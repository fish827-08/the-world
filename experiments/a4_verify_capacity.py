"""A4/R19 生态门验证（本地开发）— 修复后代码重跑，4 机制为主 + 3 机制对照。

裁定依据：
- **R19**：R4 批运行于修复提交前 ⇒ 整体降级；生态门须用**修复后**代码重跑
  **5 seed × 60k、`arbitrary_codebook=true`**。
- **V12 口径**：4 机制为主 + 3 机制对照臂；两档同 seed 集、同 tick、同 log 间隔。
- **R22 (F-D15)**：manifest 必须记录 git commit + sim_core 指纹 + 配置指纹 + 全部开关真实状态。

用法：
  python experiments/a4_verify_capacity.py --mode on --seed 42 --ticks 60000 --codebook 1 \
      --out _rerun_logs/a4_fix/asym_on_cb1_s42.csv --snapshot-every 5000

断点续跑（快照）—— ⚠️ **F-R7：快照默认被 `.gitignore` 排除**
  R19 首批的快照因数据仓 `.gitignore` 含 `*.npz|*.pkl` 而**从未入库**，
  直接后果是"本地接不上、只能从 0 重跑"，白丢约 9 小时。
  ⇒ **D-23a 处置**：快照统一写到 `--snapshot-dir`（默认 `_rerun_logs/snap/`）下的
  **固定文件名 `<tag>.snapshot.npz`（原地覆盖）**，该目录已在 `.gitignore` 中放开，
  **暂停/中断时提交一次即可入库**；每 run 只 1 个文件（≈4–6 MB），10 run ≈ 40–60 MB。
  ❌ 切勿放开 `*.npz` 通配：120 个/run × 4 MB ≈ 480 MB/run，会把仓库再撑爆。
  中断后原命令重跑即自动续跑（读取已存在的快照，从 e._tick+1 继续）
  - --fresh 强制从 tick 0 重来
  ⚠️ D2 的感知噪声走【全局 np.random】（已知缺陷 F-D2），引擎快照不含它；
     故本脚本额外存取 np.random 状态，保证续跑与"不中断连续跑"**逐位一致**。
生态门判定：跑满目标 tick 且 tick>=10000 起 N 全程 >0（= N 稳定 >0 持续 50k）。
进度可见：CSV 每采样点 flush；同目录 <name>.progress.json 每 5000 tick 更新。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import (  # noqa: E402
    CALIBRATION_M_RANGE,
    SIGNAL_ALPHABET_IMPLEMENTED,
    SIGNAL_ALPHABET_STATES,
    InfoStructureConfig,
    PredationConfig,
    SimConfig,
)
from simulation.sphere_engine import SphereEngine  # noqa: E402


# D-19：provenance 统一走 simulation.provenance（硬校验，不再本地静默 None/"unknown"）
from simulation.provenance import collect as prov_collect, validate as prov_validate
from observatory.statistics import (  # D-16：单一口径实现
    codebook_convergence, max_generation_current, predation_fraction,
)
from observatory.statistics import selection_gradient  # D-17：⑤ 单一口径


def _moments(x) -> tuple[float, float, float]:
    """样本标准差 / 偏度 / **超额**峰度（矩法）—— BC 双峰系数的输入（R135 第 -1 步①）。

    BC = (skew² + 1) / (kurt + 3(n-1)²/((n-2)(n-3)))，BC > 5/9 提示双峰。
    ⚠️ 调用方须先保证 **n ≥ 3**：峰度是四阶矩，样本过小时方差极大（可有可无的天文数字）。
    """
    a = np.asarray(x, dtype=np.float64)
    n = int(a.size)
    if n < 3:
        return 0.0, 0.0, 0.0
    d = a - a.mean()
    m2 = float((d ** 2).mean())
    m3 = float((d ** 3).mean())
    m4 = float((d ** 4).mean())
    if m2 <= 1e-300:          # 退化分布（所有值相同）
        return 0.0, 0.0, 0.0
    std = float(np.sqrt(m2 * n / (n - 1)))          # ddof=1
    skew = m3 / (m2 ** 1.5)
    kurt = m4 / (m2 ** 2) - 3.0                     # 超额峰度（正态 = 0）
    return round(std, 4), round(skew, 4), round(kurt, 4)


def build(mode: str, codebook: bool, seed: int, ticks: int, *,
          max_count: int = 5000, neutral: bool = False,
          sig_disabled: bool = False, oracle: bool = False,
          measure: bool = False, oracle_donation: float | None = None,
          oracle_persistence: int | None = None,
          gain_multiplier: float | None = None,
          calibration_arm: bool = False,
          signal_mode: str | None = None,
          signal_alphabet: str | None = None,
          gate_mode: str | None = None, gate_delta: str | None = None,
          distribution: str | None = None,
          # A′ 记忆朝向梯度（2026-09-19）：`orientation` 才启用；默认 `none` = 原式（逐位等价）
          memory_gradient: str = "none",
          memory_gradient_gain: float = 0.3,
          # PC-1（R134，2026-09-20）：S2 关捕食 / S1 软顶 / 零模型臂的瓶颈开关
          predation_enabled: bool = True,
          soft_cap_target: float = 0.0,
          learning_bottleneck: bool = True,
          reputation_weight: float = 0.0,
          # R135 第3步 A-连续（2026-09-20）：凸 trade-off 取食倍率 (1−g16)^k
          forage_tradeoff_k: float = 0.0) -> SphereEngine:
    c = SimConfig(seed=seed)
    c.simulation.ticks = ticks
    c.simulation.use_sim_core = False          # D2 须走 Python 路径（AGENTS.md）
    c.simulation.history_limit = 100           # 环形缓冲，限内存（不改变语义）
    c.population.initial_count = 200           # R4 manifest 真实口径
    c.population.max_count = max_count         # R41：标杆批口径 3240（⑤ 不饱和前提）
    # PC-1（R134）：S1 软顶（目标窗形式，冒烟后修订）/ S2 关捕食（默认 = 旧行为）
    c.population.soft_cap_target = float(soft_cap_target)
    # ⚠️ 这里是**整体替换** PredationConfig ⇒ **必须**把 k 一并传进去，
    #    否则 A-连续的凸度会被静默重置为 0（F1 同型：传了开关却没生效）。
    c.predation = PredationConfig(
        enabled=bool(predation_enabled),
        forage_tradeoff_k=float(forage_tradeoff_k),
    )
    # 2026-09-19（R127/C8）：**原为硬编码 "uniform"**（注释"R4 manifest 真实口径"）——
    # 该硬编码使 C1a / C1b / C2 / α / gate / α8 **全部跑在 uniform 世界**
    # （容量只随纬度变 ⇒ 食物位置**可由位置预测** ⇒ 信息本不值钱），**且无任何告警**：
    # C4 对账"开关 vs 设计"时两边都是 uniform ⇒ 判通过。⇒ 见 R127 的 **C8 前提对账**。
    # 现改为**可配**：默认仍 "uniform"（与历史批可比）；"patchy" 须**显式指定**且**先过前提冒烟**。
    c.resources.distribution = str(distribution) if distribution else "uniform"
    # A′ 走向构造参数（而非构造后赋值）：`__post_init__` 只在构造时跑 ⇒ 赋值不会校验（F1 同型教训）
    d2 = InfoStructureConfig(
        enabled=True,
        learning_bottleneck=bool(learning_bottleneck),
        memory_gradient=str(memory_gradient),
        memory_gradient_gain=float(memory_gradient_gain),
        reputation_weight=float(reputation_weight),   # PC-1：C3 人为拉满（R134 裁定 1.0）
    )
    d2.learning_rate = 0.05
    d2.arbitrary_codebook = codebook           # R19/V12：4 机制=True，3 机制对照=False
    d2.steels_alignment = True
    d2.measure_signal_response = measure       # D-18 ⑥ 探针（纯观测）
    if mode == "off":                          # 对称对照：全感知/无噪声/argmax
        d2.perception_radius = 8
        d2.perception_noise = 0.0
        d2.softmax_tau = 0.0
    c.info_structure = d2
    c.neutral_genes = neutral                  # 漂变零模型（冻结 g14/g15）
    c.signal_disabled = sig_disabled           # 信号禁用臂
    c.oracle.enabled = oracle                  # D-8 oracle 正向对照臂
    # D-27④-A：剂量扫描（None ⇒ 沿用配置默认，行为与旧版逐位一致）
    if oracle_donation is not None:
        c.oracle.donation = float(oracle_donation)
    if oracle_persistence is not None:
        c.oracle.persistence = int(oracle_persistence)
    # R97 增益校准档（2026-09-16）：**先登记旗、再设 m**，并在此显式做 C5 v2 自洽检查
    # （dataclass 构造后赋值不触发 __post_init__ ⇒ 不能只依赖它；F1 教训同型）。
    if calibration_arm:
        c.oracle.is_calibration_arm = True
    if gain_multiplier is not None:
        c.oracle.gain_multiplier = float(gain_multiplier)
    if c.oracle.gain_multiplier != 1.0 and not c.oracle.is_calibration_arm:
        raise SystemExit(
            f"C5 v2 硬失败：gain_multiplier={c.oracle.gain_multiplier} ≠ 1.0 "
            f"但未登记校准臂（须 --calibration-arm）"
        )
    if c.oracle.is_calibration_arm:
        lo, hi = CALIBRATION_M_RANGE
        _mm = float(c.oracle.gain_multiplier)
        if not (_mm == 1.0 or lo <= _mm <= hi):
            raise SystemExit(
                f"C5 v2 硬失败：校准臂 gain_multiplier={_mm} 只允许 1.0（配对基线）"
                f"或落在 [{lo}, {hi}]（处理臂）"
            )
    # R100 条件 7（随机信号自检）：`signal_mode="random"` ⇒ 信号与个体状态无关（**无信息**）
    # ⇒ 若 ratio/ρ 同样上升，即实证「增益不依赖信号内容」（V-1 C-4 的可执行检验）。
    if signal_mode is not None:
        c.signal_mode = str(signal_mode)
    # R113/R121 信号字母表（默认 "16" = 现状；"4" = 仅能量 2 位）
    # R123/B② 门控臂（付款闸；仪器 ⇒ 必须登记校准臂）
    if gate_mode is not None:
        c.oracle.gate_mode = str(gate_mode)
    if gate_delta is not None:
        c.oracle.gate_delta = str(gate_delta)
    if c.oracle.gate_mode == "delta_positive" and not c.oracle.is_calibration_arm:
        raise SystemExit(
            "门控臂是仪器（改付款规则）⇒ 必须同时给 --calibration-arm（R100 条件 5）"
        )
    if c.oracle.gate_mode == "delta_positive" and not c.info_structure.measure_signal_response:
        raise SystemExit(
            "门控臂需要 measure_signal_response=True（否则 Δ≡0 ⇒ 付款全消失 = 假结论源）"
        )
    if signal_alphabet is not None:
        _sa = str(signal_alphabet)
        if _sa not in SIGNAL_ALPHABET_IMPLEMENTED:
            raise SystemExit(
                f"signal_alphabet={_sa!r} 尚未实施（已实现：{SIGNAL_ALPHABET_IMPLEMENTED}）"
                "—— 不静默降级（教训 2）"
            )
        c.signal_alphabet = _sa
    return SphereEngine(c)


def _mg_frac(e, key: str) -> str:
    """A′ 朝向梯度计数占比；未适用/不可用 ⇒ **空串**（不是 0，同 R120 n/a 口径）。"""
    st = e.memory_gradient_stats()
    if not st or not st.get("counters_available"):
        return ""
    v = st.get(key)
    return "" if v is None else str(v)


def main() -> None:
    # ⚠️ 2026-09-15（内评代修，对应已上板的同类缺陷）：本文件续跑路径 print("\u21bb ...")，
    #    而 Windows 默认 GBK 控制台**无法编码 U+21BB (↻)** ⇒ UnicodeEncodeError ⇒ **rc=1 假失败**。
    #    症状：`test_f_r10_f_r12` 的 3 例失败（含 F-R13 回归），且**只在续跑路径**触发
    #    —— 而 D-27-④ 的剂量测量（4 剂量 × 3 seed，须多轮续批）**正要走这条路**。
    #    与 `preflight_check.py` 的同类修复同源（同一错型：“仪器坏了却看不见”的孪生面）。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass  # 非 TTY / 旧解释器：降级不重配，不因诊断能力缺失而阻断运行
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("on", "off"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ticks", type=int, default=60000)
    ap.add_argument("--log-interval", type=int, default=1000)
    ap.add_argument("--codebook", type=int, default=1,
                    help="1=四机制(arbitrary_codebook=True, R19 主) / 0=三机制对照")
    ap.add_argument("--out", required=True)
    ap.add_argument("--snapshot-every", type=int, default=5000,
                    help="每 N tick 写一次快照（0=不写）")
    ap.add_argument("--snapshot-dir", default="_rerun_logs/snap",
                    help="D-23a：单一最新快照目录（固定文件名原地覆盖，已放开 gitignore）")
    ap.add_argument("--fresh", action="store_true",
                    help="忽略已有快照，从 tick 0 重跑")
    # ---- D-24 小规格验证批（R47/R53）新臂型 ----
    ap.add_argument("--arm", choices=("main", "control", "zero", "sigoff", "oracle"),
                    default=None,
                    help="main=4机制主臂 / control=3机制对照 / zero=漂变零模型(冻结g14g15) / "
                         "sigoff=信号禁用 / oracle=正向对照（D-8）。指定即开 ⑥探针+⑤观测")
    ap.add_argument("--max-count", type=int, default=5000,
                    help="种群上限（R41：⑤ 不饱和前提=3240；R19 口径=5000）")
    ap.add_argument("--measure", action="store_true",
                    help="显式开 ⑤观测+⑥探针（--arm 已隐含）")
    # D-27④-A（R86 修订）：oracle 剂量扫描参数。默认 None ⇒ 沿用配置默认值（不改行为）。
    ap.add_argument("--oracle-donation", "--donation", dest="oracle_donation",
                    type=float, default=None,
                    help="覆盖 oracle.donation（剂量扫描用）；**仅 --arm oracle 生效**。"
                         "`--donation` 是等性别名（供 batch_runner `--<key>` 拼参用）")
    ap.add_argument("--oracle-persistence", type=int, default=None,
                    help="覆盖 oracle.persistence（归因窗口 tick）；仅 --arm oracle 生效")
    # R97 增益校准档（R105 放行 ④ 实施）：`m` 必须与校准臂登记成对出现（C5 v2）。
    ap.add_argument("--gain-multiplier", "--m", dest="gain_multiplier",
                    type=float, default=None,
                    help="覆盖 oracle.gain_multiplier（增益档）；**仅 --arm oracle 生效**，"
                         "且 m≠1 必须同时给 --calibration-arm")
    ap.add_argument("--signal-mode", dest="signal_mode", default=None,
                    choices=["state", "random", "evolved"],
                    help="覆盖 signal_mode；`random` = 信号与个体状态无关（**无信息**）"
                         "⇒ 供 R100 条件 7「随机信号自检」用")
    ap.add_argument("--gate-mode", dest="gate_mode", default=None,
                    choices=["none", "delta_positive"],
                    help="R123/B② 门控臂：付款闸 = 只对 Δ_i>0 的到达付款（**仪器**，"
                         "须与 --calibration-arm 同用）")
    ap.add_argument("--gate-delta", dest="gate_delta", default=None,
                    choices=["content", "full"],
                    help="Δ 口径：content=只去**内容项**（保留存在性项，**付款闸推荐**）；"
                         "full=去两项（现探针口径）")
    # A′ 记忆朝向梯度：默认 none（原式 ⇒ 与历史批逐位可比）
    ap.add_argument("--memory-gradient", dest="memory_gradient", default="none",
                    choices=("none", "orientation"),
                    help="A′：none=原式（记忆格==邻居格才加分）；orientation=朝向梯度（cos 允许为负）")
    ap.add_argument("--memory-gradient-gain", dest="memory_gradient_gain",
                    type=float, default=0.3,
                    help="A′：朝向梯度增益（默认 0.3，与原式同量级）")
    # PC-1（R134）：S2 关捕食 / S1 软顶 / 零模型臂瓶颈开关（值式 flag ⇒ preset 可用 key=value）
    ap.add_argument("--predation-enabled", dest="predation_enabled", default="true",
                    choices=("true", "false"),
                    help="PC-1 S2：false=关捕食（单营养级）；默认 true=原式")
    ap.add_argument("--soft-cap-target", dest="soft_cap_target", type=float, default=0.0,
                    help="PC-1 S1（目标窗形式）：N* = target×max_count，出生率在 N* 处线性归零；"
                         "0=关（默认=原式）。R134 PC-1 拟 0.6（窗 [0.2K,0.95K] 的中位）")
    ap.add_argument("--forage-tradeoff-k", dest="forage_tradeoff_k", type=float, default=0.0,
                    help="R135 第3步 A-连续：取食倍率 (1−g16)^k（凸 trade-off）；"
                         "0=关（默认，与旧版逐位一致）；本批取 2.0（凸/加速下降 ⇒ 中间态杂食者吃亏）")
    ap.add_argument("--learning-bottleneck", dest="learning_bottleneck", default="true",
                    choices=("true", "false"),
                    help="PC-1 零模型臂：false=关学习瓶颈（配码本关=零模型）；默认 true")
    # PC-1：C3 人为拉满（R134 裁定 reputation_weight=1.0）；signal-disabled 值式 flag
    # （--arm sigoff 的语义照旧，两者取或——禁用臂用哪个都行，preset 统一走值式）
    ap.add_argument("--reputation-weight", dest="reputation_weight",
                    type=float, default=0.0,
                    help="R2 声誉权重（C3 人为拉满用；R134 PC-1 裁定 1.0）")
    ap.add_argument("--signal-disabled", dest="signal_disabled_flag", default="false",
                    choices=("true", "false"),
                    help="true=信号常关（PC-1 禁用臂）；与 --arm sigoff 取或")
    ap.add_argument("--signal-alphabet", dest="signal_alphabet", default=None,
                    choices=list(SIGNAL_ALPHABET_IMPLEMENTED),
                    help='信号字母表档位（R113/R121）："16"=现状 4 位（默认）；'
                         '"4"=仅能量 2 位（code=e_bin+1∈{1..4}，删 f_bit/n_bit）。'
                         '⚠️ 与 "16" 批次（C1a/C1b/C2）的 ratio/codebook_conv 不可直接比较')
    ap.add_argument("--distribution", dest="distribution", default=None,
                    choices=("uniform", "patchy"),
                    help="食物分布（R127/C8 前提开关）：uniform=默认（与历史批可比）；"
                         "patchy=空间斑块（**须先过前提冒烟**：容量空间异质性）")
    ap.add_argument("--calibration-arm", action="store_true",
                    help="登记本臂为**校准臂**（`is_calibration_arm=True`）；"
                         "R100 条件 5：未登记而 m≠1 ⇒ 硬失败")
    args = ap.parse_args()

    arm = args.arm
    neutral = arm == "zero"
    sig_disabled = arm == "sigoff" or (args.signal_disabled_flag == "true")
    oracle_on = arm == "oracle"
    # R59/F-R9：control = 3 机制对照 ⇒ 关码本（arm 语义优先于 --codebook 默认值 1）。
    # 若无此行，batch grid 只传 arm 时 control 会与 main 同配置同轨迹（2026-09-13 D-24 实测复现）。
    if arm == "control":
        args.codebook = 0
    # D-27④-A：剂量参数只在 oracle 臂有意义——**非 oracle 臂传了就直接报错**，
    # 不静默忽略（教训 2：静默 no-op 最危险）。
    if arm != "oracle" and (args.oracle_donation is not None
                            or args.oracle_persistence is not None):
        raise SystemExit(
            "--oracle-donation/--oracle-persistence 仅在 --arm oracle 下生效"
            f"（当前 arm={arm!r}）——请勿静默传参"
        )
    measure = bool(args.measure) or arm is not None

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prog = out.with_suffix(".progress.json")
    # D-23a：单一最新快照（固定文件名、原地覆盖）落在已放开 gitignore 的目录；
    # 兼容旧批次：新路径不存在时回退到 <out>.snapshot.npz（旧命名）。
    snap = Path(args.snapshot_dir) / f"{out.stem}.snapshot.npz"
    rngp = snap.with_suffix(".rngstate.pkl")
    if not snap.exists():
        legacy = out.with_suffix(".snapshot.npz")
        if legacy.exists():
            snap, rngp = legacy, out.with_suffix(".rngstate.pkl")
    # ---- F-R13：快照目录可能不存在 ----
    # `--snapshot-dir` 是调用方给的路径，本脚本此前只 mkdir 了 `out.parent`
    # ⇒ `save_snapshot` 在 `np.savez_compressed` 处抛 FileNotFoundError（实测：
    # test_f_r10_f_r12 两轮续批集成 2 例失败）。旧命名回退时 snap.parent 即
    # out.parent（已建），此处幂等，无副作用。
    snap.parent.mkdir(parents=True, exist_ok=True)

    # ---- 断点续跑：快照存在且非 --fresh 则从快照恢复 ----
    resumed = False
    if not args.fresh and snap.exists():
        e = SphereEngine.load_snapshot(str(snap))       # 配置由快照自带，指纹天然一致
        if rngp.exists():                               # F-D2：补回全局 np.random 状态
            with open(rngp, "rb") as fh:
                np.random.set_state(pickle.load(fh))
        start_tick = int(e._tick)
        resumed = start_tick > 0
    else:
        e = build(args.mode, bool(args.codebook), args.seed, args.ticks,
                  max_count=args.max_count, neutral=neutral,
                  sig_disabled=sig_disabled, oracle=oracle_on, measure=measure,
                  oracle_donation=args.oracle_donation,
                  oracle_persistence=args.oracle_persistence,
                  gain_multiplier=args.gain_multiplier,
                  calibration_arm=bool(args.calibration_arm),
                  signal_mode=args.signal_mode,
                  signal_alphabet=args.signal_alphabet,
                  gate_mode=args.gate_mode, gate_delta=args.gate_delta,
                  distribution=args.distribution,
                  memory_gradient=args.memory_gradient,
                  memory_gradient_gain=args.memory_gradient_gain,
                  predation_enabled=(args.predation_enabled == "true"),
                  soft_cap_target=args.soft_cap_target,
                  learning_bottleneck=(args.learning_bottleneck == "true"),
                  reputation_weight=args.reputation_weight,
                  forage_tradeoff_k=args.forage_tradeoff_k)
        start_tick = 0
    # 🔴 内评未闭合项 #6（2026-09-20 修）：**provenance 必须在跑之前采集**。
    #   收尾时采集记的是"跑完之后的代码树"——长批期间若有人推提交（E-027 就发生过），
    #   manifest 会把**后人的代码**记成这批的产物 ⇒ 既不可复现，也会把排障带向错误方向。
    #   `rng_draws` 例外：它只有收尾才准 ⇒ 在 summary 组装处单独回填。
    prov_start = prov_collect(e.config, python_info=True)
    if resumed:
        print(f"  ↻ 从快照续跑：tick {start_tick} → {args.ticks}")
        # R121 §3.1 + C4 读回：续跑时配置**由快照自带** ⇒ 命令行若另给档位而快照不同，
        # 必须**硬失败**而非静默忽略（F-R21/C5 同族：传了开关却没生效）
        if (args.gate_mode is not None or args.gate_delta is not None) and arm != "oracle":
            raise SystemExit(
                f"--gate-mode/--gate-delta 仅在 --arm oracle 下生效（当前 arm={arm!r}）"
                "——请勿静默传参"
            )
        if args.signal_alphabet is not None and (
            str(args.signal_alphabet) != str(e.config.signal_alphabet)
        ):
            raise SystemExit(
                f"signal_alphabet 冲突：命令行 {args.signal_alphabet!r} vs "
                f"快照 {e.config.signal_alphabet!r} —— 跨档不得续跑（R121 §3.1，特性非缺陷）"
            )
        # R127/C8：distribution 是**前提开关** ⇒ 续跑时必须与快照一致（否则前提被静默换掉）
        if args.distribution is not None and (
            str(args.distribution) != str(e.config.resources.distribution)
        ):
            raise SystemExit(
                f"distribution 冲突：命令行 {args.distribution!r} vs "
                f"快照 {e.config.resources.distribution!r} —— 前提开关不得跨批混用（R127 C8）"
            )
    # R121 §3.4：**指标口径必须随档位走**（"16"⇒16、"4"⇒4）。
    # 漏传的后果：数组宽度恒 16，未用槽恒"一致" ⇒ 收敛度**系统性虚高**（静默错误）。
    _n_alpha = SIGNAL_ALPHABET_STATES[str(e.config.signal_alphabet)]

    fields = ["tick", "N", "g14", "g15", "g16", "trust",
              # R135 第 -1 步①（2026-09-20）：g16 **分布矩**——
              # 此前只有均值 ⇒ **判不了双峰**（双峰与单峰可同均值）。
              # BC 双峰系数 = (skew²+1)/(kurt+3(n-1)²/((n-2)(n-3)))，故输出 std/skew/kurt/n。
              # ⚠️ `g4` 是**搭车诊断列**：g16 与 g4（进食量倍率，强选择）的初始 LD 会把
              #    g16 捎带上漂（实测 s42 Δg16=+0.095 全由此而来）⇒ 没有 g4 就无法区分
              #    "g16 被选择" 与 "g16 被搭车"。详见 `sphere_engine._genome_summary`。
              "g16_std", "g16_skew", "g16_kurt", "g4",
              "max_gen", "max_gen_cur",       # R77：高水位 / 当刻最深（两个口径分列）
              "mean_row", "polar_frac",
              "codebook_conv", "pred_frac",   # D-16：R31③/R38③ 判据列
              "resp_a", "resp_b", "oracle_ratio",   # D-18⑥/D-8（累计口径）
              "mem_bit_frac",   # R128 §五 步骤 0：mem_bit 取值分布的**时间序列**（累计口径）
              # A′（2026-09-19）：**先证"测到了"**再判读 ⇒ 两个可观测性占比（累计口径）
              "mem_grad_slots_frac", "mem_grad_trig_frac"]
    # ---- F-R12：续跑必须**按 tick 幂等**写 CSV ----
    # 原因（2026-09-15 D-24 实测）：续跑直接 `open("a")` 追加 ⇒ 多轮续批会把
    # [start_tick 之前] 的 tick 重复写入（云端 20+ 轮续批：main_s42 16 个重复、
    # main_s43 30 个重复 ⇒ 时序非单调，且**首轮完全看不出来**）。
    # 改法：先截断到 `start_tick`（保留 tick ≤ start_tick 的行），再继续追加。
    if resumed and out.exists():
        _keep = [
            r for r in csv.DictReader(out.open(encoding="utf-8"))
            if str(r.get("tick", "")).isdigit() and int(r["tick"]) <= start_tick
        ]
        with out.open("w", encoding="utf-8", newline="") as _fh:
            _w = csv.DictWriter(_fh, fieldnames=fields)
            _w.writeheader()
            _w.writerows(_keep)
    fh = out.open("a" if resumed else "w", encoding="utf-8", newline="")
    w = csv.DictWriter(fh, fieldnames=fields)
    if not resumed:
        w.writeheader()
    fh.flush()

    last = start_tick
    for t in range(start_tick + 1, args.ticks + 1):
        e.step()
        if t % args.log_interval == 0 or e.extinct:
            P = len(e._id)
            r = (e._flat[:P] // 120) if P else np.zeros(0)
            w.writerow({
                "tick": t, "N": P,
                "g14": round(float(e._genes[:P, 14].mean()), 4) if P else "",
                "g15": round(float(e._genes[:P, 15].mean()), 4) if P else "",
                # g16 AGGRESSION 均值（所有者 09-20 01:02 派工：测"攻击基因固化"假说，
                # 捕食态内战振荡的基因层证据——此前 g16 从未入 CSV）
                "g16": round(float(e._genes[:P, 16].mean()), 4) if P else "",
                **({} if P < 3 else dict(zip(
                    ("g16_std", "g16_skew", "g16_kurt"), _moments(e._genes[:P, 16])))),
                "g4": round(float(e._genes[:P, 4].mean()), 4) if P else "",
                "trust": round(float(e._trust[:P].mean()), 4) if P else "",
                "max_gen": int(e._max_generation),          # R77：历史高水位（不回落）
                "max_gen_cur": max_generation_current(e),   # R77：当刻最深（只看存活）
                "mean_row": round(float(r.mean()), 3) if P else "",
                "polar_frac": round(float(((r <= 5) | (r >= 54)).mean()), 4) if P else "",
                # D-16：
                # R121 §3.4：n_states 随 signal_alphabet（CSV 列口径）
                "codebook_conv": (
                    round(codebook_convergence(e._codebook[:P], n_states=_n_alpha), 4)
                    if P else ""
                ),
                "pred_frac": round(predation_fraction(e.death_cause_totals()), 4),
                # D-18⑥/D-8（累计口径；未开探针时恒 0）
                "resp_a": e.signal_response_stats()["resp_a_exposure"],
                "resp_b": e.signal_response_stats()["resp_b_delta"],
                "oracle_ratio": e.oracle_stats()["oracle_return_ratio"],
                # R128 §五 步骤 0：`mem_bit` 取值分布（累计占比 = mem_bit=1 的发射 / 状态编码发射）
                # 非 "8" 档 ⇒ 空串（**未适用**，不是 0；同 R120 的 ratio n/a 口径）
                # A′：朝向梯度的**可观测性**（非 orientation 模式 ⇒ 空串 = 未适用，不是 0）
                "mem_grad_slots_frac": _mg_frac(e, "slots_frac"),
                "mem_grad_trig_frac": _mg_frac(e, "trigger_frac"),
                "mem_bit_frac": (
                    e.alphabet_stats()["mem_bit_frac"]
                    if e.alphabet_stats()["mem_bit_frac"] is not None else ""
                ),
            })
            fh.flush()
            last = t
            if e.extinct:
                break
            if t % 5000 == 0:
                prog.write_text(json.dumps({"seed": args.seed, "mode": args.mode,
                                            "codebook": args.codebook, "tick": t,
                                            "N": P}, ensure_ascii=False), encoding="utf-8")
        # 快照独立节拍（不依赖 log_interval），保证任意时刻中断最多丢 snapshot-every 个 tick
        if args.snapshot_every and t % args.snapshot_every == 0:
            e.save_snapshot(str(snap))
            with open(rngp, "wb") as fh2:
                pickle.dump(np.random.get_state(), fh2)
    fh.close()

    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    tail = [r for r in rows if int(r["tick"]) >= 10000]
    reached = last >= args.ticks
    stable50k = bool(tail) and all(int(r["N"]) > 0 for r in tail) and last >= 10000
    dc = {str(k): int(v) for k, v in e.death_cause_totals().items()}
    t0 = e.genome_t0_stats()          # R135 第 -1 步③：t=0 基线（搭车诊断）
    # provenance（**启动采集值**为准；仅 rng_draws 用收尾值）
    prov = dict(prov_start)
    prov["rng_draws"] = int(e.rng_draws)
    prov["collected_at"] = "run_start"
    # 漂移自检：跑的过程中若有人推提交 ⇒ 记录双方，供事后归因（不失败，因为产物仍有效）
    prov_end = prov_collect(e.config, python_info=False)
    if prov_end.get("code_tree_sha256") != prov.get("code_tree_sha256"):
        prov["code_tree_drift"] = True
        prov["code_tree_at_finish"] = prov_end.get("code_tree_sha256")
        prov["code_subtrees_at_finish"] = prov_end.get("code_subtrees")
    prov_validate(prov, require_sim_core=bool(e.config.simulation.use_sim_core))
    summary = {
        "manifest": {
            **prov,
            "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "script": "experiments/a4_verify_capacity.py",
        },
        "switches": {                              # R22：全部开关真实状态
            "mode": args.mode, "seed": args.seed, "ticks_target": args.ticks,
            "resumed_from_snapshot": bool(resumed), "start_tick": start_tick,
            "arbitrary_codebook": bool(args.codebook),
            "arm": arm,
            "measure_signal_response": bool(measure),
            "oracle_enabled": bool(oracle_on),
            "oracle_donation": float(e.config.oracle.donation) if oracle_on else None,
            "oracle_persistence": int(e.config.oracle.persistence) if oracle_on else None,
            "perception_radius": int(e.config.info_structure.perception_radius),
            "perception_noise": float(e.config.info_structure.perception_noise),
            "softmax_tau": float(e.config.info_structure.softmax_tau),
            "learning_bottleneck": bool(e.config.info_structure.learning_bottleneck),
            "steels_alignment": bool(e.config.info_structure.steels_alignment),
            "reputation_weight": float(e.config.info_structure.reputation_weight),
            # A′（2026-09-19）：开关**必须可读回**（C4）—— 冒烟时实测它曾缺席 ⇒
            # 无法回答"跑的到底是哪个档"，与 R127 的 C8 前提对账同型缺陷。
            "memory_gradient": str(e.config.info_structure.memory_gradient),
            "memory_gradient_gain": float(e.config.info_structure.memory_gradient_gain),
            # R135 第 -1 步③：**t=0 基线**（跨批可比 + 搭车诊断）
            # 没有它，同一个 g16 读数在 A 批是"选择"、在 B 批是"搭车"，无法分辨。
            "g16_t0_mean": round(t0["g16_mean"], 4),
            "g16_t0_std": round(t0["g16_std"], 4),
            "g4_t0_mean": round(t0["g4_mean"], 4),
            "corr_g16_g4_t0": (round(t0["corr_g16_g4"], 4)
                               if t0.get("corr_g16_g4") is not None else None),
            # PC-1（R134）：S1/S2 开关必须可读回（C4——E-027 的 16 码事故同型预防）
            "predation_enabled": bool(e.config.predation.enabled),
            "soft_cap_target": float(e.config.population.soft_cap_target),
            # R135 第3步 A-连续：凸度必须可读回（C4）
            "forage_tradeoff_k": float(e.config.predation.forage_tradeoff_k),
            # 🔴 R136 §一 增量 2（C4 自证缺口）：PC-1 三臂的 `switches.arm` **全为 main**，
            #    码本/瓶颈两个开关读不到 ⇒ **臂间开关差无法从产物自证**（外复核只能靠 preset 名）。
            "arbitrary_codebook": bool(e.config.info_structure.arbitrary_codebook),
            "learning_bottleneck": bool(e.config.info_structure.learning_bottleneck),
            "use_sim_core": bool(e.config.simulation.use_sim_core),
            "distribution": e.config.resources.distribution,
            "initial_count": int(e.config.population.initial_count),
            "max_count": int(e.config.population.max_count),
            "neutral_genes": bool(e.config.neutral_genes),
            "signal_disabled": bool(e.config.signal_disabled),
            # R97 增益校准档（条件 5 机器强制拒收依据 + C4 读回核对）
            "oracle_gain_multiplier": float(e.config.oracle.gain_multiplier),
            "is_calibration_arm": bool(e.config.oracle.is_calibration_arm),
            "signal_mode": str(e.config.signal_mode),
            # R113/R121 C4 读回：字母表档位（必须与命令行/预注册一致；跨档不可比）
            "signal_alphabet": str(e.config.signal_alphabet),
            # R123/B② C4 读回：门控臂档位（仪器参数必须可核对，防"传了没生效"）
            "oracle_gate_mode": str(e.config.oracle.gate_mode),
            "oracle_gate_delta": str(e.config.oracle.gate_delta),
        },
        "result": {
            # ⚠️ 必须用引擎真实 tick，不能用 last（=最后一次【采样】的 tick）：
            # log_interval 默认 1000，跑 1500 tick 时 last 会停在 1000 ⇒ final_tick 记错
            # （60k 恰好是 1000 的倍数才一直没暴露）。R42/R38 判据依赖该字段。
            "final_tick": int(e._tick), "final_N": len(e._id), "extinct": bool(e.extinct),
            "reached_60k": reached, "stable_N_gt0_last50k": stable50k,
            "eco_gate_pass": bool(reached and stable50k),
            # ---- F-R17：`eco_gate_pass` 的**适用域**必须随数据走 ----
            # 该门判据含 `last >= 10000` 硬编码（为 60k 批定义）⇒ **短程批恒 False**。
            # 缺适用域说明时，短程批的 False 会被误读成"生态崩溃"（本人 2026-09-15 实测踩到）。
            "eco_gate_scope": f"{args.ticks} tick 批",
            "eco_gate_applicable": bool(args.ticks >= 10000),
            "born_total": int(e.total_born), "died_total": int(e.total_died),
            "deaths_by_cause": dc,
            # D-16：终局判据值（区制分层用：饱和封顶 vs 捕食主导）
            "final_codebook_conv": (
                round(
                    codebook_convergence(
                        e._codebook[: len(e._id)], n_states=_n_alpha
                    ),
                    4,
                )
                if len(e._id) else 0.0
            ),
            "final_pred_frac": round(predation_fraction(dc), 4),
            # R77：max_gen 两个口径**分名记录**（此前同一列名 `max_gen` 在不同脚本里
            # 分别指"当刻最深"与"历史高水位"⇒ 跨脚本比较必错）
            "final_max_gen_highwater": int(e._max_generation),
            "final_max_gen_current": max_generation_current(e),
            # D-17 ⑤（R42）：非饱和窗主口径 + 饱和窗诊断（两窗不得合并）
            "selection_gradient": selection_gradient(e),
            # D-18 ⑥（R43）：三联报
            "signal_response": e.signal_response_stats(),
            # D-8 oracle（O-4/O-7）：manifest 必录 return_ratio
            # （内含 R121 §4.1 的 food_band_true_sig 零机时 counter）
            "oracle": e.oracle_stats(),
            # R121 §3：字母表档位 + §3.5 码值域守卫（bad_code_n 在 "4" 下必须恒 0）
            "alphabet": e.alphabet_stats(),
            # A′（2026-09-19）：记忆朝向梯度的可观测性计数（非 orientation ⇒ **None 未适用**）
            "memory_gradient": e.memory_gradient_stats(),
            # R135 第 -1 步④：互捕结构量化（攻击者/猎物 g16 直方图 + Δ）
            "cannibalism": e.cannibalism_stats(),
            # R135 第 -1 步③：t=0 基因组基线（搭车诊断）
            "genome_t0": {k: v for k, v in t0.items() if k != "gene_means"},
            # R121 §4.2：记忆继承卫生（"生而知之"量化；纯观测）
            "inheritance": e.inheritance_stats(),
        },
    }
    out.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # ---- F-R10：收尾**不再删除** progress.json ----
    # 原因（2026-09-15 D-24 实测）：删除会被环境 safe-delete 守卫 fail-closed 拒绝
    # （作用域内累计删除超限 ⇒ 拒删并终止进程）⇒ 子进程 rc=1 **假失败**，
    # 污染批次状态与退出码（数据无损：上一行 summary 已先写）。
    # 改法：写"完成标记"覆盖 progress.json（不删文件 ⇒ 不触发守卫）。
    # 顺序保证：summary 先写、标记后写 ⇒ **任何时刻**中断都不丢数据。
    prog.write_text(
        json.dumps(
            {
                "status": "complete",
                "final_tick": int(e._tick),
                "final_N": len(e._id),
                "finished": summary["manifest"]["finished"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    s = summary["result"]
    print(f"[{args.mode} cb={args.codebook} s{args.seed}] tick={s['final_tick']} "
          f"N={s['final_N']} eco_gate={s['eco_gate_pass']} "
          f"born={s['born_total']} died={s['died_total']}")


if __name__ == "__main__":
    main()
