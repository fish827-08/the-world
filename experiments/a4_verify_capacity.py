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
    CorpseWoundConfig,
    InfoStructureConfig,
    PredationConfig,
    ResourceDynamicsConfig,
    SimConfig,
    SubposConfig,
)
from simulation.sphere_engine import SphereEngine  # noqa: E402


# D-19：provenance 统一走 simulation.provenance（硬校验，不再本地静默 None/"unknown"）
from simulation.provenance import collect as prov_collect, validate as prov_validate
from observatory.statistics import (  # D-16：单一口径实现
    codebook_convergence, max_generation_current, predation_fraction,
)
from observatory.statistics import selection_gradient  # D-17：⑤ 单一口径


# R139/R140：g16 直方图的箱数（值域 [0,1] 均分）。**改动它等于改判据口径** ⇒ 单一真源。
G16_BINS = 10


def _hist10(x) -> list[int]:
    """g16 的 10-bin 直方图（值域 [0,1]，右开区间；越界值夹到两端）。"""
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return [0] * G16_BINS
    idx = np.clip((a * G16_BINS).astype(np.int64), 0, G16_BINS - 1)
    return np.bincount(idx, minlength=G16_BINS)[:G16_BINS].tolist()


# R141 P0 派工单 §1.3 锁定的列名（🔴 勿改名——`tools/calib_solve.py` 按此消费）
EC_CSV_COLS = ("g_lo_n", "g_lo_net_mean", "g_lo_net_p50", "g_lo_net_var",
               "g_mid_n", "g_mid_net_mean", "g_mid_net_p50", "g_mid_net_var",
               "g_hi_n", "g_hi_net_mean", "g_hi_net_p50", "g_hi_net_var",
               "forage_in_mean", "pred_in_mean", "prey_energy_mean")


def _ec_csv_row(e) -> dict:
    """把 `_ec_flush` 刚写的那一行净收入统计摊平成 CSV 列（None ⇒ 空串，R120 口径）。"""
    ts = e.ec_timeseries()
    row = ts[-1] if ts else {}
    out = {}
    for k in EC_CSV_COLS:
        v = row.get(k)
        out[k] = "" if v is None else (int(v) if k.endswith("_n") else round(float(v), 6))
    return out


def _probe_csv(probe: dict | None, key: str):
    """L1/L2 探针读数进 CSV 列（R150 B4）。

    🔴 探针为 `None`（开关关 ⇒ **未适用**）或该键为 `None`（分母为 0）⇒ **空串**，
    **不是 0**（R120 / §五.12 口径铁律："没测" ≠ "测出零"）。
    """
    if not probe:
        return ""
    v = probe.get(key)
    return "" if v is None else v


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
          forage_tradeoff_k: float = 0.0,
          # R152/P0（2026-09-22）：捕食生态位结构 6 参（**默认值 = 旧行为，逐位一致**）
          attack_cost: float = 0.1, transfer_ratio: float = 0.4,
          attack_gene_gate: float = 0.3, attack_prob_coef: float = 0.2,
          success_floor: float = 0.1, success_ceil: float = 0.9,
          # R141 P0-2（派工单 §二）：g16 初始投放（"" = 旧行为；"0.05,0.5,0.9" = 校准批）
          init_g16_clusters: str = "",
          # R144/R145：能量封顶开关（默认 False = 与 E-017~E-031/calib1 可比）
          energy_cap: bool = False,
          # R145 §七.1：光合产能覆盖（`photo_max`；None = 不覆盖）——供配对臂用
          photo_max: float | None = None,
          # R146/R149 L1/L2（R150 B1/B2；**默认全关 = 旧行为**，逐位等价）
          l1_seek: bool = False, l1_fear: bool = False, l2_dash: bool = False,
          w_seek_max: float = 0.5, w_fear: float = 0.5,
          l1_prey_mode: str = "lowagg",
          # S1 骨架（设计稿 §三/§5.2）：尸体—食腐 + 血条—受伤（**默认全关 = 旧行为**，
          # 逐位等价；机制未接线，开关仅建字段 + 读回）
          corpse_enabled: bool = False, corpse_energy_frac: float = 0.9,
          corpse_decay_ticks: int = 600, corpse_to_plant_frac: float = 0.5,
          corpse_patch_boost: float = 0.5, corpse_cap_per_cell: int = 200,
          scav_gate: float = 0.5, scav_s: float = 2.0,
          wound_enabled: bool = False, wound_base: float = 0.35,
          wound_heal_rate: float = 0.001, wound_heal_energy_cost: float = 0.05,
          contest_enabled: bool = False, holder_adv: float = 1.2,
          escalation_gap: float = 0.25, contest_cost_energy: float = 0.5,
          w_fear_health: float = 0.5, need_aggression_k: float = 0.5,
          wound_fear_threshold: float = 0.3,
          # 13.4 波 1（设计稿 §二/§三）：亚格连续坐标（**默认全关 = 旧行为**，逐位等价）
          subpos_enabled: bool = False, subdiv: int = 4,
          speed_gain: float = 4.0, speed_max: float = 2.0,
          min_energy_frac: float = 0.05, lat_floor: float = 0.3,
          stay_base: float = 0.0, stay_food_k: float = 0.0,
          stay_signal_k: float = 0.0, stay_fear_k: float = 0.0,
          stay_max: float = 0.8,
          # 13.4 波 2A（T2，R178）：资源动态（斑块休耕—死亡—轮作；**默认关 = 旧行为**）
          resource_dynamics_enabled: bool = False,
          rest_ticks: int = 300, kill_frac: float = 10.0,
          kill_denom: str = "regrowth", dead_regen_ticks: int = 2000,
          dead_cell_max_frac: float = 0.5,
          kill_patch_only: bool = True, rotate_same_row_only: bool = True,
          # 13.4 波 2B（T3）：视野 / 单格上限 / 社交归一化（默认 = 旧行为）
          perception_span: int = 1, cell_occupancy_cap: int = 3,
          cell_occupancy_cap_enabled: bool = False,
          social_norm: str = "auto") -> SphereEngine:
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
        # R152/P0：捕食生态位结构 6 参（**整体替换 ⇒ 必须一次传全**，漏一个就被静默重置为默认）
        attack_cost=float(attack_cost),
        transfer_ratio=float(transfer_ratio),
        attack_gene_gate=float(attack_gene_gate),
        attack_prob_coef=float(attack_prob_coef),
        success_floor=float(success_floor),
        success_ceil=float(success_ceil),
    )
    c.genome.init_g16_clusters = str(init_g16_clusters or "")   # R141 P0-2
    c.organisms.energy_cap_enabled = bool(energy_cap)           # R144：能量封顶（新纪元开关）
    if photo_max is not None:
        c.organisms.photo_max = float(photo_max)                # R145：光合配对臂
    # R146/R149 L1/L2（R150 B1/B2）：**默认全关 ⇒ 旧行为**（H1 逐位等价，C7 已钉死）
    c.simulation.l1_seek = bool(l1_seek)
    c.simulation.l1_fear = bool(l1_fear)
    c.simulation.l2_dash = bool(l2_dash)
    c.simulation.w_seek_max = float(w_seek_max)
    c.simulation.w_fear = float(w_fear)
    c.simulation.l1_prey_mode = str(l1_prey_mode)
    # S1 骨架（设计稿 §三）：尸体—食腐 + 血条—受伤，**整体替换** CorpseWoundConfig
    # （默认全关 = 旧行为，逐位等价；机制未接线 ⇒ 开关只进指纹/读回）
    c.corpse_wound = CorpseWoundConfig(
        corpse_enabled=bool(corpse_enabled),
        corpse_energy_frac=float(corpse_energy_frac),
        corpse_decay_ticks=int(corpse_decay_ticks),
        corpse_to_plant_frac=float(corpse_to_plant_frac),
        corpse_patch_boost=float(corpse_patch_boost),
        corpse_cap_per_cell=int(corpse_cap_per_cell),
        scav_gate=float(scav_gate),
        scav_s=float(scav_s),
        wound_enabled=bool(wound_enabled),
        wound_base=float(wound_base),
        wound_heal_rate=float(wound_heal_rate),
        wound_heal_energy_cost=float(wound_heal_energy_cost),
        contest_enabled=bool(contest_enabled),
        holder_adv=float(holder_adv),
        escalation_gap=float(escalation_gap),
        contest_cost_energy=float(contest_cost_energy),
        w_fear_health=float(w_fear_health),
        need_aggression_k=float(need_aggression_k),
        wound_fear_threshold=float(wound_fear_threshold),
    )
    # 13.4 波 1：亚格连续坐标（**整体替换** SubposConfig ⇒ 须一次传全，防漏传静默重置）
    c.subpos = SubposConfig(
        enabled=bool(subpos_enabled),
        subdiv=int(subdiv),
        speed_gain=float(speed_gain),
        speed_max=float(speed_max),
        min_energy_frac=float(min_energy_frac),
        lat_floor=float(lat_floor),
        stay_base=float(stay_base),
        stay_food_k=float(stay_food_k),
        stay_signal_k=float(stay_signal_k),
        stay_fear_k=float(stay_fear_k),
        stay_max=float(stay_max),
    )
    # 13.4 波 2A（T2，R178）：资源动态（**整体替换** ResourceDynamicsConfig ⇒ 一次传全）
    c.resource_dynamics = ResourceDynamicsConfig(
        enabled=bool(resource_dynamics_enabled),
        rest_ticks=int(rest_ticks),
        kill_frac=float(kill_frac),
        kill_denom=str(kill_denom),
        dead_regen_ticks=int(dead_regen_ticks),
        dead_cell_max_frac=float(dead_cell_max_frac),
        kill_patch_only=bool(kill_patch_only),
        rotate_same_row_only=bool(rotate_same_row_only),
    )
    # 13.4 波 2B（T3）：视野 / 单格上限 / 社交归一化（默认 = 旧行为）
    c.simulation.perception_span = int(perception_span)
    c.simulation.cell_occupancy_cap = int(cell_occupancy_cap)
    c.simulation.cell_occupancy_cap_enabled = bool(cell_occupancy_cap_enabled)
    c.simulation.social_norm = str(social_norm)
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
    ap.add_argument("--energy-cap", dest="energy_cap", default="false",
                    choices=("true", "false"),
                    help="R144 能量封顶：true=每 tick 末钳制 energy ≤ max_energy（**新纪元**）；"
                         "默认 false=旧行为（与 E-017~E-031/calib1 可比）")
    ap.add_argument("--photo-max", dest="photo_max", type=float, default=None,
                    help="R145：光合产能覆盖（默认 None=不覆盖）；配对臂用 --photo-max 0 关光合")
    # ---- R146/R149 L1 感知追击 + L2 机动性（R150 B1；**全部默认关 = 旧行为**）----
    # 🔴 臂间开关差必须从产物自证 ⇒ 全部进 `switches`（C4）。
    # ⚠️ `store_true`：preset 变体里写裸名（`"l1-seek"`）即可 —— `expand()` 对无值键
    #    只拼开关名（见 `cli_flag`/`expand` 的 `if val:` 分支）。
    ap.add_argument("--l1-seek", dest="l1_seek", action="store_true",
                    help="L1a 追猎项（默认关 = 旧行为，逐位等价）")
    ap.add_argument("--l1-fear", dest="l1_fear", action="store_true",
                    help="L1b 恐惧项（默认关 = 旧行为，逐位等价）")
    ap.add_argument("--l2-dash", dest="l2_dash", action="store_true",
                    help="L2 机动性（两段式冲刺；默认关 = 旧行为，逐位等价）")
    ap.add_argument("--w-seek-max", dest="w_seek_max", type=float, default=0.5,
                    help="L1a 权重上限（预注册两档：0.5(B)/0.25(C)，两档都过才算）")
    ap.add_argument("--w-fear", dest="w_fear", type=float, default=0.5,
                    help="L1b 固定权重（D 臂 = 0，作 fear 是否存在的操作检查）")
    ap.add_argument("--l1-prey-mode", dest="l1_prey_mode", default="lowagg",
                    choices=("lowagg", "any"),
                    help="猎物代理场：lowagg=低 g16 个体 / any=任意占格者（E′ 归因臂）")
    # ---- S1 骨架（设计稿 §三/§5.2）：尸体—食腐 + 血条—受伤 -------------------
    # 🔴 全部**默认关/默认值 = 旧行为**（逐位等价，C7 digest 钉死）。机制未接线 ⇒
    #    这些开关只进指纹与 `switches` 读回（C4），S2/S3 才接线。
    # ⚠️ 开 + `use_sim_core=True` ⇒ 引擎构造期硬报错（H3 fail-loud，同 L1/L2 先例）。
    ap.add_argument("--corpse-enabled", dest="corpse_enabled", action="store_true",
                    help="尸体—食腐通道（默认关 = 旧行为；S1 只建字段，机制不接线）")
    ap.add_argument("--corpse-energy-frac", dest="corpse_energy_frac", type=float, default=0.9,
                    help="死亡时剩余能量 × 该比例 转入尸体格（留 10%% 分解即失）")
    ap.add_argument("--corpse-decay-ticks", dest="corpse_decay_ticks", type=int, default=600,
                    help="尸体存续 tick（≈ 世代时间 ⇒ 脉冲可累积，Noy-Meir）")
    ap.add_argument("--corpse-to-plant-frac", dest="corpse_to_plant_frac", type=float, default=0.5,
                    help="腐烂归还植物池比例（其余为分解损失）")
    ap.add_argument("--corpse-patch-boost", dest="corpse_patch_boost", type=float, default=0.5,
                    help="腐烂处资源 +50%%（持续 2000 tick）")
    ap.add_argument("--corpse-cap-per-cell", dest="corpse_cap_per_cell", type=int, default=200,
                    help="单格尸体**能量**上限（R161 裁定：非尸具数；200 ≈ 一具满能量尸体）")
    ap.add_argument("--scav-gate", dest="scav_gate", type=float, default=0.5,
                    help="食腐 Hill 半效点（**不是硬门槛**；R156 陷阱修正 1）")
    ap.add_argument("--scav-s", dest="scav_s", type=float, default=2.0,
                    help="食腐 Hill 陡度（scav_mult = g16^s/(g16^s+gate^s)）")
    ap.add_argument("--wound-enabled", dest="wound_enabled", action="store_true",
                    help="血条—受伤（消耗战；默认关 = 旧行为；S1 只建字段，机制不接线）")
    ap.add_argument("--wound-base", dest="wound_base", type=float, default=0.35,
                    help="每次成功攻击扣血条 Δ（×(0.5+0.5×攻击性)；一次击杀需 3 次成功）")
    ap.add_argument("--wound-heal-rate", dest="wound_heal_rate", type=float, default=0.001,
                    help="每 tick 恢复（上限 1.0；完全愈合 1000 tick ≈ 1/10 寿命）")
    ap.add_argument("--wound-heal-energy-cost", dest="wound_heal_energy_cost",
                    type=float, default=0.05,
                    help="愈合耗能 / tick（不免费）")
    ap.add_argument("--contest-enabled", dest="contest_enabled", action="store_true",
                    help="争夺食物战（RHP；默认关 = 旧行为；S3 接线）")
    ap.add_argument("--holder-adv", dest="holder_adv", type=float, default=1.2,
                    help="持有者优势（Parker 1974；R161 裁定默认 1.2 —— 0.3 被 RHP 的 (0.3+g16) 淹没）")
    ap.add_argument("--escalation-gap", dest="escalation_gap", type=float, default=0.25,
                    help="不升级的 RHP 差阈值（只有接近才升级）")
    ap.add_argument("--contest-cost-energy", dest="contest_cost_energy", type=float, default=0.5,
                    help="驱逐战的代价（防'免费赶人'）")
    ap.add_argument("--w-fear-health", dest="w_fear_health", type=float, default=0.5,
                    help="血条恐惧项权重（低血条 ⇒ 更恐惧；能力导向）")
    ap.add_argument("--wound-fear-threshold", dest="wound_fear_threshold", type=float,
                    default=0.3,
                    help="血条恐惧触发门槛（1−health ≥ 此值才生效；13.4 波 3）")
    ap.add_argument("--need-aggression-k", dest="need_aggression_k", type=float, default=0.5,
                    help="饥饿激进项强度（固定 0.5；D 臂设 0 = 关'饥饿更激进'）")
    # ---- 13.4 波 1：亚格连续坐标（R169/R175；**默认全关 = 旧行为**）----
    # 臂身份必须能从产物自证（C4）⇒ 全部进 `switches` 读回。
    ap.add_argument("--subpos-enabled", dest="subpos_enabled", action="store_true",
                    help="亚格连续坐标（默认关 = 旧行为，逐位等价；与 l2_dash 互斥 H3）")
    ap.add_argument("--subpos-subdiv", dest="subdiv", type=int, default=4,
                    help="每格 4×4=16 亚位置 ⇒ 最小步长 0.25 格")
    ap.add_argument("--subpos-speed-gain", dest="speed_gain", type=float, default=4.0,
                    help="speed = clamp(mob_eff×gain, 0, speed_max)；4.0 ≈ 历史 1+dash_frac")
    ap.add_argument("--subpos-speed-max", dest="speed_max", type=float, default=2.0,
                    help="速度上限（格/tick）；=2 使 subpos 包含 L2 冲刺语义")
    ap.add_argument("--subpos-min-energy-frac", dest="min_energy_frac", type=float,
                    default=0.05,
                    help="移动能量门槛（纯能量阈值；0.05×max_energy=15，拦'真要饿死'）")
    ap.add_argument("--subpos-lat-floor", dest="lat_floor", type=float, default=0.3,
                    help="极区移速折减下限（speed_cap = speed_max×(lat_floor+(1−lat_floor)cos)）")
    ap.add_argument("--subpos-stay-base", dest="stay_base", type=float, default=0.0,
                    help="停留概率基座（默认 0 = 默认走；语义反转）")
    ap.add_argument("--subpos-stay-food-k", dest="stay_food_k", type=float, default=0.0,
                    help="本格还有余粮 ⇒ 停着吃（权重）")
    ap.add_argument("--subpos-stay-signal-k", dest="stay_signal_k", type=float, default=0.0,
                    help="收到信号 ⇒ 停（权重）")
    ap.add_argument("--subpos-stay-fear-k", dest="stay_fear_k", type=float, default=0.0,
                    help="邻域有威胁 ⇒ 停（权重；本波不接线，只读回）")
    ap.add_argument("--subpos-stay-max", dest="stay_max", type=float, default=0.8,
                    help="停留概率上限（反退化闸：任何个体至少 20% 概率移动）")
    # ---- 13.4 波 2A：资源动态（T2；**默认关 = 旧行为**）----
    ap.add_argument("--resource-dynamics-enabled", dest="resource_dynamics_enabled",
                    action="store_true",
                    help="斑块休耕—死亡—轮作（默认关 = 旧行为，逐位等价；H3 拦 Rust）")
    ap.add_argument("--rest-ticks", dest="rest_ticks", type=int, default=300,
                    help="被吃后休耕 tick（该格 N tick 内再生=0）")
    ap.add_argument("--kill-frac", dest="kill_frac", type=float, default=10.0,
                    help="被吃强度 > 此值（= kill_mult，**当期再生倍数**；R178 裁定 10）⇒ 斑块死亡")
    ap.add_argument("--kill-denom", dest="kill_denom", default="regrowth",
                    choices=("regrowth", "capacity"),
                    help="kill 分母：regrowth=当期再生（R178 裁定）/ capacity=设计稿字面")
    ap.add_argument("--dead-regen-ticks", dest="dead_regen_ticks", type=int, default=2000,
                    help="死格重入候选池等待（0 = 硬拒绝：不可逆荒漠化）")
    ap.add_argument("--dead-cell-max-frac", dest="dead_cell_max_frac", type=float,
                    default=0.5,
                    help="反荒漠化闸：死格占比超它 ⇒ 强制加速重生")
    ap.add_argument("--kill-patch-only", dest="kill_patch_only", default="true",
                    choices=("true", "false"),
                    help="只有斑块格会死（防背景格大范围被打散）")
    ap.add_argument("--rotate-same-row-only", dest="rotate_same_row_only", default="true",
                    choices=("true", "false"),
                    help="只在同行交换斑块加成（跨行破坏 Σcapacity 守恒）")
    # ---- 13.4 波 2B（T3）：视野 2 格 / 单格上限 / 社交归一化 ----
    ap.add_argument("--perception-span", dest="perception_span", type=int, default=1,
                    choices=(1, 2),
                    help="感知半径（跳数）：1=默认（旧行为）/ 2=两圈（构造级）")
    ap.add_argument("--cell-occupancy-cap", dest="cell_occupancy_cap", type=int, default=3,
                    help="单格个体上限（score 层剔除满格；落本格不受限）")
    ap.add_argument("--cell-occupancy-cap-enabled", dest="cell_occupancy_cap_enabled",
                    action="store_true",
                    help="单格上限开关（默认关 = 旧行为，逐位等价）")
    ap.add_argument("--social-norm", dest="social_norm", default="auto",
                    help="社交项归一化除数：auto=每格实际邻居数（F1 修复）/ 数字=冻结常量")
    ap.add_argument("--init-g16-clusters", dest="init_g16_clusters", default="",
                    help="R141 P0-2：g16 初始投放（逗号分隔，按簇等分人口）；"
                         "空=旧行为。校准批用 \"0.05,0.5,0.9\"")
    ap.add_argument("--forage-tradeoff-k", dest="forage_tradeoff_k", type=float, default=0.0,
                    help="R135 第3步 A-连续：取食倍率 (1−g16)^k（凸 trade-off）；"
                         "0=关（默认，与旧版逐位一致）；本批取 2.0（凸/加速下降 ⇒ 中间态杂食者吃亏）")
    # ---- R152/P0（2026-09-22）：**捕食生态位结构** 6 个参数上 CLI -----------------
    # 🔴 目的：P0 要测"给捕食者真实代价 + 让中间态最差"，而这些值此前**只能改源码**
    #    ⇒ 无法做臂间对照（F1 家族："传了开关却没生效"的可预防形态）。
    # ⚠️ 全部**默认 = 旧行为**（逐位一致）：0.1 / 0.4 / 0.3 / 0.2 / 0.1 / 0.9。
    ap.add_argument("--attack-cost", dest="attack_cost", type=float, default=0.1,
                    help="每次**真出手**的能耗（现 0.1 ≈ 代谢的 0.1%%；P0 拟 1.0 = 高风险）")
    ap.add_argument("--transfer-ratio", dest="transfer_ratio", type=float, default=0.4,
                    help="捕食成功抢走猎物能量比例（现 0.4；P0 拟 0.8）")
    ap.add_argument("--attack-gate", dest="attack_gene_gate", type=float, default=0.3,
                    help="攻击性低于该值不发动攻击（现 0.3；P0 拟 0.15）")
    ap.add_argument("--attack-prob-coef", dest="attack_prob_coef", type=float, default=0.2,
                    help="出手概率 ≈ 攻击性 × 该系数 × 饥饿度（现 0.2；P0 拟 0.5）")
    ap.add_argument("--success-floor", dest="success_floor", type=float, default=0.1,
                    help="成功率下限（现 0.1；P0 拟 0.35）")
    ap.add_argument("--success-ceil", dest="success_ceil", type=float, default=0.9,
                    help="成功率上限（现 0.9；P0 拟 0.95）")
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
                  forage_tradeoff_k=args.forage_tradeoff_k,
                  init_g16_clusters=args.init_g16_clusters,
                  energy_cap=(args.energy_cap == "true"),
                  photo_max=args.photo_max,
                  # R146/R149（R150 B1）：L1/L2 臂身份（默认全关 = 旧行为）
                  l1_seek=bool(args.l1_seek), l1_fear=bool(args.l1_fear),
                  l2_dash=bool(args.l2_dash), w_seek_max=args.w_seek_max,
                  w_fear=args.w_fear, l1_prey_mode=args.l1_prey_mode,
                  # S1 骨架：尸体—食腐 + 血条—受伤（默认全关 = 旧行为）
                  corpse_enabled=bool(args.corpse_enabled),
                  corpse_energy_frac=args.corpse_energy_frac,
                  corpse_decay_ticks=args.corpse_decay_ticks,
                  corpse_to_plant_frac=args.corpse_to_plant_frac,
                  corpse_patch_boost=args.corpse_patch_boost,
                  corpse_cap_per_cell=args.corpse_cap_per_cell,
                  scav_gate=args.scav_gate, scav_s=args.scav_s,
                  wound_enabled=bool(args.wound_enabled),
                  wound_base=args.wound_base,
                  wound_heal_rate=args.wound_heal_rate,
                  wound_heal_energy_cost=args.wound_heal_energy_cost,
                  contest_enabled=bool(args.contest_enabled),
                  holder_adv=args.holder_adv, escalation_gap=args.escalation_gap,
                  contest_cost_energy=args.contest_cost_energy,
                  w_fear_health=args.w_fear_health,
                  need_aggression_k=args.need_aggression_k,
                  wound_fear_threshold=args.wound_fear_threshold,
                  # 13.4 波 1：亚格连续坐标（默认全关 = 旧行为）
                  subpos_enabled=bool(args.subpos_enabled),
                  subdiv=args.subdiv, speed_gain=args.speed_gain,
                  speed_max=args.speed_max,
                  min_energy_frac=args.min_energy_frac,
                  lat_floor=args.lat_floor,
                  stay_base=args.stay_base, stay_food_k=args.stay_food_k,
                  stay_signal_k=args.stay_signal_k, stay_fear_k=args.stay_fear_k,
                  stay_max=args.stay_max,
                  # 13.4 波 2A：资源动态（默认关 = 旧行为）
                  resource_dynamics_enabled=bool(args.resource_dynamics_enabled),
                  rest_ticks=args.rest_ticks, kill_frac=args.kill_frac,
                  kill_denom=args.kill_denom,
                  dead_regen_ticks=args.dead_regen_ticks,
                  dead_cell_max_frac=args.dead_cell_max_frac,
                  kill_patch_only=(args.kill_patch_only == "true"),
                  rotate_same_row_only=(args.rotate_same_row_only == "true"),
                  # 13.4 波 2B（T3）：视野 / 单格上限 / 社交归一化
                  perception_span=args.perception_span,
                  cell_occupancy_cap=args.cell_occupancy_cap,
                  cell_occupancy_cap_enabled=bool(args.cell_occupancy_cap_enabled),
                  social_norm=args.social_norm,
                  # R152/P0：捕食生态位结构 6 参
                  attack_cost=args.attack_cost, transfer_ratio=args.transfer_ratio,
                  attack_gene_gate=args.attack_gene_gate,
                  attack_prob_coef=args.attack_prob_coef,
                  success_floor=args.success_floor, success_ceil=args.success_ceil)
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
        # R146/R149：L1/L2 开关是**臂身份**（A 臂 vs B 臂的唯一差别）⇒ 续跑时命令行若与
        # 快照不符，必须**硬失败**而非静默沿用快照（同 F-R21/C5 家族："传了开关没生效"）。
        # 段二正是"从段一快照续跑"⇒ 这条检查就是段二不错臂的机器保证。
        for _k, _cli in (("l1_seek", bool(args.l1_seek)), ("l1_fear", bool(args.l1_fear)),
                         ("l2_dash", bool(args.l2_dash))):
            _snap_v = bool(getattr(e.config.simulation, _k))
            if _cli != _snap_v:
                raise SystemExit(
                    f"{_k} 冲突：命令行 {_cli} vs 快照 {_snap_v} —— 臂身份不得静默混用"
                    "（段二续跑必须与段一同臂）"
                )
        if str(args.l1_prey_mode) != str(e.config.simulation.l1_prey_mode):
            raise SystemExit(
                f"l1_prey_mode 冲突：命令行 {args.l1_prey_mode!r} vs "
                f"快照 {e.config.simulation.l1_prey_mode!r}（E′ 臂身份）"
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
              # R139/R140 派工 **P0 观测列**（2026-09-21）：
              # ① `g16_h0..h9` = g16 的 **10-bin 直方图**（值域 [0,1] 均分）——
              #    ⚠️ 没有它，**置换零分布无从下手** ⇒ 双峰永远不可判（E-030 的教训：
              #    只存矩 ⇒ shuffle/重算全落空；且 BC 单用已证明是误判机器）。
              # ② `mean_energy` —— E_prey 推算的最大不确定源（R140 §四），优先级最高。
              # ③ 死因**时间序列**（累计值，差分可得区间）：看死因结构随相位怎么变。
              # ④ `g3` 寿命基因均值 —— 验证"长寿命是否被选择"（maturity ∝ lifespan ⇒ 应有晚熟代价）。
              *[f"g16_h{i}" for i in range(G16_BINS)],
              "mean_energy", "d_starv", "d_pred", "d_old", "g3",
              # R141 P0 派工单 §1.3（🔴 **列名锁定**——`tools/calib_solve.py` 按此消费，勿改名）
              # 三腿 = g16 三分箱 lo/mid/hi；net = forage + pred − meta − move − attack（不含光合）
              "g_lo_n", "g_lo_net_mean", "g_lo_net_p50", "g_lo_net_var",
              "g_mid_n", "g_mid_net_mean", "g_mid_net_p50", "g_mid_net_var",
              "g_hi_n", "g_hi_net_mean", "g_hi_net_p50", "g_hi_net_var",
              "forage_in_mean", "pred_in_mean", "prey_energy_mean",
              # R145 补丁②：囤积**现象**占比（R147 §二 发现 2 改名为 over_cap_frac，
              # 以区别于**仪器**口径 cap_residual_frac —— 两者正交，勿混读）
              "over_cap_frac",
              "max_gen", "max_gen_cur",       # R77：高水位 / 当刻最深（两个口径分列）
              "mean_row", "polar_frac",
              "codebook_conv", "pred_frac",   # D-16：R31③/R38③ 判据列
              "resp_a", "resp_b", "oracle_ratio",   # D-18⑥/D-8（累计口径）
              "mem_bit_frac",   # R128 §五 步骤 0：mem_bit 取值分布的**时间序列**（累计口径）
              # A′（2026-09-19）：**先证"测到了"**再判读 ⇒ 两个可观测性占比（累计口径）
              "mem_grad_slots_frac", "mem_grad_trig_frac",
              # R146/R149 L1/L2（R150 B4）：闸门读数的**时间序列**（累计口径；关档 ⇒ 空串
              # = **未适用**，不是 0）。`seek_zero_frac` 高 = **自熄**（邻域无猎物代理）——
              # 预注册允许结局「猎物池枯竭」，**不得**被读成"没接线"。
              "seek_term_mean", "seek_zero_frac", "fear_term_mean",
              "dash_frac", "mob_eff_mean",
              # S1 骨架（设计稿 §5.2 项 10）：尸体—食腐 + 血条 CSV 列
              # （关档 = 空壳读数：corpse_total/corpse_eaten/wound_n/contest_n 恒 0，
              #   health_mean 恒 1.0 —— S2/S3 接线后才非平凡）
              "corpse_total", "corpse_eaten",
              "health_mean", "health_low_frac", "wound_n", "contest_n",
              # S3 交互（设计稿 §5.4 项 6/7）：争夺战持有者胜率 + 血条恐惧项反退化
              "contest_win_by_holder_frac", "fear_health_flat_frac",
              # 13.4 波 1（R169/R175）：亚格坐标读数（关档 ⇒ 空串 = 未适用，R120 口径）
              "subpos_flat_moves", "subpos_slow_frac",
              # 13.4 波 2A（T2，R176 §12.5 四条）：资源动态读数（关档 ⇒ 空串 = 未适用）
              "dead_cell_frac", "resting_cell_frac",
              "patch_kill_n", "patch_reborn_n", "mean_capacity_effective",
              # 13.4 波 2B（T3）：视野/单格上限读数（关档 ⇒ 空串 = 未适用）
              "span_downgrade_frac", "cap_blocked_n", "cap_stay_n"]
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
            # R140 P0：死因**时间序列**（累计口径，差分可得区间值）。
            # 键名按 `DeathCause` 成员名归一化（枚举 str() 形如 "DeathCause.STARVATION"）。
            _dct = {str(k).split(".")[-1].upper(): int(v)
                    for k, v in e.death_cause_totals().items()}
            # R146/R149 L1/L2 读数（R150 B4）：**每次只取一次探针**（关档 ⇒ None ⇒ 空串）
            _l1p, _l2p = e.l1_probe(), e.l2_probe()
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
                **({} if P == 0 else dict(zip(
                    (f"g16_h{i}" for i in range(G16_BINS)), _hist10(e._genes[:P, 16])))),
                "mean_energy": round(float(e._energy[:P].mean()), 4) if P else "",
                "d_starv": _dct.get("STARVATION", 0),
                "d_pred": _dct.get("PREDATION", 0),
                "d_old": _dct.get("OLD_AGE", 0),
                "g3": round(float(e._genes[:P, 3].mean()), 4) if P else "",
                # 囤积**现象**占比（单一出处 = 引擎 `over_cap_frac()`；P=0 ⇒ None ⇒ 写空）
                "over_cap_frac": ("" if e.over_cap_frac() is None else e.over_cap_frac()),
                # R141 P0：逐 tick 净收入统计（`_ec_flush` 每 tick 追加一行；n=0 ⇒ None ⇒ 写空）
                **_ec_csv_row(e),
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
                # R146/R149 L1/L2（R150 B4）：闸门读数的时间序列（关档 ⇒ 空串 = 未适用）
                "seek_term_mean": _probe_csv(_l1p, "seek_term_mean"),
                "seek_zero_frac": _probe_csv(_l1p, "seek_zero_frac"),
                "fear_term_mean": _probe_csv(_l1p, "fear_term_mean"),
                "dash_frac": _probe_csv(_l2p, "dash_frac"),
                "mob_eff_mean": _probe_csv(_l2p, "mob_eff_mean"),
                # S1 骨架（设计稿 §5.2 项 10）：尸体—食腐 + 血条读数（空壳口径，
                # 同 result.corpse/result.wound —— S1 允许值可为 0）
                "corpse_total": round(float(e._corpse_energy.sum()), 6),
                "corpse_eaten": int(e._corpse_eaten_n),
                "health_mean": (round(float(e._health[:P].mean()), 4) if P else ""),
                "health_low_frac": (
                    round(float((e._health[:P] < 0.5).mean()), 4) if P else ""
                ),
                "wound_n": int(e._wound_n),
                "contest_n": int(e._contest_n),
                # S3 交互（设计稿 §5.4 项 6/7）：争夺战持有者胜率 + 血条恐惧项反退化
                # （None ⇒ 空串 = 未适用，R120 口径）
                "contest_win_by_holder_frac": _probe_csv(
                    e.wound_probe(), "contest_win_by_holder_frac"),
                "fear_health_flat_frac": _probe_csv(
                    e.wound_probe(), "fear_health_flat_frac"),
                # 13.4 波 1：亚格坐标读数（关档 ⇒ None ⇒ 空串 = 未适用）
                "subpos_flat_moves": _probe_csv(e.subpos_probe(), "mean_flat_moves"),
                "subpos_slow_frac": _probe_csv(e.subpos_probe(), "slow_frac"),
                # 13.4 波 2A：资源动态四条（关档 ⇒ None ⇒ 空串 = 未适用）
                "dead_cell_frac": _probe_csv(
                    e.resource_dynamics_probe(), "dead_cell_frac"),
                "resting_cell_frac": _probe_csv(
                    e.resource_dynamics_probe(), "resting_cell_frac"),
                "patch_kill_n": _probe_csv(
                    e.resource_dynamics_probe(), "patch_kill_n"),
                "patch_reborn_n": _probe_csv(
                    e.resource_dynamics_probe(), "patch_reborn_n"),
                "mean_capacity_effective": _probe_csv(
                    e.resource_dynamics_probe(), "mean_capacity_effective"),
                # 13.4 波 2B（T3）：视野/单格上限（关档 ⇒ None ⇒ 空串）
                "span_downgrade_frac": _probe_csv(
                    e.wave2b_probe(), "span_downgrade_frac"),
                "cap_blocked_n": _probe_csv(e.wave2b_probe(), "cap_blocked_n"),
                "cap_stay_n": _probe_csv(e.wave2b_probe(), "cap_stay_n"),
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
            # R152/P0：捕食生态位结构 6 参必须可读回（C4）—— 臂间差就在这几个键上，
            # 缺席 ⇒ 外复核只能靠 preset 名（R136 §一 增量 2 的同型缺口）
            "attack_cost": float(e.config.predation.attack_cost),
            "transfer_ratio": float(e.config.predation.transfer_ratio),
            "attack_gene_gate": float(e.config.predation.attack_gene_gate),
            "attack_prob_coef": float(e.config.predation.attack_prob_coef),
            "success_floor": float(e.config.predation.success_floor),
            "success_ceil": float(e.config.predation.success_ceil),
            # R141 P0-2：初始投放必须可读回（C4）
            "init_g16_clusters": str(e.config.genome.init_g16_clusters),
            # R144/R145：能量封顶与光合必须可读回（C4；且封顶开关进指纹 ⇒ 纪元可判）
            "energy_cap_enabled": bool(e.config.organisms.energy_cap_enabled),
            "photo_max": float(e.config.organisms.photo_max),
            # 🔴 R136 §一 增量 2（C4 自证缺口）：PC-1 三臂的 `switches.arm` **全为 main**，
            #    码本/瓶颈两个开关读不到 ⇒ **臂间开关差无法从产物自证**（外复核只能靠 preset 名）。
            "arbitrary_codebook": bool(e.config.info_structure.arbitrary_codebook),
            "learning_bottleneck": bool(e.config.info_structure.learning_bottleneck),
            "use_sim_core": bool(e.config.simulation.use_sim_core),
            # ---- R146/R149 L1/L2（R150 B1/B2）：**臂身份必须能从产物自证**（C4）----
            # A 臂 vs B/C/D/E′ 的唯一差别就在这几个键上 ⇒ 缺席 ⇒ 外复核只能靠 preset 名
            # （R136 §一 增量 2 的同型缺口）。
            "l1_seek": bool(e.config.simulation.l1_seek),
            "l1_fear": bool(e.config.simulation.l1_fear),
            "l2_dash": bool(e.config.simulation.l2_dash),
            "w_seek_max": float(e.config.simulation.w_seek_max),
            "w_fear": float(e.config.simulation.w_fear),
            "l1_prey_mode": str(e.config.simulation.l1_prey_mode),
            # ---- S1 骨架（设计稿 §三/§5.2 项 9）：尸体—食腐 + 血条（C4 读回）----
            # 臂间差（B/C/D/E 臂）就落在这几个键上 ⇒ 缺席 ⇒ 外复核只能靠 preset 名
            # （R136 §一 增量 2 的同型缺口）。
            "corpse_enabled": bool(e.config.corpse_wound.corpse_enabled),
            "corpse_energy_frac": float(e.config.corpse_wound.corpse_energy_frac),
            "corpse_decay_ticks": int(e.config.corpse_wound.corpse_decay_ticks),
            "corpse_to_plant_frac": float(e.config.corpse_wound.corpse_to_plant_frac),
            "corpse_patch_boost": float(e.config.corpse_wound.corpse_patch_boost),
            "corpse_cap_per_cell": int(e.config.corpse_wound.corpse_cap_per_cell),
            # 🔴 R165 0-1（2026-09-23，老工）：**单位自证** —— 尸体池以**能量**记，
            #    食腐入胃处按 ÷`eat_efficiency` 折算成质量。13.3 批（p1corpse）
            #    无此标记且用了"池当质量"的旧口径（×3 放大）⇒ 能量层读数不可跨纪比较。
            "corpse_pool_unit": str(e.config.corpse_wound.corpse_pool_unit),
            "scav_to_energy_divisor": float(e.config.organisms.eat_efficiency),
            "scav_gate": float(e.config.corpse_wound.scav_gate),
            "scav_s": float(e.config.corpse_wound.scav_s),
            "wound_enabled": bool(e.config.corpse_wound.wound_enabled),
            "wound_base": float(e.config.corpse_wound.wound_base),
            "wound_heal_rate": float(e.config.corpse_wound.wound_heal_rate),
            "wound_heal_energy_cost": float(e.config.corpse_wound.wound_heal_energy_cost),
            "contest_enabled": bool(e.config.corpse_wound.contest_enabled),
            "holder_adv": float(e.config.corpse_wound.holder_adv),
            "escalation_gap": float(e.config.corpse_wound.escalation_gap),
            "contest_cost_energy": float(e.config.corpse_wound.contest_cost_energy),
            "w_fear_health": float(e.config.corpse_wound.w_fear_health),
            "need_aggression_k": float(e.config.corpse_wound.need_aggression_k),
            "wound_fear_threshold": float(e.config.corpse_wound.wound_fear_threshold),
            # ---- 13.4 波 1：亚格连续坐标（C4 读回；臂身份 = `subpos_enabled`）----
            # 关档 = 旧行为 ⇒ 这些键仍是"默认值读回"，不叫"未适用"（开关可读回是硬要求）。
            "subpos_enabled": bool(e.config.subpos.enabled),
            "subpos_subdiv": int(e.config.subpos.subdiv),
            "subpos_speed_gain": float(e.config.subpos.speed_gain),
            "subpos_speed_max": float(e.config.subpos.speed_max),
            "subpos_min_energy_frac": float(e.config.subpos.min_energy_frac),
            "subpos_lat_floor": float(e.config.subpos.lat_floor),
            "subpos_stay_base": float(e.config.subpos.stay_base),
            "subpos_stay_food_k": float(e.config.subpos.stay_food_k),
            "subpos_stay_signal_k": float(e.config.subpos.stay_signal_k),
            "subpos_stay_fear_k": float(e.config.subpos.stay_fear_k),
            "subpos_stay_max": float(e.config.subpos.stay_max),
            # ---- 13.4 波 2A：资源动态（C4 读回；臂身份 = `resource_dynamics_enabled`）----
            "resource_dynamics_enabled": bool(e.config.resource_dynamics.enabled),
            "rest_ticks": int(e.config.resource_dynamics.rest_ticks),
            "kill_frac": float(e.config.resource_dynamics.kill_frac),
            "kill_denom": str(e.config.resource_dynamics.kill_denom),
            "dead_regen_ticks": int(e.config.resource_dynamics.dead_regen_ticks),
            "dead_cell_max_frac": float(e.config.resource_dynamics.dead_cell_max_frac),
            "kill_patch_only": bool(e.config.resource_dynamics.kill_patch_only),
            "rotate_same_row_only": bool(
                e.config.resource_dynamics.rotate_same_row_only),
            # ---- 13.4 波 2B（T3）：视野 / 单格上限 / 社交归一化（C4 读回）----
            "perception_span": int(e.config.simulation.perception_span),
            "perception_cap": int(getattr(e.config.simulation, "perception_cap", 32)),
            "cell_occupancy_cap": int(e.config.simulation.cell_occupancy_cap),
            "cell_occupancy_cap_enabled": bool(
                e.config.simulation.cell_occupancy_cap_enabled),
            "social_norm": str(e.config.simulation.social_norm),
            # L2 几何/成本参数（「参数制造分化」的可核查性）
            "dash_min_energy_frac": float(e.config.organisms.dash_min_energy_frac),
            "dash_cost_kappa": float(e.config.organisms.dash_cost_kappa),
            "dash_cost_exp": float(e.config.organisms.dash_cost_exp),
            "young_mob_mult": float(e.config.organisms.young_mob_mult),
            "old_mob_mult": float(e.config.organisms.old_mob_mult),
            "far_cap": int(e.config.organisms.far_cap),
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
            # R135 第 -1 步④：互捕结构量化（攻击者/猎物 g16 直方图 + Δ + 真决斗三级拆分）
            "cannibalism": e.cannibalism_stats(),
            # R144/R145：能量封顶探针 + 状态量边界自检
            "energy_cap": e.energy_cap_probe(),
            # R146/R149（R150 B1/B2）：L1 两项 + L2 机动性读数（**关档 ⇒ None = 未适用**）
            "l1": e.l1_probe(),
            "l2": e.l2_probe(),
            "bounds": e.state_bounds_check(),
            # S1 骨架（设计稿 §5.2 项 9）：尸体—食腐 + 血条读数块（**空壳**，值可为 0）
            "corpse": e.corpse_probe(),
            "wound": e.wound_probe(),
            # 13.4 波 1：亚格坐标读数（关档 ⇒ None = 未适用，R120 口径）
            "subpos": e.subpos_probe(),
            # 13.4 波 2A：资源动态读数（关档 ⇒ None = 未适用）+ 守恒自检（恒应过）
            "resource_dynamics": e.resource_dynamics_probe(),
            "resource_dynamics_conservation": e.resource_dynamics_conservation(),
            # 13.4 波 2B（T3）：视野/单格上限读数
            "wave2b": e.wave2b_probe(),
            # R141 P0（派工单 §1.3，🔴 段名与结构锁定 —— `calib_solve.py` 按此消费）
            "energy_ledger": e.energy_ledger(),
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
