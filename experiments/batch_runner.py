"""D-25 并行跑批包装（R48 新纪律：单机实验一律走并行包装）。

为什么需要它
------------
R48：并行化是**唯一必要手段**——152 h → 约 4 h，零语义风险，成本极低。
R48 新纪律：**单机实验一律走并行包装**，并发上限**按内存定**（须留 ≥1 GB 余量）。

相比旧 `run_d2x_parallel.ps1` 的改进
------------------------------------
1. **跨平台**（纯 Python / 无 psutil 依赖；Windows 用 GlobalMemoryStatusEx，Linux 读 /proc/meminfo）
2. **并发按内存动态定**，不是写死 20（R48）
3. **内存守卫**：启动前 + 运行中持续检查，低于阈值就不再拉起新 run（只排队，不杀已有进程）
4. **失败重试**：退出码≠0 或 summary 缺失 ⇒ 自动重试 N 次
5. **`--skip-existing`：已完成且 summary 合法的 run 直接跳过** ⇒ 断点续批，
   避免"8 臂已跑完却重跑 10 臂"这类浪费（R19 教训）
6. **汇总**：批结束后读所有 summary → 分层表（R38③：`pred_frac` 分层，不得合并均值）
   → 落 `_summary.csv` / `_summary.md`，并记录墙钟（供 R50 速率校准）

用法
----
    # 预设：R19 验收批（cb0/cb1 × seed 42-46 × 60k）
    python experiments/batch_runner.py --preset r19 --skip-existing

    # 自定义网格
    python experiments/batch_runner.py \\
        --script experiments/a4_verify_capacity.py \\
        --grid codebook=0,1 seed=42,43,44,45,46 \\
        --fixed mode=on ticks=60000 snapshot-every=5000 fresh \\
        --out-template "_rerun_logs/a4_fix/asym_on_cb{codebook}_s{seed}.csv"

    # 只看看会跑什么（不执行）
    python experiments/batch_runner.py --preset r19 --dry-run

退出码：0 = 全部成功；1 = 有 run 失败（已耗尽重试）；2 = 用法/环境错误。
"""
from __future__ import annotations

import argparse
import ctypes
import csv
import glob
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print 会让脚本 rc=1 假失败；F-R15 族）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # 非 TTY / 旧解释器：不因诊断能力缺失而阻断运行

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------- 内存（零依赖）

if sys.platform == "win32":

    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    def memory_gb() -> tuple[float, float, int]:
        """返回 (可用GB, 总GB, 负载%)。"""
        m = _MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return (m.ullAvailPhys / 1024**3, m.ullTotalPhys / 1024**3, int(m.dwMemoryLoad))

else:

    def memory_gb() -> tuple[float, float, int]:
        info = {}
        try:
            for line in open("/proc/meminfo", encoding="utf-8"):
                k, _, v = line.partition(":")
                info[k.strip()] = int(v.split()[0]) * 1024
        except Exception:
            return (0.0, 0.0, 0)
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", info.get("MemFree", 0))
        load = int(100 * (total - avail) / total) if total else 0
        return (avail / 1024**3, total / 1024**3, load)


def cpu_count() -> int:
    return os.cpu_count() or 1


def pick_concurrency(per_run_gb: float, reserve_gb: float, cap: int | None) -> int:
    """并发上限 = min(cap, cpu-1, 按内存算)。R48：必须留 ≥ reserve_gb 余量。"""
    avail, _total, _load = memory_gb()
    by_mem = int(max(1, (avail - reserve_gb) // per_run_gb)) if per_run_gb > 0 else 1
    n = min(by_mem, max(1, cpu_count() - 1))   # 留 1 核给系统/汇总
    if cap:
        n = min(n, cap)
    return max(1, n)


# ---------------------------------------------------------------- run 规格

@dataclass
class Run:
    name: str
    cmd: list[str]
    out: Path
    summary: Path
    attempt: int = 0
    status: str = "pending"      # pending | running | done | failed | skipped
    wall: float = 0.0
    note: str = ""
    proc: subprocess.Popen | None = field(default=None, repr=False)

    @property
    def log(self) -> Path:
        return self.out.with_suffix(self.out.suffix + f".attempt{self.attempt}.log")


def summary_ok(p: Path) -> bool:
    """summary 合法 = 存在 + 可解析 + 有 final_N + **跑满目标 tick**。

    ⚠️ 判据必须是"跑满目标 tick"而不是"跑满 60k / 通过生态门"：
    小规格批（R47 ≈15 run）与 D-25 自测都跑不满 60k，若用 eco_gate 判据会把
    它们一律误判为失败并无限重试。**灭绝提前结束（final_tick < 目标）才算失败**。
    """
    if not p.exists():
        return False
    try:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
    except Exception:
        return False
    r = d.get("result", {})
    if "final_N" not in r:
        return False
    target = d.get("switches", {}).get("ticks_target")
    ft = r.get("final_tick")
    if target is None or ft is None:
        return bool(r.get("eco_gate_pass"))
    try:
        return int(ft) >= int(target)
    except Exception:
        return False


def cli_flag(key: str) -> str:
    """preset/网格的**参数名 → CLI 开关**（F-R21）。

    🔴 **F-R21（2026-09-16，R108 立号）**：`r97cal` preset 的网格键写作 `gain_multiplier`
    （下划线，Python 风格），而 `expand()` 原来直接拼 `--{key}` ⇒ 生成 `--gain_multiplier`，
    但 `argparse` 注册的是 `--gain-multiplier` ⇒ **参数不被识别**，云端只能绕行 shell 执行
    （无数据损失，但 preset 事实上不可直接跑）。
    ⇒ 统一规则：**grid/fixed 的键一律用下划线写法**（便于做 `template.format()` 占位符），
    **拼 CLI 时把 `_` 换成 `-`**。`template` 仍用**原始键**（下划线）作占位符 ⇒ 旧模板不受影响。
    """
    return "--" + key.replace("_", "-")


def expand(grid: list[str], fixed: list[str], template: str | None, script: str,
           workdir: Path, variants: list[dict] | None = None) -> list[Run]:
    """笛卡尔积展开网格 → Run 列表。

    键名规则见 `cli_flag()`：**键用下划线、CLI 用短横线**（F-R21）。

    `variants`（R109 §五 新增，可空）
    --------------------------------
    一个批里要跑**参数互异的多条臂**时（如 C 步 `oracle_m1.3` / `oracle_m1.0` / `zero`），
    单靠 `grid` 的笛卡尔积表达不了"每个取值配一组不同开关"。`variants` 即为此：
    每项 = `{"name": str, "args": [str, ...], "template": str}`，最终展开为
    **`variants × combos`**，各变体自带 `--out` 模板。

    🔴 **快照目录隔离（血泪教训）**：`--snapshot-dir` 若沿用默认 `_rerun_logs/snap/`，
    而新批的 run 名与**历史批**重名（如 `zero_s42` / `main_s42` 与 D-24 同名），
    引擎会**静默从旧快照"续跑"**（`--ticks` 小于已跑 tick 时循环体为空 ⇒ 空产出、假失败）。
    ⇒ 故新批一律显式给**专属快照目录**；本函数不代管，由 preset 的 `fixed` 指定。
    """
    g: dict[str, list[str]] = {}
    for item in grid:
        k, _, vs = item.partition("=")
        g[k.strip()] = [v for v in vs.split(",") if v != ""]

    keys = list(g)
    combos: list[dict[str, str]] = [{}]
    for k in keys:
        combos = [dict(c, **{k: v}) for c in combos for v in g[k]]

    fixed_pairs: list[tuple[str, str]] = []
    for item in fixed:
        k, sep, v = item.partition("=")
        fixed_pairs.append((k.strip(), v.strip() if sep else ""))

    # ---- 变体归一：无 variants ⇒ 单一"空变体"（保持旧行为逐位不变）----
    norm_variants: list[dict] = [{"name": None, "pairs": [], "template": template}]
    if variants:
        norm_variants = []
        for v in variants:
            vp: list[tuple[str, str]] = []
            for item in v.get("args", []):
                k, sep, val = item.partition("=")
                vp.append((k.strip(), val.strip() if sep else ""))
            norm_variants.append({"name": v.get("name"), "pairs": vp,
                                  "template": v["template"]})

    runs: list[Run] = []
    for var in norm_variants:
        for c in combos:
            out_rel = var["template"].format(**c)
            out = workdir / out_rel
            args = [script, "--out", out_rel]
            for k in keys:                   # 网格参数在 out 模板里用到，也传给脚本
                args += [cli_flag(k), c[k]]
            for k, val in fixed_pairs:
                args.append(cli_flag(k))     # F-R21：`_` → `-`（argparse 只认短横线）
                if val:
                    args.append(val)
            for k, val in var["pairs"]:      # 变体专属开关（同一键可被变体覆盖）
                args.append(cli_flag(k))
                if val:
                    args.append(val)
            runs.append(Run(
                name=Path(out_rel).stem,
                cmd=args,
                out=out,
                summary=out.with_suffix(".summary.json"),
            ))
    return runs


def preset_runs(name: str, workdir: Path = Path(".")) -> list[Run]:
    """按 preset 名展开 run 列表（**CLI 与测试共用**，保证测的就是跑的那条路径）。"""
    p = PRESETS[name]
    return expand(p["grid"], p["fixed"], p.get("template"), p["script"],
                  workdir, p.get("variants"))


PRESETS = {
    # R19 验收批：V12 口径 4 机制主臂(cb1) + 3 机制对照(cb0)，seed 42-46，60k
    "r19": dict(
        script="experiments/a4_verify_capacity.py",
        grid=["codebook=0,1", "seed=42,43,44,45,46"],
        fixed=["mode=on", "ticks=60000", "snapshot-every=5000"],
        template="_rerun_logs/a4_fix/asym_on_cb{codebook}_s{seed}.csv",
    ),
    # D-24 小规格验证批（R47/R53）：5 臂 × 3 seed，max_count=3240（R41，⑤ 不饱和前提），
    # 60k tick，--arm 隐含开 ⑤观测+⑥探针（oracle 臂另开 oracle）
    "d24": dict(
        script="experiments/a4_verify_capacity.py",
        grid=["arm=main,control,zero,sigoff,oracle", "seed=42,43,44"],
        fixed=["mode=on", "ticks=60000", "max-count=3240", "snapshot-every=5000"],
        template="_rerun_logs/d24/{arm}_s{seed}.csv",
    ),
    # D-27④-A（R86 修订 / R91）：oracle **剂量-响应**四点测量。
    # 四点覆盖 ratio≥1.2 的求解区（R91 建议）；tick 固定 ⇒ 转化率可比（转化率随 run 变长而升）。
    "d27dose": dict(
        script="experiments/a4_verify_capacity.py",
        grid=["donation=0.674,1.0,1.5,2.0", "seed=42,43,44"],
        fixed=["mode=on", "arm=oracle", "ticks=8000", "max-count=3240",
               "snapshot-every=0"],
        template="_rerun_logs/d27/dose{donation}_s{seed}.csv",
    ),
    # D-27④-接收侧（内评《复核-R102方向修复》§三）：**同配置重跑**，唯一差别 = 引擎新增
    # 接收侧/对账仪器 ⇒ ① 得接收侧三条观察项 ② 与 d27dose 的 final_N/ratio **逐位对拍**
    #   ⇒ 证明「新仪器是纯观测」（不改变轨迹与随机流；F-R18 纪律的同族检查）。
    "d27recv": dict(
        script="experiments/a4_verify_capacity.py",
        grid=["donation=0.674,1.0,1.5,2.0", "seed=42,43,44"],
        fixed=["mode=on", "arm=oracle", "ticks=8000", "max-count=3240",
               "snapshot-every=0"],
        template="_rerun_logs/d27_recv/dose{donation}_s{seed}.csv",
    ),
    # R97 ⑤ **配对校准批**（R105 判据：R2 形态 + n=6）。
    # 设计：m ∈ {1.0（配对基线）, 1.3, 1.5} × seed ∈ {42..47} = **18 run**；
    #   全部登记 `--calibration-arm`（含 m=1.0 基线）⇒ **整批被 R100 条件 5 排除于科学判定**。
    #   donation 固定 1.0（R91 建议）、tick 固定 8000（与 ④ 批可比）。
    "r97cal": dict(
        script="experiments/a4_verify_capacity.py",
        grid=["gain_multiplier=1.0,1.3,1.5", "seed=42,43,44,45,46,47"],
        fixed=["mode=on", "arm=oracle", "donation=1.0", "ticks=8000",
               "max-count=3240", "snapshot-every=0", "calibration-arm"],
        template="_rerun_logs/r97cal/m{gain_multiplier}_s{seed}.csv",
    ),
    # R100 条件 7 **随机信号自检**：同档 m=1.3 + `signal_mode=random`（信号与个体状态无关 ⇒ 无信息）
    # ⇒ 若 ratio/ρ 同样上升 = 实证「增益不依赖信号内容」（V-1 C-4 的可执行检验）。6 run。
    "r97cal_rand": dict(
        script="experiments/a4_verify_capacity.py",
        grid=["seed=42,43,44,45,46,47"],
        fixed=["mode=on", "arm=oracle", "donation=1.0", "ticks=8000",
               "max-count=3240", "snapshot-every=0", "calibration-arm",
               "gain-multiplier=1.3", "signal-mode=random"],
        template="_rerun_logs/r97cal/rand_m1.3_s{seed}.csv",
    ),
    # ==================== R109 §五：C 步（本地直通车）三段 ====================
    # 共同口径：60k tick（R3 生态门本义适用）/ max_count=3240（R41 ⑤ 不饱和前提）/
    #   donation=1.0 / 快照每 5000 原地覆盖。
    # 🔴 **快照目录隔离**：`zero_s42` / `main_s42` 与 **D-24 批同名**，若沿用默认
    #   `_rerun_logs/snap/` 会**静默从 D-24 的 60k 快照续跑**（循环体为空 ⇒ 空产出）
    #   ⇒ 三段**一律**指定专属 `snapshot-dir=_rerun_logs/cstep_snap`（2026-09-16 立）。
    # C1a：仪器侧三条臂（处理 / 配对基线 / G-A 对照端点）⇒ 判 Q1（ratio 域内迁移性）
    # ⚠️ `donation` 必须放在**变体**里（不能进 `fixed`）：`a4` 对「非 oracle 臂收到
    #    oracle 专属参数」**硬失败**（防静默传参）⇒ `fixed` 里的 donation 会让 `zero` 臂
    #    rc≠0、无产出（2026-09-16 首跑实测：18 run 只起了 12 个）。
    "cstep1a": dict(
        script="experiments/a4_verify_capacity.py",
        grid=["seed=42,43,44,45,46,47"],
        fixed=["mode=on", "ticks=60000", "max-count=3240",
               "snapshot-every=5000", "snapshot-dir=_rerun_logs/cstep_snap"],
        template=None,
        variants=[
            dict(name="oracle_m1.3",
                 args=["arm=oracle", "donation=1.0", "gain-multiplier=1.3",
                       "calibration-arm"],
                 template="_rerun_logs/cstep1a/oracle_m1.3_s{seed}.csv"),
            dict(name="oracle_m1.0",
                 args=["arm=oracle", "donation=1.0", "gain-multiplier=1.0",
                       "calibration-arm"],
                 template="_rerun_logs/cstep1a/oracle_m1.0_s{seed}.csv"),
            dict(name="zero",
                 args=["arm=zero"],
                 template="_rerun_logs/cstep1a/zero_s{seed}.csv"),
        ],
    ),
    # C1b：科学臂（4 机制全开，m=1.0）⇒ 判 Q3/Q4（ρ 簇级 + 六门）
    # ⚠️ 同上：`main` 臂**不得**带 `donation`（非 oracle 臂 ⇒ a4 硬拒）。
    "cstep1b": dict(
        script="experiments/a4_verify_capacity.py",
        grid=["seed=42,43,44,45,46,47"],
        fixed=["mode=on", "arm=main", "ticks=60000",
               "max-count=3240", "snapshot-every=5000",
               "snapshot-dir=_rerun_logs/cstep_snap"],
        template="_rerun_logs/cstep1b/main_s{seed}.csv",
    ),
    # C2：条件 7 随机信号对照（同档 m=1.3 + signal_mode=random）⇒ 机制归因专项
    "cstep2rand": dict(
        script="experiments/a4_verify_capacity.py",
        grid=["seed=42,43,44,45,46,47"],
        fixed=["mode=on", "arm=oracle", "donation=1.0", "ticks=60000",
               "max-count=3240", "snapshot-every=5000",
               "snapshot-dir=_rerun_logs/cstep_snap",
               "gain-multiplier=1.3", "calibration-arm", "signal-mode=random"],
        template="_rerun_logs/cstep2rand/rand_m1.3_s{seed}.csv",
    ),
    # α（R121 步骤 2 → 步 4）：**4 码纪元基线 + 同批 16/4 对照**（纪元纪律）
    # ------------------------------------------------------------------
    # 🔴 为什么 16 与 4 必须**同批**：内评 09-17 §三.3 —— 字母表是**纪元变更**，
    #    跨批的 ratio/codebook_conv **不可直接比较** ⇒ 因果结论只能在同批内用可逆开关取。
    # 🔴 为什么是 **8k 快批**而非 60k：内评 09-18 §三 建议 —— 先过"方向门槛"再上 60k 确认批。
    #    本项目的最大时间黑洞一直是"用旗舰批做筛选"（R97→R100→R107→R120）。
    #    成本：main 臂 ≈570 tick/min/run ⇒ 8k 约 **15 min**（12 run 并发）。
    # ⚠️ `donation` 不得进 `fixed`（非 oracle 臂收到 oracle 专属参数 ⇒ a4 **硬失败**，
    #    2026-09-16 C1a 首跑实测：18 run 只起了 12 个）——本预设 arm=main，故不写 donation。
    "cstep3alpha": dict(
        script="experiments/a4_verify_capacity.py",
        grid=["seed=42,43,44,45,46,47"],
        fixed=["mode=on", "arm=main", "ticks=8000",
               "max-count=3240", "snapshot-every=2000",
               "snapshot-dir=_rerun_logs/cstep_snap3"],
        variants=[
            dict(name="a16", args=["signal-alphabet=16"],
                 template="_rerun_logs/cstep3alpha/a16_s{seed}.csv"),
            dict(name="a4", args=["signal-alphabet=4"],
                 template="_rerun_logs/cstep3alpha/a4_s{seed}.csv"),
        ],
    ),
    # R123/B② 门控臂对照批（**待令启动**）：现状付款 vs 门控付款（Δ_content）同批、同 seed。
    #   目的：直接测度"蹭归因占比"（= gate_block / arrivals），并给"只计真通信时是否仍有阳性"
    #   留数据。⚠️ 门控臂是**仪器** ⇒ `--calibration-arm` 必带（R100 条件 5）。
    "cstep3gate": dict(
        script="experiments/a4_verify_capacity.py",
        grid=["seed=42,43,44,45,46,47"],
        fixed=["mode=on", "ticks=12000", "max-count=3240",
               "snapshot-every=2000", "snapshot-dir=_rerun_logs/cstep_snap3"],
        variants=[
            dict(name="ungated",
                 args=["arm=oracle", "donation=1.0", "gain-multiplier=1.3",
                       "calibration-arm", "gate-mode=none"],
                 template="_rerun_logs/cstep3gate/ungated_s{seed}.csv"),
            dict(name="gated",
                 args=["arm=oracle", "donation=1.0", "gain-multiplier=1.3",
                       "calibration-arm", "gate-mode=delta_positive",
                       "gate-delta=content"],
                 template="_rerun_logs/cstep3gate/gated_s{seed}.csv"),
        ],
    ),
    # β（`"8"` 档 B③ 记忆位）：**信号里第一次携带接收者读不到的信息** ⇒ 是否出现方向性选择压？
    # ------------------------------------------------------------------
    # 🔴 科学问题（R125 / 内评）：`"8"` 档 `state = e_bin*2 + mem_bit`，其中 `mem_bit`
    #    标记「发送者**当前格**在它自己的 `_work_memory` 里」—— 而记忆可含**半径 4 之外**的格位
    #    ⇒ 这是**唯一**接收者无法直读、也不与直读冗余的信号内容（`n_bit`/`f_bit` 均为冗余）。
    # 🔴 为什么与 `"4"` 同批：字母表是**纪元变更**（R113/R121：跨批 `ratio`/`codebook_conv` 不可比）
    #    ⇒ 因果结论只能在**同批内**用可逆开关取；`"4"` = α 批后的**新纪元基线**。
    # 🔴 为什么 **12k**（不是 α 的 8k）：α 批实测 **8k 区制不可分**（饱和度重叠），
    #    ρ 判读需终态 N 可分层 ⇒ 12k（`N`: SAT≥2715 vs PRED≤1522，09-18 实测分离）。
    # 🔴 为什么 **main 臂**（非 oracle）：本批问的是**科学臂**里记忆位有没有用；
    #    oracle 通道与 `"8"` 档的因果问题正交，且 oracle 臂慢 2.4×（成本）。
    # ⚠️ 臂名用 `b4`/`b8` + 独立快照目录 `cstep_snap8`：**防止与 α/gate 批的快照名冲突**
    #    （`cstep_snap3` 已有 48 个快照：a16/a4/gated/ungated × 12；快照名 = `<臂名>_s<seed>`）。
    # ⚠️ `donation` 不得进 `fixed`（非 oracle 臂收到 oracle 专属参数 ⇒ a4 硬失败；C1a 首跑实测）。
    # 成本：α 批实测 12 run × 8k = 26.4 min（main 臂）⇒ 12k 外推 ≈ **40 min**（35–55 区间）。
    "cstep3alpha8": dict(
        script="experiments/a4_verify_capacity.py",
        grid=["seed=42,43,44,45,46,47"],
        fixed=["mode=on", "arm=main", "ticks=12000",
               "max-count=3240", "snapshot-every=2000",
               "snapshot-dir=_rerun_logs/cstep_snap8"],
        variants=[
            dict(name="b4", args=["signal-alphabet=4"],
                 template="_rerun_logs/cstep3alpha8/b4_s{seed}.csv"),
            dict(name="b8", args=["signal-alphabet=8"],
                 template="_rerun_logs/cstep3alpha8/a8_s{seed}.csv"),
        ],
    ),
    # γ（R127/C8）：**在"信息有价值"的世界里重测信号内容效应** —— patchy 侧（2×2 的另一格）
    # ------------------------------------------------------------------
    # 🔴 为什么必须跑这一批：`[实测]` 冒烟（2026-09-19 02:3x）证明
    #    **uniform 世界下「纬度」100% 解释了容量（R²=1.000）** ⇒ 位置完全可预测 ⇒ **信息价值 = 0**
    #    ⇒ C1a/C1b/C2/α/gate/α8 全部跑在"测信息价值 = 0"的世界里（R127 C8 首例事故）
    #    而 **patchy 下纬度只解释 31.7%**（不可解释 68.3%）⇒ 经度方向有真信息 ⇒ 前提成立。
    # 🔴 为什么是 2×2 的另一格：本批用 `distribution=patchy` × {`"4"`,`"8"`}，
    #    与 **α8（uniform × {`"4"`,`"8"`}，12k，已完成）** 同 tick / 同 seed / 同臂
    #    ⇒ 直接构成 `distribution × alphabet` 的 **2×2**，**α8 的数据不必重跑**（省一半机时）。
    # ⚠️ 臂名 `p4`/`p8` + 独立快照目录 `cstep_snap9`：防与 a16/a4/gated/ungated/b4/a8 快照名冲突。
    # ⚠️ 已过前提冒烟（R127 §12.2.1）：C4 读回 ✓ / C8 前提 ✓ / 200 tick 生态存活 ✓（N=82）。
    # 成本：α8 实测 12 run × 12k = 35.6 min；patchy 生态可能更脆（早崩更快）⇒ 估 **30–45 min**。
    "cstep3patchy": dict(
        script="experiments/a4_verify_capacity.py",
        grid=["seed=42,43,44,45,46,47"],
        fixed=["mode=on", "arm=main", "ticks=12000",
               "max-count=3240", "snapshot-every=2000",
               "snapshot-dir=_rerun_logs/cstep_snap9",
               "distribution=patchy"],
        variants=[
            dict(name="p4", args=["signal-alphabet=4"],
                 template="_rerun_logs/cstep3patchy/p4_s{seed}.csv"),
            dict(name="p8", args=["signal-alphabet=8"],
                 template="_rerun_logs/cstep3patchy/p8_s{seed}.csv"),
        ],
    ),
    # R128（2026-09-19）：`mem_bit` 语义修正版验证批 —— **"非冗余通道首次上桌"**
    #   臂：仅需 `p8f`（修正版 "8" 档）× 6 seed × 12k；**对照复用 E-023 的 `p4`**
    #   （前提已由 C7 逐位对拍证明"修正只影响 8 档、4 档逐位不变"）
    #   口径预注册见板上 R128 §三（诚实预告：修正只消冗余、不创造位置信息 ⇒ 主指标仍可能 null）
    "cstep3membit8": dict(
        script="experiments/a4_verify_capacity.py",
        grid=["seed=42,43,44,45,46,47"],
        fixed=["mode=on", "arm=main", "ticks=12000",
               "max-count=3240", "snapshot-every=2000",
               "snapshot-dir=_rerun_logs/cstep_snap10",
               "distribution=patchy"],
        variants=[
            dict(name="p8f", args=["signal-alphabet=8"],
                 template="_rerun_logs/cstep3membit8/p8f_s{seed}.csv"),
        ],
    ),
    # A′（2026-09-19，`设计-A档记忆朝向梯度-20260919.md` §七）：记忆**朝向梯度**正式批
    #   问题：个体**自己**记住的富食格位置会不会改变它往哪走？（零新通道、改决策语义）
    #   臂：`mg`（orientation，朝向梯度 on）vs `mn`（none，**原式**）⇒ **同批同 seed**
    #   🔴 对照**不可复用** E-023/E-024 的 `p4`（那是旧语义的批次）
    #   档位固定 `"4"`（最小，避免字母表变量混入）；世界 patchy（C8 lat_r2=0.317 ✓）
    #   预计 12 run × 12k ≈ **32–36 min**（参照 E-023 实测 36.0 min）
    "cstep3memgrad": dict(
        script="experiments/a4_verify_capacity.py",
        grid=["seed=42,43,44,45,46,47"],
        fixed=["mode=on", "arm=main", "ticks=12000",
               "max-count=3240", "snapshot-every=2000",
               "snapshot-dir=_rerun_logs/cstep_snap11",
               "distribution=patchy", "signal-alphabet=4"],
        variants=[
            dict(name="mg", args=["memory-gradient=orientation"],
                 template="_rerun_logs/cstep3memgrad/mg_s{seed}.csv"),
            dict(name="mn", args=["memory-gradient=none"],
                 template="_rerun_logs/cstep3memgrad/mn_s{seed}.csv"),
        ],
    ),
    # R132（2026-09-19 所有者批准）：**A′ 复测批（确证性）** —— 把 E-025 的探索性成簇信号
    #   （pred_frac 0/6 正 + 区制 4/6 单向翻转）转成可判读结论。
    #   🔴 预注册（所有者 18:45 复核 ①–④，跑前锁定）：
    #     主读数 = `pred_frac` 同向性（n=20 双侧符号检验，**判显著 ≥15/20**（p=0.0414）；
    #       功效前提 = 真实同向率 ≥0.80（0.80⇒0.804）；"20/20 p≈1.9e-6" 是事实**不是判据**）
    #     落点三档：**≥15/20 = 复现**｜**13–14/20 = 未复现**｜**≤12/20 = 方向反**
    #     副读数 = 区制方向比（二项 vs 0.5，**≥80% 单向**才算；**n_flips < 6 ⇒ undecidable**）
    #     其余指标（N/ρ/codebook_conv/resp_* 等）**只报不判**（确证性批的多重比较纪律）
    #   🔴 seed 用 **48–67（全 fresh）**：R132 原建议 42–61，但 42–47 已在 E-025 跑过
    #      （引擎确定性 ⇒ 重跑 = 逐位重复），且把它们计入"≥15/20"会**重复使用**
    #      产生假设的那 6 个数据点（正是所有者 ② 说的"被选中的极端值"问题）⇒ 偏离已上板说明
    #   配置：δ（E-026）无健康窗 ⇒ 按 R132 §二 跑**原配置 max_count=3240**（所有者 18:45 §四.3 ✓）
    "cstep3memgrad20": dict(
        script="experiments/a4_verify_capacity.py",
        grid=["seed=48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67"],
        fixed=["mode=on", "arm=main", "ticks=12000",
               "max-count=3240", "snapshot-every=2000",
               "snapshot-dir=_rerun_logs/cstep_snap12",
               "distribution=patchy", "signal-alphabet=4"],
        variants=[
            dict(name="mg", args=["memory-gradient=orientation"],
                 template="_rerun_logs/cstep3memgrad20/mg_s{seed}.csv"),
            dict(name="mn", args=["memory-gradient=none"],
                 template="_rerun_logs/cstep3memgrad20/mn_s{seed}.csv"),
        ],
    ),
    # R132 ⑤（所有者 18:45 建议，可选 +1h）：**uniform 判别臂** —— 排掉最大解释风险
    #   （"pred_frac 差异来自移动统计被改，而非记忆有用"）。
    #   机制预测：uniform 世界容量由纬度决定（lat_r2≈1.0）⇒ 位置可预测 ⇒ 记忆**不增值**
    #   ⇒ 预注册读法：uniform 下 `pred_frac` Δ 应**无成簇同向**（若 patchy 的 6/6 式下降
    #     在 uniform 复现 ⇒ "记忆有用"解释死，效应归"移动统计"）。
    #   ⚠️ n=10 对 ⇒ 判别力有限，只作方向性判别（不作显著性确证）
    "cstep3memgrad20u": dict(
        script="experiments/a4_verify_capacity.py",
        grid=["seed=48,49,50,51,52,53,54,55,56,57"],
        fixed=["mode=on", "arm=main", "ticks=12000",
               "max-count=3240", "snapshot-every=2000",
               "snapshot-dir=_rerun_logs/cstep_snap13",
               "distribution=uniform", "signal-alphabet=4"],
        variants=[
            dict(name="mg", args=["memory-gradient=orientation"],
                 template="_rerun_logs/cstep3memgrad20u/mg_s{seed}.csv"),
            dict(name="mn", args=["memory-gradient=none"],
                 template="_rerun_logs/cstep3memgrad20u/mn_s{seed}.csv"),
        ],
    ),
    # δ（R129 批准）：**区制图** —— 测绘「SAT:PRED 比例」随 max_count 的曲线（找可观测选择窗）
    # ------------------------------------------------------------------
    # 🔴 一维杠杆选 max_count：SAT 判据（N >= 0.9*max_count）由它**定义** ⇒ 唯一确定的杠杆
    # 🔴 5 点（1200/1800/2400/4200/5400）+ **复用 E-023 p4 作 3240 点**（30 run 而非 36，R129 §二）
    # 🔴 variants 的 max-count 在 fixed 之后 ⇒ argparse 后者覆盖（dry-run 已逐 run 核）
    # ⚠️ 预注册（设计稿 §四，跑前锁定）：SAT = N>=0.9*max_count；主读数 = 每点 SAT 比例 k/6；
    #    判定 = **极值两点**（1200 vs 5400）Fisher 精确 p<0.05 ⇒ 杠杆；全点 ∈[1/6,5/6] ⇒ 非杠杆；
    #    ⚠️ Fisher 6v6 最小可分辨 = **5:1 vs 1:5**（4:2 vs 2:4 检不出）⇒ "非杠杆"≠设计失败，是功效天花板
    # ⚠️ 快照独立目录 cstep_snap_delta（防与历史批同名静默续跑）
    "cstep3delta": dict(
        script="experiments/a4_verify_capacity.py",
        grid=["seed=42,43,44,45,46,47"],
        fixed=["mode=on", "arm=main", "ticks=12000",
               "max-count=3240", "snapshot-every=2000",
               "snapshot-dir=_rerun_logs/cstep_snap_delta",
               "distribution=patchy"],
        variants=[
            dict(name="d1200", args=["max-count=1200"],
                 template="_rerun_logs/cstep3delta/d1200_s{seed}.csv"),
            dict(name="d1800", args=["max-count=1800"],
                 template="_rerun_logs/cstep3delta/d1800_s{seed}.csv"),
            dict(name="d2400", args=["max-count=2400"],
                 template="_rerun_logs/cstep3delta/d2400_s{seed}.csv"),
            dict(name="d4200", args=["max-count=4200"],
                 template="_rerun_logs/cstep3delta/d4200_s{seed}.csv"),
            dict(name="d5400", args=["max-count=5400"],
                 template="_rerun_logs/cstep3delta/d5400_s{seed}.csv"),
        ],
    ),
    # A′ 复测批（R132 fish 批准）：**20 seed 双臂**验证 E-025 的成簇信号（pred 6/6 同向 + 区制单向翻转）
    # ------------------------------------------------------------------
    # 🔴 δ（E-026）结论：max_count **非杠杆**（6 点全部 4:2 或同构；崩塌 seed 与承载力无关，
    #    s42/s44 在所有点崩塌到完全相同的 N=489/194）⇒ **无健康窗** ⇒ 按 R132 预注册规则：
    #    复测批走**原配置 max_count=3240**，主读数 = pred_frac 同向性与区制翻转方向（rho 如实报、不可判）
    # 🔴 20 seed 连续（42–61）：n=20 双侧符号检验全同向 p≈1.9e-6，可判读；功效口径收尾时算
    # ⚠️ 独立快照目录 cstep_snap_mg20；判读预注册：pred_frac 同向性（双侧符号检验）+ 区制翻转方向
    #    （单向比 = 翻转中朝 SAT 的比例，二项检验 vs 0.5）
    "cstep3memgrad20": dict(
        script="experiments/a4_verify_capacity.py",
        grid=["seed=42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61"],
        fixed=["mode=on", "arm=main", "ticks=12000",
               "max-count=3240", "snapshot-every=2000",
               "snapshot-dir=_rerun_logs/cstep_snap_mg20",
               "distribution=patchy"],
        variants=[
            dict(name="mg", args=["memory-gradient=orientation"],
                 template="_rerun_logs/cstep3memgrad20/mg_s{seed}.csv"),
            dict(name="mn", args=["memory-gradient=none"],
                 template="_rerun_logs/cstep3memgrad20/mn_s{seed}.csv"),
        ],
    ),
}


# ---------------------------------------------------------------- 单实例锁
# 🔴 2026-09-16 事故与**根因更正**（内评 22:29 补录，实测 CPU/EXE 证据）：
#   · **首跑真实事故 = 参数归属错**：`donation` 放在 preset 的 `fixed` ⇒ 套给 `zero` 臂
#     ⇒ a4 硬拒 ⇒ 18 run 只起 12 个（已由 `a6b96a9` 修正；现场无数据损坏）。
#   · ⚠️ **更正我原先的表述**：「26 个进程 = 两个 batch_runner 实例」**是错的** ——
#     `.venv\Scripts\python.exe` 是**重定向器 stub**，以同一 argv 派生真实解释器并等待
#     ⇒ **每次调用天然产生两个同名进程（父=子）**。实测：36 个 a4 进程中 18 个 CPU≈0、
#     18 个 CPU≈382–439 s；总 CPU 7199.8 s ÷ 墙钟 447 s = **16.1 核** ⇒ **18 份仿真、非 36 份**。
#   ⇒ **纪律（已入 `AGENT.md` §10.5）：进程计数不可作为"重复执行"的证据；判真伪须看 CPU 时间 / EXE 路径。**
#   · 环境**确有**偶发重复执行（实测：板追加被双写、补丁脚本被重放）——但那是 harness 层现象，
#     与进程计数无关，也不是本次 C1a 的事故原因。
#   ⇒ 本锁的威胁模型 = **防"同一批被两个启动器同时推进"**（对 C1b/C2 的启动窗口有效）。
LOCK_NOTE = ("威胁模型：防同时启动的两个批跑器（偶发重复执行 ⇒ 双写同一批产出）；"
             "**不防**陈旧锁（陈旧用 PID 判活自动接管）。")
LOCK_PATH = ROOT / "_rerun_logs" / ".batch_runner.lock"


def _pid_alive(pid: int) -> bool:
    """PID 是否存活（零依赖；Windows 用 OpenProcess，POSIX 用信号 0）。"""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, int(pid))       # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        k32.CloseHandle(h)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_lock(preset: str, label: str = "", force: bool = False) -> str | None:
    """抢占单实例锁。成功返回 None；被占用则返回**占用描述**（调用方打印并退出）。"""
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"pid": os.getpid(), "preset": preset, "label": label,
         "started": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False)

    def _read_holder() -> tuple[dict, int]:
        try:
            info = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
            return info, int(info.get("pid", -1))
        except Exception:
            return {}, -1

    if force:                                # 显式接管：直接覆盖
        LOCK_PATH.write_text(payload, encoding="utf-8")
        return None

    # 🔴 R110 §三：**原子抢占**（`O_CREAT|O_EXCL`）—— 原实现是"先查后写"，
    # 两个**同一秒启动**的实例可同时通过存在性检查（而"命令被执行两次"的时序特征
    # 恰恰是同时启动）⇒ 锁与威胁模型不匹配（内评 22:29 §三 实测指出）。
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        info, pid = _read_holder()
        if _pid_alive(pid):
            return (f"已有实例在跑（pid={pid}，preset={info.get('preset')!r}，"
                    f"起于 {info.get('started')}）—— **同一时刻只允许一个批跑器**；"
                    f"如确认已死可用 --force-lock 接管")
        # 陈旧锁（持有者已死）⇒ 接管：原子替换
        try:
            LOCK_PATH.unlink()
        except FileNotFoundError:
            pass
        try:
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            info, pid = _read_holder()       # 竞争中输了 ⇒ 如实报告
            return (f"已有实例在跑（pid={pid}，preset={info.get('preset')!r}）"
                    f"—— 竞争接管失败，请重试或确认后 --force-lock")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(payload)
    return None


def release_lock() -> None:
    try:
        LOCK_PATH.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------- 调度

def poll(running: list[Run]) -> None:
    for r in list(running):
        rc = r.proc.poll()
        if rc is None:
            continue
        running.remove(r)
        r.wall = time.time() - r._t0
        # ---- F-R10：判据放宽（2026-09-15）----
        # 旧判据 `rc == 0 and summary_ok` 会把"rc=1 假失败"误判为 failed 并触发重试。
        # 假失败成因：a4 收尾删 progress.json 被环境 safe-delete 守卫 fail-closed 拒绝
        # ⇒ 子进程退出码 1，而 **summary 早已先写、数据无损**。
        # 新判据：summary 合法即可判 done。为防"读到上一轮的旧 summary"，
        # 额外要求 summary 的 mtime **不早于本次启动时刻**（`_t0`）。
        _fresh = True
        try:
            _fresh = r.summary.stat().st_mtime >= float(getattr(r, "_t0", 0.0)) - 1.0
        except OSError:
            _fresh = False
        if summary_ok(r.summary) and _fresh:
            r.status = "done"
            if rc == 0:
                print(f"  ✅ {r.name}  done  {r.wall/60:.1f} min")
            else:
                r.note = f"⚠️ rc={rc}（summary 完整且为本轮产出 ⇒ 判 done；F-R10 假失败）"
                print(f"  ✅ {r.name}  done（⚠️ rc={rc}，summary 完整）  {r.wall/60:.1f} min")
        else:
            r.status = "failed"
            _why = "summary缺失/不合法"
            if summary_ok(r.summary) and not _fresh:
                _why = "summary 为旧文件（mtime 早于本次启动）"
            r.note = f"rc={rc} {_why}"
            print(f"  ❌ {r.name}  FAILED ({r.note})  {r.wall/60:.1f} min")


def run_batch(runs: list[Run], conc: int, retries: int, python: str, workdir: Path,
              per_run_gb: float, reserve_gb: float, dry: bool) -> int:
    if dry:
        print(f"[dry-run] 并发上限 {conc}（按内存 {per_run_gb:.2f} GB/run，留 {reserve_gb:.1f} GB）")
        for r in runs:
            print(f"  {'⏭️ skip' if r.status=='skipped' else '▶ run '} {r.name}: "
                  f"{python} {' '.join(r.cmd)}")
        return 0

    pending = [r for r in runs if r.status == "pending"]
    running: list[Run] = []
    t_start = time.time()
    guard_hits = 0

    while pending or running:
        # 内存守卫：不足则不再拉起新 run（只排队，绝不杀已有进程）
        while pending and len(running) < conc:
            avail, total, load = memory_gb()
            need = reserve_gb + per_run_gb
            if avail < need:
                guard_hits += 1
                print(f"  🛡️ 内存守卫：可用 {avail:.2f} GB < 需要 {need:.2f} GB"
                      f"（负载 {load}%）⇒ 暂不拉起新 run（运行中 {len(running)}）")
                break
            r = pending.pop(0)
            r.attempt += 1
            with open(r.log, "w", encoding="utf-8") as lf:
                r.proc = subprocess.Popen([python] + r.cmd, cwd=str(workdir),
                                          stdout=lf, stderr=subprocess.STDOUT)
            r._t0 = time.time()
            r.status = "running"
            running.append(r)
            print(f"  ▶ {r.name} (attempt {r.attempt}, 运行中 {len(running)}/{conc}, "
                  f"可用内存 {avail:.2f}/{total:.2f} GB)")

        poll(running)

        # 重试回收
        for r in runs:
            if r.status == "failed" and r.attempt <= retries:
                r.status = "pending"
                pending.append(r)
                print(f"  🔁 {r.name} 重试 {r.attempt}/{retries}")

        if pending or running:
            time.sleep(2.0)

    total_wall = time.time() - t_start
    done = [r for r in runs if r.status == "done"]
    skipped = [r for r in runs if r.status == "skipped"]
    failed = [r for r in runs if r.status == "failed"]
    print(f"\n批次结束：done={len(done)} skipped={len(skipped)} failed={len(failed)} "
          f"墙钟={total_wall/60:.1f} min（守卫触发 {guard_hits} 次）")
    return 1 if failed else 0


# ---------------------------------------------------------------- 汇总

def summarize(runs: list[Run], outdir: Path) -> None:
    from observatory.statistics import predation_fraction  # D-16 单一口径

    rows = []
    for r in runs:
        if not r.summary.exists():
            continue
        try:
            with open(r.summary, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        res, sw = d.get("result", {}), d.get("switches", {})
        causes = res.get("deaths_by_cause") or {}
        pred = res.get("final_pred_frac")
        if pred is None:                      # 回算（D-16 回算口径）
            pred = round(predation_fraction(causes), 4)
        rows.append({
            "arm": r.name,
            "codebook": sw.get("arbitrary_codebook"),
            "arm_type": sw.get("arm"),
            "seed": sw.get("seed"),
            "final_N": res.get("final_N"),
            "eco_gate": res.get("eco_gate_pass"),
            "pred_frac": pred,
            "regime": ("predation_dominant" if pred is not None and pred >= 0.9
                       else "non_predation"),
            "codebook_conv": res.get("final_codebook_conv"),
            # D-17⑤ / D-18⑥ / D-8 oracle（R53 六门判读的原始量）
            "slope_g15": (res.get("selection_gradient") or {}).get("non_sat", {}).get("slope_g15"),
            "resp_a": (res.get("signal_response") or {}).get("resp_a_exposure"),
            "resp_b": (res.get("signal_response") or {}).get("resp_b_delta"),
            "resp_triple": (res.get("signal_response") or {}).get("resp_triple"),
            "oracle_ratio": (res.get("oracle") or {}).get("oracle_return_ratio"),
            "wall_min": round(r.wall / 60, 1),
        })
    if not rows:
        print("（无 summary 可汇总）")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    csvp = outdir / "_summary.csv"
    with open(csvp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== 汇总（{len(rows)} run）===")
    print(f"{'arm':<22}{'cb':>4}{'seed':>6}{'final_N':>9}{'gate':>7}{'pred_frac':>11}{'区制':>22}{'wall_min':>10}")
    print("-" * 92)
    for x in sorted(rows, key=lambda z: (str(z["codebook"]), z["seed"] or 0)):
        print(f"{x['arm']:<22}{str(x['codebook']):>4}{str(x['seed']):>6}{str(x['final_N']):>9}"
              f"{str(x['eco_gate']):>7}{x['pred_frac']:>11.4f}{x['regime']:>22}"
              f"{str(x['wall_min']):>10}")
    print("-" * 92)

    hi = [x for x in rows if x["pred_frac"] is not None and x["pred_frac"] >= 0.9]
    lo = [x for x in rows if x["pred_frac"] is not None and x["pred_frac"] < 0.9]
    print(f"分层（阈值 0.9）：捕食主导 {len(hi)} / 非捕食主导 {len(lo)}")
    if hi and lo:
        med = lambda g: sorted(x["final_N"] for x in g if isinstance(x["final_N"], int))[len(g) // 2]
        print(f"  ⚠️ R38③：两层不得合并均值。final_N 中位："
              f"捕食主导 {med(hi)} vs 非捕食主导 {med(lo)}")

    (outdir / "_summary.csv").write_text(csvp.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"\n汇总已写：{csvp}")


# ---------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser(description="D-25 并行跑批包装（R48）")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--script", default="experiments/a4_verify_capacity.py")
    ap.add_argument("--grid", nargs="*", default=[], help="如 codebook=0,1 seed=42,43")
    ap.add_argument("--fixed", nargs="*", default=[], help="如 mode=on ticks=60000 fresh")
    ap.add_argument("--out-template", default="_rerun_logs/a4_fix/run_{seed}.csv")
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--python", default=None, help="默认用主工作区 .venv")
    ap.add_argument("--concurrency", type=int, default=None, help="默认按内存自动")
    ap.add_argument("--per-run-gb", type=float, default=0.15,
                    help="单 run 内存估计（实测 68-78 MB，取 2× 余量）")
    ap.add_argument("--reserve-gb", type=float, default=1.0,
                    help="R48：必须留出的内存余量（GB）")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true",
                    help="已有合法 summary 的 run 直接跳过（断点续批）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-lock", action="store_true",
                    help="忽略已有锁（仅当确认持有者进程已死时使用）")
    ap.add_argument("--no-summarize", action="store_true")
    args = ap.parse_args()

    if args.preset:
        p = PRESETS[args.preset]
        args.script = p["script"]
        args.grid = p["grid"]
        args.fixed = p["fixed"]
        args.out_template = p.get("template")
        args.variants = p.get("variants")
    if not args.grid:
        print("错误：需要 --grid 或 --preset")
        return 2

    workdir = Path(args.workdir)
    if not workdir.is_absolute():
        workdir = ROOT / workdir
    py = args.python or str(ROOT / ".venv" / "Scripts" / "python.exe")
    if not os.path.exists(py):
        py = sys.executable

    runs = expand(args.grid, args.fixed, args.out_template, args.script, workdir,
                  getattr(args, "variants", None))
    for r in runs:                      # 产出目录可能与 workdir 不同（如 workdir=_wt_r19）
        r.out.parent.mkdir(parents=True, exist_ok=True)
    if args.skip_existing:
        for r in runs:
            if summary_ok(r.summary):
                r.status = "skipped"
                r.note = "已有合法 summary"

    avail, total, load = memory_gb()
    conc = args.concurrency or pick_concurrency(args.per_run_gb, args.reserve_gb, None)
    print(f"任务数 {len(runs)}（跳过 {sum(1 for r in runs if r.status=='skipped')}）"
          f" | 并发上限 {conc}（CPU {cpu_count()} 核）"
          f" | 内存 {avail:.2f}/{total:.2f} GB 可用（负载 {load}%）"
          f" | 守卫阈值 留 {args.reserve_gb:.1f} GB + {args.per_run_gb:.2f} GB/run")

    if not args.dry_run:                      # 真跑才上锁（dry-run 无副作用）
        hold = acquire_lock(args.preset or "<ad-hoc>", label=args.out_template or "",
                            force=args.force_lock)
        if hold:
            print(f"❌ 拒绝启动：{hold}")
            return 3
    try:
        rc = run_batch(runs, conc, args.retries, py, workdir,
                       args.per_run_gb, args.reserve_gb, args.dry_run)
    finally:
        if not args.dry_run:
            release_lock()

    if not args.no_summarize and not args.dry_run:
        outdir = runs[0].out.parent
        summarize(runs, outdir)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
