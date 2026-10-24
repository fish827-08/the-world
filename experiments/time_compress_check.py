"""时间压缩不变性检查（14.10 / 评审 P0.0 验收）—— 压时间后结果该不该变。

判据（预注册）
--------------
同一物理配置，只改「每昼夜 tick 数 D」并做三类重标（`experiments/scaling_rescale.py`）：
**逐「世界日」比较种群、食物利用率、死亡构成**。

* **通过**：两条曲线在「日」坐标下重合（相对差 < 20%）
* **不通过**：随 k 系统性漂移 ⇒ k 太大或漏标了参数（漏标 = 静默改科学）

锚点一致的速度
--------------
`v_max = cols / (8·D)` ⇒ **每「日」行程 = cols/8 格，与 D 无关** ⇒ 天然可比。

⚠️ 现有 `speed_steps = floor(speed×subdiv+0.5)` 是**每 tick 独立四舍五入** ⇒ 慢档会被量化成
   「永远不动」。速度定档按 **R204 §二 / R205**：`gain = v_max`、`subdiv = SUBDIV_STD`（**80**，R213 §四）**固定**
   （`apply_speed_std`）⇒ `R = v_max × 40` **随 v_max 变**：`R < 5` 时本工具向 stderr 报警
   （量化悬崖，设计稿 §12.1：保 ≥5 档是一条硬下界）。

用法
----
    .venv\\Scripts\\python.exe experiments/time_compress_check.py
    .venv\\Scripts\\python.exe experiments/time_compress_check.py --rows 60 --cols 120 --days 6
    .venv\\Scripts\\python.exe experiments/time_compress_check.py --ds 2400,960   # 定档默认（k=1 vs 2.5）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation.config import SimConfig                      # noqa: E402
from simulation.sphere_engine import SphereEngine            # noqa: E402
from experiments.scaling_rescale import (                    # noqa: E402
    ancher_consistent_speed, apply_post_build, apply_speed_std, ladder_warn,
    rescale_config,
)

# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print 会让脚本 rc=1 假失败；F-R15 族）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # 非 TTY / 旧解释器：不因诊断能力缺失而阻断运行

# 🔴 速度定档（R204 §二 / R205）：`gain = v_max`、`subdiv = SUBDIV_STD`（80，R213 §四）固定 —— 见 `apply_speed_std`。
#    旧常量 `A_AGE`（用 ā 反解 gain）与 `R_TARGET`（用 R 反解 subdiv）已按裁定删除。


def run_day(rows: int, cols: int, patches: int, pop: int, D: int, days: float,
            subpos: bool, seed: int = 42,
            travel_per_day: float = 0.0) -> tuple[np.ndarray, dict]:
    """跑到 `days` 个世界日，按「日」采样。返回 (按日采样表, 摘要)。"""
    c = SimConfig(seed=seed)
    c.world.rows, c.world.cols = rows, cols
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = True
    c.resources.patch_regrowth_mult = 1.195
    c.resources.patch_count = patches
    c.population.initial_count = pop

    k = 2400.0 / D
    notes = rescale_config(c, k)

    # travel_per_day > 0 ⇒ 直接给定「每日行程」（格），使两条臂可比且不吃 1 格/tick 上限
    v_max = (travel_per_day / D) if travel_per_day > 0 else ancher_consistent_speed(cols, D)
    ladder_r = None
    if subpos:
        c.simulation.use_sim_core = False
        c.subpos.enabled = True
        ladder_r = apply_speed_std(c.subpos, v_max)      # gain=v_max、subdiv=80（R213 §四）
        c.subpos.min_energy_frac = 0.0
        _w = ladder_warn(ladder_r)
        if _w:
            print(_w, file=sys.stderr)
    else:
        # 无 subpos ⇒ 位移恒 1 格/tick ⇒ 用 stay_prob 把"每 tick 必走"调成等效日行程
        c.simulation.stay_prob = float(np.clip(1.0 - v_max, 0.0, 0.95))

    eng = SphereEngine(c)
    apply_post_build(eng, notes)

    prod = eng.resources._capacity > 0
    cap_sum = float(eng.resources._capacity[prod].sum())
    ticks_per_day = int(D)
    total = int(round(days * ticks_per_day))
    rec = []
    for t in range(1, total + 1):
        eng.step()
        if t % ticks_per_day == 0:
            N = int(len(eng._flat))
            stock = float(eng.resources._grid[prod].sum())
            rec.append((t / ticks_per_day, N, stock / max(cap_sum, 1e-9),
                        float(eng._energy[:N].mean()) if N else 0.0))
    days_arr = np.array([r[0] for r in rec])
    pops = np.array([r[1] for r in rec], dtype=float)
    utils = np.array([r[2] for r in rec])
    info = {"k": k, "D": D, "v_max": v_max, "subdiv": c.subpos.subdiv if subpos else None,
            "ladder_R": ladder_r,
            "prod_cells": int(prod.sum()), "pop_final": int(pops[-1]) if len(pops) else 0,
            "util_final": float(utils[-1]) if len(utils) else float("nan"),
            "days": days_arr, "pops": pops, "utils": utils}
    return days_arr, info


def main() -> None:
    ap = argparse.ArgumentParser(description="时间压缩不变性检查")
    ap.add_argument("--rows", type=int, default=60)
    ap.add_argument("--cols", type=int, default=120)
    ap.add_argument("--patches", type=int, default=30)
    ap.add_argument("--pop", type=int, default=800)
    ap.add_argument("--days", type=float, default=6.0)
    ap.add_argument("--subpos", choices=("off", "on"), default="off")
    ap.add_argument("--travel-per-day", type=float, default=0.0,
                    help=">0 ⇒ 直接用「每日行程（格）」定速度（不吃 1 格/tick 上限）")
    ap.add_argument("--ds", default="2400,960",
                    help="对照臂的「每昼夜 tick 数」列表：**默认 2400,960 = k=1 vs 定档 k=2.5**（R205）；"
                         "旧的 k=5 臂仍可用 `--ds 2400,480`")
    a = ap.parse_args()
    ds_list = [int(x) for x in a.ds.split(",") if x.strip()]

    print(f"== 时间压缩不变性检查：{a.rows}x{a.cols}，斑块 {a.patches}，"
          f"初始 {a.pop}，{a.days} 世界日，subpos={a.subpos} ==")
    if a.travel_per_day > 0:
        print(f"   直接给定每日行程 = {a.travel_per_day} 格（与 D 无关）⇒ v_max = 行程/D\n")
    else:
        print(f"   锚点：v_max = cols/(8D) ⇒ 每日行程 = cols/8 = {a.cols / 8:.1f} 格（与 D 无关）\n")

    results = []
    print(f"   对照臂 D = {ds_list}（k = {[round(2400.0 / d, 3) for d in ds_list]}）"
          f" ｜ 定档（R205）：k=2.5 ⇔ D=960\n")
    for D in ds_list:
        d, info = run_day(a.rows, a.cols, a.patches, a.pop, D, a.days,
                          a.subpos == "on", travel_per_day=a.travel_per_day)
        results.append((d, info))
        print(f"D={D:<5} k={info['k']:<4.1f} v_max={info['v_max']:.4f}"
              f" subdiv={info['subdiv']} R={info['ladder_R']}｜末值 N={info['pop_final']}"
              f" 利用率={info['util_final']:.3f}")

    d0, i0 = results[0]
    d1, i1 = results[1]
    n = min(len(i0["pops"]), len(i1["pops"]))
    print(f"\n{'日':>5}{'N(D=2400)':>12}{'N(D=480)':>11}{'相对差':>9}"
          f"{'利用率2400':>12}{'利用率480':>11}")
    for j in range(n):
        p0, p1 = i0["pops"][j], i1["pops"][j]
        rel = abs(p1 - p0) / max(p0, 1.0)
        print(f"{i0['days'][j]:>5.0f}{p0:>12.0f}{p1:>11.0f}{rel:>9.2f}"
              f"{i0['utils'][j]:>12.3f}{i1['utils'][j]:>11.3f}")

    rel_all = [abs(i1["pops"][j] - i0["pops"][j]) / max(i0["pops"][j], 1.0)
               for j in range(n)]
    rel_tail = rel_all[max(0, n // 3):]
    print(f"\n⇒ 全程相对差 中位 {np.median(rel_all):.2f}｜"
          f"尾 2/3 相对差 中位 {np.median(rel_tail):.2f}")
    print("   判据（预注册）：尾段相对差中位 < 0.20 ⇒ **通过**（时间压缩不改科学）")
    print("   ⚠️ 不通过时先排查：① 是否有参数漏标 ② k 是否过大（离散化）"
          " ③ subdiv 是否够（量化悬崖）")


if __name__ == "__main__":
    main()
