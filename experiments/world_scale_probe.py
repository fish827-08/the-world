"""世界尺度性能探针 —— **放大世界到底贵不贵？**（底层架构讨论的实测依据）

三个问题（fish 2026-09-25 提出"把世界从格子变成画布"）
----------------------------------------------------
1. 世界放大 N 倍，每 tick 成本涨多少？（**关键是线性还是更差**）
2. 成本主要来自「世界面积」还是「个体数量」？（决定该往哪边优化）
3. 内存与一次性构造（邻居表）的代价多大？

做法
----
固定 13.5 B 臂底盘（patchy + bgzero），只动 `world.rows/cols` 与 `population.initial_count`；
斑块数按面积等比缩放（保持**斑块密度**不变，否则比的是"食物变了"而不是"世界大了"）。
每组跑 `--ticks` 步取平均，重复 `--reps` 次取**最小值**（降噪）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from simulation.config import SimConfig  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402

BASE_CELLS = 60 * 120
BASE_PATCHES = 30


def make_cfg(seed: int, rows: int, cols: int, pop: int) -> SimConfig:
    c = SimConfig(seed=seed)
    c.world.rows = rows
    c.world.cols = cols
    c.simulation.use_sim_core = False
    c.simulation.l2_dash = False
    c.population.initial_count = pop
    c.population.max_count = max(pop * 2, 6000)
    c.predation.enabled = False
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = True
    # 斑块数按面积等比 ⇒ 斑块**密度**不变（否则测的是食物变化，不是世界变化）
    c.resources.patch_count = max(4, int(round(BASE_PATCHES * rows * cols / BASE_CELLS)))
    c.resources.patch_radius = 2
    c.resources.patch_regrowth_mult = 1.195
    o = c.organisms
    o.max_energy = 600.0
    o.initial_energy = 300.0
    o.starve_frac = 0.30
    o.exhaust_frac = 0.17
    o.eat_efficiency = 7.5
    o.assim_herb = 0.4
    o.assim_carn = 0.8
    o.stomach_cap_mass = 25.0
    o.eat_threshold_frac = 0.6
    o.photo_max = 0.0
    return c


def measure(rows: int, cols: int, pop: int, ticks: int, reps: int) -> dict:
    t_build = time.perf_counter()
    eng = SphereEngine(make_cfg(42, rows, cols, pop))
    build = time.perf_counter() - t_build
    n0 = len(eng._flat)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(ticks):
            eng.step()
        best = min(best, (time.perf_counter() - t0) / ticks)
    cells = rows * cols
    return {"rows": rows, "cols": cols, "cells": cells, "pop": n0,
            "build_s": round(build, 3), "ms_tick": round(best * 1e3, 3),
            "grand_total": float(getattr(eng, "_run_born", 0))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="世界尺度性能探针")
    ap.add_argument("--sizes", default="60x120,120x240,240x480,480x960")
    ap.add_argument("--pops", default="200,2000")
    ap.add_argument("--ticks", type=int, default=150)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args(argv)

    sizes = []
    for tok in a.sizes.split(","):
        r, c = tok.lower().split("x")
        sizes.append((int(r), int(c)))
    pops = [int(p) for p in a.pops.split(",")]

    print("== 世界尺度性能探针（13.5 B 臂；斑块密度守恒；取 reps 最小值）==")
    print(f"   sizes={sizes}  pops={pops}  ticks={a.ticks}  reps={a.reps}\n")
    rows_out = []
    base = None
    for pop in pops:
        print(f"--- 个体数 {pop} ---")
        print(f"{'世界':>12} {'格数':>9} {'倍率':>6} {'构造s':>7} {'ms/tick':>9} {'相对':>7} "
              f"{'每格μs':>8} {'每个体μs':>9}")
        b0 = None
        for (r, c) in sizes:
            m = measure(r, c, pop, a.ticks, a.reps)
            if b0 is None:
                b0 = m["ms_tick"]
            cells = m["cells"]
            print(f"{r:>5}x{c:<6} {cells:>9} {cells/BASE_CELLS:>5.0f}x "
                  f"{m['build_s']:>7.3f} {m['ms_tick']:>9.3f} {m['ms_tick']/b0:>6.2f}x "
                  f"{m['ms_tick']*1000/max(cells,1):>8.3f} "
                  f"{m['ms_tick']*1000/max(m['pop'],1):>9.3f}")
            rows_out.append(m)
            if pop == pops[0] and base is None and cells == BASE_CELLS:
                base = m
        print()

    print("=== 判读 ===")
    small = [m for m in rows_out if m["cells"] == BASE_CELLS]
    big = [m for m in rows_out if m["cells"] == max(m2["cells"] for m2 in rows_out)]
    for m in small + big:
        print(f"  格数 {m['cells']:>8} 个体 {m['pop']:>6} ⇒ {m['ms_tick']:>8.3f} ms/tick")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
