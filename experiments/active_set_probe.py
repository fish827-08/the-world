"""活跃集探针（B 路线的可行性基数）—— **不改引擎**，只统计"真正需要每 tick 计算的格"。

为什么必须先测这个
-----------------
路线 B 的全部收益来自一句话：**「只算有生物的地方」**。
收益倍数 = 世界总格数 ÷ 活跃格数。所以：

    收益 = (rows×cols) ÷ (活跃 chunk 数 × chunk 格数)

这个分母**不能靠猜**——它取决于生物聚不聚、聚在哪。
🔴 关键假设（本探针要验证的）：**活跃集与世界大小无关**——世界放大，生物还是聚在同样多的斑块上
⇒ 活跃 chunk 数不涨 ⇒ 成本不涨 ⇒ 解绑成立。

口径（三种，从松到紧）
---------------------
1. **含生物的 chunk**：至少有一个个体在里面（最松，B 的下界）
2. **含生物 ± 1 chunk**（= 3×3 邻域）：移动会跨边界，邻接块也得算（**B 的推荐口径**）
3. **含生物所在格 ± 8 邻格**（逐格精确口径）：最紧，模拟"只需要活跃格"

统计量
------
* 每 tick 的活跃 chunk 数（中位/P90）
* **曾经被访问过的 chunk 累计数**（= 追赶机制要覆盖的集合上限）
* 活跃格数 vs 全场格数 ⇒ **收益倍数**
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
CHUNK = 16


def make_cfg(seed: int, rows: int, cols: int, pop: int, patches: int = 0) -> SimConfig:
    c = SimConfig(seed=seed)
    c.world.rows, c.world.cols = rows, cols
    c.simulation.use_sim_core = False
    c.simulation.l2_dash = False
    c.population.initial_count = pop
    c.population.max_count = max(pop * 3, 20000)
    c.predation.enabled = False
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = True
    # 🔴 patches=0 ⇒ 密度守恒（斑块数随面积涨）；patches>0 ⇒ **固定斑块数**
    #   （世界放大 ⇒ 斑块间距变大）。后者才是"找食物成为挑战"的目标配置。
    c.resources.patch_count = (int(patches) if patches > 0
                               else max(4, int(round(BASE_PATCHES * rows * cols / BASE_CELLS))))
    c.resources.patch_radius = 2
    c.resources.patch_regrowth_mult = 1.195
    o = c.organisms
    o.max_energy = 600.0
    o.initial_energy = 300.0
    o.starve_frac = 0.30
    o.exhaust_frac = 0.17
    o.eat_efficiency = 7.5
    o.assim_herb = 0.4
    c.corpse_wound.corpse_enabled = False
    return c


def chunk_id(flat: np.ndarray, cols: int) -> np.ndarray:
    r = flat // cols
    cc = flat % cols
    return (r // CHUNK) * ((cols + CHUNK - 1) // CHUNK) + (cc // CHUNK)


def run(rows, cols, pop, ticks, seed=42, patches=0):
    eng = SphereEngine(make_cfg(seed, rows, cols, pop, patches))
    n_chunks_r = (rows + CHUNK - 1) // CHUNK
    n_chunks_c = (cols + CHUNK - 1) // CHUNK
    total_chunks = n_chunks_r * n_chunks_c
    cells = rows * cols
    nbr_chunks = ((n_chunks_r + 1) // 2) * ((n_chunks_c + 1) // 2)   # 邻接并集的上界估计

    hist_lo, hist_mid, hist_hi, ever = [], [], [], set()
    pop_hist = []
    t0 = time.perf_counter()
    for t in range(ticks):
        eng.step()
        flat = eng._flat[: len(eng._flat)]
        pop_hist.append(len(flat))
        cid = np.unique(chunk_id(flat, cols))
        hist_lo.append(len(cid))
        ever.update(int(x) for x in cid)
        # ±1 chunk 邻域（3×3）：用 chunk 行列 + 偏移展开
        cr = (cid // n_chunks_c)
        cc = (cid % n_chunks_c)
        offs = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)])
        nb = set()
        for dr, dc in offs:
            rr = cr + dr
            ccx = cc + dc
            m = (rr >= 0) & (rr < n_chunks_r) & (ccx >= 0) & (ccx < n_chunks_c)
            nb.update(int(x) for x in (rr[m] * n_chunks_c + ccx[m]))
        hist_mid.append(len(nb))
        hist_hi.append(min(cells, len(flat) * 9))
    dt = time.perf_counter() - t0
    return {
        "world": f"{rows}x{cols}", "cells": cells, "pop": int(np.median(pop_hist)),
        "total_chunks": total_chunks,
        "lo_med": int(np.median(hist_lo)), "mid_med": int(np.median(hist_mid)),
        "hi_med": int(np.median(hist_hi)),
        "ever_chunks": len(ever),
        "ms_tick": dt / ticks * 1e3,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="活跃集探针（B 路线收益基数）")
    ap.add_argument("--worlds", default="60x120,240x480,480x960")
    ap.add_argument("--pop", type=int, default=2000)
    ap.add_argument("--ticks", type=int, default=250)
    ap.add_argument("--patches", type=int, default=0, help=">0 ⇒ 固定斑块数（世界放大则间距变大）")
    a = ap.parse_args(argv)

    print(f"== 活跃集探针：个体 {a.pop}，chunk = {CHUNK}×{CHUNK}，{a.ticks} tick，"
          f"斑块数 = {'密度守恒' if a.patches <= 0 else a.patches} ==")
    print("   （含生物 = 下界口径；±1 chunk = 推荐口径；±8 邻格 = 逐格口径）\n")
    print(f"{'世界':>10}{'总格数':>10}{'chunk数':>9}{'含生物':>8}{'±1chunk':>9}{'±8邻格':>9}"
          f"{'曾访问chunk':>12}{'活跃格数':>10}{'收益倍数':>10}{'ms/tick':>9}")
    for w in a.worlds.split(","):
        r, c = (int(x) for x in w.lower().split("x"))
        pass
        d = run(r, c, a.pop, a.ticks, patches=a.patches)
        act_cells = d["mid_med"] * CHUNK * CHUNK
        gain = d["cells"] / max(act_cells, 1)
        print(f"{d['world']:>10}{d['cells']:>10}{d['total_chunks']:>9}{d['lo_med']:>8}"
              f"{d['mid_med']:>9}{d['hi_med']:>9}{d['ever_chunks']:>12}"
              f"{act_cells:>10}{gain:>9.1f}x{d['ms_tick']:>8.1f}")
    print("\n判读：若「±1 chunk」不随世界放大而涨 ⇒ **解绑成立**，收益倍数就是 B 的净收益。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
