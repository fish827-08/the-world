"""邻居表代价探针 —— 定位 `_nb_table` 补齐到 `cols` 列带来的内存与 cache 代价。

背景（14.10）
-------------
`simulation/sphere_engine.py` 构造期建了一张**给 Rust 用的统一步长邻居表**：

    nb_stride = max(8, self.world.cols)
    self._nb_table = np.full((n_cells, nb_stride), -1, dtype=np.int64)

极点行的邻居数是**相邻纬度带整行**（`cols` 个）⇒ 为了统一步长，表被补齐到 `cols` 列。
后果是内存 **∝ rows × cols²**（二次）：

| 世界 | 形状 | 内存 |
|---|---|---|
| 60×120 | (7200, 120) | 6.9 MB |
| 240×480 | (115200, 480) | 442 MB |
| 480×960 | (460800, 960) | **3 539 MB** |

而极点行只占 2 行（0.4% 的格）。行步长 960×8 = 7680 B ⇒ 随机按格取邻居
**每次一个 cache line 之外**（TLB 也吃紧）。

本探针在同一世界/同一参配置下对比：
* **宽表**（现状）：`(n_cells, cols)` int64
* **紧凑表**：`(n_cells, 8)` int32 + `_nb_len` 截到 8

⚠️ 两者**只对极点行不同**（极点格邻居数由 cols 变 8），其余格逐位等价。
极点行占 2/rows 行 ⇒ 本探针报的差异是**性能收益的上界估计**，落地时要单独处理极点语义。

用法
----
    .venv\\Scripts\\python.exe experiments/nb_table_probe.py
    .venv\\Scripts\\python.exe experiments/nb_table_probe.py --rows 480 --cols 960 --pop 2000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation.config import SimConfig                      # noqa: E402
from simulation.sphere_engine import SphereEngine            # noqa: E402


def _cfg(seed: int, rows: int, cols: int, pop: int, sim_core: bool) -> SimConfig:
    c = SimConfig(seed=seed)
    c.world.rows, c.world.cols = rows, cols
    c.simulation.use_sim_core = bool(sim_core)
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = True
    c.resources.patch_regrowth_mult = 1.195
    c.resources.patch_count = max(4, int(round(30 * rows * cols / 7200)))
    if pop > 0:
        c.population.initial_count = pop
    return c


def _compact(eng: SphereEngine, dtype=np.int32) -> None:
    """把引擎的宽邻居表换成长度 8 的紧凑 int32 表（`_nb_len` 同步截断）。"""
    n = eng.world.n_cells
    wide = eng._nb_table
    narrow = np.full((n, 8), -1, dtype=dtype)
    lens = np.minimum(eng._nb_len.astype(np.int64), 8)
    for c in range(n):
        L = int(lens[c])
        if L > 0:
            narrow[c, :L] = wide[c, :L].astype(dtype)
    eng._nb_table = narrow
    eng._nb_len = lens.astype(np.int64)


def measure(rows: int, cols: int, pop: int, ticks: int, reps: int,
            sim_core: bool, compact: bool) -> dict:
    t0 = time.perf_counter()
    eng = SphereEngine(_cfg(42, rows, cols, pop, sim_core))
    t_build = time.perf_counter() - t0
    nb_mb = eng._nb_table.nbytes / 1e6
    if compact:
        # Rust 侧签名要求 int64 ⇒ 走 Rust 时只紧凑化、不改 dtype
        _compact(eng, np.int64 if sim_core else np.int32)
    P = int(len(eng._flat))

    for _ in range(5):                     # 预热
        eng.step()
    t0 = time.perf_counter()
    for _ in range(ticks):
        eng.step()
    dt = (time.perf_counter() - t0) / ticks * 1e3
    return {"build_s": t_build, "nb_mb": nb_mb, "narrow_mb": eng._nb_table.nbytes / 1e6,
            "ms_tick": dt, "pop": P, "cells": eng.world.n_cells,
            "dtype": str(eng._nb_table.dtype), "stride": int(eng._nb_table.shape[1])}


def main() -> None:
    ap = argparse.ArgumentParser(description="邻居表（宽 vs 紧凑）代价探针")
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--cols", type=int, default=960)
    ap.add_argument("--pop", type=int, default=2000)
    ap.add_argument("--ticks", type=int, default=40)
    ap.add_argument("--sim-core", action="store_true", help="走 Rust 路径（默认 Python）")
    a = ap.parse_args()

    print(f"== 邻居表代价探针：{a.rows}x{a.cols}，个体 {a.pop}，"
          f"{a.ticks} tick，路径 = {'Rust' if a.sim_core else 'Python'} ==")
    print(f"{'配置':<26}{'构造s':>8}{'表内存MB':>10}{'步长':>6}{'ms/tick':>10}{'相对':>8}"
          f"{'省内存':>9}")
    base = None
    for compact, tag in ((False, "宽表(现状 n_cells×cols)"), (True, "紧凑(n_cells×8,i32)")):
        m = measure(a.rows, a.cols, a.pop, a.ticks, 1, a.sim_core, compact)
        rel = "-"
        if base is not None and m["ms_tick"] > 0:
            rel = f"{m['ms_tick'] / base:.2f}x"
        base = m["ms_tick"] if base is None else base
        save = (f"{m['nb_mb'] / m['narrow_mb']:.0f}x"
                if m['narrow_mb'] > 0 else "-")
        print(f"{tag:<26}{m['build_s']:>8.2f}{m['narrow_mb']:>10.1f}"
              f"{m['stride']:>6}{m['ms_tick']:>10.2f}{rel:>8}{save:>9}")
    print("\n⚠️ 紧凑表把极点行邻居数由 cols 改成 8 ⇒ 仅极点行语义变化（2/rows 行），"
          "\n   落地时需为极点单独保留 CSR/慢路径（见评审 §6.2）。")


if __name__ == "__main__":
    main()
