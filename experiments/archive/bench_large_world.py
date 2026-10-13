#!/usr/bin/env python3
"""
大世界性能基准（bench_large_world.py）

对比小世界（60×120=7200格）vs 大世界（200×400=80000格）的性能，
定位扩大世界后的瓶颈。

用法：
  python3 experiments/bench_large_world.py
"""

import time
import sys
from pathlib import Path

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.config import SimConfig, WorldConfig, PopulationConfig
from simulation.sphere_engine import SphereEngine


def make_config(rows: int, cols: int, init_count: int, max_count: int,
                use_sim_core: bool = True, seed: int = 42) -> SimConfig:
    cfg = SimConfig()
    cfg.seed = seed
    cfg.world = WorldConfig(rows=rows, cols=cols)
    cfg.population = PopulationConfig(initial_count=init_count, max_count=max_count)
    cfg.simulation.use_sim_core = use_sim_core
    return cfg


def bench(cfg: SimConfig, warmup: int = 2000, measure: int = 1000, label: str = "") -> dict:
    """预热 + 测量，返回性能指标。"""
    print(f"\n{'='*60}")
    print(f"基准: {label}")
    print(f"  世界: {cfg.world.rows}×{cfg.world.cols} = {cfg.world.rows*cfg.world.cols} 格")
    print(f"  种群: 初始={cfg.population.initial_count}, 上限={cfg.population.max_count}")
    print(f"  use_sim_core={cfg.simulation.use_sim_core}")
    print(f"{'='*60}")

    t0 = time.time()
    e = SphereEngine(cfg)
    init_time = time.time() - t0
    print(f"  初始化耗时: {init_time:.2f}s")

    # 预热
    print(f"  预热 {warmup} tick...")
    t0 = time.time()
    for _ in range(warmup):
        e.step()
    warmup_time = time.time() - t0
    n_warmup = e.alive_count()
    print(f"  预热完成: {warmup_time:.1f}s, 存活={n_warmup}, 速率={warmup/warmup_time:.1f} tick/s")

    # 测量
    print(f"  测量 {measure} tick...")
    t0 = time.time()
    for _ in range(measure):
        e.step()
    measure_time = time.time() - t0
    n_final = e.alive_count()
    rate = measure / measure_time

    print(f"  测量结果: {measure_time:.2f}s, 存活={n_final}")
    print(f"  >>> 性能: {rate:.1f} tick/s")
    print(f"  >>> 每tick耗时: {1000/rate:.2f} ms")

    # 内存估算
    n_cells = cfg.world.rows * cfg.world.cols
    mem_mb = (
        n_cells * 8 * 5 +  # resource_grid/capacity/temperature/fruit_grid/signal_age
        n_cells * 1 +      # signal_marks (uint8)
        n_cells * 8 * 6 +  # _nb_table (int64, 6 neighbors)
        n_final * 8 * 20 + # 个体数组（energy/stomach/genes/age/... 约20个数组）
        n_final * 120 * 8  # expectation (N*120)
    ) / 1024 / 1024
    print(f"  估算内存: {mem_mb:.0f} MB")

    return {
        "label": label,
        "rows": cfg.world.rows,
        "cols": cfg.world.cols,
        "n_cells": n_cells,
        "init_count": cfg.population.initial_count,
        "max_count": cfg.population.max_count,
        "alive": n_final,
        "rate": rate,
        "ms_per_tick": 1000 / rate,
        "init_time": init_time,
        "mem_mb": mem_mb,
    }


def main():
    results = []

    # 1. 小世界基线（60×120）
    cfg_small = make_config(60, 120, 200, 5000, use_sim_core=True)
    results.append(bench(cfg_small, warmup=3000, measure=1000,
                         label="小世界 60×120 (基线)"))

    # 2. 大世界（200×400）
    cfg_large = make_config(200, 400, 2000, 50000, use_sim_core=True)
    results.append(bench(cfg_large, warmup=1000, measure=500,
                         label="大世界 200×400 (11x)"))

    # 3. 大世界纯Python（对比Rust加速比）
    cfg_large_py = make_config(200, 400, 2000, 50000, use_sim_core=False)
    results.append(bench(cfg_large_py, warmup=200, measure=100,
                         label="大世界 200×400 纯Python"))

    # 汇总对比
    print("\n" + "=" * 80)
    print("性能汇总对比")
    print("=" * 80)
    print(f"{'配置':<35} {'格子':>8} {'存活':>6} {'tick/s':>10} {'ms/tick':>10} {'内存MB':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['label']:<35} {r['n_cells']:>8} {r['alive']:>6} "
              f"{r['rate']:>10.1f} {r['ms_per_tick']:>10.2f} {r['mem_mb']:>8.0f}")

    if len(results) >= 2:
        speedup = results[1]["rate"] / results[0]["rate"] if results[0]["rate"] > 0 else 0
        scale = results[1]["n_cells"] / results[0]["n_cells"]
        print(f"\n大世界 vs 小世界: 规模 {scale:.1f}x, 性能 {speedup:.2f}x")
        print(f"效率比（性能/规模）: {speedup/scale:.3f}（1.0=完美扩展，<1=有开销）")

    if len(results) >= 3:
        rust_speedup = results[1]["rate"] / results[2]["rate"] if results[2]["rate"] > 0 else 0
        print(f"大世界 Rust vs Python: {rust_speedup:.1f}x 加速")

    print("=" * 80)


if __name__ == "__main__":
    main()
