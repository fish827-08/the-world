"""patchy_vs_uniform：L1 斑块资源（守恒版）的生态对比实验。

目的
----
验证斑块化（守恒版）是否在不破坏总食物量的前提下，带来设计效果：
  1. 种群存活：patchy 模式下种群不灭绝、能持续繁衍若干代；
  2. 空间聚集：patchy 模式下个体空间分布更聚集（变异系数更高）；
  3. 基因多样性：patchy 模式下基因多样性不塌缩（选择压不单一）；
  4. 总量守恒：patchy 与 uniform 的总食物量在同一量级（容量/再生守恒生效）。

用法
----
    python -m experiments.patchy_vs_uniform [-t 2000]

依赖
----
- simulation.sphere_engine.SphereEngine
- simulation.config.SimConfig
直接读引擎内部数组做统计，不新增接口。
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from simulation.config import SimConfig
from simulation.sphere_engine import SphereEngine

SEEDS = (42, 7, 123)
RECORD_EVERY = 100  # 每多少 tick 记录一次


def make_config(seed: int, distribution: str) -> SimConfig:
    cfg = SimConfig(seed=seed)
    cfg.population.initial_count = 200
    cfg.simulation.ticks = 10_000  # 上限，实际由 -t 控制
    cfg.resources.distribution = distribution
    return cfg


def spatial_cv(engine: SphereEngine) -> float:
    """空间聚集度：每格个体数的变异系数（std/mean），越高越聚集。"""
    counts = np.bincount(engine._flat, minlength=engine.world.n_cells)
    mean = counts.mean()
    if mean < 1e-9:
        return 0.0
    return float(counts.std() / mean)


def gene_diversity(engine: SphereEngine) -> float:
    """基因多样性：所有基因位 std 的均值。"""
    if engine.alive_count() == 0:
        return 0.0
    return float(engine._genes.std(axis=0).mean())


def run_one(seed: int, distribution: str, ticks: int) -> dict:
    """跑一组实验，返回记录列表 + 终态统计。"""
    cfg = make_config(seed, distribution)
    cfg.simulation.ticks = ticks
    engine = SphereEngine(cfg)

    records = []
    t0 = time.time()
    for t in range(1, ticks + 1):
        stats = engine.step()
        if t % RECORD_EVERY == 0 or t == ticks or engine.extinct:
            records.append({
                "tick": t,
                "population": stats.population,
                "total_food": stats.total_resource,
                "spatial_cv": spatial_cv(engine),
                "gene_div": gene_diversity(engine),
                "max_gen": engine._max_generation,
                "born": engine.total_born,
                "died": engine.total_died,
            })
        if engine.extinct:
            break
    elapsed = time.time() - t0

    final = records[-1]
    return {
        "seed": seed,
        "distribution": distribution,
        "elapsed_s": round(elapsed, 2),
        "ticks_run": final["tick"],
        "extinct": engine.extinct,
        "final_population": final["population"],
        "final_food": round(final["total_food"], 1),
        "final_spatial_cv": round(final["spatial_cv"], 3),
        "final_gene_div": round(final["gene_div"], 4),
        "max_generation": final["max_gen"],
        "total_born": final["born"],
        "total_died": final["died"],
        "records": records,
    }


def print_summary(results: list[dict]) -> None:
    """打印终态对比表。"""
    print("\n" + "=" * 90)
    print(f"{'分布':<10}{'seed':<8}{'种群':<8}{'总食物':<12}{'空间CV':<10}{'基因多样性':<12}{'世代':<8}{'出生':<8}{'死亡':<8}{'耗时s':<8}")
    print("-" * 90)
    for r in results:
        print(
            f"{r['distribution']:<10}{r['seed']:<8}{r['final_population']:<8}"
            f"{r['final_food']:<12}{r['final_spatial_cv']:<10}{r['final_gene_div']:<12}"
            f"{r['max_generation']:<8}{r['total_born']:<8}{r['total_died']:<8}{r['elapsed_s']:<8}"
        )
    print("=" * 90)

    # 分组均值
    for dist in ("uniform", "patchy"):
        group = [r for r in results if r["distribution"] == dist]
        if not group:
            continue
        print(f"\n[{dist}] 三 seed 均值：")
        print(f"  终态种群   = {np.mean([r['final_population'] for r in group]):.0f}")
        print(f"  终态总食物 = {np.mean([r['final_food'] for r in group]):.1f}")
        print(f"  空间CV     = {np.mean([r['final_spatial_cv'] for r in group]):.3f}")
        print(f"  基因多样性 = {np.mean([r['final_gene_div'] for r in group]):.4f}")
        print(f"  最大世代   = {np.mean([r['max_generation'] for r in group]):.1f}")
        extinct_count = sum(1 for r in group if r["extinct"])
        print(f"  灭绝次数   = {extinct_count}/3")


def main() -> None:
    parser = argparse.ArgumentParser(description="L1 斑块资源对比实验")
    parser.add_argument("-t", "--ticks", type=int, default=2000, help="每 run 跑多少 tick")
    args = parser.parse_args()

    print(f"L1 斑块资源对比实验：每 run {args.ticks} tick，3 个 seed，两组分布")
    print(f"记录间隔：每 {RECORD_EVERY} tick")

    results = []
    for dist in ("uniform", "patchy"):
        for seed in SEEDS:
            print(f"  运行 {dist} seed={seed} ...", end=" ", flush=True)
            r = run_one(seed, dist, args.ticks)
            print(f"完成（{r['ticks_run']} tick, 种群={r['final_population']}）")
            results.append(r)

    print_summary(results)


if __name__ == "__main__":
    main()
