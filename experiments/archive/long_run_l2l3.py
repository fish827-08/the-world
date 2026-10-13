"""long_run_l2l3：L2+L3 系统长跑实验（愉悦度+信号+感知）。

记录指标：种群数、信号密度、愉悦度均值/方差、g14/g15 基因均值、世代数。
用于验证系统长期稳定性，并观测信号/感知基因是否被自然选择保留。

用法
----
    python -m experiments.long_run_l2l3 [-t 50000] [-n 500] [--patchy]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from simulation.config import SimConfig
from simulation.sphere_engine import SphereEngine


def run(ticks: int, n: int, patchy: bool, seed: int = 42) -> list[dict]:
    cfg = SimConfig(seed=seed)
    cfg.population.initial_count = n
    cfg.simulation.ticks = ticks
    if patchy:
        cfg.resources.distribution = "patchy"
    engine = SphereEngine(cfg)

    records = []
    t0 = time.time()
    for t in range(1, ticks + 1):
        engine.step()
        if t % 500 == 0 or engine.extinct:
            ps = engine.pleasure_summary()
            records.append({
                "tick": t,
                "population": engine.alive_count(),
                "max_gen": engine._max_generation,
                "signal_cells": engine.signals.active_count(),
                "signal_marks": engine.signals.total_marks(),
                "valence_mean": round(ps["valence_mean"], 4),
                "valence_std": round(ps["valence_std"], 4),
                "arousal_mean": round(ps["arousal_mean"], 4),
                "g14_mean": round(float(engine._genes[:, 14].mean()), 4) if engine.alive_count() else 0,
                "g15_mean": round(float(engine._genes[:, 15].mean()), 4) if engine.alive_count() else 0,
                "total_energy": round(float(engine._energy.sum()), 1),
            })
        if engine.extinct:
            break
    elapsed = time.time() - t0
    print(f"\n完成：{records[-1]['tick']} tick, {elapsed:.1f}s, "
          f"{records[-1]['tick']/elapsed:.0f} tick/s")
    return records


def print_table(records: list[dict]) -> None:
    print(f"\n{'tick':<8}{'种群':<8}{'世代':<8}{'信号格':<8}{'信号点':<8}"
          f"{'valence':<10}{'arousal':<10}{'g14':<8}{'g15':<8}")
    print("-" * 80)
    for r in records:
        print(f"{r['tick']:<8}{r['population']:<8}{r['max_gen']:<8}"
              f"{r['signal_cells']:<8}{r['signal_marks']:<8}"
              f"{r['valence_mean']:<10}{r['arousal_mean']:<10}"
              f"{r['g14_mean']:<8}{r['g15_mean']:<8}")


def main() -> None:
    parser = argparse.ArgumentParser(description="L2+L3 长跑实验")
    parser.add_argument("-t", "--ticks", type=int, default=5000)
    parser.add_argument("-n", "--population", type=int, default=500)
    parser.add_argument("--patchy", action="store_true", help="启用斑块资源")
    parser.add_argument("-s", "--seed", type=int, default=42)
    args = parser.parse_args()

    mode = "patchy" if args.patchy else "uniform"
    print(f"L2+L3 长跑：{mode}, N={args.population}, {args.ticks} tick, seed={args.seed}")
    records = run(args.ticks, args.population, args.patchy, args.seed)
    print_table(records)


if __name__ == "__main__":
    main()
