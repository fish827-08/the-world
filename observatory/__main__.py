"""Headless 实验入口：py -m observatory（模块四 · 球面引擎版）。

用途：
- 直接运行实验矩阵（5 类实验：baseline / resource_pressure /
  resource_distribution / mutation_rate / repeated_seeds）；
- 全部实验输出保存到 --out 目录（每个实验一个子目录：
  manifest.json + generations.csv/json），最终结果可导出 / 复现。

示例：
    py -m observatory --out results --rows 60 --cols 120
    py -m observatory --names baseline --generations 10000 --sim-core
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from observatory.experiment import DEFAULT_MAX_TICKS, ExperimentRunner, build_plan
from persistence.io import save_survey, save_survey_markdown

_GROUP_CHOICES = [
    "baseline",
    "resource_pressure",
    "resource_distribution",
    "mutation_rate",
    "repeated_seeds",
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Evolution Observatory — 球面世界无头实验（模块四）"
    )
    p.add_argument("--out", default="results", help="输出根目录")
    p.add_argument("--rows", type=int, default=60, help="纬度行数（球面网格）")
    p.add_argument("--cols", type=int, default=120, help="经度列数（球面网格）")
    p.add_argument("--seed", type=int, default=42, help="base seed（跨运行复现）")
    p.add_argument("--tick-interval", type=int, default=200, help="兜底采样节拍")
    p.add_argument("--generations", type=int, default=1_000,
                   help="baseline 长程世代数（默认精简档；10,000 完整档会按此自动配足 tick 上限）")
    p.add_argument("--max-ticks", type=int, default=None,
                   help="baseline 长程 tick 硬上限；缺省按世代数自动推导")
    p.add_argument("--short", type=int, default=250, help="对照实验世代数")
    p.add_argument("--repeated", type=int, default=3, help="repeated seeds 种子数")
    p.add_argument("--repeated-gens", type=int, default=250, help="repeated seeds 世代数")
    p.add_argument("--sim-core", action="store_true",
                   help="种群数值管线走 Rust（sim_core；需先在 sim_core/ 安装扩展）")
    p.add_argument("--names", nargs="*", default=None,
                   help="只跑指定实验名（如 baseline pressure_low）；缺省跑计划全部")
    p.add_argument("--group", choices=_GROUP_CHOICES, default=None,
                   help="只跑指定实验组")
    args = p.parse_args(argv)

    runs = build_plan(
        base_seed=args.seed,
        max_generations=args.generations,
        short_generations=args.short,
        repeated_seed_count=args.repeated,
        repeated_generations=args.repeated_gens,
        baseline_max_ticks=args.max_ticks,
    )
    only = list(args.names) if args.names is not None else None
    if args.group is not None:
        only = [r.spec.name for r in runs if r.spec.group == args.group]

    runner = ExperimentRunner(
        base_seed=args.seed,
        world_size=(args.rows, args.cols),
        tick_interval=args.tick_interval,
        use_sim_core=args.sim_core,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = runner.run_plan(runs, only=only, quiet=False, out_dir=out)

    save_survey(results, out)
    save_survey_markdown(results, out)
    print(f"\n结果已保存到: {out.resolve()}")
    survey_md = (out / "__survey__.md").read_text(encoding="utf-8")
    print(survey_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())