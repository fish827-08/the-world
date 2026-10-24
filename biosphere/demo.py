"""biosphere.demo：MVP 冒烟入口（CPU 参考实现）。

用法（仓库根目录，一条命令）::

    python -m biosphere.demo --ticks 1000 --seed 42

- 不依赖旧 ``sphere_engine``；只 import numpy + ``biosphere`` 包；
- 输出：metrics JSON（``--out`` 指定路径）+ 终端 ASCII 计数摘要；
- 同 seed 两次运行 ⇒ metrics JSON **逐位一致**（B4 测试钉住）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from biosphere.metrics import metrics_json
from biosphere.simulator import BioMVPConfig, BiosphereSimulator


def _summary_lines(m: dict) -> List[str]:
    """ASCII 计数摘要（只用 ASCII ⇒ 任何代码页下不乱码）。"""
    pop = m["population"]
    beh = m["behaviors"]["counts"]
    lin = m["lineage"]
    en = m["energy"]
    causes = pop["death_causes"]
    cause_txt = ", ".join(f"{k}={v}" for k, v in sorted(causes.items())) or "none"
    return [
        "=== biosphere MVP smoke summary ===",
        f"seed={m['seed']} ticks={m['ticks']} "
        f"world={m['world']['rows']}x{m['world']['cols']}",
        f"final_alive={pop['final_alive']} peak_alive={pop['peak_alive']} "
        f"born={pop['total_born']} died={pop['total_died']} ({cause_txt})",
        f"energy: final_total={en['final_total']:.2f} "
        f"cost_paid={en['intent_cost_paid']:.2f} "
        f"world_inflow={en['world_inflow']:.2f}",
        f"lineage: n={lin['n_individuals']} roots={lin['n_roots']} "
        f"max_generation={lin['max_generation']}",
        "behaviors(whole-run labels): " + " ".join(
            f"{k}={v}" for k, v in sorted(beh.items())),
        f"interactions: attack_pairs={m['interactions']['n_attack_pairs']} "
        f"share_pairs={m['interactions']['n_share_pairs']} "
        f"signalers={m['interactions']['n_signalers']}",
    ]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m biosphere.demo",
        description="biosphere MVP 冒烟（CPU 参考实现，合成小世界）")
    parser.add_argument("--ticks", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=64)
    parser.add_argument("--pop", type=int, default=48)
    parser.add_argument("--out", type=str, default="",
                        help="metrics JSON 输出路径（默认 ./biosphere_metrics.json）")
    parser.add_argument("--quiet", action="store_true", help="不打印摘要")
    args = parser.parse_args(argv)

    cfg = BioMVPConfig(
        rows=args.rows, cols=args.cols, seed=args.seed, ticks=args.ticks,
        init_pop=args.pop,
    )
    sim = BiosphereSimulator(cfg)
    log = sim.run()
    metrics = metrics_json(log, sim.world, seed=args.seed, ticks=args.ticks)

    out_path = Path(args.out) if args.out else Path("biosphere_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(metrics, fh, sort_keys=True, indent=1)
        fh.write("\n")

    if not args.quiet:
        for line in _summary_lines(metrics):
            print(line)
        print(f"metrics_json: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
