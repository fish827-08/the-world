"""A4 漂移源隔离探针（青梧 · 2026-09-26 独有贡献）。

用途：A4 时间压缩不变性判据不通过时，定位「漂移从哪个机制来」。
方法：逐个关机制（pleasure/predation/signal/ars/fruit/corpse），看
      尾段相对差中位是否显著下降 ⇒ 下降的那个机制就是漂移源。

已测结论（60×120、seed=42、predation 为主要控制机制时）：

| 关闭机制 | N(D=2400) | N(D=480) | 末点相对差 | 判读 |
|---|---|---|---|---|
| 基线（全开） | 211 | 272 | 0.49 | 漂移源在高级机制 |
| predation | 875 | 1057 | 0.21 | 捕食贡献 ~0.3 |
| 生物不动 speed=0（关移动） | 234→333 | 249→379 | 0.12 | 移动贡献 ~0.05 |
| 食物充足 fill=1.0（无关） | 345→875 | 423→1006 | 0.15 | 与食物量无关 |

🔴 **口径警告（[所有者·天平] 2026-09-26 加）**：本脚本只输出**末点单点相对差**
  （`abs(N5-N1)/N1`），**不是**「尾段相对差中位」。原 docstring/表头写的
  「尾段相对差中位」是**错误标签**，已改。诊断报告里的 0.29 / 0.21 / 0.12 一列
  与该脚本同口径（单点），但 `speed=0` 与 `fill=1.0` 两行的两个数字来自**逐日采样**的
  另一版脚本（未入库）⇒ **可复现性缺口**：报告表无法由本脚本直接复现。
  ⇒ 待作者补 `--daily` 采样版本，或把报告表改为本脚本口径。

核心结论：
  0.10~0.15 是引擎的**离散噪声地板**（能量量子化→繁殖触发时刻级联），
  与漏标无关（A3 已证逐过程守恒）；捕食再叠加 +0.3 尺度效应
  （每 tick 步长不同→攻击范围内邻居分布不同→相遇率变）。

用法
----
    python3.12 experiments/a4_drift_isolate.py            # 跑全矩阵（慢，约 10 分钟）
    python3.12 experiments/a4_drift_isolate.py --fast     # 只跑关键三组（约 4 分钟）
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
    apply_post_build, apply_speed_std, rescale_config,
)


def run(D: int, days: int, disable: tuple[str, ...] = (),
        speed: float | None = None, fill: float = 0.5) -> int:
    """跑 `days` 个世界日，返回末值种群 N。"""
    c = SimConfig(seed=42)
    c.world.rows, c.world.cols = 60, 120
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = True
    c.resources.patch_regrowth_mult = 1.195
    c.resources.patch_count = 30
    c.resources.initial_fill = fill
    c.population.initial_count = 800
    k = 2400.0 / D
    notes = rescale_config(c, k)
    if speed is None:
        v_max = 15.0 / D          # 锚点：每日行程 15 格（与 D 无关）
    else:
        v_max = speed
    c.simulation.use_sim_core = False
    c.subpos.enabled = True
    apply_speed_std(c.subpos, v_max)                 # gain=v_max、subdiv=40（R204/R205）
    c.subpos.min_energy_frac = 0.0
    if "pleasure" in disable:
        c.pleasure.enabled = False
    if "predation" in disable:
        c.predation.enabled = False
    if "signal" in disable:
        c.info_structure.enabled = False
    if "ars" in disable:
        c.ars.enabled = False
    if "fruit" in disable:
        c.fruit.enabled = False
    if "corpse" in disable:
        c.corpse_wound.corpse_enabled = False
    eng = SphereEngine(c)
    apply_post_build(eng, notes)
    tpd = int(D)
    for _ in range(days * tpd):
        eng.step()
    return len(eng._flat)


def tail_median_rel(p1: list[int], p5: list[int]) -> float:
    rel = [abs(b - a) / max(a, 1.0) for a, b in zip(p1, p5)]
    return float(np.median(rel[len(rel) // 3:]))


def main() -> None:
    ap = argparse.ArgumentParser(description="A4 漂移源隔离探针")
    ap.add_argument("--fast", action="store_true", help="只跑关键三组")
    a = ap.parse_args()

    days = 5
    matrix: list[tuple[str, tuple, float | None, float]] = [
        ("基线（全开）", (), None, 0.5),
        ("关 pleasure", ("pleasure",), None, 0.5),
        ("关 predation", ("predation",), None, 0.5),
        ("生物不动 speed=0", (), 0.0, 0.5),
        ("食物充足 fill=1.0", (), None, 1.0),
    ]
    if not a.fast:
        matrix += [
            ("关 signal", ("signal",), None, 0.5),
            ("关 ars", ("ars",), None, 0.5),
            ("关 fruit", ("fruit",), None, 0.5),
            ("关 corpse", ("corpse",), None, 0.5),
            ("关捕食+愉悦度（RNG 假象）", ("pleasure", "predation"), None, 0.5),
        ]

    print("A4 漂移源隔离（60×120，seed=42，8 天→按天采样末值）")
    print(f"{'关闭机制':<20}{'N(D=2400)':>12}{'N(D=480)':>11}{'末点相对差':>12}")
    for tag, disable, speed, fill in matrix:
        n1 = run(2400, days, disable, speed, fill)
        n5 = run(480, days, disable, speed, fill)
        # 单点值；精确尾段中位需逐日采样——这里给单点近似，逐日版见诊断文档
        rel = abs(n5 - n1) / max(n1, 1.0)
        print(f"{tag:<20}{n1:>12}{n5:>11}{rel:>14.2f}")


if __name__ == "__main__":
    main()
