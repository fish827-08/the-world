"""A4 漂移源隔离探针（青梧 · 2026-09-26 独有贡献；T5 补 --daily 2026-09-26）。

用途：A4 时间压缩不变性判据不通过时，定位「漂移从哪个机制来」。
方法：逐个关机制（pleasure/predation/signal/ars/fruit/corpse），看
      尾段相对差中位是否显著下降 ⇒ 下降的那个机制就是漂移源。

已测结论（60×120、seed=42、predation 为主要控制机制时）：

| 关闭机制 | N(D=2400) | N(D=480) | 尾段相对差中位 | 判读 |
|---|---|---|---|---|
| 基线（全开） | 211 | 272 | 0.29 | 漂移源在高级机制 |
| 关 pleasure | 211 | 272 | 0.29 | 无变化（愉悦度纯观测） |
| 关 predation | 875 | 1057 | 0.21 | 捕食贡献 ~0.3 |
| 生物不动 speed=0 | 234→333 | 249→379 | 0.12 | 移动贡献 ~0.05 |
| 食物充足 fill=1.0 | 345→875 | 423→1006 | 0.15 | 与食物量无关 |

🔴 **口径（T5 修正版）**：
  · 默认模式 = **末点单点相对差**（`abs(N5-N1)/N1`），与原版一致（快）。
  · `--daily` 模式 = **逐日采样**：每天记录 N，算「尾段相对差中位」
    （取后 1/3 天的逐日相对差中位），与 A4 判据口径**一致**；
    报告表中 `speed=0` / `fill=1.0` 两行正是此口径 ⇒ 现在可由本脚本直接复现。
  · 核心结论不变：0.10~0.15 是引擎的**离散噪声地板**（能量量子化→繁殖触发
    时刻级联），与漏标无关（A3 已证逐过程守恒）；捕食再叠加 +0.3 尺度效应。
    2026-09-26 天平在 e5746b0 把捕食定位为**口径错不是漏标**：速率过程应
    `min(1, k·p)`（CAP）而非 `1-(1-p)^k`（PROB），k=5 时 PROB 少算 33% 出手
    次数 ⇒ 捕食偏弱 ⇒ 种群偏高。CAP 修正后 k=5 尾段相对差 0.48→0.32
    （`experiments/pred_rate_rule_test.py` 实测）。

用法
----
    python3.12 experiments/a4_drift_isolate.py            # 末点单点（快，全矩阵约 10 分钟）
    python3.12 experiments/a4_drift_isolate.py --fast     # 只跑关键五组
    python3.12 experiments/a4_drift_isolate.py --daily    # 逐日采样 + 尾段中位（复现报告表）
    python3.12 experiments/a4_drift_isolate.py --daily --fast
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
try:                                                          # noqa: E402
    # 云启 T3（dev/yunqi-t2t3 @ d7655dd）新增 apply_speed_std（speed_max=speed_gain=v_max、
    # subdiv=40、R<5 报警），并删除 gain_for/subdiv_for。此处做兼容：
    # 若 main 已合入 T3 用新件，否则回退旧件（main 现版仍有 gain_for/subdiv_for）。
    from experiments.scaling_rescale import apply_speed_std as _apply_speed_std
    HAS_APPLY_SPEED_STD = True
except ImportError:                                          # noqa: E402
    _apply_speed_std = None
    HAS_APPLY_SPEED_STD = False


def _speed_wiring(c: SimConfig, v_max: float, anchor_v: float) -> None:
    """速度基因接线：新件 apply_speed_std（若有）否则旧件 gain_for/subdiv_for。

    🔴 T5 修复（2026-09-26，Q2 更深根因）：`speed=0`（生物不动组）时
    `subdiv_for(0, 40)` = round(40/1e-9) = 4e10 ⇒ 引擎 `_steps_hist` 分配
    (4e10+1,) int64 ≈ 298 GiB ⇒ **原入库脚本跑 speed=0 组必崩**（报告表那两行
    数字只能来自未入库版——这就是「可复现性缺口」的机器级原因）。
    修法：`v_max==0` 时 subdiv 退化为锚点值（生物不动，步数直方图恒 0，
    与锚点 subdiv 逐位等价，仅省内存）。
    """
    c.subpos.speed_max = v_max
    if HAS_APPLY_SPEED_STD:
        _apply_speed_std(c.subpos, v_max)   # 内部已设 speed_max/speed_gain/subdiv
    else:
        from experiments.scaling_rescale import gain_for, subdiv_for
        c.subpos.speed_gain = gain_for(max(v_max, 1e-9), A_AGE)
        c.subpos.subdiv = subdiv_for(max(v_max, anchor_v), 40)


def run(D: int, days: int, disable: tuple[str, ...] = (),
        speed: float | None = None, fill: float = 0.5,
        daily: bool = False) -> list[int] | int:
    """跑 `days` 个世界日。

    daily=False → 返回末值种群 N（int，与原版一致）。
    daily=True  → 返回逐日末值 N 列表（长度 = days，首日为第一个世界日末）。
    """
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
    apply_speed_std(c.subpos, v_max)                 # gain=v_max、subdiv=80（R213 §四）
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
    if not daily:
        for _ in range(days * tpd):
            eng.step()
        return len(eng._flat)
    out: list[int] = []
    for d in range(1, days + 1):
        for _ in range(tpd):
            eng.step()
        out.append(len(eng._flat))
    return out


def tail_median_rel(p1: list[int], p5: list[int]) -> float:
    """逐日相对差的后 1/3 段中位（A4 判据口径：尾段相对差中位）。"""
    rel = [abs(b - a) / max(a, 1.0) for a, b in zip(p1, p5)]
    return float(np.median(rel[len(rel) // 3:]))


def main() -> None:
    ap = argparse.ArgumentParser(description="A4 漂移源隔离探针")
    ap.add_argument("--fast", action="store_true", help="只跑关键五组")
    ap.add_argument("--daily", action="store_true",
                    help="逐日采样模式：输出逐日 N 与尾段相对差中位（A4 判据口径）")
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

    title = ("A4 漂移源隔离（60×120，seed=42，逐日采样尾段中位）" if a.daily
             else "A4 漂移源隔离（60×120，seed=42，末点单点相对差）")
    print(title)
    print(f"{'关闭机制':<22}{'N(D=2400)':>12}{'N(D=480)':>11}{'相对差':>12}")

    if a.daily:
        # 逐日采样：先各跑一遍收集逐日序列，再统一打印尾段中位
        seqs: list[tuple[str, list[int], list[int]]] = []
        for tag, disable, speed, fill in matrix:
            p1 = run(2400, days, disable, speed, fill, daily=True)
            p5 = run(480, days, disable, speed, fill, daily=True)
            assert isinstance(p1, list) and isinstance(p5, list)
            seqs.append((tag, p1, p5))
        # 逐日明细表（全部组共用同一组日索引）
        print("\n逐日 N 明细（日 | N1 | N5 | 当日相对差）：")
        for d in range(days):
            row = []
            for tag, p1, p5 in seqs:
                rel = abs(p5[d] - p1[d]) / max(p1[d], 1.0)
                row.append(f"{p1[d]}/{p5[d]}/{rel:.2f}")
            print(f"日{d+1}: " + "  ".join(f"[{t}] {v}" for t, v in zip(
                [s[0] for s in seqs], row)))
        print()
        for tag, p1, p5 in seqs:
            rel = tail_median_rel(p1, p5)
            print(f"{tag:<22}{p1[-1]:>12}{p5[-1]:>11}{rel:>12.2f}")
    else:
        for tag, disable, speed, fill in matrix:
            n1 = run(2400, days, disable, speed, fill)
            n5 = run(480, days, disable, speed, fill)
            assert isinstance(n1, int) and isinstance(n5, int)
            rel = abs(n5 - n1) / max(n1, 1.0)
            print(f"{tag:<22}{n1:>12}{n5:>11}{rel:>12.2f}")


if __name__ == "__main__":
    main()
