"""捕食「每 tick 概率」的守恒口径对比（14.10 / P0.0 A2 追加验证）。

背景
----
时间压缩（D → D/k）时，`predation.attack_prob_coef` 是**每 tick 伯努利概率**。
它的正确改法取决于「守恒目标」是什么，而这是**两件不同的事**：

| 口径 | 公式 | 守恒的量 | k=5, p=0.2 的结果 | 每世界日尝试数 |
|---|---|---|---|---|
| **PROB（现表）** | `p' = 1 − (1 − p)^k` | 「每天**至少**出手一次」的概率 | 0.672 | 480 × 0.672 = **322.7**（−33%） |
| **CAP（本探针）** | `p' = min(1, k·p)` | 「每天出手**次数**」的期望 | 1.000 | 480 × 1.000 = **480.0**（= k=1） |
| （不变） | `p' = p` | — | 0.200 | **96.0**（−80%） |

捕食是**速率过程**（一天可以出手很多次，理论上 480 次）⇒ 物理上该守恒的是**次数**，
不是「至少一次」。⇒ 现表的 PROB 口径**目标选错**，这正是青梧隔离出来的
「捕食尺度效应 +0.3」的量化来源。

离散化有效域
------------
CAP 口径要求 `k·p ≤ 1` 才有解。`attack_prob_coef` 默认 0.2、`hunger≤1`、`g16≤1`
⇒ `p_max = 0.2` ⇒ **`k ≤ 5` 才可表示**，且 `k·p → 1` 时方差被压缩到 0（退化）。
保留方差的实用上限：**`k·p ≤ 0.5` ⇒ `k ≤ 2.5`**。

用途
----
    python.exe experiments/pred_rate_rule_test.py --days 5 --pop 800
    python.exe experiments/pred_rate_rule_test.py --days 5 --pop 800 --ks 1,2,3,5

预注册判据（跑之前写死，不许事后改）
----------------------------------
**主判据**：CAP 口径在 k=5 时与 k=1 的尾段相对差 **< PROB 口径同口径读数**
（即「换口径应减小漂移」）。
**支持 1**：CAP 口径下 `每世界日尝试数` 与 k=1 相等（解析可算，容差 1e-9）。
**支持 2**：捕食关臂（`--no-predation`）在两种口径下读数相同（口径只影响开臂）。

⚠️ **不许用单点 N 比大小代替尾段相对差**（见 `_archive/2026-10-10-退役团队-归档/docs-旧档/tasks/验收-P0.0阶段A-20260926.md` §4.1：
单次实现种子间 SD 可达均值 41%）。本探针按「世界日」逐个采样尾段。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- R98 纪律：Windows GBK 控制台兜底 ---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from simulation.config import SimConfig                              # noqa: E402
from simulation.sphere_engine import SphereEngine                    # noqa: E402
from experiments.scaling_rescale import (                            # noqa: E402
    apply_post_build, apply_speed_std, rescale_config,
)

D0 = 2400             # 基准每昼夜 tick 数


def run_series(D: int, days: int, pop: int, seed: int, travel_per_day: float,
               rule: str, predation: bool) -> list[int]:
    """跑 `days` 个世界日，**逐日**返回末值种群 N（供尾段统计）。"""
    k = D0 / D
    c = SimConfig(seed=seed)
    c.world.rows, c.world.cols = 60, 120
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = True
    c.resources.patch_regrowth_mult = 1.195
    c.resources.patch_count = 30
    c.resources.initial_fill = 0.5
    c.population.initial_count = pop

    # 先按现表（PROB 口径）重标，再按 rule 覆盖捕食那一项
    p_loc = float(c.predation.attack_prob_coef)
    notes = rescale_config(c, k)
    if rule == "cap":
        p_new = min(1.0, k * p_loc)
    elif rule == "prob":
        p_new = 1.0 - (1.0 - p_loc) ** k
    elif rule == "none":
        p_new = p_loc
    else:
        raise ValueError(f"未知 rule={rule!r}")
    c.predation.attack_prob_coef = p_new
    if not predation:
        c.predation.enabled = False

    v_max = travel_per_day / D
    c.simulation.use_sim_core = False
    c.subpos.enabled = True
    apply_speed_std(c.subpos, v_max)                 # gain=v_max、subdiv=80（R213 §四）
    c.subpos.min_energy_frac = 0.0

    eng = SphereEngine(c)
    apply_post_build(eng, notes)

    out: list[int] = []
    for _ in range(days):
        for _ in range(int(D)):
            eng.step()
        out.append(len(eng._flat))
    return out


def tail_median_rel(a: list[int], b: list[int]) -> float:
    """尾段（后 2/3）相对差中位。"""
    rel = [abs(y - x) / max(x, 1.0) for x, y in zip(a, b)]
    return float(np.median(rel[len(rel) // 3:]))


def main() -> None:
    ap = argparse.ArgumentParser(description="捕食每 tick 概率的守恒口径对比")
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--pop", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--travel-per-day", type=float, default=15.0)
    ap.add_argument("--ks", default="1,5", help="逗号分隔的 k 列表（k=2400/D）")
    ap.add_argument("--no-predation", action="store_true", help="捕食关臂（支持判据 2）")
    a = ap.parse_args()

    ks = [float(x) for x in a.ks.split(",")]
    p_loc = float(SimConfig().predation.attack_prob_coef)

    print(f"== 捕食守恒口径对比：60×120 / pop {a.pop} / seed {a.seed} / "
          f"{a.days} 世界日 / 每日行程 {a.travel_per_day} 格 ==")
    arm = "  ｜ 捕食关臂" if a.no_predation else "  ｜ 捕食开"
    print(f"   attack_prob_coef（k=1）= {p_loc}{arm}")
    print()
    print(f"{'k':>5}{'口径':>8}{'p(k)':>10}{'尝试数/日':>11}{'相对k=1':>9}"
          f"{'N(尾段中位)':>12}{'尾段相对差':>11}")
    print('-' * 72)

    for rule in ("prob", "cap"):
        p1 = p_loc
        n_day = p1 * D0
        series1 = run_series(D0, a.days, a.pop, a.seed, a.travel_per_day,
                             rule, not a.no_predation)
        print(f"{1.0:>5}{rule:>8}{p1:>10.4f}{n_day:>11.1f}{1.0:>9.2f}"
              f"{np.median(series1[len(series1)//3:]):>12.1f}{0.0:>11.2f}")
        for k in ks:
            if abs(k - 1.0) < 1e-9:
                continue
            D = int(round(D0 / k))
            p_k = (1.0 - (1.0 - p_loc) ** k) if rule == "prob" else min(1.0, k * p_loc)
            n_k = p_k * D
            s = run_series(D, a.days, a.pop, a.seed, a.travel_per_day,
                           rule, not a.no_predation)
            rel = tail_median_rel(series1, s)
            flag = ""
            if not a.no_predation:
                flag = "  ← 目标" if rule == "cap" else ""
            print(f"{k:>5.0f}{rule:>8}{p_k:>10.4f}{n_k:>11.1f}{n_k/n_day:>9.3f}"
                  f"{np.median(s[len(s)//3:]):>12.1f}{rel:>11.2f}{flag}")
        print()

    print("读法：")
    print("  · 「尝试数/日」是解析值 —— 它才该守恒（速率过程）。")
    print("  · 「尾段相对差」越小越好；CAP 行应优于 PROB 行（主判据）。")
    print("  · CAP 在 k=5 时 p→1.0 ⇒ 方差被压掉（退化），这是离散化上限的可见形态。")


if __name__ == "__main__":
    main()
