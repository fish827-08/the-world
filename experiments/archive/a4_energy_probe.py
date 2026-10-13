"""A4 能量/食物收支探针：ON vs OFF 的食物存量空间分布与种群位置。

判别：若 ON 把种群挤到极区、而中低纬食物大量未被取食 -> "食物存在但不可达"，
     生态崩溃是【空间错配/招募失败】而非【全球食物枯竭】。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import InfoStructureConfig, SimConfig  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402

TICKS = 1500
SN = 100
ROWS = 60
POLAR = np.r_[0:6, 54:60]        # 南北极带
EQUAT = np.arange(25, 35)        # 赤道带
POLAR_CELLS = (POLAR[:, None] * 120 + np.arange(120)[None, :]).ravel()
EQUAT_CELLS = (EQUAT[:, None] * 120 + np.arange(120)[None, :]).ravel()


def build(asym: bool, seed: int = 42) -> SphereEngine:
    c = SimConfig(seed=seed)
    c.simulation.ticks = TICKS
    c.simulation.use_sim_core = False
    c.population.initial_count = 200          # R4 manifest 真实口径
    c.population.max_count = 5000
    c.resources.distribution = "uniform"      # R4 manifest 真实口径
    d2 = InfoStructureConfig(enabled=True)
    d2.arbitrary_codebook = False
    d2.learning_rate = 0.05
    if not asym:
        d2.perception_radius = 8
        d2.perception_noise = 0.0
        d2.softmax_tau = 0.0
    c.info_structure = d2
    return SphereEngine(c)


for label, asym in (("OFF", False), ("ON ", True)):
    print(f"\n=== {label}  seed=42 (食物存量空间分布) ===")
    e = build(asym)
    print(f"{'tick':>6}{'N':>6}{'E':>8}{'食物总':>10}{'极区食物':>10}{'赤道食物':>10}"
          f"{'种群极区%':>10}{'出生':>7}")
    for t in range(1, TICKS + 1):
        e.step()
        if t % SN == 0 or e.extinct:
            g = e.resources._grid
            tot = float(g.sum())
            pol = float(g[POLAR_CELLS].sum())
            equ = float(g[EQUAT_CELLS].sum())
            P = len(e._id)
            if P:
                r = e._flat // 120
                pf = float(np.isin(r, POLAR).mean())
                mE = float(e._energy[:P].mean())
            else:
                pf = mE = float("nan")
            print(f"{t:>6}{P:>6}{mE:>8.1f}{tot:>10.0f}{pol:>10.0f}{equ:>10.0f}"
                  f"{pf*100:>9.1f}%{e.total_born:>7}")
            if e.extinct:
                break
