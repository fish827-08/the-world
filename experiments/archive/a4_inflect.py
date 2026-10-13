"""A4 细粒度拐点：ON vs OFF 前 240 tick，每 20 tick 采样。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import InfoStructureConfig, SimConfig  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402


def build(asym: bool, seed: int = 42) -> SphereEngine:
    c = SimConfig(seed=seed)
    c.simulation.ticks = 240
    c.simulation.use_sim_core = False
    c.population.initial_count = 500
    c.population.max_count = 5000
    c.resources.distribution = "patchy"
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
    for seed in (42, 43, 44, 45, 46):
        e = build(asym, seed)
        seq = []
        for t in range(1, 241):
            e.step()
            if t % 20 == 0:
                seq.append(len(e._id))
            if e.extinct:
                break
        peak = max(seq)
        tpk = (seq.index(peak) + 1) * 20
        print(f"{label} seed={seed}  " + " ".join(f"{n:>4}" for n in seq) +
              f"   peak={peak}@t≈{tpk}")
