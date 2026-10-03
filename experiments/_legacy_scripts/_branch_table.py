# -*- coding: utf-8 -*-
"""真支别表：直接用引擎构造世界读 `ResourceDynamics.bg_production_zero`（不跑模拟，秒级）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from experiments.steady_k_probe import make_cfg, apply_post_build
from experiments.s3_memory_probe import S1_BASE
from simulation.sphere_engine import SphereEngine
from world.resource_dynamics import ResourceDynamics

SEEDS = [165, 167, 194, 195, 196, 197, 198, 199, 192]
PCS = [400, 1700, 4000]

print("%-6s" % "seed" + "".join("%-14s" % ("pc=%d" % p) for p in PCS))
for s in SEEDS:
    line = "%-6d" % s
    for pc in PCS:
        c, n = make_cfg(s, 480, 960, 10000, pc, True,
                        S1_BASE["speed_max"], S1_BASE["gain"], S1_BASE["subdiv"],
                        S1_BASE["k"], S1_BASE["max_count"], 1.195,
                        bg_low_prod_frac=0.4, bg_low_cap_mult=0.05)
        c.simulation.use_sim_core = False
        c.resource_dynamics.enabled = True
        e = SphereEngine(c)
        apply_post_build(e, n)
        rd = ResourceDynamics.from_field(e.config.resource_dynamics, e.resources, e.world)
        cap = float(np.asarray(e.resources._capacity).sum())
        line += "%-14s" % ("%s (%.2e)" % ("A" if rd.bg_production_zero else "B", cap))
    print(line)
