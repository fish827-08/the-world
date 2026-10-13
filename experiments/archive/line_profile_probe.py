"""逐行性能剖析 `_step_population`（480×960）—— 找出"每格 0.17 μs"到底花在哪一行。

用法（必须用 venv 解释器）：
    .venv\\Scripts\\python.exe _trash_local/prof_steppop.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from line_profiler import LineProfiler  # noqa: E402

from simulation.config import SimConfig  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402

ROWS, COLS, POP, TICKS = 480, 960, 200, 12


def make_cfg():
    c = SimConfig(seed=42)
    c.world.rows, c.world.cols = ROWS, COLS
    c.simulation.use_sim_core = False
    c.simulation.l2_dash = False
    c.population.initial_count = POP
    c.population.max_count = 20000
    c.predation.enabled = False
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = True
    c.resources.patch_count = 30
    c.resources.patch_radius = 2
    c.resources.patch_regrowth_mult = 1.195
    o = c.organisms
    o.max_energy = 600.0
    o.initial_energy = 300.0
    o.starve_frac = 0.30
    o.exhaust_frac = 0.17
    o.eat_efficiency = 7.5
    o.assim_herb = 0.4
    return c


e = SphereEngine(make_cfg())
# 基线（无 profiler）
t0 = time.perf_counter()
for _ in range(TICKS):
    e.step()
base = (time.perf_counter() - t0) / TICKS * 1e3
print(f"基线（无 profiler）：{base:.2f} ms/tick @ {ROWS}x{COLS}, {POP} 个体\n")

e2 = SphereEngine(make_cfg())
lp = LineProfiler()
lp.add_function(SphereEngine._step_population)
lp.add_function(SphereEngine.step)
lp.enable()
for _ in range(TICKS):
    e2.step()
lp.disable()
print(f"=== `_step_population` 逐行（{TICKS} tick；数值已除以 tick 数 = 每 tick μs）===")
lp.print_stats()
