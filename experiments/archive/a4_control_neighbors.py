"""A4 控制实验：是"方向偏置"还是"只看4个邻居"导致崩溃？

三组对照（均 radius=4 语义、只改 nb[:4] 取到哪 4 个）：
  axis4 : {上,左,右,下}   —— 各向同性（修正目标）
  diag4 : {上左,上右,下左,下右} —— 也是各向同性，但取对角
  buggy : {上左,上,上右,左} —— 现状（只有北/西）
若 axis4 与 diag4 都能撑住、buggy 崩塌 → 病因是【方向偏置】，不是【邻居数=4】。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print 会让脚本 rc=1 假失败；F-R15 族）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # 非 TTY / 旧解释器：不因诊断能力缺失而阻断运行

from simulation.config import InfoStructureConfig, SimConfig  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402
from world.sphere_world import SphereWorld  # noqa: E402

TICKS = 3000


class ReorderWorld(SphereWorld):
    def __init__(self, rows, cols, order):
        self._order = order
        super().__init__(rows, cols)

    def neighbors(self, flat):
        nb = super().neighbors(flat)
        if len(nb) == 8:
            return nb[self._order]
        return nb


def build(order, seed=42):
    c = SimConfig(seed=seed)
    c.simulation.ticks = TICKS
    c.simulation.use_sim_core = False
    c.population.initial_count = 500
    c.population.max_count = 5000
    c.resources.distribution = "patchy"
    d2 = InfoStructureConfig(enabled=True)
    d2.arbitrary_codebook = False
    d2.learning_rate = 0.05
    c.info_structure = d2
    e = SphereEngine(c)
    if order is not None:
        e.world = ReorderWorld(c.world.rows, c.world.cols, order)
        tbl = np.full((e.world.n_cells, max(8, e.world.cols)), -1, dtype=np.int64)
        for cc in range(e.world.n_cells):
            tbl[cc, :len(e.world.neighbors(cc))] = e.world.neighbors(cc)
        e._nb_table = tbl
    return e


# Moore 列序: [0]上左 [1]上 [2]上右 [3]左 [4]右 [5]下左 [6]下 [7]下右
CASES = [
    ("buggy {上左,上,上右,左}", [0, 1, 2, 3]),
    ("axis4 {上,左,右,下}  ", [1, 3, 4, 6]),
    ("diag4 {四个对角}      ", [0, 2, 5, 7]),
]
print(f"radius=4 语义、仅改前4邻居的取法；N0=500, patchy, {TICKS} tick")
print(f"{'case':<24}{'seed42':>9}{'seed43':>9}{'seed44':>9}")
for name, order in CASES:
    vals = []
    for s in (42, 43, 44):
        e = build(order, seed=s)
        for _ in range(TICKS):
            e.step()
            if e.extinct:
                break
        vals.append(len(e._id))
    print(f"{name:<24}{vals[0]:>9}{vals[1]:>9}{vals[2]:>9}")
print("\n读法：buggy 崩、axis4/diag4 撑住 ⇒ 病因=方向偏置（非邻居数）")
