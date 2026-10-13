"""A4 崩溃溯源探针（本地开发）— 验证"感知半径4 = nb[:4] 造成方向偏置"假设。

不修改引擎：用子类 SphereWorld 重排邻居顺序来模拟"修正后的 Von Neumann 4 邻"，
以此在零引擎改动下对比：
  ON   : 现状（radius=4，nb[:4] → 北+西偏置）
  OFF  : 对称（radius=8，全 8 邻）
  FIXED: 把前 4 个邻居换成 正交 4 邻（上/下/左/右）→ 验证 N 是否恢复

观测：N、平均纬度行、极区占比、累计死因、平均能量/胃、累计出生。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import InfoStructureConfig, SimConfig  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402
from world.sphere_world import SphereWorld  # noqa: E402

TICKS = 4000
SN = 100  # 采样间隔


class AxisFirstWorld(SphereWorld):
    """把邻居顺序改为 正交优先：[上,左,右,下, 上左,上右,下左,下右]。

    这样引擎里 `nb[:4]` 恰好取到 {上,左,右,下} = 真 Von Neumann 4 邻，
    消除"只会向北/向西"的方向偏置。零引擎改动即可验证假设。
    """

    def neighbors(self, flat: int):
        nb = super().neighbors(flat)
        if len(nb) != 8:
            return nb  # 极点格（cols 个）不重排
        order = [1, 3, 4, 6, 0, 2, 5, 7]
        return nb[order]


def build(asym: bool, world_cls=SphereWorld, seed: int = 42) -> SphereEngine:
    c = SimConfig(seed=seed)
    c.simulation.ticks = TICKS
    c.simulation.use_sim_core = False           # D2 走 Python 路径
    c.population.initial_count = 500
    c.population.max_count = 5000
    c.resources.distribution = "patchy"
    d2 = InfoStructureConfig(enabled=True)
    d2.learning_bottleneck = True
    d2.learning_rate = 0.05
    d2.arbitrary_codebook = False               # 与 r4 批 manifest 一致
    d2.steels_alignment = True
    if not asym:
        d2.perception_radius = 8
        d2.perception_noise = 0.0
        d2.softmax_tau = 0.0
    c.info_structure = d2
    e = SphereEngine(c)
    e.world = world_cls(c.world.rows, c.world.cols)   # 替换世界（重排邻居）
    # 邻居表也要重建（引擎 __init__ 用的是旧 world 的表）
    n_cells = e.world.n_cells
    nb_stride = max(8, e.world.cols)
    tbl = np.full((n_cells, nb_stride), -1, dtype=np.int64)
    for cc in range(n_cells):
        nbs = e.world.neighbors(cc)
        tbl[cc, :len(nbs)] = nbs
    e._nb_table = tbl
    return e


def run(label: str, asym: bool, world_cls=SphereWorld, seed: int = 42):
    e = build(asym, world_cls, seed)
    rows = e.world.rows
    samples = []
    for t in range(1, TICKS + 1):
        e.step()
        if t % SN == 0 or e.extinct:
            P = len(e._id)
            if P:
                r = e._flat // e.world.cols
                mean_row = float(r.mean())
                polar = float(((r <= 5) | (r >= rows - 6)).mean())
                mean_e = float(e._energy[:P].mean())
                mean_s = float(e._stomach[:P].mean())
            else:
                mean_row = polar = mean_e = mean_s = float("nan")
            samples.append((t, P, mean_row, polar, mean_e, mean_s))
            if e.extinct:
                break
    dc = e.death_cause_totals()
    print(f"\n=== {label} (seed={seed}) ===")
    print(f"{'tick':>7}{'N':>6}{'meanRow':>9}{'polar%':>8}{'E':>8}{'stom':>7}")
    for t, P, mr, po, me, ms in samples:
        print(f"{t:>7}{P:>6}{mr:>9.2f}{po*100:>7.1f}%{me:>8.2f}{ms:>7.2f}")
    print(f"  deaths total={e.total_died} born={e.total_born}")
    for k, v in sorted(dc.items(), key=lambda kv: -kv[1]):
        print(f"    {k}: {v}")


if __name__ == "__main__":
    print(f"TICKS={TICKS}, sample every {SN}, N0=500, patchy, use_sim_core=False")
    run("OFF 对称 (radius=8)", asym=False)
    run("ON  现状 (radius=4, nb[:4])", asym=True)
    run("FIXED 正交4邻 (radius=4, axis-first)", asym=True, world_cls=AxisFirstWorld)
