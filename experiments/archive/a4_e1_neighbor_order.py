"""A4-E1：邻居取法矩阵（按 R4 manifest 真实口径 uniform + N0=200）。

修正 F3：既有归因探针误用 patchy/N0=500，与 R4 manifest（uniform/N0=200）不符。
本脚本按真实口径复跑，并回答："方向偏置 vs 邻居数"、"radius=6 是否静默 no-op"。

5 臂 × 3 seed × 1500 tick，落 `_rerun_logs/e1_neighbor_order.csv`。
臂：
  off_r8      radius=8（对称对照）
  buggy_r4    旧 `nb[:4]`（北+西偏置）
  vn_r4       修正 von Neumann（上/左/右/下）
  diag_r4     对角 4 邻（各向同性对照）
  r6_noop     radius=6（预期=静默 no-op，等价 r8）
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import InfoStructureConfig, SimConfig  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402
from world.sphere_world import SphereWorld  # noqa: E402

TICKS = 1500
SEEDS = (42, 43, 44)
OUT = ROOT / "_rerun_logs" / "e1_neighbor_order.csv"


class BuggyWorld(SphereWorld):
    def neighbors_von_neumann(self, flat):
        return np.asarray(self.neighbors(int(flat)))[:4]      # 旧行为：北+西


class DiagWorld(SphereWorld):
    def neighbors_von_neumann(self, flat):
        nb = self.neighbors(int(flat))
        if len(nb) == 8:
            return nb[[0, 2, 5, 7]]                            # 四个对角
        return nb


ARMS = {
    "off_r8":   (8, SphereWorld),
    "buggy_r4": (4, BuggyWorld),
    "vn_r4":    (4, SphereWorld),        # 修正后默认 = von Neumann
    "diag_r4":  (4, DiagWorld),
    "r6_noop":  (6, SphereWorld),
}


def build(radius: int, world_cls, seed: int) -> SphereEngine:
    c = SimConfig(seed=seed)
    c.simulation.ticks = TICKS
    c.simulation.use_sim_core = False
    c.population.initial_count = 200          # ← R4 manifest 真实口径
    c.population.max_count = 5000
    c.resources.distribution = "uniform"      # ← R4 manifest 真实口径
    d2 = InfoStructureConfig(enabled=True)
    d2.learning_bottleneck = True
    d2.learning_rate = 0.05
    d2.arbitrary_codebook = False
    d2.steels_alignment = True
    d2.perception_radius = radius
    if radius == 8:
        d2.perception_noise = 0.0
        d2.softmax_tau = 0.0
    c.info_structure = d2
    e = SphereEngine(c)
    if world_cls is not SphereWorld:
        e.world = world_cls(c.world.rows, c.world.cols)
        tbl = np.full((e.world.n_cells, max(8, e.world.cols)), -1, dtype=np.int64)
        for cc in range(e.world.n_cells):
            tbl[cc, :len(e.world.neighbors(cc))] = e.world.neighbors(cc)
        e._nb_table = tbl
    return e


rows = []
print(f"R4 真实口径：uniform, N0=200, use_sim_core=False, {TICKS} tick, seeds={SEEDS}")
print(f"{'arm':<10}{'radius':>7}{'seed':>6}{'N_end':>7}{'N_peak':>8}{'born':>7}"
      f"{'捕食':>7}{'饿死':>7}{'meanRow':>9}{'polar%':>8}{'g15':>7}")
for arm, (radius, wcls) in ARMS.items():
    for s in SEEDS:
        e = build(radius, wcls, s)
        peak = 0
        for _ in range(TICKS):
            e.step()
            peak = max(peak, len(e._id))
            if e.extinct:
                break
        P = len(e._id)
        dc = e.death_cause_totals()
        g = {str(k): int(v) for k, v in dc.items()}
        r = (e._flat[:P] // 120) if P else np.zeros(0, int)
        mrow = float(r.mean()) if P else float("nan")
        pf = float(((r <= 5) | (r >= 54)).mean()) if P else float("nan")
        g15 = float(e._genes[:P, 15].mean()) if P else float("nan")
        rows.append({"arm": arm, "radius": radius, "seed": s, "N_end": P, "N_peak": peak,
                     "born": int(e.total_born), "predation": g.get("DeathCause.PREDATION", 0),
                     "starvation": g.get("DeathCause.STARVATION", 0),
                     "mean_row": round(mrow, 3), "polar_frac": round(pf, 4),
                     "g15": round(g15, 4)})
        print(f"{arm:<10}{radius:>7}{s:>6}{P:>7}{peak:>8}{e.total_born:>7}"
              f"{g.get('DeathCause.PREDATION', 0):>7}{g.get('DeathCause.STARVATION', 0):>7}"
              f"{mrow:>9.2f}{pf*100:>7.1f}%{g15:>7.3f}")

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nwrote {OUT}")

print("\n--- 各臂 N_end 中位 ---")
for arm in ARMS:
    v = sorted(r["N_end"] for r in rows if r["arm"] == arm)
    print(f"  {arm:<10} {v[1]}  (全部 {v})")
