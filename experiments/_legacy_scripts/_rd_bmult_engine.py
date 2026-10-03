# 真引擎确认：b_mult 抽奖 ⇒ rd 派生状态 + sparse 守卫行为（seed 165/167/170/172）
import sys
from pathlib import Path

WT = Path(r"C:\Users\圣羽\Desktop\temp\tempCode\_wt_lq_rd")
sys.path.insert(0, str(WT))

import numpy as np  # noqa: E402

from experiments.steady_k_probe import make_cfg, apply_post_build  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402


def build(seed: int, sparse: bool):
    c, notes = make_cfg(
        seed, 480, 960, 120, 1700, True, 0.125, 0.125, 80, 2.5, 30000, 1.195,
        bg_low_prod_frac=0.4, bg_low_cap_mult=0.05,
    )
    c.simulation.use_sim_core = False
    c.resource_dynamics.enabled = True
    c.simulation.sparse_fields = bool(sparse)
    return c, notes


for seed in (165, 167, 170, 172):
    c, notes = build(seed, sparse=False)
    try:
        eng = SphereEngine(c)
    except Exception as e:  # noqa: BLE001
        print(f"seed={seed} 构造异常: {type(e).__name__}: {str(e)[:120]}")
        continue
    apply_post_build(eng, notes)
    rd = eng._rd
    f = eng.resources
    cap0 = np.asarray(f._capacity, dtype=np.float64)
    cap_rd = rd.capacity_from_base()
    neq = int((cap0 != cap_rd).sum())
    print(f"seed={seed:>3} rd.bgzero={str(rd.bg_production_zero):>5} "
          f"dead_mult={rd.dead_regen_mult} rest_mult={rd.rest_regen_mult} "
          f"moves={rd.rotate_moves_patch} | cap 失配={neq} | sparse={eng._sparse_fields}")
    del eng

    # sparse 尝试
    c2, notes2 = build(seed, sparse=True)
    try:
        eng2 = SphereEngine(c2)
        apply_post_build(eng2, notes2)
        print(f"          sparse=True ⇒ 构造 OK（_sparse_fields={eng2._sparse_fields}）")
        del eng2
    except ValueError as e:
        print(f"          sparse=True ⇒ ⛔ValueError: {str(e)[:80]}…")
