# b_mult 抽奖的**现场后果**：t=1 回写把 field 容量改成 rd 模型口径
# 种子 = 本机批 169-172 + 已知 A 支 165；装置 = s3_memory_probe 旗舰档
import sys
from pathlib import Path

WT = Path(r"C:\Users\圣羽\Desktop\temp\tempCode\_wt_lq_rd")
sys.path.insert(0, str(WT))

import numpy as np  # noqa: E402

from experiments.steady_k_probe import make_cfg, apply_post_build  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402


def build(seed: int):
    c, notes = make_cfg(
        seed, 480, 960, 10000, 1700, True, 0.125, 0.125, 80, 2.5, 30000, 1.195,
        bg_low_prod_frac=0.4, bg_low_cap_mult=0.05,
    )
    c.simulation.use_sim_core = False
    c.resource_dynamics.enabled = True
    c.simulation.sparse_fields = False
    eng = SphereEngine(c)
    apply_post_build(eng, notes)
    return eng


SEEDS = tuple(int(s) for s in sys.argv[1:]) or (165, 169, 170, 171, 172)
for seed in SEEDS:
    e = build(seed)
    rd, rf = e._rd, e.resources
    cap0 = np.asarray(rf._capacity, dtype=np.float64).copy()
    base = rd._base_capacity
    bg = ~rd._mask
    first_bg = int(np.flatnonzero(bg)[0])
    b_mult = float(cap0[first_bg] / base[first_bg])
    low = np.asarray(rf._bg_low_mask, dtype=bool)
    n_prod0 = int((cap0 > 0).sum())
    s0 = float(cap0.sum())
    g0 = float(np.asarray(rf._grid, dtype=np.float64)[low].sum())

    e.step()
    g1 = float(np.asarray(rf._grid, dtype=np.float64)[low].sum())
    e.step()
    g2 = float(np.asarray(rf._grid, dtype=np.float64)[low].sum())

    cap1 = np.asarray(rf._capacity, dtype=np.float64)
    n_prod1 = int((cap1 > 0).sum())
    s1 = float(cap1.sum())
    low_alive = int(((cap1 > 0) & low).sum())
    # 原零产能格（沙漠）被抬起来的数
    desert_raised = int(((cap0 == 0) & (cap1 > 0)).sum())
    print(f"seed={seed:>3} b_mult={b_mult:<6.3f} rd.bgzero={str(rd.bg_production_zero):>5} "
          f"dead/rest={rd.dead_regen_mult}/{rd.rest_regen_mult} moves={rd.rotate_moves_patch}")
    print(f"        low带格={int(low.sum())} | t0 产能格={n_prod0} Σcap={s0:.3e} "
          f"Σgrid(low)={g0:.3e}")
    print(f"        t1 产能格={n_prod1} Σcap={s1:.3e} | low带残存产能={low_alive} "
          f"沙漠被抬起={desert_raised} | low带Σgrid: t1={g1:.3e} t2={g2:.3e}")
    del e
