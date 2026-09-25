"""ARS 对照实验（v4 首轮）：ARS 关 vs 开，量「荒漠出生 Founder 的存活率」。

设计（预注册，照 v4 §5/§10）：
  * 主判据：创始代中出生格 food_ratio==0 的个体，在 t_read 仍存活的比例（ARS 开 vs 关）
  * 辅助：末 N、世代、ARS 读数（dec_n / 切换 / 赶路占比）
  * 臂：ARS 关 / ARS gain=1 / ARS gain=3（theta=0.5，kappa=0，giveup=20）
  * 世界：13.5 B 臂（patchy + bgzero，**不改世界底层规则**）；创始代均匀撒点（引擎默认）
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from simulation.config import SimConfig          # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402

T_READ = 1000
T_END = 3000
SEEDS = (42, 7, 11)


def make_cfg(seed: int, ars: bool, gain: float):
    c = SimConfig(seed=seed)
    c.simulation.use_sim_core = False
    c.simulation.l2_dash = False
    c.population.max_count = 600
    c.predation.forage_tradeoff_k = 0.0
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = True
    c.resources.patch_count = 30
    c.resources.patch_radius = 2
    c.resources.patch_regrowth_mult = 1.195
    o = c.organisms
    o.max_energy = 600.0; o.initial_energy = 300.0
    o.starve_frac = 0.30; o.exhaust_frac = 0.17
    o.eat_efficiency = 7.5; o.assim_herb = 0.4; o.assim_carn = 0.8
    o.stomach_cap_mass = 25.0; o.eat_threshold_frac = 0.6; o.photo_max = 0.0
    c.ars.enabled = bool(ars)
    c.ars.gain = float(gain)
    c.ars.theta = 0.5
    c.ars.giveup = 20
    c.ars.fast_tau = 50.0
    c.ars.slow_tau = 500.0
    return c


ARMS = [("ARS关", False, 0.0), ("ARS g1", True, 1.0), ("ARS g3", True, 3.0)]

print(f"== ARS 对照实验：t_read={T_READ}，t_end={T_END}，seeds={SEEDS} ==\n")
summary = {}
for name, ars, gain in ARMS:
    surv_reads = []
    for seed in SEEDS:
        e = SphereEngine(make_cfg(seed, ars, gain))
        grid0 = np.asarray(e.resources._grid)
        P0 = len(e._flat)
        ids0 = e._id[:P0].copy()
        desert0 = ids0[grid0[e._flat[:P0]] <= 0.0]     # 创始代荒漠出生
        n_hist = []
        for t in range(1, T_END + 1):
            e.step()
            if t in (T_READ, T_END):
                alive = np.isin(e._id[: len(e._flat)], desert0)
                surv_reads.append((t, float(alive.mean()),
                                   int(alive.sum()), int(len(desert0)),
                                   int(len(e._flat)), int(e._max_generation)))
        p = e.ars_probe()
        summary[(name, seed)] = surv_reads
        detail = " | ".join(
            f"t={t}: 存活率 {s*100:.1f}%({k}/{d}) N={n} 代={g}"
            for t, s, k, d, n, g in surv_reads)
        extra = (f"｜ARS: dec={p['ars_dec_n']} 赶路占比={p['ars_extensive_frac']}"
                 f" 切换={p['ars_sw_ie_n']}/{p['ars_sw_ei_n']}"
                 f" 失望重选={p['ars_rerand_n']}") if p else ""
        print(f"[{name}] seed={seed}  {detail}{extra}")

print("\n=== 汇总（t=1000 荒漠出生存活率，按 seed 分列）===")
for name, _, _ in ARMS:
    vals = [summary[(name, s)][0][1] for s in SEEDS]
    vals_str = " / ".join(f"{v*100:.0f}%" for v in vals)
    mean = float(np.mean(vals))
    print(f"  {name:<8}: {vals_str}  (均值 {mean*100:.1f}%)  "
          f"{'✅ ≥50%' if mean >= 0.5 else '（<50%）'}")
