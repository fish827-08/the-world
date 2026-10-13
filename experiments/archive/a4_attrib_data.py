"""A4 溯源数据采集（本地开发）— 为《崩溃溯源报告》提供 [实测] 数据。

产出：
 A) 邻居方向审计：证明 nb[:4] = {上左,上,上右,左}（只有北/西）
 B) 多 seed（42–46）ON vs OFF 对照：N 曲线 / 首次拐点 / 分死因 / 能量收支 / 基因漂移
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

TICKS = 2500
SEEDS = (42, 43, 44, 45, 46)
GENES = {0: "g0_move", 3: "g3_life", 10: "g10_forage", 14: "g14_perc",
         15: "g15_sig", 16: "g16_aggr"}

print("=" * 86)
print("A) 邻居方向审计（sphere_world.neighbors 的顺序 vs nb[:4] 的偏向）")
print("=" * 86)
w = SphereWorld(60, 120)
rows = np.arange(w.n_cells) // w.cols
cols = np.arange(w.n_cells) % w.cols
normal = np.flatnonzero((rows >= 10) & (rows <= 49))
sample = normal[::997][:12]
print(f"{'cell':>6}{'row':>5}{'col':>5} | nb[0..7] (drow,dcol) | nb[:4] 方向")
print("-" * 86)
north_west = 0
for c in sample:
    nb = w.neighbors(int(c))
    r0, c0 = int(rows[c]), int(cols[c])
    ds = [(int(rows[n]) - r0, int(cols[n]) - c0) for n in nb]
    first4 = ds[:4]
    all_nw = all(dr <= 0 for dr, dc in first4)  # 都不向南
    north_west += all_nw
    print(f"{c:>6}{r0:>5}{c0:>5} | {ds} | {first4}  {'全部非南向' if all_nw else '含南向'}")
print(f"\n  ⇒ 抽样 {len(sample)} 格里 {north_west}/{len(sample)} 的 nb[:4] 【全部为北向或同排】")
print("  ⇒ nb[:4] = {上左, 上, 上右, 左}：个体永远看不到 右/下左/下/下右 四个邻居")
print("  ⇒ 结论：radius=4 实际=单向(北+西)偏置，而非文档所称 Von Neumann（上下左右）")

print()
print("=" * 86)
print(f"B) 多 seed ON vs OFF 对照（N0=500, patchy, {TICKS} tick, use_sim_core=False）")
print("=" * 86)


def build(asym: bool, seed: int) -> SphereEngine:
    c = SimConfig(seed=seed)
    c.simulation.ticks = TICKS
    c.simulation.use_sim_core = False
    c.population.initial_count = 500
    c.population.max_count = 5000
    c.resources.distribution = "patchy"
    d2 = InfoStructureConfig(enabled=True)
    d2.arbitrary_codebook = False        # 与 r4 批 manifest 一致
    d2.learning_rate = 0.05
    if not asym:
        d2.perception_radius = 8
        d2.perception_noise = 0.0
        d2.softmax_tau = 0.0
    c.info_structure = d2
    return SphereEngine(c)


for label, asym in (("OFF (radius=8)", False), ("ON  (radius=4, nb[:4])", True)):
    print(f"\n--- {label} ---")
    G = {k: [] for k in GENES}
    last_N, peaks, t_peak = [], [], []
    for s in SEEDS:
        e = build(asym, s)
        traj = []
        for t in range(1, TICKS + 1):
            e.step()
            if t in (100, 500, 1000, 1500, 2000, TICKS) or e.extinct:
                traj.append((t, len(e._id)))
                if e.extinct:
                    break
        P = len(e._id)
        last_N.append(P)
        ns = [n for _, n in traj]
        peaks.append(max(ns))
        t_peak.append(traj[ns.index(max(ns))][0])
        if P:
            for k in GENES:
                G[k].append(float(e._genes[:P, k].mean()))
        dc = e.death_cause_totals()
        dsum = sum(dc.values()) or 1
        print(f"  seed={s}  N_end={P:>5}  N_peak={max(ns):>5}@t={traj[ns.index(max(ns))][0]}  "
              f"born={e.total_born:>5}  died={e.total_died:>5}  "
              f"饿{dict(dc).get('STARVATION', 0)}/捕{dict(dc).get('PREDATION', 0)}/老{dict(dc).get('OLD_AGE', 0)}")
        print(f"         轨迹 {traj}")
    print(f"  ⇒ N 终局 中位={sorted(last_N)[len(last_N)//2]}  峰值中位={sorted(peaks)[len(peaks)//2]}  "
          f"灭绝 {sum(1 for n in last_N if n == 0)}/{len(SEEDS)}")
    for k, name in GENES.items():
        v = G[k]
        if v:
            print(f"      {name:<11} 终局均值={np.mean(v):.3f}±{np.std(v):.3f}")
