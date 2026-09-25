"""只读几何探测：在**真实斑块布局**上模拟"直线搜索"，回答三件事（不改任何文件/引擎）。

1. 走多少格会"失望"该转向？   ⇒ 实测世界的**平均自由程 λ**（直线上相邻两次撞到斑块格的期望距离）
2. 根据什么转向？            ⇒ 对比几种转向规则的撞斑率
3. fish 的验收线「≥60% 素食者找到食物不饿死」在什么预算下可达？
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from simulation.config import SimConfig          # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402

# —— 13.5 B 臂同参（与冒烟一致）——
c = SimConfig(seed=42)
c.simulation.use_sim_core = False
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
c.population.max_count = 600

e = SphereEngine(c)
w = e.world
grid = np.asarray(e.resources._grid)
patch = grid > 0.0                       # t=0 所有斑块格都有存量（initial_fill=0.5）
n_patch_cells = int(patch.sum())
rho = n_patch_cells / w.n_cells
lam = 1.0 / rho                          # 平均自由程（1 格宽的直线上撞斑率的倒数）

print(f"=== 世界 ===")
print(f"网格 {w.rows}×{w.cols}={w.n_cells}｜斑块格 {n_patch_cells}（{rho*100:.1f}%）"
      f"｜**平均自由程 λ ≈ {lam:.1f} 格**\n")

# —— 实测 λ（射线法，交叉验证）——
rng = np.random.default_rng(7)
dists = []
for _ in range(4000):
    start = int(rng.integers(0, w.n_cells))
    if patch[start]:
        continue
    nb = w.neighbors(start)
    cur = int(nb[rng.integers(0, len(nb))])
    for d in range(1, 80):
        if patch[cur]:
            dists.append(d)
            break
        nb = w.neighbors(cur)
        cur = int(nb[rng.integers(0, len(nb))])
d_arr = np.array(dists)
print(f"=== 实测：从随机荒漠格走直线，到第一次撞到斑块格的距离 ===")
print(f"  样本 {len(d_arr)}｜中位 {np.median(d_arr):.0f} 格｜均值 {d_arr.mean():.1f} 格"
      f"｜P90 {np.percentile(d_arr,90):.0f} 格")
for k in (1, 2, 3, 4, 5):
    print(f"  走 {k}λ = {k*lam:.1f} 格 ⇒ 累计撞到概率 {(d_arr<=k*lam).mean()*100:.1f}%")
print()

# —— 搜索策略对比：在能量预算 B 格内，多少荒漠出生者能撞到斑块？——
def hit_prob(B: int, restart_every: int | None, trials: int = 3000,
             exclude_back: bool = True) -> float:
    """直线搜索：每走 restart_every 格没撞到就重选方向（None=永不转向，纯直线）。"""
    hit = 0
    for _ in range(trials):
        start = int(rng.integers(0, w.n_cells))
        if patch[start]:
            hit += 1
            continue
        nb = w.neighbors(start)
        prev_dir = -1
        cur = int(nb[rng.integers(0, len(nb))])
        dir_idx = None
        d = 1
        ok = patch[cur]
        while not ok and d < B:
            if restart_every and d % restart_every == 0:
                nb = w.neighbors(cur)
                cands = list(range(len(nb)))
                if exclude_back and prev_dir >= 0 and len(nb) == 8:
                    back = (prev_dir + 4) % 8     # 8 邻列序里 +4 = 正对向
                    if back in cands and len(cands) > 1:
                        cands.remove(back)
                dir_idx = int(rng.choice(cands))
            nb = w.neighbors(cur)
            if dir_idx is None or len(nb) <= (dir_idx or 0):
                dir_idx = int(rng.integers(0, len(nb)))
            nxt = int(nb[dir_idx])
            # 记录本步方向（用"从 cur 看 nxt"在 cur 的邻居表里的下标）
            nb_cur = w.neighbors(cur)
            prev_dir = int(np.flatnonzero(nb_cur == nxt)[0]) if len(nb_cur) == 8 else -1
            cur = nxt
            d += 1
            ok = patch[cur]
        hit += int(ok)
    return hit / trials


print("=== 策略对比：预算 B 格内撞到斑块的概率（荒漠出生者）===")
print(f"{'预算B(格)':>10}{'纯直线':>10}{'每16格重启':>12}{'每26格(3λ)重启':>16}{'每8格(1λ)重启':>16}")
for B in (8, 16, 26, 40, 60, 104):
    p0 = hit_prob(B, None, trials=1500)
    p1 = hit_prob(B, 16, trials=1500)
    p3 = hit_prob(B, int(round(3 * lam)), trials=1500)
    p8 = hit_prob(B, int(round(lam)), trials=1500)
    print(f"{B:>10}{p0*100:>9.1f}%{p1*100:>11.1f}%{p3*100:>15.1f}%{p8*100:>15.1f}%")

print()
print("=== 对照 fish 的验收线「≥60% 素食者找到食物」===")
for B in (16, 26, 40):
    best = max(hit_prob(B, k, trials=1500) for k in
               (None, 16, int(round(3 * lam)), int(round(lam))))
    print(f"  预算 {B:>3} 格 ⇒ 最优策略撞斑率 {best*100:.1f}%  "
          f"{'✅ 达标' if best>=0.60 else '❌ 不足'}")
