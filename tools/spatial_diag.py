"""13.5 空间诊断（R191）—— 「斑块有没有被吃？生物在哪？有没有赖着不走？」

    python.exe tools/spatial_diag.py

跑 13.5 的 C 臂配置（只食物绑定），每 1000 tick 采样引擎内部状态：
  斑块存量/容量 ｜ 生物落在斑块格比例 ｜ **被访问过的斑块格比例** ｜ 吃空斑块比例 ｜ 同格最多人

R191 实测（seed 42 / 11，6000 tick）：
  · **斑块存量/容量 = 0.94–1.00** ⇒ 斑块几乎是满的 ⇒ **食物供过于求，不是"太少"**
  · **生物 93–100% 都在斑块格上**（斑块仅占 11.3% 的格）⇒ **找到就不走**（聚集）
  · **31–47% 的斑块格从未被访问过**（6000 tick 内）；访问率缓慢上升（44.7%→53.4%）
    ⇒ 靠随机游走"撞上"，发现率低
⇒ 两个机制并存：① **发现率低**（感知 1 格 + 随机游走）② **发现后不走**（食物取之不尽）
⇒ 前者由 `perception_span=2` 直接对治；后者需**斑块被吃空**才会自然发生（现人口太低，吃空率 0%）
"""


os.chdir(r"C:\Users\圣羽\Desktop\temp\tempCode\the-world")
sys.path.insert(0, os.getcwd())

import numpy as np
from simulation.config import InfoStructureConfig, SimConfig
from simulation.sphere_engine import SphereEngine


def build(seed: int) -> SimConfig:
    c = SimConfig(seed=seed)
    c.simulation.use_sim_core = False
    c.population.max_count = 3240
    c.population.soft_cap_target = 0.6
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = True          # 13.5 ①
    c.resources.patch_regrowth_mult = 1.195        # 13.5 ②
    c.predation.forage_tradeoff_k = 0.0
    c.info_structure = InfoStructureConfig(enabled=True, learning_rate=0.05,
                                           memory_gradient="none")
    c.organisms.energy_cap_enabled = True
    c.organisms.photo_max = 0.0
    c.corpse_wound = type(c.corpse_wound)()        # 默认（全关）——与 C 臂一致
    return c


def run(seed: int, ticks: int = 6000, every: int = 1000) -> None:
    e = SphereEngine(build(seed))
    n = e.world.n_cells
    patch = np.asarray(e.resources._patch_mask, dtype=bool)
    cap = np.asarray(e.resources._capacity, dtype=np.float64)
    cap_patch = float(cap[patch].sum())
    area_patch = int(patch.sum())
    visited = np.zeros(n, dtype=bool)
    print("=" * 104)
    print("seed=%d ｜ 斑块格 %d/%d（%.1f%%）｜ 斑块总容量 %.0f ｜ 名义再生 %.1f 质量/tick"
          % (seed, area_patch, n, 100.0 * area_patch / n, cap_patch,
             float(e.resources._regrowth_amount(0).sum())))
    print("%-7s %-7s %-13s %-13s %-13s %-13s %-11s" % (
        "tick", "N", "斑块存量/容量", "生物在斑块%", "已访问斑块%", "吃空斑块%", "同格最多人"))
    for k in range(ticks):
        if e.extinct:
            print("  灭绝 @tick=%d" % k)
            break
        e.step()
        P = len(e._flat)
        if P:
            visited[e._flat[:P]] = True
        if (k + 1) % every:
            continue
        stock = np.asarray(e.resources._grid, dtype=np.float64)
        st_p = float(stock[patch].sum()) / max(1e-9, cap_patch)
        on_patch = float(np.mean(patch[e._flat[:P]])) if P else 0.0
        vis_p = float(visited[patch].sum()) / max(1, area_patch)
        empty = float(np.mean(stock[patch] < 0.05 * np.maximum(cap[patch], 1e-9)))
        if P:
            cnt = np.bincount(e._flat[:P], minlength=n)
            mx, av = int(cnt.max()), float(P) / max(1, int((cnt > 0).sum()))
        else:
            mx, av = 0, 0.0
        print("%-7d %-7d %-13.3f %-13.1f %-13.1f %-13.1f %-11s" % (
            k + 1, P, st_p, 100 * on_patch, 100 * vis_p, 100 * empty,
            "%d(均%.2f)" % (mx, av)))
    print()


for s in (42, 11):
    run(s)
