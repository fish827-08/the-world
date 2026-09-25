"""只读冒烟：用**现有** subpos 量「减速」的后果（不修改任何引擎代码）。

目的（R204 结论的前提检验）：
  P-A  减速会不会把种群打崩（随机走有效半径 ~2.5 格 < 最近斑块 7.7 格 ⇒ 预言会崩）
  P-B  减速后「每代多少 tick」（决定机时是否可接受）
  P-C  斑块存量随时间是否下降（决定「动态期待」有没有信号可用）
  P-D  满能力速度上限取 0.11–0.15 是否合理
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from simulation.config import SimConfig          # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402

TICKS = 4000
SEEDS = (42, 7)
SAMPLE = (0, 200, 500, 1000, 2000, 3000, 4000)


def make_cfg(seed: int, subpos: bool, speed_max: float | None = None,
             speed_gain: float | None = None):
    c = SimConfig(seed=seed)
    c.simulation.use_sim_core = False          # 新机制强制 Python 路径
    # —— 13.5 B 臂（食物绑定 + 能量标定）——
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
    o.assim_carn = 0.8
    o.stomach_cap_mass = 25.0
    o.eat_threshold_frac = 0.6
    o.photo_max = 0.0
    c.population.max_count = 600
    # —— subpos（已实现的减速旋钮）——
    if subpos:
        c.subpos.enabled = True
        c.subpos.subdiv = 4
        if speed_max is not None:
            c.subpos.speed_max = float(speed_max)
        if speed_gain is not None:
            c.subpos.speed_gain = float(speed_gain)
    return c


ARMS = [
    ("基线 subpos 关", dict(subpos=False)),
    ("减速 speed_max=1.0", dict(subpos=True, speed_max=1.0)),
    ("减速 speed_max=0.25", dict(subpos=True, speed_max=0.25)),
    ("减速 speed_max=0.15", dict(subpos=True, speed_max=0.15)),
]

print(f"== 冒烟：TICKS={TICKS}，seeds={SEEDS}，采样点={SAMPLE} ==\n")

results = {}
for name, kw in ARMS:
    for seed in SEEDS:
        t0 = time.time()
        try:
            e = SphereEngine(make_cfg(seed, **kw))
        except Exception as ex:
            print(f"[{name}] seed={seed} 构造失败: {type(ex).__name__}: {ex}")
            continue
        rows = []
        grid0 = float(np.asarray(e.resources._grid).sum())
        peak0 = float(np.asarray(e.resources._grid).max())
        for t in range(TICKS + 1):
            if t in SAMPLE and t > 0:
                g = np.asarray(e.resources._grid)
                rows.append((t, int(len(e._flat)),
                             float(g.sum()), float(g.max()),
                             int(e._max_generation)))
            e.step()
        # 补充 t=0 一行
        rows.insert(0, (0, -1, grid0, peak0, 0))
        wall = time.time() - t0
        results[(name, seed)] = (rows, wall)
        line = " | ".join(f"t={t}:N={n}" for t, n, _, _, _ in rows[1:])
        gens = " → ".join(f"{ge}" for _, _, _, _, ge in rows[1:])
        print(f"[{name}] seed={seed}  墙钟 {wall:.1f}s")
        print(f"   存活: {line}")
        print(f"   世代: {gens}")
        print(f"   食物总量: " + " → ".join(f"{s:.0f}" for _, _, s, _, _ in rows))
        print(f"   最富格存量: " + " → ".join(f"{p:.2f}" for _, _, _, p, _ in rows))
        print()

print("=== 判读辅助 ===")
for (name, seed), (rows, wall) in results.items():
    if seed != 42:
        continue
    print(f"  {name:<22} 末存活 N={rows[-1][1]}  末世代={rows[-1][4]}")
