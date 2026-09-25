"""只读标定：扫 patch_regrowth_mult，找"个体驻留时摄入速率真的会下降"的档位。

目的（R204 §11.2 待办 1）：
  fish 提议①「让斑块越吃越少」＋提议③「动态期待（吃得越来越少 ⇒ 就走）」。
  但 ③ 需要一个**事实前提**：个体驻留在斑块上时，摄入速率会**真的随时间下降**。
  本脚本在 13.5 B 臂 + subpos（speed_max=1.0，觅食最活跃档）下扫几档再生倍率，
  量「驻留个体的摄入速率是否随时间下降」。

判据（预注册）：
  取**始终存活的个体**，按时间窗（500 tick 一窗）算人均摄入；
  看「后窗 / 首窗」的比值：≈1 ⇒ 不下降（动态期待无从谈起）；明显 <1 ⇒ 有信号可用。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from simulation.config import SimConfig          # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402

TICKS = 6000
WINDOW = 1000


def make_cfg(seed: int, regrowth: float):
    c = SimConfig(seed=seed)
    c.simulation.use_sim_core = False
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = True
    c.resources.patch_count = 30
    c.resources.patch_radius = 2
    c.resources.patch_regrowth_mult = float(regrowth)
    o = c.organisms
    o.max_energy = 600.0; o.initial_energy = 300.0
    o.starve_frac = 0.30; o.exhaust_frac = 0.17
    o.eat_efficiency = 7.5; o.assim_herb = 0.4; o.assim_carn = 0.8
    o.stomach_cap_mass = 25.0; o.eat_threshold_frac = 0.6; o.photo_max = 0.0
    c.population.max_count = 600
    c.subpos.enabled = True
    c.subpos.subdiv = 4
    c.subpos.speed_max = 1.0            # 觅食最活跃档（冒烟里消耗最多）
    return c


ARMS = [0.30, 0.60, 1.195, 3.0]
print(f"== 斑块耗尽标定：TICKS={TICKS}，窗={WINDOW}，subpos speed_max=1.0，seed=42 ==\n")

for reg in ARMS:
    e = SphereEngine(make_cfg(42, reg))
    g0 = np.asarray(e.resources._grid)
    cap0 = float(g0.max())
    # 🔴 类级补丁（ResourceField 是 __slots__ 类，实例不可赋值）：
    #   拦截 Python 路径进食调用，累计每 tick 真实咬到的质量（不改仓库代码）
    import world.resource_field as _rf
    _orig = _rf.ResourceField.consume_many
    acc = {"sum": 0.0}

    def wrapped(self, flat, want, _orig=_orig, acc=acc):
        taken = _orig(self, flat, want)
        acc["sum"] += float(np.sum(taken))
        return taken

    _rf.ResourceField.consume_many = wrapped
    try:
        win_sum = []
        prev = 0.0
        ns = []
        for t in range(1, TICKS + 1):
            e.step()
            if t % WINDOW == 0:
                cur = acc["sum"]
                win_sum.append(cur - prev)
                prev = cur
                ns.append(int(len(e._flat)))
    finally:
        _rf.ResourceField.consume_many = _orig   # 还原，避免污染后续臂
    per_ind = [s / max(1, ns[i]) for i, s in enumerate(win_sum)]
    ratio = per_ind[-1] / per_ind[0] if per_ind[0] > 1e-9 else float("nan")
    g_end = np.asarray(e.resources._grid)
    frac_low = float((g_end[g0 > 0] < 0.2 * cap0).mean()) if (g0 > 0).any() else 0.0
    print(f"regrowth={reg:<6} 每窗总摄入(质量): " +
          " → ".join(f"{x:.0f}" for x in win_sum))
    print(f"               每窗人均摄入: " + " → ".join(f"{x:.2f}" for x in per_ind) +
          f"  ⇒ 末窗/首窗 = {ratio:.2f}"
          f"｜末态低存量斑块格 {frac_low*100:.0f}%｜末 N={ns[-1]}")

print()
print("=== 判读 ===")
print("  末窗/首窗 ≈ 1 ⇒ 摄入不随驻留下降 ⇒ 动态期待无信号（必须再调）")
print("  末窗/首窗 明显 <1 ⇒ 有信号可用 ⇒ 该档位进入正式批")
