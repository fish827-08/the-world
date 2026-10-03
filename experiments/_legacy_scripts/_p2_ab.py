"""P2 整 tick 交替 A/B（类级补丁；文化段不消费 RNG ⇒ 两变体轨迹应逐位同步）。"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "experiments")

import numpy as np  # noqa: E402

import p1_profile as P  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402

RUN_TO = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
cfg, notes = P.make_cfg(42, 480, 960, 3000, 480, False, True,
                        0.125, 0.125, 80, 2.5, 30000)
eng = SphereEngine(cfg)
P.apply_post_build(eng, notes)
for _ in range(RUN_TO):
    eng.step()
print(f"N={len(eng._flat)}（t={RUN_TO}，py+subpos / T4 档）", flush=True)


def _old_impl(self, j_idx, adult_qual):
    interp = self._interpret
    cell_adults: dict[int, list[int]] = {}
    for a in np.flatnonzero(adult_qual):
        cell_adults.setdefault(int(self._flat[int(a)]), []).append(int(a))
    for idx in j_idx:
        c = int(self._flat[int(idx)])
        lst: list[int] = []
        for nc in self.world.neighbors(c):
            v = cell_adults.get(int(nc))
            if v:
                lst.extend(v)
        if lst:
            lst = sorted(set(lst))
            interp[int(idx)] += 0.1 * (interp[lst].mean(axis=0) - interp[int(idx)])


def _digest(e):
    n = len(e._flat)
    return (n, hash(e._flat[:n].tobytes()), hash(e._energy[:n].tobytes()),
            hash(e._genes[:n].tobytes()), hash(e._interpret.tobytes()),
            hash(e._age[:n].tobytes()), hash(e._repro_cooldown[:n].tobytes()))


_orig = SphereEngine._culture_learn_python

# ① 计时：旧 10 tick / 新 10 tick 交替 3 轮（同一实例，仅比速度）
res: dict[str, list[float]] = {"old": [], "new": []}
for _ in range(3):
    for tag, impl in (("old", _old_impl), ("new", _orig)):
        SphereEngine._culture_learn_python = impl
        t0 = time.perf_counter()
        for _ in range(10):
            eng.step()
        res[tag].append((time.perf_counter() - t0) / 10 * 1e3)
SphereEngine._culture_learn_python = _orig
mo, mn = float(np.median(res["old"])), float(np.median(res["new"]))
print(f"旧: {[round(x, 2) for x in res['old']]}  中位 {mo:.2f} ms/tick")
print(f"新: {[round(x, 2) for x in res['new']]}  中位 {mn:.2f} ms/tick")
print(f"⇒ 整 tick {mo / mn:.3f}×；省 {mo - mn:.2f} ms/tick（N={len(eng._flat)}）")

# ② 轨迹一致：同一快照分别跑 10 tick（旧 vs 新），末态摘要必须逐位一致
snap = "_trash_local/_p2_snap.npz"
eng.save_snapshot(snap)
t_snap = len(eng._flat)
SphereEngine._culture_learn_python = _old_impl
for _ in range(10):
    eng.step()
d_old = _digest(eng)
SphereEngine._culture_learn_python = _orig
eng2 = SphereEngine.load_snapshot(snap)
assert len(eng2._flat) == t_snap
for _ in range(10):
    eng2.step()
d_new = _digest(eng2)
print(f"② 快照后 10 tick（旧 vs 新）末态摘要："
      f"{'逐位一致 ✅' if d_old == d_new else '有差异 ❌'}  N {t_snap} → {len(eng._flat)}")
if d_old != d_new:
    print("   旧:", d_old)
    print("   新:", d_new)
