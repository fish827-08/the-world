"""P2 端到端等价：同配置同种子，两个**全新**引擎分别用旧/新实现跑 N tick，末态逐位比对。"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "experiments")

import numpy as np  # noqa: E402
import p1_profile as P  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402

RUN_TO = int(sys.argv[1]) if len(sys.argv) > 1 else 2500


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


def _build():
    cfg, notes = P.make_cfg(42, 480, 960, 3000, 480, False, True,
                            0.125, 0.125, 80, 2.5, 30000)
    eng = SphereEngine(cfg)
    P.apply_post_build(eng, notes)
    return eng


def _digest(e):
    n = len(e._flat)
    return (n, hash(e._flat[:n].tobytes()), hash(e._energy[:n].tobytes()),
            hash(e._genes[:n].tobytes()), hash(e._interpret.tobytes()),
            hash(e._age[:n].tobytes()), hash(e._stomach[:n].tobytes()),
            hash(e._repro_cooldown[:n].tobytes()), hash(e._id[:n].tobytes()),
            hash(e._parent[:n].tobytes()))


_orig = SphereEngine._culture_learn_python
res = {}
for tag, impl in (("旧", _old_impl), ("新", _orig)):
    SphereEngine._culture_learn_python = impl
    eng = _build()
    t0 = time.perf_counter()
    for _ in range(RUN_TO):
        eng.step()
    dt = time.perf_counter() - t0
    res[tag] = _digest(eng)
    print(f"{tag}：{RUN_TO} tick 用时 {dt:.1f}s（{dt / RUN_TO * 1e3:.2f} ms/tick）"
          f"  末态 N={res[tag][0]}", flush=True)
SphereEngine._culture_learn_python = _orig
print("⇒ 末态摘要（N/flat/energy/genes/interpret/age/stomach/cooldown/id/parent）"
      f"逐位一致：{'✅' if res['旧'] == res['新'] else '❌ ' + str(res)}")
