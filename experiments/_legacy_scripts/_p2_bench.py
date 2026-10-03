"""P2 文化学习向量化 A/B（一次性探针，不入库）：
   ① 全程 N 轨迹与 P1 基线（旧代码产出）逐点比对 —— 端到端未漂移的证据；
   ② 同状态块级 A/B：旧逐个体循环 vs 新批式实现（同进程背靠背，中位数）；
   ③ 稳态整 tick 计时（与 P1 CSV 同 tick 切片对照）。
用法：python _trash_local/_p2_bench.py [run_to]
"""
import csv
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "experiments")
sys.path.insert(0, "tests")

import numpy as np  # noqa: E402

import p1_profile as P  # noqa: E402
from simulation.genes import Gene  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402
from test_p2_culture_vec import _ref_culture_learn  # noqa: E402

RUN_TO = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
SLICE = 250

cfg, notes = P.make_cfg(42, 480, 960, 3000, 480, False, True,
                        0.125, 0.125, 80, 2.5, 30000)
t0 = time.time()
eng = SphereEngine(cfg)
P.apply_post_build(eng, notes)
print(f"构造 {time.time() - t0:.1f}s；py+subpos / T4 档，跑到 t={RUN_TO}", flush=True)

base = {}
with open("results/p1_profile/py_subpos.csv", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        base[int(row["t"])] = (int(row["N"]), float(row["ms_per_tick"]))

t, bad = 0, 0
while t < RUN_TO:
    s0 = time.perf_counter()
    for _ in range(SLICE):
        eng.step()
    dt = (time.perf_counter() - s0) / SLICE * 1e3
    t += SLICE
    N = len(eng._flat)
    ref = base.get(t)
    tag = ""
    if ref is not None:
        ok = ref[0] == N
        bad += (not ok)
        tag = (f"  基线 N={ref[0]} ✅" if ok
               else f"  基线 N={ref[0]} ❌ 漂移")
        tag += f"（基线 {ref[1]:.2f} ms/t）"
    print(f"  t={t:5d}  N={N:6d}  {dt:7.2f} ms/tick{tag}", flush=True)

print(f"\n① 轨迹一致性：比对 {sum(1 for k in base if k <= RUN_TO)} 点，漂移 {bad} 点")

N = len(eng._flat)
ocfg = eng.config.organisms
age_f = eng._age[:N].astype(np.float64)
maturity_age = ocfg.maturity_fraction * eng._lifespan(eng._genes[:, Gene.LIFE_GENE])
adult_qual = age_f >= maturity_age
j_idx = np.flatnonzero(~adult_qual)
print(f"\n② 块级 A/B（N={N}，成年={int(adult_qual.sum())} 未成年={j_idx.size}）")


def timed(fn, reps=25):
    store = eng._interpret[j_idx].copy()
    ts = []
    for _ in range(reps):
        t1 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t1)
        eng._interpret[j_idx] = store
    return float(np.median(ts) * 1e3)


old_ms = timed(lambda: _ref_culture_learn(eng, j_idx, adult_qual))
new_ms = timed(lambda: eng._culture_learn_python(j_idx, adult_qual))
print(f"  旧（逐个体循环）  : {old_ms:7.3f} ms/tick")
print(f"  新（批式向量化）  : {new_ms:7.3f} ms/tick"
      f"  ⇒ 提速 {old_ms / new_ms:.1f}×，省 {old_ms - new_ms:.3f} ms/tick")

t0 = time.perf_counter()
for _ in range(20):
    eng.step()
print(f"\n③ 稳态整 tick（20 tick 均值）：{(time.perf_counter() - t0) / 20 * 1e3:.2f} ms/tick"
      f"（N={len(eng._flat)}）")

# ④ 整 tick 交替 A/B：文化段不消费 RNG ⇒ 两变体状态轨迹应逐位同步
import types  # noqa: E402


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
            hash(e._age[:n].tobytes()))


new_impl = eng._culture_learn_python
res: dict[str, list[float]] = {"old": [], "new": []}
digs = []
for _ in range(3):
    for tag, impl in (("old", types.MethodType(_old_impl, eng)), ("new", new_impl)):
        eng._culture_learn_python = impl
        t0 = time.perf_counter()
        for _ in range(10):
            eng.step()
        res[tag].append((time.perf_counter() - t0) / 10 * 1e3)
        digs.append((tag, _digest(eng)))
eng._culture_learn_python = new_impl
mo, mn = float(np.median(res["old"])), float(np.median(res["new"]))
print("\n④ 整 tick 交替 A/B（10 tick × 3 轮，同进程同状态）")
print(f"  旧: {[round(x, 2) for x in res['old']]}  中位 {mo:.2f} ms/tick")
print(f"  新: {[round(x, 2) for x in res['new']]}  中位 {mn:.2f} ms/tick")
print(f"  ⇒ 整 tick {mo / mn:.3f}×；省 {mo - mn:.2f} ms/tick（N={len(eng._flat)}）")
print(f"  状态摘要逐轮：{'全部一致 ✅' if len(set(d for _, d in digs)) == 1 else '有差异 ❌'}"
      f"（每轮 = 10 tick 后 flat/energy/genes/interpret/age 摘要哈希）")
