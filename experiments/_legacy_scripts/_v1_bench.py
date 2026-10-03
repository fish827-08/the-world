"""R244 v1 收益实测：rd 开档（bgzero）下**资源侧子集化**的实收
（480×960 T4 同构臂，静默机、并发 1）。

配对（同 seed / 同 rd 开 / 仅 `sparse_fields` 差 ⇒ 两臂轨迹逐位同轨，计时才可比）：
  [地板] N=0（pop=1 → 灭绝后）300 tick —— 全场臂 = 旧 rd 每 tick 全额结算；
         稀疏臂 = 脏格子集（N=0 收敛后脏集≈空 ⇒ **收益上界**）
  [活体] pop=2000 前 200 tick（N>0 窗口）—— 真实 N 下的净收益
  [微基] `_regrowth_amount` 全场 vs 子集（|idx| 扫描）—— 解释上面两段的构成

对照 T-D 口径：`_td_rd_floor.py`（同 480×960 / pop=1 / subpos / speed_max=gain=0.25）。

用法：.venv\\Scripts\\python.exe _trash_local/_v1_bench.py
"""
from __future__ import annotations

import hashlib
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from experiments.steady_k_probe import SUBDIV_STD, make_cfg  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402

WARM = 20
TICKS_FLOOR = 300
TICKS_LIVE = 200


def make(sparse: bool, pop: int) -> SphereEngine:
    """rd 开（bgzero ⇒ v1 放行资源侧惰性）；仅 `sparse_fields` 差。"""
    cfg, _ = make_cfg(seed=42, rows=480, cols=960, pop=pop, patches=480,
                      subpos=True, speed_max=0.25, gain=0.25, subdiv=SUBDIV_STD,
                      k=2.5, max_count=30000, rgm=1.195)
    cfg.resource_dynamics.enabled = True
    cfg.simulation.sparse_fields = bool(sparse)
    return SphereEngine(cfg)


def digest(e: SphereEngine) -> tuple:
    h = lambda a: hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]  # noqa: E731
    return (e.tick, len(e._id), int(e._flat.sum()), round(float(e._energy.sum()), 6),
            h(e.resources._grid), h(e.signals._marks), h(e.signals._age))


def run_to_extinct(e: SphereEngine) -> None:
    for _ in range(300):
        if e.extinct:
            break
        e.step()


def bench_pair(a: SphereEngine, b: SphereEngine, n: int,
               sample: int = 0) -> tuple[list[float], list[float], list[int]]:
    """交错逐 tick 计时（消顺序/漂移偏差；两臂轨迹逐位同轨 ⇒ 配对公平）。"""
    ta: list[float] = []
    tb: list[float] = []
    ns: list[int] = []
    for i in range(n):
        t0 = time.perf_counter()
        a.step()
        ta.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        b.step()
        tb.append(time.perf_counter() - t0)
        if sample and (i % sample == 0):
            ns.append(len(a._id))
    return ta, tb, ns


def _summ(ms: list[float]) -> tuple[float, float]:
    return statistics.mean(ms) * 1e3, statistics.median(ms) * 1e3


def _profile_arm(e: SphereEngine, n: int, label: str) -> None:
    import cProfile
    import pstats
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(n):
        e.step()
    pr.disable()
    st = pstats.Stats(pr)
    rows = sorted(
        ((tt / n * 1e3, ct / n * 1e3, fname, fn)
         for (fname, _l, fn), (_cc, _nc, tt, ct, _c) in st.stats.items()),
        reverse=True,
    )
    print(f"   [{label}] self top6（ms/tick）： " + "; ".join(
        f"{fn}[{fname.split(chr(92))[-1]}] {tt:.2f}" for tt, _c, fname, fn in rows[:6]))


def main() -> None:
    print("== R244 v1：rd 开（bgzero）· 资源侧子集化 vs 全场（480×960 / k=2.5 / subpos / 并发 1）==")

    # ---------- [地板] N=0 ----------
    a, b = make(False, 1), make(True, 1)
    run_to_extinct(a)
    run_to_extinct(b)
    assert a.extinct and b.extinct, "地板段要求 N=0"
    assert a.resources._lazy is False and a.signals._sparse is False, "全场臂被误开"
    assert b.resources._lazy is True and b.signals._sparse is True, "稀疏臂惰性未放行（v1 回归？）"
    assert a._rd.enabled and b._rd.enabled and a._rd.kill_denom == "regrowth"
    assert digest(a) == digest(b), "构造/跑段已分化 ⇒ 配对前提不成立"
    dirty0 = int(b.resources._dirty_mask.sum())
    bench_pair(a, b, WARM)                        # 同预热（交错 ⇒ 同漂移）
    ta, tb, _ = bench_pair(a, b, TICKS_FLOOR)
    assert digest(a) == digest(b), "地板段跑完分化 ⇒ 计时不可比"
    ma, da = _summ(ta)
    mb, db = _summ(tb)
    dirty1 = int(b.resources._dirty_mask.sum())
    print(f"[地板] N=0（灭绝后 {TICKS_FLOOR} tick 交错配对；脏格 {dirty0}→{dirty1}/{b.resources._grid.size}）：")
    print(f"   rd 全场 {ma:6.2f} 均值 / {da:6.2f} 中位  子集 {mb:6.2f} / {db:6.2f} ms/tick"
          f"  ⇒ 省 {ma - mb:5.2f} 均值 / {da - db:5.2f} 中位（{100 * (1 - mb / ma):.0f}%）")
    _profile_arm(a, 100, "地板·全场")
    _profile_arm(b, 100, "地板·子集")

    # ---------- [活体] pop=2000 ----------
    a2, b2 = make(False, 2000), make(True, 2000)
    bench_pair(a2, b2, WARM)                      # 两臂同预热（否则 tick 错位）
    n0 = len(a2._id)
    ta2, tb2, ns = bench_pair(a2, b2, TICKS_LIVE, sample=10)
    assert digest(a2) == digest(b2), "活体段跑完分化 ⇒ 计时不可比（或 v1 破等价！）"
    ma2, da2 = _summ(ta2)
    mb2, db2 = _summ(tb2)
    dirty2 = int(b2.resources._dirty_mask.sum())
    print(f"[活体] pop 起步 {n0} → {TICKS_LIVE} tick 交错配对（N 采样 10 tick/次；"
          f"末脏格 {dirty2}/{b2.resources._grid.size}）：")
    print(f"   N 区间 ≈ {min(ns)}–{max(ns)}（末值 {ns[-1]}；两臂逐位同轨）")
    print(f"   rd 全场 {ma2:6.2f} 均值 / {da2:6.2f} 中位  子集 {mb2:6.2f} / {db2:6.2f} ms/tick"
          f"  ⇒ 省 {ma2 - mb2:5.2f} 均值 / {da2 - db2:5.2f} 中位（{100 * (1 - mb2 / ma2):.0f}%）")
    _profile_arm(a2, 100, "活体·全场")
    _profile_arm(b2, 100, "活体·子集")

    # ---------- [微基] `_regrowth_amount` 全场 vs 子集 ----------
    rf = a.resources
    t_ref = int(a._tick)
    rng = np.random.default_rng(7)
    n_cells = rf._grid.size
    reps = 30
    print("[微基] `_regrowth_amount` 单调用（中位 30 样本；热缓存；|idx| 由 rng 抽）：")
    t0 = time.perf_counter()
    for _ in range(reps):
        rf._regrowth_amount(t_ref)
    full_ms = (time.perf_counter() - t0) / reps * 1e3
    print(f"   全场 |idx|=n_cells={n_cells}  {full_ms:7.3f} ms")
    for k in (0, 2_000, 20_000, 100_000):
        idx = (rng.choice(n_cells, size=k, replace=False) if k
               else np.zeros(0, np.int64))
        t0 = time.perf_counter()
        for _ in range(reps):
            rf._regrowth_amount(t_ref, idx)
        ms = (time.perf_counter() - t0) / reps * 1e3
        print(f"   子集 |idx|={k:6d}  {ms:7.3f} ms  （{100 * ms / full_ms:5.1f}% 全场）")


if __name__ == "__main__":
    main()
