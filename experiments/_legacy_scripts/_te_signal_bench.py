"""R231 T-E 收益实测：rd 开档下"信号稀疏"的实收（480×960 T4 同构臂，静默机、并发 1）。

三问（全部**配对**：同 seed / 同 rd 开 / 仅 `sparse_fields` 差 ⇒ 轨迹逐位相同，计时才可比）：
  [地板] N=0（pop=1 → 灭绝后）300 tick = 信号全场的**固定成本上界**（与 T-A/T-D 地板同口径）
  [活体] pop=2000 前 200 tick（N>0 窗口）= 真实 N 下的**净收益**（含 `_activate` 维护成本）
  [微基] `signals.tick()` 单调用成本 vs |active| —— 解释上面两段的构成

用法：.venv\\Scripts\\python.exe _trash_local/_te_signal_bench.py
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
    """rd **开**；仅 `sparse_fields` 差（资源侧因 rd 锁恒为全场 ⇒ 差值 = 信号侧净效应）。"""
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


def bench(e: SphereEngine, n: int, sample: int = 0) -> tuple[tuple[float, float], list[int]]:
    ns: list[int] = []
    w0, c0 = time.perf_counter(), time.process_time()
    for i in range(n):
        e.step()
        if sample and (i % sample == 0):
            ns.append(len(e._id))
    wall = (time.perf_counter() - w0) / n * 1e3
    cpu = (time.process_time() - c0) / n * 1e3
    return (wall, cpu), ns


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
    print(f"   [{label}] self top6： " + "; ".join(
        f"{fn}[{fname.split(chr(92))[-1]}] {tt:.2f}" for tt, _c, fname, fn in rows[:6]))
    sig = [r for r in rows if "signal_field" in r[2]]
    print(f"   [{label}] SignalField self/cum： " + "; ".join(
        f"{fn} {tt:.3f}/{ct:.3f}" for tt, ct, _f, fn in sig[:6]))


def main() -> None:
    print("== R231 T-E：rd 开档 · 信号稀疏 vs 全场（480×960 / k=2.5 / subpos / 并发 1）==")

    # ---------- [地板] N=0 ----------
    a, b = make(False, 1), make(True, 1)
    run_to_extinct(a)
    run_to_extinct(b)
    assert a.extinct and b.extinct, "地板段要求 N=0"
    assert a.signals._sparse is False and b.signals._sparse is True
    assert a.resources._lazy is False and b.resources._lazy is False   # rd 锁未放开
    assert digest(a) == digest(b), "构造/跑段已分化 ⇒ 配对前提不成立"
    bench_pair(a, b, WARM)                        # 同预热（交错 ⇒ 同漂移）
    ta, tb, _ = bench_pair(a, b, TICKS_FLOOR)
    assert digest(a) == digest(b), "地板段跑完分化 ⇒ 计时不可比"
    ma, da = _summ(ta)
    mb, db = _summ(tb)
    print(f"[地板] N=0（灭绝后 {TICKS_FLOOR} tick 交错配对；signals.duration={a.signals.duration}）：")
    print(f"   信号全场 {ma:6.2f} 均值 / {da:6.2f} 中位  稀疏 {mb:6.2f} / {db:6.2f} ms/tick"
          f"  ⇒ 省 {ma - mb:5.2f} 均值 / {da - db:5.2f} 中位")
    _profile_arm(a, 100, "地板·全场")
    _profile_arm(b, 100, "地板·稀疏")

    # ---------- [活体] pop=2000 ----------
    a2, b2 = make(False, 2000), make(True, 2000)
    bench_pair(a2, b2, WARM)                      # 两臂同预热（否则 tick 错位）
    n0 = len(a2._id)
    ta2, tb2, ns = bench_pair(a2, b2, TICKS_LIVE, sample=10)
    assert digest(a2) == digest(b2), "活体段跑完分化 ⇒ 计时不可比"
    ma2, da2 = _summ(ta2)
    mb2, db2 = _summ(tb2)
    print(f"[活体] pop 起步 {n0} → {TICKS_LIVE} tick 交错配对（N 采样 10 tick/次）：")
    print(f"   N 区间 ≈ {min(ns)}–{max(ns)}（末值 {ns[-1]}；两臂逐位同轨）")
    print(f"   信号全场 {ma2:6.2f} 均值 / {da2:6.2f} 中位  稀疏 {mb2:6.2f} / {db2:6.2f} ms/tick"
          f"  ⇒ 省 {ma2 - mb2:5.2f} 均值 / {da2 - db2:5.2f} 中位")
    _profile_arm(a2, 100, "活体·全场")
    _profile_arm(b2, 100, "活体·稀疏")

    # ---------- [微基] signals.tick() vs |active| ----------
    n_cells = a.resources._grid.size
    rng = np.random.default_rng(7)
    print("[微基] signals.tick() 单调用（中位 50 样本；|active| 由 write_many 维持；热缓存）：")
    for k in (0, 2_000, 20_000):
        cells = rng.choice(n_cells, size=k, replace=False) if k else np.zeros(0, np.int64)
        pats = np.ones(k, dtype=np.uint8)
        ta: list[float] = []
        tb: list[float] = []
        for _ in range(50):
            if k:
                a.signals.write_many(cells, pats)
                b.signals.write_many(cells, pats)
            t0 = time.perf_counter()
            a.signals.tick()
            ta.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            b.signals.tick()
            tb.append(time.perf_counter() - t0)
        act = int(b.signals._active_idx.size)
        print(f"   |active|={k:6d}（实 {act:6d}）  全场 {statistics.median(ta)*1e3:7.3f} ms"
              f"  稀疏 {statistics.median(tb)*1e3:7.3f} ms")


if __name__ == "__main__":
    main()
