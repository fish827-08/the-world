"""稀疏化 B 引擎级 A/B —— 480×960 T4 档 sparse on/off 逐位对拍 + 配对计时。

用途（R217 §五 #1 交付证据）
------------------------------
单元测试（`tests/test_sparse_fields.py`）在 60×120 小世界；本脚本补**实尺寸**证据：
  * A 臂 = 默认档（`sparse_fields=False`），B 臂 = 稀疏档（`True`），
    **同 cfg / 同 seed / 同构造路径**（构造确定 ⇒ 构造态即逐位同）；
  * 逐 tick 比 digest（tick / 存活数 / flat / genes / grid / marks / age /
    energy / stomach / 死因账本）——任一 tick 分裂即报错退出；
  * 同报**配对** CPU 计时（引擎 A/B 交替 step ⇒ 轨迹逐位相同 ⇒ 严格配对；
    本机常有并发批跑，墙钟失真，CPU 列更稳）；
  * 统计"脏格非空 / 活跃信号格非空"的 tick 数 ⇒ 证明稀疏路径**非空转**
    （空转等价是平凡真，不构成证据）。

🔴 注意：本脚本构造的 pop=300 在 T4 档（背景产能归零）会快速衰减到个位数
——这是**环境**使然，不影响逐位对拍；A/B 两臂轨迹完全相同 ⇒ 计时配对仍成立。

用法
----
    .venv\\Scripts\\python.exe experiments/sparse_lockstep_ab.py
    .venv\\Scripts\\python.exe experiments/sparse_lockstep_ab.py --ticks 120 --pop 500
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation.sphere_engine import SphereEngine            # noqa: E402
from experiments.scaling_rescale import apply_post_build     # noqa: E402
from experiments.steady_k_probe import make_cfg              # noqa: E402

# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print 会让脚本 rc=1 假失败；F-R15 族）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def digest(e: SphereEngine) -> tuple:
    """全状态摘要；**任何**一项不等即判分裂（含死因账本）。"""
    return (
        e._tick, len(e._id), sha(e._flat), sha(e._genes),
        sha(e.resources._grid), sha(e.signals._marks), sha(e.signals._age),
        round(float(e._energy.sum()), 6),
        round(float(e._stomach.sum()), 6),
        tuple(sorted((str(k), int(v)) for k, v in e._run_deaths.items())),
    )


def build(sparse: bool, args: argparse.Namespace) -> SphereEngine:
    cfg, notes = make_cfg(seed=args.seed, rows=args.rows, cols=args.cols,
                          pop=args.pop, patches=args.patches, subpos=True,
                          speed_max=args.speed_max, gain=args.speed_max,
                          subdiv=args.subdiv, k=args.k,
                          max_count=args.max_count, rgm=args.rgm)
    cfg.simulation.sparse_fields = bool(sparse)
    t0 = time.perf_counter()
    e = SphereEngine(cfg)
    apply_post_build(e, notes)
    print(f"   [构造] sparse={sparse}  引擎闸={e._sparse_fields}  "
          f"资源惰性={e.resources._lazy}  信号稀疏={e.signals._sparse}  "
          f"耗时 {time.perf_counter() - t0:.1f}s")
    return e


def main() -> int:
    ap = argparse.ArgumentParser(description="稀疏化 B 480×960 引擎级 A/B")
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--cols", type=int, default=960)
    ap.add_argument("--patches", type=int, default=480)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pop", type=int, default=300)
    ap.add_argument("--ticks", type=int, default=80)
    ap.add_argument("--k", type=float, default=2.5)
    ap.add_argument("--max-count", type=int, default=30000)
    ap.add_argument("--speed-max", type=float, default=0.125)
    ap.add_argument("--subdiv", type=int, default=80)
    ap.add_argument("--rgm", type=float, default=1.195)
    ap.add_argument("--n-min", type=int, default=0,
                    help=">0 时额外报「N≥该值」的配对段（R219 §二-3 要求比值绑 N）")
    args = ap.parse_args()

    print(f"== {args.rows}×{args.cols} T4 档 sparse on/off 逐位对拍 ==")
    ea = build(False, args)
    eb = build(True, args)
    if digest(ea) != digest(eb):
        print("   🔴 构造态即不一致（不该发生）")
        return 1
    print(f"   [构造态] digest 一致 ✓ tick={ea._tick} N={len(ea._id)}")

    ta = tb = wa = wb = 0.0
    per: list[tuple[int, float, float, float, float]] = []
    saw_dirty = saw_active = 0
    for t in range(args.ticks):
        c0 = time.process_time()
        w0 = time.perf_counter()
        ea.step()
        ca = time.process_time() - c0
        waa = time.perf_counter() - w0
        c0 = time.process_time()
        w0 = time.perf_counter()
        eb.step()
        cb = time.process_time() - c0
        wbb = time.perf_counter() - w0
        ta, wa = ta + ca, wa + waa
        tb, wb = tb + cb, wb + wbb
        per.append((len(ea._id), ca, cb, waa, wbb))
        da, db = digest(ea), digest(eb)
        if da != db:
            print(f"   🔴 tick {t + 1} digest 分裂：\n     A={da}\n     B={db}")
            return 1
        saw_dirty += int(bool(eb.resources._dirty_mask.any()))
        saw_active += int(eb.signals._active_idx.size > 0)
        if (t + 1) % 20 == 0:
            print(f"   [t={t + 1:3d}] N={len(ea._id):4d}  digest 一致 ✓  "
                  f"脏格={int(eb.resources._dirty_mask.sum()):6d}  "
                  f"活跃信号格={int(eb.signals._active_idx.size):4d}")

    n = max(args.ticks, 1)

    def seg(rows: list[tuple[int, float, float, float, float]]) -> str:
        k = len(rows)
        sa = sum(r[1] for r in rows) / k * 1e3
        sb = sum(r[2] for r in rows) / k * 1e3
        wa_ = sum(r[3] for r in rows) / k * 1e3
        wb_ = sum(r[4] for r in rows) / k * 1e3
        n0, n1 = rows[0][0], rows[-1][0]
        return (f"N {n0}→{n1}：墙钟 默认 {wa_:.1f} / 稀疏 {wb_:.1f} ms/tick（省 "
                f"{100 - wb_ / wa_ * 100:.1f}%）| CPU 默认 {sa:.1f} / 稀疏 {sb:.1f}"
                f"（省 {100 - sb / sa * 100:.1f}%）")

    print(f"\n   [逐位] {args.ticks} tick 全程 digest 一致 ✓ "
          f"（脏格非空 {saw_dirty}/{n}，活跃非空 {saw_active}/{n} ⇒ 非空转）")
    print(f"   [计时] 全程 {seg(per)}")
    q = max(n // 4, 1)
    print(f"   [计时] 首1/4 {seg(per[:q])}")
    print(f"   [计时] 末1/4 {seg(per[-q:])}")
    if args.n_min > 0:                            # R219 §二-3：比值必须与 N 绑定
        hi = [r for r in per if r[0] >= args.n_min]
        print(f"   [计时] N≥{args.n_min} 段（{len(hi)} tick）{seg(hi)}" if hi else
              f"   [计时] N≥{args.n_min} 段：无样本")
    print(f"   [末态] N={len(ea._id)}  tick={ea._tick}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
