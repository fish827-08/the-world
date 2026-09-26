"""地板税分解探针 —— 480×960 T4 档「N≈0 每 tick 固定开销」逐段归因。

背景（R217 §五 #1 = 稀疏化 B）
------------------------------
`[实测]` P1（2026-09-26）：480×960 世界里**没有生物**时，py 路径每 tick 仍烧
**16.7 ms**（rust 路径 5.4–6.4 ms）——"世界放大"的固定税，与 K 无关。
本探针把这份税**逐段归因**（消融法），为稀疏化 B 回答"哪一段能省多少"。

方法（纯观测，不改引擎源码；类级 monkeypatch，逐项测完还原）
------------------------------------------------------------
1. 构造 T4 同款配置（`steady_k_probe.make_cfg` + `apply_post_build`），pop=1 跑到灭绝；
2. 基线：测 K tick 的均 ms/tick（N=0，无逐个体项 ⇒ 纯粹格子级开销）；
3. 消融：逐项把类方法换成 no-op/等价空转，差值 = 该项成本：
   * `ResourceField.regrow`            —— 资源再生（含其触发的 LT 全格缓存）
   * `LightAndTemperature._ensure_cache` —— 光照/温度全格计算（regrow 内部再分离）
   * `ResourceField.total`             —— TickStats 的全场求和
   * `SignalField.tick`                —— 信号场全格推进
   * `SphereEngine._populate_history`  —— 历史/分通道账本
4. 报告每项 ms/tick、占地板比；并给「全消融」后的引擎残余 = 其余固定开销。

🔴 口径
* 消融只改"算力"，不改状态语义（N=0 时无个体消费/读格，网格值不影响后续耗时）
  ⇒ 计时有效；但**消融期间状态会漂**（如跳过 regrow ⇒ 格子不满）⇒ 只用于计时，
  不用于任何数值结论。
* 消融顺序固定、每项独立测量（从基线态出发），互不叠加（除"全消融"组合）。

用法
----
    .venv\\Scripts\\python.exe experiments/floor_tax_probe.py --ticks 300
    .venv\\Scripts\\python.exe experiments/floor_tax_probe.py --sparse --ticks 300   ← 稀疏化 B 对照
    .venv\\Scripts\\python.exe experiments/floor_tax_probe.py --arm rust --ticks 300

`--sparse`（2026-09-27，R217 §五 #1）：从**构造期**就走 `cfg.simulation.sparse_fields=True`
（引擎闸真路径）⇒ 与默认档跑同配置做前后对照。稀疏档下消融表含义反转：
「res.regrow 省 X ms」= 惰性路径的**残余**成本；`light.ensure_cache` 在 N=0 应≈0
（无全场刷新触发者）。对照口径：同一机器、紧邻两跑（本机常并发批跑 ⇒ 两列同报）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation.sphere_engine import SphereEngine            # noqa: E402
from world.resource_field import ResourceField               # noqa: E402
from world.light_and_temperature import LightAndTemperature  # noqa: E402
from world.signal_field import SignalField                   # noqa: E402
from experiments.scaling_rescale import apply_post_build     # noqa: E402
from experiments.steady_k_probe import make_cfg              # noqa: E402

# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print 会让脚本 rc=1 假失败；F-R15 族）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ---- 消融替身（签名与真方法一致；no-op 返回 None 即可） ----------------------

def _noop_regrow(self, tick):                       # noqa: ANN001
    return None


def _noop_ensure_cache(self, tick):                 # noqa: ANN001
    # 只跳过"重算"，保留旧缓存（时间分支不再前进；仅用于计时）
    return None


def _noop_total(self):                              # noqa: ANN001
    return 0.0


def _noop_signal_tick(self):                        # noqa: ANN001
    return None


def _noop_populate_history(self, stats):            # noqa: ANN001
    # 只跳过账本/历史累计；stats 本身照常构造（其成本含在基线里）
    return None


ABLATIONS: list[tuple[str, type, str, object]] = [
    ("res.regrow",        ResourceField,            "regrow",           _noop_regrow),
    ("light.ensure_cache", LightAndTemperature,     "_ensure_cache",    _noop_ensure_cache),
    ("res.total",         ResourceField,            "total",            _noop_total),
    ("signals.tick",      SignalField,              "tick",             _noop_signal_tick),
    ("history",           SphereEngine,             "_populate_history", _noop_populate_history),
]


def measure(eng: SphereEngine, ticks: int) -> tuple[float, float]:
    """返回 (墙钟 ms/tick, CPU ms/tick)。

    机器有并发批跑时墙钟会被放大（本项目本机常态）⇒ CPU 时间更接近"这份工作的本征成本"，
    两列同报（口径：wall = 墙钟，cpu = time.process_time，含本进程全部线程）。
    """
    w0, c0 = time.perf_counter(), time.process_time()
    for _ in range(ticks):
        eng.step()
    dw = time.perf_counter() - w0
    dc = time.process_time() - c0
    n = max(ticks, 1)
    return dw / n * 1e3, dc / n * 1e3


def run_arm(arm: str, args: argparse.Namespace) -> int:
    subpos = arm == "py"
    use_sim_core = arm == "rust"
    cfg, notes = make_cfg(
        seed=args.seed, rows=args.rows, cols=args.cols, pop=1,
        patches=args.patches, subpos=subpos,
        speed_max=args.speed_max, gain=args.speed_max, subdiv=args.subdiv,
        k=args.k, max_count=args.max_count, rgm=args.rgm,
    )
    if use_sim_core:
        cfg.simulation.use_sim_core = True
    sparse_on = bool(getattr(args, "sparse", False)) and not use_sim_core
    if getattr(args, "sparse", False) and use_sim_core:
        print("   ⚠️ --sparse 在 rust 臂被引擎闸忽略（use_sim_core ⇒ sparse off）")
    cfg.simulation.sparse_fields = sparse_on
    eng = SphereEngine(cfg)
    apply_post_build(eng, notes)
    n_cells = eng.world.n_cells
    print(f"== 臂={arm}  世界={args.rows}x{args.cols}（{n_cells} 格）  "
          f"subpos={subpos}  use_sim_core={use_sim_core}  k={args.k:g}  "
          f"sparse={sparse_on} ==")
    if sparse_on:
        print(f"   [稀疏] 引擎闸={eng._sparse_fields}  资源惰性={eng.resources._lazy}"
              f"  信号稀疏={eng.signals._sparse}")
    print(f"   [构造] LT Rust 加速: "
          f"{'on' if getattr(eng.light, '_rust_lt', None) else 'off'}")

    # ---- 跑到灭绝（pop=1） -------------------------------------------------
    ext_t0 = time.perf_counter()
    t = 0
    while len(eng._id) > 0 and t < args.ext_cap:
        eng.step()
        t += 1
    print(f"   [灭绝] pop=1 在 tick {t} 灭绝（cap={args.ext_cap}），"
          f"耗时 {time.perf_counter() - ext_t0:.1f}s")
    if len(eng._id) > 0:
        print("   ⚠️ 未灭绝 ⇒ 地板里仍含逐个体现，读数偏高（请加大 --ext-cap）")

    # ---- 基线 --------------------------------------------------------------
    measure(eng, 20)                                # 预热（缓存/分支稳定）
    t_base, c_base = measure(eng, args.ticks)
    dirty = (int(eng.resources._dirty_mask.sum())
             if getattr(eng.resources, "_lazy", False) else -1)
    active = (int(eng.signals._active_idx.size)
              if getattr(eng.signals, "_sparse", False) else -1)
    extra = ""
    if dirty >= 0:
        extra += f"  脏格={dirty}/{n_cells}"
    if active >= 0:
        extra += f"  活跃信号格={active}"
    print(f"\n   [基线] {t_base:.2f} ms/tick（墙钟）/ {c_base:.2f} ms/tick（CPU）"
          f"（N=0，{args.ticks} tick）{extra}")

    # ---- 逐项消融 ----------------------------------------------------------
    rows_out: list[tuple[str, float, float]] = []
    for name, cls, attr, stub in ABLATIONS:
        orig = getattr(cls, attr)
        setattr(cls, attr, stub)
        try:
            measure(eng, 10)                        # 消融态预热
            t_off, c_off = measure(eng, args.ticks)
        finally:
            setattr(cls, attr, orig)
        delta = c_base - c_off
        rows_out.append((name, delta, delta / c_base * 100.0))
        print(f"   [消融] {name:22s} 省 {delta:6.2f} ms/tick CPU"
              f"（{delta / c_base * 100:5.1f}%）  → 剩 {c_off:6.2f}（墙钟 {t_off:6.2f}）")

    # ---- 全消融（组合） ----------------------------------------------------
    saved = [(cls, attr, getattr(cls, attr)) for _, cls, attr, _ in ABLATIONS]
    for _, cls, attr, stub in ABLATIONS:
        setattr(cls, attr, stub)
    try:
        measure(eng, 10)
        t_all, c_all = measure(eng, args.ticks)
    finally:
        for cls, attr, orig in saved:
            setattr(cls, attr, orig)
    print(f"   [全消融] 剩 {c_all:.2f} ms/tick CPU（墙钟 {t_all:.2f}）—— 引擎残余"
          f"（tick 调度/TickStats/能量求和/其它全场项）")

    # ---- 单算子微基准（解释 regrow 内部构成；n_cells 同尺寸） ---------------
    print("\n   [微基准] 单算子 @ 同尺寸（仅解释构成，非消融）")
    a = np.ones(n_cells, dtype=np.float64)
    b = np.full(n_cells, 0.5, dtype=np.float64)
    for label, fn in (
        ("clip((T+20)/20,0,1)", lambda: np.clip((a + 20.0) / 20.0, 0.0, 1.0)),
        ("np.power(f, 1.0)", lambda: np.power(b, 1.0)),
        ("where(mask, g*pm, g*bm)", lambda: np.where(a > 0, b * 1.195, b * 0.0)),
        ("minimum(cap, g+grow,out)", lambda: np.minimum(a, b + b, out=b)),
        ("sum()", lambda: float(a.sum())),
    ):
        reps = 20
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        print(f"     {label:28s} {(time.perf_counter() - t0) / reps * 1e3:6.3f} ms")

    # ---- 汇总 --------------------------------------------------------------
    print("\n   [汇总] 地板税分项（ms/tick）")
    print("   | 项 | ms/tick | 占比 | 稀疏化 B 能否省 |")
    print("   |---|---|---|---|")
    can = {
        "res.regrow": "✅ 闭式惰性（只结算脏格）",
        "light.ensure_cache": "🟡 随 regrow 一起惰性化（子集查询）",
        "res.total": "❌ 统计口径，全场求和保留",
        "signals.tick": "✅ 活跃集推进（只推进有标记格）",
        "history": "❌ 账本，与格数无关（本项应≈0）",
    }
    for name, delta, pct in rows_out:
        print(f"   | {name} | {delta:.2f} | {pct:.1f}% | {can.get(name, '')} |")
    print(f"   | **合计可省（regrow+LT+signals）** | "
          f"**{sum(d for n, d, _ in rows_out if n in ('res.regrow', 'signals.tick')):.2f}**"
          f" | — | — |")
    print(f"   | 引擎残余（全消融后） | {c_all:.2f} | {c_all / c_base * 100:.1f}% | — |")
    if sparse_on:
        print("   ℹ️ 本跑 = 稀疏档（构造期 sparse_fields=True）："
              "上表「省 X」= 各段**残余**成本；对照默认档请不加 --sparse 同参数再跑一次")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="地板税分解（消融法）")
    ap.add_argument("--arm", choices=("py", "rust"), default="py",
                    help="py = T4 实测臂（subpos on / use_sim_core off）；rust = 已下沉臂")
    ap.add_argument("--sparse", action="store_true",
                    help="构造期开 sparse_fields（仅 py 臂有效）⇒ 与默认档同配置前后对照")
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--cols", type=int, default=960)
    ap.add_argument("--patches", type=int, default=480)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ticks", type=int, default=300, help="每项消融的计时 tick 数")
    ap.add_argument("--ext-cap", type=int, default=40000, help="跑到灭绝的 tick 上限")
    ap.add_argument("--k", type=float, default=2.5)
    ap.add_argument("--max-count", type=int, default=30000)
    ap.add_argument("--speed-max", type=float, default=0.125)
    ap.add_argument("--subdiv", type=int, default=80)
    ap.add_argument("--rgm", type=float, default=1.195)
    args = ap.parse_args()
    return run_arm(args.arm, args)


if __name__ == "__main__":
    raise SystemExit(main())
