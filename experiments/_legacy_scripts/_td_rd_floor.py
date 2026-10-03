"""T-D 证据：rd 开启后的"每 tick 地板"分解（N=0，480×960，T4 档参数）。

问题（R230 T-D）：`sparse_fields` 能否与 `resource_dynamics` 共存？
  · 现状 gate：`sparse_fields` 需 rd 关（engine L842-846）⇒ rd 开 = 自动静默退回全场。
  · 本脚本只量事实：rd 开的每 tick 地板 = ? 其中
      (a) regrow 链（`_regrowth_amount` 全场 + `growth_multiplier` + `minimum`）——惰性再生的候选
      (b) rd 记账块（`capacity_from_base`/`note_tick`/`rotate`/`validate_writeback`）——惰性消不掉
      (c) 其它全场项（信号场 / LT 缓存）
口径：T4 档（480×960 / patches=480 / subpos on / k=2.5 / max_count=30000 / rgm=1.195），
     `pop=1` 跑到灭绝后计时（N=0 ⇒ 地板；与 R219 T-A 同口径）。

用法：.venv\\Scripts\\python.exe _trash_local/_td_rd_floor.py
"""
from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from experiments.steady_k_probe import SUBDIV_STD, make_cfg  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402

TICKS = 150
WARM = 20


def build(rd_on: bool) -> SphereEngine:
    cfg, _ = make_cfg(seed=42, rows=480, cols=960, pop=1, patches=480,
                      subpos=True, speed_max=0.25, gain=0.25, subdiv=SUBDIV_STD,
                      k=2.5, max_count=30000, rgm=1.195)
    cfg.resource_dynamics.enabled = bool(rd_on)
    e = SphereEngine(cfg)
    for _ in range(200):                      # 跑到灭绝（pop=1）
        if e.extinct:
            break
        e.step()
    return e


def bench(e: SphereEngine, n: int) -> tuple[float, float]:
    w0, c0 = time.perf_counter(), time.process_time()
    for _ in range(n):
        e.step()
    return ((time.perf_counter() - w0) / n * 1e3,
            (time.process_time() - c0) / n * 1e3)


def main() -> None:
    for rd_on in (False, True):
        e = build(rd_on)
        assert e.extinct, f"未灭绝（N={len(e._id)}），本测要求 N=0 地板"
        import numpy as _np
        mask_ref = _np.array(e._rd._mask, copy=True)          # rd 掩码基准
        cap_ref = _np.array(e.resources._capacity, copy=True)  # 容量基准
        bench(e, WARM)                        # 预热
        wall, cpu = bench(e, TICKS)
        print(f"[rd={'on ' if rd_on else 'off'}] N=0 地板 = {wall:.2f} ms/tick 墙钟 / "
              f"{cpu:.2f} ms/tick CPU（{TICKS} tick）  "
              f"sparse={e._sparse_fields} rd={bool(getattr(e._rd, 'enabled', False))}")
        if rd_on:
            # 掩码/容量是否恒定（决定"惰性再生前提是否被 rd 破坏"的事实基础）
            import numpy as _np
            mask_now = _np.array_equal(e._rd._mask, mask_ref)
            cap_now = _np.array_equal(e.resources._capacity, cap_ref)
            print(f"  bg_production_zero={e.config.resources.bg_production_zero}  "
                  f"掩码自构造以来恒定={mask_now}  容量恒定={cap_now}")
            pr = cProfile.Profile()
            pr.enable()
            for _ in range(100):
                e.step()
            pr.disable()
            st = pstats.Stats(pr)
            rows = []
            for (fname, lineno, fn), (cc, nc, tt, ct, _callers) in st.stats.items():
                if tt < 0.1:                  # 自耗时 ≥1 ms/tick 才列（cum 会被嵌套放大）
                    continue
                rows.append((tt / 100 * 1e3, ct / 100 * 1e3, fn, fname.split("/")[-1], lineno))
            rows.sort(reverse=True)
            print("  cProfile（self / cum ms per tick，按 self 排序，列 ≥1 ms/tick）：")
            for tt_ms, ct_ms, fn, f, ln in rows[:30]:
                print(f"    self {tt_ms:7.2f}  cum {ct_ms:8.2f}  {fn}  [{f}:{ln}]")


if __name__ == "__main__":
    main()
