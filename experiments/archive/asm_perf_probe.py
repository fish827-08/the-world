"""R239/R258 ASM（模式仲裁）性能实测 —— **配对口径**（R219：相对值 + N + 并发）。

问题：开仲裁档（`action_selection.mode="arbitration"`）相对融合档（`"fusion"`）
到底多花/少花多少 ms/tick？（设计稿 §四-5 `[估算]` < 0.5 ms/tick，须实测。）

口径（写死，防"数字没出处"）
------------------------------
* **配对**：同一世界/种子/tick 数/气味通道（三通道**两臂都开** ⇒ 隔离出"仲裁逻辑"本身的
  增量），唯一差别 = `asm_mode`；报 `Δ = arbitration − fusion` 与相对比；
* **并发**：本探针单进程、串行各臂 ⇒ 报 `concurrency=1`；
* **双路径**：Python（`use_sim_core=False`）与 Rust（`True`）各测一组 —— 仲裁段在
  Python 是逐个体循环 + 向量化快路径，在 Rust 走 `step_movement`；
* **装置**：S3 同源（480×960 / patchy / bgzero / rgm=1.195 / patches=1700 / pop=3000
  ≈ T4 实测 K 中位） + `--quick` 可换 60×120 / pop=1000 冒烟。

用法
----
    python3.12 experiments/asm_perf_probe.py --quick
    python3.12 experiments/asm_perf_probe.py               # 默认 480×960 / pop 3000
输出：`results/asm_v1/asm_perf.json`（**产物入数据仓**，主仓 gitignore）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation.config import SimConfig, ActionSelectionConfig   # noqa: E402
from simulation.sphere_engine import SphereEngine                # noqa: E402

# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print 会让脚本 rc=1 假失败；F-R15 族）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # 非 TTY / 旧解释器：不因诊断能力缺失而阻断运行

CHANNELS = ("food", "risk", "kin")


def _mk(rows, cols, pop, patches, mode, use_sim_core, seed=42) -> SimConfig:
    c = SimConfig(seed=seed)
    c.world.rows, c.world.cols = rows, cols
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = True
    c.resources.patch_regrowth_mult = 1.195
    c.resources.patch_count = patches
    c.population.initial_count = pop
    c.smell.channels = CHANNELS                     # 两臂都开（隔离仲裁增量）
    c.action_selection = ActionSelectionConfig(mode=mode)
    c.simulation.use_sim_core = bool(use_sim_core)
    return c


def run_arm(rows, cols, pop, patches, mode, use_sim_core, ticks, warm=20, seed=42,
            freeze_repro=False) -> dict:
    cfg = _mk(rows, cols, pop, patches, mode, use_sim_core, seed=seed)
    eng = SphereEngine(cfg)
    if freeze_repro:
        # 🔴 **只用于性能口径**（非科学跑）：繁殖基因置满上界 ⇒ 阈值 0.9×max ⇒ 测窗内
        #   几乎不繁殖 ⇒ 两臂 N 轨迹贴近（N 差异 ≪ 自然档）⇒ Δ 不含"N 轨迹差"混淆。
        from simulation.genes import Gene
        eng._genes[:, Gene.REPRO_THRESHOLD] = 1.0
    for _ in range(warm):
        eng.step()
    n_warm = int(len(eng._id))
    t0 = time.perf_counter()
    for _ in range(ticks):
        eng.step()
    ms = (time.perf_counter() - t0) / max(ticks, 1) * 1e3
    return {
        "mode": mode, "use_sim_core": bool(use_sim_core),
        "ms_per_tick": round(ms, 4), "N_warm": n_warm, "N_end": int(len(eng._id)),
        "ticks": int(ticks), "warm": int(warm), "seed": int(seed),
        "cells": int(rows * cols), "freeze_repro": bool(freeze_repro),
        "asm": eng.asm_probe(),
    }


def pair(rows, cols, pop, patches, use_sim_core, ticks, warm, seed,
         freeze_repro=False) -> dict:
    a = run_arm(rows, cols, pop, patches, "fusion", use_sim_core, ticks, warm, seed,
                freeze_repro)
    b = run_arm(rows, cols, pop, patches, "arbitration", use_sim_core, ticks, warm, seed,
                freeze_repro)
    return {
        "fusion": a, "arbitration": b,
        "delta_ms": round(b["ms_per_tick"] - a["ms_per_tick"], 4),
        "rel_pct": round((b["ms_per_tick"] - a["ms_per_tick"])
                         / max(a["ms_per_tick"], 1e-9) * 100.0, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="冒烟档：60×120 / pop 1000 / 60 tick（默认 480×960 / pop 3000）")
    ap.add_argument("--ticks", type=int, default=None)
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reps", type=int, default=3,
                    help="每档重复次数（本机噪声 ±2 ms/tick 量级 ⇒ 报中位 + 极差，默认 3）")
    ap.add_argument("--freeze-repro", action="store_true",
                    help="繁殖基因置满 ⇒ N 轨迹贴近（隔离 'N 轨迹差' 混淆；见 run_arm 注释）")
    ap.add_argument("--out", default="results/asm_v1/asm_perf.json")
    args = ap.parse_args()

    if args.quick:
        rows, cols, pop, patches, ticks = 60, 120, 1000, 30, 60
    else:
        # pop0 ≈ T4 实测 K 中位 3004 ⇒ 测窗内 N 稳定（配对可比性优先；S3 的 10000 档
        #   会向 K 回落 ⇒ N 大幅下滑 ⇒ 低估逐个体成本）
        rows, cols, pop, patches, ticks = 480, 960, 3000, 1700, 60
    if args.ticks is not None:
        ticks = args.ticks

    res = {
        "device": {"rows": rows, "cols": cols, "pop0": pop, "patches": patches,
                   "channels": list(CHANNELS), "ticks": ticks, "warm": args.warm,
                   "seed": args.seed, "concurrency": 1, "reps": args.reps,
                   "freeze_repro": bool(args.freeze_repro)},
    }
    for path, sim_core in (("py", False), ("rust", True)):
        reps = [pair(rows, cols, pop, patches, sim_core, ticks, args.warm, args.seed,
                     args.freeze_repro) for _ in range(max(int(args.reps), 1))]
        d = [r["delta_ms"] for r in reps]
        res[path] = {"reps": reps, "delta_ms_median": round(float(np.median(d)), 4),
                     "delta_ms_min": min(d), "delta_ms_max": max(d)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    for path in ("py", "rust"):
        p = res[path]
        for k, r in enumerate(p["reps"]):
            print(f"[{path} rep{k}] fusion {r['fusion']['ms_per_tick']:8.3f} "
                  f"arb {r['arbitration']['ms_per_tick']:8.3f} "
                  f"Δ {r['delta_ms']:+7.3f} ({r['rel_pct']:+6.2f}%) "
                  f"N {r['fusion']['N_warm']}→{r['fusion']['N_end']}/"
                  f"{r['arbitration']['N_end']}")
        print(f"[{path}] Δ 中位 {p['delta_ms_median']:+.3f} ms/tick "
              f"(极差 {p['delta_ms_min']:+.3f} .. {p['delta_ms_max']:+.3f})")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
