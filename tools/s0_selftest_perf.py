"""S0-kernel-b 自测性能表（wall/tick 微基准）——Pre-Flight 用的真值件，**非跑批**。

口径声明（照 PI 10-09 20:5x 班：本机微基准、零入数据仓、夜墙前收口）
--------------------------------------------------------------------
- 测的是 **`s0_kernel.harness` 的最小世界**（S0 双臂容器），**不是**主引擎——PI 板上的
  云机数（1500 tick = 4.39s／seed7 五腿逐位同）走的是主引擎＋步A 探针链路，两条码路**不可互推**。
- 🔴 **计时口径＝独立进程**（`--mode fresh`，默认）：实测同一进程内连跑会把 wall 抬到冷启动的
  **2～2.4 倍**（同 seed 轨迹逐位相同、pop 与 cap_hits 完全一致，只有耗时漂 ⇒ 是解释器/内存侧的
  自身污染，不是行为差）。chain 脚本本来就**逐 run 起独立进程** ⇒ 机时规划只认 fresh 那一栏。
- 每档跑 `ticks` 次 `step()`，报 `ms/tick`（中位与最差）＋ 末人口 ＋ `cap_hits`；
  人口顶到 `max_ind` 时耗时会被规模支配 ⇒ 表里同时给"封顶层"与"未封顶层"两类档，别把两者混成一格。
- **不写数据仓、不产 CSV**（除 `--out` 显式指定的本机路径）；不消费主引擎任何状态。
- 外推公式就一行：`秒/run ≈ ms_per_tick × ticks / 1000`，正式档 20,000 tick 由中位数外推，
  标 [外推]；标定批/正式批的**实测**计时项仍归砚（§七bis #9 唯一源＝点火程序令 @dda9d5d）。

用法
----
    ../../.venv/Scripts/python.exe tools/s0_selftest_perf.py            # 默认六档
    ../../.venv/Scripts/python.exe tools/s0_selftest_perf.py --reps 5 --out /tmp/perf.json
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s0_kernel.arms import Landscape          # noqa: E402
from s0_kernel.harness import Demography, S0World   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # R98：GBK 控制台兜底
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

#: 六档：S0 正式规模档＋隔离用的未封顶档（rows/cols 合计 28,800 格＝120×240，草案 §五 同一尺度）
PRESETS = [
    {"name": "s0_formal", "n_ind0": 500, "rows": 120, "cols": 240, "max_ind": 2000,
     "ticks": 300, "arm": "A", "alpha_W": None},
    {"name": "s0_formal_B", "n_ind0": 500, "rows": 120, "cols": 240, "max_ind": 2000,
     "ticks": 300, "arm": "B", "alpha_W": 1.0},
    {"name": "s0_z2_B", "n_ind0": 500, "rows": 120, "cols": 240, "max_ind": 2000,
     "ticks": 300, "arm": "B", "alpha_W": 0.0},
    {"name": "uncapped_small", "n_ind0": 60, "rows": 20, "cols": 40, "max_ind": 60,
     "ticks": 400, "arm": "A", "alpha_W": None},
    {"name": "uncapped_small_B", "n_ind0": 60, "rows": 20, "cols": 40, "max_ind": 60,
     "ticks": 400, "arm": "B", "alpha_W": 1.0},
    {"name": "cap_probe", "n_ind0": 500, "rows": 120, "cols": 240, "max_ind": 5000,
     "ticks": 120, "arm": "A", "alpha_W": None},
]


def bench_one(p: dict, reps: int, seed: int) -> dict:
    """跑一档并返回 wall/tick 统计。preset 里出现的任何 `Demography` 字段都透传
    ⇒ 可以拿装置档做敏感性（如 `max_ind`／`lifespan`），不必改脚本。"""
    dem_kw = {k: v for k, v in p.items() if k in Demography.__dataclass_fields__}
    dem = Demography(**dem_kw)
    per_tick: list[float] = []
    wall_total = 0.0
    last: S0World | None = None
    for _ in range(reps):
        w = S0World(seed, p["arm"], p["alpha_W"], dem, land=Landscape())
        t0 = time.perf_counter()
        for _t in range(p["ticks"]):
            w.step()
            if w.pop <= 0:
                raise RuntimeError(f"微基准里灭绝了（{p['name']}）⇒ 装置档不合适，请换档而不是继续测")
        dt = time.perf_counter() - t0
        wall_total += dt
        per_tick.append(dt / p["ticks"] * 1000.0)
        last = w
    assert last is not None
    med = statistics.median(per_tick)
    return {
        "name": p["name"], "arm": p["arm"], "alpha_W": p["alpha_W"], "reps": reps,
        "n_ind0": p["n_ind0"], "cells": p["rows"] * p["cols"], "max_ind": p["max_ind"],
        "ticks_benched": p["ticks"],
        "ms_per_tick_median": round(med, 4),
        "ms_per_tick_worst": round(max(per_tick), 4),
        "pop_end": last.pop, "cap_hits": last.cap_hits, "capped": last.pop >= p["max_ind"],
        "wall_seconds_total": round(wall_total, 3),
        "extrapolated_20000t_seconds": round(med * 20000 / 1000.0, 1),
        "extrapolated_32run_serial_hours": round(med * 20000 * 32 / 1000.0 / 3600.0, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=207)
    ap.add_argument("--out", default="", help="本机 JSON 路径（默认不落盘，不进数据仓）")
    ap.add_argument("--mode", choices=("fresh", "inproc"), default="fresh",
                    help="fresh=每档起独立进程（机时口径，默认）；inproc=同进程连跑（会自污染，仅对比用）")
    ap.add_argument("--only", default=None, help="只跑某一档（配 --mode inproc 给子进程用）")
    ap.add_argument("--ticks", type=int, default=0, help="覆盖档位 tick（0=用档位值）")
    ap.add_argument("--quiet-json", action="store_true", help="只打印单档 JSON（子进程内部用）")
    args = ap.parse_args()
    if args.reps < 1:
        raise RuntimeError("reps ≥ 1")
    if args.ticks < 0:
        raise RuntimeError("--ticks ≥ 0")

    if args.mode == "fresh":
        rows = []
        for preset in PRESETS:
            cmd = [sys.executable, str(Path(__file__).resolve()), "--mode", "inproc",
                   "--reps", "1", "--seed", str(args.seed), "--only", preset["name"],
                   "--quiet-json", "--ticks", str(preset["ticks"])]
            samples = []
            for _ in range(args.reps):
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=1800)
                if r.returncode != 0:
                    raise RuntimeError(f"fresh 子进程失败（{preset['name']}）：{r.stderr[-400:]}")
                samples.append(json.loads(r.stdout.strip().splitlines()[-1]))
            med = statistics.median([x["ms_per_tick_median"] for x in samples])
            row = dict(min(samples, key=lambda x: abs(x["ms_per_tick_median"] - med)))
            row.update({"ms_per_tick_median": round(med, 4),
                        "ms_per_tick_worst": round(max(x["ms_per_tick_median"] for x in samples), 4),
                        "mode": "fresh", "n_procs": len(samples)})
            row["extrapolated_20000t_seconds"] = round(med * 20000 / 1000.0, 1)
            row["extrapolated_32run_serial_hours"] = round(med * 20000 * 32 / 1000.0 / 3600.0, 2)
            rows.append(row)
    else:
        picked = [p for p in PRESETS if args.only in (None, p["name"])] if args.only else PRESETS
        if args.ticks:
            picked = [dict(p, ticks=args.ticks) for p in picked]
        rows = [bench_one(p, args.reps, args.seed) for p in picked]
    if args.only and args.mode == "inproc" and len(rows) == 1 and args.quiet_json:
        print(json.dumps(rows[0], ensure_ascii=False))
        return 0
    print(f"=== S0-kernel-b 自测性能表（本机 {platform.machine()}，reps={args.reps}，"
          f"seed={args.seed}；[微基准，非跑批]）===")
    print("档 | 臂 | α_W | 格数 | max_ind | ms/tick中位 | ms/tick最差 | 末人口 | cap_hits | "
          "外推20000t秒 | 外推32run串行小时")
    for r in rows:
        print(f"{r['name']} | {r['arm']} | {r['alpha_W']} | {r['cells']} | {r['max_ind']} | "
              f"{r['ms_per_tick_median']} | {r['ms_per_tick_worst']} | {r['pop_end']} | "
              f"{r['cap_hits']} | {r['extrapolated_20000t_seconds']} | "
              f"{r['extrapolated_32run_serial_hours']}")

    payload = {
        "tool": "tools/s0_selftest_perf.py",
        "nature": "本机微基准（Pre-Flight wall/tick 真值）——非跑批、未入数据仓（R117 无涉）",
        "what_is_measured": "s0_kernel.harness.S0World.step()（S0 双臂容器），"
                            "与主引擎＋步A 探针链路是两条码路，禁互推",
        "env": {"python": sys.version.split()[0], "platform": platform.platform(),
                "machine": platform.machine()},
        "reps": args.reps, "seed": args.seed, "rows": rows,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                                  encoding="utf-8", newline="\n")
        print(f"\n[本机落盘] {args.out}（未进数据仓；入仓与否由砚/PI 裁）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
