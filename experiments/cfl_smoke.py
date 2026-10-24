"""A19.2 CFL 冒烟：把大气子步 `dt_s` 从 120 s 放大到 1200 s（= 世界切 360 s/tick 的配套改动）。

要回答的问题
------------
"世界 1 tick 从 36 s 变 360 s"要求大气子步同比放大（120 → 1200 s）。放大 10 倍后
迎风水汽平流的 CFL 数 ν = u·Δt/Δx 也会涨 ≈10 倍；本脚本只测**稳定不稳定**，不改默认。

判据（派工口径）
----------------
* ν_max < 1（硬；超过 `_CFL_STABILITY_MAX` 引擎会直接抛）
* dt/τ < 0.5（阻尼/辐射的时间分辨率）
* 无 NaN / 无温度越界

用法
----
    python experiments/cfl_smoke.py --device s1 --ticks 200 --dt-list 120,1200
    python experiments/cfl_smoke.py --device s2 --ticks 200 --dt-list 1200 --substeps 30

产物：stdout JSON（每档一行 + 汇总）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import (  # noqa: E402
    AtmosphereConfig, EnvFieldConfig, LightConfig, SimConfig, WorldConfig,
)
from simulation.sphere_engine import SphereEngine  # noqa: E402
from tools.device_presets import DEVICE_PRESETS  # noqa: E402

TIER = "full"
NU_MAX = 1.0
DT_TAU_MAX = 0.5
T_FLOOR_C, T_CEIL_C = -150.0, 120.0


def run_one(rows: int, cols: int, ticks: int, dt_s: float, substeps: int,
            every_ticks: int, seed: int) -> dict:
    # 🔴 世界 1 tick 的**物理秒数** = substeps×dt_s/every_ticks（v38/P0.3 汇率）；
    #   config 会断言它等于 86400/rotation_period ⇒ 放大 dt_s 必须同步改 rotation_period。
    sec_per_tick = float(substeps) * float(dt_s) / float(every_ticks)
    rotation = 86400.0 / sec_per_tick
    cfg = SimConfig(
        seed=int(seed),
        world=WorldConfig(rows=rows, cols=cols),
        light=LightConfig(rotation_period=int(round(rotation))),
        env_field=EnvFieldConfig(enabled=True, c0b_enabled=True),
        atmosphere=AtmosphereConfig(
            enabled=True, dt_s=float(dt_s), substeps=int(substeps),
            every_ticks=int(every_ticks), moisture_on=True, coriolis_on=False,
        ),
    )
    cfg.simulation.use_sim_core = False
    rec: dict = {"dt_s": float(dt_s), "substeps": int(substeps),
                 "every_ticks": int(every_ticks),
                 "sec_per_tick": round(sec_per_tick, 3),
                 "rotation_period": int(round(rotation))}
    t0 = time.time()
    try:
        eng = SphereEngine(cfg)
        for _ in range(int(ticks)):
            eng.step()
        pr = eng.atmosphere.probe()
        t_now = np.asarray(eng.atmosphere.temp_c if hasattr(eng.atmosphere, "temp_c")
                           else eng.env_field.temp, dtype=np.float64)
        rec.update({
            "ok": True,
            "wall_s": round(time.time() - t0, 1),
            "cfl_max": round(float(pr.get("cfl_max", float("nan"))), 4),
            "dt_over_tau": round(float(pr.get("dt_over_tau", float("nan"))), 4),
            "t_min_c": round(float(np.nanmin(t_now)), 2),
            "t_max_c": round(float(np.nanmax(t_now)), 2),
            "nan_cells": int(np.sum(~np.isfinite(t_now))),
        })
    except Exception as exc:                      # noqa: BLE001 - 稳定性测试要抓"炸"
        rec.update({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                    "wall_s": round(time.time() - t0, 1)})
    cfl = rec.get("cfl_max")
    rec["pass_cfl"] = bool(cfl is not None and np.isfinite(cfl) and cfl < NU_MAX)
    rec["pass_dt_tau"] = bool(np.isfinite(rec.get("dt_over_tau", np.nan))
                              and rec["dt_over_tau"] < DT_TAU_MAX)
    rec["pass_nan"] = bool(rec.get("nan_cells", 1) == 0)
    rec["pass_range"] = bool(T_FLOOR_C <= rec.get("t_min_c", 1e9)
                             and rec.get("t_max_c", -1e9) <= T_CEIL_C)
    rec["pass_all"] = bool(rec["ok"] and rec["pass_cfl"] and rec["pass_dt_tau"]
                           and rec["pass_nan"] and rec["pass_range"])
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="s1")
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--cols", type=int, default=None)
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--dt-list", default="120,1200")
    ap.add_argument("--substeps", type=int, default=3)
    ap.add_argument("--every-ticks", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-json", default=None)
    args = ap.parse_args()

    preset = DEVICE_PRESETS.get(args.device) if args.device else None
    rows = args.rows or (int(preset["rows"]) if preset else 120)
    cols = args.cols or (int(preset["cols"]) if preset else 240)

    out = {"device": args.device, "grid": [rows, cols], "ticks": int(args.ticks),
           "seed": int(args.seed), "runs": []}
    for dt in [float(x) for x in args.dt_list.split(",") if x.strip()]:
        rec = run_one(rows, cols, int(args.ticks), dt, int(args.substeps),
                      int(args.every_ticks), int(args.seed))
        out["runs"].append(rec)
        print(f"[dt={dt:g}s] ok={rec['ok']} cfl={rec.get('cfl_max')} "
              f"dt/τ={rec.get('dt_over_tau')} T=[{rec.get('t_min_c')},{rec.get('t_max_c')}] "
              f"nan={rec.get('nan_cells')} pass={rec['pass_all']}", flush=True)
        if rec.get("error"):
            print("   error:", rec["error"], flush=True)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.save_json:
        Path(args.save_json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
