"""世界读 GCM 表 + NPP 斑块：一次冒烟跑（A16b / A16c 共用入口）。

三种模式（互斥）：
  * `--table <npz>`：注入 `GCMTableSource`（世界用 GCM 的温度；降水由表提供，但世界**尚未消费**）
  * `--npp`：开 NPP 派生容量/再生（`distribution="npp"`，需要 `env_field.enabled`）
  * `--patches N`：旧口径对照（`distribution="patchy"`, patch_count=N, bg_production_zero）

输出 JSON（stdout）：墙钟、种群、容量统计、NPP/斑块统计、气候源 probe。
**不接生物**（用默认初始种群，跑 N tick 只为看世界侧读数）。

用法：
    python experiments/run_world_gcm_smoke.py --device s1 --table <npz> --ticks 20
    python experiments/run_world_gcm_smoke.py --device s2 --npp --ticks 2000
    python experiments/run_world_gcm_smoke.py --device s2 --patches 1700 --ticks 2000
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
    EnvFieldConfig, NPPConfig, PopulationConfig, ResourceConfig, SimConfig, WorldConfig,
)
from simulation.sphere_engine import SphereEngine  # noqa: E402
from tools.device_presets import DEVICE_PRESETS  # noqa: E402
from world.climate import GCMTableSource  # noqa: E402

TIER = "full"          # 2000 tick 档属长跑，跑位不进档②


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="s1")
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--cols", type=int, default=None)
    ap.add_argument("--ticks", type=int, default=20)
    ap.add_argument("--pop", type=int, default=2000)
    ap.add_argument("--table", default=None)
    ap.add_argument("--year-ticks", type=int, default=87600)
    ap.add_argument("--npp", action="store_true")
    ap.add_argument("--patches", type=int, default=0, help="旧口径：patch_count（0=不用）")
    ap.add_argument("--c0b", action="store_true",
                    help="开 C0b 海陆掩码（有海洋 ⇒ 容量在海洋为 0；A17.2 生产口径必开）")
    ap.add_argument("--patch-keep-bg", action="store_true",
                    help="patchy 档保留背景产能（= 与 uniform 同总容量预算；默认关=旧 A16c 口径）")
    ap.add_argument("--save-json", default=None)
    args = ap.parse_args()

    preset = DEVICE_PRESETS.get(args.device) if args.device else None
    rows = args.rows or (int(preset["rows"]) if preset else 120)
    cols = args.cols or (int(preset["cols"]) if preset else 240)

    #   `--c0b` 也要求构造 EnvField（否则海陆掩码不存在 ⇒ uniform 档没有海洋零容量）
    env = EnvFieldConfig(enabled=bool(args.npp or args.patches or args.c0b),
                         c0b_enabled=bool(args.c0b),
                         npp=NPPConfig(enabled=bool(args.npp)))
    if args.patches:
        res = ResourceConfig(distribution="patchy", patch_count=int(args.patches),
                             bg_production_zero=not bool(args.patch_keep_bg))
    elif args.npp:
        res = ResourceConfig(distribution="npp")
    else:
        res = ResourceConfig()

    cfg = SimConfig(world=WorldConfig(rows=rows, cols=cols), env_field=env, resources=res,
                    population=PopulationConfig(initial_count=int(args.pop),
                                                max_count=max(int(args.pop), 5000)))
    src = None
    if args.table:
        src = GCMTableSource(Path(args.table), year_ticks=int(args.year_ticks))

    t0 = time.time()
    engine = SphereEngine(cfg, climate_source=src)
    t_construct = time.time() - t0

    t1 = time.time()
    for i in range(int(args.ticks)):
        engine.step()
    t_run = time.time() - t1

    out: dict = {
        "device": args.device, "grid": [rows, cols], "ticks": int(args.ticks),
        "construct_s": round(t_construct, 2), "run_s": round(t_run, 2),
        "ms_per_tick": round(1000.0 * t_run / max(args.ticks, 1), 2),
        "pop_alive": int(engine.alive_count()),
        "mode": ("gcm_table" if args.table else ("npp" if args.npp else
                 ("patchy" if args.patches else "uniform"))),
    }
    # 气候源读数（表源：T 的全球范围；env 源：probe）
    try:
        t_now = engine.climate.temperature_c(engine._flat, engine._tick)
        out["climate_temperature_c"] = {"min": round(float(np.min(t_now)), 2),
                                        "max": round(float(np.max(t_now)), 2),
                                        "mean": round(float(np.mean(t_now)), 2)}
    except Exception as exc:  # noqa: BLE001 - 只为诊断，不掩盖主流程
        out["climate_temperature_c"] = f"N/A ({type(exc).__name__})"
    if src is not None:
        out["climate_probe"] = src.probe()

    # 容量统计（世界侧）
    cap = engine.resources._capacity if hasattr(engine.resources, "_capacity") else None
    if cap is not None:
        cap = np.asarray(cap, dtype=np.float64).ravel()
        out["capacity"] = {"sum": round(float(np.sum(cap)), 1),
                           "p50": round(float(np.percentile(cap, 50)), 3),
                           "p99": round(float(np.percentile(cap, 99)), 3),
                           "zero_frac": round(float(np.mean(cap <= 0.0)), 4)}
        lm = getattr(engine.env_field, "land_mask", None) if engine.env_field is not None else None
        if lm is not None:
            lm = np.asarray(lm, dtype=bool).ravel()
            out["capacity"].update({
                "land_frac": round(float(lm.mean()), 4),
                "land_sum": round(float(np.sum(cap[lm])), 1),
                "land_zero_frac": round(float(np.mean(cap[lm] <= 0.0)), 4),
                "ocean_zero_frac": round(float(np.mean(cap[~lm] <= 0.0)), 4),
            })
    if engine.env_field is not None:
        pr = engine.env_field.probe()
        keep = ("npp_on", "npp_p50", "npp_p99", "npp_max", "n_patches",
                "patch_mode", "patch_cover_frac", "patch_cover_frac_land",
                "patch_max_share", "patch_area_cells_p50", "patch_area_cells_max",
                "patch_radius_deg_mean", "patch_radius_km_mean", "sea_level_m",
                "land_area_frac", "pr_source", "pr_corr_proxy", "pr_pattern_rel_dev",
                "pr_scale", "pr_policy", "pr_runoff_corr", "pr_runoff_ratio",
                "river_frac", "lake_count", "river_cells_c0b", "flood_filled")
        out["env"] = {k: pr[k] for k in keep if k in pr}
        pm = getattr(engine.resources, "_patch_mask", None)
        if pm is not None:
            pm = np.asarray(pm, dtype=bool).ravel()
            item = {"cells": int(np.sum(pm)), "cover_frac": round(float(np.mean(pm)), 4)}
            lm2 = getattr(engine.env_field, "land_mask", None)
            if lm2 is not None:
                lm2 = np.asarray(lm2, dtype=bool).ravel()
                item["cover_frac_land"] = round(float(np.mean(pm[lm2])), 4)
            out["legacy_patch_mask"] = item

    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.save_json:
        Path(args.save_json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
