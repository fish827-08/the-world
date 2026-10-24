"""A20.1 季节动态 NPP 演示：一年（可压缩年长）里 NPP / 斑块 / 容量的月度变化。

口径
----
* 气候源 = 30 年表的 **12 月层**（`exopl_at21_earth_v0_monthly.npz`），`year_ticks` 可压缩（演示用 1200）。
* `NPPConfig(dynamic=True)`：每 `update_every` tick 用 `climate.temperature_c(tick)` 重算 NPP，
  再把 NPP 作为容量/再生因子热更新到资源场。
* 季节**只由温度驱动**（降水用年平，见 `EnvField.refresh_npp_from_climate` 的说明）。

判据（派司口径 "季节曲线可读"）
------------------------------
热带（|lat|<10）双峰、中纬（40–60）单峰——本脚本只**打印**，不设硬判据。

用法
----
    python experiments/seasonal_npp_demo.py --device s2 --year-ticks 1200 --samples 12
"""
from __future__ import annotations

import argparse
import json
import subprocess
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

TIER = "full"


def _git_head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:      # noqa: BLE001
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="s2")
    ap.add_argument("--table", required=True)
    ap.add_argument("--year-ticks", type=int, default=1200)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--ticks", type=int, default=None, help="默认 = year_ticks（跑一整年）")
    ap.add_argument("--update-every", type=int, default=10)
    ap.add_argument("--pop", type=int, default=10000)
    ap.add_argument("--save-json", default=None)
    ap.add_argument("--save-csv", default=None)
    args = ap.parse_args()

    preset = DEVICE_PRESETS.get(args.device) if args.device else None
    rows = int(preset["rows"]) if preset else 120
    cols = int(preset["cols"]) if preset else 240
    ticks = int(args.ticks or args.year_ticks)

    cfg = SimConfig(
        seed=42,
        world=WorldConfig(rows=rows, cols=cols),
        env_field=EnvFieldConfig(enabled=True, c0b_enabled=True,
                                 npp=NPPConfig(enabled=True, dynamic=True,
                                               update_every=int(args.update_every))),
        resources=ResourceConfig(distribution="npp"),
        population=PopulationConfig(initial_count=int(args.pop),
                                    max_count=max(int(args.pop), 5000)),
    )
    src = GCMTableSource(Path(args.table), year_ticks=int(args.year_ticks))
    t0 = time.time()
    eng = SphereEngine(cfg, climate_source=src)
    t_build = time.time() - t0

    lat = np.degrees(np.asarray(eng.world._lat, dtype=np.float64)).ravel()
    # 纬度是**每行一个**，NPP 是**每格一个** ⇒ 必须广播到格（否则布尔索引长度不匹配）
    def _cells(mask_row: np.ndarray) -> np.ndarray:
        return np.repeat(np.asarray(mask_row, dtype=bool), eng.world.cols)
    trop = _cells(np.abs(lat) <= 10.0)
    midn = _cells((lat >= 40.0) & (lat <= 60.0))
    mids = _cells((lat <= -40.0) & (lat >= -60.0))

    samples = []
    step = max(ticks // int(args.samples), 1)
    t1 = time.time()
    for i in range(ticks):
        eng.step()
        if (i + 1) % step == 0 or i == ticks - 1:
            pr = eng.env_field.probe()
            npp = np.asarray(eng.env_field.npp, dtype=np.float64)
            phase = (i + 1) / float(args.year_ticks) * 12.0
            samples.append({
                "tick": i + 1,
                "month": round(phase % 12.0, 2),
                "npp_p50": round(float(np.percentile(npp, 50)), 4),
                "npp_p99": round(float(np.percentile(npp, 99)), 4),
                "npp_tropic": round(float(npp[trop].mean()), 4),
                "npp_mid_north": round(float(npp[midn].mean()), 4),
                "npp_mid_south": round(float(npp[mids].mean()), 4),
                "capacity_sum": round(float(np.asarray(eng.resources._capacity).sum()), 1),
                "n_patches": pr.get("n_patches"),
                "patch_cover_frac_land": pr.get("patch_cover_frac_land"),
                "pop_alive": int(eng.alive_count()),
            })
    wall = time.time() - t1

    def amp(key: str) -> float:
        v = [s[key] for s in samples if s[key] is not None]
        return round(max(v) - min(v), 4) if v else 0.0

    out = {
        "git_head": _git_head(), "device": args.device, "grid": [rows, cols],
        "table": str(args.table), "year_ticks": int(args.year_ticks), "ticks": ticks,
        "build_s": round(t_build, 2), "run_s": round(wall, 1),
        "ms_per_tick": round(1000.0 * wall / max(ticks, 1), 2),
        "npp_amplitude": {k: amp(k) for k in ("npp_tropic", "npp_mid_north", "npp_mid_south",
                                             "npp_p50", "npp_p99", "capacity_sum",
                                             "n_patches", "patch_cover_frac_land")},
        "samples": samples,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "samples"},
                     ensure_ascii=False, indent=2))
    for s in samples:
        print(f"  t={s['tick']:5d} m={s['month']:5.2f} npp50={s['npp_p50']:.3f} "
              f"trop={s['npp_tropic']:.3f} midN={s['npp_mid_north']:.3f} "
              f"midS={s['npp_mid_south']:.3f} cap={s['capacity_sum']:.0f} "
              f"patches={s['n_patches']} pop={s['pop_alive']}")
    if args.save_json:
        Path(args.save_json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    if args.save_csv:
        keys = list(samples[0])
        lines = [",".join(keys)] + [",".join(str(s[k]) for k in keys) for s in samples]
        Path(args.save_csv).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
