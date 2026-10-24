"""A20.4 标定第一轮：NPP 三形式 × 3 seed × 2000 tick（同世界 / 同装置）。

协议（`汇报-20261019-A19.md` §四 / 账本 H3）
------------------------------------------
只看**结论方向**是否一致；不一致 ⇒ 报"对形式敏感"、**不选边**。
禁止用存活数反推形式或参数。

用法
----
    python experiments/npp_form_calibration.py --device s2 --ticks 2000 --seeds 42,7,13
    python experiments/npp_form_calibration.py --device s1 --ticks 200 --seeds 42

产物：stdout JSON（含 `git_head` / `sea_level_mode` / 每臂的存活与容量分布）。
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

TIER = "full"
FORMS = ("multiplicative", "liebig", "light_temp_only")


def _git_head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:      # noqa: BLE001
        return ""


def run_arm(form: str, seed: int, rows: int, cols: int, ticks: int, pop: int) -> dict:
    cfg = SimConfig(
        seed=int(seed),
        world=WorldConfig(rows=rows, cols=cols),
        env_field=EnvFieldConfig(enabled=True, c0b_enabled=True,
                                 npp=NPPConfig(enabled=True, form=form)),
        resources=ResourceConfig(distribution="npp"),
        population=PopulationConfig(initial_count=int(pop), max_count=max(int(pop), 5000)),
    )
    t0 = time.time()
    eng = SphereEngine(cfg)
    pop_trace = []
    for i in range(int(ticks)):
        eng.step()
        if (i + 1) % max(ticks // 10, 1) == 0:
            pop_trace.append(int(eng.alive_count()))
    wall = time.time() - t0
    cap = np.asarray(eng.resources._capacity, dtype=np.float64).ravel()
    lm = np.asarray(eng.env_field.land_mask, dtype=bool).ravel()
    pr = eng.env_field.probe()
    return {
        "form": form, "seed": int(seed), "ticks": int(ticks),
        "wall_s": round(wall, 1),
        "pop_alive": int(eng.alive_count()),
        "pop_trace": pop_trace,
        "capacity_land_p50": round(float(np.percentile(cap[lm], 50)), 3),
        "capacity_land_p99": round(float(np.percentile(cap[lm], 99)), 3),
        "capacity_land_sum": round(float(cap[lm].sum()), 1),
        "land_frac": round(float(pr.get("land_area_frac", np.nan)), 4),
        "n_patches": pr.get("n_patches"),
        "patch_cover_frac_land": pr.get("patch_cover_frac_land"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="s2")
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--cols", type=int, default=None)
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--seeds", default="42,7,13")
    ap.add_argument("--pop", type=int, default=10000)
    ap.add_argument("--save-json", default=None)
    args = ap.parse_args()

    preset = DEVICE_PRESETS.get(args.device) if args.device else None
    rows = args.rows or (int(preset["rows"]) if preset else 120)
    cols = args.cols or (int(preset["cols"]) if preset else 240)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    out = {"git_head": _git_head(), "device": args.device, "grid": [rows, cols],
           "ticks": int(args.ticks), "pop": int(args.pop), "seeds": seeds,
           "arms": []}
    for form in FORMS:
        for seed in seeds:
            rec = run_arm(form, seed, rows, cols, int(args.ticks), int(args.pop))
            out["arms"].append(rec)
            print(f"[{form:17s} seed={seed}] pop={rec['pop_alive']:5d} "
                  f"cap_p50={rec['capacity_land_p50']:7.2f} "
                  f"patches={rec['n_patches']} wall={rec['wall_s']:.0f}s", flush=True)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.save_json:
        Path(args.save_json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
