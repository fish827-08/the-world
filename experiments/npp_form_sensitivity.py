"""A18.1 形式敏感性：NPP 三种形式在**同一世界 / 同一 seed**下的容量与斑块读数。

为什么要这个脚本（写死审计 H3）
--------------------------------
NPP 是"生产力定律"的手写关系。写哪一种形式，直接决定"哪里富饶"——世界族扫描的结论
可能来自公式选择而不是世界差异。本脚本只出读数，**不改默认形式**（默认仍是现状的
multiplicative，等第一轮生态读数再预注册）。

用法
----
    python experiments/npp_form_sensitivity.py --device s2 --ticks 300
    python experiments/npp_form_sensitivity.py --device s2 --ticks 300 --seed 7 \
        --save-json results/npp_forms_s2.json

产物：stdout JSON（每形式一行）+ 可选落盘。
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

TIER = "full"
FORMS = ("multiplicative", "liebig", "light_temp_only")


def run_one(form: str, rows: int, cols: int, ticks: int, seed: int, pop: int) -> dict:
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
    t_build = time.time() - t0
    t1 = time.time()
    for _ in range(int(ticks)):
        eng.step()
    t_run = time.time() - t1

    cap = np.asarray(eng.resources._capacity, dtype=np.float64).ravel()
    lm = np.asarray(eng.env_field.land_mask, dtype=bool).ravel()
    pr = eng.env_field.probe()
    return {
        "form": form,
        "ticks": int(ticks),
        "build_s": round(t_build, 2),
        "run_s": round(t_run, 2),
        "pop_alive": int(eng.alive_count()),
        "capacity": {
            "sum": round(float(cap.sum()), 1),
            "land_sum": round(float(cap[lm].sum()), 1),
            "land_p50": round(float(np.percentile(cap[lm], 50)), 3),
            "land_p90": round(float(np.percentile(cap[lm], 90)), 3),
            "land_p99": round(float(np.percentile(cap[lm], 99)), 3),
            "land_max": round(float(cap[lm].max()), 3),
            "land_zero_frac": round(float(np.mean(cap[lm] <= 0.0)), 4),
            "land_gini": round(_gini(cap[lm]), 4),
        },
        "sea_level_mode": str(eng.config.env_field.sea_level.mode),
        "realized_land_frac": round(float(pr.get("land_area_frac", np.nan)), 4),
        "sea_level_m": round(float(pr.get("sea_level_m", np.nan)), 1),
        "npp": {
            "p50": round(float(pr.get("npp_p50", np.nan)), 4),
            "p99": round(float(pr.get("npp_p99", np.nan)), 4),
            "max": round(float(pr.get("npp_max", np.nan)), 4),
        },
        "patch": {k: pr[k] for k in pr if k.startswith("patch_") or k == "n_patches"},
    }


def _git_head() -> str:
    """当前 commit（**读数归属**：A19.1 的教训 —— 汇报必须能对上 commit）。"""
    import subprocess
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:      # noqa: BLE001 - 只用于记账
        return ""


def _gini(x: np.ndarray) -> float:
    """容量不平等度（0 = 完全均匀，1 = 全集中在一格）。"""
    v = np.sort(np.asarray(x, dtype=np.float64))
    n = v.size
    if n == 0 or v.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * v) / (n * v.sum())) - (n + 1.0) / n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="s2")
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--cols", type=int, default=None)
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pop", type=int, default=10000)
    ap.add_argument("--save-json", default=None)
    args = ap.parse_args()

    preset = DEVICE_PRESETS.get(args.device) if args.device else None
    rows = args.rows or (int(preset["rows"]) if preset else 120)
    cols = args.cols or (int(preset["cols"]) if preset else 240)

    out = {"git_head": _git_head(), "device": args.device, "grid": [rows, cols],
           "seed": int(args.seed),
           "ticks": int(args.ticks), "pop": int(args.pop),
           "note": "默认形式仍是 multiplicative（本脚本只出读数，不改默认）",
           "forms": []}
    for form in FORMS:
        out["forms"].append(run_one(form, rows, cols, int(args.ticks), int(args.seed),
                                    int(args.pop)))
        f = out["forms"][-1]
        print(f"[{form}] pop={f['pop_alive']} cap_sum={f['capacity']['sum']:.4g} "
              f"p50={f['capacity']['land_p50']} gini={f['capacity']['land_gini']} "
              f"patches={f['patch'].get('n_patches')} "
              f"cover_land={f['patch'].get('patch_cover_frac_land')}", flush=True)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.save_json:
        Path(args.save_json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
