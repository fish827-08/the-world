"""A3.5 联调 demo：s1（60×120）/ 200 tick，**合成** atmo 场。

只证链路：世界边界 → atmo 核（合成）→ 出口场 → 世界消费
（`climate.temperature_c` / `EnvField.pr` 空间型 / 动态 NPP）。
🔴 合成场不是天气结论；判读只看"世界确实吃到了动态场"。

运行（仓库根）：
    .venv/Scripts/python.exe experiments/atmo_coupling_demo.py --ticks 200
产物默认写 `_rerun_logs/20261020-A3/atmo_demo_s1.json`（不落仓库根）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atmo.coupling import AtmoConfig, SyntheticCore, make_weather_engine  # noqa: E402
from simulation.config import SimConfig  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402


def _config(seed: int) -> SimConfig:
    cfg = SimConfig(seed=seed)
    cfg.simulation.use_sim_core = False          # 天气模式硬前提（IC1/Rust 未接）
    cfg.env_field.enabled = True
    cfg.env_field.c0b_enabled = True             # h_m / land_mask / surface_type
    cfg.env_field.npp.enabled = True             # 让 t2m 消费点可观测（动态 NPP）
    cfg.env_field.npp.dynamic = True
    cfg.env_field.npp.update_every = 10
    return cfg


def _run_baseline(ticks: int, seed: int) -> SphereEngine:
    eng = SphereEngine(_config(seed))
    for _ in range(ticks):
        eng.step()
    return eng


def _run_weather(ticks: int, seed: int, acfg: AtmoConfig):
    eng, src = make_weather_engine(_config(seed), acfg, SyntheticCore(acfg))
    for _ in range(ticks):
        src.on_tick(eng.tick + 1)                # IC3 未批准 ⇒ 驱动显式推进交换
        eng.step()
    return eng, src


def _stats(a: np.ndarray) -> dict[str, float]:
    x = np.asarray(a, dtype=np.float64)
    return {"mean": float(x.mean()), "min": float(x.min()),
            "max": float(x.max()), "std": float(x.std())}


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _awm(field: np.ndarray, world) -> float:
    """面积权全球平均（keep_total 的判据必须用这把尺，不能用 unweighted mean）。"""
    f = np.asarray(field, dtype=np.float64).reshape(-1)
    a = np.asarray(world.cell_area(np.arange(f.size, dtype=np.int64)),
                   dtype=np.float64)
    return float(np.sum(f * a) / np.sum(a))


def _main_root() -> Path:
    """产物落**主工作树**的 `_rerun_logs/`（worktree 里跑也一样）。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"], cwd=ROOT,
            capture_output=True, text=True, encoding="utf-8", check=True)
        p = Path(out.stdout.strip())
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        return p.parent
    except Exception:                                  # noqa: BLE001
        return ROOT


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, encoding="utf-8", check=True)
        return out.stdout.strip()
    except Exception:                                  # noqa: BLE001 - 产物标注用
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description="A3.5 耦合联调 demo（合成 atmo 场）")
    ap.add_argument("--ticks", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    acfg = AtmoConfig(enabled=True, weather_source="atmo",
                      nlat=32, nlon=64, nlev=8, seed=args.seed)
    base = _run_baseline(args.ticks, args.seed)
    weather, src = _run_weather(args.ticks, args.seed, acfg)

    b_pr = np.asarray(base.env_field.pr, dtype=np.float64)
    w_pr = np.asarray(weather.env_field.pr, dtype=np.float64)
    b_npp = np.asarray(base.env_field.npp, dtype=np.float64)
    w_npp = np.asarray(weather.env_field.npp, dtype=np.float64)
    probe = src.probe()
    bridge = probe["bridge"]
    wm = src.manifest()

    result = {
        "meta": {
            "commit": _git_head(),
            "ticks": int(args.ticks),
            "grid": {"name": "s1", "rows": int(base.world.rows),
                     "cols": int(base.world.cols)},
            "seed": int(args.seed),
            "synthetic_atmo": True,
            "chain_proof_only": True,
            "note": "合成 atmo 场；不得据此下天气/生态结论",
        },
        "baseline": {
            "alive": int(base.alive_count()),
            "total_energy": float(base.population_view().total_energy),
            "pr_proxy": _stats(b_pr),
            "npp": _stats(b_npp),
            "pr_source": getattr(base.env_field, "pr_source", None),
            "atmosphere_enabled": base.atmosphere is not None,
        },
        "weather": {
            "alive": int(weather.alive_count()),
            "total_energy": float(weather.population_view().total_energy),
            "pr_proxy": _stats(w_pr),
            "npp": _stats(w_npp),
            "pr_source": getattr(weather.env_field, "pr_source", None),
            "pr_diag": getattr(weather.env_field, "pr_diag", None),
            "band_is_none": weather.env_field.band is None,
            "atmosphere_enabled": weather.atmosphere is not None,
            "t2m_c": {k: bridge[f"world_t2m_c_{k}"] for k in
                      ("min", "mean", "max") if f"world_t2m_c_{k}" in bridge},
            "precip_mm_yr": {
                "mean": bridge.get("world_precip_mm_yr_mean"),
                "max": bridge.get("world_precip_mm_yr_max"),
            },
            "exchanges": int(probe["bridge"]["exchanges"]),
            "t2m_calls": int(probe["t2m_calls"]),
            "precip_refresh_count": int(probe["precip_refresh_count"]),
            "clock": wm["time"],
            "manifest": wm,
        },
        "compare": {
            "pr_awm_baseline": _awm(b_pr, base.world),
            "pr_awm_weather": _awm(w_pr, weather.world),
            "pr_awm_delta": _awm(w_pr, weather.world) - _awm(b_pr, base.world),
            "pr_unweighted_mean_delta": float(w_pr.mean() - b_pr.mean()),
            "pr_pattern_corr": _corr(b_pr, w_pr),
            "npp_awm_ratio": float(_awm(w_npp, weather.world)
                                   / _awm(b_npp, base.world))
            if _awm(b_npp, base.world) != 0.0 else None,
            "npp_pattern_corr": _corr(b_npp, w_npp),
            "alive_delta": int(weather.alive_count() - base.alive_count()),
        },
    }

    out = Path(args.out) if args.out else (
        _main_root() / "_rerun_logs" / "20261020-A3" / "atmo_demo_s1.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["compare"], ensure_ascii=False, indent=2))
    print(f"[A3.5] written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
