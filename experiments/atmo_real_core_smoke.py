"""A3 真核集成冒烟：真实 `atmo_core` ↔ `ModuleAtmoCore`（**不是天气结论**）。

范围（与总审追加任务一致）
--------------------------
1. `contract()/grid_spec()/make_state()/digest()/export(st,g)` 真实扩展路径；
2. `ac.step`（W2 原始方程）在 dt=288 下跑 N 步：无 NaN、可复现、两次逐位一致；
3. INTERFACE v1.0.3 的 M2′ 时钟：tick=36 ⇒ k=8、dt_step=288、109,500 交换/年；
4. `shallow_water_step` 独立入口冒烟；
5. **边界消费检查**：同一初值、两组截然不同的 sst/mask/h/alb/seaice，若 digest 相同
   ⇒ W2 的 `step` 尚未消费世界边界（如实记录，不假装已接通）；
6. `CouplingBridge.exchange(0)` 的真实核路径：能推进 step，但因 A1 `export` 无 `precip`
   而显式 `ValueError`（A2 湿物理未就绪；拒绝用 0 降水占位）。

🔴 不构造完整天气引擎、不替换世界温度/降水；本脚本只证适配器与真实扩展能对接。

运行（仓库内任意 cwd；产物落主工作树 `_rerun_logs/20261020-A3/`）：
    .venv/Scripts/python.exe experiments/atmo_real_core_smoke.py --steps 300
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atmo.coupling import AtmoConfig, CouplingClock, ModuleAtmoCore  # noqa: E402
from atmo.coupling.bridge import CouplingBridge  # noqa: E402
from world.sphere_world import SphereWorld  # noqa: E402

_ORBIT = {
    "year_s": 31536000.0,
    "rotation_period_s": 86400.0,
    "obliquity_rad": 0.0,
    "eccentricity": 0.0167,
    "equinox_offset_s": 0.0,
    "solar_constant_w_m2": 1361.0,
    "radius_m": 6371000.0,
}


def _main_root() -> Path:
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
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, encoding="utf-8", check=True)
        return out.stdout.strip()
    except Exception:                                  # noqa: BLE001
        return "unknown"


def _sha_arrays(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()


def _stats(a: np.ndarray) -> dict[str, float]:
    x = np.asarray(a, dtype=np.float64)
    return {"min": float(x.min()), "mean": float(x.mean()),
            "max": float(x.max()), "std": float(x.std())}


def _shape_map(state) -> dict[str, list[int]]:
    return {k: list(np.asarray(v).shape) for k, v in state.items()}


def _nan_total(state) -> int:
    return int(sum(np.asarray(v, dtype=np.float64).size -
                   np.isfinite(np.asarray(v, dtype=np.float64)).sum()
                   for v in state.values()))


def _run_steps(core, state, steps: int, dt: float) -> dict:
    t0 = time.perf_counter()
    last = None
    nan_steps = 0
    for _ in range(steps):
        last = core.step(state, {}, dt)
        nan_steps += int(last.get("nan_count", 0))
    elapsed = time.perf_counter() - t0
    return {"steps": steps, "elapsed_s": round(elapsed, 4),
            "ms_per_step": round(1000.0 * elapsed / steps, 4),
            "nan_steps_sum": nan_steps, "last_diag": dict(last)}


def _boundary(rows: int, cols: int) -> dict:
    lat = np.degrees(np.linspace(-90.0, 90.0, rows + 1)[:-1] + 90.0 / rows)
    land = np.where(np.abs(lat) < 40.0, 1.0, 0.0)[:, None]
    mask = np.broadcast_to(land, (rows, cols)).copy()
    sst = (300.0 - 40.0 * np.sin(np.radians(lat)) ** 2)[:, None]
    return {
        "h": np.zeros((rows, cols)),
        "mask": mask,
        "alb": np.where(mask > 0.5, 0.17, 0.06) * np.ones((rows, cols)),
        "sst": np.broadcast_to(sst, (rows, cols)).copy(),
        "ocean": np.broadcast_to(1.0 - land, (rows, cols)).copy(),
        "seaice": np.zeros((rows, cols)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="A3 real atmo_core adapter smoke")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--dt", type=float, default=288.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    try:
        import atmo_core as ac
    except ImportError as exc:                        # pragma: no cover - 环境门
        raise SystemExit(
            "atmo_core 未安装；先按 `内部团队文档/身份-A1.md` F7/F8 配方 "
            "`maturin develop --release`") from exc

    cfg = AtmoConfig(enabled=True, weather_source="atmo", dt_atmo=args.dt,
                     nlat=32, nlon=64, nlev=8, seed=42)
    core = ModuleAtmoCore(ac, cfg)

    grid = core.grid_spec()
    required = ("nlat", "nlon", "nlev", "lat_deg", "lon_deg", "area", "mu_edges")
    missing = [k for k in required if k not in grid]
    if missing:
        raise RuntimeError(f"grid_spec 缺键 {missing}（INTERFACE v1.0.1 起必须给 mu_edges）")

    state = core.make_state()
    digest0 = core.digest(state)
    clock = CouplingClock(36.0, args.dt)

    # ---- 1) 300 步 + 出口场 ----
    run1 = _run_steps(core, state, args.steps, clock.dt_step)
    digest1 = core.digest(state)
    exports = core.export(state)
    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in exports.items()
              if k not in ("approx", "synthetic", "placeholder")}
    export_stats = {k: _stats(v) for k, v in arrays.items()}
    nan_state = _nan_total(state)

    # ---- 2) 同初值、同配置第二次 ⇒ digest / export 逐位一致 ----
    state2 = core.make_state()
    run2 = _run_steps(core, state2, args.steps, clock.dt_step)
    digest2 = core.digest(state2)
    exports2 = core.export(state2)
    bitwise = (digest1 == digest2) and all(
        np.array_equal(arrays[k], np.asarray(exports2[k], dtype=np.float64))
        for k in arrays)

    # ---- 3) 边界消费检查：两组截然不同的 sst/mask/h/alb/seaice ----
    base = _boundary(grid["nlat"], grid["nlon"])
    forcing_a = dict(base)
    forcing_b = {k: (np.asarray(v, dtype=np.float64) * 0.0 + 1.0)
                 if k != "sst" else np.full_like(np.asarray(v), 320.0)
                 for k, v in base.items()}
    forcing_a["sst"] = np.full_like(np.asarray(base["sst"]), 250.0)
    forcing_b["sst"] = np.full_like(np.asarray(base["sst"]), 320.0)
    forcing_a.update({"orbit": _ORBIT, "tick": 0, "time_s": 0.0,
                      "dt_step_s": clock.dt_step, "synthetic": False})
    forcing_b.update({"orbit": _ORBIT, "tick": 0, "time_s": 0.0,
                      "dt_step_s": clock.dt_step, "synthetic": False})
    sa = core.make_state()
    sb = core.make_state()
    core.step(sa, forcing_a, clock.dt_step)
    core.step(sb, forcing_b, clock.dt_step)
    boundary_identical = (core.digest(sa) == core.digest(sb))

    # ---- 4) 真实核 + CouplingBridge：step 能跑，但 export 缺 precip ⇒ fail-loud ----
    world = SphereWorld(24, 48)
    bridge = CouplingBridge(
        world, cfg, core, 36.0,
        boundary_fn=lambda: _boundary(24, 48), orbit=_ORBIT)
    bridge_digest0 = bridge.initial_digest
    bridge_error = None
    try:
        bridge.exchange(0)
    except Exception as exc:                          # noqa: BLE001 - 负结果要记录
        bridge_error = f"{type(exc).__name__}: {exc}"
    bridge_digest1 = core.digest(bridge.state)

    # ---- 5) shallow_water_step 独立入口 ----
    sw = {
        "u1": np.zeros((grid["nlat"], grid["nlon"])),
        "v1": np.zeros((grid["nlat"], grid["nlon"])),
        "h1": np.full((grid["nlat"], grid["nlon"]), 300.0),
        "u2": np.zeros((grid["nlat"], grid["nlon"])),
        "v2": np.zeros((grid["nlat"], grid["nlon"])),
        "h2": np.full((grid["nlat"], grid["nlon"]), 3700.0),
    }
    sw_before = _sha_arrays(*sw.values())
    sw_last = None
    sw_nan = 0
    for _ in range(10):
        sw_last = ac.shallow_water_step(sw, grid, clock.dt_step)
        sw_nan += int(sw_last.get("nan_count", 0))
    sw_after = _sha_arrays(*sw.values())

    result = {
        "meta": {
            "commit": _git_head(),
            "script": "experiments/atmo_real_core_smoke.py",
            "extension_file": str(ac.__file__),
            "extension_interface_version": ac.INTERFACE_VERSION,
            "doc_interface_version": "1.0.3",
            "synthetic": False,
            "note": "真核适配器冒烟；未接 A2 降水、未替换世界温度/降水 ⇒ 不是天气结论",
            "dt_atmo_s": args.dt,
            "tick_seconds": 36.0,
        },
        "contract": dict(ac.contract()),
        "grid": {
            "keys": sorted(grid.keys()),
            "nlat": int(grid["nlat"]), "nlon": int(grid["nlon"]),
            "nlev": int(grid["nlev"]),
            "area_sum": float(np.sum(grid["area"])),
            "mu_edges_len": int(np.asarray(grid["mu_edges"]).size),
            "lat_first_last": [float(grid["lat_deg"][0]), float(grid["lat_deg"][-1])],
        },
        "clock": {
            "k": clock.k, "dt_step_s": clock.dt_step, "n_sub": clock.n_sub,
            "dt_remainder_s": clock.dt_remainder,
            "exchanges_per_year": clock.exchanges_per_year(31536000.0),
            "year_ticks": int(31536000 // 36),
        },
        "state": {
            "shapes": _shape_map(state),
            "digest_initial": digest0,
            "digest_after": digest1,
            "nan_total": nan_state,
        },
        "steps_run1": run1,
        "steps_run2": run2,
        "determinism": {
            "digest_run2": digest2,
            "digest_equal": digest1 == digest2,
            "exports_bitwise_equal": bool(bitwise),
        },
        "exports": {
            "keys": sorted(exports.keys()),
            "stats": export_stats,
            "approx": bool(exports.get("approx", False)),
        },
        "boundary_consumption": {
            "identical_digest_with_different_boundary": bool(boundary_identical),
            "interpretation": ("W2 step 只读 grid/held_suarez/filter；世界边界字段未消费"
                               if boundary_identical else "边界字段进入了 step（需复核）"),
        },
        "bridge_exchange": {
            "initial_digest": bridge_digest0,
            "digest_after_step": bridge_digest1,
            "step_advanced": bridge_digest0 != bridge_digest1,
            "error": bridge_error,
            "expected": "ValueError: core.export 缺 'precip'（A2 未就绪；拒绝 0 降水占位）",
        },
        "shallow_water": {
            "steps": 10, "nan_steps_sum": sw_nan,
            "digest_before": sw_before, "digest_after": sw_after,
            "last_diag": dict(sw_last) if sw_last else None,
        },
    }

    out = args.out or (_main_root() / "_rerun_logs" / "20261020-A3" /
                       "atmo_real_core_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float),
                   encoding="utf-8")

    summary = {
        "extension": result["meta"]["extension_interface_version"],
        "clock": result["clock"],
        "steps": run1,
        "determinism": result["determinism"],
        "t2m": export_stats.get("t2m"),
        "boundary_consumed": not boundary_identical,
        "bridge_error": bridge_error,
        "shallow_nan": sw_nan,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    print(f"[A3-real] written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
