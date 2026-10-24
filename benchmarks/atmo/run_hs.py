"""A1 atmo_core Held–Suarez 跑批器：产出 loader 契约的 run 目录（AV 裁判输入）。

口径
----
- 初值：HS94 平衡温度剖面 T_eq(φ,σ) + 静止 u=v=0 + ps0=100000 Pa；
  再叠加固定 seed 的确定性热力扰动（1 K，打破纬向对称，否则斜压涡不生长）。
- 强迫：``ac.step(state, forcing=None)`` 走 A1 PrimParams 默认 = HS94 标准
  （ΔTy=60, Δθz=10, σb=0.7, ka=1/40d, ks=1/4d, kf=1/d）+ 已声明偏差
  （海绵层 σ<0.1、nu=1e7、极区截断滤波；见 atmo_core/README.md §5.2，HS94 原文没有）。
- 输出：``meta.json`` + ``coords.npz`` + ``snap_*.npz``（loader 契约）+ ``diag.npz``
  （每快照 cfl/每日 mass/tmin/tmax；F9 mass 口径核查用）。
- 确定性：同 seed 两次运行逐位（CONS-DET 由套件 --run-dir-b 判）。

用法::

    python -m benchmarks.atmo.run_hs --out _rerun_logs/av-hs-a --days 120
    python -m benchmarks.atmo.run_hs --out _rerun_logs/av-hs-b --days 120   # 同 seed 第二跑
    python -m benchmarks.atmo.run_hs --probe-mass 300                        # F9 mass 口径探针
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

DAY_S = 86400.0

HS_FORCING_META = {
    "omega": 2.0 * math.pi / 86400.0,   # A1 grid：Ω = 2π/rotation_period_s
    "g": 9.81,
    "R": 287.0,
    "cp": 1004.0,
    "radius": 6371000.0,
    "p0": 100000.0,
    "delta_Ty": 60.0,
    "delta_theta_z": 10.0,
    "sigma_b": 0.7,
    "ka": 1.0 / (40.0 * DAY_S),
    "ks": 1.0 / (4.0 * DAY_S),
    "kf": 1.0 / DAY_S,
    # 已声明偏差（HS94 原文没有；A1 W2 实测必需，见 atmo_core/README.md §5.2）
    "nu": 1.0e7,
    "sponge_sigma": 0.10,
    "sponge_rate": 1.0 / (0.5 * DAY_S),
    "filter_mode": 2,
    "filter_cfl_target": 0.2,
}

# HS94_PARAMS 的判据对照键（不含已声明偏差键）
_HS94_COMPARE_KEYS = ("omega", "g", "R", "cp", "radius", "p0",
                      "delta_Ty", "delta_theta_z", "sigma_b", "ka", "ks", "kf")


def import_atmo_core(ext_dir: str | None):
    if ext_dir:
        sys.path.insert(0, str(ext_dir))
    import atmo_core as ac
    return ac


def hs_teq(mu: np.ndarray, sigma_full: np.ndarray, t0: float, d_ty: float,
           d_thz: float, kappa: float) -> np.ndarray:
    """与 A1 prim.rs 相同的 HS94 平衡温度剖面（σ 坐标）；返回 (nlev, nlat, 1) 可广播。"""
    cos2 = 1.0 - mu * mu
    teq = (t0 - d_ty * mu[None, :] * mu[None, :]
           - d_thz * np.log(sigma_full)[:, None] * cos2[None, :])
    teq = teq * (sigma_full ** kappa)[:, None]
    return np.maximum(teq, 200.0)[:, :, None]


def make_initial_state(ac, g, seed: int, ic: str = "iso_w1"):
    """ic="iso_w1"：等温静止 + 1K·sin(lon) 波数 1 光滑扰动（打破纬向对称，最温和的
    常规 HS 初值）；ic="teq_noise"：HS94 平衡剖面 + 1K 白噪声（实测 ~105 步即炸，
    仅留作对照臂）。"""
    if ic == "teq_noise":
        st = ac.make_state(g, T0=288.0, ps0=100000.0)
        mu = np.asarray(g["mu"], dtype=np.float64)
        sig = np.asarray(g["sigma_full"], dtype=np.float64)
        teq = hs_teq(mu, sig, 288.0, 60.0, 10.0, 287.0 / 1004.0)
        T = np.array(st["T"], dtype=np.float64, copy=True)
        T[:] = teq
        rng = np.random.default_rng(seed)
        T += 1.0 * rng.standard_normal(T.shape)
        st["T"] = np.ascontiguousarray(T)
        return st
    st = ac.make_state(g, T0=288.0, ps0=100000.0)
    T = np.array(st["T"], dtype=np.float64, copy=True)
    lon = np.asarray(g["lon_deg"], dtype=np.float64)
    T += 1.0 * np.sin(np.radians(lon))[None, None, :]
    st["T"] = np.ascontiguousarray(T)
    return st


def _save_snapshot(run: Path, idx: int, day: float, st, ac) -> str:
    np.savez_compressed(
        run / f"snap_{idx:05d}.npz",
        u=np.ascontiguousarray(st["u"], dtype=np.float64),
        v=np.ascontiguousarray(st["v"], dtype=np.float64),
        T=np.ascontiguousarray(st["T"], dtype=np.float64),
        ps=np.ascontiguousarray(st["ps"], dtype=np.float64),
    )
    return f"snap_{idx:05d}.npz"


def run_hs(out: str, days: float, dt: float, seed: int, snap_start_day: float,
           snap_every_day: float, ext_dir: str | None, ic: str = "iso_w1") -> dict:
    ac = import_atmo_core(ext_dir)
    g = ac.grid_spec(nlat=32, nlon=64, nlev=8)
    st = make_initial_state(ac, g, seed, ic)
    run = Path(out)
    run.mkdir(parents=True, exist_ok=True)

    steps_per_day = DAY_S / dt
    total_steps = int(round(days * steps_per_day))
    snap_days: list[float] = []
    snap_files: list[str] = []
    cfl_reported: list[float] = []
    mass_daily: list[float] = []
    tmin_daily: list[float] = []
    tmax_daily: list[float] = []
    t_start = time.time()
    next_snap = snap_start_day
    for step_i in range(1, total_steps + 1):
        diag = ac.step(st, None, dt)
        day = step_i / steps_per_day
        if step_i % int(round(steps_per_day)) == 0:
            mass_daily.append(float(diag["mass"]))
            tmin_daily.append(float(diag["tmin"]))
            tmax_daily.append(float(diag["tmax"]))
        if day >= next_snap - 1e-9:
            snap_days.append(round(day, 6))
            snap_files.append(_save_snapshot(run, len(snap_days), day, st, ac))
            cfl_reported.append(float(diag["cfl_max"]))
            next_snap += snap_every_day
        if step_i % 3000 == 0:
            el = time.time() - t_start
            print(f"  step {step_i}/{total_steps} day {day:.1f} "
                  f"cfl {diag['cfl_max']:.3f} tmin {diag['tmin']:.1f} "
                  f"({el:.0f}s, {el / step_i * 1000:.1f} ms/step)", flush=True)

    np.savez_compressed(
        run / "diag.npz",
        mass_daily=np.asarray(mass_daily), tmin_daily=np.asarray(tmin_daily),
        tmax_daily=np.asarray(tmax_daily), cfl_reported=np.asarray(cfl_reported),
    )
    config = {"nlat": 32, "nlon": 64, "nlev": 8, "kind": "gaussian",
              "dt_atmo_s": dt, "seed": seed, "forcing": "PrimParams::default (HS94)",
              "ext": "atmo_core atmo_core.pyd (cargo rustc release)"}
    meta = {
        "benchmark": "held_suarez",
        "dt_atmo_s": dt,
        "days": days,
        "snap_days": snap_days,
        "obliquity_deg": 0.0,
        "seed": seed,
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16],
        "cfl_reported": cfl_reported,
        "forcing": {k: HS_FORCING_META[k] for k in _HS94_COMPARE_KEYS},
        "declared_deviations": {k: HS_FORCING_META[k] for k in
                                ("nu", "sponge_sigma", "sponge_rate",
                                 "filter_mode", "filter_cfl_target")},
        "initial_state": (f"ic={ic}; " +
                          ("HS94 T_eq(φ,σ) + rest + 1K rng(default_rng(seed)) white noise" if ic == "teq_noise"
                           else "isothermal T0=288 rest + 1K·sin(lon) wavenumber-1 smooth perturbation")),
        "config": config,
        "snap_files": snap_files,
    }
    (run / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(
        run / "coords.npz",
        lat=np.asarray(g["lat_deg"], dtype=np.float64),
        lon=np.asarray(g["lon_deg"], dtype=np.float64),
        sigma=np.asarray(g["sigma_full"], dtype=np.float64),
        sigma_edges=np.asarray(g["sigma_half"], dtype=np.float64),
        mu_edges=np.asarray(g["mu_edges"], dtype=np.float64),
    )
    wall = time.time() - t_start
    print(f"[run_hs] 完成 {days} 天：{total_steps} 步，{len(snap_days)} 快照，"
          f"wall {wall:.0f}s（{wall / total_steps * 1000:.2f} ms/step）→ {run}", flush=True)
    return meta


def probe_mass(n_steps: int, dt: float, ext_dir: str | None, mode: str) -> dict:
    """F9 mass 口径探针：逐步记录 ac.step 的 diag.mass 与独立复算的两种全球平均 ps。

    mode: "isothermal_rest"（A1 README §5.2 口径）| "teq_perturbed"（本跑批器初值口径）
    """
    ac = import_atmo_core(ext_dir)
    g = ac.grid_spec(nlat=32, nlon=64, nlev=8)
    if mode == "isothermal_rest":
        st = ac.make_state(g, T0=288.0, ps0=101325.0)
    else:
        st = make_initial_state(ac, g, seed=42, ic="teq_noise")
    area = np.asarray(g["area"], dtype=np.float64)
    lat = np.asarray(g["lat_deg"], dtype=np.float64)
    w_cos = np.cos(np.radians(lat))
    w_cos = w_cos / w_cos.sum()
    masses = {"diag": [], "area_ps": [], "unw_ps": [], "cos_ps": []}
    for i in range(1, n_steps + 1):
        diag = ac.step(st, None, dt)
        ps = np.asarray(st["ps"], dtype=np.float64)
        zm = ps.mean(axis=1)
        masses["diag"].append(float(diag["mass"]))
        masses["area_ps"].append(float(zm @ area))
        masses["unw_ps"].append(float(ps.mean()))
        masses["cos_ps"].append(float(zm @ w_cos))
    out = {"mode": mode, "dt": dt, "n_steps": n_steps}
    for k, v in masses.items():
        arr = np.asarray(v)
        out[k] = {"rel_drift": float(np.max(np.abs((arr - arr[0]) / arr[0]))),
                  "final_minus_initial_over_initial": float((arr[-1] - arr[0]) / arr[0])}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="av-run-hs")
    ap.add_argument("--out", default=None, help="run 目录（loader 契约）")
    ap.add_argument("--days", type=float, default=120.0)
    ap.add_argument("--dt", type=float, default=288.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--snap-start-day", type=float, default=30.0)
    ap.add_argument("--snap-every-day", type=float, default=2.0)
    ap.add_argument("--ext-dir", default=None, help="atmo_core.pyd 所在目录")
    ap.add_argument("--ic", default="iso_w1", choices=["iso_w1", "teq_noise"],
                    help="初值：iso_w1=等温静止+波数1光滑扰动；teq_noise=HS平衡剖面+白噪声（对照臂）")
    ap.add_argument("--probe-mass", type=int, default=0, metavar="N",
                    help="只跑 N 步 mass 口径探针（F9），不产出 run 目录")
    ap.add_argument("--probe-mode", default="isothermal_rest",
                    choices=["isothermal_rest", "teq_perturbed"])
    args = ap.parse_args(argv)

    if args.probe_mass:
        rep = probe_mass(args.probe_mass, args.dt, args.ext_dir, args.probe_mode)
        print(json.dumps(rep, ensure_ascii=False, indent=1))
        return 0
    if not args.out:
        ap.error("--out 必填（或用 --probe-mass）")
    run_hs(args.out, args.days, args.dt, args.seed,
           args.snap_start_day, args.snap_every_day, args.ext_dir, args.ic)
    return 0


if __name__ == "__main__":
    sys.exit(main())
