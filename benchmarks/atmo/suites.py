"""AV 套件定义与判据执行：把 loader 读到的实测场喂给 metrics，再按 criteria 冻结阈值判定。"""
from dataclasses import dataclass

import numpy as np

from benchmarks.atmo import conservation, criteria, loader, metrics

# Track A：区间累积通量的登记键（单源；jia_remap 透传、本件切片共用）
FLUX_META_KEYS = ("flux_semantics", "flux_interval_s", "flux_accumulation",
                  "flux_valid_from_snap", "flux_first_interval_s")

# Held & Suarez (1994) 标准强迫参数（本基准的"标准"定义；A1 侧 config 必须与此逐项一致，容差 1%）
HS94_PARAMS = {
    "omega": 7.292e-5,        # s-1
    "g": 9.81,                # m s-2
    "R": 287.0,               # J kg-1 K-1
    "cp": 1004.0,             # J kg-1 K-1
    "radius": 6.371e6,        # m
    "p0": 100000.0,           # Pa
    "delta_Ty": 60.0,         # K，赤道-极点 T_eq 差
    "delta_theta_z": 10.0,    # K，位温垂直梯度项
    "sigma_b": 0.7,           # 边界层/近地面过渡 sigma
    "ka": 1.0 / (40.0 * 86400.0),   # 内部牛顿冷却率 s-1（40 天）
    "ks": 1.0 / (4.0 * 86400.0),    # 近地面冷却率 s-1（4 天）
    "kf": 1.0 / (1.0 * 86400.0),    # 地表拖曳率 s-1（1 天）
}


@dataclass(frozen=True)
class SuiteDef:
    name: str
    wet: bool
    criteria_ids: tuple


HELD_SUAREZ = SuiteDef(
    "held_suarez", False,
    ("HS-FORCING", "HS-JET-U", "HS-JET-LAT", "HS-SFC-U", "HS-T-TROP-SFC",
     "HS-T-POLE-SFC", "HS-T-TROP-UP", "HS-EKE-ML", "HS-EKE-RATIO",
     "CONS-NAN", "CONS-MASS", "CONS-CFL", "CONS-POLAR-AMP", "CONS-POLAR-TREND", "CONS-DET"),
)

AQUAPLANET = SuiteDef(
    "aquaplanet", True,
    ("AQ-FORCING", "AQ-ITCZ-LAT", "AQ-ITCZ-P", "AQ-EP-SUBTROP", "AQ-TOA",
     "AQ-SST-TREND", "AQ-HEMI-DIFF",
     "CONS-NAN", "CONS-MASS", "CONS-CFL", "CONS-POLAR-AMP", "CONS-POLAR-TREND",
     "CONS-WATER", "CONS-ENERGY", "CONS-DET"),
)

SUITES = {s.name: s for s in (HELD_SUAREZ, AQUAPLANET)}


def _both_hemi(field_zm, lat, lat_lo, lat_hi):
    """field_zm (nlev,nlat) 或 (nlat,)：南北半球同 |lat| 带各自面积平均后取平均。"""
    north = metrics.region_mean(field_zm, lat, lat_lo, lat_hi) \
        if field_zm.ndim > 1 else _lat_mean_1d(field_zm, lat, lat_lo, lat_hi)
    south = metrics.region_mean(field_zm, lat, -lat_hi, -lat_lo) \
        if field_zm.ndim > 1 else _lat_mean_1d(field_zm, lat, -lat_hi, -lat_lo)
    return 0.5 * (north + south)


def _lat_mean_1d(v, lat, lo, hi):
    lat = np.asarray(lat, dtype=np.float64)
    m = (lat >= min(lo, hi)) & (lat <= max(lo, hi))
    if not m.any():
        raise ValueError(f"空纬度带 [{lo},{hi}]")
    w = metrics.area_weights(lat)[m]
    return float(np.sum(np.asarray(v)[m] * w) / w.sum())


def _lev_mask(plev, lo, hi):
    m = (plev >= lo) & (plev <= hi)
    if not m.any():
        raise ValueError(f"空气压层窗口 [{lo},{hi}] hPa")
    return m


def evaluate(suite: SuiteDef, run_dir, run_dir_b=None) -> dict:
    run = loader.load_run(run_dir)
    if run["meta"]["benchmark"] != suite.name:
        raise ValueError(
            f"meta.benchmark={run['meta']['benchmark']!r} 与套件 {suite.name} 不符")
    meta, coords, snaps = run["meta"], run["coords"], run["snaps"]
    lat, lon = coords["lat"], coords["lon"]
    if suite.wet and "sigma_edges" not in coords:
        raise ValueError("湿套件需 sigma 坐标 + sigma_edges（垂直积分水账）")
    if "plev_hpa" in coords:
        plev = coords["plev_hpa"]
    else:
        plev = coords["sigma"] * 1000.0
    snap_days = np.asarray(meta["snap_days"], dtype=np.float64)
    if len(snap_days) != len(snaps):
        raise ValueError(f"snap_days 长度 {len(snap_days)} != 快照数 {len(snaps)}")

    rows = []

    def add(cid, measured, status, note=""):
        c = criteria.get(cid)
        rows.append({"id": cid, "desc": c.desc, "measured": measured,
                     "limit_lo": c.limit_lo, "limit_hi": c.limit_hi, "unit": c.unit,
                     "mandatory": c.mandatory, "status": status, "note": note})

    def check(cid, x, note=""):
        # 非有限实测值（NaN/inf）不得 PASS（如全 NaN 场算出的 0/0=inf）
        ok = bool(np.isfinite(x)) and criteria.check_value(cid, x)
        add(cid, x, "PASS" if ok else "FAIL", note)
        return rows[-1]["status"] == "PASS"

    stack = lambda k: np.stack([s[k] for s in snaps])  # noqa: E731
    u, v, T = stack("u"), stack("v"), stack("T")
    ps = stack("ps")

    # ---- 守恒/稳定性（共用）----
    nan_bad = metrics.nan_scan({k: stack(k) for k in snaps[0]})
    add("CONS-NAN", {"n_bad": len(nan_bad), "fields": nan_bad},
        "PASS" if not nan_bad else "FAIL")

    mu_edges = coords.get("mu_edges")
    check("CONS-MASS", conservation.mass_drift(ps, lat, mu_edges=mu_edges),
          note="weights=mu_edges" if mu_edges is not None else "weights=v1(midpoint)")

    dt = float(meta["dt_atmo_s"])
    cfl_re = max(metrics.cfl_advection(s["u"], s["v"], lat, lon, dt) for s in snaps)
    rep = meta.get("cfl_reported") or []
    cfl_rep = float(np.max(rep)) if len(rep) else float("nan")
    cfl_meas = max(x for x in (cfl_re, cfl_rep) if np.isfinite(x)) \
        if np.isfinite(cfl_re) or np.isfinite(cfl_rep) else float("nan")
    check("CONS-CFL", cfl_meas,
          note=f"recomputed={cfl_re:.4g} reported={cfl_rep:.4g}" if np.isfinite(cfl_rep) else "仅重算值")

    pol = metrics.polar_stability(u)   # 保留 (nt, nlev) 前导维
    pol_max_t = pol["polar_max"].reshape(len(snaps), -1).max(axis=1)
    ratio_t = pol["ratio"].reshape(len(snaps), -1).max(axis=1)
    check("CONS-POLAR-AMP", float(np.max(ratio_t)))
    n = len(snaps)
    idx = int(0.8 * n)
    if n - idx >= 2:
        tail_slope = metrics.linear_slope(pol_max_t[idx:], snap_days[idx:])
        check("CONS-POLAR-TREND", tail_slope)
    else:
        add("CONS-POLAR-TREND", None, "NOT_RUN",
            note=f"末 20% 时段快照数 {n - idx} < 2，斜率不可定义")

    # ---- 套件专属 ----
    if not suite.wet:
        forcing = meta.get("forcing", {})
        missing = [k for k in HS94_PARAMS if k not in forcing]
        mismatch = []
        for k, ref in HS94_PARAMS.items():
            if k in forcing:
                got = float(forcing[k])
                if not np.isfinite(got) or abs(got - ref) > 0.01 * abs(ref):
                    mismatch.append(k)
        add("HS-FORCING", {"missing": missing, "mismatch": mismatch},
            "PASS" if not missing and not mismatch else "FAIL",
            note="对照 HS94_PARAMS（rel 容差 1%）")

        u_zm = metrics.zonal_mean(u).mean(axis=0)   # (nlev, nlat)
        T_zm = metrics.zonal_mean(T).mean(axis=0)
        jet = metrics.jet_metrics(u_zm, lat, plev)
        check("HS-JET-U", jet["u_max"])
        check("HS-JET-LAT", jet["lat_deg"])

        sfc_lev = plev >= 900.0
        if not sfc_lev.any():
            raise ValueError("无 sigma>=0.9 的层（近地面窗）")
        check("HS-SFC-U", metrics.region_mean(u_zm[sfc_lev], lat, -30.0, 30.0))
        check("HS-T-TROP-SFC", metrics.region_mean(T_zm[sfc_lev], lat, -15.0, 15.0))
        check("HS-T-POLE-SFC", _both_hemi(T_zm[sfc_lev], lat, 75.0, 90.0))
        up_lev = _lev_mask(plev, 200.0, 250.0)
        check("HS-T-TROP-UP", metrics.region_mean(T_zm[up_lev], lat, -15.0, 15.0))

        eke = metrics.transient_eke(u, v)   # (nlev, nlat)
        ek_lev = _lev_mask(plev, 300.0, 700.0)
        aml = np.abs(lat)
        eke_ml = float(eke[ek_lev][:, (aml >= 40) & (aml <= 60)].mean())
        eke_tr = float(eke[ek_lev][:, aml <= 15].mean())
        check("HS-EKE-ML", eke_ml, note=f"tropical EKE={eke_tr:.4g}")
        check("HS-EKE-RATIO", eke_ml / eke_tr if eke_tr > 0 else float("inf"))
    else:
        missing = [k for k in ("radiation", "slab_heat_capacity") if k not in meta]
        add("AQ-FORCING", {"missing": missing},
            "PASS" if not missing else "FAIL", note="可追溯性：辐射方案与 slab 热容必须登记")

        # Track A：区间累积通量的 pr/evap 判据从第 2 个快照起（首快照区间在门窗外）。
        # CONS-WATER 的 W/days 必须用**全窗**：P/E 覆盖 (day0, dayN]，ΔW 也是 day0→dayN；
        # 把 W/days 跟着切到 [1:] 会把比较窗错开一个区间（评审台账 §36.3/§36.4）。
        export_meta = meta.get("export")
        if isinstance(export_meta, dict):
            misplaced = [k for k in FLUX_META_KEYS if k in export_meta and k not in meta]
            if misplaced:
                raise ValueError(
                    f"flux 登记键错放在 meta['export'] 而顶层缺失: {misplaced}"
                    "（契约要求顶层；见 gate/J2_trackA_reader_20261022.md）")
        flux_semantics = meta.get("flux_semantics")
        if flux_semantics not in (None, "instant", "interval_mean"):
            raise ValueError(f"未知 flux_semantics={flux_semantics!r}（只许 instant / interval_mean）")
        if flux_semantics == "interval_mean":
            if meta.get("flux_valid_from_snap") != 2:
                raise ValueError(
                    "flux_semantics=interval_mean 需 flux_valid_from_snap=2（1-based），"
                    f"实为 {meta.get('flux_valid_from_snap')!r}")
            flux_off = 1
        else:
            flux_off = 0
        pr_flux = stack("pr")[flux_off:]
        evap_flux = stack("evap")[flux_off:]

        pr_zm = metrics.zonal_mean(pr_flux).mean(axis=0) * 86400.0  # mm/d
        evap_zm = metrics.zonal_mean(evap_flux).mean(axis=0) * 86400.0
        itcz = metrics.itcz_metrics(pr_zm, lat)
        obliq = float(meta["obliquity_deg"])
        if obliq == 0.0:
            check("AQ-ITCZ-LAT", itcz["lat_deg"])
            hemi_diff = abs(
                _lat_mean_1d(pr_zm, lat, 0.0, 60.0) - _lat_mean_1d(pr_zm, lat, -60.0, 0.0))
            check("AQ-HEMI-DIFF", hemi_diff)
        else:
            add("AQ-ITCZ-LAT", itcz["lat_deg"], "N/A", note=f"obliquity={obliq} 非对称配置，不判")
            add("AQ-HEMI-DIFF", None, "N/A", note=f"obliquity={obliq} 非对称配置，不判")
        check("AQ-ITCZ-P", itcz["pr_max"])
        ep_zm = evap_zm - pr_zm
        check("AQ-EP-SUBTROP", _both_hemi(ep_zm, lat, 20.0, 35.0))

        half = n // 2
        days = snap_days
        en = conservation.energy_closure(
            stack("sw_toa")[half:], stack("lw_out")[half:], stack("sst")[half:],
            float(meta["slab_heat_capacity"]), days[half:], lat)
        check("AQ-TOA", en["net_toa"])
        check("AQ-SST-TREND", en["sst_trend_k_per_day"])
        check("CONS-ENERGY", en["residual"])

        edges = coords["sigma_edges"]
        dp_sigma = np.diff(edges)
        if np.any(dp_sigma <= 0):
            raise ValueError("sigma_edges 必须升序（0..1）")
        q = stack("q")
        dp = dp_sigma[None, :, None, None] * ps[:, None, :, :]
        w_col = conservation.column_water(q, dp)   # (nt, nlat, nlon)
        w_ser = np.array([metrics.area_mean(w_col[i], lat) for i in range(n)])
        wc = conservation.water_closure(pr_flux, evap_flux, w_ser, days, lat)
        check("CONS-WATER", wc["relative"],
              note=f"flux={flux_semantics or 'instant'}, "
                   f"residual={wc['residual_si']:.3e} kg m-2 s-1, P={wc['p_mean']:.3e}")

    # ---- 确定性 ----
    if run_dir_b:
        mism = conservation.determinism_compare(run_dir, run_dir_b)
        add("CONS-DET", {"mismatches": len(mism), "sample": mism[:5]},
            "PASS" if not mism else "FAIL")
    else:
        add("CONS-DET", None, "NOT_RUN", note="需 --run-dir-b（同配置同 seed 第二次运行）")

    statuses = {r["status"] for r in rows}
    if "FAIL" in statuses:
        verdict = "FAIL"
    elif statuses - {"PASS", "N/A"}:
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS"
    return {
        "suite": suite.name, "run_dir": str(run_dir), "run_dir_b": str(run_dir_b) if run_dir_b else None,
        "n_snaps": len(snaps), "days": [float(snap_days[0]), float(snap_days[-1])],
        "rows": rows, "verdict": verdict,
        "meta_summary": {"seed": meta.get("seed"), "config_hash": meta.get("config_hash"),
                         "days_total": meta.get("days")},
    }
