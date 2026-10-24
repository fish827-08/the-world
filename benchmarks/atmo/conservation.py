"""AV 守恒/稳定性检查：只算实测值，判定阈值在 criteria.py。"""
from pathlib import Path

import numpy as np

from benchmarks.atmo.metrics import area_weights, linear_slope

G_EARTH = 9.81  # m s-2


def ps_global_mean_series(ps, lat_deg, mu_edges=None):
    """ps (nt, nlat, nlon) Pa -> 逐时刻面积加权全球平均，形状 (nt,)。

    lat 加权后必须再对经度取均值：只做 tensordot 会留下 (nt, nlon)，下游按最大值归约时
    测到的就成了纬向图案重排，而不是全球质量（评审 2026-10-21 F11）。
    `mu_edges` 透传给 `area_weights`（R-8：Gaussian 原生网格用精确带边）。
    """
    ps = np.asarray(ps, dtype=np.float64)
    w = area_weights(lat_deg, mu_edges=mu_edges)
    if ps.shape[-2] != w.size:
        raise ValueError(f"lat 轴不匹配: field {ps.shape[-2]} vs lat {w.size}")
    return np.tensordot(ps, w, axes=([-2], [0])).mean(axis=-1) / w.sum()


def mass_drift(ps, lat_deg, mu_edges=None):
    """全期相对 t0 的最大漂移 |dM|/M0（CONS-MASS 的实测值）。"""
    s = ps_global_mean_series(np.asarray(ps, dtype=np.float64), lat_deg, mu_edges=mu_edges)
    return float(np.max(np.abs((s - s[0]) / s[0])))


def column_water(q, dp):
    """气柱水重 (nt,...) kg m-2。q (nt, nlev, nlat, nlon)，dp (nt, nlev, nlat, nlon) 各层气压厚 Pa。"""
    return np.sum(np.asarray(q, dtype=np.float64) * np.asarray(dp, dtype=np.float64), axis=1) / G_EARTH


def water_closure(pr, evap, w_series, days, lat_deg):
    """全球水账闭合（SI）。pr/evap (nt, nlat, nlon) kg m-2 s-1；w_series (nt,) kg m-2；days (nt,)。
    返回 residual（SI）与相对 P 的比值（CONS-WATER 实测值）。面积加权。"""
    pr = np.asarray(pr, dtype=np.float64)
    evap = np.asarray(evap, dtype=np.float64)
    w_series = np.asarray(w_series, dtype=np.float64)
    days = np.asarray(days, dtype=np.float64)
    w = area_weights(lat_deg)
    p_bar = np.tensordot(pr, w, axes=([-2], [0])).mean() / w.sum()
    e_bar = np.tensordot(evap, w, axes=([-2], [0])).mean() / w.sum()
    dw_dt = (w_series[-1] - w_series[0]) / ((days[-1] - days[0]) * 86400.0)
    residual = e_bar - p_bar - dw_dt
    rel = abs(residual) / abs(p_bar) if p_bar != 0 else float("inf")
    return {"residual_si": float(residual), "relative": float(rel),
            "p_mean": float(p_bar), "e_mean": float(e_bar), "dw_dt_si": float(dw_dt)}


def energy_closure(sw_toa_net, lw_out, sst, slab_heat_capacity, days, lat_deg):
    """气柱+slab 能量闭合（SI）。sw_toa_net/lw_out (nt, nlat, nlon) W m-2；sst (nt, nlat, nlon) K。
    residual = mean(SW-LW) - C_slab * dSST/dt（CONS-ENERGY 实测值；时窗由调用方截取后传入）。面积加权。"""
    sw = np.asarray(sw_toa_net, dtype=np.float64)
    lw = np.asarray(lw_out, dtype=np.float64)
    sst = np.asarray(sst, dtype=np.float64)
    days = np.asarray(days, dtype=np.float64)
    w = area_weights(lat_deg)

    def gmean(x):
        return np.tensordot(x, w, axes=([-2], [0])).mean() / w.sum()

    net_toa = gmean(sw - lw)
    sst_bar = np.array([gmean(sst[i]) for i in range(sst.shape[0])])
    dt_days = (days[-1] - days[0]) * 86400.0
    storage = slab_heat_capacity * (sst_bar[-1] - sst_bar[0]) / dt_days
    residual = net_toa - storage
    return {"residual": float(residual), "net_toa": float(net_toa),
            "storage": float(storage), "sst_trend_k_per_day": float(linear_slope(sst_bar, days))}


def determinism_compare(dir_a, dir_b):
    """两次运行逐位对比（CONS-DET）。比较 meta.json 与全部 snap_*.npz。返回不匹配清单（空=一致）。"""
    da, db = Path(dir_a), Path(dir_b)
    mismatches = []

    def cmp_array_set(xa, xb, name):
        if set(xa.files) != set(xb.files):
            mismatches.append({"file": name, "why": "keys differ"})
            return
        for k in xa.files:
            va, vb = xa[k], xb[k]
            # 逐位 = 字节级比较：np.array_equal 会把位型相同的 NaN 判为不等
            if va.dtype != vb.dtype or va.shape != vb.shape or va.tobytes() != vb.tobytes():
                mismatches.append({"file": name, "key": k, "why": "bitwise differ"})

    names = sorted({p.name for p in da.iterdir()} | {p.name for p in db.iterdir()})
    for name in names:
        pa, pb = da / name, db / name
        if name not in {p.name for p in da.iterdir()} or name not in {p.name for p in db.iterdir()}:
            mismatches.append({"file": name, "why": "missing in one side"})
            continue
        if name == "meta.json":
            if pa.read_bytes() != pb.read_bytes():
                mismatches.append({"file": name, "why": "meta bytes differ"})
        elif name.endswith(".npz"):
            with np.load(pa, allow_pickle=False) as xa, np.load(pb, allow_pickle=False) as xb:
                cmp_array_set(xa, xb, name)
        else:
            if pa.read_bytes() != pb.read_bytes():
                mismatches.append({"file": name, "why": "bytes differ"})
    return mismatches
