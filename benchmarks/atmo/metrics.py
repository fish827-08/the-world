"""AV 指标库：纯 numpy，无引擎依赖。

只做计算、返回实测值；判定一律在 suites.py 用 criteria.py 的冻结阈值。
所有场约定（依派工件 §二）：float64 (nlev, nlat, nlon) C 序；lat/lon 度。
"""
import numpy as np

R_EARTH = 6.371e6  # m


def zonal_mean(x, axis=-1):
    return np.mean(x, axis=axis)


def area_weights(lat_deg, mu_edges=None):
    """纬度带面积权 ∝ Δ(sin φ)，默认按行中心反推带边（相邻中点；两端各延伸半个带距）。

    cos φ 只在**等 Δφ** 网格上是面积权；本包目标网格是**等 Δμ 行**，用 cos 会把热带权重
    抬高、极区压低（评审 2026-10-21 F12 / A1 补件 §三.1 指控，AV 独立复核：CONS-MASS
    3.22e-3 → 4.13e-5）。带边不补极帽（与映射"不填极"同约定），权重和 = 2μ_max ≤ 2。

    `mu_edges`（nlat+1 个 μ 带边）：直接给真实积分边界，返回 `np.diff(mu_edges)`。
    **Gaussian 原生网格必须给**——中点反推对 Gauss 节点只是近似，会把带边推错 ~0.7°，
    造成 1e-6 量级的伪质量漂移（R-8，`gate/R8_area_weights_fix_20261022.md`）。
    不给时行为与 v1 逐位一致（旧 run 目录回退口径）。
    """
    lat = np.asarray(lat_deg, dtype=np.float64)
    if lat.ndim != 1 or lat.size == 0:
        raise ValueError(f"lat 必须是 1D 非空，实得 shape {lat.shape}")
    if lat.size > 1 and np.any(np.diff(lat) <= 0):
        raise ValueError("lat 必须严格升序（网格契约）")
    if mu_edges is not None:
        e = np.asarray(mu_edges, dtype=np.float64)
        if e.ndim != 1 or e.size != lat.size + 1:
            raise ValueError(
                f"mu_edges 必须是长度 nlat+1={lat.size + 1} 的 1D，实得 shape {e.shape}")
        if not np.all(np.isfinite(e)):
            raise ValueError("mu_edges 含非有限值")
        if np.any(np.diff(e) <= 0):
            raise ValueError("mu_edges 必须严格升序")
        if e[0] < -1.0 or e[-1] > 1.0:
            raise ValueError(f"mu_edges 必须落在 [-1,1]，实得 [{e[0]}, {e[-1]}]")
        return np.diff(e)
    mu = np.sin(np.deg2rad(lat))
    if mu.size == 1:
        return np.array([2.0])
    mid = 0.5 * (mu[:-1] + mu[1:])
    hi = min(1.0, float(mu[-1] + 0.5 * (mu[-1] - mu[-2])))
    lo = max(-1.0, float(mu[0] - 0.5 * (mu[1] - mu[0])))
    return np.diff(np.concatenate(([lo], mid, [hi])))


def area_mean(x, lat_deg):
    """全球面积加权平均（对全部维度归约）。x 形状 (..., nlat, nlon)：lat 轴 = axis -2。"""
    w = area_weights(lat_deg)
    if x.shape[-2] != w.size:
        raise ValueError(f"lat 轴不匹配: field {x.shape[-2]} vs lat {w.size}")
    wb = w.reshape((1,) * (x.ndim - 2) + (w.size, 1))
    lead = int(np.prod(x.shape[:-2])) if x.ndim > 2 else 1
    denom = w.sum() * x.shape[-1] * lead
    return float(np.sum(x * wb) / denom)


def region_mean(x, lat_deg, lat_lo, lat_hi):
    """在纬度带 [lat_lo, lat_hi] 的面积加权平均（对全部维度归约）。
    纬轴自动判定：-2 优先（场带 lon 维），否则 -1（纬向平均场 (..., nlat)）。"""
    w = area_weights(lat_deg)
    m = (lat_deg >= min(lat_lo, lat_hi)) & (lat_deg <= max(lat_lo, lat_hi))
    if not m.any():
        raise ValueError("空纬度带")
    if x.shape[-2] == w.size:
        ax = -2
    elif x.shape[-1] == w.size:
        ax = -1
    else:
        raise ValueError(f"lat 轴不匹配: field {x.shape} vs lat {w.size}")
    xs = np.take(x, np.where(m)[0], axis=ax)
    if ax == -2:
        wb = w[m].reshape((1,) * (xs.ndim - 2) + (w[m].size, 1))
    else:
        wb = w[m].reshape((1,) * (xs.ndim - 1) + (w[m].size,))
    other = int(np.prod(xs.shape)) // w[m].size
    return float(np.sum(xs * wb) / (w[m].sum() * other))


def jet_metrics(u_zm, lat_deg, plev_hpa, p_window=(200.0, 400.0), lat_band=20.0):
    """急流指标：副热带急流最大风速与其纬度。u_zm (nlev, nlat) 时均纬向平均。"""
    u_zm = np.asarray(u_zm)
    lat_deg = np.asarray(lat_deg, dtype=np.float64)
    pm = (plev_hpa >= p_window[0]) & (plev_hpa <= p_window[1])
    lm = np.abs(lat_deg) >= lat_band
    if not pm.any() or not lm.any():
        raise ValueError("急流窗口为空（plev 或 lat 掩膜）")
    sub = u_zm[np.ix_(pm, lm)]
    i = np.unravel_index(int(np.argmax(sub)), sub.shape)
    return {"u_max": float(sub[i]), "lat_deg": float(np.abs(lat_deg[lm])[i[1]])}


def transient_eke(u, v):
    """瞬变涡动动能（对时-纬平均的偏差）。u,v (nt, nlev, nlat, nlon) -> (nlev, nlat)。"""
    def dev(x):
        return x - x.mean(axis=0, keepdims=True) - x.mean(axis=-1, keepdims=True) \
            + x.mean(axis=(0, -1), keepdims=True)
    return 0.5 * (np.mean(dev(u) ** 2, axis=(0, -1)) + np.mean(dev(v) ** 2, axis=(0, -1)))


def itcz_metrics(pr_zm, lat_deg, tropics=(-20.0, 20.0)):
    """ITCZ 指标。pr_zm (nlat,) 时均纬向平均降水（mm/d 或 SI 均可，量纲随输入）。"""
    lat_deg = np.asarray(lat_deg, dtype=np.float64)
    m = (lat_deg >= tropics[0]) & (lat_deg <= tropics[1])
    lat_t, pr_t = lat_deg[m], np.asarray(pr_zm)[m]
    i = int(np.argmax(pr_t))
    w = np.clip(pr_t, 0.0, None)
    centroid = float(np.sum(w * lat_t) / np.sum(w)) if w.sum() > 0 else float("nan")
    return {"lat_deg": float(lat_t[i]), "pr_max": float(pr_t[i]), "centroid_deg": centroid}


def linear_slope(y, x=None):
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n < 2:
        return float("nan")
    if x is None:
        x = np.arange(n, dtype=np.float64)
    xm, ym = x.mean(), y.mean()
    denom = np.sum((x - xm) ** 2)
    if denom == 0:
        return float("nan")
    return float(np.sum((x - xm) * (y - ym)) / denom)


def mm_per_day_to_si(x):
    return np.asarray(x, dtype=np.float64) / 86400.0


def cfl_advection(u, v, lat_deg, lon_deg, dt_s, radius=R_EARTH, w=None, dz=None,
                  pole_rows=2):
    """AV 独立重算平流 CFL。u,v (..., nlat, nlon)（单快照，nlev 维可有可无）。
    假定 lat/lon 等距；dx = R cos(phi) dlambda, dy = R dphi。
    pole_rows：排除两极各 n 行（经纬网格极区 dx→0 的已知格点效应，
    极区行为由 CONS-POLAR-* 判据另管）。"""
    lat_deg = np.asarray(lat_deg, dtype=np.float64)
    lon_deg = np.asarray(lon_deg, dtype=np.float64)
    if lat_deg.size < 2 or lon_deg.size < 2:
        raise ValueError("lat/lon 至少 2 点")
    dphi = np.deg2rad(np.abs(np.diff(lat_deg)).max())
    dlam = np.deg2rad(np.abs(np.diff(lon_deg)).max())
    m = np.ones(lat_deg.size, dtype=bool)
    if pole_rows > 0 and lat_deg.size > 2 * pole_rows:
        m[:pole_rows] = False
        m[-pole_rows:] = False
    dx = radius * np.cos(np.deg2rad(lat_deg)) * dlam  # (nlat,)
    shape = (1,) * (u.ndim - 2) + (lat_deg.size, 1)
    cu = np.abs(u) / np.reshape(dx, shape)
    cv = np.abs(v) / (radius * dphi)
    c = dt_s * (cu + cv)
    if w is not None and dz is not None and dz > 0:
        c = c + dt_s * np.abs(w) / dz
    return float(c[..., m, :].max())


def nan_scan(fields):
    """fields: {name: array}。返回含 NaN/Inf 的场名列表。"""
    bad = []
    for name, arr in fields.items():
        a = np.asarray(arr, dtype=np.float64)
        if not np.all(np.isfinite(a)):
            bad.append(name)
    return bad


def polar_stability(u_series, n_pole_rows=2, lat=None, lat_polar_min_deg=None):
    """极区稳定性诊断。u_series (..., nlat, nlon)，lat 轴 = axis -2（首轴约定为时间）。
    ratio = 极区最大|风| / 其余区域最大|风|（>1 即极区伪放大）；保留前导维。

    选行口径两套（R-7(c) v2 提案；`gate/R7c_v2_proposal_20261021.md`，未冻结）：
    - 默认（v1，冻结口径）：按行序号取最外 n_pole_rows 行；
    - 给 lat 与 lat_polar_min_deg：按 |lat| >= 阈值选**全部**行（与网格的等距/等 μ 无关）。
    两参必须成对给；默认路径与 v1 逐位一致。
    """
    a = np.abs(np.asarray(u_series))
    if (lat is None) != (lat_polar_min_deg is None):
        raise ValueError("lat 与 lat_polar_min_deg 必须成对提供（或都不给走 v1 行号口径）")
    if lat is None:
        mask = np.ones(a.shape[-2], dtype=bool)
        mask[:n_pole_rows] = False
        mask[-n_pole_rows:] = False
        polar = np.maximum(
            a[..., :n_pole_rows, :].max(axis=(-1, -2)),
            a[..., -n_pole_rows:, :].max(axis=(-1, -2)),
        )
    else:
        lat = np.asarray(lat, dtype=np.float64)
        if lat.ndim != 1 or lat.size != a.shape[-2]:
            raise ValueError(f"lat 须为长度 {a.shape[-2]} 的 1D，实得 shape {lat.shape}")
        if not np.all(np.isfinite(lat)):
            raise ValueError("lat 含非有限值")
        if lat.size > 1 and np.any(np.diff(lat) <= 0):
            raise ValueError("lat 必须严格升序（网格契约）")
        thr = float(lat_polar_min_deg)
        if not 0.0 < thr < 90.0:
            raise ValueError(f"lat_polar_min_deg 须在 (0,90)，实得 {thr}")
        mask = np.abs(lat) < thr
        if not np.any(~mask):
            raise ValueError(f"|lat| >= {thr} 未选中任何行")
        if not np.any(mask):
            raise ValueError(f"|lat| >= {thr} 选中了全部行，其余区域为空")
        polar = a[..., ~mask, :].max(axis=(-1, -2))
    rest = a[..., mask, :].max(axis=(-1, -2))
    ratio = polar / np.maximum(rest, 1e-30)
    return {"polar_max": polar, "rest_max": rest, "ratio": ratio,
            "n_polar_rows": int((~mask).sum())}
