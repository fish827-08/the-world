# TIER = "fast"
"""AV 指标库单测：合成场解析答案（高斯急流/ITCZ/CFL 闭式值/守恒注入）。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.atmo import metrics
from benchmarks.atmo.conservation import (mass_drift, ps_global_mean_series,
                                          water_closure)


def _grid():
    lat = np.linspace(-90.0, 90.0, 61)
    lon = np.linspace(0.0, 360.0, 48, endpoint=False)
    return lat, lon


def test_area_weights_equal_mu_rows_are_uniform():
    """回归（评审 2026-10-21 F12 / A1 补件 §三.1）：等 μ 行（本包目标网格）面积权必须相等。
    旧 `cos φ` 在等 μ 行上把热带抬高、极区压低 ⇒ CONS-MASS 假读数 3.22e-3（真值 4.13e-5）。"""
    nlat = 48
    mu_max = np.sin(np.deg2rad(87.1591))
    edges = np.linspace(-mu_max, mu_max, nlat + 1)
    lat = np.rad2deg(np.arcsin(0.5 * (edges[:-1] + edges[1:])))
    w = metrics.area_weights(lat)
    assert np.ptp(w) / w.mean() < 1e-12, np.ptp(w) / w.mean()
    assert abs(w.sum() - 2.0 * mu_max) < 1e-12
    assert abs(metrics.area_mean(np.full((nlat, 8), 3.0), lat) - 3.0) < 1e-12


def test_area_weights_uniform_lat_matches_cos_on_inner_rows():
    """等 Δφ 网格上 Δsinφ 与 cos φ 只差中点近似 O(Δφ²) ⇒ 旧口径在该网格上仍可用。"""
    lat = np.linspace(-90.0, 90.0, 61)
    w = metrics.area_weights(lat)
    assert abs(w.sum() - 2.0) < 1e-12
    m = np.abs(lat) < 80.0
    wn = w[m] / w[m].mean()
    cn = np.cos(np.deg2rad(lat[m])); cn = cn / cn.mean()
    rel = np.abs(wn / cn - 1.0)
    assert rel.max() < 3e-4, rel.max()


def _v1_area_weights(lat_deg):
    """v1 逐行公式（R-8 之前的冻结实现）——用于证明默认路径逐位不变。"""
    mu = np.sin(np.deg2rad(np.asarray(lat_deg, dtype=np.float64)))
    if mu.size == 1:
        return np.array([2.0])
    mid = 0.5 * (mu[:-1] + mu[1:])
    hi = min(1.0, float(mu[-1] + 0.5 * (mu[-1] - mu[-2])))
    lo = max(-1.0, float(mu[0] - 0.5 * (mu[1] - mu[0])))
    return np.diff(np.concatenate(([lo], mid, [hi])))


def _gauss_edges(n):
    """atmo_core.grid_spec 同款带边：-1 + 2·cumsum(area)，area = Gauss 权/2。"""
    _, gw = np.polynomial.legendre.leggauss(n)
    return np.concatenate(([-1.0], -1.0 + 2.0 * np.cumsum(gw / 2.0)[:-1], [1.0]))


def test_area_weights_default_path_bitwise_v1_three_grids():
    """R-8：不给 mu_edges 时必须与 v1 **逐位**相同（旧 run 目录回退口径）。

    三网格：等 μ 目标网格 48 行 / 等 Δφ 61 行 / Gaussian 原生 32 行；另含单行特例。
    """
    nlat = 48
    mu_max = np.sin(np.deg2rad(87.1591))
    e = np.linspace(-mu_max, mu_max, nlat + 1)
    equal_mu = np.rad2deg(np.arcsin(0.5 * (e[:-1] + e[1:])))
    uniform = np.linspace(-90.0, 90.0, 61)
    gauss = np.degrees(np.arcsin(np.polynomial.legendre.leggauss(32)[0]))
    for lat in (equal_mu, uniform, gauss):
        w = metrics.area_weights(lat)
        assert np.array_equal(w, _v1_area_weights(lat)), "默认路径必须逐位不变"
        assert np.array_equal(metrics.area_weights(lat, mu_edges=None), w)
    assert np.array_equal(metrics.area_weights(np.array([10.0])), np.array([2.0]))


def test_area_weights_mu_edges_match_gauss_weights():
    """R-8：32 节点 Gaussian 的精确带边 ⇒ 面积权即 Gauss 权（Σ=2）。"""
    n = 32
    nodes, gw = np.polynomial.legendre.leggauss(n)
    lat = np.degrees(np.arcsin(nodes))
    w = metrics.area_weights(lat, mu_edges=_gauss_edges(n))
    np.testing.assert_allclose(w, gw, rtol=0, atol=1e-15)
    assert abs(w.sum() - 2.0) < 1e-15


def test_area_weights_mu_edges_match_atmo_core_grid():
    """R-8 金标：32 节点 Gaussian 权 = A1 `atmo_core.grid_spec` 的 area（本包权为 2×area）。"""
    ac = pytest.importorskip("atmo_core", reason="atmo_core 扩展未编译（需 maturin develop --release）")
    g = ac.grid_spec(nlat=32, nlon=64, nlev=8)
    w = metrics.area_weights(np.asarray(g["lat_deg"]), mu_edges=np.asarray(g["mu_edges"]))
    assert abs(w.sum() - 2.0) < 1e-15, w.sum()
    np.testing.assert_allclose(w / 2.0, np.asarray(g["area"]), rtol=0, atol=1e-15)


def test_area_weights_mu_edges_fail_loud():
    lat = np.linspace(-80.0, 80.0, 5)
    e = _gauss_edges(lat.size)
    with pytest.raises(ValueError, match=r"长度 nlat\+1"):
        metrics.area_weights(lat, mu_edges=e[:-1])
    with pytest.raises(ValueError, match="1D"):
        metrics.area_weights(lat, mu_edges=e.reshape(2, 3))
    bad = e.copy(); bad[2] = np.nan
    with pytest.raises(ValueError, match="非有限"):
        metrics.area_weights(lat, mu_edges=bad)
    bad = e.copy(); bad[2] = bad[1]
    with pytest.raises(ValueError, match="严格升序"):
        metrics.area_weights(lat, mu_edges=bad)
    bad = e.copy(); bad[-1] = 1.5
    with pytest.raises(ValueError, match=r"\[-1,1\]"):
        metrics.area_weights(lat, mu_edges=bad)


def test_mass_drift_gaussian_exact_weights_remove_pseudo_drift():
    """R-8 合成闭合：Gauss 原生 32 节点上，v1 中点带边把 P2 模态积错 1.54e-3；
    让 P2 振幅随时间线性增长 ⇒ 伪质量漂移 3.6e-6 量级（台账 R-8 实测同量级）。
    精确带边下 P2（deg 2 ≤ 2n−1）被 Gauss 求积精确积掉 ⇒ 漂移 ~1e-16。
    """
    n = 32
    nodes, _ = np.polynomial.legendre.leggauss(n)
    lat = np.degrees(np.arcsin(nodes))
    p2 = (3.0 * nodes**2 - 1.0) / 2.0
    c = np.linspace(0.0, 2.34e-3, 11)
    ps = np.stack([1e5 * (1.0 + ci * p2)[:, None] * np.ones((n, 4)) for ci in c])
    old = mass_drift(ps, lat)
    new = mass_drift(ps, lat, mu_edges=_gauss_edges(n))
    assert 3.0e-6 < old < 4.2e-6, f"v1 伪漂移量级应 ≈3.6e-6，实得 {old:.3e}"
    assert new < 1e-15, f"精确带边下应 ≈0，实得 {new:.3e}"


def test_zonal_mean_and_area_mean_constant_field():
    lat, lon = _grid()
    x = np.full((3, lat.size, lon.size), 7.5)
    assert np.allclose(metrics.zonal_mean(x), 7.5)
    assert abs(metrics.area_mean(x, lat) - 7.5) < 1e-12


def test_jet_metrics_recovers_gaussian_jet():
    lat, _ = _grid()
    plev = np.array([900.0, 500.0, 250.0])
    u_zm = np.zeros((3, lat.size))
    u_zm[2] = 30.0 * np.exp(-((lat - 35.0) / 12.0) ** 2)
    j = metrics.jet_metrics(u_zm, lat, plev)
    assert abs(j["u_max"] - 30.0) < 0.5
    assert abs(j["lat_deg"] - 35.0) < 2.0


def test_jet_metrics_empty_window_raises():
    lat, _ = _grid()
    plev = np.array([900.0, 500.0])
    with pytest.raises(ValueError):
        metrics.jet_metrics(np.zeros((2, lat.size)), lat, plev)  # 无 200-400 hPa 层


def test_itcz_metrics_recovers_off_equator_band():
    lat, _ = _grid()
    pr_zm = 10.0 * np.exp(-((lat - 5.0) / 6.0) ** 2)
    itcz = metrics.itcz_metrics(pr_zm, lat)
    assert abs(itcz["lat_deg"] - 5.0) < 2.0  # lat 格距 3°，最近格点 6
    assert abs(itcz["pr_max"] - 10.0) < 0.3
    assert abs(itcz["centroid_deg"] - 5.0) < 2.0


def test_transient_eke_zero_for_steady_and_positive_for_fluctuating():
    nt, nlev, nlat, nlon = 6, 2, 16, 24
    rng = np.random.default_rng(42)
    steady = np.ones((nt, nlev, nlat, nlon))
    e0 = metrics.transient_eke(steady, steady)
    assert np.allclose(e0, 0.0, atol=1e-12)
    fluc = 1.0 + rng.normal(0.0, 1.0, (nt, nlev, nlat, nlon))
    e1 = metrics.transient_eke(fluc, fluc)
    assert e1.mean() > 0.5


def test_cfl_advection_closed_form():
    lat = np.array([-1.0, 0.0, 1.0])
    lon = np.linspace(0.0, 360.0, 96, endpoint=False)
    u = np.full((2, 3, lon.size), 10.0)
    v = np.full_like(u, 0.0)
    dt = 600.0
    dlam = np.deg2rad(360.0 / 96)
    expect = dt * 10.0 / (metrics.R_EARTH * np.cos(0.0) * dlam)  # 赤道 dx 最大 |u|/dx？cos 最大 ⇒ dx 最大 ⇒ 最小 CFL
    # 实测 max 应在 cos 最小处（|lat|=1）：cfl = dt*10/(R cos(1°) dlam)
    expect_max = dt * 10.0 / (metrics.R_EARTH * np.cos(np.deg2rad(1.0)) * dlam)
    got = metrics.cfl_advection(u, v, lat, lon, dt)
    assert abs(got - expect_max) / expect_max < 1e-9
    assert got > expect


def test_nan_scan_catches_injected_nan():
    lat, lon = _grid()
    fields = {"T": np.zeros((2, 4, lat.size, lon.size)),
              "ps": np.ones((lat.size, lon.size))}
    assert metrics.nan_scan(fields) == []
    fields["T"][1, 2, 10, 5] = np.nan
    assert metrics.nan_scan(fields) == ["T"]


def test_polar_stability_detects_pole_amplification():
    nt, nlev, nlat, nlon = 4, 2, 16, 24
    u = np.ones((nt, nlev, nlat, nlon))
    res = metrics.polar_stability(u)
    assert np.allclose(res["ratio"], 1.0)
    u[:, :, 0, :] = 5.0  # SH 极行放大 5 倍（其余区域 max=1 ⇒ ratio=5）
    res2 = metrics.polar_stability(u)
    assert res2["ratio"].max() > 3.0


# 原生 octahedral T32 环纬（A1 `native_run1` 48 环实测，24 个北半球值；南半球镜像）
NATIVE_T32_LAT_NH = (
    1.855571, 5.566714, 9.277853, 12.988989, 16.700118, 20.411238, 24.122348,
    27.833444, 31.544523, 35.255580, 38.966610, 42.677606, 46.388558, 50.099453,
    53.810274, 57.520994, 61.231573, 64.941949, 68.652017, 72.361581,
    76.070244, 79.777046, 83.478937, 87.159095,
)


def _equal_mu_lat(nlat=48, mu_max=np.sin(np.deg2rad(87.159103))):
    edges = np.linspace(-mu_max, mu_max, nlat + 1)
    return np.rad2deg(np.arcsin(0.5 * (edges[:-1] + edges[1:])))


def _polar_bump(lat, lon_n=8, bump_lat=77.95, amp=5.0):
    u = np.ones((1, 1, lat.size, lon_n))
    j = int(np.argmin(np.abs(lat - bump_lat)))
    u[0, 0, j, :] = amp
    return u, j


def test_polar_stability_v2_equal_mu_rows():
    """v2（R-7(c) 提案）：等 μ 48 行目标网格上按 |lat|>=75 选行 ⇒ 每半球仅最外 1 行（±77.95°）。
    第二行（±69.45°）的放大**不算极区**——v1 行号口径会把它算进去（R-7 的 69–78° 带）。"""
    lat = _equal_mu_lat()
    u, j_outer = _polar_bump(lat, bump_lat=77.95)
    assert abs(lat[j_outer]) > 75.0
    v2 = metrics.polar_stability(u, lat=lat, lat_polar_min_deg=75.0)
    assert v2["n_polar_rows"] == 2 and v2["ratio"].max() == 5.0

    u2, j2 = _polar_bump(lat, bump_lat=69.45)
    assert abs(lat[j2]) < 75.0
    assert metrics.polar_stability(u2, lat=lat, lat_polar_min_deg=75.0)["ratio"].max() == 0.2  # 放大落在 rest ⇒ 不触发
    assert metrics.polar_stability(u2)["ratio"].max() == 5.0  # v1 两行口径会选中 69.45°


def test_polar_stability_v2_uniform_lat_rows():
    """等距纬网格（2° 间隔）：|lat|>=75 选中 16 行（每半球 8 行），与行号无关。"""
    lat = np.arange(-89.0, 90.0, 2.0)
    u, j = _polar_bump(lat, bump_lat=75.0)
    assert abs(lat[j]) == 75.0
    v2 = metrics.polar_stability(u, lat=lat, lat_polar_min_deg=75.0)
    assert v2["n_polar_rows"] == 16 and v2["ratio"].max() == 5.0
    u2, _ = _polar_bump(lat, bump_lat=73.0)
    assert metrics.polar_stability(u2, lat=lat, lat_polar_min_deg=75.0)["ratio"].max() == 0.2  # 放大落在 rest ⇒ 不触发


def test_polar_stability_v2_native_rings():
    """原生 48 环（48 环 = 24 北 + 24 南）：75° ⇒ 4 环/半球（≥76.07°）；70° ⇒ 加 72.36° 环。
    阈值选行是语义决定：72.36° 的放大只在 70° 口径下算极区。"""
    lat = np.array([-x for x in reversed(NATIVE_T32_LAT_NH)] + list(NATIVE_T32_LAT_NH))
    u, j = _polar_bump(lat, bump_lat=72.361581)
    assert abs(lat[j]) == 72.361581
    assert metrics.polar_stability(u, lat=lat, lat_polar_min_deg=75.0)["n_polar_rows"] == 8
    assert metrics.polar_stability(u, lat=lat, lat_polar_min_deg=75.0)["ratio"].max() == 0.2
    assert metrics.polar_stability(u, lat=lat, lat_polar_min_deg=70.0)["n_polar_rows"] == 10
    assert metrics.polar_stability(u, lat=lat, lat_polar_min_deg=70.0)["ratio"].max() == 5.0
    assert metrics.polar_stability(u, lat=lat, lat_polar_min_deg=80.0)["n_polar_rows"] == 4
    # v1 行号（最外 2 行/端）与 80° 口径选到同一批环 ⇒ 读数应逐位相同
    v1 = metrics.polar_stability(u, n_pole_rows=2)
    v80 = metrics.polar_stability(u, lat=lat, lat_polar_min_deg=80.0)
    assert np.array_equal(v1["polar_max"], v80["polar_max"])
    assert np.array_equal(v1["ratio"], v80["ratio"])


def test_polar_stability_v1_default_and_guards():
    """默认路径必须与 v1 行号口径逐位一致；成对参数与空集护栏须 fail loud。"""
    rng = np.random.default_rng(11)
    u = rng.standard_normal((3, 2, 16, 8))
    v1 = metrics.polar_stability(u, n_pole_rows=2)
    a = np.abs(u)
    polar = np.maximum(a[:, :, :2, :].max(axis=(-1, -2)), a[:, :, -2:, :].max(axis=(-1, -2)))
    rest = a[:, :, 2:-2, :].max(axis=(-1, -2))
    assert np.array_equal(v1["polar_max"], polar)
    assert np.array_equal(v1["rest_max"], rest)
    assert np.array_equal(v1["ratio"], polar / np.maximum(rest, 1e-30))
    assert v1["n_polar_rows"] == 4

    lat = np.linspace(-85.0, 85.0, 16)
    with pytest.raises(ValueError, match="成对"):
        metrics.polar_stability(u, lat=lat)
    with pytest.raises(ValueError, match="成对"):
        metrics.polar_stability(u, lat_polar_min_deg=75.0)
    with pytest.raises(ValueError, match="长度"):
        metrics.polar_stability(u, lat=lat[:-1], lat_polar_min_deg=75.0)
    with pytest.raises(ValueError, match="严格升序"):
        metrics.polar_stability(u, lat=lat[::-1], lat_polar_min_deg=75.0)
    with pytest.raises(ValueError, match=r"\(0,90\)"):
        metrics.polar_stability(u, lat=lat, lat_polar_min_deg=90.0)
    with pytest.raises(ValueError, match="未选中"):
        metrics.polar_stability(u, lat=lat, lat_polar_min_deg=89.0)
    with pytest.raises(ValueError, match="全部行"):
        metrics.polar_stability(u, lat=lat, lat_polar_min_deg=1.0)


def test_mass_drift_zero_then_detects_injection():
    lat, lon = _grid()
    ps = np.full((3, lat.size, lon.size), 100000.0)
    assert mass_drift(ps, lat) == 0.0
    ps2 = ps.copy()
    ps2[1:] *= 1.001  # 0.1% 漂移
    assert abs(mass_drift(ps2, lat) - 0.001) < 1e-12
    s = ps_global_mean_series(ps, lat)
    assert np.allclose(s, 100000.0)


def test_mass_drift_is_global_mean_not_zonal_pattern():
    """回归（评审 2026-10-21 F11）：ps 全球面积加权均值恒定、但纬向图案随时间重排 ⇒ 漂移必须为 0。

    缺陷：旧实现只对 lat 加权、漏了经向均值，返回 (nt,nlon)，`mass_drift` 再在经度维取 max
    ⇒ 测到的是图案重排（A1 件上误报 1.02e-2），不是全球质量。
    构造：P0·(1+0.02cosφ)·(1+0.01·sin(λ+0.3t)) —— 经度均匀取和时 sin 项严格相消 ⇒ 总质量恒定。
    """
    lat, lon = _grid()
    t = np.arange(6)[:, None, None]
    phi = np.deg2rad(lat)[None, :, None]
    lam = np.deg2rad(lon)[None, None, :]
    ps = 1e5 * (1.0 + 0.02 * np.cos(phi)) * (1.0 + 0.01 * np.sin(lam + 0.3 * t))

    s = ps_global_mean_series(ps, lat)
    assert s.shape == (6,), f"全球均值必须逐时刻一维，实为 {s.shape}"
    assert np.ptp(s) / s[0] < 1e-12
    assert mass_drift(ps, lat) < 1e-12


def test_water_closure_detects_imbalance():
    lat, lon = _grid()
    nt = 4
    days = np.array([0.0, 1.0, 2.0, 3.0])
    p = 1e-5  # kg m-2 s-1 (~0.86 mm/d)
    pr = np.full((nt, lat.size, lon.size), p)
    evap = pr.copy()
    w_series = np.full(nt, 20.0)  # 柱水不变
    zero = water_closure(pr, evap, w_series, days, lat)
    assert abs(zero["relative"]) < 1e-12
    evap2 = pr * 1.01  # 蒸发多 1%
    bad = water_closure(pr, evap2, w_series, days, lat)
    assert bad["relative"] > 1e-3
    assert abs(bad["residual_si"] - 0.01 * p) < 1e-9


def test_linear_slope():
    assert metrics.linear_slope([0.0, 1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert metrics.linear_slope([5.0, 5.0, 5.0]) == pytest.approx(0.0)
