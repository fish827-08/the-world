"""A2 大气物理包 pytest 入口（单柱、不依赖 A1 动力）。

覆盖
----
M1 `selfcheck` 的 34 项模块级断言（太阳/短波/长波/湿力学/云/降水/对流/地表/单柱）；
M2 单柱账本 + 确定性（5 步，同配置逐位一致）；
M3 配置序列化/缺失回退 + fail-loud；
M4 太阳参考实现的解析锚点（`solar.py` 按 INTERFACE v1.0.2 §1 仅作对拍参考）；
M5 跨列批量整步与标量 `ColumnModel` 路径逐位对拍（状态/出口场/账本）。

判据阈值均从被测模块/量纲导入或为机器精度量级，不按结果回调。
TIER = "mainline"（入档②；本文件实测 <1 s，不标 slow）。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from atmo.physics import (
    AtmoPhysicsConfig,
    AtmoPhysicsDriver,
    ColumnConfig,
    ColumnModel,
    DriverConfig,
    SolarConfig,
    SurfaceConfig,
    saturation_specific_humidity,
    solar,
)
from atmo.physics import constants as pc
from atmo.physics.selfcheck import (
    Checker,
    check_clouds,
    check_column,
    check_convection,
    check_longwave,
    check_precip,
    check_shortwave,
    check_solar,
    check_surface,
    check_thermo,
)

TIER = "mainline"

_SELFCHECK_FUNCS = (
    check_solar,
    check_shortwave,
    check_longwave,
    check_thermo,
    check_clouds,
    check_precip,
    check_convection,
    check_surface,
    check_column,
)


def test_module_selfcheck_all_pass():
    """34 项模块断言全过；覆盖数不得低于首版 34。"""
    cfg = AtmoPhysicsConfig()
    chk = Checker(verbose=False)
    for func in _SELFCHECK_FUNCS:
        func(cfg, chk)
    failed = [item["name"] for item in chk.items if not item["passed"]]
    assert not failed, f"模块检查失败: {failed}"
    assert len(chk.items) >= 34, f"检查项被删减: {len(chk.items)} < 34"


def test_column_ledger_and_determinism():
    """5 步账本闭合 + 同配置两次逐位一致（不跑长 RCE）。"""
    model_a = ColumnModel(AtmoPhysicsConfig())
    model_b = ColumnModel(AtmoPhysicsConfig())
    for _ in range(5):
        diag_a = model_a.step()
        diag_b = model_b.step()

    for name in ("T", "qv", "qc", "soil_T"):
        assert np.array_equal(
            getattr(model_a.state, name), getattr(model_b.state, name)
        ), f"{name} 非同配置逐位一致"
    assert model_a.state.Ts == model_b.state.Ts
    assert model_a.state.t == model_b.state.t

    assert diag_a.nan_free and diag_b.nan_free
    assert abs(diag_a.energy_residual) < 0.1, diag_a.energy_residual  # J/m²/step
    assert abs(diag_a.water_residual) < 1.0e-9, diag_a.water_residual  # kg/m²/step


def test_config_roundtrip_and_fail_loud():
    """to_dict→from_dict 原样往返；缺失字段回退默认；非法值 fail-loud。"""
    cfg = AtmoPhysicsConfig()
    assert AtmoPhysicsConfig.from_dict(cfg.to_dict()) == cfg

    partial = AtmoPhysicsConfig.from_dict(
        {"solar": {"solar_constant": 1300.0}, "surface": {"surface_type": "land"}}
    )
    assert partial.solar.solar_constant == 1300.0
    assert partial.solar.mode == "global_mean"
    assert partial.surface.surface_type == "land"

    with pytest.raises(ValueError):
        SolarConfig(mode="bogus")
    with pytest.raises(ValueError):
        SurfaceConfig(soil_layers=3)
    with pytest.raises(ValueError):
        ColumnConfig(pressure_edges_pa=(1000.0, 100000.0))
    with pytest.raises(ValueError):
        DriverConfig(vectorized="yes")


_SIGMA_HALF = np.array([0.0, 0.05, 0.14, 0.26, 0.40, 0.55, 0.70, 0.85, 1.0])
_SIGMA_FULL = np.array([0.025, 0.095, 0.200, 0.330, 0.475, 0.625, 0.775, 0.925])


def _interface_grid(nlat: int, nlon: int) -> dict:
    """INTERFACE v1.0.2 §2.3/§3 的 Gaussian 网格子集（A1 交付前的契约夹具）。"""
    nodes, weights = np.polynomial.legendre.leggauss(nlat)
    return {
        "nlat": nlat,
        "nlon": nlon,
        "nlev": 8,
        "kind": "gaussian",
        "sigma_half": _SIGMA_HALF,
        "sigma_full": _SIGMA_FULL,
        "area": weights / 2.0,
        "lat_rad": np.arcsin(nodes),
        "lon_rad": np.arange(nlon, dtype=np.float64) * 2.0 * np.pi / nlon,
    }


def _a1_state(driver: AtmoPhysicsDriver, nlat: int, nlon: int) -> dict:
    """A1 布局（k=0 模式顶）的合成状态；q 用同一柱压算，保持物理一致。"""
    ps = 101325.0
    p_edges = driver.pressure_edges(ps)
    p_mid = 0.5 * (p_edges[:-1] + p_edges[1:])
    t_col = np.linspace(288.0, 210.0, 8)  # 地表→顶
    q_col = 0.7 * saturation_specific_humidity(
        t_col, p_mid, driver.cfg.thermo
    )
    nlev = 8
    T = np.broadcast_to(t_col[::-1, None, None], (nlev, nlat, nlon)).copy()
    q = np.broadcast_to(q_col[::-1, None, None], (nlev, nlat, nlon)).copy()
    return {
        "T": T,
        "q": q,
        "ps": np.full((nlat, nlon), ps),
        "u": np.full((nlev, nlat, nlon), 3.0),
        "v": np.zeros((nlev, nlat, nlon)),
    }


def test_driver_a1_layout_mapping():
    """层序翻转、sigma 层压/质量、单柱等价与配置 fail-loud。"""
    cfg = AtmoPhysicsConfig()
    cfg.column.fixed_sst = True  # 与 prescribed 海洋（外部热库）对齐
    grid = _interface_grid(1, 1)
    driver = AtmoPhysicsDriver(cfg, grid)
    st = _a1_state(driver, 1, 1)
    forcing = {
        "mask": np.zeros((1, 1)),
        "sst": np.full((1, 1), 300.0),
        "alb": np.full((1, 1), 0.08),
    }
    aux = driver.make_state(st, forcing)
    col = driver.column_state(st, aux, 0, 0)

    assert col.T[0] == st["T"][-1, 0, 0]
    assert col.T[-1] == st["T"][0, 0, 0]
    assert col.qv[0] == st["q"][-1, 0, 0]
    assert col.p_edges[0] == 101325.0
    assert col.p_edges[-1] == cfg.driver.top_pressure_pa
    expected_mass = (101325.0 - cfg.driver.top_pressure_pa) / pc.GRAVITY
    assert np.isclose(col.mass.sum(), expected_mass, rtol=0.0, atol=1e-9)

    toa = np.array([[250.0]])
    cosz = np.array([[0.6]])
    direct = ColumnModel(cfg, col.copy())
    d_direct = direct.step(
        900.0, toa_flux=250.0, mu0=0.6, surface_albedo=0.08
    )
    diag = driver.step(
        st, forcing, aux, 900.0, toa_sw=toa, cosz=cosz
    )

    assert np.array_equal(st["T"], direct.state.T[::-1, None, None])
    assert np.array_equal(st["q"], direct.state.qv[::-1, None, None])
    assert np.array_equal(aux.qc, direct.state.qc[::-1, None, None])
    assert aux.Ts[0, 0] == direct.state.Ts
    assert np.isclose(diag["energy_residual"], d_direct.energy_residual,
                      rtol=0.0, atol=1e-12)
    assert diag["nan_count"] == 0

    exports = driver.export()
    for key in ("precip", "cloud", "sw_sfc", "lw_sfc"):
        assert exports[key].shape == (1, 1)
    assert 0.0 <= exports["cloud"][0, 0] <= 1.0
    assert exports["precip"][0, 0] >= 0.0

    with pytest.raises(ValueError):
        DriverConfig(surface_mode="bogus")
    with pytest.raises(ValueError):
        DriverConfig(top_pressure_pa=0.0)
    with pytest.raises(ValueError):
        DriverConfig(land_mask_threshold=1.5)


def test_driver_global_ledger_and_night():
    """混合海陆 T21 子集：面积加权账本、出口边界、夜间零短波。"""
    cfg = AtmoPhysicsConfig()
    grid = _interface_grid(4, 8)
    driver = AtmoPhysicsDriver(cfg, grid)
    st = _a1_state(driver, 4, 8)
    mask = np.zeros((4, 8))
    mask[0] = 1.0  # 最南行全陆，其余海
    forcing = {
        "mask": mask,
        "sst": np.full((4, 8), 300.0),
        "alb": np.where(mask > 0.5, 0.20, 0.06),
    }
    aux = driver.make_state(st, forcing)
    toa = np.full((4, 8), 294.0)
    cosz = np.full((4, 8), 0.8660254)

    for _ in range(2):
        diag = driver.step(st, forcing, aux, 900.0, toa_sw=toa, cosz=cosz)

    assert abs(diag["energy_residual"]) < 0.1, diag["energy_residual"]
    assert abs(diag["water_residual"]) < 1.0e-9, diag["water_residual"]
    assert diag["nan_count"] == 0
    assert 0.0 <= diag["cloud_mean"] <= 1.0
    assert diag["precip_mean"] >= 0.0
    assert 0.0 < diag["land_fraction"] < 1.0
    exports = driver.export()
    for key in ("precip", "cloud", "sw_sfc", "lw_sfc"):
        assert exports[key].shape == (4, 8)
        assert np.all(np.isfinite(exports[key]))
    assert exports["cloud"].min() >= 0.0 and exports["cloud"].max() <= 1.0
    assert exports["precip"].min() >= 0.0

    night = AtmoPhysicsDriver(cfg, _interface_grid(1, 1))
    st_n = _a1_state(night, 1, 1)
    f_n = {
        "mask": np.zeros((1, 1)),
        "sst": np.full((1, 1), 300.0),
        "alb": np.full((1, 1), 0.06),
    }
    aux_n = night.make_state(st_n, f_n)
    d_n = night.step(
        st_n, f_n, aux_n, 900.0,
        toa_sw=np.zeros((1, 1)), cosz=np.zeros((1, 1)),
    )
    assert d_n["sw_sfc_mean"] == 0.0
    assert night.export()["sw_sfc"][0, 0] == 0.0
    assert d_n["nan_count"] == 0


def test_driver_batch_matches_scalar_bitwise():
    """批量整步与标量逐柱：状态/出口场/逐列账本必须逐位一致。"""
    cfg_scalar = AtmoPhysicsConfig()
    cfg_scalar.driver = DriverConfig(vectorized=False)
    cfg_batch = AtmoPhysicsConfig()
    assert cfg_batch.driver.vectorized is True

    nlat, nlon = 8, 16
    mask = np.zeros((nlat, nlon))
    mask[:, : nlon // 2] = 1.0
    mask[0] = 1.0
    forcing = {
        "mask": mask,
        "sst": np.full((nlat, nlon), 300.0),
        "alb": np.where(mask > 0.5, 0.20, 0.06),
    }
    lon = np.arange(nlon)[None, :]
    day = ((lon + np.arange(nlat)[:, None]) % 4) != 0
    toa = np.where(day, 320.0, 0.0)
    cosz = np.where(day, 0.6, 0.0)

    results = []
    for cfg in (cfg_scalar, cfg_batch):
        driver = AtmoPhysicsDriver(cfg, _interface_grid(nlat, nlon))
        st = _a1_state(driver, nlat, nlon)
        aux = driver.make_state(st, forcing)
        diags = [
            driver.step(st, forcing, aux, 900.0, toa_sw=toa, cosz=cosz)
            for _ in range(2)
        ]
        results.append((st, aux, driver.export(), diags))

    (st_a, aux_a, ex_a, d_a), (st_b, aux_b, ex_b, d_b) = results
    for key in ("T", "q"):
        assert np.array_equal(st_a[key], st_b[key]), f"state {key} 非逐位"
    for key in ("qc", "Ts", "soil_T"):
        assert np.array_equal(getattr(aux_a, key), getattr(aux_b, key)), f"aux {key} 非逐位"
    for key in ("precip", "cloud", "sw_sfc", "lw_sfc"):
        assert np.array_equal(ex_a[key], ex_b[key]), f"export {key} 非逐位"
    for ia, ib in zip(d_a, d_b):
        assert ia == ib, f"诊断不一致: {ia} vs {ib}"


def test_driver_relaxation_falls_back_to_scalar():
    """开启 T/水汽松弛时批量档必须回退（对流→松弛的编排顺序不能反）。"""
    cfg = AtmoPhysicsConfig()
    cfg.column.t_relax_tau_s = 86400.0
    cfg.column.t_ref_profile_k = cfg.column.initial_temperature_k
    assert AtmoPhysicsDriver(cfg, _interface_grid(1, 1)).vectorized is False
    cfg2 = AtmoPhysicsConfig()
    cfg2.column.q_relax_tau_s = 86400.0
    cfg2.column.q_relax_relative_humidity = (0.7,) * 8
    assert AtmoPhysicsDriver(cfg2, _interface_grid(1, 1)).vectorized is False


def test_solar_reference_anchors():
    """A1↔A2 对拍基线：至日赤纬、正午/午夜通量、柱平均 S0/4。"""
    cfg = SolarConfig()
    dec_summer = float(solar.solar_declination(172.0, cfg))
    dec_winter = float(solar.solar_declination(355.0, cfg))
    assert abs(dec_summer - math.radians(23.44)) < math.radians(0.4)
    assert dec_winter < -math.radians(23.0)

    noon, mu_noon = solar.toa_insolation(0.0, 80.0, 12.0, cfg)
    midnight, mu_midnight = solar.toa_insolation(0.0, 80.0, 0.0, cfg)
    assert noon > 1000.0 and mu_noon > 0.0
    assert midnight == 0.0 and mu_midnight == 0.0

    toa, mu0 = solar.column_toa_forcing(cfg)
    assert toa == pytest.approx(0.25 * cfg.solar_constant)
    assert mu0 == pytest.approx(cfg.global_mean_cos_zenith)
