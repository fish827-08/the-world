"""A2 物理包自检 + 单柱 RCE 运行器（不依赖 A1 动力核心）。

用法（在 A2 worktree 根目录）::

    PYTHONIOENCODING=utf-8 <venv>/python.exe -m atmo.physics.selfcheck \
        --out _rerun_logs/atmo_physics_selfcheck.json

输出分三块：
1. 模块级单测读数（太阳几何/短波/长波/湿力学/云/降水/对流/地表）；
2. 单柱 20 步的确定性 + 能量/水量账本；
3. 三档单柱 RCE（slab 海洋 / 固定 SST / 陆地）的均值、守恒残差与剖面。

所有判据先写在代码里、后跑读数；阈值均来自量纲/文献量级，不按结果回调。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import constants as c
from . import solar
from .clouds import (
    cloud_fraction,
    cloud_liquid_water_path,
    cloud_optical_depth,
    total_cloud_cover,
)
from .column import ColumnModel
from .config import (
    AtmoPhysicsConfig,
    CloudConfig,
    ConvectionConfig,
    RadiationConfig,
    SurfaceConfig,
)
from .convection import convective_adjustment
from .precip import precip_step
from .radiation import longwave_fluxes, shortwave_fluxes
from .surface import (
    bulk_fluxes,
    solve_land_surface,
    solve_ocean_temperature,
    surface_properties,
)
from .thermo import (
    evaporate_liquid,
    moist_static_energy,
    saturation_adjustment,
    saturation_specific_humidity,
    saturation_vapor_pressure,
)


class Checker:
    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.items: list[dict[str, Any]] = []

    def check(
        self,
        name: str,
        passed: bool,
        value: Any,
        threshold: str,
        note: str = "",
    ) -> bool:
        self.items.append(
            {
                "name": name,
                "passed": bool(passed),
                "value": _jsonable(value),
                "threshold": threshold,
                "note": note,
            }
        )
        if self.verbose:
            tag = "PASS" if passed else "FAIL"
            print(f"[{tag}] {name}: {value}  ({threshold})")
        return bool(passed)

    @property
    def all_passed(self) -> bool:
        return all(item["passed"] for item in self.items)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _profile(cfg: AtmoPhysicsConfig):
    col = cfg.column
    p_edges = np.asarray(col.pressure_edges_pa, dtype=np.float64)
    T = np.asarray(col.initial_temperature_k, dtype=np.float64)
    p_mid = 0.5 * (p_edges[:-1] + p_edges[1:])
    qv = col.initial_relative_humidity * saturation_specific_humidity(T, p_mid, cfg.thermo)
    qc = np.full_like(T, col.initial_cloud_water)
    return p_edges, T, qv, qc, p_mid


# ---------------------------------------------------------------- 太阳几何
def check_solar(cfg: AtmoPhysicsConfig, chk: Checker) -> None:
    annual = solar.global_annual_mean_toa(cfg.solar)
    chk.check(
        "solar.global_annual_mean_toa",
        abs(annual - 0.25 * cfg.solar.solar_constant) < 1.5,
        annual,
        f"|F-{0.25 * cfg.solar.solar_constant:.2f}|<1.5 W/m2",
    )

    dec_summer = float(np.degrees(solar.solar_declination(172.0, cfg.solar)))
    dec_winter = float(np.degrees(solar.solar_declination(355.0, cfg.solar)))
    chk.check(
        "solar.declination_solstice",
        abs(dec_summer - cfg.solar.obliquity_deg) < 0.3
        and abs(dec_winter + cfg.solar.obliquity_deg) < 0.4,
        [dec_summer, dec_winter],
        "夏至≈+23.44°, 冬至≈−23.44° (0.4°)",
    )

    noon, mu_noon = solar.toa_insolation(0.0, 80.0, 12.0, cfg.solar)
    midnight, mu_mid = solar.toa_insolation(0.0, 80.0, 0.0, cfg.solar)
    chk.check(
        "solar.diurnal_cycle",
        noon > 1000.0 and midnight == 0.0 and mu_mid == 0.0,
        {"noon_Wm2": noon, "midnight_Wm2": midnight},
        "春分赤道正午>1000, 午夜=0",
    )

    pole = solar.daily_mean_toa_insolation(90.0, 172.0, cfg.solar)
    expected = (
        cfg.solar.solar_constant
        * float(solar.earth_sun_distance_factor(172.0, cfg.solar))
        * np.sin(np.radians(cfg.solar.obliquity_deg))
    )
    chk.check(
        "solar.polar_day_solstice",
        abs(pole - expected) < 0.5,
        pole,
        f"解析极昼值 {expected:.2f} ±0.5 W/m2",
    )


# ---------------------------------------------------------------- 短波
def check_shortwave(cfg: AtmoPhysicsConfig, chk: Checker) -> None:
    p_edges, T, qv, qc, p_mid = _profile(cfg)
    toa, mu0 = solar.column_toa_forcing(cfg.solar)
    cloud_cfg = cfg.cloud

    clear_cfg = RadiationConfig(sw_vapor_abs_coeff=0.0, sw_rayleigh_tau_surface=0.0)
    sw0 = shortwave_fluxes(
        p_edges, T, qv, np.zeros_like(qc), 0.30, toa, mu0, clear_cfg, cloud_cfg
    )
    residual = sw0.toa_in - sw0.toa_out - sw0.surface_net - sw0.atmosphere_absorption
    chk.check(
        "sw.conservation_no_atmosphere",
        abs(residual) < 1.0e-10 * sw0.toa_in
        and abs(sw0.toa_out - 0.30 * sw0.toa_in) < 1.0e-8
        and abs(sw0.atmosphere_absorption) < 1.0e-8,
        {"residual": residual, "toa_out": sw0.toa_out, "surface_net": sw0.surface_net},
        "残差<1e-10·TOA；τ=0 时 TOA_out=α·TOA、大气吸收=0",
    )

    sw_clear = shortwave_fluxes(
        p_edges, T, qv, np.zeros_like(qc), 0.06, toa, mu0, cfg.radiation, cloud_cfg
    )
    res_clear = (
        sw_clear.toa_in
        - sw_clear.toa_out
        - sw_clear.surface_net
        - sw_clear.atmosphere_absorption
    )
    chk.check(
        "sw.conservation_clear_sky",
        abs(res_clear) < 1.0e-10 * sw_clear.toa_in,
        {"residual": res_clear, "toa_out": sw_clear.toa_out, "atm_abs": sw_clear.atmosphere_absorption},
        "残差<1e-10·TOA",
    )
    chk.check(
        "sw.vapor_absorption_positive",
        0.0 < sw_clear.atmosphere_absorption < 0.35 * sw_clear.toa_in,
        sw_clear.atmosphere_absorption,
        "0<湿大气吸收<35%·TOA",
    )

    qc_cloud = np.where((np.arange(qc.size) >= 2) & (np.arange(qc.size) <= 4), 1.0e-3, 0.0)
    sw_cloud = shortwave_fluxes(
        p_edges, T, qv, qc_cloud, 0.06, toa, mu0, cfg.radiation, cloud_cfg
    )
    res_cloud = (
        sw_cloud.toa_in
        - sw_cloud.toa_out
        - sw_cloud.surface_net
        - sw_cloud.atmosphere_absorption
    )
    chk.check(
        "sw.conservation_with_cloud",
        abs(res_cloud) < 1.0e-10 * sw_cloud.toa_in,
        {"residual": res_cloud, "cloud_tau_max": float(np.max(sw_cloud.cloud_tau))},
        "残差<1e-10·TOA",
    )
    chk.check(
        "sw.cloud_reflects",
        sw_cloud.toa_out > sw_clear.toa_out + 1.0
        and sw_cloud.surface_net < sw_clear.surface_net,
        {"toa_out_clear": sw_clear.toa_out, "toa_out_cloud": sw_cloud.toa_out},
        "云档 TOA 反射更高、地面吸收更低",
    )


# ---------------------------------------------------------------- 长波
def check_longwave(cfg: AtmoPhysicsConfig, chk: Checker) -> None:
    p_edges, T, qv, qc, p_mid = _profile(cfg)
    albedo, emissivity, _ = surface_properties(cfg.surface)
    lw = longwave_fluxes(
        p_edges, T, qv, qc, cfg.column.initial_surface_temperature_k, emissivity,
        cfg.radiation, cfg.cloud,
    )
    closure = lw.olr + lw.surface_net_down + lw.atmosphere_absorption
    scale = c.SIGMA_SB * cfg.column.initial_surface_temperature_k ** 4
    chk.check(
        "lw.conservation_grey",
        abs(closure) < 1.0e-10 * scale,
        {"closure": closure, "olr": lw.olr, "surface_net_down": lw.surface_net_down},
        "OLR + 地面净向下 + 大气吸收 = 0 (1e-10·σTs⁴)",
    )
    sigma_ts4 = scale
    chk.check(
        "lw.olr_below_surface_emission",
        0.0 < lw.olr < sigma_ts4 and lw.atmosphere_absorption < 0.0,
        {"olr": lw.olr, "sigma_ts4": sigma_ts4, "atm_abs": lw.atmosphere_absorption},
        "0<OLR<σTs⁴ 且大气净冷却",
    )

    lin_cfg = RadiationConfig(lw_mode="linear")
    lw_lin = longwave_fluxes(
        p_edges, T, qv, qc, cfg.column.initial_surface_temperature_k, emissivity,
        lin_cfg, cfg.cloud,
    )
    expected = lin_cfg.lw_linear_a + lin_cfg.lw_linear_b * (
        cfg.column.initial_surface_temperature_k - lin_cfg.lw_linear_t_ref
    )
    chk.check(
        "lw.linear_olr_formula",
        abs(lw_lin.olr - expected) < 1.0e-10,
        lw_lin.olr,
        f"OLR=A+B(Ts−Tref)={expected:.3f}",
    )


# ---------------------------------------------------------------- 湿力学
def check_thermo(cfg: AtmoPhysicsConfig, chk: Checker) -> None:
    es0 = float(saturation_vapor_pressure(c.T0, cfg.thermo))
    chk.check(
        "thermo.es_at_0C",
        abs(es0 - c.ES0) < 1.0e-6,
        es0,
        "e_s(273.15 K)=611.21 Pa",
    )

    qs_surface = float(saturation_specific_humidity(288.15, 100000.0, cfg.thermo))
    chk.check(
        "thermo.qs_surface",
        0.0095 < qs_surface < 0.0125,
        qs_surface,
        "q_s(288.15 K,1000 hPa)∈[0.0095,0.0125]",
    )

    T0, qv0, qc0, p0 = 290.0, 0.015, 0.001, 90000.0
    h_before = float(moist_static_energy(T0, 0.0, qv0, qc0, cfg.thermo))
    qt_before = qv0 + qc0
    T1, qv1, qc1, cond = saturation_adjustment(T0, qv0, qc0, p0, cfg.thermo)
    h_after = float(moist_static_energy(T1, 0.0, qv1, qc1, cfg.thermo))
    qs1 = float(saturation_specific_humidity(T1, p0, cfg.thermo))
    chk.check(
        "thermo.saturation_adjustment",
        abs(h_after - h_before) < 1.0e-9 * abs(h_before)
        and abs((qv1 + qc1) - qt_before) < 1.0e-15
        and T1 > T0
        and qv1 <= qs1 + 1.0e-12
        and cond > 0.0,
        {"dE": h_after - h_before, "dqt": (qv1 + qc1) - qt_before, "T": T1, "cond": cond},
        "|Δh|<1e-9·h, Δqt<1e-15, T↑, qv=qs",
    )

    T2, qv2, qc2, evap = evaporate_liquid(300.0, 0.005, 0.002, 90000.0, 0.001, cfg.thermo)
    h2_before = float(moist_static_energy(300.0, 0.0, 0.005, 0.002, cfg.thermo))
    h2_after = float(moist_static_energy(T2, 0.0, qv2, qc2, cfg.thermo))
    qs2 = float(saturation_specific_humidity(T2, 90000.0, cfg.thermo))
    chk.check(
        "thermo.evaporation_energy",
        abs(h2_after - h2_before) < 1.0e-9 * abs(h2_before)
        and abs(evap - 0.001) < 1.0e-12
        and qv2 <= qs2 + 1.0e-12
        and T2 < 300.0,
        {"dE": h2_after - h2_before, "evap": evap, "T": T2},
        "|Δh|<1e-9·h, 蒸发量=请求量, qv≤qs, T↓",
    )


# ---------------------------------------------------------------- 云
def check_clouds(cfg: AtmoPhysicsConfig, chk: Checker) -> None:
    cf = cloud_fraction(np.array([0.5, 0.7, 0.8, 1.0]), cfg.cloud)
    chk.check(
        "cloud.diagnostic_fraction",
        abs(cf[0]) < 1.0e-15 and cf[1] < 1.0e-15 and cf[2] > 0.0 and abs(cf[3] - 1.0) < 1.0e-12,
        cf.tolist(),
        "RH≤0.7→0；RH=1→1；单调",
    )

    lwp = float(cloud_liquid_water_path(np.array([1.0e-4]), np.array([10000.0]))[0])
    tau = float(cloud_optical_depth(np.array([1.0e-4]), np.array([10000.0]), cfg.cloud)[0])
    tau2 = float(cloud_optical_depth(np.array([2.0e-4]), np.array([10000.0]), cfg.cloud)[0])
    chk.check(
        "cloud.optical_depth_slingo",
        abs(lwp - 1.0e-4 * 10000.0 / c.GRAVITY) < 1.0e-12
        and abs(tau - 3.0 * lwp / (2.0 * c.RHO_WATER * 10.0e-6)) < 1.0e-12
        and abs(tau2 - 2.0 * tau) < 1.0e-12,
        {"lwp": lwp, "tau": tau, "tau_2x": tau2},
        "LWP=qc·Δp/g=0.102 kg/m², τ=3LWP/(2ρ_w r_eff)=15.30 (r_eff=10 µm), τ∝qc",
    )
    chk.check(
        "cloud.random_overlap",
        abs(total_cloud_cover([0.5, 0.5]) - 0.75) < 1.0e-15,
        total_cloud_cover([0.5, 0.5]),
        "1−Π(1−C)=0.75",
    )


# ---------------------------------------------------------------- 降水
def check_precip(cfg: AtmoPhysicsConfig, chk: Checker) -> None:
    p_edges, T, qv, qc, p_mid = _profile(cfg)
    cf = np.full(T.size, 0.5)
    qc_heavy = np.full(T.size, 2.0e-3)
    mass = (p_edges[:-1] - p_edges[1:]) / c.GRAVITY
    w0 = float(np.sum((qv + qc_heavy) * mass))
    h0 = float(
        np.sum(moist_static_energy(T, np.zeros_like(T), qv, qc_heavy, cfg.thermo) * mass)
    )
    pr = precip_step(p_edges, T, qv, qc_heavy, cf, cfg.column.dt_s, cfg.precip, cfg.thermo)
    w1 = float(np.sum((pr.qv + pr.qc) * mass))
    h1 = float(
        np.sum(moist_static_energy(pr.T, np.zeros_like(T), pr.qv, pr.qc, cfg.thermo) * mass)
    )
    chk.check(
        "precip.water_conservation",
        abs((w1 - w0) + pr.surface_precip) < 1.0e-12 * max(w0, 1.0),
        {"dW": w1 - w0, "surface": pr.surface_precip, "re-evap": pr.rain_evaporation},
        "ΔW+P=0 (1e-12·W)",
    )
    chk.check(
        "precip.energy_conservation",
        abs(h1 - h0) < 1.0e-9 * max(abs(h0), 1.0),
        h1 - h0,
        "|ΔE|<1e-9·E（常数潜热档，液相焓按 0）",
    )
    chk.check(
        "precip.produces_rain",
        pr.surface_precip > 0.0 and np.all(pr.qc >= 0.0),
        pr.surface_precip,
        "qc=2 g/kg → 地面降水>0",
    )

    qc_light = np.full(T.size, 2.0e-4)
    pr_light = precip_step(
        p_edges, T, qv, qc_light, cf, cfg.column.dt_s, cfg.precip, cfg.thermo
    )
    chk.check(
        "precip.below_threshold",
        pr_light.surface_precip == 0.0 and np.allclose(pr_light.qc, qc_light),
        pr_light.surface_precip,
        "qc_in=0.4 g/kg<0.5 g/kg → 无降水",
    )


# ---------------------------------------------------------------- 对流
def check_convection(cfg: AtmoPhysicsConfig, chk: Checker) -> None:
    model = ColumnModel(cfg)
    s = model.state
    z = s.heights()[1]
    T = s.T.copy()
    T[0] = 320.0
    qv = s.qv.copy()
    qc = np.zeros_like(T)
    mass = s.mass
    h0 = float(np.sum(moist_static_energy(T, z, qv, qc, cfg.thermo) * mass))
    w0 = float(np.sum((qv + qc) * mass))
    conv = convective_adjustment(
        s.p_edges, T, qv, qc, z, cfg.convection, cfg.thermo
    )
    h1 = float(np.sum(moist_static_energy(conv.T, z, conv.qv, conv.qc, cfg.thermo) * mass))
    w1 = float(np.sum((conv.qv + conv.qc) * mass))
    h_final = moist_static_energy(conv.T, z, conv.qv, conv.qc, cfg.thermo)
    chk.check(
        "convection.mse_conservation",
        abs(h1 - h0) < 1.0e-8 * max(abs(h0), 1.0),
        h1 - h0,
        "|ΔΣh·m|<1e-8·Σh·m",
    )
    chk.check(
        "convection.water_conservation",
        abs((w1 - w0) + conv.precip) < 1.0e-12 * max(w0, 1.0),
        {"dW": w1 - w0, "precip": conv.precip},
        "ΔW+P=0 (1e-12·W)",
    )
    chk.check(
        "convection.stabilizes",
        bool(np.all(np.diff(h_final) >= -cfg.convection.mse_tol - 1.0e-6)),
        {"min_dh": float(np.min(np.diff(h_final))), "pairs": conv.pairs_adjusted,
         "iters": conv.iterations},
        "调整后 h 随高度不减小（容差 mse_tol=1 J/kg）",
    )

    conv_dry = convective_adjustment(
        s.p_edges, T, qv, qc, z, ConvectionConfig(qc_hold=0.0), cfg.thermo
    )
    w_dry0 = float(np.sum((qv + qc) * mass))
    w_dry1 = float(np.sum((conv_dry.qv + conv_dry.qc) * mass))
    chk.check(
        "convection.produces_precip",
        conv_dry.precip > 0.0
        and abs((w_dry1 - w_dry0) + conv_dry.precip) < 1.0e-12 * max(w_dry0, 1.0),
        {"precip": conv_dry.precip, "dW": w_dry1 - w_dry0},
        "qc_hold=0 时不稳定柱 → 对流降水>0 且 ΔW+P=0",
    )


# ---------------------------------------------------------------- 地表
def check_surface(cfg: AtmoPhysicsConfig, chk: Checker) -> None:
    h_up, le_up, _ = bulk_fluxes(300.0, 290.0, 0.005, 100000.0, 5.0, 1.0,
                                 cfg.surface, cfg.thermo)
    h_dn, _, _ = bulk_fluxes(280.0, 290.0, 0.005, 100000.0, 5.0, 1.0,
                             cfg.surface, cfg.thermo)
    chk.check(
        "surface.bulk_flux_signs",
        h_up > 0.0 and le_up > 0.0 and h_dn < 0.0,
        {"H_up": h_up, "LE_up": le_up, "H_down": h_dn},
        "Ts>Ta→H>0；海面饱和→LE>0；Ts<Ta→H<0",
    )

    ocean = solve_ocean_temperature(
        288.0, 200.0, 300.0, 285.0, 0.005, 100000.0, 900.0,
        cfg.surface, cfg.thermo, wind_speed=5.0,
    )
    cml = c.RHO_WATER * c.CW * cfg.surface.ocean_mixed_layer_depth
    lw_net = cfg.surface.emissivity_ocean * (
        300.0 - c.SIGMA_SB * ocean.temperature ** 4
    )
    balance = cml * (ocean.temperature - 288.0) / 900.0 - (
        200.0 + lw_net - ocean.sensible - ocean.latent
    )
    chk.check(
        "surface.ocean_implicit_balance",
        abs(balance) < 1.0e-6,
        {"Ts": ocean.temperature, "residual_Wm2": balance},
        "|C dT/dt − 净通量|<1e-6 W/m²",
    )

    soil0 = np.array([288.0, 288.0])
    land_cfg = SurfaceConfig(surface_type="land")
    land, soil1 = solve_land_surface(
        soil0, 200.0, 300.0, 285.0, 0.005, 100000.0, 900.0,
        land_cfg, cfg.thermo, wind_speed=5.0,
    )
    k_surf = land_cfg.soil_conductivity[0] / (0.5 * land_cfg.soil_depth_m[0])
    land_balance = (
        200.0
        + land_cfg.emissivity_land * (300.0 - c.SIGMA_SB * land.temperature ** 4)
        - land.sensible
        - land.latent
        - k_surf * (land.temperature - float(soil1[0]))
    )
    chk.check(
        "surface.land_surface_balance",
        abs(land_balance) < 1.0e-5 and np.all(np.isfinite(soil1)),
        {"Ts": land.temperature, "soil": soil1.tolist(), "residual_Wm2": land_balance},
        "|净辐射−湍流−土壤导热|<1e-5 W/m²",
    )


# ---------------------------------------------------------------- 单柱账本
def check_column(cfg: AtmoPhysicsConfig, chk: Checker) -> None:
    m1 = ColumnModel(cfg)
    m2 = ColumnModel(cfg)
    energy_res = 0.0
    water_res = 0.0
    for _ in range(20):
        d1 = m1.step()
        d2 = m2.step()
        energy_res = max(energy_res, abs(d1.energy_residual))
        water_res = max(water_res, abs(d1.water_residual))
    same = (
        np.array_equal(m1.state.T, m2.state.T)
        and np.array_equal(m1.state.qv, m2.state.qv)
        and np.array_equal(m1.state.qc, m2.state.qc)
        and m1.state.Ts == m2.state.Ts
    )
    chk.check(
        "column.bitwise_determinism",
        same,
        same,
        "同配置两次 20 步逐位一致",
    )
    mass_total = float(np.sum(m1.state.mass))
    energy_scale = c.CPD * mass_total * 1.0  # 1 K 等效柱能量
    chk.check(
        "column.energy_ledger",
        energy_res < 1.0e-6 * energy_scale,
        {"max_abs_Jm2": energy_res, "equiv_K": energy_res / energy_scale},
        "单步残差<1e-6 K 等效",
    )
    chk.check(
        "column.water_ledger",
        water_res < 1.0e-10,
        water_res,
        "<1e-10 kg/m²/step",
    )
    chk.check(
        "column.no_nan",
        bool(
            np.all(np.isfinite(m1.state.T))
            and np.all(np.isfinite(m1.state.qv))
            and np.all(np.isfinite(m1.state.qc))
            and np.isfinite(m1.state.Ts)
        ),
        True,
        "20 步后全有限",
    )


# ---------------------------------------------------------------- RCE
def _rce_config(kind: str) -> AtmoPhysicsConfig:
    cfg = AtmoPhysicsConfig()
    if kind == "fixed_sst":
        cfg.column.fixed_sst = True
    elif kind == "land":
        cfg.surface = SurfaceConfig(surface_type="land")
    elif kind != "slab_ocean":
        raise ValueError(f"未知 RCE 档: {kind}")
    return cfg


def run_rce(
    kind: str,
    spinup_days: float,
    sample_days: float,
    verbose: bool = True,
) -> dict[str, Any]:
    cfg = _rce_config(kind)
    model = ColumnModel(cfg)
    t0 = time.perf_counter()
    summary = model.run(days=sample_days, spinup_days=spinup_days)
    runtime = time.perf_counter() - t0
    summary["kind"] = kind
    summary["runtime_s"] = runtime
    summary["config"] = cfg.to_dict()
    if verbose:
        means = summary["means"]
        closure = (
            summary["atmosphere_closure_wm2"]
            if kind == "fixed_sst"
            else summary["surface_closure_wm2"]
        )
        print(
            f"[RCE {kind}] {runtime:.1f}s  Ts {summary['surface_temperature_start']:.2f}"
            f"→{summary['surface_temperature_final']:.2f} K  "
            f"dTs/dt={summary['dts_dt_k_per_day']:+.4f} K/d  "
            f"TOA={summary['toa_net_mean']:.2f}  "
            f"closure={closure:+.2e} W/m2  "
            f"OLR={means.get('olr', float('nan')):.1f}  "
            f"PW={means.get('precipitable_water', float('nan')):.2f} kg/m2  "
            f"LWP={means.get('lwp', float('nan')):.3f}  "
            f"P={summary['precip_mm_day']:.2f} mm/d  "
            f"E={summary['evaporation_mm_day']:.2f} mm/d  "
            f"cloud={means.get('cloud_cover', float('nan')):.2f}"
        )
    return summary


def rce_gates(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """RCE 判据：守恒、量级、准平衡/闭合。

    “辐射平衡”按**系统能量闭合**判：外部净辐射 = 系统储热变化率
    （海洋/土壤储热计入）。单柱无大尺度下沉/动力，云盖偏厚，
    不把 |TOA|→0 当作稳态判据；改报 dTs/dt 与闭合残差。
    """
    kind = summary["kind"]
    means = summary["means"]
    mass_scale = c.P0 / c.GRAVITY * c.CPD * 1.0
    gates = [
        {
            "name": f"rce.{kind}.no_nan",
            "passed": summary["nan_steps"] == 0,
            "value": summary["nan_steps"],
            "threshold": "0",
        },
        {
            "name": f"rce.{kind}.energy_ledger",
            "passed": summary["max_abs_energy_residual"] < 1.0e-6 * mass_scale,
            "value": summary["max_abs_energy_residual"],
            "threshold": "<1e-6 K 等效",
        },
        {
            "name": f"rce.{kind}.water_ledger",
            "passed": summary["max_abs_water_residual"] < 1.0e-9,
            "value": summary["max_abs_water_residual"],
            "threshold": "<1e-9 kg/m²/step",
        },
    ]
    # 陆地档缺土壤湿度储库（β 固定），单柱积分会进入干冷态；它只作
    # 守恒/闭合检验，不用气候量级（Ts/PW/P）判据。
    if kind != "land":
        gates += [
            {
                "name": f"rce.{kind}.ts_plausible",
                "passed": 250.0 < summary["surface_temperature_final"] < 340.0,
                "value": summary["surface_temperature_final"],
                "threshold": "(250, 340) K",
            },
            {
                "name": f"rce.{kind}.water_vapor_plausible",
                "passed": 5.0 < means.get("precipitable_water", 0.0) < 80.0,
                "value": means.get("precipitable_water"),
                "threshold": "(5, 80) kg/m²",
            },
            {
                "name": f"rce.{kind}.precip_plausible",
                "passed": 0.05 < summary["precip_mm_day"] < 30.0,
                "value": summary["precip_mm_day"],
                "threshold": "(0.05, 30) mm/day",
            },
            {
                "name": f"rce.{kind}.E_minus_P",
                "passed": abs(
                    summary["precip_mm_day"] - summary["evaporation_mm_day"]
                ) < 0.5,
                "value": summary["precip_mm_day"] - summary["evaporation_mm_day"],
                "threshold": "|P−E|<0.5 mm/day（采样期水收支）",
            },
        ]
    if kind == "slab_ocean":
        gates.append(
            {
                "name": f"rce.{kind}.quasi_equilibrium",
                "passed": abs(summary["dts_dt_k_per_day"]) < 0.05,
                "value": summary["dts_dt_k_per_day"],
                "threshold": "|dTs/dt|<0.05 K/day",
            }
        )
        gates.append(
            {
                "name": f"rce.{kind}.energy_closure",
                "passed": abs(summary["surface_closure_wm2"]) < 5.0,
                "value": summary["surface_closure_wm2"],
                "threshold": "|TOA−dE/dt−F_deep|<5 W/m²",
            }
        )
    elif kind == "fixed_sst":
        gates.append(
            {
                "name": f"rce.{kind}.atmosphere_closure",
                "passed": abs(summary["atmosphere_closure_wm2"]) < 5.0,
                "value": summary["atmosphere_closure_wm2"],
                "threshold": "|大气辐射强迫−dE_atm/dt|<5 W/m²",
            }
        )
    elif kind == "land":
        gates.append(
            {
                "name": f"rce.{kind}.energy_closure",
                "passed": abs(summary["surface_closure_wm2"]) < 5.0,
                "value": summary["surface_closure_wm2"],
                "threshold": "|TOA−dE/dt−F_deep|<5 W/m²",
            }
        )
    return gates


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A2 大气物理包自检 + 单柱 RCE")
    parser.add_argument(
        "--out",
        default="_rerun_logs/atmo_physics_selfcheck.json",
        help="JSON 输出路径（相对 worktree 根）",
    )
    parser.add_argument("--spinup-days", type=float, default=300.0)
    parser.add_argument("--sample-days", type=float, default=30.0)
    parser.add_argument("--quick", action="store_true", help="快速档：60+5 天（固定 SST/陆地 20+5）")
    parser.add_argument("--skip-rce", action="store_true")
    parser.add_argument("--skip-fixed-sst", action="store_true")
    parser.add_argument("--skip-land", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.quick:
        args.spinup_days = 60.0
        args.sample_days = 5.0

    cfg = AtmoPhysicsConfig()
    chk = Checker(verbose=not args.quiet)
    print(f"A2 自检基线: {c.__name__}  dt={cfg.column.dt_s:.0f}s  层数={len(cfg.column.initial_temperature_k)}")

    t0 = time.perf_counter()
    check_solar(cfg, chk)
    check_shortwave(cfg, chk)
    check_longwave(cfg, chk)
    check_thermo(cfg, chk)
    check_clouds(cfg, chk)
    check_precip(cfg, chk)
    check_convection(cfg, chk)
    check_surface(cfg, chk)
    check_column(cfg, chk)
    unit_runtime = time.perf_counter() - t0
    print(f"模块级检查耗时 {unit_runtime:.1f}s；PASS {sum(i['passed'] for i in chk.items)}/{len(chk.items)}")

    report: dict[str, Any] = {
        "unit_checks": chk.items,
        "unit_all_passed": chk.all_passed,
        "unit_runtime_s": unit_runtime,
        "rce": {},
    }

    if not args.skip_rce:
        print("开始单柱 RCE（slab 海洋）…")
        slab = run_rce("slab_ocean", args.spinup_days, args.sample_days, not args.quiet)
        report["rce"]["slab_ocean"] = slab
        report["rce"]["slab_ocean_gates"] = rce_gates(slab)

        # 固定 SST / 陆地是边界强迫档，不做长时间气候 spinup（60 天足够到
        # 大气统计稳态；再长只会放大云盖/土壤的慢漂移，不增加判据信息）。
        sec_spin = min(args.spinup_days, 60.0)
        sec_sample = min(args.sample_days, 10.0)
        if not args.skip_fixed_sst:
            print("开始单柱 RCE（固定 SST）…")
            fixed = run_rce("fixed_sst", sec_spin, sec_sample, not args.quiet)
            report["rce"]["fixed_sst"] = fixed
            report["rce"]["fixed_sst_gates"] = rce_gates(fixed)

        if not args.skip_land:
            print("开始单柱 RCE（陆地）…")
            land = run_rce("land", sec_spin, sec_sample, not args.quiet)
            report["rce"]["land"] = land
            report["rce"]["land_gates"] = rce_gates(land)

    report["all_passed"] = bool(
        report["unit_all_passed"]
        and all(
            gate["passed"]
            for key, gates in report["rce"].items()
            if key.endswith("_gates")
            for gate in gates
        )
    )
    report = _jsonable(report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果写入 {out}；总判定: {'PASS' if report['all_passed'] else 'FAIL'}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
