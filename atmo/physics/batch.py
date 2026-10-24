"""跨列批量物理内核（A1 布局驱动专用）。

定位
----
``column.py`` 的单柱 ``ColumnModel`` 是参考实现与单柱验证入口；本模块把
同一套公式沿"列"轴向量化，供 ``AtmoPhysicsDriver`` 一次推进所有列。
两套实现的方程、参数与算术顺序一致；``tests/test_atmo_physics.py`` 用
``driver(..., DriverConfig(vectorized=False))`` 与批量档逐位对拍。

约定
----
- 所有 ``(ncol, nlev)`` 数组的层索引 0 = 地表；``p_edges`` ``(ncol, nlev+1)``。
- 辐射公式与参数来源见 ``radiation.py``；本模块只改数组形状，不改物理。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from . import constants as c
from .clouds import cloud_fraction_from_state, cloud_optical_depth
from .config import AtmoPhysicsConfig, CloudConfig, PrecipConfig, RadiationConfig, SurfaceConfig, ThermoConfig
from .convection import convective_adjustment_columns
from .radiation import _delta_eddington
from .thermo import (
    d_saturation_specific_humidity_dT,
    latent_heat_vaporization,
    saturation_adjustment_columns,
    saturation_specific_humidity,
)


@dataclass
class ShortwaveColumns:
    toa_in: NDArray[np.float64]  # (ncol,)
    toa_out: NDArray[np.float64]
    surface_net: NDArray[np.float64]
    atmosphere_absorption: NDArray[np.float64]
    heating_rate: NDArray[np.float64]  # (ncol,nlev)
    flux_down: NDArray[np.float64]  # (ncol,nlev+1)
    flux_up: NDArray[np.float64]
    direct_down: NDArray[np.float64]
    optical_depth: NDArray[np.float64]  # (ncol,nlev)
    cloud_tau: NDArray[np.float64]


@dataclass
class LongwaveColumns:
    olr: NDArray[np.float64]
    surface_net_down: NDArray[np.float64]
    atmosphere_absorption: NDArray[np.float64]
    heating_rate: NDArray[np.float64]
    flux_down: NDArray[np.float64]
    flux_up: NDArray[np.float64]
    optical_depth: NDArray[np.float64]


def _sw_layer_response_columns(
    tau: NDArray[np.float64],
    omega: NDArray[np.float64],
    g: NDArray[np.float64],
    mu0: NDArray[np.float64],
    f0_perp: NDArray[np.float64],
    f_down_top: NDArray[np.float64],
    f_up_top: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """跨列版 ``radiation._sw_layer_response``（同一分支与算术顺序）。

    输入末轴为列；``f_down_top``/``f_up_top`` 可带前沿轴（仿射映射的
    3 组边界条件）。返回与输入同形状的层底扩散通量。
    """
    a = 1.5 * (1.0 - omega * g)
    b = 2.0 * (1.0 - omega)
    csrc = omega * f0_perp
    p = 1.5 * omega * f0_perp * g * mu0
    sigma0 = f_down_top + f_up_top
    d0 = f_down_top - f_up_top
    e_bot = np.exp(-tau / mu0)

    # 保守散射（b≈0）分支
    d1_cons = d0 + csrc * mu0 * (1.0 - e_bot)
    sigma1_cons = (
        sigma0
        - a * d0 * tau
        - a * csrc * mu0 * (tau - mu0 * (1.0 - e_bot))
        + p * mu0 * (1.0 - e_bot)
    )

    # 一般分支；b=0 处用安全值计算后被 np.where 丢弃
    b_safe = np.where(b > 1.0e-12, b, 1.0)
    s = np.sqrt(a * b_safe)
    denom = a * b * mu0 * mu0 - 1.0
    denom = np.where(
        np.abs(denom) < 1.0e-12,
        np.where(denom >= 0.0, 1.0e-12, -1.0e-12),
        denom,
    )
    amp = (a * csrc * mu0 * mu0 + p * mu0) / denom
    coef = p + amp / mu0
    s_safe = np.where(s > 0.0, s, 1.0)
    c1 = 0.5 * ((sigma0 - amp) + coef / s_safe - (a / s_safe) * d0)
    c2 = 0.5 * ((sigma0 - amp) - coef / s_safe + (a / s_safe) * d0)
    e_plus = np.exp(np.minimum(s * tau, 700.0))
    e_minus = np.exp(np.maximum(-s * tau, -700.0))
    sigma1_gen = c1 * e_plus + c2 * e_minus + amp * e_bot
    d1_gen = -(s / a) * (c1 * e_plus - c2 * e_minus) + (coef / a) * e_bot

    use_gen = b > 1.0e-12
    sigma1 = np.where(use_gen, sigma1_gen, sigma1_cons)
    d1 = np.where(use_gen, d1_gen, d1_cons)

    # 标量路径对 tau≤0 且无直射源直接返回输入（避免无谓舍入）
    identity = (tau <= 0.0) & (f0_perp <= 0.0)
    f_down_bot = 0.5 * (sigma1 + d1)
    f_up_bot = 0.5 * (sigma1 - d1)
    f_down_bot = np.where(identity, f_down_top, f_down_bot)
    f_up_bot = np.where(identity, f_up_top, f_up_bot)
    return f_down_bot, f_up_bot


def _sw_layer_affine_columns(
    tau: NDArray[np.float64],
    omega: NDArray[np.float64],
    g: NDArray[np.float64],
    mu0: NDArray[np.float64],
    f0_perp: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """层映射仿射形式 ``y = M x + c``；返回 ``M`` (ncol,2,2)、``c`` (ncol,2)。"""
    zeros = np.zeros_like(tau)
    ones = np.ones_like(tau)
    fd_top = np.stack([zeros, ones, zeros], axis=0)  # (3,ncol)
    fu_top = np.stack([zeros, zeros, ones], axis=0)
    fd_bot, fu_bot = _sw_layer_response_columns(
        tau[None, :], omega[None, :], g[None, :], mu0[None, :],
        f0_perp[None, :], fd_top, fu_top,
    )
    c_out = np.stack([fd_bot[0], fu_bot[0]], axis=1)
    col1 = np.stack([fd_bot[1] - c_out[:, 0], fu_bot[1] - c_out[:, 1]], axis=1)
    col2 = np.stack([fd_bot[2] - c_out[:, 0], fu_bot[2] - c_out[:, 1]], axis=1)
    mat = np.stack([col1, col2], axis=2)  # M[:, :, 0]=col1, M[:, :, 1]=col2
    return mat, c_out


def shortwave_fluxes_columns(
    p_edges: NDArray[np.float64],
    T: NDArray[np.float64],
    qv: NDArray[np.float64],
    qc: NDArray[np.float64],
    surface_albedo: NDArray[np.float64],
    toa_flux: NDArray[np.float64],
    cos_zenith: NDArray[np.float64],
    rad_cfg: RadiationConfig,
    cloud_cfg: CloudConfig,
) -> ShortwaveColumns:
    """跨列短波双流；夜列（``toa_flux≤0`` 或 ``cosz≤1e-6``）全零。"""
    p = np.asarray(p_edges, dtype=np.float64)
    t = np.asarray(T, dtype=np.float64)
    q = np.asarray(qv, dtype=np.float64)
    qcl = np.asarray(qc, dtype=np.float64)
    ncol, nlev = t.shape
    dp = p[:, :-1] - p[:, 1:]

    day = (np.asarray(toa_flux, dtype=np.float64) > 0.0) & (
        np.asarray(cos_zenith, dtype=np.float64) > 1.0e-6
    )
    toa = np.where(day, np.asarray(toa_flux, dtype=np.float64), 0.0)
    mu0 = np.where(day, np.asarray(cos_zenith, dtype=np.float64), 0.5)
    f0_perp = toa / mu0
    direct_top = mu0 * f0_perp

    mass_path = dp / c.GRAVITY
    tau_vap = rad_cfg.sw_vapor_abs_coeff * q * mass_path
    tau_ray = (
        rad_cfg.sw_rayleigh_tau_surface
        * dp
        / np.maximum(dp.sum(axis=1, keepdims=True), 1.0e-12)
    )
    tau_cloud = cloud_optical_depth(qcl, dp, cloud_cfg)
    tau = tau_vap + tau_ray + tau_cloud

    nz = tau > 1.0e-14
    tau_safe = np.where(nz, tau, 1.0)
    omega = np.where(
        nz,
        (tau_ray + rad_cfg.cloud_sw_single_scatter_albedo * tau_cloud) / tau_safe,
        0.0,
    )
    denom_g = tau_ray + tau_cloud
    g = np.where(
        nz,
        np.where(
            denom_g > 1.0e-14,
            rad_cfg.cloud_sw_asymmetry * tau_cloud / np.maximum(denom_g, 1.0e-14),
            0.0,
        ),
        0.0,
    )
    omega = np.clip(omega, 0.0, 1.0)
    g = np.clip(g, -1.0, 1.0)
    tau_s, omega_s, g_s = _delta_eddington(tau, omega, g)

    # 直射：界面 nlev=大气顶；逐层向下衰减
    direct = np.zeros((ncol, nlev + 1), dtype=np.float64)
    direct[:, nlev] = direct_top
    for i in range(nlev - 1, -1, -1):
        direct[:, i] = direct[:, i + 1] * np.exp(
            -np.minimum(tau_s[:, i] / mu0, 700.0)
        )

    mats = np.empty((ncol, nlev, 2, 2), dtype=np.float64)
    offs = np.empty((ncol, nlev, 2), dtype=np.float64)
    for i in range(nlev):
        f0_local = direct[:, i + 1] / mu0
        mat, off = _sw_layer_affine_columns(
            tau_s[:, i], omega_s[:, i], g_s[:, i], mu0, f0_local
        )
        mats[:, i] = mat
        offs[:, i] = off

    # 自顶向下传播：F↓_top=0，F↑_top=U 未知，各界面对 U 仿射
    a_d = np.zeros(ncol, dtype=np.float64)
    b_d = np.zeros(ncol, dtype=np.float64)
    a_u = np.ones(ncol, dtype=np.float64)
    b_u = np.zeros(ncol, dtype=np.float64)
    for i in range(nlev - 1, -1, -1):
        nd = mats[:, i, 0, 0] * a_d + mats[:, i, 0, 1] * a_u
        bd = mats[:, i, 0, 0] * b_d + mats[:, i, 0, 1] * b_u + offs[:, i, 0]
        nu = mats[:, i, 1, 0] * a_d + mats[:, i, 1, 1] * a_u
        bu = mats[:, i, 1, 0] * b_d + mats[:, i, 1, 1] * b_u + offs[:, i, 1]
        a_d, b_d, a_u, b_u = nd, bd, nu, bu

    alb = np.clip(np.asarray(surface_albedo, dtype=np.float64), 0.0, 1.0)

    def propagate(u_top_value: NDArray[np.float64]):
        fd = np.zeros((ncol, nlev + 1), dtype=np.float64)
        fu = np.zeros((ncol, nlev + 1), dtype=np.float64)
        fu[:, nlev] = u_top_value
        for k in range(nlev - 1, -1, -1):
            fd_top = fd[:, k + 1]
            fu_top = fu[:, k + 1]
            fd[:, k] = np.maximum(
                mats[:, k, 0, 0] * fd_top
                + mats[:, k, 0, 1] * fu_top
                + offs[:, k, 0],
                0.0,
            )
            fu[:, k] = np.maximum(
                mats[:, k, 1, 0] * fd_top
                + mats[:, k, 1, 1] * fu_top
                + offs[:, k, 1],
                0.0,
            )
        return fd, fu

    denom = a_u - alb * a_d
    u_top = np.where(
        np.abs(denom) < 1.0e-14,
        0.0,
        (alb * (b_d + direct[:, 0]) - b_u) / np.where(np.abs(denom) < 1.0e-14, 1.0, denom),
    )
    u_top = np.maximum(u_top, 0.0)
    flux_down, flux_up = propagate(u_top)

    scale = np.maximum(np.asarray(toa_flux, dtype=np.float64), 1.0)
    for _ in range(12):
        residual = flux_up[:, 0] - alb * (flux_down[:, 0] + direct[:, 0])
        done = np.abs(residual) <= 1.0e-12 * scale
        if np.all(done):
            break
        deriv = np.where(np.abs(a_u) > 1.0e-9, a_u, 1.0)
        u_top = np.where(
            done,
            u_top,
            np.maximum(u_top - residual / deriv, 0.0),
        )
        flux_down, flux_up = propagate(u_top)

    net_top = (flux_down[:, 1:] + direct[:, 1:]) - flux_up[:, 1:]
    net_bot = (flux_down[:, :-1] + direct[:, :-1]) - flux_up[:, :-1]
    absorbed = net_top - net_bot
    mass = dp / c.GRAVITY
    heating = absorbed / np.maximum(c.CPD * mass, 1.0e-12)
    surface_net = (1.0 - alb) * (flux_down[:, 0] + direct[:, 0])
    atmos_abs = np.sum(absorbed, axis=1)
    toa_net = direct[:, nlev] - flux_up[:, nlev]

    # 夜列：标量路径整体早退为 0
    day_i = day[:, None]
    heating = np.where(day_i, heating, 0.0)
    flux_down = np.where(day_i, flux_down, 0.0)
    flux_up = np.where(day_i, flux_up, 0.0)
    direct = np.where(day_i, direct, 0.0)
    tau_s = np.where(day_i, tau_s, 0.0)
    tau_cloud = np.where(day_i, tau_cloud, 0.0)
    surface_net = np.where(day, surface_net, 0.0)
    atmos_abs = np.where(day, atmos_abs, 0.0)
    toa_net = np.where(day, toa_net, 0.0)

    return ShortwaveColumns(
        toa_in=direct[:, nlev].copy(),
        toa_out=(flux_up[:, nlev]).copy(),
        surface_net=surface_net,
        atmosphere_absorption=atmos_abs,
        heating_rate=heating,
        flux_down=flux_down,
        flux_up=flux_up,
        direct_down=direct,
        optical_depth=tau_s,
        cloud_tau=tau_cloud,
    )


def longwave_fluxes_columns(
    p_edges: NDArray[np.float64],
    T: NDArray[np.float64],
    qv: NDArray[np.float64],
    qc: NDArray[np.float64],
    surface_temperature: NDArray[np.float64],
    surface_emissivity: NDArray[np.float64],
    rad_cfg: RadiationConfig,
    cloud_cfg: CloudConfig,
) -> LongwaveColumns:
    """跨列长波灰体（或线性 OLR 对照档）。"""
    p = np.asarray(p_edges, dtype=np.float64)
    t = np.asarray(T, dtype=np.float64)
    q = np.asarray(qv, dtype=np.float64)
    qcl = np.asarray(qc, dtype=np.float64)
    ncol, nlev = t.shape
    dp = p[:, :-1] - p[:, 1:]
    mass = dp / c.GRAVITY
    ts = np.asarray(surface_temperature, dtype=np.float64)
    eps_s = np.clip(np.asarray(surface_emissivity, dtype=np.float64), 0.0, 1.0)
    sigma_ts4 = c.SIGMA_SB * ts ** 4

    if rad_cfg.lw_mode == "linear":
        olr = rad_cfg.lw_linear_a + rad_cfg.lw_linear_b * (ts - rad_cfg.lw_linear_t_ref)
        olr = np.maximum(olr, 0.0)
        flux_down_surface = rad_cfg.lw_linear_down_fraction * olr
        surface_net_down = flux_down_surface - eps_s * sigma_ts4
        atm_abs = -olr - surface_net_down
        tau_lw = rad_cfg.lw_vapor_abs_coeff * q * mass
        tau_lw = tau_lw + rad_cfg.lw_cloud_abs_coeff * (
            qcl * dp / c.GRAVITY
        )
        w = tau_lw + 1.0e-12 * mass
        w = w / np.maximum(w.sum(axis=1, keepdims=True), 1.0e-30)
        heating = atm_abs[:, None] * w / np.maximum(c.CPD * mass, 1.0e-12)
        return LongwaveColumns(
            olr=olr,
            surface_net_down=surface_net_down,
            atmosphere_absorption=atm_abs,
            heating_rate=heating,
            flux_down=np.broadcast_to(flux_down_surface[:, None], (ncol, nlev + 1)).copy(),
            flux_up=np.broadcast_to(olr[:, None], (ncol, nlev + 1)).copy(),
            optical_depth=tau_lw,
        )

    tau_lw = rad_cfg.lw_vapor_abs_coeff * q * mass
    tau_lw = tau_lw + rad_cfg.lw_cloud_abs_coeff * (qcl * dp / c.GRAVITY)
    eps = 1.0 - np.exp(-np.maximum(tau_lw, 0.0))
    sigma_t4 = c.SIGMA_SB * t ** 4

    f_down = np.zeros((ncol, nlev + 1), dtype=np.float64)
    for i in range(nlev - 1, -1, -1):
        f_down[:, i] = (
            f_down[:, i + 1] * (1.0 - eps[:, i]) + eps[:, i] * sigma_t4[:, i]
        )

    f_up = np.zeros((ncol, nlev + 1), dtype=np.float64)
    f_up[:, 0] = eps_s * sigma_ts4 + (1.0 - eps_s) * f_down[:, 0]
    for i in range(nlev):
        f_up[:, i + 1] = (
            f_up[:, i] * (1.0 - eps[:, i]) + eps[:, i] * sigma_t4[:, i]
        )

    net_top = f_down[:, 1:] - f_up[:, 1:]
    net_bot = f_down[:, :-1] - f_up[:, :-1]
    convergence = net_top - net_bot
    heating = convergence / np.maximum(c.CPD * mass, 1.0e-12)
    surface_net_down = f_down[:, 0] - f_up[:, 0]
    atm_abs = np.sum(convergence, axis=1)

    return LongwaveColumns(
        olr=f_up[:, nlev].copy(),
        surface_net_down=surface_net_down,
        atmosphere_absorption=atm_abs,
        heating_rate=heating,
        flux_down=f_down,
        flux_up=f_up,
        optical_depth=tau_lw,
    )


# ---------------------------------------------------------------- 湿过程
def _energy_no_geopotential_columns(
    T: NDArray[np.float64],
    qv: NDArray[np.float64],
    qc: NDArray[np.float64],
    cfg: ThermoConfig,
) -> NDArray[np.float64]:
    """跨列版 ``thermo._energy_no_geopotential``（同一算术顺序）。"""
    base = c.CPD * (T - c.T0) + c.LV0 * qv
    if cfg.latent_heat_mode == "temperature_dependent":
        base = base + qv * c.CPV * (T - c.T0) + qc * c.CW * (T - c.T0)
    return base


def _solve_temperature_columns(
    qv: NDArray[np.float64],
    qc: NDArray[np.float64],
    target_h_no_z: NDArray[np.float64],
    cfg: ThermoConfig,
) -> NDArray[np.float64]:
    """跨列版 ``thermo.solve_temperature_for_energy``（h 对 T 线性，解析解）。"""
    if cfg.latent_heat_mode == "constant":
        denom = c.CPD
    else:
        denom = c.CPD + qv * c.CPV + qc * c.CW
    return c.T0 + (target_h_no_z - c.LV0 * qv) / denom


def _evaporate_liquid_columns(
    T: NDArray[np.float64],
    qv: NDArray[np.float64],
    liquid: NDArray[np.float64],
    p: NDArray[np.float64],
    requested: NDArray[np.float64],
    cfg: ThermoConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """跨列版 ``thermo.evaporate_liquid``（同一 6 次迭代与收敛判据）。"""
    t0 = np.asarray(T, dtype=np.float64)
    q0 = np.asarray(qv, dtype=np.float64)
    l0 = np.asarray(liquid, dtype=np.float64)
    req = np.asarray(requested, dtype=np.float64)
    press = np.asarray(p, dtype=np.float64)
    active = (req > 0.0) & (l0 > 0.0)
    if not np.any(active):
        return t0.copy(), q0.copy(), l0.copy(), np.zeros_like(t0)

    h_target = _energy_no_geopotential_columns(t0, q0, l0, cfg)
    evap = np.where(active, np.minimum(req, l0), 0.0)
    qs = saturation_specific_humidity(t0, press, cfg)
    evap = np.where(active, np.minimum(evap, np.maximum(qs - q0, 0.0)), evap)
    active = active & (evap > 0.0)
    if not np.any(active):
        return t0.copy(), q0.copy(), l0.copy(), np.zeros_like(t0)

    alive = active.copy()
    dead = np.zeros_like(active)
    for _ in range(6):
        if not np.any(alive):
            break
        t_try = _solve_temperature_columns(q0 + evap, l0 - evap, h_target, cfg)
        qs_new = saturation_specific_humidity(t_try, press, cfg)
        evap_new = np.minimum(
            np.minimum(req, l0), np.maximum(qs_new - q0, 0.0)
        )
        conv = alive & (np.abs(evap_new - evap) <= 1.0e-15)
        gone = alive & ~conv & (evap_new <= 0.0)
        evap = np.where(alive, evap_new, evap)
        alive = alive & ~conv & ~gone
        dead |= gone

    t_new = _solve_temperature_columns(q0 + evap, l0 - evap, h_target, cfg)
    ok = active & ~dead
    return (
        np.where(ok, t_new, t0),
        np.where(ok, q0 + evap, q0),
        np.where(ok, l0 - evap, l0),
        np.where(ok, evap, 0.0),
    )


def _evaporate_cloud_water_columns(
    T: NDArray[np.float64],
    qv: NDArray[np.float64],
    qc: NDArray[np.float64],
    p: NDArray[np.float64],
    dt: float,
    cloud_cfg: CloudConfig,
    thermo_cfg: ThermoConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """跨列版 ``clouds.evaporate_cloud_water``。"""
    qs = saturation_specific_humidity(T, p, thermo_cfg)
    rh = np.clip(qv / np.maximum(qs, 1.0e-12), 0.0, 1.0)
    requested = cloud_cfg.cloud_evap_coeff * (1.0 - rh) * qc * dt
    return _evaporate_liquid_columns(T, qv, qc, p, requested, thermo_cfg)


def _precip_step_columns(
    p_edges: NDArray[np.float64],
    T: NDArray[np.float64],
    qv: NDArray[np.float64],
    qc: NDArray[np.float64],
    cloud_fraction_layers: NDArray[np.float64],
    dt: float,
    cfg: PrecipConfig,
    thermo_cfg: ThermoConfig,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
    NDArray[np.float64], NDArray[np.float64],
]:
    """跨列版 ``precip.precip_step``（顶层→地表扫描，逐列掩码）。

    返回 ``(T, qv, qc, surface_flux, surface_precip, rain_evap, auto, acc)``。
    """
    p = np.asarray(p_edges, dtype=np.float64)
    t = np.array(T, dtype=np.float64, copy=True)
    q = np.array(qv, dtype=np.float64, copy=True)
    qcl = np.array(qc, dtype=np.float64, copy=True)
    cf_in = np.asarray(cloud_fraction_layers, dtype=np.float64)
    ncol, nlev = t.shape
    dp = p[:, :-1] - p[:, 1:]
    mass = dp / c.GRAVITY
    p_mid = 0.5 * (p[:, :-1] + p[:, 1:])
    rho = p_mid / (c.RD * np.maximum(t, 150.0))
    zeros = np.zeros(ncol, dtype=np.float64)

    if not cfg.enabled or dt <= 0.0:
        return t, q, qcl, zeros, zeros, zeros, zeros, zeros

    rain_flux = np.zeros(ncol, dtype=np.float64)
    evap_total = np.zeros(ncol, dtype=np.float64)
    auto_total = np.zeros(ncol, dtype=np.float64)
    acc_total = np.zeros(ncol, dtype=np.float64)
    vt = max(cfg.rain_terminal_velocity, 1.0e-6)

    for i in range(nlev - 1, -1, -1):
        if cfg.cloud_fraction_weighted:
            cf = np.clip(cf_in[:, i], cfg.cloud_fraction_min, 1.0)
            qc_in = np.maximum(qcl[:, i] / cf, 0.0)
        else:
            cf = np.ones(ncol, dtype=np.float64)
            qc_in = np.maximum(qcl[:, i], 0.0)
        qr_local = np.where(rain_flux > 0.0, rain_flux / (rho[:, i] * vt), 0.0)

        auto_rate = cfg.autoconversion_rate * np.maximum(qc_in - cfg.qc_crit, 0.0)
        auto_grid = cf * auto_rate if cfg.cloud_fraction_weighted else auto_rate
        acc_rate = cfg.accretion_coeff * qc_in * qr_local
        acc_grid = cf * acc_rate if cfg.cloud_fraction_weighted else acc_rate

        loss_rate = np.maximum(auto_grid + acc_grid, 0.0)
        loss_q = np.minimum(np.maximum(qcl[:, i], 0.0), loss_rate * dt)
        has_loss = loss_q > 0.0
        if np.any(has_loss):
            total_rate = np.maximum(auto_grid + acc_grid, 1.0e-30)
            auto_part = loss_q * auto_grid / total_rate
            acc_part = loss_q * acc_grid / total_rate
            auto_total += np.where(has_loss, auto_part * mass[:, i], 0.0)
            acc_total += np.where(has_loss, acc_part * mass[:, i], 0.0)
        qcl[:, i] -= loss_q
        rain_flux += np.where(has_loss, loss_q * mass[:, i] / dt, 0.0)

        if np.any(rain_flux > 0.0):
            qs = saturation_specific_humidity(t[:, i], p_mid[:, i], thermo_cfg)
            rh = np.clip(q[:, i] / np.maximum(qs, 1.0e-12), 0.0, 1.0)
            pos = rain_flux > 0.0
            qr_now = np.where(pos, rain_flux / (rho[:, i] * vt), 0.0)
            evap_q = cfg.rain_evap_coeff * (1.0 - rh) * qr_now * dt
            deficit = np.maximum(qs - q[:, i], 0.0)
            evap_q = np.minimum(
                np.minimum(evap_q, deficit),
                np.where(pos, rain_flux * dt / mass[:, i], 0.0),
            )
            evap_q = np.where(pos, evap_q, 0.0)
            do = evap_q > 0.0
            if np.any(do):
                t_new, q_new, _, evap_actual = _evaporate_liquid_columns(
                    t[:, i], q[:, i], evap_q, p_mid[:, i], evap_q, thermo_cfg
                )
                t[:, i] = np.where(do, t_new, t[:, i])
                q[:, i] = np.where(do, q_new, q[:, i])
                evap_mass = np.where(do, evap_actual * mass[:, i], 0.0)
                rain_flux = np.maximum(rain_flux - evap_mass / dt, 0.0)
                evap_total += evap_mass

    surface_flux = np.maximum(rain_flux, 0.0)
    return (
        t, q, np.maximum(qcl, 0.0), surface_flux, surface_flux * dt,
        evap_total, auto_total, acc_total,
    )


# ---------------------------------------------------------------- 地表
def _bulk_fluxes_columns(
    surface_temperature: NDArray[np.float64],
    air_temperature: NDArray[np.float64],
    air_specific_humidity: NDArray[np.float64],
    air_pressure: NDArray[np.float64],
    wind_speed: NDArray[np.float64],
    beta: NDArray[np.float64],
    drag_coeff_heat: float,
    drag_coeff_moisture: float,
    thermo_cfg: ThermoConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """跨列版 ``surface.bulk_fluxes``。"""
    rho_a = air_pressure / (c.RD * np.maximum(air_temperature, 150.0))
    lv = latent_heat_vaporization(surface_temperature, thermo_cfg)
    qs = saturation_specific_humidity(surface_temperature, air_pressure, thermo_cfg)
    sensible = rho_a * c.CPD * drag_coeff_heat * wind_speed * (
        surface_temperature - air_temperature
    )
    evaporation = (
        rho_a * drag_coeff_moisture * wind_speed * np.maximum(beta, 0.0)
        * (qs - air_specific_humidity)
    )
    return sensible, lv * evaporation, evaporation


def _solve_ocean_temperature_columns(
    surface_temperature: NDArray[np.float64],
    sw_net_surface: NDArray[np.float64],
    lw_down_surface: NDArray[np.float64],
    air_temperature: NDArray[np.float64],
    air_specific_humidity: NDArray[np.float64],
    air_pressure: NDArray[np.float64],
    dt: float,
    wind_speed: NDArray[np.float64],
    emissivity: NDArray[np.float64],
    mixed_layer_depth: float,
    drag_coeff_heat: float,
    drag_coeff_moisture: float,
    thermo_cfg: ThermoConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """跨列版 ``surface.solve_ocean_temperature``（Newton，逐列掩码）。"""
    ts0 = np.asarray(surface_temperature, dtype=np.float64)
    heat_capacity = c.RHO_WATER * c.CW * mixed_layer_depth
    rho_a = air_pressure / (c.RD * np.maximum(air_temperature, 150.0))
    lv = latent_heat_vaporization(ts0, thermo_cfg)
    ts = ts0.copy()
    active = np.ones_like(ts, dtype=bool)
    for _ in range(10):
        if not np.any(active):
            break
        h, le, _ = _bulk_fluxes_columns(
            ts, air_temperature, air_specific_humidity, air_pressure,
            wind_speed, np.ones_like(ts), drag_coeff_heat, drag_coeff_moisture,
            thermo_cfg,
        )
        f = heat_capacity * (ts - ts0) / dt - (
            sw_net_surface + emissivity * (lw_down_surface - c.SIGMA_SB * ts ** 4)
            - h - le
        )
        qs_prime = d_saturation_specific_humidity_dT(ts, air_pressure, thermo_cfg)
        df = (
            heat_capacity / dt
            + 4.0 * emissivity * c.SIGMA_SB * ts ** 3
            + rho_a * c.CPD * drag_coeff_heat * wind_speed
            + rho_a * lv * drag_coeff_moisture * wind_speed * qs_prime
        )
        step = f / np.maximum(df, 1.0e-12)
        ts_new = np.clip(ts - step, 150.0, 400.0)
        done = active & (np.abs(step) < 1.0e-8)
        ts = np.where(active, ts_new, ts)
        active = active & ~done
    h, le, evap = _bulk_fluxes_columns(
        ts, air_temperature, air_specific_humidity, air_pressure,
        wind_speed, np.ones_like(ts), drag_coeff_heat, drag_coeff_moisture,
        thermo_cfg,
    )
    return h, le, evap, ts


def _solve_land_surface_columns(
    soil_temperature: NDArray[np.float64],
    sw_net_surface: NDArray[np.float64],
    lw_down_surface: NDArray[np.float64],
    air_temperature: NDArray[np.float64],
    air_specific_humidity: NDArray[np.float64],
    air_pressure: NDArray[np.float64],
    dt: float,
    wind_speed: NDArray[np.float64],
    beta: NDArray[np.float64],
    cfg: SurfaceConfig,
    thermo_cfg: ThermoConfig,
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
    NDArray[np.float64], NDArray[np.float64],
]:
    """跨列版 ``surface.solve_land_surface``（土壤隐式仿射 + 表层 Newton）。

    返回 ``(H, LE, evaporation, Ts, soil_new)``；``soil_new`` 形状 (nsoil,ncol)。
    """
    emissivity = cfg.emissivity_land
    n = cfg.soil_layers
    t_soil = np.asarray(soil_temperature, dtype=np.float64)[:n]
    d = np.asarray(cfg.soil_depth_m[:n], dtype=np.float64)
    cap = np.asarray(cfg.soil_volumetric_heat_capacity[:n], dtype=np.float64)
    k = np.asarray(cfg.soil_conductivity[:n], dtype=np.float64)
    k_surf = k[0] / (0.5 * d[0])
    rho_a = air_pressure / (c.RD * np.maximum(air_temperature, 150.0))

    if n == 1:
        k_deep = k[0] / (0.5 * d[0])
        denom = cap[0] * d[0] / dt + k_surf + k_deep
        alpha = (cap[0] * d[0] / dt * t_soil[0]
                 + k_deep * cfg.deep_soil_temperature) / denom
        alpha = alpha[None, :]
        beta_vec = np.full((1, t_soil.shape[1]), k_surf / denom, dtype=np.float64)
    else:
        k12 = (d[0] + d[1]) / (d[0] / k[0] + d[1] / k[1])
        k_deep = k[1] / (0.5 * d[1])
        c1 = cap[0] * d[0] / dt
        c2 = cap[1] * d[1] / dt
        a11 = c1 + k_surf + k12
        a12 = -k12
        a21 = -k12
        a22 = c2 + k12 + k_deep
        det = a11 * a22 - a12 * a21
        alpha = np.stack([
            (c1 * t_soil[0] * a22
             + k12 * (c2 * t_soil[1] + k_deep * cfg.deep_soil_temperature)) / det,
            (a11 * (c2 * t_soil[1] + k_deep * cfg.deep_soil_temperature)
             + k12 * c1 * t_soil[0]) / det,
        ], axis=0)
        beta_vec = np.broadcast_to(
            np.stack([k_surf * a22 / det, k_surf * k12 / det], axis=0)[:, None],
            (2, t_soil.shape[1]),
        ).copy()

    ts = t_soil[0].copy()
    active = np.ones_like(ts, dtype=bool)
    for _ in range(50):
        if not np.any(active):
            break
        t1_new = alpha[0] + beta_vec[0] * ts
        h, le, _ = _bulk_fluxes_columns(
            ts, air_temperature, air_specific_humidity, air_pressure,
            wind_speed, beta, cfg.drag_coeff_heat, cfg.drag_coeff_moisture,
            thermo_cfg,
        )
        f = (
            sw_net_surface
            + emissivity * (lw_down_surface - c.SIGMA_SB * ts ** 4)
            - h - le
            - k_surf * (ts - t1_new)
        )
        qs_prime = d_saturation_specific_humidity_dT(ts, air_pressure, thermo_cfg)
        df = (
            -4.0 * emissivity * c.SIGMA_SB * ts ** 3
            - rho_a * c.CPD * cfg.drag_coeff_heat * wind_speed
            - rho_a * cfg.drag_coeff_moisture * wind_speed * np.maximum(beta, 0.0)
            * latent_heat_vaporization(ts, thermo_cfg) * qs_prime
            - k_surf * (1.0 - beta_vec[0])
        )
        step = -f / np.minimum(df, -1.0e-12)
        ts_new = np.clip(ts + step, 150.0, 400.0)
        done = active & (np.abs(step) < 1.0e-10)
        ts = np.where(active, ts_new, ts)
        active = active & ~done

    soil_new = alpha + beta_vec * ts[None, :]
    h, le, evap = _bulk_fluxes_columns(
        ts, air_temperature, air_specific_humidity, air_pressure,
        wind_speed, beta, cfg.drag_coeff_heat, cfg.drag_coeff_moisture,
        thermo_cfg,
    )
    return h, le, evap, ts, soil_new


# ---------------------------------------------------------------- 整步编排
@dataclass
class BatchStepResult:
    T: NDArray[np.float64]
    qv: NDArray[np.float64]
    qc: NDArray[np.float64]
    Ts: NDArray[np.float64]
    soil_T: NDArray[np.float64]
    sw_toa_in: NDArray[np.float64]
    sw_toa_out: NDArray[np.float64]
    sw_surface_net: NDArray[np.float64]
    sw_atm_absorption: NDArray[np.float64]
    olr: NDArray[np.float64]
    lw_surface_net_down: NDArray[np.float64]
    lw_atm_absorption: NDArray[np.float64]
    sensible: NDArray[np.float64]
    latent: NDArray[np.float64]
    evaporation: NDArray[np.float64]
    precip_large_scale: NDArray[np.float64]
    precip_convective: NDArray[np.float64]
    precip_total: NDArray[np.float64]
    rain_evaporation: NDArray[np.float64]
    autoconversion: NDArray[np.float64]
    accretion: NDArray[np.float64]
    cloud_cover: NDArray[np.float64]
    lwp: NDArray[np.float64]
    precipitable_water: NDArray[np.float64]
    toa_net: NDArray[np.float64]
    energy_residual: NDArray[np.float64]
    water_residual: NDArray[np.float64]
    deep_flux: NDArray[np.float64]
    min_temperature: NDArray[np.float64]
    max_temperature: NDArray[np.float64]
    nan_free: NDArray[np.bool_]


def batch_step_columns(
    p_edges: NDArray[np.float64],
    T: NDArray[np.float64],
    qv: NDArray[np.float64],
    qc: NDArray[np.float64],
    Ts: NDArray[np.float64],
    soil_T: NDArray[np.float64],
    land: NDArray[np.bool_],
    surface_albedo: NDArray[np.float64],
    toa_sw: NDArray[np.float64],
    cosz: NDArray[np.float64],
    wind_low: NDArray[np.float64] | None,
    dt: float,
    cfg: AtmoPhysicsConfig,
    ocean_fixed_sst: bool,
) -> BatchStepResult:
    """跨列整步：与 ``ColumnModel.step`` 同一顺序、同一公式、同一舍入。

    输入层索引 0 = 地表；``soil_T`` 形状 ``(nsoil,ncol)``。批量档不包含
    T 剖面/水汽松弛（驱动侧在开启松弛时回退逐列标量路径）。
    """
    p = np.asarray(p_edges, dtype=np.float64)
    t = np.array(T, dtype=np.float64, copy=True)
    q = np.array(qv, dtype=np.float64, copy=True)
    qcl = np.array(qc, dtype=np.float64, copy=True)
    ts = np.array(Ts, dtype=np.float64, copy=True)
    soil = np.array(soil_T, dtype=np.float64, copy=True)
    land = np.asarray(land, dtype=bool)
    ncol, nlev = t.shape
    dt = float(dt)

    dp = p[:, :-1] - p[:, 1:]
    mass = dp / c.GRAVITY
    p_mid = 0.5 * (p[:, :-1] + p[:, 1:])
    nsoil = int(cfg.surface.soil_layers)

    # 静力高度（界面 0 = 地表）：与 ColumnState.heights() 同一累加顺序。
    z_edges = np.zeros((ncol, nlev + 1), dtype=np.float64)
    for i in range(nlev):
        z_edges[:, i + 1] = z_edges[:, i] + (
            c.RD * t[:, i] / c.GRAVITY * np.log(p[:, i] / p[:, i + 1])
        )
    z_mid = 0.5 * (z_edges[:, :-1] + z_edges[:, 1:])

    # ---- 初态账本 ----
    e0_atm = np.sum(
        _energy_no_geopotential_columns(t, q, qcl, cfg.thermo) * mass, axis=1
    )
    e0_surface = np.where(
        land,
        np.sum(
            np.asarray(cfg.surface.soil_volumetric_heat_capacity[:nsoil])[:, None]
            * np.asarray(cfg.surface.soil_depth_m[:nsoil])[:, None]
            * (soil[:nsoil] - c.T0),
            axis=0,
        ),
        c.RHO_WATER * c.CW * cfg.surface.ocean_mixed_layer_depth * (ts - c.T0),
    )
    e0 = e0_atm + e0_surface
    w0 = np.sum((q + qcl) * mass, axis=1)

    emissivity = np.where(
        land, cfg.surface.emissivity_land, cfg.surface.emissivity_ocean
    )
    beta = np.where(land, cfg.surface.land_evap_beta, 1.0)
    default_wind = float(cfg.surface.wind_speed)
    if wind_low is None:
        wind = np.full(ncol, default_wind, dtype=np.float64)
    else:
        local = np.hypot(np.asarray(wind_low[0]), np.asarray(wind_low[1]))
        wind = np.where(local > 0.0, local, default_wind)

    # ---- 短波 / 先按旧 Ts 的长波 ----
    sw = shortwave_fluxes_columns(
        p, t, q, qcl, surface_albedo, toa_sw, cosz, cfg.radiation, cfg.cloud
    )
    lw_surface = longwave_fluxes_columns(
        p, t, q, qcl, ts, emissivity, cfg.radiation, cfg.cloud
    )

    # ---- 地表温度隐式更新 ----
    p_low = p_mid[:, 0]
    t_low = t[:, 0]
    q_low = q[:, 0]
    ts_ocean = ts.copy()
    soil_ocean = soil.copy()
    h_ocean, le_ocean, evap_ocean = _bulk_fluxes_columns(
        ts, t_low, q_low, p_low, wind, np.ones(ncol), cfg.surface.drag_coeff_heat,
        cfg.surface.drag_coeff_moisture, cfg.thermo,
    )
    if not ocean_fixed_sst:
        h_ocean, le_ocean, evap_ocean, ts_ocean = _solve_ocean_temperature_columns(
            ts, sw.surface_net, lw_surface.flux_down[:, 0], t_low, q_low, p_low,
            dt, wind, emissivity, cfg.surface.ocean_mixed_layer_depth,
            cfg.surface.drag_coeff_heat, cfg.surface.drag_coeff_moisture,
            cfg.thermo,
        )
    h_land, le_land, evap_land, ts_land, soil_land = _solve_land_surface_columns(
        soil, sw.surface_net, lw_surface.flux_down[:, 0], t_low, q_low, p_low,
        dt, wind, beta, cfg.surface, cfg.thermo,
    )
    h_flux = np.where(land, h_land, h_ocean)
    le_flux = np.where(land, le_land, le_ocean)
    evap = np.where(land, evap_land, evap_ocean)
    ts_new = np.where(land, ts_land, ts_ocean)
    soil_new = np.where(land[None, :], soil_land, soil_ocean)

    # ---- 新 Ts 的长波（加热率与账本用这一份）----
    lw = longwave_fluxes_columns(
        p, t, q, qcl, ts_new, emissivity, cfg.radiation, cfg.cloud
    )

    t = t + (sw.heating_rate + lw.heating_rate) * dt
    mass_low = mass[:, 0]
    evap = np.maximum(evap, -q[:, 0] * mass_low / dt)
    t[:, 0] += h_flux * dt / (c.CPD * mass_low)
    q[:, 0] = np.maximum(q[:, 0] + evap * dt / mass_low, 0.0)

    # ---- 相变 / 云水蒸发 ----
    t, q, qcl, _ = saturation_adjustment_columns(
        t, q, qcl, p_mid, cfg.thermo
    )
    t, q, qcl, _ = _evaporate_cloud_water_columns(
        t, q, qcl, p_mid, dt, cfg.cloud, cfg.thermo
    )

    # ---- 降水 ----
    cf = cloud_fraction_from_state(q, t, p_mid, cfg.cloud, cfg.thermo)
    (t, q, qcl, precip_flux, precip_large, rain_evap,
     auto_c, acc_c) = _precip_step_columns(
        p, t, q, qcl, cf, dt, cfg.precip, cfg.thermo
    )

    # ---- 对流 ----
    conv = convective_adjustment_columns(
        p, t, q, qcl, z_mid, cfg.convection, cfg.thermo
    )
    t, q, qcl = conv.T, conv.qv, conv.qc

    q = np.maximum(q, 0.0)
    qcl = np.maximum(qcl, 0.0)

    # ---- 末态账本 ----
    e1_atm = np.sum(
        _energy_no_geopotential_columns(t, q, qcl, cfg.thermo) * mass, axis=1
    )
    e1_surface = np.where(
        land,
        np.sum(
            np.asarray(cfg.surface.soil_volumetric_heat_capacity[:nsoil])[:, None]
            * np.asarray(cfg.surface.soil_depth_m[:nsoil])[:, None]
            * (soil_new[:nsoil] - c.T0),
            axis=0,
        ),
        c.RHO_WATER * c.CW * cfg.surface.ocean_mixed_layer_depth * (ts_new - c.T0),
    )
    e1 = e1_atm + e1_surface
    w1 = np.sum((q + qcl) * mass, axis=1)

    toa_net = sw.toa_in - sw.toa_out - lw.olr
    fixed = (~land) & bool(ocean_fixed_sst)
    forcing_energy = np.where(
        fixed,
        (sw.atmosphere_absorption + lw.atmosphere_absorption
         + h_flux + le_flux) * dt,
        toa_net * dt,
    )
    d_deep = cfg.surface.soil_depth_m[nsoil - 1]
    k_deep = cfg.surface.soil_conductivity[nsoil - 1] / (0.5 * d_deep)
    deep_flux = np.where(
        land,
        k_deep * (soil_new[nsoil - 1] - cfg.surface.deep_soil_temperature),
        0.0,
    )
    forcing_energy = forcing_energy - np.where(land, deep_flux * dt, 0.0)

    energy_residual = (e1 - e0) - forcing_energy + conv.precip_energy
    water_residual = (
        (w1 - w0) - (evap * dt - precip_large - conv.precip)
    )

    cf_out = cloud_fraction_from_state(q, t, p_mid, cfg.cloud, cfg.thermo)
    cloud_cover = 1.0 - np.prod(
        1.0 - np.clip(cf_out, 0.0, 1.0), axis=1
    )
    lwp = np.sum(qcl * dp / c.GRAVITY, axis=1)
    pw = np.sum(q * mass, axis=1)
    nan_free = (
        np.all(np.isfinite(t), axis=1)
        & np.all(np.isfinite(q), axis=1)
        & np.all(np.isfinite(qcl), axis=1)
        & np.isfinite(ts_new)
    )

    return BatchStepResult(
        T=t, qv=q, qc=qcl, Ts=ts_new, soil_T=soil_new,
        sw_toa_in=sw.toa_in, sw_toa_out=sw.toa_out,
        sw_surface_net=sw.surface_net,
        sw_atm_absorption=sw.atmosphere_absorption,
        olr=lw.olr, lw_surface_net_down=lw.surface_net_down,
        lw_atm_absorption=lw.atmosphere_absorption,
        sensible=h_flux, latent=le_flux, evaporation=evap,
        precip_large_scale=precip_large,
        precip_convective=conv.precip,
        precip_total=precip_large + conv.precip,
        rain_evaporation=rain_evap,
        autoconversion=auto_c, accretion=acc_c,
        cloud_cover=cloud_cover, lwp=lwp, precipitable_water=pw,
        toa_net=toa_net, energy_residual=energy_residual,
        water_residual=water_residual, deep_flux=deep_flux,
        min_temperature=np.min(t, axis=1),
        max_temperature=np.max(t, axis=1),
        nan_free=nan_free,
    )


__all__ = [
    "ShortwaveColumns",
    "LongwaveColumns",
    "BatchStepResult",
    "shortwave_fluxes_columns",
    "longwave_fluxes_columns",
    "batch_step_columns",
]
