"""短波双流与长波灰体辐射。

短波（delta-Eddington 双流）
---------------------------
- 单层光学性质：水汽吸收 τ_v = k_v·q_v Δp/g；Rayleigh τ_R 按层质量比例；
  云 τ_c 来自 ``clouds.cloud_optical_depth``（Slingo 1989）。
- 组合：ω0 = (τ_R + ω_c τ_c)/τ，g = g_c τ_c/(τ_R+τ_c)；
  delta-Eddington 标度（Wiscombe 1977；Toon et al. 1989）：
  f=g²，τ'=(1−ω0 f)τ，ω0'=(1−f)ω0/(1−ω0 f)，g'=(g−f)/(1−f)。
- 双流方程（Eddington 闭包，含不对称因子；τ 向下增加）：
    dΣ/dτ = −a D + p e^{−τ/μ0}
    dD/dτ = −b Σ + c e^{−τ/μ0}
  Σ=F↓+F↑，D=F↓−F↑，a=1.5(1−ω0 g)，b=2(1−ω0)，
  c=ω0 F0，p=1.5 ω0 F0 g μ0（F0 为垂直于光束的辐照度）。
  该形式由辐射传输方程的一阶/二阶矩在 Eddington 近似下得到，
  满足 d(净总通量)/dτ = −吸收（逐层能量守恒）。
- 层内解析解 + 仿射映射；整层用 TOA F↓=0 与地表
  F↑=α(F↓+直射) 两个边界条件求唯一解。散射相函数按 delta 标度后
  的 g' 参与闭包（不把云当纯吸收体）。

长波（灰体）
------------
- 逐层光学厚度 τ = k_lw q_v Δp/g + k_c,lw LWP；层发射率 ε=1−e^{−τ}，
  无散射；TOA 向下为 0，地表 F↑=ε_s σT_s⁴+(1−ε_s)F↓。
  这是 Schwarzschild 方程的逐层解，逐层能量守恒。
- ``lw_mode="linear"`` 为 OLR=A+B(T_s−T_ref) 的简单闭合对照档，
  用于隔离测试；其大气冷却按长波光学厚度权重分配。

公式来源：Liou, *An Introduction to Atmospheric Radiation*, 2nd ed.,
Ch. 6–7；Petty, *A First Course in Atmospheric Radiation*, 2nd ed.,
Ch. 13；Toon et al. (1989), *J. Geophys. Res.* 94, 16287–16301。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import constants as c
from .clouds import cloud_liquid_water_path, cloud_optical_depth
from .config import CloudConfig, RadiationConfig, ThermoConfig


@dataclass
class ShortwaveFluxes:
    toa_in: float
    toa_out: float
    surface_net: float
    atmosphere_absorption: float
    heating_rate: NDArray[np.float64]  # K/s，逐层
    flux_down: NDArray[np.float64]  # 扩散向下，界面 (n+1)
    flux_up: NDArray[np.float64]  # 扩散向上，界面 (n+1)
    direct_down: NDArray[np.float64]  # 直射水平通量，界面 (n+1)
    optical_depth: NDArray[np.float64]  # 逐层总光学厚度（标度后）
    cloud_tau: NDArray[np.float64]


@dataclass
class LongwaveFluxes:
    olr: float
    surface_net_down: float  # 向下为正
    atmosphere_absorption: float  # 净向下通量收敛（正=加热）
    heating_rate: NDArray[np.float64]  # K/s，逐层
    flux_down: NDArray[np.float64]  # 界面 (n+1)
    flux_up: NDArray[np.float64]  # 界面 (n+1)
    optical_depth: NDArray[np.float64]


def _delta_eddington(
    tau: NDArray[np.float64],
    omega: NDArray[np.float64],
    g: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    f = g * g
    denom = np.maximum(1.0 - omega * f, 1.0e-12)
    tau_s = (1.0 - omega * f) * tau
    omega_s = (1.0 - f) * omega / denom
    g_s = np.where(f < 1.0 - 1.0e-12, (g - f) / np.maximum(1.0 - f, 1.0e-12), 0.0)
    return tau_s, np.clip(omega_s, 0.0, 1.0), np.clip(g_s, -1.0, 1.0)


def _sw_layer_response(
    tau: float,
    omega: float,
    g: float,
    mu0: float,
    f0_perp: float,
    f_down_top: float,
    f_up_top: float,
) -> tuple[float, float]:
    """单层双流解析响应：由层顶 [F↓, F↑] 得层底 [F↓, F↑]。"""
    if tau <= 0.0 and f0_perp <= 0.0:
        return f_down_top, f_up_top

    a = 1.5 * (1.0 - omega * g)
    b = 2.0 * (1.0 - omega)
    csrc = omega * f0_perp
    p = 1.5 * omega * f0_perp * g * mu0

    sigma0 = f_down_top + f_up_top
    d0 = f_down_top - f_up_top
    e_bot = np.exp(-tau / mu0)

    if b <= 1.0e-12:
        # 保守散射 + 直射源；D(τ)=D0+c μ0(1−e)，Σ 解析积分。
        d1 = d0 + csrc * mu0 * (1.0 - e_bot)
        sigma1 = (
            sigma0
            - a * d0 * tau
            - a * csrc * mu0 * (tau - mu0 * (1.0 - e_bot))
            + p * mu0 * (1.0 - e_bot)
        )
    else:
        s = float(np.sqrt(a * b))
        # 特解 A e^{−τ/μ0}
        denom = a * b * mu0 * mu0 - 1.0
        if abs(denom) < 1.0e-12:
            denom = 1.0e-12 if denom >= 0.0 else -1.0e-12
        amp = (a * csrc * mu0 * mu0 + p * mu0) / denom
        coef = p + amp / mu0
        c1 = 0.5 * ((sigma0 - amp) + coef / s - (a / s) * d0)
        c2 = 0.5 * ((sigma0 - amp) - coef / s + (a / s) * d0)
        e_plus = np.exp(min(s * tau, 700.0))
        e_minus = np.exp(max(-s * tau, -700.0))
        sigma1 = c1 * e_plus + c2 * e_minus + amp * e_bot
        d1 = -(s / a) * (c1 * e_plus - c2 * e_minus) + (coef / a) * e_bot

    f_down_bot = 0.5 * (sigma1 + d1)
    f_up_bot = 0.5 * (sigma1 - d1)
    return float(f_down_bot), float(f_up_bot)


def _sw_layer_affine(
    tau: float, omega: float, g: float, mu0: float, f0_perp: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """层映射的仿射形式：y = M x + c，x=[F↓,F↑]_top。"""
    c_out = np.array(_sw_layer_response(tau, omega, g, mu0, f0_perp, 0.0, 0.0))
    col1 = np.array(_sw_layer_response(tau, omega, g, mu0, f0_perp, 1.0, 0.0)) - c_out
    col2 = np.array(_sw_layer_response(tau, omega, g, mu0, f0_perp, 0.0, 1.0)) - c_out
    mat = np.column_stack([col1, col2])
    return mat, c_out


def shortwave_fluxes(
    p_edges: ArrayLike,
    T: ArrayLike,
    qv: ArrayLike,
    qc: ArrayLike,
    surface_albedo: float,
    toa_flux: float,
    cos_zenith: float,
    rad_cfg: RadiationConfig,
    cloud_cfg: CloudConfig,
) -> ShortwaveFluxes:
    """整层短波双流。``toa_flux`` 为水平面通量 [W/m²]，``mu0`` 为有效天顶角余弦。"""
    p = np.asarray(p_edges, dtype=np.float64)
    nlev = p.size - 1
    t = np.asarray(T, dtype=np.float64)
    q = np.asarray(qv, dtype=np.float64)
    qcl = np.asarray(qc, dtype=np.float64)
    dp = p[:-1] - p[1:]
    zeros = np.zeros(nlev, dtype=np.float64)
    zeros_i = np.zeros(nlev + 1, dtype=np.float64)

    if toa_flux <= 0.0 or cos_zenith <= 1.0e-6:
        return ShortwaveFluxes(
            toa_in=0.0,
            toa_out=0.0,
            surface_net=0.0,
            atmosphere_absorption=0.0,
            heating_rate=zeros,
            flux_down=zeros_i,
            flux_up=zeros_i,
            direct_down=zeros_i,
            optical_depth=zeros,
            cloud_tau=zeros,
        )

    mu0 = float(cos_zenith)
    f0_perp = float(toa_flux) / mu0
    # 直射水平通量在 TOA 等于 toa_flux
    direct_top = mu0 * f0_perp

    mass_path = dp / c.GRAVITY
    tau_vap = rad_cfg.sw_vapor_abs_coeff * q * mass_path
    tau_ray = rad_cfg.sw_rayleigh_tau_surface * dp / np.maximum(dp.sum(), 1.0e-12)
    tau_cloud = cloud_optical_depth(qcl, dp, cloud_cfg)
    tau = tau_vap + tau_ray + tau_cloud

    omega = np.zeros(nlev, dtype=np.float64)
    g = np.zeros(nlev, dtype=np.float64)
    nz = tau > 1.0e-14
    omega[nz] = (
        tau_ray[nz] + rad_cfg.cloud_sw_single_scatter_albedo * tau_cloud[nz]
    ) / tau[nz]
    denom_g = tau_ray + tau_cloud
    g[nz] = np.where(
        denom_g[nz] > 1.0e-14,
        rad_cfg.cloud_sw_asymmetry * tau_cloud[nz] / np.maximum(denom_g[nz], 1.0e-14),
        0.0,
    )
    omega = np.clip(omega, 0.0, 1.0)
    g = np.clip(g, -1.0, 1.0)

    tau_s, omega_s, g_s = _delta_eddington(tau, omega, g)

    # 直射衰减：界面 k：0=地表，nlev=大气顶
    direct_iface = np.zeros(nlev + 1, dtype=np.float64)
    direct_iface[nlev] = direct_top
    for i in range(nlev - 1, -1, -1):
        direct_iface[i] = direct_iface[i + 1] * np.exp(-min(float(tau_s[i]) / mu0, 700.0))

    # 逐层仿射映射；每层的散射源用该层顶的**局地**直射（不是 TOA 值）。
    mats: list[NDArray[np.float64]] = []
    offs: list[NDArray[np.float64]] = []
    for i in range(nlev):
        f0_local = float(direct_iface[i + 1]) / mu0
        mat, off = _sw_layer_affine(
            float(tau_s[i]), float(omega_s[i]), float(g_s[i]), mu0, f0_local
        )
        mats.append(mat)
        offs.append(off)

    # 传播：自大气顶向下。F↓_top=0，F↑_top=U 未知；各界面对 U 仿射。
    a_d, b_d = 0.0, 0.0
    a_u, b_u = 1.0, 0.0
    for i in range(nlev - 1, -1, -1):
        mat, off = mats[i], offs[i]
        nd = mat[0, 0] * a_d + mat[0, 1] * a_u
        bd = mat[0, 0] * b_d + mat[0, 1] * b_u + off[0]
        nu = mat[1, 0] * a_d + mat[1, 1] * a_u
        bu = mat[1, 0] * b_d + mat[1, 1] * b_u + off[1]
        a_d, b_d, a_u, b_u = nd, bd, nu, bu

    alb = float(np.clip(surface_albedo, 0.0, 1.0))

    def propagate(u_top_value: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        fd = np.zeros(nlev + 1, dtype=np.float64)
        fu = np.zeros(nlev + 1, dtype=np.float64)
        fu[nlev] = float(u_top_value)
        for k in range(nlev - 1, -1, -1):
            # 显式乘加而非 ``mats[k] @ vec``：2×2 matmul 走 BLAS 归约，
            # 与跨列批量核的逐项乘加相差 1 ulp，会破坏逐位对拍。
            fd[k] = max(
                float(mats[k][0, 0] * fd[k + 1] + mats[k][0, 1] * fu[k + 1]
                      + offs[k][0]),
                0.0,
            )
            fu[k] = max(
                float(mats[k][1, 0] * fd[k + 1] + mats[k][1, 1] * fu[k + 1]
                      + offs[k][1]),
                0.0,
            )
        return fd, fu

    # 无截断仿射解：不出现负漫射通量时即为最终解（严格满足地表边界）。
    denom = a_u - alb * a_d
    if abs(denom) < 1.0e-14:
        u_top = 0.0
    else:
        u_top = (alb * (b_d + direct_iface[0]) - b_u) / denom
    u_top = max(u_top, 0.0)
    flux_down, flux_up = propagate(u_top)

    # 双流闭合在"纯吸收 + 仅上行漫射"时会把 F↓ 算成小负值；propagate 将其
    # 截断为非负，截断后地表边界不再自动成立 ⇒ 对 u_top 重新迭代求解。
    # 截断情形下 ∂F↑/∂u 的解析导数 ≈ a_u，迭代为压缩映射。
    scale = max(float(toa_flux), 1.0)
    for _ in range(12):
        residual = flux_up[0] - alb * (flux_down[0] + direct_iface[0])
        if abs(residual) <= 1.0e-12 * scale:
            break
        deriv = a_u if abs(a_u) > 1.0e-9 else 1.0
        u_top = max(u_top - residual / deriv, 0.0)
        flux_down, flux_up = propagate(u_top)

    net_top = (flux_down[1:] + direct_iface[1:]) - flux_up[1:]
    net_bot = (flux_down[:-1] + direct_iface[:-1]) - flux_up[:-1]
    absorbed = net_top - net_bot
    mass = dp / c.GRAVITY
    heating = absorbed / np.maximum(c.CPD * mass, 1.0e-12)
    surface_net = (1.0 - alb) * (flux_down[0] + direct_iface[0])
    atmos_abs = float(np.sum(absorbed))
    toa_net = direct_iface[nlev] - flux_up[nlev]

    return ShortwaveFluxes(
        toa_in=float(direct_iface[nlev]),
        toa_out=float(flux_up[nlev]),
        surface_net=float(surface_net),
        atmosphere_absorption=atmos_abs,
        heating_rate=heating,
        flux_down=flux_down,
        flux_up=flux_up,
        direct_down=direct_iface,
        optical_depth=tau_s,
        cloud_tau=tau_cloud,
    )


def longwave_fluxes(
    p_edges: ArrayLike,
    T: ArrayLike,
    qv: ArrayLike,
    qc: ArrayLike,
    surface_temperature: float,
    surface_emissivity: float,
    rad_cfg: RadiationConfig,
    cloud_cfg: CloudConfig,
) -> LongwaveFluxes:
    """整层长波灰体（或线性 OLR 对照档）。"""
    p = np.asarray(p_edges, dtype=np.float64)
    nlev = p.size - 1
    t = np.asarray(T, dtype=np.float64)
    q = np.asarray(qv, dtype=np.float64)
    qcl = np.asarray(qc, dtype=np.float64)
    dp = p[:-1] - p[1:]
    mass = dp / c.GRAVITY
    ts = float(surface_temperature)
    eps_s = float(np.clip(surface_emissivity, 0.0, 1.0))
    sigma_ts4 = c.SIGMA_SB * ts ** 4

    if rad_cfg.lw_mode == "linear":
        olr = rad_cfg.lw_linear_a + rad_cfg.lw_linear_b * (ts - rad_cfg.lw_linear_t_ref)
        olr = max(olr, 0.0)
        flux_down_surface = rad_cfg.lw_linear_down_fraction * olr
        surface_net_down = flux_down_surface - eps_s * sigma_ts4
        atm_abs = -olr - surface_net_down  # = ε_sσTs⁴ − (1+f_dn)OLR
        # 按层长波光学厚度（水汽+云）权重分配
        tau_lw = rad_cfg.lw_vapor_abs_coeff * q * mass
        tau_lw += rad_cfg.lw_cloud_abs_coeff * cloud_liquid_water_path(qcl, dp)
        w = tau_lw + 1.0e-12 * mass
        w = w / np.maximum(w.sum(), 1.0e-30)
        heating = atm_abs * w / np.maximum(c.CPD * mass, 1.0e-12)
        return LongwaveFluxes(
            olr=float(olr),
            surface_net_down=float(surface_net_down),
            atmosphere_absorption=float(atm_abs),
            heating_rate=heating,
            # 线性档没有逐层通量解；界面数组填 OLR / 地面向下通量，供地表平衡使用。
            flux_down=np.full(nlev + 1, flux_down_surface, dtype=np.float64),
            flux_up=np.full(nlev + 1, olr, dtype=np.float64),
            optical_depth=tau_lw,
        )

    tau_lw = rad_cfg.lw_vapor_abs_coeff * q * mass
    tau_lw += rad_cfg.lw_cloud_abs_coeff * cloud_liquid_water_path(qcl, dp)
    eps = 1.0 - np.exp(-np.maximum(tau_lw, 0.0))
    sigma_t4 = c.SIGMA_SB * t ** 4

    # 界面 0 = 地表，nlev = 大气顶；向下从顶扫到底，向上从底扫到顶。
    f_down = np.zeros(nlev + 1, dtype=np.float64)
    for i in range(nlev - 1, -1, -1):
        f_down[i] = f_down[i + 1] * (1.0 - eps[i]) + eps[i] * sigma_t4[i]

    f_up = np.zeros(nlev + 1, dtype=np.float64)
    f_up[0] = eps_s * sigma_ts4 + (1.0 - eps_s) * f_down[0]
    for i in range(nlev):
        f_up[i + 1] = f_up[i] * (1.0 - eps[i]) + eps[i] * sigma_t4[i]

    net_top = f_down[1:] - f_up[1:]
    net_bot = f_down[:-1] - f_up[:-1]
    convergence = net_top - net_bot
    heating = convergence / np.maximum(c.CPD * mass, 1.0e-12)
    surface_net_down = float(f_down[0] - f_up[0])
    atm_abs = float(np.sum(convergence))

    return LongwaveFluxes(
        olr=float(f_up[nlev]),
        surface_net_down=surface_net_down,
        atmosphere_absorption=atm_abs,
        heating_rate=heating,
        flux_down=f_down,
        flux_up=f_up,
        optical_depth=tau_lw,
    )
