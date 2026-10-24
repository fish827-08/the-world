"""地表：整体公式感热/潜热、风驱动蒸发、海洋混合层、陆地土壤热传导。

公式与来源
----------
- 整体公式（bulk formula）：
    H  = ρ_a c_p C_H |V| (T_s − T_a)
    LE = ρ_a L_v C_E |V| β (q_s(T_s) − q_a)
  来源：Stull, *Boundary Layer Meteorology*, §5；Bonan, *Ecological
  Climatology*, Ch. 7；C_H=C_E=1.2e-3 为中性层结典型值。
- 海表混合层：C_ml dT_s/dt = SW_net + LW_net − H − LE，
  C_ml=ρ_w c_w h_ml（h_ml=50 m 典型）。
- 陆地：表层零热容能量平衡，土壤 1–2 层隐式热传导
    C_i d_i dT_i/dt = F_{i−1} − F_i，F = −k dT/dz。
  节点在层中心，界面用串联热阻；深层固定温度边界。
- 风驱动蒸发：LE ∝ |V|；无风时仅有分子扩散（本模型取 0，登记为参数）。

来源：Bonan (2019), *Climate Change and Terrestrial Ecosystem Modeling*,
Ch. 5–7；Stull (1988), *An Introduction to Boundary Layer Meteorology*。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as c
from .config import SurfaceConfig, ThermoConfig
from .thermo import (
    d_saturation_specific_humidity_dT,
    latent_heat_vaporization,
    saturation_specific_humidity,
)


@dataclass
class SurfaceFluxes:
    sensible: float  # W m-2，向上为正
    latent: float  # W m-2，向上为正
    evaporation: float  # kg m-2 s-1，向上为正
    temperature: float  # K，更新后的表层温度（海洋混合层/陆表）


def surface_properties(cfg: SurfaceConfig) -> tuple[float, float, float]:
    """(反照率, 发射率, 蒸发抑制 β)。"""
    if cfg.surface_type == "ocean":
        return cfg.albedo_ocean, cfg.emissivity_ocean, 1.0
    return cfg.albedo_land, cfg.emissivity_land, cfg.land_evap_beta


def bulk_fluxes(
    surface_temperature: float,
    air_temperature: float,
    air_specific_humidity: float,
    air_pressure: float,
    wind_speed: float,
    beta: float,
    cfg: SurfaceConfig,
    thermo_cfg: ThermoConfig,
) -> tuple[float, float, float]:
    """整体公式：返回 (H, LE, evaporation) [W/m², W/m², kg/m²/s]，向上为正。"""
    rho_a = air_pressure / (c.RD * max(air_temperature, 150.0))
    lv = float(latent_heat_vaporization(surface_temperature, thermo_cfg))
    qs = float(saturation_specific_humidity(surface_temperature, air_pressure, thermo_cfg))
    sensible = rho_a * c.CPD * cfg.drag_coeff_heat * wind_speed * (
        surface_temperature - air_temperature
    )
    evaporation = (
        rho_a
        * cfg.drag_coeff_moisture
        * wind_speed
        * max(beta, 0.0)
        * (qs - air_specific_humidity)
    )
    latent = lv * evaporation
    return float(sensible), float(latent), float(evaporation)


def solve_ocean_temperature(
    surface_temperature: float,
    sw_net_surface: float,
    lw_down_surface: float,
    air_temperature: float,
    air_specific_humidity: float,
    air_pressure: float,
    dt: float,
    cfg: SurfaceConfig,
    thermo_cfg: ThermoConfig,
    wind_speed: float | None = None,
) -> SurfaceFluxes:
    """海洋混合层隐式温度更新（Newton），返回新温度与最终通量。"""
    _, emissivity, _ = surface_properties(cfg)
    heat_capacity = c.RHO_WATER * c.CW * cfg.ocean_mixed_layer_depth
    wind = cfg.wind_speed if wind_speed is None else float(wind_speed)
    rho_a = air_pressure / (c.RD * max(air_temperature, 150.0))
    lv = float(latent_heat_vaporization(surface_temperature, thermo_cfg))
    ts = float(surface_temperature)

    def fluxes(t: float) -> tuple[float, float, float]:
        return bulk_fluxes(t, air_temperature, air_specific_humidity, air_pressure,
                           wind, 1.0, cfg, thermo_cfg)

    for _ in range(10):
        h, le, _ = fluxes(ts)
        f = heat_capacity * (ts - surface_temperature) / dt - (
            sw_net_surface + emissivity * (lw_down_surface - c.SIGMA_SB * ts ** 4)
            - h - le
        )
        qs_prime = float(
            d_saturation_specific_humidity_dT(ts, air_pressure, thermo_cfg)
        )
        df = (
            heat_capacity / dt
            + 4.0 * emissivity * c.SIGMA_SB * ts ** 3
            + rho_a * c.CPD * cfg.drag_coeff_heat * wind
            + rho_a * lv * cfg.drag_coeff_moisture * wind * qs_prime
        )
        step = f / max(df, 1.0e-12)
        ts -= step
        ts = float(np.clip(ts, 150.0, 400.0))
        if abs(step) < 1.0e-8:
            break

    h, le, evap = fluxes(ts)
    return SurfaceFluxes(sensible=h, latent=le, evaporation=evap, temperature=ts)


def _soil_interface_conductance(cfg: SurfaceConfig, i: int, j: int) -> float:
    """节点 i/j 之间的等效导热（串联热阻）。"""
    ki = cfg.soil_conductivity[i]
    kj = cfg.soil_conductivity[j]
    di = cfg.soil_depth_m[i]
    dj = cfg.soil_depth_m[j]
    return (di + dj) / (di / ki + dj / kj)


def update_soil_temperature(
    soil_temperature: np.ndarray,
    surface_temperature: float,
    dt: float,
    cfg: SurfaceConfig,
) -> np.ndarray:
    """土壤 1–2 层隐式热传导；表层通量 G = k_surf (Ts − T1)。"""
    n = cfg.soil_layers
    t = np.array(soil_temperature[:n], dtype=np.float64, copy=True)
    d = np.asarray(cfg.soil_depth_m[:n], dtype=np.float64)
    cap = np.asarray(cfg.soil_volumetric_heat_capacity[:n], dtype=np.float64)
    k = np.asarray(cfg.soil_conductivity[:n], dtype=np.float64)
    k_surf = k[0] / (0.5 * d[0])

    if n == 1:
        k_deep = k[0] / (0.5 * d[0])
        denom = cap[0] * d[0] / dt + k_surf + k_deep
        return np.array([
            (cap[0] * d[0] / dt * t[0] + k_surf * surface_temperature
             + k_deep * cfg.deep_soil_temperature) / denom
        ])

    k12 = _soil_interface_conductance(cfg, 0, 1)
    k_deep = k[1] / (0.5 * d[1])
    c1 = cap[0] * d[0] / dt
    c2 = cap[1] * d[1] / dt
    a11 = c1 + k_surf + k12
    a12 = -k12
    a21 = -k12
    a22 = c2 + k12 + k_deep
    b1 = c1 * t[0] + k_surf * surface_temperature
    b2 = c2 * t[1] + k_deep * cfg.deep_soil_temperature
    det = a11 * a22 - a12 * a21
    return np.array([
        (b1 * a22 - a12 * b2) / det,
        (a11 * b2 - b1 * a21) / det,
    ])


def solve_land_surface(
    soil_temperature: np.ndarray,
    sw_net_surface: float,
    lw_down_surface: float,
    air_temperature: float,
    air_specific_humidity: float,
    air_pressure: float,
    dt: float,
    cfg: SurfaceConfig,
    thermo_cfg: ThermoConfig,
    wind_speed: float | None = None,
) -> tuple[SurfaceFluxes, np.ndarray]:
    """陆表：零热容表层能量平衡 + 土壤隐式更新，返回 (通量, 新土壤温度)。

    土壤隐式更新对表层温度是仿射的（T_new = α + β·Ts），把它代入表层
    能量平衡后只对 Ts 做一维 Newton ⇒ 返回的 (Ts, T_soil) 严格满足同一
    步的表层平衡与土壤热传导，无算子劈裂残差。
    """
    _, emissivity, beta = surface_properties(cfg)
    wind = cfg.wind_speed if wind_speed is None else float(wind_speed)
    rho_a = air_pressure / (c.RD * max(air_temperature, 150.0))
    n = cfg.soil_layers
    t_soil = np.array(soil_temperature[:n], dtype=np.float64, copy=True)
    d = np.asarray(cfg.soil_depth_m[:n], dtype=np.float64)
    cap = np.asarray(cfg.soil_volumetric_heat_capacity[:n], dtype=np.float64)
    k = np.asarray(cfg.soil_conductivity[:n], dtype=np.float64)
    k_surf = k[0] / (0.5 * d[0])

    # 土壤隐式更新写成 T_new = alpha + beta_vec·Ts（Ts 为待求表层温度）。
    if n == 1:
        k_deep = k[0] / (0.5 * d[0])
        denom = cap[0] * d[0] / dt + k_surf + k_deep
        alpha = (
            cap[0] * d[0] / dt * t_soil[0] + k_deep * cfg.deep_soil_temperature
        ) / denom
        beta_vec = np.array([k_surf / denom])
    else:
        k12 = _soil_interface_conductance(cfg, 0, 1)
        k_deep = k[1] / (0.5 * d[1])
        c1 = cap[0] * d[0] / dt
        c2 = cap[1] * d[1] / dt
        a11 = c1 + k_surf + k12
        a12 = -k12
        a21 = -k12
        a22 = c2 + k12 + k_deep
        det = a11 * a22 - a12 * a21
        # b1 = c1·T1 + k_surf·Ts；b2 = c2·T2 + k_deep·Tdeep
        alpha = np.array([
            (c1 * t_soil[0] * a22 + k12 * (c2 * t_soil[1]
                                           + k_deep * cfg.deep_soil_temperature)) / det,
            (a11 * (c2 * t_soil[1] + k_deep * cfg.deep_soil_temperature)
             + k12 * c1 * t_soil[0]) / det,
        ])
        beta_vec = np.array([k_surf * a22 / det, k_surf * k12 / det])

    def skin_temperature(ts0: float) -> float:
        t_skin = float(ts0)
        for _ in range(50):
            t1_new = alpha[0] + beta_vec[0] * t_skin
            h, le, _ = bulk_fluxes(t_skin, air_temperature, air_specific_humidity,
                                   air_pressure, wind, beta, cfg, thermo_cfg)
            f = (
                sw_net_surface
                + emissivity * (lw_down_surface - c.SIGMA_SB * t_skin ** 4)
                - h - le
                - k_surf * (t_skin - t1_new)
            )
            qs_prime = float(
                d_saturation_specific_humidity_dT(t_skin, air_pressure, thermo_cfg)
            )
            df = (
                -4.0 * emissivity * c.SIGMA_SB * t_skin ** 3
                - rho_a * c.CPD * cfg.drag_coeff_heat * wind
                - rho_a * cfg.drag_coeff_moisture * wind * max(beta, 0.0)
                * latent_heat_vaporization(t_skin, thermo_cfg) * qs_prime
                - k_surf * (1.0 - beta_vec[0])
            )
            step = -f / min(df, -1.0e-12)
            t_skin = float(np.clip(t_skin + step, 150.0, 400.0))
            if abs(step) < 1.0e-10:
                break
        return t_skin

    ts = skin_temperature(float(t_soil[0]))
    soil_new = alpha + beta_vec * ts
    h, le, evap = bulk_fluxes(ts, air_temperature, air_specific_humidity,
                              air_pressure, wind, beta, cfg, thermo_cfg)
    return SurfaceFluxes(sensible=h, latent=le, evaporation=evap, temperature=ts), soil_new
