"""太阳几何与大气顶短波通量（公转 + 自转 + 黄赤交角）。

定位（`atmo/INTERFACE.md` FROZEN v1.0.2 §1）
--------------------------------------------
正式耦合的太阳几何与大气顶短波由 **A1**（`atmo_core.solar` / `export`）提供；
本模块**仅作列模式 stub、离线自检与 A1↔A2 对拍参考**，不得作为世界天气的
第二个真源。A3 接线时应使用 A1 的 `toa_sw` 作为本包辐射的上边界条件。

公式与来源
----------
- 太阳赤纬：sin δ = sin ε · sin λ，λ 为黄道经度（春分起算）；
  ε = 23.44°（J2000）；见 Hartmann, *Global Physical Climatology*,
  2nd ed., §2.3 / Eq. (2.6)。
- 日地距离修正：(a/r)² = (1 + e cos ν)² / (1 − e²)²，ν 为真近点角；
  同上 §2.3；e = 0.0167。
- 太阳天顶角：cos μ0 = sin φ sin δ + cos φ cos δ cos H；
  同上 Eq. (2.9)。
- 日平均（解析）：⟨cos μ0⟩ = (1/π)[H0 sinφ sinδ + cosφ cosδ sin H0]，
  H0 = arccos(−tanφ tanδ)；同上 Eq. (2.12) 附近。
- 全球年平均：S0/4（地球截面积/球面积），Hartmann Eq. (2.13)。

三种柱模式
----------
- ``global_mean``：RCE 标准，TOA = S0/4，有效 μ0 = ``global_mean_cos_zenith``；
- ``fixed``：固定 μ0 与 TOA 通量（模块隔离测试）；
- ``geometry``：按纬度/日序/地方太阳时算瞬时通量（昼夜+季节）。
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import constants as c
from .config import SolarConfig


def solar_declination(day_of_year: ArrayLike, cfg: SolarConfig) -> NDArray[np.float64]:
    """太阳赤纬 δ [rad]；春分（day≈80）为 0，夏至为正。"""
    d = np.asarray(day_of_year, dtype=np.float64)
    eps = math.radians(cfg.obliquity_deg)
    lam = 2.0 * math.pi * (d - cfg.vernal_equinox_day) / cfg.year_days
    return np.arcsin(np.sin(eps) * np.sin(lam))


def earth_sun_distance_factor(day_of_year: ArrayLike, cfg: SolarConfig) -> NDArray[np.float64]:
    """(a/r)² 日地距离修正因子（近日点 > 1）。"""
    d = np.asarray(day_of_year, dtype=np.float64)
    e = cfg.eccentricity
    nu = 2.0 * math.pi * (d - cfg.perihelion_day) / cfg.year_days
    return ((1.0 + e * np.cos(nu)) / (1.0 - e * e)) ** 2


def cos_solar_zenith(
    latitude_rad: ArrayLike,
    declination_rad: ArrayLike,
    hour_angle_rad: ArrayLike,
) -> NDArray[np.float64]:
    """cos μ0 = sinφ sinδ + cosφ cosδ cosH。"""
    phi = np.asarray(latitude_rad, dtype=np.float64)
    dec = np.asarray(declination_rad, dtype=np.float64)
    hour = np.asarray(hour_angle_rad, dtype=np.float64)
    return np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(hour)


def hour_angle_from_local_time(local_solar_time_h: float, rotation_period_s: float) -> float:
    """地方太阳时 → 时角 [rad]；正午 0，午夜 ±π。"""
    frac = (local_solar_time_h / 24.0) % 1.0
    return 2.0 * math.pi * (frac - 0.5)


def toa_insolation(
    latitude_deg: float,
    day_of_year: float,
    local_solar_time_h: float,
    cfg: SolarConfig,
) -> tuple[float, float]:
    """瞬时大气顶水平面短波通量 [W/m²] 与 cos μ0。"""
    phi = math.radians(latitude_deg)
    dec = float(solar_declination(day_of_year, cfg))
    hour = hour_angle_from_local_time(local_solar_time_h, cfg.rotation_period_s)
    mu0 = float(cos_solar_zenith(phi, dec, hour))
    if mu0 <= 0.0:
        return 0.0, 0.0
    factor = float(earth_sun_distance_factor(day_of_year, cfg))
    return cfg.solar_constant * factor * mu0, mu0


def daily_mean_toa_insolation(
    latitude_deg: float, day_of_year: float, cfg: SolarConfig
) -> float:
    """日平均大气顶短波通量 [W/m²]（解析积分，含极昼/极夜）。"""
    phi = math.radians(latitude_deg)
    dec = float(solar_declination(day_of_year, cfg))
    factor = float(earth_sun_distance_factor(day_of_year, cfg))
    arg = -math.tan(phi) * math.tan(dec)
    if arg <= -1.0:  # 极昼
        h0 = math.pi
    elif arg >= 1.0:  # 极夜
        return 0.0
    else:
        h0 = math.acos(arg)
    mean_mu = (h0 * math.sin(phi) * math.sin(dec)
               + math.cos(phi) * math.cos(dec) * math.sin(h0)) / math.pi
    return cfg.solar_constant * factor * max(mean_mu, 0.0)


def daily_mean_cos_zenith(
    latitude_deg: float, day_of_year: float, cfg: SolarConfig
) -> float:
    """日平均 cos μ0（供辐射传递的 μ0 近似；日平均通量/日平均 μ0 不直接等价）。"""
    phi = math.radians(latitude_deg)
    dec = float(solar_declination(day_of_year, cfg))
    arg = -math.tan(phi) * math.tan(dec)
    if arg <= -1.0:
        h0 = math.pi
    elif arg >= 1.0:
        return 0.0
    else:
        h0 = math.acos(arg)
    mean_mu = (h0 * math.sin(phi) * math.sin(dec)
               + math.cos(phi) * math.cos(dec) * math.sin(h0)) / math.pi
    return max(mean_mu, 0.0)


def global_annual_mean_toa(cfg: SolarConfig, n_lat: int = 180, n_day: int = 365) -> float:
    """全球年平均 TOA 短波 [W/m²]（数值面积加权，用于自检）。"""
    lats = np.linspace(-89.5, 89.5, n_lat)
    days = np.linspace(1.0, cfg.year_days, n_day, endpoint=False)
    weights = np.cos(np.radians(lats))
    total = 0.0
    for day in days:
        vals = np.array([daily_mean_toa_insolation(float(lat), float(day), cfg) for lat in lats])
        total += float(np.sum(vals * weights)) / float(np.sum(weights))
    return total / n_day


def column_toa_forcing(cfg: SolarConfig) -> tuple[float, float]:
    """柱模式的大气顶水平短波通量 [W/m²] 与有效 cos μ0。

    返回的 μ0 用于双流的直射路径；``global_mean`` 下按
    ``F_toa = μ0_eff · F0_perp`` 反推 F0_perp，使水平通量保持 S0/4。
    """
    if cfg.mode == "global_mean":
        return 0.25 * cfg.solar_constant, cfg.global_mean_cos_zenith
    if cfg.mode == "fixed":
        return cfg.fixed_toa_flux, cfg.fixed_cos_zenith
    toa, mu0 = toa_insolation(
        cfg.latitude_deg, cfg.day_of_year, cfg.local_solar_time_h, cfg
    )
    return toa, (mu0 if mu0 > 0.0 else cfg.fixed_cos_zenith)
