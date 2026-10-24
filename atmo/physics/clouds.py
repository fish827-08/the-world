"""云：诊断云量、云水路径、云光学厚度、未饱和云水蒸发。

公式与来源
----------
- 诊断云量：C = [(RH − RH_c)/(1 − RH_c)]^b，RH_c = 0.70；
  Slingo (1987), *Q. J. R. Meteorol. Soc.* 113, 899–927 的 RH 型云量方案
  的简化版（取指数 b=2）。
- 云水路径：LWP = q_c Δp / g [kg/m²]。
- 短波云光学厚度：τ = 3 LWP / (2 ρ_w r_eff)；
  Slingo (1989), *Q. J. R. Meteorol. Soc.* 115, 487–511；r_eff=10 μm。
- 云量重叠：总云量按随机重叠 C_tot = 1 − Π(1 − C_i)（诊断用；
  辐射按网格平均云水计算，不做云量分区）。

云水是网格平均量（kg/kg 干空气）；云内云水 q_c/C 只在降水参数化里
用于换算，辐射用网格平均 LWP ⇒ 无需再乘云量。
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import constants as c
from .config import CloudConfig, ThermoConfig
from .thermo import evaporate_liquid, relative_humidity, saturation_specific_humidity


def cloud_fraction(rh: ArrayLike, cfg: CloudConfig) -> NDArray[np.float64]:
    """诊断云量 [0, 1]；RH 低于阈值时为 0。"""
    rh_arr = np.asarray(rh, dtype=np.float64)
    x = np.clip((rh_arr - cfg.rh_crit) / (1.0 - cfg.rh_crit), 0.0, 1.0)
    cf = x ** cfg.rh_exponent
    return np.clip(cf, cfg.cloud_fraction_floor, cfg.cloud_fraction_ceiling)


def cloud_fraction_from_state(
    qv: ArrayLike,
    T: ArrayLike,
    p: ArrayLike,
    cfg: CloudConfig,
    thermo_cfg: ThermoConfig,
) -> NDArray[np.float64]:
    """由层状态诊断云量。"""
    rh = relative_humidity(qv, T, p, thermo_cfg)
    return cloud_fraction(rh, cfg)


def cloud_liquid_water_path(
    qc: ArrayLike, delta_p: ArrayLike
) -> NDArray[np.float64]:
    """网格平均云水路径 LWP [kg/m²]。"""
    return np.asarray(qc, dtype=np.float64) * np.asarray(delta_p, dtype=np.float64) / c.GRAVITY


def cloud_optical_depth(
    qc: ArrayLike, delta_p: ArrayLike, cfg: CloudConfig
) -> NDArray[np.float64]:
    """短波云光学厚度（Slingo 1989 或线性档）。"""
    lwp = cloud_liquid_water_path(qc, delta_p)
    if cfg.optical_depth_mode == "linear":
        return cfg.linear_optical_coeff * lwp
    re_m = cfg.effective_radius_um * 1.0e-6
    return 3.0 * lwp / (2.0 * c.RHO_WATER * re_m)


def total_cloud_cover(cloud_fraction_layers: ArrayLike) -> float:
    """随机重叠总云量。"""
    cf = np.clip(np.asarray(cloud_fraction_layers, dtype=np.float64), 0.0, 1.0)
    return float(1.0 - np.prod(1.0 - cf))


def evaporate_cloud_water(
    T: float,
    qv: float,
    qc: float,
    p: float,
    dt: float,
    cloud_cfg: CloudConfig,
    thermo_cfg: ThermoConfig,
) -> tuple[float, float, float, float]:
    """未饱和云水弛豫蒸发（潜热冷却）；返回 (T, qv, qc, evaporated)。

    速率 = ``cloud_evap_coeff · (1 − RH) · q_c``，再受饱和差与云水量限制；
    相变能量由 thermo 的湿静力能求根保证。
    """
    if qc <= 0.0 or dt <= 0.0:
        return T, qv, qc, 0.0
    qs = float(saturation_specific_humidity(T, p, thermo_cfg))
    rh = min(max(qv / max(qs, 1.0e-12), 0.0), 1.0)
    requested = cloud_cfg.cloud_evap_coeff * (1.0 - rh) * qc * dt
    return evaporate_liquid(T, qv, qc, p, requested, thermo_cfg)
