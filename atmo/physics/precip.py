"""降水：Kessler 型云水自动转化/碰并、雨通量落出与再蒸发。

公式与来源
----------
- 自动转化（Kessler 1969, *Meteor. Monogr.* 10(32)）：
  P_auto = k_auto · max(q_c − q_c,crit, 0)，k_auto ≈ 1e-3 s-1，
  q_c,crit ≈ 0.5 g/kg。
- 碰并：P_acc = k_acc · q_c · q_r，k_acc ≈ 2.2 s-1（Kessler 1969）。
- 雨滴落速：V_t ≈ 5 m/s（层平均质量加权，Kessler 型简化）。
- 雨滴再蒸发：E = k_evap (1 − RH) q_r，k_evap ≈ 1e-3 s-1
  （Kessler 1969 的简化线性形式；原式含 q_r^0.5，本实现用线性档并在
  参数表登记）。
- 云内/网格平均：q_c,in = q_c / C；网格损失率 = C × 云内率。

实现口径
--------
雨不作为预报量存列（避免 V_t·dt/Δz 的 CFL 刚性）：每一步按"稳态雨通量"
自上而下扫描。层内生成的雨水立即加入向下通量 P [kg m-2 s-1]，穿过下层
时按该层 RH 再蒸发；到达地表的 P 即地面降水。水量逐层严格记账：
  Δ(q_c m) + Δ(q_v m) + P_out dt = P_in dt
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import constants as c
from .config import PrecipConfig, ThermoConfig
from .thermo import evaporate_liquid, relative_humidity, saturation_specific_humidity


@dataclass
class PrecipResult:
    T: NDArray[np.float64]
    qv: NDArray[np.float64]
    qc: NDArray[np.float64]
    surface_flux: float  # kg m-2 s-1，到达地表
    surface_precip: float  # kg m-2，本步累计
    rain_evaporation: float  # kg m-2，本步累计
    autoconversion: float  # kg m-2，本步累计
    accretion: float  # kg m-2，本步累计


def precip_step(
    p_edges: ArrayLike,
    T: ArrayLike,
    qv: ArrayLike,
    qc: ArrayLike,
    cloud_fraction_layers: ArrayLike,
    dt: float,
    cfg: PrecipConfig,
    thermo_cfg: ThermoConfig,
) -> PrecipResult:
    p = np.asarray(p_edges, dtype=np.float64)
    t = np.array(T, dtype=np.float64, copy=True)
    q = np.array(qv, dtype=np.float64, copy=True)
    qcl = np.array(qc, dtype=np.float64, copy=True)
    cf_in = np.asarray(cloud_fraction_layers, dtype=np.float64)
    dp = p[:-1] - p[1:]
    mass = dp / c.GRAVITY
    p_mid = 0.5 * (p[:-1] + p[1:])
    rho = p_mid / (c.RD * np.maximum(t, 150.0))

    if not cfg.enabled or dt <= 0.0:
        return PrecipResult(t, q, qcl, 0.0, 0.0, 0.0, 0.0, 0.0)

    rain_flux = 0.0  # kg m-2 s-1
    surface_flux = 0.0
    evap_total = 0.0
    auto_total = 0.0
    acc_total = 0.0
    vt = max(cfg.rain_terminal_velocity, 1.0e-6)

    # 层索引 0 = 地表；雨自顶向底扫描（i 从 nlev-1 到 0）。
    for i in range(t.size - 1, -1, -1):
        cf = float(np.clip(cf_in[i], cfg.cloud_fraction_min, 1.0)) if cfg.cloud_fraction_weighted else 1.0
        qc_in = max(qcl[i] / cf, 0.0) if cfg.cloud_fraction_weighted else max(qcl[i], 0.0)
        qr_local = rain_flux / (rho[i] * vt) if rain_flux > 0.0 else 0.0

        auto_rate = cfg.autoconversion_rate * max(qc_in - cfg.qc_crit, 0.0)
        auto_grid = cf * auto_rate if cfg.cloud_fraction_weighted else auto_rate
        acc_rate = cfg.accretion_coeff * qc_in * qr_local
        acc_grid = cf * acc_rate if cfg.cloud_fraction_weighted else acc_rate

        loss_rate = max(auto_grid + acc_grid, 0.0)
        loss_q = min(max(qcl[i], 0.0), loss_rate * dt)
        if loss_q > 0.0:
            # 按比例拆分诊断量
            total_rate = max(auto_grid + acc_grid, 1.0e-30)
            auto_part = loss_q * auto_grid / total_rate
            acc_part = loss_q * acc_grid / total_rate
            auto_total += auto_part * mass[i]
            acc_total += acc_part * mass[i]
            qcl[i] -= loss_q
            rain_flux += loss_q * mass[i] / dt

        # 雨通量在本层再蒸发（用进入本层的雨通量诊断 q_r；线性 Kessler 档）
        if rain_flux > 0.0:
            qs = float(saturation_specific_humidity(t[i], p_mid[i], thermo_cfg))
            rh = min(max(q[i] / max(qs, 1.0e-12), 0.0), 1.0)
            qr_now = rain_flux / (rho[i] * vt)
            evap_q = cfg.rain_evap_coeff * (1.0 - rh) * qr_now * dt
            deficit = max(qs - q[i], 0.0)
            evap_q = min(evap_q, deficit, rain_flux * dt / mass[i])
            if evap_q > 0.0:
                t[i], q[i], _, evap_q = evaporate_liquid(
                    float(t[i]), float(q[i]), float(evap_q), float(p_mid[i]),
                    float(evap_q), thermo_cfg,
                )
                evap_mass = evap_q * mass[i]
                rain_flux = max(rain_flux - evap_mass / dt, 0.0)
                evap_total += evap_mass

    surface_flux = max(rain_flux, 0.0)
    surface_precip = surface_flux * dt
    return PrecipResult(
        T=t,
        qv=q,
        qc=np.maximum(qcl, 0.0),
        surface_flux=float(surface_flux),
        surface_precip=float(surface_precip),
        rain_evaporation=float(evap_total),
        autoconversion=float(auto_total),
        accretion=float(acc_total),
    )
