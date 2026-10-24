"""湿对流调整：把条件不稳定柱松弛到湿静力能（MSE）中性。

方案
----
相邻两层若上层 MSE 低于下层（h_i > h_{i+1}），即条件不稳定，则按质量
加权混合两层的 MSE 与总水 q_t = q_v + q_c；混合后两层各自满足同一个
h_mix（而不是同一温度——否则会人为造出等温层）：
- 每层先按无水凝结算出 T_dry，使 h(T_dry, q_t, 0) = h_mix；
- 若 q_t > q_s(T_dry)，调用 ``thermo.saturation_adjustment`` 凝结到饱和，
  潜热留在空气中；凝结水超过 ``qc_hold`` 的部分作为对流降水移除。
每一步混合都严格守恒 MSE 与总水（移除的降水单独记账），迭代至全柱
MSE 随高度不减少（或达到 ``max_iter``）。移除降水时温度相关档把液相焓
记入 ``precip_energy``；常数潜热档液相焓按 0 计。

与 Betts–Miller 的关系
----------------------
这是 Betts–Miller "adjust to a reference moist adiabat" 的硬调整极限
（时间尺度 → 0）；参考剖面由混合本身给出（MSE 中性），不写死目标 T/q。
来源：Betts & Miller (1986), *Q. J. R. Meteorol. Soc.* 112, 677–691；
Manabe & Wetherald (1967) 对流调整框架。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import constants as c
from .config import ConvectionConfig, ThermoConfig
from .thermo import (
    moist_static_energy,
    saturation_adjustment,
    saturation_adjustment_columns,
    saturation_specific_humidity,
    solve_temperature_for_energy,
)


@dataclass
class ConvectionResult:
    T: NDArray[np.float64]
    qv: NDArray[np.float64]
    qc: NDArray[np.float64]
    precip: float  # kg m-2，本步对流降水
    precip_energy: float  # J m-2，降水带走的液相焓（仅温度相关档非零）
    iterations: int
    pairs_adjusted: int


@dataclass
class ConvectionColumnsResult:
    """跨列对流调整结果；``precip``/``precip_energy``/``pairs_adjusted`` 为逐列数组。"""

    T: NDArray[np.float64]
    qv: NDArray[np.float64]
    qc: NDArray[np.float64]
    precip: NDArray[np.float64]
    precip_energy: NDArray[np.float64]
    iterations: int
    pairs_adjusted: NDArray[np.int64]


def _state_for_target_energy_columns(
    qt: NDArray[np.float64],
    target_h: NDArray[np.float64],
    p: NDArray[np.float64],
    t_guess: NDArray[np.float64],
    thermo_cfg: ThermoConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """跨列版 :func:`_state_for_target_energy`。"""
    t_dry = solve_temperature_for_energy(qt, 0.0, target_h, thermo_cfg, t_guess)
    qs_dry = saturation_specific_humidity(t_dry, p, thermo_cfg)
    sat = qt > qs_dry
    if not np.any(sat):
        return t_dry, qt.copy(), np.zeros_like(qt)
    t_adj, qv_adj, qc_adj, _ = saturation_adjustment_columns(
        t_dry, qt, np.zeros_like(qt), p, thermo_cfg
    )
    return (
        np.where(sat, t_adj, t_dry),
        np.where(sat, qv_adj, qt),
        np.where(sat, qc_adj, 0.0),
    )


def convective_adjustment_columns(
    p_edges: ArrayLike,
    T: ArrayLike,
    qv: ArrayLike,
    qc: ArrayLike,
    z: ArrayLike,
    cfg: ConvectionConfig,
    thermo_cfg: ThermoConfig,
) -> ConvectionColumnsResult:
    """跨列湿对流调整：逐列复刻 :func:`convective_adjustment` 的判据与顺序。

    ``p_edges`` ``(ncol,nlev+1)``，``T/qv/qc/z`` ``(ncol,nlev)``；层索引 0 =
    地表。列之间独立，故按列向量化不改变任何一列的算术顺序（成对混合
    仍是自下而上、逐对更新）。返回逐列降水/液相焓与成对调整计数。
    """
    p = np.asarray(p_edges, dtype=np.float64)
    t = np.array(T, dtype=np.float64, copy=True)
    q = np.array(qv, dtype=np.float64, copy=True)
    qcl = np.array(qc, dtype=np.float64, copy=True)
    zz = np.asarray(z, dtype=np.float64)
    if t.ndim != 2:
        raise ValueError("T 必须为 (ncol, nlev)")
    ncol, nlev = t.shape
    if p.shape != (ncol, nlev + 1):
        raise ValueError(f"p_edges 形状 {p.shape} ≠ {(ncol, nlev + 1)}")
    if q.shape != t.shape or qcl.shape != t.shape or zz.shape != t.shape:
        raise ValueError("T/qv/qc/z 形状必须一致 (ncol, nlev)")

    dp = p[:, :-1] - p[:, 1:]
    mass = dp / c.GRAVITY
    p_mid = 0.5 * (p[:, :-1] + p[:, 1:])
    precip = np.zeros(ncol, dtype=np.float64)
    precip_energy = np.zeros(ncol, dtype=np.float64)
    pairs = np.zeros(ncol, dtype=np.int64)
    iterations = 0

    if not cfg.enabled or nlev < 2:
        return ConvectionColumnsResult(
            t, q, qcl, precip, precip_energy, iterations, pairs
        )

    for it in range(cfg.max_iter):
        iterations = it + 1
        h = moist_static_energy(t, zz, q, qcl, thermo_cfg)
        if not np.any(h[:, :-1] > h[:, 1:] + cfg.mse_tol):
            break
        for i in range(nlev - 1):
            h_i = moist_static_energy(t[:, i], zz[:, i], q[:, i], qcl[:, i], thermo_cfg)
            h_j = moist_static_energy(
                t[:, i + 1], zz[:, i + 1], q[:, i + 1], qcl[:, i + 1], thermo_cfg
            )
            mask = h_i > h_j + cfg.mse_tol
            if not np.any(mask):
                continue
            mt = mass[:, i] + mass[:, i + 1]
            h_mix = (mass[:, i] * h_i + mass[:, i + 1] * h_j) / mt
            qt = (
                mass[:, i] * (q[:, i] + qcl[:, i])
                + mass[:, i + 1] * (q[:, i + 1] + qcl[:, i + 1])
            ) / mt
            t_new, qv_new, qc_new = _state_for_target_energy_columns(
                qt, h_mix - c.GRAVITY * zz[:, i], p_mid[:, i], t[:, i], thermo_cfg
            )
            t_next, q_next, qc_next = _state_for_target_energy_columns(
                qt, h_mix - c.GRAVITY * zz[:, i + 1],
                p_mid[:, i + 1], t[:, i + 1], thermo_cfg,
            )
            for j, (tj, qj, qcj, mj) in enumerate((
                (t_new, qv_new, qc_new, mass[:, i]),
                (t_next, q_next, qc_next, mass[:, i + 1]),
            )):
                idx = i + j
                excess = np.maximum(qcj - cfg.qc_hold, 0.0)
                qcj = np.where(mask, qcj - excess, qcj)
                precip += np.where(mask, excess * mj, 0.0)
                if thermo_cfg.latent_heat_mode == "temperature_dependent":
                    precip_energy += np.where(
                        mask, c.CW * (tj - c.T0) * excess * mj, 0.0
                    )
                t[:, idx] = np.where(mask, tj, t[:, idx])
                q[:, idx] = np.where(mask, qj, q[:, idx])
                qcl[:, idx] = np.where(mask, qcj, qcl[:, idx])
            pairs += mask.astype(np.int64)

    return ConvectionColumnsResult(
        T=t,
        qv=np.maximum(q, 0.0),
        qc=np.maximum(qcl, 0.0),
        precip=precip,
        precip_energy=precip_energy,
        iterations=int(iterations),
        pairs_adjusted=pairs,
    )


def _state_for_target_energy(
    qt: float,
    target_h: float,
    p: float,
    t_guess: float,
    thermo_cfg: ThermoConfig,
) -> tuple[float, float, float]:
    """给定总水与目标 h（含 g z），返回 (T, qv, qc)。"""
    t_dry = solve_temperature_for_energy(qt, 0.0, target_h, thermo_cfg, t_guess)
    qs_dry = float(saturation_specific_humidity(t_dry, p, thermo_cfg))
    if qt <= qs_dry:
        return float(t_dry), float(qt), 0.0
    t_new, qv_new, qc_new, _ = saturation_adjustment(t_dry, qt, 0.0, p, thermo_cfg)
    return float(t_new), float(qv_new), float(qc_new)


def convective_adjustment(
    p_edges: ArrayLike,
    T: ArrayLike,
    qv: ArrayLike,
    qc: ArrayLike,
    z: ArrayLike,
    cfg: ConvectionConfig,
    thermo_cfg: ThermoConfig,
) -> ConvectionResult:
    p = np.asarray(p_edges, dtype=np.float64)
    t = np.array(T, dtype=np.float64, copy=True)
    q = np.array(qv, dtype=np.float64, copy=True)
    qcl = np.array(qc, dtype=np.float64, copy=True)
    zz = np.asarray(z, dtype=np.float64)
    dp = p[:-1] - p[1:]
    mass = dp / c.GRAVITY
    p_mid = 0.5 * (p[:-1] + p[1:])

    if not cfg.enabled or t.size < 2:
        return ConvectionResult(t, q, qcl, 0.0, 0.0, 0, 0)

    precip_total = 0.0
    precip_energy_total = 0.0
    pairs_total = 0
    iterations = 0

    for it in range(cfg.max_iter):
        iterations = it + 1
        h = moist_static_energy(t, zz, q, qcl, thermo_cfg)
        if not np.any(h[:-1] > h[1:] + cfg.mse_tol):
            break
        # 自下而上处理（层索引 0 = 地表）；混合后两层 h 相同，成对稳定。
        for i in range(t.size - 1):
            h_i = float(moist_static_energy(t[i], zz[i], q[i], qcl[i], thermo_cfg))
            h_j = float(moist_static_energy(t[i + 1], zz[i + 1], q[i + 1], qcl[i + 1], thermo_cfg))
            if h_i <= h_j + cfg.mse_tol:
                continue
            mt = mass[i] + mass[i + 1]
            h_mix = (mass[i] * h_i + mass[i + 1] * h_j) / mt
            qt = (mass[i] * (q[i] + qcl[i]) + mass[i + 1] * (q[i + 1] + qcl[i + 1])) / mt
            t_new, qv_new, qc_new = _state_for_target_energy(
                float(qt), float(h_mix - c.GRAVITY * zz[i]),
                float(p_mid[i]), float(t[i]), thermo_cfg,
            )
            t_next, q_next, qc_next = _state_for_target_energy(
                float(qt), float(h_mix - c.GRAVITY * zz[i + 1]),
                float(p_mid[i + 1]), float(t[i + 1]), thermo_cfg,
            )
            for j, (tj, qj, qcj, mj) in enumerate((
                (t_new, qv_new, qc_new, mass[i]),
                (t_next, q_next, qc_next, mass[i + 1]),
            )):
                idx = i + j
                if qcj > cfg.qc_hold:
                    excess = qcj - cfg.qc_hold
                    qcj = cfg.qc_hold
                    precip_total += excess * mj
                    if thermo_cfg.latent_heat_mode == "temperature_dependent":
                        precip_energy_total += c.CW * (tj - c.T0) * excess * mj
                t[idx] = tj
                q[idx] = qj
                qcl[idx] = qcj
            pairs_total += 1

    return ConvectionResult(
        T=t,
        qv=np.maximum(q, 0.0),
        qc=np.maximum(qcl, 0.0),
        precip=float(precip_total),
        precip_energy=float(precip_energy_total),
        iterations=int(iterations),
        pairs_adjusted=int(pairs_total),
    )
