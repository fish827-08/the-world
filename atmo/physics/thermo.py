"""湿热力学：Clausius–Clapeyron、饱和比湿、潜热、饱和调整、液相蒸发。

公式与参数来源
--------------
- Clausius–Clapeyron 积分式：e_s(T) = e_s0 · exp[L_v/R_v (1/T0 − 1/T)]，
  e_s0 = 611.21 Pa @ 273.15 K；Rogers & Yau, *A Short Course in Cloud
  Physics*, 3rd ed., Eq. (2.16) 附近。
- 饱和比湿：q_s = ε e_s / [p − (1−ε)e_s]，ε = R_d/R_v（同上，Eq. 2.13）。
- Buck (1981) 备选式见 ``_buck_es``（J. Appl. Meteor. 20, 1527–1532）。
- 潜热：常数档 L_v = 2.501e6 J/kg（0 °C）；温度相关档
  L_v(T) = L_v0 + (c_pv − c_w)(T − T0)（Rogers & Yau §2.4 线性近似）。
- 相变温度更新不写死 L_v/c_p 的近似系数，而是对莫湿静力能做一维求根，
  使相变前后 ``moist_static_energy`` 守恒到机器精度。

约定
----
- 所有函数是纯函数；不做隐式状态修改。
- ``qc``/``rain`` 均为 kg/kg（干空气基）；通量换算由调用方负责。
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import constants as c
from .config import ThermoConfig


def latent_heat_vaporization(T: ArrayLike, cfg: ThermoConfig) -> NDArray[np.float64]:
    """水汽化潜热 L_v(T) [J/kg]。"""
    t = np.asarray(T, dtype=np.float64)
    if cfg.latent_heat_mode == "constant":
        return np.full_like(t, c.LV0)
    return c.LV0 + (c.CPV - c.CW) * (t - c.T0)


def saturation_vapor_pressure(T: ArrayLike, cfg: ThermoConfig) -> NDArray[np.float64]:
    """液面饱和水汽压 e_s(T) [Pa]。"""
    t = np.asarray(T, dtype=np.float64)
    if cfg.saturation_formula == "buck":
        return _buck_es(t)
    # CC 积分（L_v 取 T0 常数；与常数潜热档一致）
    return c.ES0 * np.exp(c.LV0 / c.RV * (1.0 / c.T0 - 1.0 / t))


def _buck_es(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Buck (1981) 液面饱和水汽压 [Pa]；t 为 K。"""
    tc = t - c.T0
    return 611.21 * np.exp((18.678 - tc / 234.5) * (tc / (257.14 + tc)))


def saturation_specific_humidity(
    T: ArrayLike, p: ArrayLike, cfg: ThermoConfig
) -> NDArray[np.float64]:
    """饱和比湿 q_s(T, p) [kg/kg]，分母下限避免 p ≤ e_s 的数值爆炸。"""
    t = np.asarray(T, dtype=np.float64)
    pres = np.asarray(p, dtype=np.float64)
    es = saturation_vapor_pressure(t, cfg)
    denom = np.maximum(pres - (1.0 - c.EPSILON) * es, 1.0e-6)
    return c.EPSILON * es / denom


def d_saturation_specific_humidity_dT(
    T: ArrayLike, p: ArrayLike, cfg: ThermoConfig
) -> NDArray[np.float64]:
    """d q_s / dT 的解析式（用于地面通量 Newton 迭代）。"""
    t = np.asarray(T, dtype=np.float64)
    pres = np.asarray(p, dtype=np.float64)
    es = saturation_vapor_pressure(t, cfg)
    denom = np.maximum(pres - (1.0 - c.EPSILON) * es, 1.0e-6)
    lv = latent_heat_vaporization(t, cfg)
    qs = c.EPSILON * es / denom
    return qs * lv / (c.RV * t * t) * pres / denom


def relative_humidity(
    qv: ArrayLike, T: ArrayLike, p: ArrayLike, cfg: ThermoConfig
) -> NDArray[np.float64]:
    """相对湿度 RH = q_v/q_s，截断到 [0, 1.2]（仅诊断云量用）。"""
    qs = saturation_specific_humidity(T, p, cfg)
    return np.clip(np.asarray(qv, dtype=np.float64) / np.maximum(qs, 1.0e-12), 0.0, 1.2)


def moist_static_energy(
    T: ArrayLike,
    z: ArrayLike,
    qv: ArrayLike,
    qc: ArrayLike,
    cfg: ThermoConfig,
) -> NDArray[np.float64]:
    """湿静力能 h = c_pd(T−T0) + g z + L_v0 q_v (+ 液相焓项) [J/kg 干空气]。

    ``latent_heat_mode="constant"``：液相水焓按 0 计（简单模式常用；
    降水带走显热是一项已知的 <1% 近似，见交付报告"未做/风险"）。
    ``temperature_dependent``：h 含 q_v c_pv(T−T0) 与 q_c c_w(T−T0)，
    相变求根用同一 h ⇒ 能量闭合。
    """
    t = np.asarray(T, dtype=np.float64)
    zz = np.asarray(z, dtype=np.float64)
    qvv = np.asarray(qv, dtype=np.float64)
    qcc = np.asarray(qc, dtype=np.float64)
    base = c.CPD * (t - c.T0) + c.GRAVITY * zz + c.LV0 * qvv
    if cfg.latent_heat_mode == "temperature_dependent":
        base = base + qvv * c.CPV * (t - c.T0) + qcc * c.CW * (t - c.T0)
    return base


def _energy_no_geopotential(
    T: float, qv: float, qc: float, cfg: ThermoConfig
) -> float:
    """不含 g z 的湿静力能（相变求根用，z 不变）。"""
    base = c.CPD * (T - c.T0) + c.LV0 * qv
    if cfg.latent_heat_mode == "temperature_dependent":
        base += qv * c.CPV * (T - c.T0) + qc * c.CW * (T - c.T0)
    return base


def _bisect_root(f: Callable[[float], float], lo: float, hi: float, tol: float) -> float:
    """单调函数 f 的二分求根；要求 f(lo)·f(hi) ≤ 0。"""
    flo = f(lo)
    fhi = f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        return lo if abs(flo) < abs(fhi) else hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < 1.0e-9 or (hi - lo) < tol:
            return mid
        if flo * fm <= 0.0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def solve_temperature_for_energy(
    qv: float,
    qc: float,
    target_h_no_z: float,
    cfg: ThermoConfig,
    t_guess: float,
) -> float:
    """给定 qv、qc 与目标能量，解 T 使 ``_energy_no_geopotential`` 守恒。

    两档潜热模式下 h 都是 T 的**线性**函数（qv/qc 固定），直接解析求解；
    ``t_guess`` 仅为调用侧兼容保留。支持标量与数组（跨列批量）。
    """
    if cfg.latent_heat_mode == "constant":
        denom = c.CPD
    else:
        denom = c.CPD + qv * c.CPV + qc * c.CW
    return c.T0 + (target_h_no_z - c.LV0 * qv) / denom


def saturation_adjustment(
    T: float,
    qv: float,
    qc: float,
    p: float,
    cfg: ThermoConfig,
) -> tuple[float, float, float, float]:
    """层内饱和调整：凝结或蒸发云水，使 q_v 回到饱和并守恒湿静力能。

    返回 ``(T_new, qv_new, qc_new, condensation)``。
    - 总水 q_t = qv + qc 守恒；
    - 相变前后湿静力能守恒（对 T 一维求根，两档潜热模式同一路径）；
    - q_t ≤ q_s(T) 时不调整。

    ``qv < q_s(T)`` 但 ``q_t > q_s(T)``（云水多、水汽欠饱和）时，云水蒸发
    会**吸热降温**，根在 T 下方；因此求根区间必须向两侧展开，只向上找
    会退化为端点、把潜热凭空注入柱内。
    """
    q_t = qv + qc
    qs_old = float(saturation_specific_humidity(T, p, cfg))
    if q_t <= qs_old:
        return T, qv, qc, 0.0

    h_target = _energy_no_geopotential(T, qv, qc, cfg)

    # 平衡态 q_v = min(q_t, q_s(T))；h 对 T 单调递增，双侧括根。
    def f(t_abs: float) -> float:
        qv_t = min(q_t, float(saturation_specific_humidity(t_abs, p, cfg)))
        return _energy_no_geopotential(t_abs, qv_t, q_t - qv_t, cfg) - h_target

    t_new = _bisect_root(f, max(T - 200.0, 100.0), T + 200.0, cfg.phase_change_tol_k)

    qv_new = min(q_t, float(saturation_specific_humidity(t_new, p, cfg)))
    qc_new = q_t - qv_new
    return float(t_new), float(qv_new), float(qc_new), float(qc_new - qc)


def saturation_adjustment_columns(
    T: ArrayLike,
    qv: ArrayLike,
    qc: ArrayLike,
    p: ArrayLike,
    cfg: ThermoConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """跨列饱和调整：逐元素复刻 :func:`saturation_adjustment` 的二分路径。

    与标量版同一方程、同一初值区间、同一停止判据与同一更新顺序；仅在
    数组上以掩码并行执行（未触发的元素原样返回）。用于多柱驱动的批量对流，
    与标量路径逐位一致（``np.exp`` 标量/数组同实现）。
    """
    t = np.asarray(T, dtype=np.float64)
    q_v = np.asarray(qv, dtype=np.float64)
    q_c = np.asarray(qc, dtype=np.float64)
    pres = np.asarray(p, dtype=np.float64)
    q_t = q_v + q_c
    qs_old = saturation_specific_humidity(t, pres, cfg)
    adjust = q_t > qs_old

    t_new = t.copy()
    qv_new = q_v.copy()
    qc_new = q_c.copy()
    if not np.any(adjust):
        return t_new, qv_new, qc_new, np.zeros_like(t)

    h_target = _energy_no_geopotential(t, q_v, q_c, cfg)
    lo = np.where(adjust, np.maximum(t - 200.0, 100.0), t)
    hi = np.where(adjust, t + 200.0, t)

    def f(t_abs: NDArray[np.float64]) -> NDArray[np.float64]:
        qs = saturation_specific_humidity(t_abs, pres, cfg)
        qv_eff = np.minimum(q_t, qs)
        return _energy_no_geopotential(t_abs, qv_eff, q_t - qv_eff, cfg) - h_target

    flo = f(lo)
    fhi = f(hi)
    root = np.where(
        flo == 0.0,
        lo,
        np.where(
            fhi == 0.0,
            hi,
            np.where(np.abs(flo) < np.abs(fhi), lo, hi),
        ),
    )
    active = adjust & (flo != 0.0) & (fhi != 0.0) & (flo * fhi <= 0.0)
    tol = cfg.phase_change_tol_k
    for _ in range(80):
        if not np.any(active):
            break
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        done = active & ((np.abs(fm) < 1.0e-9) | ((hi - lo) < tol))
        root = np.where(done, mid, root)
        go_hi = active & ~done & (flo * fm <= 0.0)
        go_lo = active & ~done & ~(flo * fm <= 0.0)
        hi = np.where(go_hi, mid, hi)
        fhi = np.where(go_hi, fm, fhi)
        lo = np.where(go_lo, mid, lo)
        flo = np.where(go_lo, fm, flo)
        active = active & ~done
    else:
        root = np.where(active, 0.5 * (lo + hi), root)

    qs_new = saturation_specific_humidity(root, pres, cfg)
    qv_adj = np.minimum(q_t, qs_new)
    qc_adj = q_t - qv_adj
    t_new = np.where(adjust, root, t_new)
    qv_new = np.where(adjust, qv_adj, qv_new)
    qc_new = np.where(adjust, qc_adj, qc_new)
    return t_new, qv_new, qc_new, qc_new - q_c


def evaporate_liquid(
    T: float,
    qv: float,
    liquid: float,
    p: float,
    requested: float,
    cfg: ThermoConfig,
) -> tuple[float, float, float, float]:
    """液相（云水或雨）蒸发：加入 qv、冷却，受液量和饱和差限制。

    返回 ``(T_new, qv_new, liquid_new, evaporated)``。
    能量用与相变同一湿静力能求根，保证列能量账闭合。
    """
    if requested <= 0.0 or liquid <= 0.0:
        return T, qv, liquid, 0.0

    h_target = _energy_no_geopotential(T, qv, liquid, cfg)
    evap = min(requested, liquid)
    qs = float(saturation_specific_humidity(T, p, cfg))
    evap = min(evap, max(qs - qv, 0.0))
    if evap <= 0.0:
        return T, qv, liquid, 0.0

    t_new = T
    for _ in range(6):
        t_new = solve_temperature_for_energy(qv + evap, liquid - evap, h_target, cfg, t_new)
        qs_new = float(saturation_specific_humidity(t_new, p, cfg))
        evap_new = min(requested, liquid, max(qs_new - qv, 0.0))
        if abs(evap_new - evap) <= 1.0e-15:
            evap = evap_new
            break
        evap = evap_new
        if evap <= 0.0:
            return T, qv, liquid, 0.0

    t_new = solve_temperature_for_energy(qv + evap, liquid - evap, h_target, cfg, t_new)
    return float(t_new), float(qv + evap), float(liquid - evap), float(evap)
