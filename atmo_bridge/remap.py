"""规则经纬网格 → 规则经纬网格：**保守（面积权）重映射**。

口径（写死，改口径 = 作废旧图/旧表）
----------------------------------
1. 两侧都是**格中心**坐标（度）。纬向边界 = 相邻中心中点，两端按等距外推到 ±90°；
   经向是周期网格，边界按环向间距补（见 `_edges`）。
2. 纬度向权重 = **球带面积**交叠 `sin(φ_hi) − sin(φ_lo)`（不是 Δφ —— 高纬用 Δφ 会正偏差）。
3. 经度向权重 = 角度交叠占比（同一纬度带内 `R²cosφ` 因子在归一化时消掉），跨 0/360 拆两段。
4. 每个目标格的行和归一到 **1** ⇒ 两侧都覆盖整球时**全球积分守恒**。
5. 二维权重 = 纬度权重 ⊗ 经度权重（可分离），应用 = 两次一维矩阵乘 ⇒ 不做 O(n⁴) 的逐格求交。

⚠️ 只处理**规则**经纬网格。GCM 的环状网格（Gaussian/Octahedral）须先由导出脚本转成规则网格
（见 `experiments/export_speedyweather_t31.jl`），本模块不做环→经纬。
⚠️ 输入纬度/经度**升降序都支持**（内部排序后回填），但必须单调。
"""
from __future__ import annotations

import numpy as np


def _edges(centers: np.ndarray, *, periodic: bool, period: float = 360.0) -> np.ndarray:
    """格中心（升序）→ 格边界（长度 n+1，升序）。"""
    c = np.asarray(centers, dtype=np.float64).ravel()
    if c.size < 2:
        raise ValueError("网格至少 2 点")
    d = np.diff(c)
    if np.any(d <= 0):
        raise ValueError("坐标必须严格单调升序（调用方负责排序）")
    if periodic:
        first = c[0] - 0.5 * (c[1] - c[0])
        last = c[-1] + 0.5 * (c[0] + period - c[-1])
        return np.concatenate(([first], 0.5 * (c[1:] + c[:-1]), [last]))
    return np.concatenate(([c[0] - 0.5 * d[0]], 0.5 * (c[1:] + c[:-1]),
                           [c[-1] + 0.5 * d[-1]]))


def _circular_overlap(a0: float, a1: float, b0: float, b1: float, period: float) -> float:
    """两个经度区间的环向交叠长度（度），处理跨 0/360 接缝（**两侧都可能跨**）。"""
    a0m, a1m = a0 % period, a1 % period
    b0m, b1m = b0 % period, b1 % period
    if a1m <= a0m:                      # 区间跨 0/360 ⇒ 解缠到 [a0, a0+period]
        a1m += period
    if b1m <= b0m:
        b1m += period
    total = 0.0
    for shift in (0.0, period, -period):
        lo = max(a0m, b0m + shift)
        hi = min(a1m, b1m + shift)
        if hi > lo:
            total += hi - lo
    return min(total, period)


def _sorted_with_perm(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(np.asarray(x, dtype=np.float64).ravel())
    return np.asarray(x, dtype=np.float64).ravel()[idx], idx


def _lat_edges_deg(centers: np.ndarray) -> np.ndarray:
    """纬度格边界（度）——**全球网格的两端钉到 ±90°**。

    🔴 2026-10-19 实测踩坑：ExoPlaSim 是 **Gaussian 网格**，端点中心在 ±85.76°，
    按"中点外推"只有 ±88.24° ⇒ 世界的极点行（±89.8°）与源网格"无交叠"直接抛
    （s1 更阴：不报错但**南北镜像**）。
    判据：中心已到 ±80° 以外 ⇒ 视为覆盖到极点（区域网格不会被误钉）。
    """
    e = _edges(centers, periodic=False).copy()
    if centers[0] <= -80.0:
        e[0] = -90.0
    if centers[-1] >= 80.0:
        e[-1] = 90.0
    return np.clip(e, -90.0, 90.0)


def lat_weights(lat_src: np.ndarray, lat_dst: np.ndarray) -> np.ndarray:
    """(ndst, nsrc) 纬度权重，行和 = 1；列顺序与 `lat_src` 输入顺序一致。"""
    s, perm = _sorted_with_perm(lat_src)
    d, _ = _sorted_with_perm(lat_dst)
    es = np.sin(np.radians(_lat_edges_deg(s)))
    ed = np.sin(np.radians(_lat_edges_deg(d)))
    w = np.zeros((ed.size - 1, es.size - 1), dtype=np.float64)
    for i in range(w.shape[0]):
        lo = np.maximum(es[:-1], ed[i])
        hi = np.minimum(es[1:], ed[i + 1])
        ov = np.clip(hi - lo, 0.0, None)
        tot = ov.sum()
        if tot <= 0:
            raise ValueError(f"目标纬行 {i} 与源网格无交叠（两侧纬度是否都覆盖 [-90,90]？）")
        w[i] = ov / tot
    out = np.empty_like(w)
    out[:, perm] = w
    return out


def lon_weights(lon_src: np.ndarray, lon_dst: np.ndarray) -> np.ndarray:
    """(ndst, nsrc) 经度权重，行和 = 1；列顺序与 `lon_src` 输入顺序一致。"""
    s, perm = _sorted_with_perm(lon_src)
    d, _ = _sorted_with_perm(lon_dst)
    es = _edges(s, periodic=True)
    ed = _edges(d, periodic=True)
    w = np.zeros((ed.size - 1, es.size - 1), dtype=np.float64)
    for i in range(w.shape[0]):
        row = np.array([_circular_overlap(ed[i], ed[i + 1], es[j], es[j + 1], 360.0)
                        for j in range(es.size - 1)])
        tot = row.sum()
        if tot <= 0:
            raise ValueError(f"目标经列 {i} 与源网格无交叠")
        w[i] = row / tot
    out = np.empty_like(w)
    out[:, perm] = w
    return out


def build_weights(lat_src: np.ndarray, lon_src: np.ndarray,
                  lat_dst: np.ndarray, lon_dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (Wlat, Wlon)：`field_dst = Wlat @ field_src @ Wlon.T`。"""
    return (lat_weights(lat_src, lat_dst), lon_weights(lon_src, lon_dst))


def remap_field(field: np.ndarray, wlat: np.ndarray, wlon: np.ndarray) -> np.ndarray:
    """(nlat_s, nlon_s) → (nlat_d, nlon_d)。"""
    f = np.asarray(field, dtype=np.float64)
    if f.shape != (wlat.shape[1], wlon.shape[1]):
        raise ValueError(f"场形状 {f.shape} 与权重 {(wlat.shape[1], wlon.shape[1])} 不符")
    return wlat @ f @ wlon.T


def remap_fields(fields: dict[str, np.ndarray], wlat: np.ndarray,
                 wlon: np.ndarray) -> dict[str, np.ndarray]:
    return {k: remap_field(v, wlat, wlon) for k, v in fields.items()}


def global_mean(field: np.ndarray, lat: np.ndarray) -> float:
    """面积权全球平均（纬度 sin 权重）。"""
    f = np.asarray(field, dtype=np.float64)
    s, _ = _sorted_with_perm(lat)
    e = np.sin(np.radians(_edges(s, periodic=False)))
    aw = np.diff(e)
    aw = aw / aw.sum()
    return float((f * aw[:, None]).sum() / f.shape[1])
