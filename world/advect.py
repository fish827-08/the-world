"""风场平流（A19.3，**默认关**）：半拉格朗日 + 双线性插值，经度周期、纬度夹取。

为什么单独一个模块
------------------
"风怎么用"是世界级决策（见 `内部技术文档/技术-A19-风的口径决策-20261019.md`）。
本模块只提供**一种**算子（被动标量平流），由各场自己决定要不要用、怎么用；
默认不接任何场 ⇒ 关档逐位等价。

口径（写死，改口径 = 作废旧读数）
--------------------------------
1. 位移用**格数**表示：`d_col = u·Δt / (R·cosφ·Δλ)`、`d_row = v·Δt / (R·Δφ)`。
2. 上游采样 = 双线性；经度按 cols 取模（0/360 接缝连续），纬度**夹取**（极行不外推）。
3. `u/v` 可以是标量或 (rows, cols) 场；单位 m/s，`Δt` 秒。
4. 不保证严格守恒（双线性会轻微抹平），故只用于**诊断量/气味**这类被动场，
   **不用于**任何参与账本守恒的量（水/能量）。
"""
from __future__ import annotations

import numpy as np

R_EARTH_M = 6.371e6


def advect_scalar(field: np.ndarray, u, v, dt_s: float, *,
                  radius_m: float = R_EARTH_M, wrap_lon: bool = True) -> np.ndarray:
    """把 (rows, cols) 的被动标量按风场平流 `dt_s` 秒（半拉格朗日）。

    `u>0` = 向东、`v>0` = 向北（与 GCM 表/大气诊断同号约定）。
    """
    f = np.asarray(field, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError(f"advect_scalar 只接受 2D 场，收到 {f.shape}")
    rows, cols = f.shape
    if cols < 2 or rows < 2:
        raise ValueError("网格至少 2×2")

    dphi = np.pi / rows
    dlam = 2.0 * np.pi / cols
    phi = -0.5 * np.pi + (np.arange(rows, dtype=np.float64) + 0.5) * dphi
    cosphi = np.maximum(np.cos(phi), 1e-6)          # 极点附近夹一下，避免除零

    uu = np.asarray(u, dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    if uu.ndim == 0:
        uu = np.full((rows, cols), float(uu))
    if vv.ndim == 0:
        vv = np.full((rows, cols), float(vv))

    d_row = vv * float(dt_s) / (radius_m * dphi)
    d_col = uu * float(dt_s) / (radius_m * cosphi[:, None] * dlam)

    # 上游位置（拉格朗日回溯）
    src_r = np.arange(rows, dtype=np.float64)[:, None] - d_row
    src_c = np.arange(cols, dtype=np.float64)[None, :] - d_col

    r0 = np.floor(src_r)
    c0 = np.floor(src_c)
    wr = src_r - r0
    wc = src_c - c0

    r0i = np.clip(r0.astype(np.int64), 0, rows - 1)
    r1i = np.clip(r0i + 1, 0, rows - 1)
    if wrap_lon:
        c0i = np.mod(c0.astype(np.int64), cols)
        c1i = np.mod(c0i + 1, cols)
    else:
        c0i = np.clip(c0.astype(np.int64), 0, cols - 1)
        c1i = np.clip(c0i + 1, 0, cols - 1)

    out = ((1.0 - wr) * (1.0 - wc) * f[r0i, c0i]
           + (1.0 - wr) * wc * f[r0i, c1i]
           + wr * (1.0 - wc) * f[r1i, c0i]
           + wr * wc * f[r1i, c1i])
    return out


def centroid(field: np.ndarray, wrap_lon: bool = True) -> tuple[float, float]:
    """被动场的质心 (row, col)；经度按环向重心（跨接缝也对）。仅用于测试/诊断。"""
    f = np.asarray(field, dtype=np.float64)
    tot = float(f.sum())
    if tot <= 0.0:
        return (float("nan"), float("nan"))
    rows, cols = f.shape
    r = float((f.sum(axis=1) * np.arange(rows)).sum() / tot)
    j = np.arange(cols, dtype=np.float64)
    if wrap_lon:
        ang = 2.0 * np.pi * j / cols
        x = float((f.sum(axis=0) * np.cos(ang)).sum() / tot)
        y = float((f.sum(axis=0) * np.sin(ang)).sum() / tot)
        c = (np.degrees(np.arctan2(y, x)) / 360.0 * cols) % cols
    else:
        c = float((f.sum(axis=0) * j).sum() / tot)
    return (r, c)
