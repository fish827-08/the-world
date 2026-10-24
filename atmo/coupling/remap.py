"""保守（面积权）重映射：atmo Gaussian/规则网格 ↔ 世界规则网格。

与 `atmo_bridge/remap.py`（A17）的关系
-------------------------------------
* 经向权重口径**同 A17**：周期角度交叠、跨 0/360 拆段、按目标行归一化；
  本模块把 A17 的 Python 逐格双重循环改成 numpy 广播（480×960 ↔ T21 建权 <10 ms），
  单测在小网格上与 A17 对拍（机器精度 ≤1e-13；纬度边界计算路径不同，非逐位）。
* 纬向**不同**：A17 用"格中心中点作边界"；Gaussian 网格的真实面积权是 `dmu/2`，
  其 μ 边界必须由 `cumsum(area)` 生成（否则 T21 首行面积多算 22%，见 `grid.py`）。
  对规则经纬网格，`cumsum(area)` 与 A17 的中点边**完全一致**（面积权 = Δsin 边界）。

保守性
------
两侧都用各自声明的 `area` 生成 μ 区间 ⇒ 对任意场 `f`：
`Σ_j src.area_j·f_j == Σ_i dst.area_i·(Wf)_i`（机器精度）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .grid import Grid


def mu_edges(area: np.ndarray) -> np.ndarray:
    """行面积权（Σ=1）→ μ=sinφ 的格边界（长度 nlat+1，两端精确 −1/+1）。"""
    a = np.asarray(area, dtype=np.float64).ravel()
    if a.size < 2:
        raise ValueError("area 至少 2 行")
    if not np.all(np.isfinite(a)) or np.any(a <= 0.0):
        raise ValueError("area 必须有限且全为正")
    total = float(a.sum())
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"area 和 = {total!r}，应为 1（只支持全球网格）")
    a = a / total
    edges = np.concatenate(([-1.0], -1.0 + 2.0 * np.cumsum(a[:-1]), [1.0]))
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("由 area 生成的 μ 边界非单调（面积权异常）")
    return edges


def _periodic_edges(centers: np.ndarray) -> np.ndarray:
    """经度格中心（升序）→ 格边界（同 A17 `_edges(..., periodic=True)` 口径）。"""
    c = np.asarray(centers, dtype=np.float64).ravel()
    if c.size < 2:
        raise ValueError("经度网格至少 2 点")
    d = np.diff(c)
    if np.any(d <= 0.0):
        raise ValueError("lon_deg 必须严格升序")
    first = c[0] - 0.5 * (c[1] - c[0])
    last = c[-1] + 0.5 * (c[0] + 360.0 - c[-1])
    return np.concatenate(([first], 0.5 * (c[1:] + c[:-1]), [last]))


def lat_weights(src: Grid, dst: Grid) -> np.ndarray:
    """(ndst, nsrc) 纬向权重，行和 = 1；用两侧 `area` 的累计 μ 边界。"""
    es = mu_edges(src.area)
    ed = mu_edges(dst.area)
    ov = np.clip(
        np.minimum(es[None, 1:], ed[1:, None])
        - np.maximum(es[None, :-1], ed[:-1, None]),
        0.0, None)
    tot = ov.sum(axis=1)
    if np.any(tot <= 0.0):
        raise ValueError("目标纬行与源网格无交叠（两侧是否都覆盖整球？）")
    return ov / tot[:, None]


def lon_weights(src: Grid, dst: Grid) -> np.ndarray:
    """(ndst, nsrc) 经向权重，行和 = 1；周期角度交叠（跨接缝）。"""
    es = _periodic_edges(src.lon_deg)
    ed = _periodic_edges(dst.lon_deg)
    sw = np.diff(es)
    dw = np.diff(ed)
    if np.any(sw > 180.0) or np.any(dw > 180.0):
        raise ValueError("经度格宽 >180°，周期交叠口径不成立（网格太粗）")
    a0 = ed[:-1][:, None]
    aw = dw[:, None]
    b0 = es[:-1][None, :]
    bw = sw[None, :]
    ov = np.zeros((dst.nlon, src.nlon), dtype=np.float64)
    for shift in (0.0, 360.0, -360.0):
        lo = np.maximum(a0, b0 + shift)
        hi = np.minimum(a0 + aw, b0 + bw + shift)
        ov += np.clip(hi - lo, 0.0, None)
    # 同一对区间最多只有自身宽度那么多；防"两侧都跨 0/360"时两段重复计数
    ov = np.minimum(ov, np.minimum(aw, bw))
    tot = ov.sum(axis=1)
    if np.any(tot <= 0.0):
        raise ValueError("目标经列与源网格无交叠")
    return ov / tot[:, None]


def build_weights(src: Grid, dst: Grid) -> tuple[np.ndarray, np.ndarray]:
    """返回 `(Wlat, Wlon)`：`field_dst = Wlat @ field_src @ Wlon.T`。"""
    return lat_weights(src, dst), lon_weights(src, dst)


def remap_field(field: np.ndarray, wlat: np.ndarray, wlon: np.ndarray) -> np.ndarray:
    """`(nlat_s,nlon_s)` 或 `(nlev,nlat_s,nlon_s)` → 目标网格（float64、C 序）。"""
    f = np.ascontiguousarray(field, dtype=np.float64)
    nlat_s, nlon_s = wlat.shape[1], wlon.shape[1]
    if f.ndim == 2:
        if f.shape != (nlat_s, nlon_s):
            raise ValueError(f"场形状 {f.shape} 与源网格 {(nlat_s, nlon_s)} 不符")
        return wlat @ f @ wlon.T
    if f.ndim == 3:
        if f.shape[1:] != (nlat_s, nlon_s):
            raise ValueError(f"场形状 {f.shape} 与源网格 {(nlat_s, nlon_s)} 不符")
        out = np.empty((f.shape[0], wlat.shape[0], wlon.shape[0]), dtype=np.float64)
        for k in range(f.shape[0]):
            out[k] = wlat @ f[k] @ wlon.T
        return out
    raise ValueError(f"只支持 2D/3D 场，实得 ndim={f.ndim}")


@dataclass(frozen=True, eq=False)
class RemapPair:
    """一对网格 + 缓存权重；`remap()` 自动处理北→南输入的场行翻转。"""

    src: Grid
    dst: Grid
    wlat: np.ndarray
    wlon: np.ndarray
    _shape: tuple[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.wlat.shape != (self.dst.nlat, self.src.nlat):
            raise ValueError(
                f"Wlat 形状 {self.wlat.shape} ≠ ({self.dst.nlat}, {self.src.nlat})")
        if self.wlon.shape != (self.dst.nlon, self.src.nlon):
            raise ValueError(
                f"Wlon 形状 {self.wlon.shape} ≠ ({self.dst.nlon}, {self.src.nlon})")
        object.__setattr__(self, "_shape", (self.dst.nlat, self.dst.nlon))

    @classmethod
    def build(cls, src: Grid, dst: Grid) -> "RemapPair":
        wlat, wlon = build_weights(src, dst)
        return cls(src=src, dst=dst, wlat=wlat, wlon=wlon)

    def remap(self, field: np.ndarray) -> np.ndarray:
        """源场 → 目标场；场行序必须与构造 `src` 时的输入序一致。"""
        f = np.asarray(field, dtype=np.float64)
        if self.src.flipped_from_descending:
            f = f[..., ::-1, :]
        out = remap_field(f, self.wlat, self.wlon)
        if self.dst.flipped_from_descending:
            out = out[..., ::-1, :]
        return np.ascontiguousarray(out, dtype=np.float64)

    def remap_fields(self, fields: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: self.remap(v) for k, v in fields.items()}


def global_mean(field: np.ndarray, grid: Grid) -> np.ndarray:
    """面积权全球平均；2D → 标量，3D → `(nlev,)`。"""
    f = np.asarray(field, dtype=np.float64)
    if f.shape[-2:] != (grid.nlat, grid.nlon):
        raise ValueError(f"场形状 {f.shape} 与网格 {grid.nlat}×{grid.nlon} 不符")
    row_mean = f.mean(axis=-1)                      # 经向等权（周期格同宽）
    if f.ndim == 2:
        return float(np.sum(row_mean * grid.area))
    return np.sum(row_mean * grid.area[None, :], axis=1)
