"""网格描述：给重映射一副「坐标 + 自己声明的面积权」。

口径（写死，改口径 = 作废旧权重）
--------------------------------
1. 纬度 `lat_deg` **升序（南→北）**；经度 `lon_deg` **升序、[0,360)**。
2. `area` 是该网格**自己的**行面积权（`Σ area = 1`）——不是"按中心纬度 cos 算一个"。
   * atmo Gaussian：`area = dmu/2`（`atmo_core.grid_spec()` 给的高斯求积权）。
   * 世界：`area ∝ cosφ`（`SphereWorld.cell_area` 的行值），与引擎全局平均同一把尺。
   * 规则经纬：`area = Δsin(φ_edges)/2`（A1 `grid_spec(kind="regular")` 同一口径）。
3. 🔴 保守重映射的面积边必须由 `cumsum(area)` 生成（μ 空间），**不能**用
   `atmo_core.grid_spec()["lat_edges_deg"]`：后者是 μ 中点边，T21 首行与真实高斯面积权
   差 22%（实测 −82.4969° vs 面积边 −83.2007°）。
4. 北→南的输入（外部表常见）在 `grid_from_atmo_spec` 里**显式翻转**并置
   `flipped_from_descending=True`；`RemapPair` 据此自动翻转场行，禁止静默南北镜像。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


def _as_1d_float(name: str, value: Any) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name} 必须是一维数组，实得 shape={a.shape}")
    if a.size < 2:
        raise ValueError(f"{name} 至少 2 点，实得 {a.size}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} 含 NaN/Inf")
    return np.ascontiguousarray(a)


def _strictly_increasing(name: str, a: np.ndarray) -> None:
    if np.any(np.diff(a) <= 0.0):
        raise ValueError(f"{name} 必须严格单调升序（调用方先归一纬序）")


@dataclass(frozen=True, eq=False)
class Grid:
    """一副全球网格：升序经纬中心 + 行面积权。"""

    name: str
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    area: np.ndarray
    kind: str = "regular"
    #: 原始输入是北→南时置 True ⇒ 与之对齐的**场行**也需要翻转（RemapPair 自动做）
    flipped_from_descending: bool = False
    _nlat: int = field(init=False, repr=False)
    _nlon: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        lat = _as_1d_float("lat_deg", self.lat_deg)
        lon = _as_1d_float("lon_deg", self.lon_deg)
        area = _as_1d_float("area", self.area)
        _strictly_increasing("lat_deg", lat)
        _strictly_increasing("lon_deg", lon)
        if np.any(lat <= -90.0) or np.any(lat >= 90.0):
            raise ValueError("lat_deg 必须在 (−90, 90) 内（这是格中心，不是边界）")
        if lon[0] < 0.0 or lon[-1] >= 360.0:
            raise ValueError("lon_deg 必须在 [0,360) 且升序（负经度请先归一）")
        if area.size != lat.size:
            raise ValueError(f"area 长度 {area.size} ≠ nlat {lat.size}")
        if np.any(area <= 0.0):
            raise ValueError("area 必须全为正（负/零权重的网格不支持）")
        total = float(area.sum())
        if abs(total - 1.0) > 1e-8:
            raise ValueError(
                f"area 行权和 = {total!r}，应为 1（本模块只支持覆盖整球的全球网格）")
        object.__setattr__(self, "lat_deg", lat)
        object.__setattr__(self, "lon_deg", lon)
        object.__setattr__(self, "area", area / total)
        object.__setattr__(self, "_nlat", int(lat.size))
        object.__setattr__(self, "_nlon", int(lon.size))

    @property
    def nlat(self) -> int:
        return self._nlat

    @property
    def nlon(self) -> int:
        return self._nlon

    def __repr__(self) -> str:
        return (f"Grid(name={self.name!r}, kind={self.kind!r}, "
                f"nlat={self.nlat}, nlon={self.nlon}, "
                f"flipped={self.flipped_from_descending})")


def grid_from_atmo_spec(spec: Mapping[str, Any], *, name: str = "atmo") -> Grid:
    """`atmo_core.grid_spec()` 的返回 dict → `Grid`（纬序守卫在此）。

    只读 `lat_deg/lon_deg/area/kind`；`lat_edges_deg` **故意不用**（见模块 docstring 第 3 条）。
    """
    for key in ("lat_deg", "lon_deg", "area"):
        if key not in spec:
            raise ValueError(f"atmo grid_spec 缺键 {key!r}")
    lat = _as_1d_float("lat_deg", spec["lat_deg"])
    area = _as_1d_float("area", spec["area"])
    if area.size != lat.size:
        raise ValueError(f"area 长度 {area.size} ≠ nlat {lat.size}")
    flipped = False
    d = np.diff(lat)
    if np.all(d < 0.0):
        lat = np.ascontiguousarray(lat[::-1])
        area = np.ascontiguousarray(area[::-1])
        flipped = True
    elif not np.all(d > 0.0):
        raise ValueError("atmo lat_deg 必须单调（升序或降序），实得非单调")
    return Grid(name=name, lat_deg=lat, lon_deg=spec["lon_deg"], area=area,
                kind=str(spec.get("kind", "gaussian")),
                flipped_from_descending=flipped)


def world_grid(world: Any, *, name: str = "world") -> Grid:
    """世界 `SphereWorld` → `Grid`（面积权 = 引擎自己的 `cell_area`，归一化）。

    用 `world.latitude_of()` / `world.cell_area()` 公共方法，不读 `_lat` 私有属性。
    """
    rows, cols = int(world.rows), int(world.cols)
    lat_deg = np.degrees(np.asarray(
        world.latitude_of(np.arange(rows, dtype=np.int64)), dtype=np.float64))
    lon_deg = 360.0 * np.arange(cols, dtype=np.float64) / cols
    cells = np.arange(rows * cols, dtype=np.int64)
    ca = np.asarray(world.cell_area(cells), dtype=np.float64).reshape(rows, cols)
    row_area = ca[:, 0]
    if not np.array_equal(ca, np.repeat(row_area[:, None], cols, axis=1)):
        raise ValueError("world.cell_area 不是行常数（世界网格口径变了？拒绝猜）")
    return Grid(name=name, lat_deg=lat_deg, lon_deg=lon_deg,
                area=row_area / row_area.sum(), kind="regular")


def regular_grid(lat_deg: Any, lon_deg: Any, *, name: str = "regular") -> Grid:
    """规则经纬网格 → `Grid`；面积权 = `Δsin(边界)/2`（与 A1 `kind="regular"` 同口径）。

    ⚠️ 世界不用本函数：世界的面积权是引擎的 `cosφ`（`world_grid`）。本函数用于
    atmo `kind="regular"` 与外部规则表。
    """
    lat = _as_1d_float("lat_deg", lat_deg)
    _strictly_increasing("lat_deg", lat)
    d = np.diff(lat)
    edges = np.concatenate(([lat[0] - 0.5 * d[0]], 0.5 * (lat[1:] + lat[:-1]),
                            [lat[-1] + 0.5 * d[-1]]))
    if lat[0] <= -80.0:
        edges[0] = -90.0
    if lat[-1] >= 80.0:
        edges[-1] = 90.0
    edges = np.clip(edges, -90.0, 90.0)
    area = np.diff(np.sin(np.radians(edges))) / 2.0
    return Grid(name=name, lat_deg=lat, lon_deg=lon_deg, area=area, kind="regular")
