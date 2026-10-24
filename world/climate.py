"""气候源接口（B3）：把「世界读气候」抽成一个可替换的实现类。

设计目标（派工 B3）
------------------
**换气候源 = 加一个实现类**，不改世界代码。之后接活大气（P4）、接外部 GCM 表，
都只换实现。

本文件的纪律
------------
* **零新物理、零新 RNG、零配置字段**：默认路径逐位等价（`EnvFieldSource` 是纯包装）。
* **零开销**：`refresh()` 对静态源是 no-op，且引擎**不调用**它（不引入每 tick 新调用）。
* **未接的通道显式报未接**：`precip_mm_yr()` / `wind()` 抛 `NotImplementedError`，
  **不许返回 0 冒充**（C5 命名/口径纪律）。
* 不加指纹：`climate_source_name` 只进 probe/诊断，不进 `switches`。

当前实现
--------
`EnvFieldSource`：包装 `world/env_field.py`（地形→气候态水分→产能因子）与
`world/light_and_temperature.py`（温度剖面），**不复制任何公式**——同对象同方法透传，
因此逐位等价。

未来实现（B4 起）
----------------
`LiveAtmosphereSource`：读 `Atmosphere` 的 `t_surf`/`p_surf`/`u_*`/降水率；
`GCMTableSource`：读外部月平均气候表（W0-A 已建骨架，见文件末尾）。两者都只需实现本 Protocol。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ClimateSource(Protocol):
    """气候源协议：世界只通过这 8 个入口读气候。"""

    name: str

    def bootstrap(self, world: Any) -> None:
        """构造期一次性初始化（静态源 = no-op；活源可在此建网格/读表）。"""

    def refresh(self, tick: int) -> None:
        """按源自定的节拍刷新（静态源 = no-op；引擎**不调用**它）。"""

    def growth_factor(self):
        """资源层：斑块再生倍率（`None` = 不介入 = 逐位等价）。"""

    def capacity_factor(self):
        """资源层：斑块容量倍率（`None` = 不介入）。"""

    def temperature_c(self, *args, **kwargs):
        """生物层每 tick 温度（℃）。签名与调用形态照抄现状以保逐位。"""

    def precip_mm_yr(self):
        """降水（mm/yr）。🔴 P4 前未接。"""

    def wind(self):
        """风（m/s 分量）。🔴 P4 前未接。"""

    def probe(self) -> dict:
        """只读诊断键（进 probe/CSV 诊断段，**不进指纹**）。"""

    def state(self) -> dict:
        """快照载体（静态源无状态 ⇒ 空 dict）。"""

    def restore(self, payload: dict) -> None:
        """从快照恢复（静态源 = no-op）。"""


class EnvFieldSource:
    """现有路径的**纯包装**：`EnvField` + `LightAndTemperature`，零公式复制。

    为什么是包装而不是重写：派工要求「默认路径逐位等价」。同对象同方法透传时，
    浮点运算顺序完全不变；重写公式必然引入 ULP 级差异（C7 红线）。
    """

    name = "env_field"

    def __init__(self, env_field, light) -> None:
        self._env = env_field
        self._light = light

    # ---- 生命周期 ----
    def bootstrap(self, world: Any) -> None:  # noqa: ARG002 - 接口签名
        """静态源：`EnvField` 在引擎构造期已算完（构造期零 tick）⇒ 这里无事可做。"""

    def refresh(self, tick: int) -> None:  # noqa: ARG002 - 接口签名
        """静态源 = no-op。🔴 引擎不调用本方法（不引入每 tick 新调用）。"""

    # ---- 资源层 ----
    def growth_factor(self):
        return self._env.growth_factor() if self._env is not None else None

    def capacity_factor(self):
        return self._env.capacity_factor() if self._env is not None else None

    # ---- 生物层温度（透传 ⇒ 逐位）----
    def temperature_c(self, flat, tick: int) -> np.ndarray:
        return self._light.temperature(flat, tick)

    # ---- 未接通道：显式报未接（不许 0 冒充）----
    def precip_mm_yr(self):
        raise NotImplementedError(
            "未接：降水仍由 EnvField 的 proxy 提供（P4 接活大气后再换本实现）")

    def wind(self):
        raise NotImplementedError(
            "未接：风尚未接世界（P4）；当前 `Atmosphere.u_*` 是诊断量、不回写主网格")

    # ---- 诊断与快照 ----
    def probe(self) -> dict:
        return {"climate_source_name": self.name,
                "climate_has_growth_factor": self._env is not None}

    def state(self) -> dict:
        return {}

    def restore(self, payload: dict) -> None:  # noqa: ARG002 - 接口签名
        """静态源无状态 ⇒ no-op。"""


# ============================================================================
# W0-A / A5：外部 GCM 气候表源（骨架，未接通）
# ============================================================================
#: 气候表格式契约（写入即冻结；改格式 = 改接口，须走「接口变更单」）
#:
#: ``.npz``（numpy 压缩包，零额外依赖）至少包含：
#:   ``lat``          (nlat,)        纬度（度，北→南或南→北均可，读入时归一为南→北）
#:   ``lon``          (nlon,)        经度（度，0..360 或 -180..180 均可，读入时归一为 0..360）
#:   ``temperature_c``(nlat, nlon) 或 (12, nlat, nlon)   近地面气温（℃）
#:   ``precip_mm_yr`` (nlat, nlon) 或 (12, nlat, nlon)   年降水（mm/yr）
#:   ``u``            (nlat, nlon) 或 (12, nlat, nlon)   纬向风（m/s，东正）
#:   ``v``            (nlat, nlon) 或 (12, nlat, nlon)   经向风（m/s，北正）
#: 可选 ``meta_json``（字符串）：来源 / 版本 / 分辨率 / 积分年数，原样进 probe。
#:
#: 约定：表是**气候态**（多年平均），不是某一时刻的天气；季节表用 12 个月层，
#: 月层插值方式**未定**（W1 决定，见 `refresh()`）。
GCM_TABLE_FORMAT = "npz-climatology-v0"

_REQUIRED_1D_KEYS = ("lat", "lon")
_OPTIONAL_FIELD_KEYS = ("temperature_c", "precip_mm_yr", "u", "v")



class GCMTableSource:
    """外部 GCM 离线气候表源（W1 已接通物理通道；进程外、只读表）。

    设计约束（B3/总纲 §三）
    ----------------------
    * 只实现 `ClimateSource` 协议，**不改世界代码**、**不加 config 字段**（显式构造注入）。
    * 默认路径仍是 `EnvFieldSource` ⇒ 本类零副作用（关档 digest 不受影响）。
    * 表是**气候态**（年平或 12 月层），不是逐时天气；无表/未定参数时**显式报错**，不返回 0 冒充。

    网格与重映射
    ------------
    `bootstrap(world)` 会把表从 GCM 规则网格**保守（面积权）重映射**到世界的
    `rows × cols` 网格（`atmo_bridge.remap`，纬度按 sin 面积、经度按角度交叠，行和归一）。
    ⇒ 气候表自身的值域/积分被保住（降水不会凭空增减）。

    季节（12 月层）
    --------------
    世界"一年"多长是**世界参数**（`LightConfig.season_period`），与本类无关；
    因此必须**显式**给 `year_ticks`（构造参数，不进 config）。给了 ⇒ 读取时按 tick 做
    **月层线性插值**（首尾循环）；没给 + 表是 12 月层 ⇒ `bootstrap` 直接报错（不猜）。
    """

    name = "gcm_table"

    def __init__(self, table_path: Optional[str | Path] = None, *,
                 year_ticks: Optional[int] = None) -> None:
        self._path = Path(table_path) if table_path is not None else None
        self._year_ticks = int(year_ticks) if year_ticks else None
        self._loaded = False
        self._table: dict[str, np.ndarray] = {}
        self._meta: dict[str, Any] = {}
        self._rows = 0
        self._cols = 0
        self._annual: dict[str, np.ndarray] = {}      # key -> (rows, cols)
        self._monthly_cache: dict[str, np.ndarray] = {}  # key -> (12, rows, cols)
        self._wl = None
        self._wo = None

    # ---- 生命周期 ----
    def bootstrap(self, world: Any) -> None:
        """读表 → 校验 → 建重映射权重 → 预置年平场（**只做构造期工作**）。"""
        if self._path is None:
            raise NotImplementedError(
                "GCMTableSource 未给 table_path（格式见 world.climate.GCM_TABLE_FORMAT）")
        if not self._path.exists():
            raise FileNotFoundError(f"GCM 气候表不存在：{self._path}")

        with np.load(self._path, allow_pickle=False) as npz:
            for key in _REQUIRED_1D_KEYS:
                if key not in npz:
                    raise ValueError(f"GCM 气候表缺必需键 {key!r}：{self._path}")
            lat = np.asarray(npz["lat"], dtype=np.float64).ravel()
            lon = np.asarray(npz["lon"], dtype=np.float64).ravel()
            self._table = {"lat": lat, "lon": lon}
            for key in _OPTIONAL_FIELD_KEYS:
                if key in npz:
                    self._table[key] = np.asarray(npz[key], dtype=np.float64)
            if "meta_json" in npz:
                import json

                self._meta = json.loads(str(npz["meta_json"]))

        # ---- 🔴 纬度/经度顺序归一（2026-10-19 A17.3 实测踩坑）----
        #   ExoPlaSim 的 lat 是**北→南递减**；`atmo_bridge.remap` 按升序算区间。
        #   不归一 ⇒ 区间为负 ⇒ "目标纬行 0 与源网格无交叠"（s2 直接抛），
        #   或更糟：s1 那种不报错但**南北镜像**（温度场上下颠倒，肉眼很难发现）。
        lat_order = np.argsort(lat, kind="stable")
        lon_order = np.argsort(lon, kind="stable")
        need_lat = not np.array_equal(lat_order, np.arange(lat.size))
        need_lon = not np.array_equal(lon_order, np.arange(lon.size))
        if need_lat or need_lon:
            lat = lat[lat_order]
            lon = lon[lon_order]
            for key, arr in self._table.items():
                if key in ("lat", "lon"):
                    continue
                a = np.take(arr, lat_order, axis=-2)
                self._table[key] = np.take(a, lon_order, axis=-1)
            self._table["lat"] = lat
            self._table["lon"] = lon
        self._lat_order_fixed = bool(need_lat or need_lon)

        self._validate(lat, lon)
        if any(v.ndim == 3 for v in self._table.values() if isinstance(v, np.ndarray)
               and v.ndim == 3) and self._year_ticks is None:
            raise NotImplementedError(
                "气候表是 12 月层，但未给 year_ticks ⇒ 无法定'世界一年'的相位。"
                "请显式传 GCMTableSource(table_path=..., year_ticks=<LightConfig.season_period>)")

        # 世界网格 → 权重（世界经度中心 = j*360/cols，见 world/env_field.py:346 同口径）
        self._rows, self._cols = int(world.rows), int(world.cols)
        lat_world = np.degrees(np.asarray(world._lat, dtype=np.float64)).ravel()
        lon_world = 360.0 * np.arange(self._cols, dtype=np.float64) / self._cols

        from atmo_bridge.remap import build_weights, remap_field

        self._wl, self._wo = build_weights(lat, lon, lat_world, lon_world)
        for key, arr in self._table.items():
            if key in ("lat", "lon"):
                continue
            self._annual[key] = remap_field(arr.mean(axis=0) if arr.ndim == 3 else arr,
                                            self._wl, self._wo)

        self._meta.setdefault("format", GCM_TABLE_FORMAT)
        self._meta.setdefault("table_path", str(self._path))
        self._meta["world_grid"] = [self._rows, self._cols]
        self._meta["year_ticks"] = self._year_ticks
        self._loaded = True

    def refresh(self, tick: int) -> None:  # noqa: ARG002 - 接口签名
        """表是静态气候态 ⇒ no-op（季节插值发生在读取时，按 tick 算相位）。"""
        if not self._loaded:
            raise RuntimeError("refresh 前必须先 bootstrap(world)")

    # ---- 资源层：表里没有再生规则，显式报未接 ----
    def growth_factor(self):
        raise NotImplementedError(
            "GCMTableSource.growth_factor 未接：表只有 T/P/风，没有斑块再生规则；"
            "W2 决定'由降水派生再生倍率'之前不得猜测")

    def capacity_factor(self):
        raise NotImplementedError("GCMTableSource.capacity_factor 未接：同 growth_factor")

    # ---- 物理通道 ----
    def temperature_c(self, flat, tick: int) -> np.ndarray:
        """按 `flat`（格索引数组）取近地面气温（℃）；12 月表按 tick 线性插值。"""
        self._require_loaded()
        key = "temperature_c"
        arr = self._field_at(key, tick)
        return np.asarray(arr).ravel()[np.asarray(flat, dtype=np.int64)]

    def precip_mm_yr(self) -> np.ndarray:
        """年降水（mm/yr），(rows, cols)；12 月表取 12 月平均。"""
        self._require_loaded()
        if "precip_mm_yr" not in self._annual:
            raise NotImplementedError("该气候表没有 precip_mm_yr 字段")
        return self._annual["precip_mm_yr"]

    def wind(self) -> dict:
        """风（m/s），返回 `{"u": (rows,cols), "v": (rows,cols) | None}`。

        ⚠️ 极行（第一/最后一行）方向退化 ⇒ `u` 取该行**纬向平均**、`v` 置 0（并在此声明）。
        当前世界还没有"风如何进移动/热通量"的口径（W2 决定）⇒ 本方法只提供场，不接线。
        """
        self._require_loaded()
        out: dict[str, Any] = {}
        for comp in ("u", "v"):
            if comp not in self._annual:
                out[comp] = None
                continue
            f = np.array(self._annual[comp], dtype=np.float64, copy=True)
            f[0, :] = f[0, :].mean()
            f[-1, :] = f[-1, :].mean()
            if comp == "v":
                f[0, :] = 0.0
                f[-1, :] = 0.0
            out[comp] = f
        if out.get("u") is None and out.get("v") is None:
            raise NotImplementedError("该气候表没有 u/v 字段")
        return out

    # ---- 诊断与快照 ----
    def probe(self) -> dict:
        return {
            "climate_source_name": self.name,
            "climate_loaded": self._loaded,
            "climate_table_path": str(self._path) if self._path is not None else None,
            "climate_table_fields": sorted(k for k in self._table if k not in ("lat", "lon")),
            "climate_table_shape": [int(self._table["lat"].size), int(self._table["lon"].size)]
            if "lat" in self._table else None,
            "climate_table_months": 12 if any(
                isinstance(v, np.ndarray) and v.ndim == 3 for v in self._table.values()) else 1,
            "climate_year_ticks": self._year_ticks,
            "climate_grid_reordered": bool(getattr(self, "_lat_order_fixed", False)),
            "climate_table_meta": dict(self._meta),
        }

    def state(self) -> dict:
        """表是外部静态文件 ⇒ 快照不携带（可由路径 + 文件哈希重建）。"""
        return {}

    def restore(self, payload: dict) -> None:  # noqa: ARG002 - 接口签名
        """静态表无状态 ⇒ no-op。"""

    # ---- 内部 ----
    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("GCMTableSource 未 bootstrap：先调用 bootstrap(world)")

    def _field_at(self, key: str, tick: int) -> np.ndarray:
        """取某字段在 `tick` 的场（年表直接返回；月表做月层线性插值）。"""
        if key not in self._annual:
            raise NotImplementedError(f"该气候表没有 {key} 字段")
        raw = self._table.get(key)
        if raw is None or raw.ndim == 2:
            return self._annual[key]
        # 12 月层：phase ∈ [0,12)
        phase = (int(tick) % self._year_ticks) / float(self._year_ticks) * 12.0
        m0 = int(np.floor(phase)) % 12
        w = phase - np.floor(phase)
        m1 = (m0 + 1) % 12
        cache = self._monthly(key)
        return (1.0 - w) * cache[m0] + w * cache[m1]

    def _monthly(self, key: str) -> np.ndarray:
        """按月层重映射并缓存（12, rows, cols）。"""
        if key in self._monthly_cache:
            return self._monthly_cache[key]
        from atmo_bridge.remap import remap_field

        raw = self._table[key]
        arr = np.empty((12, self._rows, self._cols), dtype=np.float64)
        for m in range(12):
            arr[m] = remap_field(raw[m], self._wl, self._wo)
        self._monthly_cache[key] = arr
        return arr

    # ---- 内部：形状校验 ----
    def _validate(self, lat: np.ndarray, lon: np.ndarray) -> None:
        if lat.size < 2 or lon.size < 2:
            raise ValueError(f"GCM 气候表网格太小：nlat={lat.size}, nlon={lon.size}")
        if not (np.all(np.isfinite(lat)) and np.all(np.isfinite(lon))):
            raise ValueError("GCM 气候表的 lat/lon 含非有限值")
        nlat, nlon = lat.size, lon.size
        for key in _OPTIONAL_FIELD_KEYS:
            if key not in self._table:
                continue
            arr = self._table[key]
            if arr.shape == (nlat, nlon) or arr.shape == (12, nlat, nlon):
                continue
            raise ValueError(
                f"GCM 气候表 {key!r} 形状 {arr.shape} 与网格 ({nlat}, {nlon}) 不符；"
                "允许 (nlat, nlon) 或 (12, nlat, nlon)")
