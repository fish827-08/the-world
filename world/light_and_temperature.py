"""LightAndTemperature：球面世界的光照与温度场（模块一 · 文件 2）。

模块职责
--------
承接文件 1（SphereWorld 拓扑）的空间框架，加入"环境能量"维度：
- 光照：固定太阳 + 世界自转 → 昼夜在经度方向扫掠；
- 温度：由纬度（基温分布）与光照（昼夜微调）共同决定；
- 活性（activity）：温度 → 生物行为的速率/能耗因子（供引擎使用）。

关键设计（前期做简，不引入季节倾角）
-----------------------------------
- 太阳视为固定在空间中的一个方向（经度 λ_sun），世界整体绕极轴
  自转，时间推进即"太阳子午线"在经度上扫掠 → 产生昼夜；
- 单格光照 = cos(纬度) × max(0, cos(经度差))：
  赤道正午光照=1；极地无论昼夜光照都趋近 0（天然近似极夜/极昼）；
- 温度 = 纬度基温 + 光照×昼夜温差。昼夜温差刻意设小（夜晚只比白天
  低一点），极地永远冷；
- activity = 温度的函数：温度偏低时生物移动/发育变慢、能耗升高
  （具体倍率映射见 methods）。

坐标/时间约定
-------------
- flat : 平铺索引（同 SphereWorld，行优先 row*cols+col）；
- tick : 整数时间步。tick % rotation_period 决定太阳子午线经度。
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from world.sphere_world import SphereWorld


class LightAndTemperature:
    """球面光照温度场（依赖 SphereWorld 的拓扑与经度环绕语义）。

    实例属性（__slots__ 声明的全部字段）说明
    ---------------------------------------
    world : SphereWorld
        所属网格（拓扑来源：rows/cols、纬度、flat↔rc）。
    rotation_period : int
        世界自转一圈需要的 tick 数（昼夜一整个周期）。
    tilt_rad : float
        黄道倾角（弧度）。本阶段固定为 0（不做四季），保留字段仅为
        未来扩展，不影响当前计算。
    lat_base_ref : float
        参考光照基准：cos(纬度) 的指数（=1 时线性，
        调大可让极地更冷，见 illumination()）。
    t_equator : float
        赤道（纬度=0）的基温（抽象温度单位）。
    t_pole : float
        极点（|纬度|=90°）的基温（恒低温）。
    day_boost : float
        昼夜温差幅度：温度 = 基温 + 光照 × day_boost。
        刻意取小 → 夜晚只比白天冷一点。
    _daily_cos : NDArray[float64], 形状 (cols,)
        日变化系数预计算表：cos(经度差) 的原始值所在经度区间，
        见 illumination()。缓存避免每 tick 重复建表。
    """

    __slots__ = (
        "world",
        "rotation_period",
        "tilt_rad",
        "season_period",
        "_season_on",
        "lat_base_ref",
        "t_equator",
        "t_pole",
        "day_boost",
        "_daily_cos",
        # 每 tick 缓存：光照温度只依赖 tick，不依赖种群状态，
        # 每 tick 内重复调用 8+ 次，缓存可减少 80%+ 计算量。
        "_cache_tick",
        "_cache_illum_all",
        "_cache_temp_all",
        "_cache_base_all",
        # 13.8 日历—罗盘式定向迁徙：日照时长占比 P ∈ [0,1]（半日角 / π），
        # 每 tick 缓存一次（与光照/温度同范式）。
        "_cache_pp_all",
        "_cache_pp_tick",
        # 预计算表（永久不变）
        "_pre_rows",
        "_pre_cols",
        "_pre_cos_lat",
        "_pre_lat_term",
        "_lat_term_is_cos",
        "_pre_col_rad",
        # Rust 加速版
        "_rust_lt",
        "_rust_illum",
        "_rust_temp",
    )

    def __init__(
        self,
        world: SphereWorld,
        rotation_period: int = 2400,
        t_equator: float = 30.0,
        t_pole: float = -20.0,
        day_boost: float = 6.0,
        lat_base_ref: float = 1.0,
        tilt_rad: float = 0.0,
        season_period: int = 0,
    ) -> None:
        """构造光照温度场。

        参数
        ----
        world : SphereWorld
            网格拓扑对象（必须先构造，作为本场的位置编码来源）。
        rotation_period : int, 默认 2400
            自转一圈的 tick 数。2400 tick 一圈 = 昼夜各 1200 tick。
            决定"太阳子午线"扫掠速度。
        t_equator : float, 默认 30.0
            赤道基温（抽象单位，可不视为现实摄氏温度）。
        t_pole : float, 默认 -20.0
            极点基温（极地恒冷，与赤道形成纬度梯度）。
        day_boost : float, 默认 6.0
            昼夜温差幅度：正午比同纬度夜晚高 6 个单位。
            刻意小 → "夜晚只比白天冷一点"。
        lat_base_ref : float, 默认 1.0
            光照对纬度的敏感指数。越大极地光照衰减越快（极地更冷）。
        tilt_rad : float, 默认 0.0（= 无季节，与旧行为逐位等价）
            黄赤交角（弧度）。≠0 ⇒ **季节**：太阳直射点纬度（赤纬 δ）随
            `season_period` 作正弦摆动 ⇒ 富集纬度带南北移动 ⇒ 迁徙的驱动源。
            δ(t) = tilt_rad · sin(2πt / season_period)
        season_period : int, 默认 0（= 无季节）
            一个完整季节循环的 tick 数（"一年"）。仅当 `tilt_rad ≠ 0` 时生效。
            须 ≥ 3× 观测长度才看得到完整周期（R149 前置门）。

        返回
        ----
        None。构造完成后即可调用查询方法。
        """
        self.world = world
        self.rotation_period = int(rotation_period)
        self.tilt_rad = float(tilt_rad)
        self.season_period = int(season_period)
        # 季节开关：tilt≠0 且 period>1 ⇒ 启用；否则完全走旧路径（逐位等价）
        self._season_on = (abs(self.tilt_rad) > 1e-12
                           and self.season_period > 1)
        if self._season_on:
            assert self.season_period >= 2, "season_period 至少 2 tick"
            # 🔴 季节需要新的光照公式，Rust 版未实现 ⇒ 禁用 Rust 光照，
            #    强制走 Python 路径（fail-loud 由引擎侧 use_sim_core 守卫负责）。
            self._rust_lt = None
        self.lat_base_ref = float(lat_base_ref)
        self.t_equator = float(t_equator)
        self.t_pole = float(t_pole)
        self.day_boost = float(day_boost)
        # 日变化预计算：经度差 ∈ [-π, π)，cos 为此时各地相对太阳的角度
        col_rad = np.linspace(0.0, 2.0 * np.pi, self.world.cols, endpoint=False)
        # 经度差从 0（对太阳）向 π 变化，cos → 光照衰减（后面按 tick 平移）
        self._daily_cos = np.cos(col_rad)
        # 缓存初始化
        self._cache_tick = -1
        self._cache_illum_all = None
        self._cache_temp_all = None
        self._cache_base_all = None
        # 13.8：日照时长缓存（独立 tick 键 ⇒ 季节关时不产生任何额外计算）
        self._cache_pp_all = None
        self._cache_pp_tick = -1
        # 预计算：全格 cos(纬度)（永久不变，避免每 tick 重复算）
        all_flat = np.arange(self.world.n_cells, dtype=np.int64)
        all_rows, all_cols = self.world.flat_to_rc(all_flat)
        self._pre_rows = all_rows
        self._pre_cols = all_cols
        self._pre_cos_lat = np.cos(self.world.latitude_of(all_rows))
        # lat_base_ref=1.0 时 lat_term = cos_lat，无需 power
        self._lat_term_is_cos = abs(self.lat_base_ref - 1.0) < 1e-12
        if not self._lat_term_is_cos:
            self._pre_lat_term = np.power(self._pre_cos_lat, self.lat_base_ref)
        # 预计算每列经度弧度
        self._pre_col_rad = all_cols / self.world.cols * (2.0 * np.pi)
        # Rust 加速版（可用时自动启用，大世界加速 2x+）
        self._rust_lt = None
        try:
            import sim_core
            self._rust_lt = sim_core.LightTempRust(
                self.world.n_cells, self._pre_cos_lat, self._pre_col_rad,
                self.t_equator, self.t_pole, self.day_boost,
                float(self.rotation_period),
            )
            self._rust_illum = np.zeros(self.world.n_cells, dtype=np.float64)
            self._rust_temp = np.zeros(self.world.n_cells, dtype=np.float64)
        except (ImportError, AttributeError):
            pass

    # ---- 缓存（每 tick 只算一次全格） ------------------------------------

    def _ensure_cache(self, tick: int) -> None:
        """确保当前 tick 的全格光照/温度缓存已计算。每 tick 只算一次。

        优先使用 Rust+Rayon 版（大世界加速 2x+），不可用时回退 Python numpy。
        """
        if self._cache_tick == tick:
            return

        # 基温（永久缓存，只算一次）
        if self._cache_base_all is None:
            self._cache_base_all = self.t_pole + (self.t_equator - self.t_pole) * self._pre_cos_lat

        if not self._season_on and self._rust_lt is not None:
            # Rust+Rayon 版（就地写入 _rust_illum/_rust_temp）——仅无季节时可用
            self._rust_lt.compute(tick, self._rust_illum, self._rust_temp)
            self._cache_illum_all = self._rust_illum
            self._cache_temp_all = self._rust_temp
        elif self._season_on:
            # ── 季节路径（Python，标准天球公式）──
            # 太阳赤纬 δ(t) = tilt · sin(2πt / season_period)
            decl = self.tilt_rad * np.sin(
                2.0 * np.pi * (tick % self.season_period) / self.season_period
            )
            sun_lon = self.sun_longitude(tick)
            # ⚠️ `_pre_col_rad` 是**每格**经度（长度 n_cells），不是每列；
            #    reshape 成 (rows, cols) 后取第一行即得每列经度（同行内列索引一致）。
            col_rad = self._pre_col_rad.reshape(
                self.world.rows, self.world.cols)[0]        # (cols,)
            lon_diff = col_rad - sun_lon
            lon_diff = (lon_diff + np.pi) % (2.0 * np.pi) - np.pi
            cos_h = np.cos(lon_diff)                        # (cols,)
            # 天顶角余弦 = sin(φ)·sin(δ) + cos(φ)·cos(δ)·cos(h)
            lat_rows = self.world.latitude_of(
                np.arange(self.world.rows, dtype=np.int64))  # (rows,)
            sin_lat = np.sin(lat_rows)[:, None]              # (rows,1)
            cos_lat = np.cos(lat_rows)[:, None]              # (rows,1)
            z = (sin_lat * np.sin(decl)
                 + cos_lat * np.cos(decl) * cos_h[None, :])  # (rows, cols)
            illum_grid = np.maximum(0.0, z)                  # 地平线下 ⇒ 0
            # lat_base_ref ≠ 1：与旧实现同义——对 cos(φ) 取幂
            if not self._lat_term_is_cos:
                ratio = np.where(
                    cos_lat[:, 0] > 1e-12,
                    np.power(cos_lat[:, 0], self.lat_base_ref - 1.0),
                    0.0,
                )
                illum_grid = illum_grid * ratio[:, None]
            illum_flat = np.ascontiguousarray(illum_grid.reshape(-1))
            self._cache_illum_all = illum_flat
            if self._cache_base_all is None:
                self._cache_base_all = (self.t_pole
                                        + (self.t_equator - self.t_pole)
                                        * self._pre_cos_lat)
            self._cache_temp_all = (self._cache_base_all
                                    + illum_flat * self.day_boost)
        else:
            # Python numpy 回退
            lat_term = self._pre_cos_lat if self._lat_term_is_cos else self._pre_lat_term
            sun_lon = self.sun_longitude(tick)
            lon_diff = self._pre_col_rad - sun_lon
            lon_diff = (lon_diff + np.pi) % (2.0 * np.pi) - np.pi
            day_term = np.maximum(0.0, np.cos(lon_diff))
            self._cache_illum_all = lat_term * day_term
            self._cache_temp_all = self._cache_base_all + self._cache_illum_all * self.day_boost

        self._cache_tick = tick

    # ---- 光照 --------------------------------------------------------------

    def sun_longitude(self, tick: int) -> float:
        """计算给定 tick 时刻太阳子午线所在经度（弧度，0..2π）。

        参数
        ----
        tick : int
            当前时间步（可为任意非负整数，内部对周期取模）。

        返回
        ----
        float : 太阳子午线经度（弧度，范围 [0, 2π)）。
        """
        phase = (tick % self.rotation_period) / self.rotation_period
        return phase * 2.0 * np.pi

    def illumination(self, flat, tick: int) -> np.ndarray:
        """计算格子光照强度（0=全黑，1=正午对日）。

        优化：全格查询返回每 tick 缓存；子集查询从缓存中索引（O(N) 查表 vs O(N) 重算）。
        """
        flat = np.asarray(flat, dtype=np.int64)
        scalar = flat.ndim == 0
        flat = flat.reshape(-1)

        # 确保缓存已计算（全格）
        self._ensure_cache(tick)

        # 从全格缓存中索引（无论是全格还是子集，都走这条路）
        out = self._cache_illum_all[flat]
        return out.item(0) if scalar else out

    # ---- 日照时长（13.8 日历—罗盘式定向迁徙） ------------------------------

    def photoperiod(self, flat, tick: int) -> np.ndarray:
        """日照时长占比 P ∈ [0, 1]（= 半日角 H₀ / π）。无季节时恒 0.5。

        定义
        ----
        该纬度在该"日"内，太阳位于地平线以上的时间占比：

            cos(H₀) = −tan(φ)·tan(δ(t)) ,  H₀ = arccos(clip(·, −1, 1)) ∈ [0, π]
            P(φ, t) = H₀ / π

        语义对照：``H₀ = π/2``（P=0.5）⇒ 昼夜等长；``H₀ = 0`` ⇒ 极夜（P=0）；
        ``H₀ = π`` ⇒ 极昼（P=1）。

        🔴 **绝不用 `illumination` 反推 P** —— ``illumination`` 是**光照强度**
        且被 ``lat_base_ref`` 的纬度因子乘过；``P`` 是**昼夜时间占比**。
        两者是不同的量，互推会把它们混成一个（C9 型错误）。
        本方法用**独立的解析式**，只消费已有的 ``tilt_rad`` / ``season_period``。

        与真实鸟类的对应：日照时长是**跨类群的主导迁徙触发器**，且它是**先行信号**
        （光照变化**发生在食物变化之前**，且绝对可靠）⇒ 鸟在食物**还充足**时就开始迁徙
        （Gwinner 1986/1996；Berthold 1996/2001）。这正是本方法存在的理由：
        13.7 已证明"感知食物梯度"这条路被信噪比堵死（食物带移动比个体随机运动慢
        17–20 倍），故改用一个**远离噪声的日历信号**。

        参数
        ----
        flat : int | NDArray[int64]
            全格平坦索引（单格或数组均可）。
        tick : int
            当前时间步。查季节相位用。

        返回
        ----
        float（标量输入）或 NDArray[float64]，形状与 `flat` 一致，值域 [0, 1]。

        实现要点
        --------
        * **每 tick 缓存一次**（`_cache_pp_all` / `_cache_pp_tick`，照 `_ensure_cache` 范式）。
        * 🔴 **极点数值守卫**：``|φ| → π/2`` 时 ``tan(φ) → ∞``。用
          ``φ_clip = clip(φ, ±(π/2 − ε))``（ε=1e-9）截断，**并断言结果 `isfinite`**。
          这是本设计**最可能产生 NaN 并静默污染全批**的点（设计稿 §3.3 守卫 2 / 风险 R1）。
        * 无季节（``_season_on=False``，即 δ≡0）⇒ ``P ≡ 0.5``，
          **不进新分支、不影响任何既有行为**。
        """
        flat_arr = np.asarray(flat, dtype=np.int64)
        scalar = flat_arr.ndim == 0
        flat_arr = flat_arr.reshape(-1)

        if self._cache_pp_tick != tick:
            if not self._season_on:
                # 无季节：δ ≡ 0 ⇒ tanδ = 0 ⇒ cos H₀ = 0 ⇒ H₀ = π/2 ⇒ P ≡ 0.5
                self._cache_pp_all = np.full(
                    self.world.n_cells, 0.5, dtype=np.float64)
            else:
                # 太阳赤纬 δ(t)（与 `_ensure_cache` 季节分支**逐字同式**）
                decl = self.tilt_rad * np.sin(
                    2.0 * np.pi * (tick % self.season_period) / self.season_period
                )
                # 🔴 极点守卫：|φ| → π/2 时 tan(φ) → ∞ ⇒ 必须先截断，
                #    否则 cos_h0 出 ±inf，arccos 出 NaN，静默污染全批（R1）。
                eps = 1e-9
                lat_rows = self.world.latitude_of(
                    np.arange(self.world.rows, dtype=np.int64)).astype(np.float64)
                phi_c = np.clip(lat_rows, -(np.pi / 2.0 - eps),
                                np.pi / 2.0 - eps)
                cos_h0 = -np.tan(phi_c) * np.tan(decl)
                cos_h0 = np.clip(cos_h0, -1.0, 1.0)     # 极昼/极夜：|·|>1 ⇒ 饱和
                h0 = np.arccos(cos_h0)                  # ∈ [0, π]
                pp_rows = h0 / np.pi                    # ∈ [0, 1]
                # 展开到每格（同一行内所有格共用该行纬度）
                self._cache_pp_all = np.ascontiguousarray(
                    np.repeat(pp_rows, self.world.cols))
            # 🔴 P3 断言：NaN / 越界 ⇒ 立刻 fail-loud（不许静默污染全批）
            if not np.isfinite(self._cache_pp_all).all():
                raise FloatingPointError(
                    "photoperiod 产生非有限值（NaN/Inf）——极点 tan(φ) 守卫失效。"
                    "这会在全批静默传播，故 fail-loud（设计稿 R1 / P3）。"
                )
            if not ((self._cache_pp_all >= 0.0) & (self._cache_pp_all <= 1.0)).all():
                raise FloatingPointError(
                    f"photoperiod 越界 [0,1]（min={self._cache_pp_all.min():.6g}, "
                    f"max={self._cache_pp_all.max():.6g}）（设计稿 P3）。"
                )
            self._cache_pp_tick = tick

        out = self._cache_pp_all[flat_arr]
        return out.item(0) if scalar else out

    # ---- 温度 --------------------------------------------------------------

    def base_temperature(self, flat) -> np.ndarray:
        """计算纬度基温（无昼夜影响，仅随纬度变化）。

        优化：基温不依赖 tick，全格永久缓存；子集查询从缓存索引。
        """
        flat = np.asarray(flat, dtype=np.int64)
        scalar = flat.ndim == 0
        flat = flat.reshape(-1)

        # 确保基温缓存已计算
        if self._cache_base_all is None:
            self._cache_base_all = self.t_pole + (self.t_equator - self.t_pole) * self._pre_cos_lat

        out = self._cache_base_all[flat]
        return out.item(0) if scalar else out

    def temperature(self, flat, tick: int) -> np.ndarray:
        """计算当前时刻的实时温度（基温 + 昼夜微调）。

        优化：全格缓存 + 子集查询从缓存索引。
        """
        flat = np.asarray(flat, dtype=np.int64)
        scalar = flat.ndim == 0
        flat = flat.reshape(-1)

        self._ensure_cache(tick)
        out = self._cache_temp_all[flat]
        return out.item(0) if scalar else out

    # ---- 活性（生物响应） ---------------------------------------------------

    def activity_factor(self, flat, tick: int) -> np.ndarray:
        """计算温度 → 生物行为活性因子（范围 0..1，温度越高活性越高）。

        活性 = clip((T - t_min) / (t_ref - t_min), 0, 1)。
        - 温度 ≥ t_ref（默认 10°）→ 活性 1（正常行动）；
        - 温度 ≤ t_min（默认 -10°）→ 活性 0（近乎停滞）；
        - 中间线性过渡。极地因温度低活性低 → 移动/发育慢、
          单位行动能耗更高（能耗反向，见引擎用法）。

        参数
        ----
        flat : int 或 NDArray[int64]
            平铺索引（支持标量或数组批量查询）。
        tick : int
            当前时间步（温度随时间变化，活性随之变化）。

        返回
        ----
        NDArray[float64] : 与入参同形状的活性因子，范围 [0, 1]。
        """
        t = self.temperature(flat, tick)
        t_min, t_ref = -10.0, 10.0
        out = np.asarray(t)
        out = np.clip((out - t_min) / (t_ref - t_min), 0.0, 1.0)
        return out.item(0) if out.ndim == 0 else out

    # ---- 调试 / 展示 -------------------------------------------------------

    def __repr__(self) -> str:
        """生成对象的人类可读字符串。

        参数
        ----
        无。

        返回
        ----
        str : 例如
            "LightAndTemperature(rotation=2400, equator=30, pole=-20,
            day_boost=6)"
        """
        return (
            f"LightAndTemperature(rotation={self.rotation_period}, "
            f"equator={self.t_equator}, pole={self.t_pole}, "
            f"day_boost={self.day_boost})"
        )