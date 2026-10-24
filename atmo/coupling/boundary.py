"""世界 → atmo 边界场（设计页 §2）；只读 `EnvField` / `LightAndTemperature`。

口径（写死）：
* `h = where(land_mask, max(h_m − sea_level_m, 0), 0)`：海洋恒 0；湖/河暂按陆地（v0）。
* `mask = land_mask`（0/1；重映射后是目标格的陆地**面积份数**）。
* `alb` 按 `SURFACE_THERMOPHYSICS[surface_type][0]` 查表。
* `sst` = `light.base_temperature(°C) + 273.15`，**只在海洋格有权重**（分子/分母重映射
  在 bridge 里做；这里给原始场与 ocean 掩码）。
* `seaice` v0 恒 0。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from world.env_field import SURFACE_THERMOPHYSICS


def world_boundary_fields(env_field: Any, light: Any) -> dict[str, np.ndarray]:
    """世界网格上的边界场（2D，升序南→北；面积权由 `world_grid` 负责）。"""
    if env_field is None:
        raise ValueError("天气模式需要 EnvField（env_field.enabled=True ∧ c0b_enabled=True）")
    for name in ("h_m", "sea_level_m", "land_mask", "surface_type"):
        if getattr(env_field, name, None) is None:
            raise ValueError(
                f"EnvField.{name} 为 None —— 天气模式硬前提是 C0b 开档"
                "（env_field.c0b_enabled=True）；拒绝静默退化成'全球一种地表'")
    rows, cols = int(env_field.world.rows), int(env_field.world.cols)
    land = np.asarray(env_field.land_mask, dtype=bool).reshape(rows, cols)
    h_m = np.asarray(env_field.h_m, dtype=np.float64).reshape(rows, cols)
    sea = np.asarray(env_field.sea_level_m, dtype=np.float64)
    if sea.ndim != 0:
        sea = np.asarray(sea).reshape(-1)
        if sea.size != 1:
            raise ValueError("sea_level_m 必须是标量（世界一个海平面）")
        sea = sea[0]
    h = np.where(land, np.maximum(h_m - float(sea), 0.0), 0.0)
    st = np.asarray(env_field.surface_type).reshape(rows, cols).astype(np.int64)
    flat_st = st.ravel().tolist()
    try:
        alb = np.array([SURFACE_THERMOPHYSICS[int(s)][0] for s in flat_st],
                       dtype=np.float64).reshape(rows, cols)
    except KeyError as exc:
        raise ValueError(f"surface_type 出现未登记类别 {exc} —— 拒绝猜反照率") from None
    n = rows * cols
    temp_c = np.asarray(light.base_temperature(np.arange(n, dtype=np.int64)),
                        dtype=np.float64).reshape(rows, cols)
    return {
        "h": np.ascontiguousarray(h),
        "mask": np.ascontiguousarray(land.astype(np.float64)),
        "alb": np.ascontiguousarray(alb),
        "sst": np.ascontiguousarray(temp_c + 273.15),
        "ocean": np.ascontiguousarray((~land).astype(np.float64)),
        "seaice": np.zeros((rows, cols), dtype=np.float64),
    }


def orbit_from_world(sim_config: Any, light: Any, atmo_cfg: Any) -> dict[str, float]:
    """世界族参数 → INTERFACE §9.3 orbit（世界只有一个真源）。

    `follow_world_season=True`：一年/倾角跟世界。世界默认 `tilt_rad=0` ⇒ atmo 也无季节；
    要做真实季节必须同时改世界族（设计页 §1.3，不在 A3 偷改）。
    """
    tick_s = float(sim_config.time_scale.tick_seconds)
    season = int(getattr(light, "season_period", 0))
    year_ticks = int(sim_config.time_scale.year_ticks)
    if bool(atmo_cfg.follow_world_season):
        year_s = float(season if season > 1 else year_ticks) * tick_s
        obliquity = float(getattr(light, "tilt_rad", 0.0))
    else:
        year_s = 365.25 * 86400.0
        obliquity = float(atmo_cfg.obliquity_rad)
    rot_s = float(light.rotation_period) * tick_s
    if year_s <= 0.0 or rot_s <= 0.0:
        raise ValueError(f"orbit 年长/自转周期必须 >0（year_s={year_s}, rot_s={rot_s}）")
    return {
        "year_s": year_s,
        "rotation_period_s": rot_s,
        "obliquity_rad": obliquity,
        "eccentricity": float(atmo_cfg.eccentricity),
        "equinox_offset_s": float(atmo_cfg.equinox_offset_s),
        "solar_constant_w_m2": float(atmo_cfg.solar_constant_w_m2),
        "radius_m": float(atmo_cfg.radius_m),
    }


def validate_atmo_boundary(env_field: Any) -> None:
    """构造期硬检查（不满足就炸，不静默）。"""
    if env_field is None:
        raise ValueError("天气模式需要 env_field（enabled=True ∧ c0b_enabled=True）")
    st = getattr(env_field, "surface_type", None)
    if st is None:
        raise ValueError("天气模式需要 C0b 的 surface_type（c0b_enabled=True）")
    if not np.all(np.isfinite(np.asarray(env_field.h_m, dtype=np.float64))):
        raise ValueError("EnvField.h_m 含 NaN/Inf —— 拒绝灌入 atmo")
