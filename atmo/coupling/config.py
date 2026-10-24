"""耦合层配置（独立于 `SimConfig`，v0 由实验驱动显式注入）。

🔴 **不挂 `SimConfig`**：跨线改动（IC2）未批准前，`simulation/config.py` 不动。
默认 `enabled=False` ⇒ 本包不构造任何对象，世界 tick 路径零调用。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

#: INTERFACE §9.3 默认（世界无季节时 atmo 也用这个倾角）
_DEFAULT_OBLIQUITY_RAD = 0.40910518  # 23.4393°


@dataclass
class AtmoConfig:
    """A3 耦合配置；字段名与 INTERFACE §9 对齐（`grid.*` / `time.*` 在这里扁平化）。

    注意：`AtmoConfig` 只描述**耦合层**，不描述大气物理参数（那是 A1/A2 的
    `held_suarez` 等，其默认值由 `atmo_core` 自己给；A3 只透传 `orbit`）。
    """

    enabled: bool = False
    weather_source: str = "off"          # "off" | "atmo"

    # ---- grid（INTERFACE §9.1）----
    kind: str = "gaussian"               # "gaussian" | "regular"
    nlat: int = 32
    nlon: int = 64
    nlev: int = 8

    # ---- time（INTERFACE §9.2）----
    dt_atmo: float = 900.0
    step_mode: str = "tick_multiple"     # v0 只实现 M2′；tick_split 显式 NotImplementedError

    # ---- orbit（INTERFACE §9.3；follow_world_season=True 时由世界覆盖）----
    follow_world_season: bool = True
    obliquity_rad: float = _DEFAULT_OBLIQUITY_RAD
    eccentricity: float = 0.0167
    equinox_offset_s: float = 0.0
    solar_constant_w_m2: float = 1361.0
    radius_m: float = 6371000.0

    # ---- 耦合 ----
    precip_refresh: str = "every_exchange"   # "every_exchange" | "off"
    sst_mode: str = "world_light_base"       # v0 占位；"atmo_t2m_feedback" 留 A2
    initial_t_k: float = 288.0
    initial_ps_pa: float = 101325.0
    #: 仅合成核/桩使用；A1 动力核不消费（manifest 照样记录，防"同 seed 不同读数"）
    seed: int = 0

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        if self.weather_source not in ("off", "atmo"):
            raise ValueError(
                f"weather_source 只能是 'off'/'atmo'，实得 {self.weather_source!r}")
        if self.enabled and self.weather_source != "atmo":
            raise ValueError(
                "enabled=True 必须显式 weather_source='atmo'（防'开了但没接源'的静默）")
        if self.weather_source == "atmo" and not self.enabled:
            raise ValueError("weather_source='atmo' 但 enabled=False —— 语义矛盾，拒绝构造")
        if self.kind not in ("gaussian", "regular"):
            raise ValueError(f"grid.kind 只能是 gaussian/regular，实得 {self.kind!r}")
        for name in ("nlat", "nlon", "nlev"):
            v = int(getattr(self, name))
            if v <= 0:
                raise ValueError(f"{name} 必须为正，实得 {v}")
            setattr(self, name, v)
        if self.nlev != 8:
            raise NotImplementedError(
                f"v0 只支持 nlev=8（INTERFACE §8.2/OPEN-6），实得 {self.nlev}")
        self.dt_atmo = float(self.dt_atmo)
        if not (self.dt_atmo > 0.0) or not _isfinite(self.dt_atmo):
            raise ValueError(f"dt_atmo 必须有限且 >0，实得 {self.dt_atmo!r}")
        if self.step_mode not in ("tick_multiple", "tick_split"):
            raise ValueError(
                f"step_mode 只能是 tick_multiple/tick_split，实得 {self.step_mode!r}")
        if self.precip_refresh not in ("every_exchange", "off"):
            raise ValueError(
                f"precip_refresh 只能是 every_exchange/off，实得 {self.precip_refresh!r}")
        if self.sst_mode not in ("world_light_base", "atmo_t2m_feedback"):
            raise ValueError(f"sst_mode 非法：{self.sst_mode!r}")
        self.follow_world_season = bool(self.follow_world_season)
        self.seed = int(self.seed)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def manifest(self) -> dict[str, Any]:
        """manifest 里的 `atmo` 配置子块（时钟/网格由 `CouplingBridge.manifest` 补）。"""
        d = self.as_dict()
        return {
            "enabled": d["enabled"],
            "weather_source": d["weather_source"],
            "grid": {k: d[k] for k in ("kind", "nlat", "nlon", "nlev")},
            "time": {"dt_atmo_s": d["dt_atmo"], "step_mode": d["step_mode"]},
            "orbit": {
                "follow_world_season": d["follow_world_season"],
                # 实际生效值见 bridge.manifest()["orbit"]；这里只记配置默认，防两份读数打架
                "effective_source": ("world" if d["follow_world_season"] else "config"),
                "config_obliquity_rad": d["obliquity_rad"],
                "config_eccentricity": d["eccentricity"],
                "config_equinox_offset_s": d["equinox_offset_s"],
                "solar_constant_w_m2": d["solar_constant_w_m2"],
                "radius_m": d["radius_m"],
            },
            "coupling": {
                "precip_refresh": d["precip_refresh"],
                "sst_mode": d["sst_mode"],
                "initial_t_k": d["initial_t_k"],
                "initial_ps_pa": d["initial_ps_pa"],
                "seed": d["seed"],
            },
        }


def _isfinite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))
