"""状态桥：边界场→重映射→`core.step`→出口场→重映射回世界（设计页 §2–§4）。

时钟 = `CouplingClock`（M2′）；交换相位由整数 `t % k` 决定，不用 RNG。
🔴 OPEN-5：`core.step` 原地更新 ⇒ 本模块**不持有跨步别名**：每次交换后把出口场
copy 成世界侧缓存；`state` 只做零拷贝引用（`bridge.state is core_state`）。
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from .boundary import orbit_from_world, world_boundary_fields
from .clock import CouplingClock
from .grid import Grid, grid_from_atmo_spec, world_grid
from .remap import RemapPair

SECONDS_PER_YEAR = 31_557_600.0
_BOUNDARY_KEYS = ("h", "mask", "alb", "sst", "ocean", "seaice")
_EXPORT_OPTIONAL = ("u10", "v10", "cloud", "sw_sfc", "lw_sfc")


class CouplingBridge:
    """一世界 ↔ 一 atmo 核的桥；`enabled=False` 时**不构造**本类。"""

    def __init__(
        self,
        world: Any,
        atmo_cfg: Any,
        core: Any,
        tick_seconds: float,
        *,
        boundary_fn: Callable[[], Mapping[str, np.ndarray]],
        orbit: Mapping[str, float],
        sim_config: Any = None,
    ) -> None:
        if not bool(getattr(atmo_cfg, "enabled", False)):
            raise ValueError("CouplingBridge 只在 AtmoConfig.enabled=True 时构造")
        self.cfg = atmo_cfg
        self.core = core
        self.sim_config = sim_config
        self.clock = CouplingClock(tick_seconds, atmo_cfg.dt_atmo, atmo_cfg.step_mode)
        self._boundary_fn = boundary_fn
        self.orbit = {k: float(v) for k, v in orbit.items()}
        self.world_grid: Grid = world_grid(world)
        self.atmo_grid: Grid = grid_from_atmo_spec(core.grid_spec(), name="atmo")
        self.to_atmo = RemapPair.build(self.world_grid, self.atmo_grid)
        self.to_world = RemapPair.build(self.atmo_grid, self.world_grid)
        self.state = core.make_state()
        self.initial_digest = str(core.digest(self.state))
        self.exchanges = 0
        self.last_tick: int | None = None
        self.last_atmo_exports: dict[str, Any] | None = None
        self.last_world_exports: dict[str, np.ndarray] | None = None
        self.last_diag: Mapping[str, Any] | None = None
        self.last_forcing_summary: dict[str, Any] | None = None

    @classmethod
    def from_engine(cls, engine: Any, atmo_cfg: Any, core: Any) -> "CouplingBridge":
        """从已构造引擎取 world/light/env_field（A3.4 生产接线用）。"""
        sim_cfg = engine.config
        tick_s = float(sim_cfg.time_scale.tick_seconds)
        return cls(
            engine.world, atmo_cfg, core, tick_s,
            boundary_fn=lambda: world_boundary_fields(engine.env_field, engine.light),
            orbit=orbit_from_world(sim_cfg, engine.light, atmo_cfg),
            sim_config=sim_cfg,
        )

    # ---------------- 交换 ----------------

    def exchange(self, tick: int) -> Mapping[str, Any]:
        """在世界 tick `t`（必须 `t % k == 0`）做一次完整交换，返回 diag。"""
        t = int(tick)
        if not self.clock.is_exchange(t):
            raise ValueError(
                f"tick {t} 不是交换相位（k={self.clock.k} ⇒ t % k == 0）")
        if self.last_tick is not None and t < self.last_tick:
            raise ValueError(f"tick 回退：{t} < 上次交换 {self.last_tick}（零阶保持不允许倒带）")
        forcing = self._boundary_forcing(t)
        diag = self.core.step(self.state, forcing, dt=self.clock.dt_step)
        if not isinstance(diag, Mapping):
            raise RuntimeError("core.step 必须返回 diag dict（INTERFACE §8.3）")
        atmo_exports = self.core.export(self.state)
        if not isinstance(atmo_exports, Mapping):
            raise RuntimeError("core.export 必须返回 dict（INTERFACE §10）")
        self.last_atmo_exports = self._validated_exports(atmo_exports)
        self.last_world_exports = self._remap_exports(self.last_atmo_exports)
        self.last_diag = diag
        self.last_forcing_summary = self._forcing_summary(forcing, t)
        self.last_tick = t
        self.exchanges += 1
        return diag

    def on_tick(self, tick: int) -> bool:
        """驱动入口：仅在交换相位推进 atmo；同一 tick 重复调用是幂等的。

        ⚠️ 引擎 `step()` 先 `_tick += 1` 再推进 ⇒ 驱动应在 `engine.step()` **之前**
        调 `on_tick(engine.tick + 1)`，让边界/出口对应当前正要推进的那一个 tick。
        """
        t = int(tick)
        if self.last_tick is not None:
            if t < self.last_tick:
                raise ValueError(f"tick 回退：{t} < 上次交换 {self.last_tick}")
            if t == self.last_tick:
                return False
        if not self.clock.is_exchange(t):
            return False
        self.exchange(t)
        return True

    def _boundary_forcing(self, tick: int) -> dict[str, Any]:
        raw = self._boundary_fn()
        if not isinstance(raw, Mapping):
            raise RuntimeError("boundary_fn 必须返回 dict")
        missing = [k for k in _BOUNDARY_KEYS if k not in raw]
        if missing:
            raise ValueError(f"boundary_fn 缺键 {missing}")
        src_shape = (self.world_grid.nlat, self.world_grid.nlon)
        for k in ("h", "mask", "alb", "sst", "ocean"):
            a = np.asarray(raw[k], dtype=np.float64)
            if a.shape != src_shape:
                raise ValueError(f"边界场 {k} 形状 {a.shape} ≠ {src_shape}")
            if not np.all(np.isfinite(a)):
                raise ValueError(f"边界场 {k} 含 NaN/Inf")

        w2a = self.to_atmo
        h = w2a.remap(raw["h"])
        mask = np.clip(w2a.remap(raw["mask"]), 0.0, 1.0)
        alb = w2a.remap(raw["alb"])
        seaice = w2a.remap(raw["seaice"])
        ocean = np.asarray(raw["ocean"], dtype=np.float64)
        sst_raw = np.asarray(raw["sst"], dtype=np.float64)
        num = w2a.remap(sst_raw * ocean)
        den = w2a.remap(ocean)
        w = self.world_grid.area
        row_ocean = ocean.mean(axis=1)
        ocean_mean = float(np.sum((sst_raw * ocean).mean(axis=1) * w)
                           / max(float(np.sum(row_ocean * w)), 1e-30))
        sst = np.where(den > 1e-12, num / np.maximum(den, 1e-30), ocean_mean)
        return {
            "h": h, "mask": mask, "alb": alb, "sst": sst, "seaice": seaice,
            "orbit": dict(self.orbit),
            "tick": int(tick),
            "time_s": self.clock.time_s(tick),
            "dt_step_s": self.clock.dt_step,
            "synthetic": bool(getattr(self.core, "synthetic", False)),
        }

    def _validated_exports(self, exports: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        need = (self.atmo_grid.nlat, self.atmo_grid.nlon)
        for key in ("t2m", "precip"):
            if key not in exports:
                raise ValueError(f"core.export 缺 {key!r}（INTERFACE §10 契约）")
            a = np.asarray(exports[key], dtype=np.float64)
            if a.shape != need:
                raise ValueError(f"出口场 {key} 形状 {a.shape} ≠ {need}")
            if not np.all(np.isfinite(a)):
                raise ValueError(f"出口场 {key} 含 NaN/Inf")
            if key == "precip" and float(a.min()) < 0.0:
                raise ValueError(f"出口场 precip 出现负值 min={float(a.min()):.3g}")
            out[key] = np.array(a, dtype=np.float64, copy=True)
        for key in _EXPORT_OPTIONAL:
            if key in exports:
                a = np.asarray(exports[key], dtype=np.float64)
                if a.shape != need:
                    raise ValueError(f"出口场 {key} 形状 {a.shape} ≠ {need}")
                if not np.all(np.isfinite(a)):
                    raise ValueError(f"出口场 {key} 含 NaN/Inf")
                out[key] = np.array(a, dtype=np.float64, copy=True)
        out["synthetic"] = bool(exports.get("synthetic", False))
        out["placeholder"] = bool(exports.get("placeholder", False))
        return out

    def _remap_exports(self, exports: Mapping[str, Any]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for key, value in exports.items():
            if isinstance(value, np.ndarray):
                out[key] = self.to_world.remap(value)
        out["t2m_c"] = out["t2m"] - 273.15
        out["precip_mm_yr"] = out["precip"] * SECONDS_PER_YEAR
        return out

    def _forcing_summary(self, forcing: Mapping[str, Any],
                         tick: int) -> dict[str, Any]:
        return {
            "tick": int(tick),
            "time_s": float(forcing["time_s"]),
            "dt_step_s": float(forcing["dt_step_s"]),
            "mask_mean": float(np.mean(forcing["mask"])),
            "sst_mean_k": float(np.mean(forcing["sst"])),
            "h_max_m": float(np.max(forcing["h"])),
            "alb_mean": float(np.mean(forcing["alb"])),
            "orbit": dict(forcing["orbit"]),
        }

    # ---------------- 读数 ----------------

    def world_t2m_c(self) -> np.ndarray:
        if self.last_world_exports is None:
            raise RuntimeError("尚未交换过（无 t2m）")
        return self.last_world_exports["t2m_c"]

    def world_precip_mm_yr(self) -> np.ndarray:
        if self.last_world_exports is None:
            raise RuntimeError("尚未交换过（无 precip）")
        return self.last_world_exports["precip_mm_yr"]

    def manifest(self) -> dict[str, Any]:
        m: dict[str, Any] = {
            "config": self.cfg.manifest(),
            "grid": {
                "atmo": {"kind": self.atmo_grid.kind,
                         "nlat": self.atmo_grid.nlat, "nlon": self.atmo_grid.nlon,
                         "nlev": int(self.cfg.nlev)},
                "world": {"nlat": self.world_grid.nlat, "nlon": self.world_grid.nlon},
            },
            "time": self.clock.as_dict(),
            "orbit": dict(self.orbit),
            "boundary": {
                "h": "env_field.h_m - sea_level_m (ocean=0)",
                "mask": "env_field.land_mask",
                "alb": "SURFACE_THERMOPHYSICS[surface_type][0]",
                "sst": f"light.base_temperature (ocean only; {self.cfg.sst_mode})",
                "seaice": "zero",
            },
            "core": {
                "kind": type(self.core).__name__,
                "synthetic": bool(getattr(self.core, "synthetic", False)),
                "is_stub": bool(getattr(self.core, "is_stub", False)),
            },
            "digest": {
                "atmo_initial": self.initial_digest,
                "atmo_last": (str(self.core.digest(self.state))
                              if self.exchanges else self.initial_digest),
            },
            "exchanges": int(self.exchanges),
        }
        if self.sim_config is not None:
            ts = self.sim_config.time_scale
            m["time_scale"] = {
                "tick_seconds": int(ts.tick_seconds),
                "rotation_period": int(ts.rotation_period),
                "year_ticks": int(ts.year_ticks),
            }
            y = self.orbit["year_s"]
            m["time"]["exchanges_per_year"] = self.clock.exchanges_per_year(y)
            m["time"]["year_s"] = float(y)
        return m

    def probe(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "enabled": True,
            "core": type(self.core).__name__,
            "synthetic": bool(getattr(self.core, "synthetic", False)),
            "k_ticks": self.clock.k,
            "dt_step_s": self.clock.dt_step,
            "exchanges": int(self.exchanges),
            "last_tick": self.last_tick,
            "atmo_digest": str(self.core.digest(self.state)),
        }
        if self.last_world_exports is not None:
            t2m = self.last_world_exports["t2m_c"]
            pr = self.last_world_exports["precip_mm_yr"]
            d.update({
                "world_t2m_c_min": float(t2m.min()),
                "world_t2m_c_mean": float(t2m.mean()),
                "world_t2m_c_max": float(t2m.max()),
                "world_precip_mm_yr_mean": float(pr.mean()),
                "world_precip_mm_yr_max": float(pr.max()),
            })
        if self.last_forcing_summary is not None:
            d["forcing"] = dict(self.last_forcing_summary)
        return d
