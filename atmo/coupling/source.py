"""`AtmoClimateSource`：把 A3 桥接到 `ClimateSource` 协议（A16b 注入通路）。

世界消费点（设计页 §4）：
* `temperature_c(flat, tick)` → **零阶保持**的 atmo `t2m`（°C），覆盖生物层与动态 NPP；
* `precip_mm_yr()` → 构造期降水换源（A18.3）；运行期 `set_precip_pattern`（A17.3）；
* `growth_factor` / `capacity_factor` → bind 后委托 `EnvField`（bind 前为 None）。

🔴 资源再生温度**不走本接口**（IC1 未批准）——汇报必须如实写。
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .bridge import CouplingBridge
from .synthetic import synthetic_precip_mm_yr

class AtmoClimateSource:
    """满足 `world.climate.ClimateSource` 的注入源；`name="atmo"`。"""

    name = "atmo"

    def __init__(self, atmo_cfg: Any, core: Any, tick_seconds: float | None = None,
                 *, seed: int | None = None) -> None:
        if not bool(getattr(atmo_cfg, "enabled", False)):
            raise ValueError("AtmoClimateSource 只在 AtmoConfig.enabled=True 时构造")
        if str(getattr(atmo_cfg, "weather_source", "off")) != "atmo":
            raise ValueError("weather_source 必须是 'atmo'")
        self.cfg = atmo_cfg
        self.core = core
        self._tick_seconds = tick_seconds
        self._seed = int(atmo_cfg.seed if seed is None else seed)
        self._world = None
        self._engine = None
        self._env = None
        self._bridge: CouplingBridge | None = None
        self._prov_flat: np.ndarray | None = None
        self._t2m_flat: np.ndarray | None = None
        self._temp_calls = 0
        self._precip_refresh_count = 0
        self._last_precip_diag: dict[str, Any] | None = None
        self.guards: dict[str, Any] = {}

    # ---------------- 构造期（引擎调用）----------------

    def bootstrap(self, world: Any) -> None:
        """引擎在构造 `EnvField` **之前**调用：只依赖 world 几何。"""
        self._world = world
        rows, cols = int(world.rows), int(world.cols)
        lat = np.degrees(np.asarray(
            world.latitude_of(np.arange(rows, dtype=np.int64)), dtype=np.float64))
        lon = 360.0 * np.arange(cols, dtype=np.float64) / cols
        if bool(getattr(self.core, "synthetic", False)):
            # 仅合成核：给 EnvField 一个**临时**空间型，manifest 标 provisional；
            # bind() 后的第一次交换会立刻换成核出口重映射的型。
            self._prov_flat = synthetic_precip_mm_yr(
                lat, lon, seed=self._seed).ravel()
        else:
            # 真实核：不伪造降水（A2 v0 precip=0 ⇒ 构造期不换源，运行期再看）。
            self._prov_flat = None

    def precip_mm_yr(self) -> np.ndarray | None:
        """降水场（mm/yr 数值、世界格序）；None = 不换源。

        引擎只在构造期调用一次（bind 前 ⇒ 临时合成型）；bind 之后返回核出口的
        当前空间型（零阶保持），便于探针/演示核对。
        """
        if self._bridge is not None:
            return self._bridge.world_precip_mm_yr().ravel().copy()
        return None if self._prov_flat is None else self._prov_flat.copy()

    def bind(self, engine: Any) -> None:
        """引擎构造完成后调用：建桥 → 首次交换 → 运行期换降水型。"""
        if self._world is None:
            raise RuntimeError("AtmoClimateSource.bind 之前必须先 bootstrap(world)")
        if engine is None or getattr(engine, "env_field", None) is None:
            raise ValueError("AtmoClimateSource.bind 需要带 EnvField 的引擎")
        if getattr(engine.world, "rows", None) != self._world.rows:
            raise ValueError("bind 的 world 与 bootstrap 的不是同一个")
        ef = engine.env_field
        if float(getattr(ef.cfg, "water_sensitivity", 0.0)) != 0.0 or \
                float(getattr(ef.cfg, "cap_sensitivity", 0.0)) != 0.0:
            raise NotImplementedError(
                "天气模式要求 env_field.water_sensitivity=0 ∧ cap_sensitivity=0："
                "引擎在 bind 之前就读取了 growth/capacity 因子，非零灵敏度会被静默丢掉。"
                "（资源再生温度换源 = IC1，未批准。）")
        self._engine = engine
        self._env = ef
        if self._tick_seconds is None:
            self._tick_seconds = float(engine.config.time_scale.tick_seconds)
        self._bridge = CouplingBridge.from_engine(engine, self.cfg, self.core)
        self._bridge.exchange(0)
        self._refresh_from_bridge()

    # ---------------- 运行期 ----------------

    def on_tick(self, tick: int) -> bool:
        """驱动入口：交换相位推进 atmo 并刷新世界侧缓存（IC3 未批准，引擎不自动调）。"""
        if self._bridge is None:
            raise RuntimeError("AtmoClimateSource 未 bind(engine)：拒绝静默返回旧值")
        did = self._bridge.on_tick(tick)
        if did:
            self._refresh_from_bridge()
        return did

    def _refresh_from_bridge(self) -> None:
        assert self._bridge is not None
        t2m = self._bridge.world_t2m_c().ravel()
        if not np.all(np.isfinite(t2m)):
            raise ValueError("atmo t2m 含 NaN/Inf —— 拒绝灌进世界温度链")
        self._t2m_flat = np.array(t2m, dtype=np.float64, copy=True)
        if str(self.cfg.precip_refresh) == "every_exchange":
            pr = self._bridge.world_precip_mm_yr().ravel()
            if float(pr.max()) > 0.0:
                diag = self._env.set_precip_pattern(
                    pr, source="atmo", keep_total=True)
                self._last_precip_diag = dict(diag)
                self._precip_refresh_count += 1

    # ---------------- ClimateSource 协议 ----------------

    def refresh(self, tick: int) -> None:
        """显式钩子（IC3 未批准 ⇒ 引擎不调用；保留给实验驱动）。"""
        self.on_tick(tick)

    def temperature_c(self, flat: Any, tick: int) -> Any:
        if self._t2m_flat is None:
            raise RuntimeError(
                "AtmoClimateSource.temperature_c 在 bind() 之前被调用 —— "
                "拒绝返回假温度（先构造引擎并 bind）")
        self._temp_calls += 1
        f = np.asarray(flat, dtype=np.int64)
        out = self._t2m_flat[f]
        return out.item(0) if f.ndim == 0 else out

    def growth_factor(self) -> Any:
        if self._env is None:
            return None
        return self._env.growth_factor()

    def capacity_factor(self) -> Any:
        if self._env is None:
            return None
        return self._env.capacity_factor()

    def wind(self) -> Any:
        raise NotImplementedError(
            "A3 v0 不消费 atmo 风场（设计页 §4：世界当前无消费点）；"
            "需要风平流时请走变更单，不在耦合层静默返回 None。")

    def probe(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "weather_source": str(self.cfg.weather_source),
            "synthetic": bool(getattr(self.core, "synthetic", False)),
            "bound": self._bridge is not None,
            "provisional_precip": self._prov_flat is not None,
            "t2m_calls": int(self._temp_calls),
            "precip_refresh_count": int(self._precip_refresh_count),
            "precip_diag": self._last_precip_diag,
            "snapshot_state": "not_serialized_v0",
        }
        if self._bridge is not None:
            d["bridge"] = self._bridge.probe()
            d["pr_source"] = getattr(self._env, "pr_source", None)
        return d

    def manifest(self, guards: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self._bridge is None:
            raise RuntimeError("manifest 需要先 bind(engine)")
        m = self._bridge.manifest()
        m["exports"] = {
            "t2m_consumer": "climate.temperature_c",
            "precip_policy": "keep_total",
            "precip_refresh": str(self.cfg.precip_refresh),
            "weather_precip_active": bool(
                self._bridge.exchanges
                and float(self._bridge.world_precip_mm_yr().max()) > 0.0),
            "wind_consumed": False,
            "cloud_consumed": False,
            "sw_lw_consumed": False,
            "resource_regrowth_temperature": "not_wired (IC1)",
        }
        m["guards"] = dict(guards if guards is not None else self.guards)
        return m

    def state(self) -> dict[str, Any]:
        """快照 state：v0 不序列化 atmo 状态（汇报如实写，不假装可续跑）。"""
        return {}

    def restore(self, state: Mapping[str, Any]) -> None:
        if state:
            raise NotImplementedError(
                "A3 v0 未实现 atmo 耦合态的快照恢复（不静默丢弃非空 state）")
