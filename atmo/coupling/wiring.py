"""天气模式的世界侧守卫与引擎工厂（设计页 §4.2）。

天气模式必须**同时**满足四条硬前提（任一不满足 ⇒ 构造期 fail-loud）：
1. `config.atmosphere.enabled == False` —— 否则 P̄=990 mm/yr 锚与 atmo 降水两套口径并行；
2. `config.env_field.c1_enabled == False` —— 否则 C1 手画降水带乘在 atmo 降水型上；
3. `env_field.enabled == True ∧ c0b_enabled == True` —— 没有 h/land_mask/surface_type 就灌不了边界；
4. `water_sensitivity == cap_sensitivity == 0` + `use_sim_core == False` —— 见下。

第 4 条的理由（不是随手加的）：引擎在 `EnvField` 构造后、`bind()` 之前就读取
`growth_factor()/capacity_factor()`；若灵敏度非零，注入源此时还拿不到 EnvField ⇒
只能返回 None ⇒ **静默丢掉** ENV 产能因子。宁炸不静默（等 IC1 批准后再支持）。
"""
from __future__ import annotations

from typing import Any, Mapping

from .source import AtmoClimateSource


def validate_weather_config(sim_config: Any) -> dict[str, Any]:
    """四条守卫；返回写进 manifest 的 `guards` 子块。"""
    atm = getattr(sim_config, "atmosphere", None)
    ef = getattr(sim_config, "env_field", None)
    sim = getattr(sim_config, "simulation", None)
    if atm is None or ef is None or sim is None:
        raise ValueError("SimConfig 缺 atmosphere/env_field/simulation 子配置")
    if bool(getattr(atm, "enabled", False)):
        raise ValueError(
            "天气模式要求 config.atmosphere.enabled=False："
            "world/atmosphere.py 的 P̄=990 mm/yr 锚（_P_ANCHOR_KG_M2_S）"
            "会与 atmo 降水形成两套口径（设计页 §4.2）。")
    if bool(getattr(ef, "c1_enabled", False)):
        raise ValueError(
            "天气模式要求 env_field.c1_enabled=False：C1 手画降水带（band）"
            "会乘在 atmo 降水型上（设计页 §4.2）。")
    if not bool(getattr(ef, "enabled", False)):
        raise ValueError("天气模式要求 env_field.enabled=True（拿不到 pr/land_mask/h_m）")
    if not bool(getattr(ef, "c0b_enabled", False)):
        raise ValueError(
            "天气模式要求 env_field.c0b_enabled=True（拿不到 h_m/land_mask/surface_type）")
    if float(getattr(ef, "water_sensitivity", 0.0)) != 0.0 or \
            float(getattr(ef, "cap_sensitivity", 0.0)) != 0.0:
        raise NotImplementedError(
            "天气模式要求 env_field.water_sensitivity=0 ∧ cap_sensitivity=0："
            "引擎在 bind 之前读取 growth/capacity 因子，非零灵敏度会被静默丢掉"
            "（资源再生温度换源 = IC1，未批准）")
    if bool(getattr(sim, "use_sim_core", False)):
        raise NotImplementedError(
            "天气模式要求 simulation.use_sim_core=False：Rust regrow 路径尚未接"
            "atmo 温度/降水因子（IC1），开 Rust 会静默 no-op")
    if bool(getattr(getattr(sim_config, "resource_dynamics", None), "enabled", False)):
        raise ValueError(
            "天气模式与 resource_dynamics.enabled=True 未验证"
            "（EnvField 自身也有同一道闸）；请先二选一")
    return {
        "world_atmosphere_enabled": bool(getattr(atm, "enabled", False)),
        "c1_enabled": bool(getattr(ef, "c1_enabled", False)),
        "env_field_enabled": bool(getattr(ef, "enabled", False)),
        "c0b_enabled": bool(getattr(ef, "c0b_enabled", False)),
        "water_sensitivity": float(getattr(ef, "water_sensitivity", 0.0)),
        "cap_sensitivity": float(getattr(ef, "cap_sensitivity", 0.0)),
        "use_sim_core": bool(getattr(sim, "use_sim_core", False)),
        "resource_dynamics_enabled": False,
        "P_ANCHOR_KG_M2_S": "not_active (atmosphere disabled)",
    }


def make_weather_engine(sim_config: Any, atmo_cfg: Any, core: Any) -> tuple[Any, AtmoClimateSource]:
    """构造引擎 + 注入源，并完成 `bind`（首次交换 + 运行期降水型）。

    返回 `(engine, source)`。驱动循环里必须在每次 `engine.step()` 前调用
    `source.on_tick(engine.tick + 1)`（IC3 未批准，引擎不会自动调）。
    """
    guards = validate_weather_config(sim_config)
    tick_s = float(sim_config.time_scale.tick_seconds)
    source = AtmoClimateSource(atmo_cfg, core, tick_s)
    from simulation.sphere_engine import SphereEngine
    source.guards = guards
    engine = SphereEngine(sim_config, climate_source=source)
    source.bind(engine)
    return engine, source
