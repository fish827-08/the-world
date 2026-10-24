"""A3 大气↔世界耦合层（默认关）。

本包只做**接口层**：网格描述、保守重映射、时钟与边界/出口桥。
不实现大气动力（A1 `atmo_core/`）与物理（A2 `atmo/physics/`）。

关档纪律：`AtmoConfig.enabled=False`（默认）时本包不构造任何对象，
世界 tick 路径零调用、零 import 副作用 ⇒ 世界 digest 逐位不变。
"""
from __future__ import annotations

from .bridge import CouplingBridge
from .clock import CouplingClock
from .config import AtmoConfig
from .grid import Grid, grid_from_atmo_spec, regular_grid, world_grid
from .protocol import AtmoCoreLike, ModuleAtmoCore
from .remap import RemapPair, build_weights, remap_field
from .source import AtmoClimateSource
from .synthetic import AtmoCoreStub, SyntheticCore, synthetic_precip_mm_yr
from .wiring import make_weather_engine, validate_weather_config

__all__ = [
    "AtmoClimateSource",
    "AtmoConfig",
    "AtmoCoreLike",
    "AtmoCoreStub",
    "CouplingBridge",
    "CouplingClock",
    "Grid",
    "ModuleAtmoCore",
    "RemapPair",
    "SyntheticCore",
    "build_weights",
    "grid_from_atmo_spec",
    "make_weather_engine",
    "regular_grid",
    "remap_field",
    "synthetic_precip_mm_yr",
    "validate_weather_config",
    "world_grid",
]
