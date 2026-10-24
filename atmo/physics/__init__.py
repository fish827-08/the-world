"""A2 大气物理包（Python，单柱可独立运行）。

模块顺序与派工件一致：太阳几何 → 短波/长波辐射 → 湿力学 → 云 → 降水
→ 对流 → 地表；``column`` 提供不依赖 A1 动力核心的单柱驱动与账本。
接口冻结后由 A3 负责把层数组映射到 ``(nlev, nlat, nlon)``。
"""
from . import constants, solar
from .clouds import (
    cloud_fraction,
    cloud_fraction_from_state,
    cloud_liquid_water_path,
    cloud_optical_depth,
    evaporate_cloud_water,
    total_cloud_cover,
)
from .column import ColumnModel, ColumnState, StepDiagnostics
from .config import (
    AtmoPhysicsConfig,
    CloudConfig,
    ColumnConfig,
    ConvectionConfig,
    DriverConfig,
    PrecipConfig,
    RadiationConfig,
    SolarConfig,
    SurfaceConfig,
    ThermoConfig,
)
from .driver import AtmoPhysicsDriver, PhysicsAuxState
from .convection import ConvectionResult, convective_adjustment
from .precip import PrecipResult, precip_step
from .radiation import (
    LongwaveFluxes,
    ShortwaveFluxes,
    longwave_fluxes,
    shortwave_fluxes,
)
from .surface import (
    SurfaceFluxes,
    bulk_fluxes,
    solve_land_surface,
    solve_ocean_temperature,
    surface_properties,
    update_soil_temperature,
)
from .thermo import (
    evaporate_liquid,
    latent_heat_vaporization,
    moist_static_energy,
    relative_humidity,
    saturation_adjustment,
    saturation_specific_humidity,
    saturation_vapor_pressure,
)

__all__ = [
    "constants",
    "solar",
    "AtmoPhysicsConfig",
    "SolarConfig",
    "RadiationConfig",
    "ThermoConfig",
    "CloudConfig",
    "PrecipConfig",
    "ConvectionConfig",
    "SurfaceConfig",
    "ColumnConfig",
    "DriverConfig",
    "ColumnModel",
    "ColumnState",
    "StepDiagnostics",
    "AtmoPhysicsDriver",
    "PhysicsAuxState",
    "ShortwaveFluxes",
    "LongwaveFluxes",
    "shortwave_fluxes",
    "longwave_fluxes",
    "bulk_fluxes",
    "solve_ocean_temperature",
    "solve_land_surface",
    "surface_properties",
    "update_soil_temperature",
    "SurfaceFluxes",
    "saturation_vapor_pressure",
    "saturation_specific_humidity",
    "saturation_adjustment",
    "evaporate_liquid",
    "latent_heat_vaporization",
    "relative_humidity",
    "moist_static_energy",
    "cloud_fraction",
    "cloud_fraction_from_state",
    "cloud_liquid_water_path",
    "cloud_optical_depth",
    "evaporate_cloud_water",
    "total_cloud_cover",
    "precip_step",
    "PrecipResult",
    "convective_adjustment",
    "ConvectionResult",
]
