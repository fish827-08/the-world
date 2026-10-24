"""大气物理包的全部可调参数（dataclass 配置树）。

纪律（派工件 §二 / 写死账本）：
- 模块代码不得出现裸数值；所有可调参数从这里读取；
- 默认值必须能通过 ``to_dict()`` 原样进入 manifest；
- ``from_dict`` 对缺失字段回退默认值（旧存档兼容）。

参数来源逐项见各字段注释；参数表与账本登记单见交付汇报。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, get_type_hints


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """按 dataclass 字段类型递归构造；缺失键回退默认值。"""
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = hints.get(f.name, f.type)
        if is_dataclass(ftype) and isinstance(value, dict):
            value = _from_dict(ftype, value)
        kwargs[f.name] = value
    return cls(**kwargs)


class _ConfigMixin:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        return _from_dict(cls, data or {})


# ---------------------------------------------------------------- 太阳几何
@dataclass
class SolarConfig(_ConfigMixin):
    """太阳几何与大气顶短波强迫。

    mode:
      - ``global_mean``：柱模式用全球年平均 TOA 辐照度 S0/4（默认，RCE 标准）；
      - ``fixed``：固定太阳天顶角，TOA 水平通量 = ``fixed_toa_flux``；
      - ``geometry``：由纬度/日序/地方太阳时算太阳高度角（昼夜+季节）。
    """

    mode: str = "global_mean"
    solar_constant: float = 1361.0  # W/m²，TSIS-1/WMO
    obliquity_deg: float = 23.4392911  # J2000 黄赤交角
    eccentricity: float = 0.0167086  # J2000 地球轨道偏心率
    perihelion_day: float = 3.0  # 近日点日序（1 月 3 日）
    vernal_equinox_day: float = 80.0  # 春分日序（3 月 21 日）
    year_days: float = 365.2422  # 回归年
    rotation_period_s: float = 86400.0  # 自转周期
    global_mean_cos_zenith: float = 0.5  # 全球平均柱的有效 μ0（几何日平均近似）
    fixed_cos_zenith: float = 0.5
    fixed_toa_flux: float = 340.25  # W/m²，S0/4
    latitude_deg: float = 0.0
    day_of_year: float = 80.0
    local_solar_time_h: float = 12.0

    def __post_init__(self) -> None:
        if self.mode not in {"global_mean", "fixed", "geometry"}:
            raise ValueError(f"未知 solar.mode: {self.mode}")
        if self.solar_constant <= 0.0:
            raise ValueError("solar_constant 必须为正")
        if not (0.0 <= self.obliquity_deg <= 90.0):
            raise ValueError("obliquity_deg 必须在 [0, 90]")
        if not (0.0 <= self.eccentricity < 1.0):
            raise ValueError("eccentricity 必须在 [0, 1)")
        if self.rotation_period_s <= 0.0 or self.year_days <= 0.0:
            raise ValueError("自转/公转周期必须为正")
        if not (0.0 < self.global_mean_cos_zenith <= 1.0):
            raise ValueError("global_mean_cos_zenith 必须在 (0, 1]")
        if not (0.0 < self.fixed_cos_zenith <= 1.0):
            raise ValueError("fixed_cos_zenith 必须在 (0, 1]")


# ---------------------------------------------------------------- 辐射
@dataclass
class RadiationConfig(_ConfigMixin):
    """短波双流 + 长波灰体。

    短波：delta-Eddington 标度后的双流（Eddington 闭包，含不对称因子）；
    长波：逐层灰体发射/吸收（无散射），``lw_mode="linear"`` 提供
    OLR = A + B (Ts − Tref) 的诊断档作对照。
    """

    # ---- 短波 ----
    sw_vapor_abs_coeff: float = 0.003  # m²/kg，水汽宽带吸收截面（按晴空大气吸收≈54 W/m² 标定）
    sw_rayleigh_tau_surface: float = 0.10  # 整层 Rayleigh 光学厚度（宽带）
    cloud_sw_single_scatter_albedo: float = 0.999  # 云滴短波单散射反照率
    cloud_sw_asymmetry: float = 0.85  # 云滴短波不对称因子
    # ---- 长波 ----
    lw_mode: str = "grey"  # "grey" | "linear"
    lw_vapor_abs_coeff: float = 0.160  # m²/kg，水汽宽带长波吸收截面（按 Ts=288 K 晴空 OLR≈270 W/m² 标定）
    lw_cloud_abs_coeff: float = 100.0  # m²/kg，云水长波质量消光系数
    lw_linear_a: float = 210.0  # W/m²，OLR = a + b (Ts − Tref)
    lw_linear_b: float = 2.2  # W/m²/K
    lw_linear_t_ref: float = 273.15  # K
    lw_linear_down_fraction: float = 0.75  # 大气向下长波 / OLR（线性档闭合假设）

    def __post_init__(self) -> None:
        if self.lw_mode not in {"grey", "linear"}:
            raise ValueError(f"未知 lw_mode: {self.lw_mode}")
        for name in (
            "sw_vapor_abs_coeff",
            "sw_rayleigh_tau_surface",
            "lw_vapor_abs_coeff",
            "lw_cloud_abs_coeff",
            "lw_linear_b",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} 必须非负")
        if not (0.0 <= self.cloud_sw_single_scatter_albedo <= 1.0):
            raise ValueError("云单散射反照率必须在 [0, 1]")
        if not (-1.0 <= self.cloud_sw_asymmetry < 1.0):
            raise ValueError("云不对称因子必须在 [-1, 1)")
        if not (0.0 <= self.lw_linear_down_fraction <= 1.0):
            raise ValueError("lw_linear_down_fraction 必须在 [0, 1]")


# ---------------------------------------------------------------- 湿热力学
@dataclass
class ThermoConfig(_ConfigMixin):
    """Clausius–Clapeyron、饱和调整、潜热。"""

    saturation_formula: str = "cc"  # "cc"（CC 积分）| "buck"（Buck 1981）
    latent_heat_mode: str = "constant"  # "constant" | "temperature_dependent"
    phase_change_max_iter: int = 50
    phase_change_tol_k: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.saturation_formula not in {"cc", "buck"}:
            raise ValueError(f"未知 saturation_formula: {self.saturation_formula}")
        if self.latent_heat_mode not in {"constant", "temperature_dependent"}:
            raise ValueError(f"未知 latent_heat_mode: {self.latent_heat_mode}")
        if self.phase_change_max_iter < 5:
            raise ValueError("phase_change_max_iter 至少 5")
        if self.phase_change_tol_k <= 0.0:
            raise ValueError("phase_change_tol_k 必须为正")


# ---------------------------------------------------------------- 云
@dataclass
class CloudConfig(_ConfigMixin):
    """诊断云量、云水、云光学厚度。"""

    rh_crit: float = 0.70  # 云量起报相对湿度（Slingo 1987 类阈值）
    rh_exponent: float = 2.0  # 云量对 RH 的超线性指数
    cloud_fraction_floor: float = 0.0
    cloud_fraction_ceiling: float = 1.0
    effective_radius_um: float = 10.0  # 云滴有效半径（Slingo 1989 典型值）
    optical_depth_mode: str = "slingo"  # "slingo" | "linear"
    linear_optical_coeff: float = 100.0  # m²/kg，线性档 τ = k·LWP
    cloud_evap_coeff: float = 1.0e-4  # s-1，未饱和云水蒸发弛豫系数

    def __post_init__(self) -> None:
        if not (0.0 <= self.rh_crit < 1.0):
            raise ValueError("rh_crit 必须在 [0, 1)")
        if self.rh_exponent <= 0.0:
            raise ValueError("rh_exponent 必须为正")
        if not (0.0 <= self.cloud_fraction_floor <= self.cloud_fraction_ceiling <= 1.0):
            raise ValueError("云量上下限必须满足 0 ≤ floor ≤ ceiling ≤ 1")
        if self.effective_radius_um <= 0.0:
            raise ValueError("effective_radius_um 必须为正")
        if self.optical_depth_mode not in {"slingo", "linear"}:
            raise ValueError(f"未知 optical_depth_mode: {self.optical_depth_mode}")
        if self.linear_optical_coeff < 0.0 or self.cloud_evap_coeff < 0.0:
            raise ValueError("云光学/蒸发系数必须非负")


# ---------------------------------------------------------------- 降水
@dataclass
class PrecipConfig(_ConfigMixin):
    """Kessler 型云水自动转化 + 碰并 + 雨通量落出/再蒸发。"""

    enabled: bool = True
    qc_crit: float = 0.5e-3  # kg/kg，自动转化阈值（Kessler 1969 量级）
    autoconversion_rate: float = 1.0e-3  # s-1，超过阈值后的转化速率
    accretion_coeff: float = 2.2  # s-1，碰并系数（Kessler 1969）
    rain_terminal_velocity: float = 5.0  # m/s，雨滴质量加权落速
    rain_evap_coeff: float = 1.0e-3  # s-1，雨滴再蒸发系数
    cloud_fraction_weighted: bool = True  # 自动转化/碰并按云内云水计算
    cloud_fraction_min: float = 0.01  # 云内换算的云量下限

    def __post_init__(self) -> None:
        if self.qc_crit < 0.0:
            raise ValueError("qc_crit 必须非负")
        for name in (
            "autoconversion_rate",
            "accretion_coeff",
            "rain_terminal_velocity",
            "rain_evap_coeff",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} 必须非负")
        if not (0.0 < self.cloud_fraction_min <= 1.0):
            raise ValueError("cloud_fraction_min 必须在 (0, 1]")


# ---------------------------------------------------------------- 对流
@dataclass
class ConvectionConfig(_ConfigMixin):
    """湿对流调整（松弛到湿绝热/MSE 中性）。"""

    enabled: bool = True
    scheme: str = "moist_adjust"  # 目前仅 "moist_adjust"
    max_iter: int = 64
    mse_tol: float = 1.0  # J/kg，判定不稳定的 MSE 容差
    qc_hold: float = 0.5e-3  # kg/kg，调整后保留为云水、超出部分降为对流降水

    def __post_init__(self) -> None:
        if self.scheme != "moist_adjust":
            raise ValueError(f"未知 convection.scheme: {self.scheme}")
        if self.max_iter < 1:
            raise ValueError("max_iter 至少 1")
        if self.mse_tol < 0.0:
            raise ValueError("mse_tol 必须非负")
        if self.qc_hold < 0.0:
            raise ValueError("qc_hold 必须非负")


# ---------------------------------------------------------------- 地表
@dataclass
class SurfaceConfig(_ConfigMixin):
    """整体公式地表通量、海洋混合层、陆地土壤热传导。"""

    surface_type: str = "ocean"  # "ocean" | "land"
    wind_speed: float = 5.0  # m/s，单柱 stub 的近地面风速（无动力时可配）
    drag_coeff_heat: float = 1.2e-3  # 感热整体交换系数
    drag_coeff_moisture: float = 1.2e-3  # 潜热整体交换系数
    albedo_ocean: float = 0.06  # 海表宽带反照率
    albedo_land: float = 0.20  # 陆表宽带反照率
    emissivity_ocean: float = 0.98  # 海表长波发射率
    emissivity_land: float = 0.95  # 陆表长波发射率
    ocean_mixed_layer_depth: float = 50.0  # m，slab 海洋混合层深度
    land_evap_beta: float = 0.3  # 土壤湿度蒸发抑制因子
    soil_layers: int = 2  # 1 或 2
    soil_depth_m: tuple[float, ...] = (0.1, 1.0)  # 各层厚度
    soil_volumetric_heat_capacity: tuple[float, ...] = (2.0e6, 2.0e6)  # J m-3 K-1
    soil_conductivity: tuple[float, ...] = (1.0, 1.0)  # W m-1 K-1
    deep_soil_temperature: float = 288.15  # K，深层边界温度

    def __post_init__(self) -> None:
        if self.surface_type not in {"ocean", "land"}:
            raise ValueError(f"未知 surface_type: {self.surface_type}")
        if self.wind_speed < 0.0:
            raise ValueError("wind_speed 必须非负")
        if not (0.0 <= self.albedo_ocean < 1.0) or not (0.0 <= self.albedo_land < 1.0):
            raise ValueError("反照率必须在 [0, 1)")
        if not (0.0 < self.emissivity_ocean <= 1.0) or not (0.0 < self.emissivity_land <= 1.0):
            raise ValueError("发射率必须在 (0, 1]")
        if self.ocean_mixed_layer_depth <= 0.0:
            raise ValueError("海洋混合层深度必须为正")
        if not (0.0 <= self.land_evap_beta <= 1.0):
            raise ValueError("land_evap_beta 必须在 [0, 1]")
        if self.soil_layers not in {1, 2}:
            raise ValueError("soil_layers 只支持 1 或 2")
        if len(self.soil_depth_m) < self.soil_layers:
            raise ValueError("soil_depth_m 长度不足")
        if len(self.soil_volumetric_heat_capacity) < self.soil_layers:
            raise ValueError("soil_volumetric_heat_capacity 长度不足")
        if len(self.soil_conductivity) < self.soil_layers:
            raise ValueError("soil_conductivity 长度不足")
        if any(d <= 0.0 for d in self.soil_depth_m[: self.soil_layers]):
            raise ValueError("土壤层厚度必须为正")
        if any(c <= 0.0 for c in self.soil_volumetric_heat_capacity[: self.soil_layers]):
            raise ValueError("土壤体积热容必须为正")
        if any(k <= 0.0 for k in self.soil_conductivity[: self.soil_layers]):
            raise ValueError("土壤导热率必须为正")


# ---------------------------------------------------------------- 单柱
@dataclass
class ColumnConfig(_ConfigMixin):
    """单柱 stub 的垂直网格、初始场与时间步（接口冻结后由 A1 网格替换）。

    数组顺序统一为**地表→顶层**：``pressure_edges_pa[0]`` 为最大气压（地表），
    ``initial_temperature_k[0]`` 为最低层；辐射/降水/对流/地表均按此约定。
    """

    pressure_edges_pa: tuple[float, ...] = (
        100000.0,
        90000.0,
        80000.0,
        70000.0,
        55000.0,
        40000.0,
        25000.0,
        10000.0,
        1000.0,
    )
    initial_temperature_k: tuple[float, ...] = (
        286.0,
        281.5,
        277.0,
        271.0,
        261.0,
        247.0,
        226.0,
        205.0,
    )
    initial_relative_humidity: float = 0.70
    initial_cloud_water: float = 0.0  # kg/kg
    initial_surface_temperature_k: float = 288.15
    initial_soil_temperature_k: float = 288.15
    dt_s: float = 900.0
    fixed_sst: bool = False  # True：海表温度固定（大气 RCE 的固定 SST 档）
    t_relax_tau_s: float = 0.0  # >0 时把 T 拉向 t_ref_profile_k（固定剖面松弛档）
    t_ref_profile_k: tuple[float, ...] = ()
    # 单柱 RCE 的可选大尺度水汽强迫（默认关；用于敏感性实验）：
    # 把 qv 拉向 q_relax_relative_humidity · qsat(T_ref)。目标温度必须固定，
    # 用当前 T 会形成“增湿→凝结→升温→目标再升”的正反馈（实测数天内发散）。
    q_relax_tau_s: float = 0.0
    q_relax_relative_humidity: tuple[float, ...] = ()
    q_relax_reference_temperature_k: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        p = self.pressure_edges_pa
        if len(p) < 3:
            raise ValueError("至少需要 2 层（3 个气压界面）")
        if any(p[i] <= p[i + 1] for i in range(len(p) - 1)):
            raise ValueError("pressure_edges_pa 必须自上而下递减")
        if len(self.initial_temperature_k) != len(p) - 1:
            raise ValueError("initial_temperature_k 长度必须等于层数")
        if len(self.t_ref_profile_k) not in {0, len(p) - 1}:
            raise ValueError("t_ref_profile_k 长度必须为 0 或等于层数")
        if len(self.q_relax_relative_humidity) not in {0, len(p) - 1}:
            raise ValueError("q_relax_relative_humidity 长度必须为 0 或等于层数")
        if len(self.q_relax_reference_temperature_k) not in {0, len(p) - 1}:
            raise ValueError("q_relax_reference_temperature_k 长度必须为 0 或等于层数")
        if any(not (0.0 <= rh <= 1.0) for rh in self.q_relax_relative_humidity):
            raise ValueError("q_relax_relative_humidity 必须在 [0, 1]")
        if self.dt_s <= 0.0:
            raise ValueError("dt_s 必须为正")
        if not (0.0 <= self.initial_relative_humidity <= 1.0):
            raise ValueError("initial_relative_humidity 必须在 [0, 1]")
        if self.t_relax_tau_s < 0.0:
            raise ValueError("t_relax_tau_s 必须非负")
        if self.q_relax_tau_s < 0.0:
            raise ValueError("q_relax_tau_s 必须非负")


# ---------------------------------------------------------------- A1 布局驱动
@dataclass
class DriverConfig(_ConfigMixin):
    """A1 sigma 网格 → 逐柱物理的映射参数（联调前置件，不影响单柱默认档）。

    ``surface_mode``：
      - ``prescribed``：海洋柱的 ``Ts`` 固定为 forcing ``sst``（外部热库，默认）；
      - ``slab``：海洋柱用本包混合层热容演化，``sst`` 只作初值。
    """

    surface_mode: str = "prescribed"
    # A1 的 sigma_half[0] = 0（模式顶）⇒ 物理层需要正的顶界面气压；
    # 1 Pa 远小于最上层厚度（0.05·ps），对质量的影响 ~1e-5。
    top_pressure_pa: float = 1.0
    # 陆地分数 ≥ 阈值按陆面算，否则按海面；v0 不做部分陆地混合。
    land_mask_threshold: float = 0.5
    # 跨列批量整步（逐列掩码，与标量 ColumnModel 路径逐位一致）；开启
    # T 剖面/水汽松弛时自动回退标量以保持"对流→松弛"的编排顺序。
    vectorized: bool = True

    def __post_init__(self) -> None:
        if self.surface_mode not in {"prescribed", "slab"}:
            raise ValueError(f"未知 driver.surface_mode: {self.surface_mode}")
        if self.top_pressure_pa <= 0.0:
            raise ValueError("driver.top_pressure_pa 必须为正")
        if not (0.0 <= self.land_mask_threshold <= 1.0):
            raise ValueError("driver.land_mask_threshold 必须在 [0, 1]")
        if not isinstance(self.vectorized, bool):
            raise ValueError("driver.vectorized 必须为 bool")


# ---------------------------------------------------------------- 顶层
@dataclass
class AtmoPhysicsConfig(_ConfigMixin):
    """大气物理包总配置（默认全部启用，但只影响 atmo 自身；世界侧开关由 A3 管）。"""

    solar: SolarConfig = field(default_factory=SolarConfig)
    radiation: RadiationConfig = field(default_factory=RadiationConfig)
    thermo: ThermoConfig = field(default_factory=ThermoConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    precip: PrecipConfig = field(default_factory=PrecipConfig)
    convection: ConvectionConfig = field(default_factory=ConvectionConfig)
    surface: SurfaceConfig = field(default_factory=SurfaceConfig)
    column: ColumnConfig = field(default_factory=ColumnConfig)
    driver: DriverConfig = field(default_factory=DriverConfig)

    def __post_init__(self) -> None:
        # 嵌套配置已各自校验；这里只做跨组一致性检查。
        if self.column.dt_s <= 0.0:
            raise ValueError("dt_s 必须为正")
