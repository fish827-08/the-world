"""单柱 stub：状态、逐步物理编排、能量/水量账本、RCE 运行器。

用途
----
在 A1 动力核心/``atmo/INTERFACE.md`` 冻结前，用一维列模式验证 A2 物理包
（辐射、湿过程、云、降水、对流、地表）单独可跑、守恒、无 NaN。
接口冻结后由 A3 把本模块的层数组映射到 ``(nlev, nlat, nlon)``。

列模式假设
----------
- 无水平/垂直动力输运（w=0）；风只通过 ``SurfaceConfig.wind_speed`` 或
  传入的 u/v 进入地表整体公式；
- 垂直网格固定（p 坐标），层厚由 ``ColumnConfig.pressure_edges_pa`` 给出；
- 水汽/云水为层平均比湿；雨以稳态通量处理（见 ``precip.py``）。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import constants as c
from . import solar
from .clouds import (
    cloud_fraction_from_state,
    cloud_liquid_water_path,
    evaporate_cloud_water,
    total_cloud_cover,
)
from .config import AtmoPhysicsConfig, ThermoConfig
from .convection import convective_adjustment
from .precip import precip_step
from .radiation import longwave_fluxes, shortwave_fluxes
from .surface import (
    bulk_fluxes,
    solve_land_surface,
    solve_ocean_temperature,
    surface_properties,
)
from .thermo import (
    moist_static_energy,
    saturation_adjustment,
    saturation_specific_humidity,
)


@dataclass
class ColumnState:
    p_edges: NDArray[np.float64]
    T: NDArray[np.float64]
    qv: NDArray[np.float64]
    qc: NDArray[np.float64]
    Ts: float
    soil_T: NDArray[np.float64] = field(default_factory=lambda: np.zeros(2))
    u: NDArray[np.float64] | None = None
    v: NDArray[np.float64] | None = None
    t: float = 0.0

    @property
    def nlev(self) -> int:
        return int(self.T.size)

    @property
    def dp(self) -> NDArray[np.float64]:
        return self.p_edges[:-1] - self.p_edges[1:]

    @property
    def p_mid(self) -> NDArray[np.float64]:
        return 0.5 * (self.p_edges[:-1] + self.p_edges[1:])

    @property
    def mass(self) -> NDArray[np.float64]:
        return self.dp / c.GRAVITY

    @property
    def rho(self) -> NDArray[np.float64]:
        return self.p_mid / (c.RD * np.maximum(self.T, 150.0))

    def heights(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """静力高度：返回 (界面 z, 层中心 z)，界面 0 = 地表（z=0）。

        数组顺序为**地表→顶层**（``p_edges[0]=p_s`` 最大、``p_edges[-1]=p_top``
        最小），与 ``pressure_edges_pa`` / ``initial_temperature_k`` 一致。
        """
        z_edges = np.zeros(self.nlev + 1, dtype=np.float64)
        for i in range(self.nlev):
            z_edges[i + 1] = z_edges[i] + (
                c.RD * self.T[i] / c.GRAVITY
                * np.log(self.p_edges[i] / self.p_edges[i + 1])
            )
        z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])
        return z_edges, z_mid

    def cloud_fraction(self, cfg: AtmoPhysicsConfig) -> NDArray[np.float64]:
        return cloud_fraction_from_state(self.qv, self.T, self.p_mid, cfg.cloud, cfg.thermo)

    def column_water(self) -> float:
        """柱内水汽+云水 [kg/m²]。"""
        return float(np.sum((self.qv + self.qc) * self.mass))

    def atmosphere_energy(self, cfg: AtmoPhysicsConfig) -> float:
        # 固定气压网格下柱内（内能+重力势能）= Σ c_p T m：∫g z dm = ∫R T dm
        # 已由静力平衡吸收，若再叠加 MSE 的 g z 项会重复计能（见报告守恒节）。
        h = moist_static_energy(
            self.T, np.zeros(self.nlev), self.qv, self.qc, cfg.thermo
        )
        return float(np.sum(h * self.mass))

    def surface_energy(self, cfg: AtmoPhysicsConfig) -> float:
        if cfg.surface.surface_type == "ocean":
            return c.RHO_WATER * c.CW * cfg.surface.ocean_mixed_layer_depth * (
                self.Ts - c.T0
            )
        n = cfg.surface.soil_layers
        depth = np.asarray(cfg.surface.soil_depth_m[:n], dtype=np.float64)
        cap = np.asarray(cfg.surface.soil_volumetric_heat_capacity[:n], dtype=np.float64)
        return float(np.sum(cap * depth * (self.soil_T[:n] - c.T0)))

    def total_energy(self, cfg: AtmoPhysicsConfig) -> float:
        return self.atmosphere_energy(cfg) + self.surface_energy(cfg)

    def copy(self) -> "ColumnState":
        return ColumnState(
            p_edges=self.p_edges.copy(),
            T=self.T.copy(),
            qv=self.qv.copy(),
            qc=self.qc.copy(),
            Ts=float(self.Ts),
            soil_T=self.soil_T.copy(),
            u=None if self.u is None else self.u.copy(),
            v=None if self.v is None else self.v.copy(),
            t=float(self.t),
        )


@dataclass
class StepDiagnostics:
    sw_toa_in: float
    sw_toa_out: float
    sw_surface_net: float
    sw_atm_absorption: float
    olr: float
    lw_surface_net_down: float
    lw_atm_absorption: float
    sensible: float
    latent: float
    evaporation: float  # kg m-2 s-1
    precip_large_scale: float  # kg m-2 / step
    precip_convective: float  # kg m-2 / step
    rain_evaporation: float  # kg m-2 / step
    autoconversion: float
    accretion: float
    cloud_cover: float
    lwp: float  # kg/m²
    precipitable_water: float  # kg/m²
    toa_net: float  # W/m²
    surface_temperature: float
    energy_residual: float  # J/m² / step
    water_residual: float  # kg/m² / step
    deep_flux: float  # W/m²，陆地深层边界流出（海洋恒为 0）
    t_relax_energy: float  # J/m² / step，T 剖面松弛注入的能量
    q_relax_mass: float  # kg/m² / step，水汽松弛注入的质量
    min_temperature: float
    max_temperature: float
    nan_free: bool

    @property
    def precip_total(self) -> float:
        return self.precip_large_scale + self.precip_convective

    def as_dict(self) -> dict[str, Any]:
        return {
            "sw_toa_in": self.sw_toa_in,
            "sw_toa_out": self.sw_toa_out,
            "sw_surface_net": self.sw_surface_net,
            "sw_atm_absorption": self.sw_atm_absorption,
            "olr": self.olr,
            "lw_surface_net_down": self.lw_surface_net_down,
            "lw_atm_absorption": self.lw_atm_absorption,
            "sensible": self.sensible,
            "latent": self.latent,
            "evaporation": self.evaporation,
            "precip_large_scale": self.precip_large_scale,
            "precip_convective": self.precip_convective,
            "precip_total": self.precip_total,
            "rain_evaporation": self.rain_evaporation,
            "autoconversion": self.autoconversion,
            "accretion": self.accretion,
            "cloud_cover": self.cloud_cover,
            "lwp": self.lwp,
            "precipitable_water": self.precipitable_water,
            "toa_net": self.toa_net,
            "surface_temperature": self.surface_temperature,
            "energy_residual": self.energy_residual,
            "water_residual": self.water_residual,
            "deep_flux": self.deep_flux,
            "t_relax_energy": self.t_relax_energy,
            "q_relax_mass": self.q_relax_mass,
            "min_temperature": self.min_temperature,
            "max_temperature": self.max_temperature,
            "nan_free": self.nan_free,
        }


def _initial_state(cfg: AtmoPhysicsConfig) -> ColumnState:
    col = cfg.column
    p_edges = np.asarray(col.pressure_edges_pa, dtype=np.float64)
    T = np.asarray(col.initial_temperature_k, dtype=np.float64).copy()
    p_mid = 0.5 * (p_edges[:-1] + p_edges[1:])
    qv = col.initial_relative_humidity * saturation_specific_humidity(T, p_mid, cfg.thermo)
    qc = np.full_like(T, col.initial_cloud_water)
    n_soil = cfg.surface.soil_layers
    soil_T = np.full(n_soil, col.initial_soil_temperature_k, dtype=np.float64)
    return ColumnState(
        p_edges=p_edges,
        T=T,
        qv=qv,
        qc=qc,
        Ts=float(col.initial_surface_temperature_k),
        soil_T=soil_T,
        u=np.zeros_like(T),
        v=np.zeros_like(T),
        t=0.0,
    )


class ColumnModel:
    """单柱物理模型；不依赖 A1 动力核心。"""

    def __init__(self, config: AtmoPhysicsConfig | None = None, state: ColumnState | None = None):
        self.cfg = config or AtmoPhysicsConfig()
        self.state = state or _initial_state(self.cfg)

    def step(
        self,
        dt: float | None = None,
        *,
        toa_flux: float | None = None,
        mu0: float | None = None,
        surface_albedo: float | None = None,
    ) -> StepDiagnostics:
        """推进一个物理步。

        ``toa_flux``/``mu0`` 省略时用 ``solar.column_toa_forcing``（单柱默认档）；
        A1 布局驱动会显式传入 A1 的 ``toa_sw`` 与 ``cosz``（INTERFACE §1 唯一真源）。
        ``surface_albedo`` 覆盖地表配置反照率（A1 forcing 的 ``alb``）。
        """
        cfg = self.cfg
        s = self.state
        dt = cfg.column.dt_s if dt is None else float(dt)
        if dt <= 0.0:
            raise ValueError("dt 必须为正")
        if (toa_flux is None) != (mu0 is None):
            raise ValueError("toa_flux 与 mu0 必须同时提供")

        e0 = s.total_energy(cfg)
        w0 = s.column_water()
        p_edges = s.p_edges
        p_mid = s.p_mid
        mass = s.mass
        z_edges, z_mid = s.heights()

        if toa_flux is None:
            toa_flux, mu0 = solar.column_toa_forcing(cfg.solar)
        albedo, emissivity, _ = surface_properties(cfg.surface)
        if surface_albedo is not None:
            albedo = float(surface_albedo)

        sw = shortwave_fluxes(
            p_edges, s.T, s.qv, s.qc, albedo, toa_flux, mu0,
            cfg.radiation, cfg.cloud,
        )
        # 先解地表温度，再用新 Ts 算长波：避免"大气按旧 Ts 冷却、地表按新 Ts
        # 发射"的一步滞后把 O(ΔTs) 的劈裂误差记进能量账。
        lw_surface = longwave_fluxes(
            p_edges, s.T, s.qv, s.qc, s.Ts, emissivity,
            cfg.radiation, cfg.cloud,
        )

        qv = s.qv.copy()
        qc = s.qc.copy()

        # ---- 地表：温度隐式更新 + 通量 ----
        surface_wind = cfg.surface.wind_speed
        if s.u is not None and s.v is not None:
            local = float(np.hypot(s.u[0], s.v[0]))
            if local > 0.0:
                surface_wind = local
        p_low = float(p_mid[0])
        if cfg.surface.surface_type == "ocean":
            if cfg.column.fixed_sst:
                h_flux, le_flux, evap = bulk_fluxes(
                    s.Ts, float(s.T[0]), float(qv[0]), p_low, surface_wind, 1.0,
                    cfg.surface, cfg.thermo,
                )
                ts_new = float(s.Ts)
                soil_new = s.soil_T.copy()
            else:
                sf = solve_ocean_temperature(
                    s.Ts, sw.surface_net, float(lw_surface.flux_down[0]),
                    float(s.T[0]), float(qv[0]), p_low, dt,
                    cfg.surface, cfg.thermo, wind_speed=surface_wind,
                )
                h_flux, le_flux, evap = sf.sensible, sf.latent, sf.evaporation
                ts_new = sf.temperature
                soil_new = s.soil_T.copy()
        else:
            sf, soil_new = solve_land_surface(
                s.soil_T, sw.surface_net, float(lw_surface.flux_down[0]),
                float(s.T[0]), float(qv[0]), p_low, dt,
                cfg.surface, cfg.thermo, wind_speed=surface_wind,
            )
            h_flux, le_flux, evap = sf.sensible, sf.latent, sf.evaporation
            ts_new = sf.temperature

        lw = longwave_fluxes(
            p_edges, s.T, s.qv, s.qc, ts_new, emissivity,
            cfg.radiation, cfg.cloud,
        )
        T = s.T + (sw.heating_rate + lw.heating_rate) * dt

        # 蒸发不能把最低层水汽抽成负；把通量限制在可用水以内。
        mass_low = mass[0]
        evap = max(evap, -float(qv[0]) * mass_low / dt)
        T[0] += h_flux * dt / (c.CPD * mass_low)
        qv[0] = max(qv[0] + evap * dt / mass_low, 0.0)

        # ---- 相变：大尺度饱和调整 ----
        for i in range(T.size):
            T[i], qv[i], qc[i], _ = saturation_adjustment(
                float(T[i]), float(qv[i]), float(qc[i]), float(p_mid[i]), cfg.thermo
            )

        # ---- 云水蒸发（未饱和层）----
        for i in range(T.size):
            T[i], qv[i], qc[i], _ = evaporate_cloud_water(
                float(T[i]), float(qv[i]), float(qc[i]), float(p_mid[i]), dt,
                cfg.cloud, cfg.thermo,
            )

        # ---- 降水（云水→雨通量，落出+再蒸发）----
        cf = cloud_fraction_from_state(qv, T, p_mid, cfg.cloud, cfg.thermo)
        pr = precip_step(p_edges, T, qv, qc, cf, dt, cfg.precip, cfg.thermo)
        T, qv, qc = pr.T, pr.qv, pr.qc

        # ---- 对流调整 ----
        conv = convective_adjustment(p_edges, T, qv, qc, z_mid, cfg.convection, cfg.thermo)
        T, qv, qc = conv.T, conv.qv, conv.qc

        # ---- 可选固定 T 剖面松弛（外部强迫；默认关）----
        relax_energy = 0.0
        if cfg.column.t_relax_tau_s > 0.0 and cfg.column.t_ref_profile_k:
            t_ref = np.asarray(cfg.column.t_ref_profile_k, dtype=np.float64)
            delta = (t_ref - T) * min(dt / cfg.column.t_relax_tau_s, 1.0)
            T = T + delta
            relax_energy = float(np.sum(c.CPD * mass * delta))

        # ---- 可选水汽松弛（大尺度水汽强迫；默认关）----
        # 目标水汽固定在参考温度上（不用当前 T）：用当前 T 会形成
        # "增湿→凝结→升温→q_ref↑→再增湿"的正反馈，数天内发散。
        q_relax_mass = 0.0
        q_relax_energy = 0.0
        if cfg.column.q_relax_tau_s > 0.0 and cfg.column.q_relax_relative_humidity:
            rh_ref = np.asarray(cfg.column.q_relax_relative_humidity, dtype=np.float64)
            t_ref_q = cfg.column.q_relax_reference_temperature_k or cfg.column.initial_temperature_k
            t_ref_q = np.asarray(t_ref_q, dtype=np.float64)
            q_ref = rh_ref * saturation_specific_humidity(t_ref_q, p_mid, cfg.thermo)
            dq = (q_ref - qv) * min(dt / cfg.column.q_relax_tau_s, 1.0)
            qv_old = qv.copy()
            qv = qv + dq
            q_relax_mass = float(np.sum(dq * mass))
            h_old = moist_static_energy(T, np.zeros_like(T), qv_old, qc, cfg.thermo)
            h_new = moist_static_energy(T, np.zeros_like(T), qv, qc, cfg.thermo)
            q_relax_energy = float(np.sum((h_new - h_old) * mass))

        # ---- 写回状态 ----
        new_state = ColumnState(
            p_edges=p_edges,
            T=T,
            qv=np.maximum(qv, 0.0),
            qc=np.maximum(qc, 0.0),
            Ts=ts_new,
            soil_T=soil_new,
            u=s.u,
            v=s.v,
            t=s.t + dt,
        )
        self.state = new_state

        e1 = new_state.total_energy(cfg)
        w1 = new_state.column_water()

        toa_net = sw.toa_in - sw.toa_out - lw.olr
        fixed_sst = cfg.column.fixed_sst and cfg.surface.surface_type == "ocean"
        if fixed_sst:
            # 海洋是外部热库：系统 = 大气柱，强迫 = 大气辐射吸收 + 地表湍流。
            forcing_energy = (
                sw.atmosphere_absorption
                + lw.atmosphere_absorption
                + h_flux
                + le_flux
            ) * dt
        else:
            forcing_energy = toa_net * dt
        # 陆地深层边界热通量是柱外交换
        deep_flux = 0.0
        if cfg.surface.surface_type == "land":
            n = cfg.surface.soil_layers
            d = cfg.surface.soil_depth_m[n - 1]
            k_deep = cfg.surface.soil_conductivity[n - 1] / (0.5 * d)
            deep_flux = float(
                k_deep * (float(soil_new[n - 1]) - cfg.surface.deep_soil_temperature)
            )
            forcing_energy -= deep_flux * dt
        energy_residual = (
            (e1 - e0) - forcing_energy - relax_energy - q_relax_energy
            + conv.precip_energy
        )

        evap_mass = evap * dt
        water_residual = (
            (w1 - w0) - (evap_mass - pr.surface_precip - conv.precip) - q_relax_mass
        )
        nan_free = bool(
            np.all(np.isfinite(new_state.T))
            and np.all(np.isfinite(new_state.qv))
            and np.all(np.isfinite(new_state.qc))
            and np.isfinite(new_state.Ts)
            and np.isfinite(ts_new)
        )
        cf_out = cloud_fraction_from_state(new_state.qv, new_state.T, p_mid, cfg.cloud, cfg.thermo)
        lwp = float(np.sum(cloud_liquid_water_path(new_state.qc, new_state.dp)))
        pw = float(np.sum(new_state.qv * new_state.mass))

        return StepDiagnostics(
            sw_toa_in=sw.toa_in,
            sw_toa_out=sw.toa_out,
            sw_surface_net=sw.surface_net,
            sw_atm_absorption=sw.atmosphere_absorption,
            olr=lw.olr,
            lw_surface_net_down=lw.surface_net_down,
            lw_atm_absorption=lw.atmosphere_absorption,
            sensible=h_flux,
            latent=le_flux,
            evaporation=float(evap),
            precip_large_scale=pr.surface_precip,
            precip_convective=conv.precip,
            rain_evaporation=pr.rain_evaporation,
            autoconversion=pr.autoconversion,
            accretion=pr.accretion,
            cloud_cover=total_cloud_cover(cf_out),
            lwp=lwp,
            precipitable_water=pw,
            toa_net=float(toa_net),
            surface_temperature=float(ts_new),
            energy_residual=float(energy_residual),
            water_residual=float(water_residual),
            deep_flux=deep_flux,
            t_relax_energy=float(relax_energy),
            q_relax_mass=float(q_relax_mass),
            min_temperature=float(np.min(new_state.T)),
            max_temperature=float(np.max(new_state.T)),
            nan_free=nan_free,
        )

    def run(
        self,
        days: float,
        spinup_days: float = 0.0,
        dt: float | None = None,
        collect_profiles: bool = True,
    ) -> dict[str, Any]:
        """积分若干天；返回采样期均值、守恒残差与末态剖面。"""
        cfg = self.cfg
        dt = cfg.column.dt_s if dt is None else float(dt)
        steps_per_day = c.DAY_SECONDS / dt
        spin_steps = int(round(spinup_days * steps_per_day))
        sample_steps = max(int(round(days * steps_per_day)), 1)

        for _ in range(spin_steps):
            self.step(dt)

        energy_start = float(self.state.total_energy(cfg))
        water_start = self.state.column_water()
        ts_start = float(self.state.Ts)
        t_start = float(self.state.t)

        samples: dict[str, list[float]] = {}
        max_energy_res = 0.0
        max_water_res = 0.0
        nan_count = 0
        for _ in range(sample_steps):
            d = self.step(dt)
            row = d.as_dict()
            for key, value in row.items():
                if isinstance(value, (int, float, bool, np.floating, np.integer)):
                    samples.setdefault(key, []).append(float(value))
            max_energy_res = max(max_energy_res, abs(d.energy_residual))
            max_water_res = max(max_water_res, abs(d.water_residual))
            nan_count += 0 if d.nan_free else 1

        elapsed = float(self.state.t - t_start)
        energy_end = float(self.state.total_energy(cfg))
        water_end = self.state.column_water()
        energy_tendency = (energy_end - energy_start) / max(elapsed, 1.0)
        water_tendency = (water_end - water_start) / max(elapsed, 1.0)
        dts_dt = (float(self.state.Ts) - ts_start) / max(elapsed, 1.0)
        atmos_forcing = float(
            np.mean(
                np.asarray(samples.get("sw_atm_absorption", [0.0]))
                + np.asarray(samples.get("lw_atm_absorption", [0.0]))
                + np.asarray(samples.get("sensible", [0.0]))
                + np.asarray(samples.get("latent", [0.0]))
            )
        )
        toa_mean = float(np.mean(np.asarray(samples.get("toa_net", [0.0]))))
        deep_flux_mean = float(np.mean(np.asarray(samples.get("deep_flux", [0.0]))))
        # 闭合 = 外部辐射强迫 − 系统储能变化率（海洋/土壤储热均计入）。
        system_closure = toa_mean - energy_tendency
        atmosphere_closure = atmos_forcing - energy_tendency
        surface_closure = system_closure - deep_flux_mean

        means = {key: float(np.mean(np.asarray(vals))) for key, vals in samples.items() if key != "nan_free"}
        means["nan_free_fraction"] = float(np.mean(np.asarray(samples.get("nan_free", [0.0]))))
        summary: dict[str, Any] = {
            "dt_s": dt,
            "spinup_days": float(spinup_days),
            "sample_days": float(days),
            "sample_steps": sample_steps,
            "means": means,
            "max_abs_energy_residual": float(max_energy_res),
            "max_abs_water_residual": float(max_water_res),
            "nan_steps": int(nan_count),
            "toa_net_mean": toa_mean,
            "atmosphere_forcing_mean": atmos_forcing,
            "energy_tendency_wm2": float(energy_tendency),
            "water_tendency_kgm2s": float(water_tendency),
            "deep_flux_mean": deep_flux_mean,
            "system_closure_wm2": float(system_closure),
            "atmosphere_closure_wm2": float(atmosphere_closure),
            "surface_closure_wm2": float(surface_closure),
            "surface_temperature_start": ts_start,
            "surface_temperature_final": float(self.state.Ts),
            "dts_dt_k_per_day": float(dts_dt * c.DAY_SECONDS),
            "precip_mm_day": 86400.0 * means.get("precip_total", 0.0) / dt,
            "evaporation_mm_day": 86400.0 * means.get("evaporation", 0.0),
            "column_water_final": float(self.state.column_water()),
            "total_energy_final": float(self.state.total_energy(cfg)),
        }
        if collect_profiles:
            summary["profiles"] = {
                "p_mid_hpa": (self.state.p_mid / 100.0).tolist(),
                "T_k": self.state.T.tolist(),
                "qv": self.state.qv.tolist(),
                "qc": self.state.qc.tolist(),
                "rh": (
                    self.state.qv
                    / np.maximum(
                        saturation_specific_humidity(self.state.T, self.state.p_mid, cfg.thermo),
                        1.0e-12,
                    )
                ).tolist(),
                "cloud_fraction": self.state.cloud_fraction(cfg).tolist(),
            }
        return summary
