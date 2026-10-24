"""A1 sigma 布局 → 逐柱物理驱动（联调前置件）。

定位
----
`atmo/INTERFACE.md` FROZEN v1.0.2 规定 A1 的状态数组为
``(nlev, nlat, nlon)``、``k=0`` 在模式顶、层压诊断 ``p_k = ps·sigma_full[k]``；
本包的 ``ColumnModel`` 用**固定气压层、索引 0 在地表**。本模块只做这两处的
显式换算，逐柱调用同一套物理，不复制物理公式：

1. **层序**：``T[k=7]``（A1 最低层）↔ 柱索引 ``0``（地表）；
2. **垂直坐标**：柱界面 ``p_edges[0]=ps`` … ``p_edges[-1]=max(0, top_pressure_pa)``；
   层压由 ``sigma_half`` 重算，A1 的 ``p`` 诊断量不参与物理；
3. **辐射上边界**：必须显式传入 A1 的 ``toa_sw`` 与 ``cosz``（INTERFACE §1
   唯一真源），本模块不调用 ``solar.py`` 的全球平均档；
4. **A2 自有状态**：云水 ``qc``、地表温度 ``Ts``、土壤温度 ``soil_T`` 由本驱动
   持有（A1 状态只有 ``q`` 留位；INTERFACE §4/§5），以 :class:`PhysicsAuxState`
   跨步保存。

账本：每柱沿用 ``ColumnModel.step`` 的能量/水量残差；本模块按 A1 网格面积权
``area`` 做全球平均。``surface_mode="prescribed"`` 时海洋是外部热库（大气柱
闭合，海表热容不计入）；``"slab"`` 时海洋混合层计入系统储能。

未做（见汇报风险）
------------------
- 部分陆地格：``land_mask_threshold`` 二选一，不做海陆通量混合；
- 冰相、土壤水：无；
- 平流/动力输送由 A1 负责，本模块不做。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from . import constants as c
from .batch import batch_step_columns
from .column import ColumnModel, ColumnState, StepDiagnostics
from .config import AtmoPhysicsConfig

_REQUIRED_GRID = (
    "nlat", "nlon", "nlev", "sigma_half", "sigma_full", "area",
    "lat_rad", "lon_rad",
)
_REQUIRED_FORCING = ("mask", "sst", "alb")


@dataclass
class PhysicsAuxState:
    """A2 自有、A1 状态里没有的跨步量（均为 A1 布局）。

    ``qc`` ``(nlev,nlat,nlon)`` kg/kg；``Ts`` ``(nlat,nlon)`` K；
    ``soil_T`` ``(soil_layers,nlat,nlon)`` K。
    """

    qc: NDArray[np.float64]
    Ts: NDArray[np.float64]
    soil_T: NDArray[np.float64]

    def copy(self) -> "PhysicsAuxState":
        return PhysicsAuxState(
            qc=self.qc.copy(), Ts=self.Ts.copy(), soil_T=self.soil_T.copy()
        )


def _check_shape(name: str, arr: np.ndarray, shape: tuple[int, ...]) -> None:
    if arr.shape != shape:
        raise ValueError(f"{name} 形状 {arr.shape} ≠ {shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 含 NaN/Inf")


class AtmoPhysicsDriver:
    """在 A1 的 ``(nlev,nlat,nlon)`` sigma 状态上逐柱推进 A2 物理。"""

    synthetic = False

    def __init__(
        self,
        config: AtmoPhysicsConfig | None = None,
        grid: Mapping[str, Any] | None = None,
    ) -> None:
        self.cfg = config or AtmoPhysicsConfig()
        if grid is None:
            raise ValueError("grid 必填（INTERFACE §2.3 的 grid_spec 子集）")
        missing = [k for k in _REQUIRED_GRID if k not in grid]
        if missing:
            raise ValueError(f"grid 缺键 {missing}")
        self.sigma_half = np.asarray(grid["sigma_half"], dtype=np.float64)
        self.sigma_full = np.asarray(grid["sigma_full"], dtype=np.float64)
        self.nlat = int(grid["nlat"])
        self.nlon = int(grid["nlon"])
        self.nlev = int(grid["nlev"])
        self.area = np.asarray(grid["area"], dtype=np.float64)
        self.lat_rad = np.asarray(grid["lat_rad"], dtype=np.float64)
        self.lon_rad = np.asarray(grid["lon_rad"], dtype=np.float64)

        if self.sigma_half.shape != (self.nlev + 1,):
            raise ValueError("sigma_half 长度必须为 nlev+1")
        if self.sigma_full.shape != (self.nlev,):
            raise ValueError("sigma_full 长度必须为 nlev")
        if np.any(np.diff(self.sigma_half) <= 0.0):
            raise ValueError("sigma_half 必须严格递增（模式顶→地表）")
        if not (np.isclose(self.sigma_half[0], 0.0) and np.isclose(self.sigma_half[-1], 1.0)):
            raise ValueError("sigma_half 两端必须是 0（顶）与 1（地表）")
        if self.area.shape != (self.nlat,) or self.lat_rad.shape != (self.nlat,):
            raise ValueError("area/lat_rad 形状必须为 (nlat,)")
        if self.lon_rad.shape != (self.nlon,):
            raise ValueError("lon_rad 形状必须为 (nlon,)")
        if not np.isclose(float(self.area.sum()), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError(f"area 必须归一化到 1，实际 {float(self.area.sum())!r}")

        self.mode = str(self.cfg.driver.surface_mode)
        self._cfg_ocean = replace(
            self.cfg,
            surface=replace(self.cfg.surface, surface_type="ocean"),
            column=replace(self.cfg.column, fixed_sst=self.mode == "prescribed"),
        )
        self._cfg_land = replace(
            self.cfg,
            surface=replace(self.cfg.surface, surface_type="land"),
            column=replace(self.cfg.column, fixed_sst=False),
        )
        # 跨列批量整步；松弛档必须走标量（松弛在对流之后，批量核不含松弛）。
        self.vectorized = bool(
            self.cfg.driver.vectorized
            and self.cfg.column.t_relax_tau_s <= 0.0
            and self.cfg.column.q_relax_tau_s <= 0.0
        )
        self.step_index = 0
        self.time_s = 0.0
        self.last_exports: dict[str, Any] | None = None
        self.last_surface_temperature: NDArray[np.float64] | None = None

    # ---------------- 状态映射 ----------------

    def pressure_edges(self, ps: float) -> NDArray[np.float64]:
        """A1 的 ``ps`` → 地表→顶的柱界面气压（``p_edges[0]=ps``）。"""
        ps = float(ps)
        floor = float(self.cfg.driver.top_pressure_pa)
        if ps * float(self.sigma_half[1]) <= floor:
            raise ValueError(f"ps={ps:.3g} Pa 过低，无法构造 {self.nlev} 层柱")
        p = np.empty(self.nlev + 1, dtype=np.float64)
        p[0] = ps
        for k in range(1, self.nlev):
            p[k] = ps * self.sigma_half[self.nlev - k]
        p[-1] = floor
        if np.any(np.diff(p) >= 0.0):
            raise ValueError("构造出的 p_edges 非严格递减")
        return p

    def column_state(
        self, st: Mapping[str, Any], aux: PhysicsAuxState, j: int, i: int
    ) -> ColumnState:
        """取一格 → 单柱状态（层序翻转在此显式完成）。"""
        T = np.asarray(st["T"], dtype=np.float64)
        q = np.asarray(st["q"], dtype=np.float64)
        ps = np.asarray(st["ps"], dtype=np.float64)
        u = st.get("u")
        v = st.get("v")
        return ColumnState(
            p_edges=self.pressure_edges(float(ps[j, i])),
            T=T[::-1, j, i].copy(),
            qv=q[::-1, j, i].copy(),
            qc=aux.qc[::-1, j, i].copy(),
            Ts=float(aux.Ts[j, i]),
            soil_T=aux.soil_T[:, j, i].copy(),
            u=None if u is None else np.asarray(u)[::-1, j, i].copy(),
            v=None if v is None else np.asarray(v)[::-1, j, i].copy(),
            t=0.0,
        )

    # ---------------- A2 自有状态 ----------------

    def make_state(
        self, st: Mapping[str, Any], forcing: Mapping[str, Any]
    ) -> PhysicsAuxState:
        """由 A1 初态/边界场构造 A2 自有状态。

        海洋：``Ts = sst``；陆地：``Ts`` 取最低层温度按干绝热率外推到层中心，
        土壤各层初值同 ``Ts``。``qc`` 全 0。均为 v0 初始化，不做气候态猜测。
        """
        mask, sst, _ = self._forcing_arrays(forcing)
        T = np.asarray(st["T"], dtype=np.float64)
        _check_shape("T", T, (self.nlev, self.nlat, self.nlon))
        T_low = T[-1]
        sigma_mid_low = 0.5 * (self.sigma_half[-1] + self.sigma_half[-2])
        z_low = -c.RD * T_low / c.GRAVITY * np.log(sigma_mid_low)
        ts_land = T_low + (c.GRAVITY / c.CPD) * z_low
        land = mask >= float(self.cfg.driver.land_mask_threshold)
        Ts = np.where(land, ts_land, sst)
        nsoil = int(self.cfg.surface.soil_layers)
        soil_T = np.where(
            land[None, :, :],
            Ts[None, :, :],
            float(self.cfg.surface.deep_soil_temperature),
        )
        soil_T = np.broadcast_to(soil_T, (nsoil,) + Ts.shape).copy()
        qc = np.zeros((self.nlev, self.nlat, self.nlon), dtype=np.float64)
        return PhysicsAuxState(qc=qc, Ts=Ts, soil_T=soil_T)

    def _forcing_arrays(
        self, forcing: Mapping[str, Any]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        missing = [k for k in _REQUIRED_FORCING if k not in forcing]
        if missing:
            raise ValueError(f"forcing 缺键 {missing}")
        need = (self.nlat, self.nlon)
        mask = np.asarray(forcing["mask"], dtype=np.float64)
        sst = np.asarray(forcing["sst"], dtype=np.float64)
        alb = np.asarray(forcing["alb"], dtype=np.float64)
        _check_shape("mask", mask, need)
        _check_shape("sst", sst, need)
        _check_shape("alb", alb, need)
        if np.any((mask < 0.0) | (mask > 1.0)):
            raise ValueError("mask 必须在 [0, 1]")
        if np.any((alb < 0.0) | (alb >= 1.0)):
            raise ValueError("alb 必须在 [0, 1)")
        return mask, sst, alb

    # ---------------- 推进 ----------------

    def step(
        self,
        st: Mapping[str, Any],
        forcing: Mapping[str, Any],
        aux: PhysicsAuxState,
        dt: float,
        *,
        toa_sw: NDArray[np.float64],
        cosz: NDArray[np.float64],
    ) -> dict[str, Any]:
        """推进一个物理步；原地更新 ``st`` 的 ``T``/``q`` 与 ``aux``。

        ``toa_sw``/``cosz`` 必须是 A1 的出日几何（INTERFACE §1）；本方法不
        自行用全球平均太阳强迫兜底，缺失/形状不符直接报错。
        """
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt 必须为正")
        mask, _, alb = self._forcing_arrays(forcing)
        need = (self.nlat, self.nlon)
        toa_sw = np.asarray(toa_sw, dtype=np.float64)
        cosz = np.asarray(cosz, dtype=np.float64)
        _check_shape("toa_sw", toa_sw, need)
        _check_shape("cosz", cosz, need)
        if np.any((cosz < 0.0) | (cosz > 1.0)):
            raise ValueError("cosz 必须在 [0, 1]")

        T = np.asarray(st["T"], dtype=np.float64)
        q = np.asarray(st["q"], dtype=np.float64)
        ps = np.asarray(st["ps"], dtype=np.float64)
        u_arr = None if st.get("u") is None else np.asarray(st["u"], dtype=np.float64)
        v_arr = None if st.get("v") is None else np.asarray(st["v"], dtype=np.float64)
        _check_shape("T", T, (self.nlev, self.nlat, self.nlon))
        _check_shape("q", q, (self.nlev, self.nlat, self.nlon))
        _check_shape("ps", ps, need)
        if u_arr is not None:
            _check_shape("u", u_arr, (self.nlev, self.nlat, self.nlon))
        if v_arr is not None:
            _check_shape("v", v_arr, (self.nlev, self.nlat, self.nlon))
        if not T.flags.writeable or not q.flags.writeable:
            raise ValueError("st['T']/st['q'] 必须可写（INTERFACE §8.3 原地更新）")
        _check_shape("qc", np.asarray(aux.qc, dtype=np.float64),
                     (self.nlev, self.nlat, self.nlon))
        _check_shape("Ts", np.asarray(aux.Ts, dtype=np.float64), need)
        _check_shape("soil_T", np.asarray(aux.soil_T, dtype=np.float64),
                     (int(self.cfg.surface.soil_layers),) + need)

        T_out = T.copy()
        q_out = q.copy()
        qc_out = aux.qc.copy()
        ts_out = aux.Ts.copy()
        soil_out = aux.soil_T.copy()

        precip = np.zeros(need, dtype=np.float64)
        cloud = np.zeros(need, dtype=np.float64)
        sw_sfc = np.zeros(need, dtype=np.float64)
        lw_sfc = np.zeros(need, dtype=np.float64)
        e_res = np.zeros(need, dtype=np.float64)
        w_res = np.zeros(need, dtype=np.float64)
        toa_net = np.zeros(need, dtype=np.float64)
        nan_mask = np.zeros(need, dtype=bool)
        land = mask >= float(self.cfg.driver.land_mask_threshold)
        if self.vectorized:
            ncol = self.nlat * self.nlon
            p_edges_all = np.empty((ncol, self.nlev + 1), dtype=np.float64)
            for col_index, ps_col in enumerate(ps.ravel()):
                p_edges_all[col_index] = self.pressure_edges(float(ps_col))
            T_cols = np.ascontiguousarray(T_out[::-1]).reshape(self.nlev, ncol).T
            q_cols = np.ascontiguousarray(q_out[::-1]).reshape(self.nlev, ncol).T
            qc_cols = np.ascontiguousarray(qc_out[::-1]).reshape(self.nlev, ncol).T
            wind_low = (
                None if u_arr is None
                else (u_arr[-1].ravel(), v_arr[-1].ravel())
            )
            res = batch_step_columns(
                p_edges_all,
                T_cols,
                q_cols,
                qc_cols,
                ts_out.ravel(),
                soil_out.reshape(int(self.cfg.surface.soil_layers), ncol),
                land.ravel(),
                alb.ravel(),
                toa_sw.ravel(),
                cosz.ravel(),
                wind_low,
                dt,
                self.cfg,
                self.mode == "prescribed",
            )
            T_out[::-1] = res.T.T.reshape(self.nlev, self.nlat, self.nlon)
            q_out[::-1] = res.qv.T.reshape(self.nlev, self.nlat, self.nlon)
            qc_out[::-1] = res.qc.T.reshape(self.nlev, self.nlat, self.nlon)
            ts_out[...] = res.Ts.reshape(self.nlat, self.nlon)
            soil_out[...] = res.soil_T.reshape(
                int(self.cfg.surface.soil_layers), self.nlat, self.nlon
            )
            precip[...] = res.precip_total.reshape(self.nlat, self.nlon) / dt
            cloud[...] = res.cloud_cover.reshape(self.nlat, self.nlon)
            sw_sfc[...] = res.sw_surface_net.reshape(self.nlat, self.nlon)
            lw_sfc[...] = res.lw_surface_net_down.reshape(self.nlat, self.nlon)
            e_res[...] = res.energy_residual.reshape(self.nlat, self.nlon)
            w_res[...] = res.water_residual.reshape(self.nlat, self.nlon)
            toa_net[...] = res.toa_net.reshape(self.nlat, self.nlon)
            nan_mask[...] = (
                ~res.nan_free
                | ~np.isfinite(res.precip_total)
                | ~np.isfinite(res.cloud_cover)
            ).reshape(self.nlat, self.nlon)
        else:
            for j in range(self.nlat):
                for i in range(self.nlon):
                    col = ColumnState(
                        p_edges=self.pressure_edges(float(ps[j, i])),
                        T=T_out[::-1, j, i].copy(),
                        qv=q_out[::-1, j, i].copy(),
                        qc=qc_out[::-1, j, i].copy(),
                        Ts=float(ts_out[j, i]),
                        soil_T=soil_out[:, j, i].copy(),
                        u=None if u_arr is None else u_arr[::-1, j, i].copy(),
                        v=None if v_arr is None else v_arr[::-1, j, i].copy(),
                        t=0.0,
                    )
                    cfg_col = self._cfg_land if land[j, i] else self._cfg_ocean
                    model = ColumnModel(cfg_col, col)
                    diag: StepDiagnostics = model.step(
                        dt,
                        toa_flux=float(toa_sw[j, i]),
                        mu0=float(cosz[j, i]),
                        surface_albedo=float(alb[j, i]),
                    )
                    new = model.state
                    T_out[::-1, j, i] = new.T
                    q_out[::-1, j, i] = new.qv
                    qc_out[::-1, j, i] = new.qc
                    ts_out[j, i] = new.Ts
                    soil_out[:, j, i] = new.soil_T

                    precip[j, i] = diag.precip_total / dt
                    cloud[j, i] = diag.cloud_cover
                    sw_sfc[j, i] = diag.sw_surface_net
                    lw_sfc[j, i] = diag.lw_surface_net_down
                    e_res[j, i] = diag.energy_residual
                    w_res[j, i] = diag.water_residual
                    toa_net[j, i] = diag.toa_net
                    nan_mask[j, i] = not (
                        diag.nan_free
                        and np.isfinite(diag.precip_total)
                        and np.isfinite(diag.cloud_cover)
                    )

        st["T"][...] = T_out
        st["q"][...] = q_out
        aux.qc[...] = qc_out
        aux.Ts[...] = ts_out
        aux.soil_T[...] = soil_out
        self.step_index += 1
        self.time_s += dt
        self.last_surface_temperature = ts_out.copy()
        self.last_exports = {
            "precip": precip,
            "cloud": cloud,
            "sw_sfc": sw_sfc,
            "lw_sfc": lw_sfc,
        }

        # area 只含纬向权重（Σ=1）；经向等权 ⇒ 全球平均还要除以 nlon。
        w = self.area[:, None] / float(self.nlon)
        return {
            "step_index": int(self.step_index),
            "time_s": float(self.time_s),
            "dt_s": dt,
            "surface_mode": self.mode,
            "energy_residual": float(np.sum(e_res * w)),        # J/m² / step
            "water_residual": float(np.sum(w_res * w)),         # kg/m² / step
            "max_abs_column_energy_residual": float(np.max(np.abs(e_res))),
            "max_abs_column_water_residual": float(np.max(np.abs(w_res))),
            "precip_mean": float(np.sum(precip * w)),           # kg/m²/s
            "cloud_mean": float(np.sum(cloud * w)),
            "toa_net_mean": float(np.sum(toa_net * w)),         # W/m²
            "sw_sfc_mean": float(np.sum(sw_sfc * w)),
            "lw_sfc_mean": float(np.sum(lw_sfc * w)),
            "nan_count": int(np.count_nonzero(nan_mask)),
            "land_fraction": float(np.sum(mask * w)),
        }

    def export(self) -> dict[str, Any]:
        """A2 出口场（INTERFACE §10 的 A2 部分）；须在 ``step`` 之后调用。"""
        if self.last_exports is None:
            raise RuntimeError("AtmoPhysicsDriver.export 之前必须先 step")
        out: dict[str, Any] = {
            k: np.array(v, copy=True) for k, v in self.last_exports.items()
        }
        out["synthetic"] = False
        out["placeholder"] = False
        return out


__all__ = ["AtmoPhysicsDriver", "PhysicsAuxState"]
