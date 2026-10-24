"""A3 自用的**合成** atmo 核（显式标记 `synthetic=True`）与 §8 签名桩。

🔴 纪律：合成场只用于验证「时钟 / 边界重映射 / 出口消费」这条**链路**，
**不是**大气模拟结果，不得据它下任何天气结论。所有出口 dict 都带
`synthetic=True` / `placeholder=True`；manifest/probe 也会记录。

真实动力核接入时换 `protocol.ModuleAtmoCore`，本文件不参与生产路径。
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

#: INTERFACE §3 的 8 层 sigma（层中心）
SIGMA_FULL = np.array([0.025, 0.095, 0.200, 0.330, 0.475, 0.625, 0.775, 0.925],
                      dtype=np.float64)
#: 界面表（模式顶 0 / 地表 1）
SIGMA_HALF = np.array([0.0, 0.05, 0.14, 0.26, 0.40, 0.55, 0.70, 0.85, 1.0],
                      dtype=np.float64)
SECONDS_PER_YEAR = 31_557_600.0          # 365.25 d（INTERFACE §7.3）
_DRY_LAPSE_K_PER_M = 0.0098              # INTERFACE §10 t2m v0 外推
_SCALE_HEIGHT_M = 8000.0
_RADIUS_M = 6371000.0


def _gaussian_spec(nlat: int, nlon: int, nlev: int) -> dict[str, Any]:
    nodes, weights = np.polynomial.legendre.leggauss(nlat)   # 升序 −1→+1
    area = weights / 2.0
    lon = np.arange(nlon, dtype=np.float64) * 360.0 / nlon
    return {
        "nlat": nlat, "nlon": nlon, "nlev": nlev, "kind": "gaussian",
        "mu": nodes, "dmu": weights, "area": area,
        "lat_deg": np.degrees(np.arcsin(nodes)),
        "lon_deg": lon,
        "lat_rad": np.arcsin(nodes),
        "lon_rad": np.radians(lon),
        "sigma_half": SIGMA_HALF.copy(),
        "sigma_full": SIGMA_FULL.copy(),
        "radius_m": _RADIUS_M,
    }


def _regular_spec(nlat: int, nlon: int, nlev: int) -> dict[str, Any]:
    edges = np.radians(np.linspace(-90.0, 90.0, nlat + 1))
    area = np.diff(np.sin(edges)) / 2.0
    lat = np.degrees(0.5 * (edges[1:] + edges[:-1]))
    lon = np.arange(nlon, dtype=np.float64) * 360.0 / nlon
    return {
        "nlat": nlat, "nlon": nlon, "nlev": nlev, "kind": "regular",
        "mu": np.sin(np.radians(lat)), "dmu": 2.0 * area, "area": area,
        "lat_deg": lat, "lon_deg": lon,
        "lat_rad": np.radians(lat), "lon_rad": np.radians(lon),
        "sigma_half": SIGMA_HALF.copy(),
        "sigma_full": SIGMA_FULL.copy(),
        "radius_m": _RADIUS_M,
    }


def synthetic_precip_mm_yr(lat_deg: np.ndarray, lon_deg: np.ndarray,
                           seed: int = 0) -> np.ndarray:
    """解析降水空间型（mm/yr）：ITCZ + 中纬风暴带 + 副热带干槽 + 弱波。

    世界格里没有 atmo 核时，`AtmoClimateSource` 用它给 EnvField 一个**临时**降水型
    （manifest 标 `provisional=true`）；bind 后立刻换成核出口重映射的型。
    """
    lat = np.asarray(lat_deg, dtype=np.float64).ravel()
    lon = np.asarray(lon_deg, dtype=np.float64).ravel()
    phi = np.radians(lat)[:, None]
    lam = np.radians(lon)[None, :]
    phase = 0.37 * float(seed)
    itcz = 2200.0 * np.exp(-(phi / math.radians(8.0)) ** 2)
    storm = 800.0 * (
        np.exp(-((phi - math.radians(48.0)) / math.radians(11.0)) ** 2)
        + np.exp(-((phi + math.radians(48.0)) / math.radians(11.0)) ** 2))
    wave = 80.0 * np.cos(2.0 * lam + phase) * np.exp(-(phi / math.radians(30.0)) ** 2)
    return np.maximum(100.0 + itcz + storm + wave, 1.0)


def _fnv1a64(data: bytes) -> str:
    h = 0xCBF29CE484222325
    prime = 0x100000001B3
    mask = 0xFFFFFFFFFFFFFFFF
    for b in data:
        h ^= b
        h = (h * prime) & mask
    return f"{h:016x}"


class SyntheticCore:
    """确定性的解析玩具核：只把 forcing 的作用**显式**传进 T/u，便于测链路。

    - `step()` 把 T 向 `mask·T_land(φ) + (1−mask)·SST` 松弛（τ=5 d），
      把 u 向 `18·sin(3φ)` 松弛（τ=10 d）；无 RNG。
    - `export()` 的 t2m 按 INTERFACE §10 的**占位**口径从最低层 T 外推到 2 m；
      precip 是解析空间型（kg·m⁻²·s⁻¹）。
    """

    synthetic = True
    is_stub = False

    def __init__(self, cfg: Any, seed: int | None = None) -> None:
        self.cfg = cfg
        self.seed = int(cfg.seed if seed is None else seed)
        if str(cfg.kind) == "gaussian":
            self._spec = _gaussian_spec(cfg.nlat, cfg.nlon, cfg.nlev)
        else:
            self._spec = _regular_spec(cfg.nlat, cfg.nlon, cfg.nlev)
        self._lat = np.radians(self._spec["lat_deg"])
        self._lon = self._spec["lon_deg"]
        self._step_count = 0
        self._time_s = 0.0
        # 最低层中心高度（§10 的 z[7]，静力近似；v0 占位）
        self._z_lowest = float(-_SCALE_HEIGHT_M * math.log(float(SIGMA_FULL[-1])))

    def grid_spec(self) -> Mapping[str, Any]:
        return self._spec

    def make_state(self) -> dict[str, np.ndarray]:
        c = self.cfg
        nlev, nlat, nlon = int(c.nlev), int(c.nlat), int(c.nlon)
        T = np.full((nlev, nlat, nlon), float(c.initial_t_k), dtype=np.float64)
        ps = np.full((nlat, nlon), float(c.initial_ps_pa), dtype=np.float64)
        st = {
            "u": np.zeros((nlev, nlat, nlon), dtype=np.float64),
            "v": np.zeros((nlev, nlat, nlon), dtype=np.float64),
            "w": np.zeros((nlev, nlat, nlon), dtype=np.float64),
            "T": T,
            "ps": ps,
            "q": np.zeros((nlev, nlat, nlon), dtype=np.float64),
            "p": ps[None, :, :] * SIGMA_FULL[:, None, None],
        }
        self._step_count = 0
        self._time_s = 0.0
        return st

    def step(self, state: Mapping[str, np.ndarray], forcing: Mapping[str, Any],
             dt: float) -> Mapping[str, Any]:
        dt = float(dt)
        T = state["T"]
        u = state["u"]
        mask = np.asarray(forcing.get("mask", np.zeros_like(T[-1])), dtype=np.float64)
        mask = np.clip(mask, 0.0, 1.0)
        sst = forcing.get("sst")
        lat2 = self._lat[:, None]
        t_land = 289.0 - 38.0 * np.sin(lat2) ** 2
        if sst is None:
            target = np.broadcast_to(t_land, T[-1].shape).copy()
        else:
            sst = np.asarray(sst, dtype=np.float64)
            target = mask * t_land + (1.0 - mask) * sst
        a_t = 1.0 - math.exp(-dt / (5.0 * 86400.0))
        T += a_t * (target[None, :, :] - T)
        jet = 18.0 * np.sin(3.0 * lat2) * np.ones_like(T[-1])
        a_u = 1.0 - math.exp(-dt / (10.0 * 86400.0))
        u += a_u * (jet[None, :, :] - u)
        state["p"] = state["ps"][None, :, :] * SIGMA_FULL[:, None, None]
        self._step_count += 1
        self._time_s += dt
        return self.diag(state, dt=dt)

    def diag(self, state: Mapping[str, np.ndarray],
             dt: float | None = None) -> Mapping[str, Any]:
        T = state["T"]
        ps = state["ps"]
        area = np.asarray(self._spec["area"], dtype=np.float64)
        dx = (2.0 * math.pi * _RADIUS_M * np.cos(self._lat)
              / float(self._spec["nlon"]))
        cfl = 0.0
        if dt is not None and float(dt) > 0.0:
            cfl = float(np.max(np.abs(state["u"])) * float(dt) / dx.min())
        mass = float(np.sum(ps.mean(axis=1) * area) * 4.0 * math.pi * _RADIUS_M ** 2)
        return {
            "cfl_max": cfl, "cfl_lat_deg": float(np.degrees(self._lat[0])),
            "cfl_k": 0, "mass": mass, "mass_rel_err": 0.0,
            "energy": float(T.mean()), "energy_rel_err": 0.0,
            "tmin": float(T.min()), "tmax": float(T.max()),
            "ps_min": float(ps.min()), "ps_max": float(ps.max()),
            "nan_count": int(np.size(T) - np.count_nonzero(np.isfinite(T))),
            "step_index": int(self._step_count), "time_s": float(self._time_s),
            "synthetic": True, "placeholder": True,
        }

    def export(self, state: Mapping[str, np.ndarray]) -> Mapping[str, Any]:
        T_low = np.asarray(state["T"][-1], dtype=np.float64)
        t2m = T_low + _DRY_LAPSE_K_PER_M * (self._z_lowest - 2.0)
        precip_mm_yr = synthetic_precip_mm_yr(
            self._spec["lat_deg"], self._lon, seed=self.seed)
        precip = precip_mm_yr / SECONDS_PER_YEAR
        cloud = np.clip(precip / max(float(precip.max()), 1e-30), 0.0, 1.0)
        u10 = np.asarray(state["u"][-1], dtype=np.float64) * 0.8
        v10 = np.asarray(state["v"][-1], dtype=np.float64) * 0.8
        zeros = np.zeros_like(t2m)
        return {
            "t2m": t2m, "precip": precip, "u10": u10, "v10": v10,
            "cloud": cloud, "sw_sfc": zeros.copy(), "lw_sfc": zeros.copy(),
            "precip_mm_yr": precip_mm_yr,
            "synthetic": True, "placeholder": True,
        }

    def digest(self, state: Mapping[str, np.ndarray]) -> str:
        chunks = [np.ascontiguousarray(state[k], dtype="<f8").tobytes()
                  for k in ("u", "v", "w", "T", "ps", "q")]
        return _fnv1a64(b"".join(chunks))


class AtmoCoreStub:
    """§8 签名桩：所有方法**显式** `NotImplementedError`，绝不返回假数据。

    用途：证明 A3 的接线在 A1 未交付时**失败方式是响亮的**，而不是静默退化。
    """

    synthetic = False
    is_stub = True

    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def _raise(self, name: str) -> None:
        raise NotImplementedError(
            f"atmo_core.{name}() 尚未实现（A1 未交付；§8 签名桩不返回假数据）")

    def grid_spec(self) -> Mapping[str, Any]:
        self._raise("grid_spec")

    def make_state(self) -> dict[str, np.ndarray]:
        self._raise("make_state")

    def step(self, state, forcing, dt: float):
        self._raise("step")

    def export(self, state):
        self._raise("export")

    def digest(self, state) -> str:
        self._raise("digest")
