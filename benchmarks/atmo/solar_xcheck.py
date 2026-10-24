"""太阳几何对拍与 sanity 检查（atmo/INTERFACE.md FROZEN v1.0.2 §11 OPEN-2 指定 AV 增加）。

口径
----
- A2 ``atmo/physics/solar.py`` 仅作参考实现（OPEN-2 裁定），绝不作为生产输入；
- 对拍分两层：
  1) **几何公式层**（SOLAR-XCHECK-COSZ / -TOA）：以 A1 自报的 ``subsolar_lat_deg``
     与 ``earth_sun_dist_au`` 为锚，逐格验证 §8.5 公式
     ``cosz = max(0, sinφ sinδ + cosφ cosδ cosH)``、``toa_sw = S0·(a/r)²·cosz``；
     时角由经度与日下点经度推得：``H = 2π·(lon − subsolar_lon)/360``。
     该层剥离历元/相位约定差异，专测几何公式本身。
  2) **年路径层**（SOLAR-XCHECK-DEC）：A1 的 ``subsolar_lat(t)`` 逐采样日与 A2 解析
     赤纬 ``asin(sinε·sinλ)`` 对拍。相位约定按 INTERFACE §9.3：
     ``t = equinox_offset_s`` 为北半球春分；A2 ``day_of_year`` 以 vernal_equinox_day=80
     为锚；两边统一用 orbit 的 ``year_s`` 作年长（SolarConfig.year_days = year_s/86400）。
- sanity 层（SOLAR-COSZ-RANGE / -GLOBAL-MEAN / -OBLIQ0-SYMM / -POLAR-NIGHT）
  只依赖 ``ac.solar`` 本身，不需要 A2。
- A2 的 perihelion_day 相位约定不在对拍面内（INTERFACE §9.3 未为 A1 定义该键，
  距离因子已在几何公式层用 A1 自报 ``earth_sun_dist_au`` 自洽校验）。

用法（atmo_core 交付后）::

    python benchmarks/atmo/solar_xcheck.py          # 用 ac.grid_spec/ac.solar 默认档
    # 或编程调用：run_xcheck(ac.solar, orbit, lat_deg, lon_deg)
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

from . import criteria

DAY_S = 86400.0
A2_VERNAL_EQUINOX_DAY = 80.0  # atmo/physics/config.py SolarConfig 默认锚
DEFAULT_YEAR_S = 365.25 * DAY_S  # INTERFACE §9.3


def _orbit_defaults(orbit: dict | None) -> dict:
    o = {
        "year_s": DEFAULT_YEAR_S,
        "rotation_period_s": 86400.0,
        "obliquity_rad": 0.40910518,
        "eccentricity": 0.0167,
        "equinox_offset_s": 0.0,
        "solar_constant_w_m2": 1361.0,
    }
    o.update(orbit or {})
    return o


def _a2_solar_config(orbit: dict):
    """orbit dict → A2 SolarConfig（year_days 统一取 orbit 的 year_s，见模块 docstring）。"""
    from atmo.physics.config import SolarConfig

    return SolarConfig(
        mode="geometry",
        solar_constant=float(orbit["solar_constant_w_m2"]),
        obliquity_deg=math.degrees(float(orbit["obliquity_rad"])),
        eccentricity=float(orbit["eccentricity"]),
        year_days=float(orbit["year_s"]) / DAY_S,
        rotation_period_s=float(orbit["rotation_period_s"]),
        vernal_equinox_day=A2_VERNAL_EQUINOX_DAY,
    )


def day_to_t(day_of_year: float, orbit: dict) -> float:
    """A2 day_of_year → A1 t_seconds（§9.3 相位约定；两边年长统一为 year_s）。"""
    o = _orbit_defaults(orbit)
    return float(o["equinox_offset_s"]) + (float(day_of_year) - A2_VERNAL_EQUINOX_DAY) * DAY_S


def _sample_days(orbit: dict, n_days: int) -> list[float]:
    o = _orbit_defaults(orbit)
    year_days = float(o["year_s"]) / DAY_S
    return [A2_VERNAL_EQUINOX_DAY + year_days * k / n_days for k in range(n_days)]


def xcheck_geometry(solar_fn, orbit: dict, lat_deg, lon_deg, t_seconds: float) -> dict:
    """单时刻逐格对拍（几何公式层）。返回该时刻的 cosz/toa 最大差。"""
    out = solar_fn(orbit, t_seconds)
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.asarray(lon_deg, dtype=np.float64)
    dec = math.radians(float(out["subsolar_lat_deg"]))
    ssl = float(out["subsolar_lon_deg"]) % 360.0
    dist_au = float(out["earth_sun_dist_au"])
    if dist_au <= 0.0:
        raise ValueError("earth_sun_dist_au 必须 > 0")
    f = 1.0 / (dist_au * dist_au)
    s0 = float(orbit["solar_constant_w_m2"])

    cosz_ref = np.maximum(
        0.0,
        np.sin(lat)[:, None] * math.sin(dec)
        + np.cos(lat)[:, None] * math.cos(dec)
        * np.cos(2.0 * np.pi * (lon - ssl) / 360.0)[None, :],
    )
    cosz_a1 = np.asarray(out["cosz"], dtype=np.float64)
    toa_a1 = np.asarray(out["toa_sw"], dtype=np.float64)
    if cosz_a1.shape != cosz_ref.shape or toa_a1.shape != cosz_ref.shape:
        raise ValueError(
            f"cosz/toa 形状 {cosz_a1.shape}/{toa_a1.shape} 与网格 {(lat.size, lon.size)} 不符"
        )
    d_cosz = float(np.max(np.abs(cosz_a1 - cosz_ref)))
    denom = s0 * f
    d_toa = float(np.max(np.abs(toa_a1 - denom * cosz_ref))) / denom
    return {"cosz_max_abs_diff": d_cosz, "toa_max_rel_diff": d_toa, "cosz_ref": cosz_ref}


def xcheck_declination(solar_fn, orbit: dict, days) -> float:
    """年路径层：A1 subsolar_lat 与 A2 解析赤纬的最大绝对差 [deg]。"""
    from atmo.physics.solar import solar_declination

    cfg = _a2_solar_config(orbit)
    worst = 0.0
    for d in days:
        out = solar_fn(orbit, day_to_t(d, orbit))
        ref = math.degrees(float(solar_declination(float(d), cfg)))
        worst = max(worst, abs(float(out["subsolar_lat_deg"]) - ref))
    return worst


def sanity_cosz_range(outputs) -> int:
    """采样全体 cosz 越界 [0,1]（1e-12 容差）计数。"""
    n = 0
    for out in outputs:
        c = np.asarray(out["cosz"], dtype=np.float64)
        n += int(np.count_nonzero((c < -1e-12) | (c > 1.0 + 1e-12)))
    return n


def sanity_global_mean(solar_fn, orbit: dict, weights, n_days: int = 73) -> float:
    """全年采样瞬时 TOA 全球面积平均相对 |mean − S0/4| / (S0/4)。

    weights: (nlat,) 且 Σ=1（A1 grid_spec 的 area）；缺省用 cos(lat) 归一。
    几何恒等式：任一时刻全球平均 = S0·(a/r)²/4，年均 ≈ S0/4（偏心率修 ~4e-4）。
    """
    o = _orbit_defaults(orbit)
    w = np.asarray(weights, dtype=np.float64)
    s0 = float(o["solar_constant_w_m2"])
    acc = []
    for k in range(n_days):
        t = float(o["equinox_offset_s"]) + float(o["year_s"]) * k / n_days
        out = solar_fn(o, t)
        zonal = np.asarray(out["toa_sw"], dtype=np.float64).mean(axis=-1)
        acc.append(float(zonal @ w))
    mean = float(np.mean(acc))
    return abs(mean - 0.25 * s0) / (0.25 * s0)


def sanity_obliquity0_symmetry(solar_fn, orbit: dict, lat_deg, lon_deg) -> float:
    """obliquity=0 时 cosz 关于赤道对称的最大偏差 max|cosz(φ) − cosz(−φ)|。"""
    o = _orbit_defaults({**orbit, "obliquity_rad": 0.0})
    lat = np.asarray(lat_deg, dtype=np.float64)
    out = solar_fn(o, float(o["equinox_offset_s"]))
    cosz = np.asarray(out["cosz"], dtype=np.float64)
    worst = 0.0
    for x in (5.0, 15.0, 30.0, 45.0, 60.0, 75.0):
        i = int(np.argmin(np.abs(lat - x)))
        j = int(np.argmin(np.abs(lat + x)))
        worst = max(worst, float(np.max(np.abs(cosz[i] - cosz[j]))))
    return worst


def sanity_polar_night(solar_fn, orbit: dict, lat_deg, n_days: int = 73) -> int:
    """obliquity>0 时极圈外（|lat| ≥ 90−ε−2）toa_sw=0 的采样点计数（>0 即存在极夜）。"""
    o = _orbit_defaults(orbit)
    eps_deg = math.degrees(float(o["obliquity_rad"]))
    lat = np.asarray(lat_deg, dtype=np.float64)
    polar = np.abs(lat) >= (90.0 - eps_deg - 2.0)
    n = 0
    for d in _sample_days(o, n_days):
        out = solar_fn(o, day_to_t(d, o))
        toa = np.asarray(out["toa_sw"], dtype=np.float64)
        n += int(np.count_nonzero(polar[:, None] & (toa <= 0.0)))
    return n


def run_xcheck(solar_fn, orbit: dict | None, lat_deg, lon_deg, n_days: int = 73) -> dict:
    """完整对拍：返回 metrics + 判据 status（对齐 criteria.check_value）。

    orbit 为 None 时用 INTERFACE §9.3 默认值。
    """
    o = _orbit_defaults(orbit)
    days = _sample_days(o, n_days)

    d_cosz = d_toa = 0.0
    outputs = []
    for d in days:
        m = xcheck_geometry(solar_fn, o, lat_deg, lon_deg, day_to_t(d, o))
        d_cosz = max(d_cosz, m["cosz_max_abs_diff"])
        d_toa = max(d_toa, m["toa_max_rel_diff"])
        outputs.append(solar_fn(o, day_to_t(d, o)))

    dec_diff = xcheck_declination(solar_fn, o, days)
    violations = sanity_cosz_range(outputs)
    gm_rel = sanity_global_mean(solar_fn, o, weights=_default_weights(lat_deg))
    obliq = float(o["obliquity_rad"])
    # OBLIQ0-SYMM 内部强制 obliquity=0（见 sanity_obliquity0_symmetry），任意配置都可判
    asym = sanity_obliquity0_symmetry(solar_fn, o, lat_deg, lon_deg)
    if obliq > 0.0:
        pn = sanity_polar_night(solar_fn, o, lat_deg, n_days=n_days)
    else:
        pn = None

    metrics = {
        "cosz_max_abs_diff": d_cosz,
        "toa_max_rel_diff": d_toa,
        "dec_max_abs_diff": dec_diff,
        "cosz_range_violations": violations,
        "global_mean_rel_err": gm_rel,
        "obliquity0_max_asym": asym,
        "polar_night_count": pn,
        "n_days": n_days,
    }
    judged = {}
    for cid, key in (
        ("SOLAR-XCHECK-COSZ", "cosz_max_abs_diff"),
        ("SOLAR-XCHECK-TOA", "toa_max_rel_diff"),
        ("SOLAR-XCHECK-DEC", "dec_max_abs_diff"),
        ("SOLAR-COSZ-RANGE", "cosz_range_violations"),
        ("SOLAR-GLOBAL-MEAN", "global_mean_rel_err"),
    ):
        judged[cid] = {"measured": metrics[key], "status": _status(cid, metrics[key])}
    judged["SOLAR-OBLIQ0-SYMM"] = {"measured": asym, "status": _status("SOLAR-OBLIQ0-SYMM", asym)}
    if obliq > 0.0:
        judged["SOLAR-POLAR-NIGHT"] = {"measured": pn, "status": _status("SOLAR-POLAR-NIGHT", pn)}
    else:
        judged["SOLAR-POLAR-NIGHT"] = {"measured": None, "status": "N/A"}
    verdict = "PASS" if all(v["status"] in ("PASS", "N/A") for v in judged.values()) else "FAIL"
    return {"metrics": metrics, "criteria": judged, "verdict": verdict}


def _status(cid: str, x: float) -> str:
    return "PASS" if criteria.check_value(cid, float(x)) else "FAIL"


def _default_weights(lat_deg) -> np.ndarray:
    lat = np.asarray(lat_deg, dtype=np.float64)
    w = np.cos(np.radians(lat))
    return w / w.sum()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="A1 ac.solar ↔ A2 solar.py 对拍（OPEN-2）")
    parser.add_argument("--days", type=int, default=73, help="全年采样天数")
    parser.add_argument("--out", default=None, help="JSON 报告落点（缺省只打印）")
    args = parser.parse_args(argv)

    try:
        import atmo_core as ac
    except ImportError:
        print("atmo_core 尚未交付（A1 线）：对拍需要 ac.solar()/ac.grid_spec()。")
        print("判据与工具已就绪：benchmarks/atmo/solar_xcheck.py + criteria.py SOLAR-*。")
        return 2

    g = ac.grid_spec()
    orbit = {
        "year_s": 365.25 * DAY_S,
        "rotation_period_s": 86400.0,
        "obliquity_rad": 0.40910518,
        "eccentricity": 0.0167,
        "equinox_offset_s": 0.0,
        "solar_constant_w_m2": 1361.0,
    }
    report = run_xcheck(ac.solar, orbit, g["lat_deg"], g["lon_deg"], n_days=args.days)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
