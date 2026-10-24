# TIER = "fast"
"""SOLAR-* 判据与太阳几何对拍工具的单测（INTERFACE §11 OPEN-2 / 附加 2）。

用合成 "A1"（直接按 §8.5 公式实现 ac.solar 的黑盒替身）验证对拍 mechanics：
正确实现必须全 PASS；分别破坏 δ / cosz / toa 的三种实现必须被对应判据抓住。
阈值本身在 test_atmo_criteria.py 的钉住表冻结。
"""
import math

import numpy as np
import pytest

from benchmarks.atmo import criteria, solar_xcheck as sx

LATS = np.linspace(-85.0, 85.0, 33)
LONS = np.linspace(0.0, 360.0, 64, endpoint=False)

ORBIT = {
    "year_s": 365.25 * 86400.0,
    "rotation_period_s": 86400.0,
    "obliquity_rad": 0.40910518,
    "eccentricity": 0.0167,
    "equinox_offset_s": 0.0,
    "solar_constant_w_m2": 1361.0,
}


def fake_ac_solar(orbit, t_seconds, dec_bias_deg=0.0, cosz_scale=1.0, toa_bias=0.0):
    """按 §8.5 公式实现的 ac.solar 替身（黑盒：只暴露 §8.5 的输出键）。"""
    year_s = float(orbit["year_s"])
    eq = float(orbit.get("equinox_offset_s", 0.0))
    eps = float(orbit["obliquity_rad"])
    e = float(orbit.get("eccentricity", 0.0167))
    s0 = float(orbit["solar_constant_w_m2"])
    rot = float(orbit["rotation_period_s"])
    lam = 2.0 * math.pi * (t_seconds - eq) / year_s
    dec = math.asin(math.sin(eps) * math.sin(lam))
    f = ((1.0 + e * math.cos(lam)) / (1.0 - e * e)) ** 2  # 近点角=黄经（测试自洽即可）
    ssl = ((t_seconds - eq) / rot * 360.0) % 360.0
    phi = np.radians(LATS)[:, None]
    hh = 2.0 * np.pi * (LONS - ssl) / 360.0
    cosz = np.maximum(
        0.0,
        np.sin(phi) * math.sin(dec) + np.cos(phi) * math.cos(dec) * np.cos(hh)[None, :],
    )
    dec_report = dec + math.radians(dec_bias_deg)
    cosz = cosz * cosz_scale
    toa = s0 * f * cosz * (1.0 + toa_bias)
    return {
        "cosz": cosz,
        "toa_sw": toa,
        "subsolar_lat_deg": math.degrees(dec_report),
        "subsolar_lon_deg": ssl,
        "earth_sun_dist_au": 1.0 / math.sqrt(f),
    }


# ---- 判据注册表 ----

def test_solar_criteria_registered():
    for cid in (
        "SOLAR-XCHECK-COSZ", "SOLAR-XCHECK-TOA", "SOLAR-XCHECK-DEC",
        "SOLAR-COSZ-RANGE", "SOLAR-GLOBAL-MEAN", "SOLAR-OBLIQ0-SYMM",
        "SOLAR-POLAR-NIGHT", "RCE-COL-ENERGY", "RCE-COL-EQUIL",
    ):
        assert cid in criteria.CRITERIA
        assert "INTERFACE" in criteria.CRITERIA[cid].citation


def test_rce_column_criteria_frozen():
    c1 = criteria.CRITERIA["RCE-COL-ENERGY"]
    c2 = criteria.CRITERIA["RCE-COL-EQUIL"]
    assert (c1.limit_lo, c1.limit_hi) == (None, 5.0)
    assert (c2.limit_lo, c2.limit_hi) == (None, 0.05)
    # 附加 2 口径必须写明：单柱不要求 TOA≈0
    assert "不要求 TOA≈0" in c2.desc or "不要求 TOA≈0" in c1.desc


def test_day_to_t_phase_mapping():
    # §9.3：t = equinox_offset_s ≡ 春分 = A2 day 80
    assert sx.day_to_t(80.0, {}) == 0.0
    assert sx.day_to_t(80.0, {"equinox_offset_s": 1000.0}) == 1000.0
    assert sx.day_to_t(81.0, {}) == 86400.0


# ---- 正确实现必须全 PASS ----

def test_correct_impl_passes_all_solar_criteria():
    report = sx.run_xcheck(fake_ac_solar, ORBIT, LATS, LONS, n_days=25)
    assert report["verdict"] == "PASS", report["criteria"]
    m = report["metrics"]
    assert m["cosz_max_abs_diff"] <= 1e-6
    assert m["toa_max_rel_diff"] <= 1e-6
    assert m["dec_max_abs_diff"] <= 1e-6
    assert m["cosz_range_violations"] == 0
    assert m["global_mean_rel_err"] <= 0.02
    assert report["criteria"]["SOLAR-POLAR-NIGHT"]["measured"] > 0


def test_obliquity0_path_na_polar_night():
    orbit = {**ORBIT, "obliquity_rad": 0.0}
    report = sx.run_xcheck(fake_ac_solar, orbit, LATS, LONS, n_days=5)
    assert report["criteria"]["SOLAR-POLAR-NIGHT"]["status"] == "N/A"
    # obliquity=0 时对称性仍判，且正确实现应 PASS
    assert report["criteria"]["SOLAR-OBLIQ0-SYMM"]["status"] == "PASS"


# ---- 破坏实现必须被抓住 ----

def test_declination_bias_detected():
    report = sx.run_xcheck(
        lambda o, t: fake_ac_solar(o, t, dec_bias_deg=0.01), ORBIT, LATS, LONS, n_days=5)
    assert report["criteria"]["SOLAR-XCHECK-COSZ"]["status"] == "FAIL"
    assert report["criteria"]["SOLAR-XCHECK-DEC"]["status"] == "FAIL"


def test_cosz_scale_detected():
    report = sx.run_xcheck(
        lambda o, t: fake_ac_solar(o, t, cosz_scale=1.0 + 1e-4), ORBIT, LATS, LONS, n_days=5)
    assert report["criteria"]["SOLAR-XCHECK-COSZ"]["status"] == "FAIL"


def test_toa_bias_detected():
    report = sx.run_xcheck(
        lambda o, t: fake_ac_solar(o, t, toa_bias=1e-4), ORBIT, LATS, LONS, n_days=5)
    assert report["criteria"]["SOLAR-XCHECK-TOA"]["status"] == "FAIL"
    # cosz 本身没动 ⇒ 几何公式层仍 PASS
    assert report["criteria"]["SOLAR-XCHECK-COSZ"]["status"] == "PASS"


# ---- 形状/契约 ----

def test_shape_mismatch_raises():
    def bad(orbit, t):
        out = fake_ac_solar(orbit, t)
        out["cosz"] = out["cosz"][:, :10]
        return out
    with pytest.raises(ValueError, match="形状"):
        sx.xcheck_geometry(bad, ORBIT, LATS, LONS, 0.0)


def test_obliquity0_symmetry_sanity_value():
    val = sx.sanity_obliquity0_symmetry(fake_ac_solar, ORBIT, LATS, LONS)
    assert val == pytest.approx(0.0, abs=1e-12)


def test_cli_without_atmo_core_returns_2(capsys, monkeypatch):
    # atmo_core（A1）尚未交付：CLI 必须明确说"未交付"而非假跑。
    # 注入 None 强制 import 失败，测试不依赖本机是否装了 atmo_core。
    import sys
    monkeypatch.setitem(sys.modules, "atmo_core", None)
    rc = sx.main([])
    assert rc == 2
    assert "尚未交付" in capsys.readouterr().out
