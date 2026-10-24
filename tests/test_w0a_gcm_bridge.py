"""`atmo_bridge.remap` 与 `world.climate.GCMTableSource` 的回归测试（W1/A12 补）。

覆盖（派工 A12 点名五项）
------------------------
1. **守恒**：重映射后全球面积权均值 = 源全球均值（温度/降水）。
2. **恒等**：同网格重映射 = 原场（逐位）。
3. **越界**：目标格与源网格无交叠 ⇒ `ValueError`（不静默给 0）。
4. **月表插值**：12 月层按 tick 线性插值；1/4 年与年初的差 = 季节振幅。
5. **错误路径**：无表 / 月表缺 `year_ticks` / 缺字段 ⇒ 显式报错。

规范性（W0-T 新规）
------------------
* 不重算物理常数：本文件只做**纯几何/纯数组**断言，不涉及任何物理常数。
* 网格取小尺寸（16×32 → 24×48），单例 < 1 s，`tier = fast`。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atmo_bridge.remap import build_weights, global_mean, remap_field  # noqa: E402
from world.climate import GCMTableSource  # noqa: E402
from world.sphere_world import SphereWorld  # noqa: E402

TIER = "fast"
NLAT, NLON = 16, 32


def _lat(nlat: int) -> np.ndarray:
    return np.linspace(-90 + 90 / nlat, 90 - 90 / nlat, nlat)


def _lon(nlon: int) -> np.ndarray:
    return np.arange(nlon) * 360.0 / nlon


def _table(path: Path, *, months: int = 1, with_v: bool = True) -> dict:
    """解析表：赤道 25℃ / 极 -30℃；降水赤道峰；u 为西风带。"""
    lat, lon = _lat(NLAT), _lon(NLON)
    phi = np.radians(lat)
    t = (25.0 * np.cos(phi) ** 2 - 30.0 * np.sin(phi) ** 4)[:, None] * np.ones((1, NLON))
    p = (1200.0 * np.exp(-(lat / 10.0) ** 2))[:, None] * np.ones((1, NLON))
    u = (18.0 * np.sin(2 * phi) ** 2 * np.sign(lat))[:, None] * np.ones((1, NLON))
    if months == 12:
        season = np.sin(2 * np.pi * np.arange(12) / 12)
        t = t[None] + 8.0 * season[:, None, None]
        p = np.repeat(p[None], 12, axis=0) * 1.0
        u = np.repeat(u[None], 12, axis=0) * 1.0
    payload = {"lat": lat, "lon": lon, "temperature_c": t, "precip_mm_yr": p, "u": u}
    if with_v:
        payload["v"] = np.zeros_like(u)
    payload["meta_json"] = np.array(json.dumps(
        {"format": "npz-climatology-v0", "source": "unit-test", "months": months}))
    np.savez_compressed(path, **payload)
    return payload


# ---------------------------------------------------------------- remap

def test_remap_row_sums_are_one_and_constant_is_preserved():
    wl, wo = build_weights(_lat(NLAT), _lon(NLON), _lat(24), _lon(48))
    assert np.allclose(wl.sum(axis=1), 1.0, atol=1e-12)
    assert np.allclose(wo.sum(axis=1), 1.0, atol=1e-12)
    out = remap_field(np.full((NLAT, NLON), 7.0), wl, wo)
    assert np.allclose(out, 7.0, atol=1e-12)


def test_remap_identity_on_same_grid():
    lat, lon = _lat(24), _lon(48)
    wl, wo = build_weights(lat, lon, lat, lon)
    field = np.arange(24 * 48, dtype=np.float64).reshape(24, 48)
    assert np.array_equal(remap_field(field, wl, wo), field)


def test_remap_global_mean_is_conserved():
    lat_s = _lat(NLAT)
    field = np.cos(np.radians(lat_s))[:, None] * np.ones((1, NLON))
    wl, wo = build_weights(lat_s, _lon(NLON), _lat(24), _lon(48))
    out = remap_field(field, wl, wo)
    assert abs(global_mean(out, _lat(24)) - global_mean(field, lat_s)) < 1e-12


def test_remap_accepts_descending_source():
    lat_s, lon_s = _lat(NLAT)[::-1], _lon(NLON)[::-1]
    wl, wo = build_weights(lat_s, lon_s, _lat(24), _lon(48))
    out = remap_field(np.full((NLAT, NLON), 3.0), wl, wo)
    assert np.allclose(out, 3.0, atol=1e-12)


def test_remap_out_of_range_lat_raises():
    """源网格只覆盖中纬（10–20°N），目标覆盖整球 ⇒ 极行无交叠必须报错，不许给 0。"""
    with pytest.raises(ValueError):
        build_weights(np.array([10.0, 20.0]), _lon(4), _lat(8), _lon(8))


# ---------------------------------------------------------------- GCMTableSource

def test_bootstrap_and_readback_shapes(tmp_path):
    p = tmp_path / "t.npz"
    _table(p)
    world = SphereWorld(24, 48)
    src = GCMTableSource(p)
    src.bootstrap(world)
    flat = np.arange(world.n_cells)
    t = src.temperature_c(flat, 0)
    assert t.shape == (world.n_cells,)
    assert np.isfinite(t).all()
    assert src.precip_mm_yr().shape == (24, 48)
    w = src.wind()
    assert w["u"].shape == (24, 48) and w["v"].shape == (24, 48)


def test_precip_is_nonnegative_and_conserved(tmp_path):
    p = tmp_path / "t.npz"
    src_tab = _table(p)
    world = SphereWorld(24, 48)
    src = GCMTableSource(p)
    src.bootstrap(world)
    out = src.precip_mm_yr()
    assert out.min() >= 0.0
    assert abs(global_mean(out, _lat(24)) - global_mean(src_tab["precip_mm_yr"], src_tab["lat"])) < 1e-9


def test_monthly_interpolation_uses_tick(tmp_path):
    p = tmp_path / "m.npz"
    _table(p, months=12)
    year = 1200
    src = GCMTableSource(p, year_ticks=year)
    src.bootstrap(SphereWorld(24, 48))
    flat = np.array([0, 100, 500])
    t0 = src.temperature_c(flat, 0)
    tq = src.temperature_c(flat, year // 4)     # phase=3 → sin(π/2)=1
    assert np.abs(tq - t0).max() == pytest.approx(8.0, abs=1e-9)
    th = src.temperature_c(flat, year // 2)     # phase=6 → sin(π)=0
    assert np.abs(th - t0).max() == pytest.approx(0.0, abs=1e-9)


def test_wind_polar_rows_are_degenerate_safe(tmp_path):
    p = tmp_path / "t.npz"
    _table(p)
    src = GCMTableSource(p)
    src.bootstrap(SphereWorld(24, 48))
    w = src.wind()
    assert np.allclose(w["v"][0], 0.0) and np.allclose(w["v"][-1], 0.0)
    assert np.allclose(w["u"][0], w["u"][0].mean())


def test_missing_table_path_raises():
    with pytest.raises(NotImplementedError):
        GCMTableSource(None).bootstrap(SphereWorld(8, 16))


def test_monthly_without_year_ticks_raises(tmp_path):
    p = tmp_path / "m.npz"
    _table(p, months=12)
    with pytest.raises(NotImplementedError):
        GCMTableSource(p).bootstrap(SphereWorld(8, 16))


def test_missing_field_raises(tmp_path):
    p = tmp_path / "noprecip.npz"
    payload = _table(p)
    lat, lon = payload["lat"], payload["lon"]
    np.savez_compressed(p, lat=lat, lon=lon, temperature_c=payload["temperature_c"])
    src = GCMTableSource(p)
    src.bootstrap(SphereWorld(8, 16))
    with pytest.raises(NotImplementedError):
        src.precip_mm_yr()


def test_probe_reports_grid_and_months(tmp_path):
    p = tmp_path / "t.npz"
    _table(p)
    src = GCMTableSource(p)
    src.bootstrap(SphereWorld(24, 48))
    pr = src.probe()
    assert pr["climate_table_shape"] == [NLAT, NLON]
    assert pr["climate_table_months"] == 1
    assert pr["climate_table_meta"]["world_grid"] == [24, 48]
