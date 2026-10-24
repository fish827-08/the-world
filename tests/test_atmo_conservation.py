# TIER = "fast"
"""AV 守恒/确定性检查单测：能量闭合、逐位确定性对比。"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.atmo import conservation


def _grid():
    lat = np.linspace(-90.0, 90.0, 32)
    lon = np.linspace(0.0, 360.0, 64, endpoint=False)
    return lat, lon


def test_energy_closure_exact_when_balanced():
    lat, lon = _grid()
    nt = 4
    days = np.array([10.0, 11.0, 12.0, 13.0])
    c_slab = 1e7
    net = 20.0  # W m-2 TOA 净通量
    sst = np.empty((nt, lat.size, lon.size))
    for i, d in enumerate(days):
        sst[i] = 300.0 + net * (d - days[0]) * 86400.0 / c_slab
    sw = np.full((nt, lat.size, lon.size), net)
    lw = np.zeros_like(sw)
    en = conservation.energy_closure(sw, lw, sst, c_slab, days, lat)
    assert abs(en["residual"]) < 1e-6
    assert en["net_toa"] == pytest.approx(net)
    assert en["sst_trend_k_per_day"] == pytest.approx(net * 86400.0 / c_slab, rel=1e-6)


def test_energy_closure_detects_leak():
    lat, lon = _grid()
    nt = 4
    days = np.array([0.0, 1.0, 2.0, 3.0])
    c_slab = 1e7
    sw = np.full((nt, lat.size, lon.size), 20.0)
    lw = np.zeros_like(sw)
    sst = np.broadcast_to(300.0, (nt, lat.size, lon.size)).copy()  # SST 不变 ⇒ 20 W/m2 全漏
    en = conservation.energy_closure(sw, lw, sst, c_slab, days, lat)
    assert abs(en["residual"] - 20.0) < 1e-6


def _write_run(tmp_path, tag, seed_arr):
    d = tmp_path / tag
    d.mkdir()
    lat, lon = _grid()
    np.savez(d / "coords.npz", lat=lat, lon=lon, sigma=np.array([0.9, 0.6, 0.3]))
    snaps = []
    for t in range(3):
        arr = {"u": seed_arr.copy(), "v": np.zeros_like(seed_arr),
               "T": np.full_like(seed_arr, 300.0),
               "ps": np.full((lat.size, lon.size), 100000.0)}
        np.savez(d / f"snap_{t:06d}.npz", **arr)
        snaps.append(arr)
    (d / "meta.json").write_text(json.dumps({
        "benchmark": "held_suarez", "dt_atmo_s": 600.0, "days": 3,
        "snap_days": [1.0, 2.0, 3.0], "obliquity_deg": 0.0, "seed": 7,
        "config_hash": "abc",
        "forcing": {"omega": 7.292e-5, "g": 9.81, "R": 287.0, "cp": 1004.0,
                    "radius": 6.371e6, "p0": 100000.0, "delta_Ty": 60.0,
                    "delta_theta_z": 10.0, "sigma_b": 0.7, "ka": 1 / (40 * 86400.0),
                    "ks": 1 / (4 * 86400.0), "kf": 1 / 86400.0},
    }), encoding="utf-8")
    return d, snaps


def test_determinism_identical_runs_pass(tmp_path):
    lat, lon = _grid()
    arr = np.zeros((2, lat.size, lon.size)) + 5.0
    da, _ = _write_run(tmp_path, "a", arr)
    db, _ = _write_run(tmp_path, "b", arr)
    assert conservation.determinism_compare(da, db) == []


def test_determinism_one_bit_flip_fails(tmp_path):
    lat, lon = _grid()
    arr = np.zeros((2, lat.size, lon.size)) + 5.0
    da, _ = _write_run(tmp_path, "a", arr)
    arr2 = arr.copy()
    arr2[0, 1, 3] += 1e-12  # 末位扰动
    db, _ = _write_run(tmp_path, "b", arr2)
    mism = conservation.determinism_compare(da, db)
    assert len(mism) == 3  # 三张快照的 u 均含该末位扰动
    assert all(m["why"] == "bitwise differ" for m in mism)
