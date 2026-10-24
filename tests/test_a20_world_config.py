"""A20 回归：时间尺度 / 动态 NPP / 群系分类（全部默认关）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import (  # noqa: E402
    AtmosphereConfig, EnvFieldConfig, NPPConfig, SimConfig, TimeScaleConfig, WorldConfig,
)
from world.biome import BIOME_NAMES, biome_stats, classify  # noqa: E402
from world.env_field import EnvField  # noqa: E402
from world.light_and_temperature import LightAndTemperature  # noqa: E402
from world.sphere_world import SphereWorld  # noqa: E402

TIER = "fast"


# ---------------------------------------------------------------- A20.3 时间尺度

def test_timescale_default_is_36s_and_changes_nothing():
    cfg = SimConfig(world=WorldConfig(rows=30, cols=60))
    assert cfg.time_scale.tick_seconds == 36
    assert cfg.light.rotation_period == 2400
    assert cfg.atmosphere.dt_s == pytest.approx(120.0)


def test_timescale_360_syncs_rotation_and_dt():
    cfg = SimConfig(world=WorldConfig(rows=30, cols=60),
                    time_scale=TimeScaleConfig(tick_seconds=360),
                    atmosphere=AtmosphereConfig(enabled=True))
    assert cfg.light.rotation_period == 240
    assert cfg.atmosphere.dt_s == pytest.approx(1200.0)
    assert cfg.time_scale.year_ticks == 240 * 365
    # 汇率自洽（config 的硬断言已过 ⇒ 这里再钉一次读数）
    assert cfg.atmosphere.substeps * cfg.atmosphere.dt_s / cfg.atmosphere.every_ticks == 360.0


def test_timescale_rejects_other_values():
    with pytest.raises(AssertionError):
        TimeScaleConfig(tick_seconds=100)


# ---------------------------------------------------------------- A20.1 动态 NPP

def _env(**kw) -> EnvField:
    w = SphereWorld(30, 60)
    lt = LightAndTemperature(w)
    return EnvField(w, lt, EnvFieldConfig(enabled=True, c0b_enabled=True, **kw))


def test_dynamic_npp_default_off():
    assert NPPConfig().dynamic is False
    e = _env(npp=NPPConfig(enabled=True))
    # 关档时引擎不会调 refresh；这里只钉"默认值"
    assert e.npp is not None


def test_refresh_npp_changes_with_temperature():
    e = _env(npp=NPPConfig(enabled=True))
    n0 = e.npp.copy()
    lat = np.repeat(np.asarray(e.world._lat, dtype=np.float64), e.world.cols)
    warm = np.full(e.world.n_cells, 25.0)
    cold = np.full(e.world.n_cells, -30.0)
    e.refresh_npp_from_climate(warm)
    n_warm = e.npp.copy()
    e.refresh_npp_from_climate(cold)
    n_cold = e.npp.copy()
    assert not np.allclose(n_warm, n_cold), "冷/暖重算必须不同"
    assert float(n_cold.max()) < float(n_warm.max()) + 1e-9
    assert np.isfinite(n0).all()


def test_refresh_npp_rejects_bad_input():
    e = _env(npp=NPPConfig(enabled=True))
    with pytest.raises(ValueError):
        e.refresh_npp_from_climate(np.zeros(e.world.n_cells - 1))
    bad = np.full(e.world.n_cells, np.nan)
    with pytest.raises(ValueError):
        e.refresh_npp_from_climate(bad)


def test_set_env_factors_validates():
    from simulation.config import ResourceConfig
    from simulation.sphere_engine import SphereEngine
    eng = SphereEngine(SimConfig(world=WorldConfig(rows=30, cols=60),
                                 env_field=EnvFieldConfig(enabled=True, c0b_enabled=True),
                                 resources=ResourceConfig(distribution="uniform")))
    n = eng.world.n_cells
    with pytest.raises(ValueError):
        eng.resources.set_env_factors(capacity=np.ones(n - 1))
    with pytest.raises(ValueError):
        eng.resources.set_env_factors(capacity=-np.ones(n))
    before = np.array(eng.resources._capacity, copy=True)
    eng.resources.set_env_factors(capacity=np.ones(n))
    assert eng.resources._capacity.shape == before.shape


# ---------------------------------------------------------------- A20.2 群系

def test_biome_basic_cases():
    t = np.array([30.0, 25.0, 25.0, 15.0, 15.0, 0.0, -20.0, -40.0])
    p = np.array([2500.0, 300.0, 2500.0, 800.0, 800.0, 300.0, 100.0, 50.0])
    amp = np.array([2.0, 2.0, 2.0, 3.0, 20.0, 3.0, 3.0, 3.0])
    b = classify(t, p, season_amp_c=amp)
    names = [BIOME_NAMES[i] for i in b]
    assert names[0] == "tropical_rainforest"
    assert names[1] == "savanna"
    assert names[3] == "temperate_oceanic"
    assert names[4] == "temperate_continental"
    assert names[6] == "ice"


def test_biome_threshold_sensitivity_is_monotone():
    """阈值整体 ±20%：雨林/荒漠的覆盖率应随阈值升高而**单调**变化。"""
    rng = np.random.default_rng(0)
    t = rng.uniform(-30, 35, (30, 60))
    p = rng.uniform(0, 3000, (30, 60))
    from world.biome import BiomeThresholds
    base = dict(ice_t=-10.0, tundra_t=-2.0, desert_p=250.0, rainforest_p=1800.0,
                tropical_t=20.0, temperate_t=8.0, continental_amp=12.0)
    lo = classify(t, p, thresholds=BiomeThresholds(**{k: v * 0.8 for k, v in base.items()}))
    hi = classify(t, p, thresholds=BiomeThresholds(**{k: v * 1.2 for k, v in base.items()}))
    f_lo = biome_stats(lo)["frac"]
    f_hi = biome_stats(hi)["frac"]
    assert f_lo["tropical_rainforest"] >= f_hi["tropical_rainforest"]
    assert f_lo["desert"] <= f_hi["desert"]


def test_biome_stats_are_fractions_and_reject_bad_input():
    b = classify(np.zeros((4, 8)), np.full((4, 8), 1000.0))
    st = biome_stats(b)
    assert sum(st["frac"].values()) == pytest.approx(1.0, abs=1e-9)
    with pytest.raises(ValueError):
        classify(np.zeros((4, 8)), np.full((4, 7), 1000.0))
    with pytest.raises(ValueError):
        classify(np.full((4, 8), np.nan), np.full((4, 8), 1000.0))
