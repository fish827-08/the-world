"""A17.3 回归：世界消费 GCM 降水（`EnvField.pr` 换空间型，总量不变）。

覆盖
----
1. **默认不换**：`EnvFieldSource` 下 `pr_source is None`，probe **不新增任何键**（逐位等价前提）。
2. **换源守恒**：面积权均值不变（`keep_total=True`），只改空间型；诊断有 corr/缩放系数。
3. **fail-loud**：形状不符 / 含 NaN / 有负值 / 均值为 0 ⇒ `ValueError`（不静默）。
4. **引擎接线**：注入 `GCMTableSource` ⇒ `env_field.pr_source == "gcm_table"`，降水确实变了；
   默认引擎（无表源）⇒ 不换。
5. **NPP 随降水重算**：开 NPP 换源后 `npp` 场改变（不是拿旧 proxy 算完就锁死）。

规模：60×120 世界 + 8×16 合成表，全部 < 1 s；tier = fast。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import (  # noqa: E402
    EnvFieldConfig, NPPConfig, SimConfig, WorldConfig,
)
from simulation.sphere_engine import SphereEngine  # noqa: E402
from world.climate import GCMTableSource  # noqa: E402
from world.env_field import EnvField  # noqa: E402
from world.light_and_temperature import LightAndTemperature  # noqa: E402
from world.sphere_world import SphereWorld  # noqa: E402

TIER = "fast"
ROWS, COLS = 60, 120


def _env(**kw) -> EnvField:
    w = SphereWorld(ROWS, COLS)
    lt = LightAndTemperature(w)
    return EnvField(w, lt, EnvFieldConfig(enabled=True, c0b_enabled=True, **kw))


def _area_mean(v: np.ndarray, w: SphereWorld) -> float:
    a = w.cell_area(np.arange(w.n_cells))
    return float(np.sum(np.asarray(v, dtype=np.float64) * a) / np.sum(a))


def test_default_has_no_precip_override():
    env = _env()
    assert env.pr_source is None and env.pr_proxy is None
    pr = env.probe()
    assert not any(k.startswith("pr_") and k != "pr" for k in pr), \
        f"关档不该新增 pr_* 键，实得 {[k for k in pr if k.startswith('pr_')]}"


def test_set_precip_pattern_keeps_total_and_reports():
    env = _env()
    before = _area_mean(env.pr, env.world)
    lat = np.degrees(np.asarray(env.world._lat, dtype=np.float64))
    fake = np.repeat((0.2 + np.cos(np.radians(lat)))[:, None], COLS, axis=1)
    diag = env.set_precip_pattern(fake, source="unit-test")
    after = _area_mean(env.pr, env.world)
    assert after == pytest.approx(before, rel=1e-12), "面积权均值必须不变（keep_total）"
    assert diag["mean_before"] == pytest.approx(diag["mean_after"], rel=1e-9)
    assert np.isfinite(diag["pattern_rel_dev"])
    pr = env.probe()
    assert pr["pr_source"] == "unit-test"
    assert "pr_corr_proxy" in pr and "pr_scale" in pr


def test_set_precip_pattern_rejects_bad_fields():
    env = _env()
    n = env.world.n_cells
    bad = np.ones(n)
    for arr in (bad[:-1],                                  # 形状不符
                np.full(n, np.nan),                        # NaN
                np.concatenate([-np.ones(1), np.ones(n - 1)]),  # 负值
                np.zeros(n)):                              # 均值 0
        with pytest.raises(ValueError):
            env.set_precip_pattern(arr)
    assert env.pr_source is None, "失败调用不能留半截状态"


def test_engine_with_gcm_table_swaps_precip(tmp_path):
    nlat, nlon = 16, 32
    lat = np.linspace(-90 + 180 / nlat / 2, 90 - 180 / nlat / 2, nlat)
    lon = np.arange(nlon) * 360.0 / nlon
    # 降水：热带峰 + 经向 4 波（明显区别于世界 proxy 的纬度带型）
    base = 3000.0 * np.exp(-(lat / 20.0) ** 2) + 100.0
    pr = base[:, None] * (1.0 + 0.5 * np.cos(np.radians(lon * 4)))[None, :]
    npz = tmp_path / "synth.npz"
    np.savez_compressed(npz, lat=lat, lon=lon,
                        temperature_c=np.repeat(lat[:, None] * 0.3, nlon, axis=1) + 10.0,
                        precip_mm_yr=pr)

    cfg = SimConfig(world=WorldConfig(rows=ROWS, cols=COLS),
                    env_field=EnvFieldConfig(enabled=True, c0b_enabled=True))
    src = GCMTableSource(npz, year_ticks=1200)
    eng = SphereEngine(cfg, climate_source=src)
    env = eng.env_field
    assert env.pr_source == "gcm_table"
    # A18.3 起换源发生在**构造期**（水文之前）⇒ 没有"旧 pr"可留，改用 pr_diag 记账
    assert env.pr_diag and env.pr_diag["policy"] == "keep_total"
    assert env.pr_diag["mean_before"] == pytest.approx(env.pr_diag["mean_after"], rel=1e-9)

    cfg2 = SimConfig(world=WorldConfig(rows=ROWS, cols=COLS),
                     env_field=EnvFieldConfig(enabled=True, c0b_enabled=True))
    ref = SphereEngine(cfg2).env_field
    assert not np.allclose(env.pr, ref.pr), "换了源却没改降水场 = 假接线"
    assert _area_mean(env.pr, env.world) == pytest.approx(_area_mean(ref.pr, ref.world),
                                                          rel=1e-9)


def test_engine_default_does_not_swap(tmp_path):
    cfg = SimConfig(world=WorldConfig(rows=ROWS, cols=COLS),
                    env_field=EnvFieldConfig(enabled=True, c0b_enabled=True))
    eng = SphereEngine(cfg)
    assert eng.env_field.pr_source is None
    assert "pr_source" not in eng.env_field.probe()


def test_npp_is_recomputed_after_precip_swap(tmp_path):
    """NPP 吃的是 pr ⇒ 换源后必须重算（否则容量还是旧 proxy 算的）。"""
    env = _env(npp=NPPConfig(enabled=True))
    npp_before = env.npp.copy()
    lat = np.degrees(np.asarray(env.world._lat, dtype=np.float64))
    fake = np.repeat((0.6 + 0.4 * np.cos(np.radians(lat * 6.0)))[:, None], COLS, axis=1)
    env.set_precip_pattern(fake, source="unit-test")
    assert env.npp is not None
    assert not np.allclose(env.npp, npp_before), "换降水后 NPP 没重算"
