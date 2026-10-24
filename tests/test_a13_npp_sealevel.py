"""A13 回归：NPP 派生容量 + 海平面总水量反解 + 斑块派生统计（A16a）。

覆盖（审核线裁定 A1 的三条件 + 变更单验收）
------------------------------------------
1. **默认关**：`npp.enabled=False` ⇒ `env.npp is None` 且 probe **不新增任何键**（逐位等价的前提）。
2. **NPP v1 只用真实场**：只吃 temp/pr/纬度；不含 soil_water/养分（本测试以"输入签名"钉死）。
3. **斑块是派生量**：连通块数/覆盖率/块径进 `patch_stats`；阈值升高 ⇒ 块数不增。
4. **海平面极端水量**：0 ⇒ 几乎全陆；极大 ⇒ 全淹没；单调性（水量↑ ⇒ 陆地↓）；守恒（体积 = 目标）。
5. **互斥 fail-loud**：`distribution='npp'` 与 `patch_count` 显式改值 ⇒ 报错；与 `npp.enabled` 不配对 ⇒ 报错。

规模：60×120（7200 格），单例 < 1 s；tier = fast。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import (  # noqa: E402
    EnvFieldConfig, NPPConfig, ResourceConfig, SeaLevelConfig, SimConfig,
)
from world.env_field import EnvField, _sea_level_from_water  # noqa: E402
from world.light_and_temperature import LightAndTemperature  # noqa: E402
from world.npp import compute_npp, label_components, patch_stats  # noqa: E402
from world.sphere_world import SphereWorld  # noqa: E402

TIER = "fast"
ROWS, COLS = 60, 120


def _env(**kw) -> EnvField:
    w = SphereWorld(ROWS, COLS)
    lt = LightAndTemperature(w)
    return EnvField(w, lt, EnvFieldConfig(c0b_enabled=True, **kw))


# ---------------------------------------------------------------- 默认关

def test_npp_off_by_default_and_probe_unchanged():
    env = _env()
    assert env.npp is None
    pr = env.probe()
    assert "npp_on" not in pr and "n_patches" not in pr


def test_npp_on_produces_field_and_patch_stats():
    env = _env(npp=NPPConfig(enabled=True))
    assert env.npp is not None and env.npp.shape == (ROWS * COLS,)
    # 归一化口径：**开海陆掩码时锚定「陆地面积权均值 = 1」**（容量只在陆地存在）；
    # 关掩码时才是全球均值 = 1。两者都保证总容量与 uniform 同量纲（A17.2 实测差 0.0%）。
    assert 0.0 <= float(env.npp.min())
    assert env.land_mask is not None
    a = env.world.cell_area(np.arange(env.world.n_cells))
    lm = np.asarray(env.land_mask, dtype=bool)
    land_mean = float(np.sum(env.npp[lm] * a[lm]) / np.sum(a[lm]))
    assert land_mean == pytest.approx(1.0, rel=1e-12)
    pr = env.probe()
    assert pr["npp_on"] is True
    assert pr["n_patches"] >= 0
    assert 0.0 <= pr["patch_cover_frac"] <= 1.0
    assert pr["patch_radius_km_mean"] >= 0.0


def test_npp_ignores_soil_and_nutrients():
    """v1 签名钉死：compute_npp 只接受 temp/precip/lat（没有 soil/nutrient 形参）。"""
    import inspect

    params = set(inspect.signature(compute_npp).parameters)
    assert {"temp_c", "precip_proxy", "lat_rad"} <= params
    assert not ({"soil_water", "nutrient", "nutrients"} & params)


# ---------------------------------------------------------------- 斑块派生

def test_components_count_and_wrap():
    mask = np.zeros((4, 8), dtype=bool)
    mask[0, 0] = True
    mask[0, 7] = True          # 经度环向与 (0,0) 连通
    labels, n = label_components(mask, wrap_lon=True)
    assert n == 1, "0/360 接缝应连通成一块"
    labels2, n2 = label_components(mask, wrap_lon=False)
    assert n2 == 2


def test_patch_stats_monotone_in_global_threshold():
    """旧口径（mode="global"）保留并保持单调：阈值越高（frac 越小）⇒ 覆盖越小。"""
    w = SphereWorld(ROWS, COLS)
    areas = w.cell_area(np.arange(w.n_cells))
    rng = np.random.default_rng(0)
    npp = np.abs(rng.random((ROWS, COLS)))
    low = patch_stats(npp.ravel(), ROWS, COLS, areas, mode="global", threshold_frac=0.5)
    high = patch_stats(npp.ravel(), ROWS, COLS, areas, mode="global", threshold_frac=0.05)
    assert high["patch_cover_frac"] < low["patch_cover_frac"]
    assert high["n_patches"] >= 1 and low["n_patches"] >= 1


def _lat_rows() -> np.ndarray:
    w = SphereWorld(ROWS, COLS)
    return np.asarray(w._lat, dtype=np.float64)


def test_local_threshold_collapses_with_delta():
    """局部阈值单调性：δ 越大 ⇒ 覆盖越小（相对周边的高值越少）。"""
    lat = _lat_rows()
    base = 0.3 + 0.7 * np.cos(lat)[:, None] * np.ones((1, COLS))
    rng = np.random.default_rng(1)
    npp = base * (1.0 + 0.15 * rng.standard_normal((ROWS, COLS)))
    w = SphereWorld(ROWS, COLS)
    areas = w.cell_area(np.arange(w.n_cells))
    covers = [patch_stats(npp.ravel(), ROWS, COLS, areas, mode="local",
                          local_window=9, local_delta=d)["patch_cover_frac"]
              for d in (0.01, 0.05, 0.2)]
    assert covers[0] >= covers[1] >= covers[2]
    assert covers[0] > 0.0


def test_global_threshold_degenerates_without_land_mask():
    """A16c 缺陷的可复现证据：**全网格 + 全局阈值** 在纯纬度场上连成一条环球带。

    这条测试是"为什么要 A17.4"的回归锚：若有人把它删了，退化会悄悄回来。
    """
    lat = _lat_rows()
    npp = 0.3 + 0.7 * np.cos(lat)[:, None] * np.ones((1, COLS))
    w = SphereWorld(ROWS, COLS)
    areas = w.cell_area(np.arange(w.n_cells))
    s = patch_stats(npp.ravel(), ROWS, COLS, areas, mode="global", threshold_frac=0.35)
    assert s["n_patches"] == 1
    assert s["patch_max_share"] > 0.99, "全局阈值在平滑纬度场上就是一个闭环带"


def test_local_threshold_does_not_make_a_belt_for_blobs():
    """平滑背景 + 局地凸起 ⇒ 局部阈值挑出**离散块**，最大块占比低（不再是环球带）。"""
    lat = _lat_rows()
    smooth = 0.3 + 0.7 * np.cos(lat)[:, None] * np.ones((1, COLS))
    npp = smooth.copy()
    bumps = [(10, 10), (20, 40), (35, 80), (48, 25), (52, 100)]
    rr, cc = np.meshgrid(np.arange(ROWS), np.arange(COLS), indexing="ij")
    for r0, c0 in bumps:
        d2 = (rr - r0) ** 2 + np.minimum(np.abs(cc - c0), COLS - np.abs(cc - c0)) ** 2
        npp = npp + 0.6 * smooth * np.exp(-d2 / (2 * 3.0 ** 2))
    w = SphereWorld(ROWS, COLS)
    areas = w.cell_area(np.arange(w.n_cells))
    s = patch_stats(npp.ravel(), ROWS, COLS, areas, mode="local",
                    local_window=9, local_delta=0.05)
    assert s["n_patches"] >= 3, f"应挑出多个离散块，实得 {s['n_patches']}"
    assert s["patch_max_share"] < 0.5, f"最大块占比 {s['patch_max_share']:.2f} 偏大（疑似连成带）"


def test_patch_stats_respects_land_mask():
    """陆地掩码：**只在陆地上找斑块**（A16c 的"环球带"正是海洋把结果连起来的）。

    构造：NPP 随经度递增 ⇒ 全局阈值只选东半球（后 50% 列）。
    ① 陆地也在东半球 ⇒ 覆盖 = 全球 50%、占陆地 100%；
    ② 陆地改到西半球 ⇒ 覆盖 = **0**（东半球那些高值格全在海洋，不算斑块）。
    """
    w = SphereWorld(ROWS, COLS)
    areas = w.cell_area(np.arange(w.n_cells))
    npp = (1.0 + 0.5 * np.arange(COLS) / COLS)[None, :] * np.ones((ROWS, 1))

    land_east = np.zeros((ROWS, COLS), dtype=bool)
    land_east[:, COLS // 2:] = True
    s1 = patch_stats(npp.ravel(), ROWS, COLS, areas, mode="global",
                     threshold_frac=0.5, land_mask=land_east)
    assert s1["land_frac"] == pytest.approx(0.5, abs=0.02)
    assert s1["patch_cover_frac"] == pytest.approx(0.5, abs=0.02)
    assert s1["patch_cover_frac_land"] == pytest.approx(1.0, abs=1e-6)

    land_west = ~land_east
    s2 = patch_stats(npp.ravel(), ROWS, COLS, areas, mode="global",
                     threshold_frac=0.5, land_mask=land_west)
    assert s2["patch_cover_frac"] == 0.0, "海洋里的高值格不该被算成斑块"
    assert s2["n_patches"] == 0


# ---------------------------------------------------------------- 海平面

def test_sea_level_extremes_and_conservation():
    env = _env()
    h, areas = env.h_m, env.world.cell_area(np.arange(env.world.n_cells))
    a_sum = float(areas.sum())

    sea0 = _sea_level_from_water(h, areas, 0.0)
    assert sea0 == pytest.approx(float(h.min()))

    sea_big = _sea_level_from_water(h, areas, 1.0e5)   # 10 万米等效水深 ⇒ 全淹没
    assert sea_big >= float(h.max())

    target = 1500.0
    sea = _sea_level_from_water(h, areas, target)
    vol = float(np.sum(areas * np.clip(sea - h, 0.0, None))) / a_sum
    assert vol == pytest.approx(target, rel=1e-6)


def test_volume_mode_matches_quantile_when_given_its_water():
    """一致性：把 quantile 档的海平面换算成水量，再喂给 volume 档 ⇒ 陆地比例应一致。"""
    base = _env()
    h, areas = base.h_m, base.world.cell_area(np.arange(base.world.n_cells))
    a_sum = float(areas.sum())
    water = float(np.sum(areas * np.clip(base.sea_level_m - h, 0.0, None))) / a_sum
    vol_env = _env(sea_level=SeaLevelConfig(mode="volume", total_water_m=water))
    land_q = float(base.conservation["land_area_frac"])
    land_v = float(vol_env.conservation["land_area_frac"])
    assert land_v == pytest.approx(land_q, abs=2e-3)


def test_volume_mode_land_fraction_monotone_in_water():
    land = []
    for water in (500.0, 1500.0, 3000.0):
        e = _env(sea_level=SeaLevelConfig(mode="volume", total_water_m=water))
        land.append(float(e.conservation["land_area_frac"]))
    assert land[0] >= land[1] >= land[2]
    assert land[0] > 0.0 and land[-1] < 1.0


# ---------------------------------------------------------------- 互斥 fail-loud

def test_npp_and_patch_count_conflict_raises():
    with pytest.raises(AssertionError):
        ResourceConfig(distribution="npp", patch_count=1700)


def test_npp_needs_matching_distribution():
    with pytest.raises(ValueError):
        SimConfig(resources=ResourceConfig(distribution="npp"))       # 没开 npp
    with pytest.raises(ValueError):
        SimConfig(env_field=EnvFieldConfig(npp=NPPConfig(enabled=True)))  # 开 npp 但 distribution 不是 npp
