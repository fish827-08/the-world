"""A3.2 重映射骨架回归：Gaussian T21/T31 ↔ 世界规则网格（480×960）。

覆盖
----
R1 常值场双向偏差 ≤1e-12（T21/T31）
R2 全球面积权均值双向守恒（随机场，T21/T31）
R3 跨接缝 + 极点：无 NaN；纬向常值场在极点行仍为经向常数；源格跨接缝权重分配完整
R4 真 `EnvField.land_mask`（60×120）→ T21 → 回世界：面积权陆地比例一致
R5 北→南纬序：显式翻转标记 + 输出与升序输入逐位一致（防 A17 南北镜像）
R6 规则经纬上与 `atmo_bridge.remap`（A17）权重对拍（机器精度）
R7 fail-loud：面积和/纬序/场形状非法一律 ValueError
R8 真 480×960 尺寸 + 3D（nlev,nlat,nlon）通道

A3.3（时钟 + 状态桥）
C1 M2′ 两档（36 s ⇒ k=25；360 s ⇒ k=2）N=10,000 tick 零漂移
C2 时钟 fail-loud（dt 太短 / tick_split / 年长非整数倍）
B1 交换相位 + 幂等 + 倒带拒绝
B2 同 seed 两次逐位一致（state digest / t2m / precip）；换 seed 只动降水型
B3 OPEN-5：state 原地（零拷贝）+ 出口缓存是 copy（旧缓存不被新步改写）
B4 桩显式 NotImplementedError（不返回假数据）
B5 世界边界（SST）真的进入 atmo 并回到世界 t2m（链路方向性）

A3.4（出口场 + 世界消费）
W1 `AtmoConfig` 默认关 + 非法组合 fail-loud
W2 天气模式四条守卫（P̄ 锚 / C1 band / C0b / 灵敏度 & Rust 路径）
W3 真引擎：构造期降水换源（`pr_source=="atmo"`、`band is None`、`atmosphere is None`）
W4 世界消费：`temperature_c` 调用次数、t2m 缓存、keep_total 均值不变、交换节拍
P1 `ModuleAtmoCore` 按 v1.0.2 §8.7 调 `export(st, g)`（网格透传）
P2 适配器 fail-loud（未知 INTERFACE_VERSION / 缺 export）

tier = fast（全部 <2 s；真 480×960 只建权不跑世界）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atmo.coupling import (  # noqa: E402
    AtmoConfig, AtmoCoreStub, CouplingBridge, CouplingClock, ModuleAtmoCore,
    SyntheticCore, make_weather_engine, validate_weather_config,
)
from atmo.coupling.grid import (  # noqa: E402
    Grid, grid_from_atmo_spec, regular_grid, world_grid,
)
from atmo.coupling.remap import (  # noqa: E402
    RemapPair, build_weights, global_mean, lat_weights, lon_weights, mu_edges,
)
from simulation.config import SimConfig  # noqa: E402
from world.sphere_world import SphereWorld  # noqa: E402

TIER = "fast"


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
def _gauss_spec(nlat: int, nlon: int, *, reverse: bool = False) -> dict:
    """T21/T31 的高斯节点（numpy 版，测试夹具；生产走 `atmo_core.grid_spec`）。"""
    nodes, weights = np.polynomial.legendre.leggauss(nlat)
    lat = np.degrees(np.arcsin(nodes))
    area = weights / 2.0
    lon = np.arange(nlon, dtype=np.float64) * 360.0 / nlon
    if reverse:
        lat, area = lat[::-1], area[::-1]
    return {"nlat": nlat, "nlon": nlon, "kind": "gaussian",
            "lat_deg": lat, "lon_deg": lon, "area": area}


def _gauss_grid(nlat: int, nlon: int, *, name: str = "atmo") -> Grid:
    return grid_from_atmo_spec(_gauss_spec(nlat, nlon), name=name)


@pytest.fixture(scope="module")
def world_small():
    return SphereWorld(60, 120)


@pytest.fixture(scope="module")
def world_small_grid(world_small):
    return world_grid(world_small)


@pytest.fixture(scope="module")
def world_s2():
    return SphereWorld(480, 960)


# ---------------------------------------------------------------------------
# R0 夹具锚：Gaussian 节点/面积权必须与 A1 `atmo_core.grid_spec` 金标一致
# ---------------------------------------------------------------------------
def test_gaussian_fixture_matches_a1_golden():
    g = _gauss_grid(32, 64)
    assert g.lat_deg[0] == pytest.approx(-85.76058712, abs=1e-8)
    assert g.lat_deg[-1] == pytest.approx(85.76058712, abs=1e-8)
    assert g.area[0] == pytest.approx(0.003509305004735068, rel=1e-12)
    assert g.area.sum() == pytest.approx(1.0, abs=1e-14)

    # 🔴 面积边 ≠ A1 的 lat_edges_deg（μ 中点边）：T21 首行差 0.7°
    a1_midpoint_edge = np.degrees(np.arcsin(0.5 * (np.sin(np.radians(g.lat_deg[0]))
                                                  + np.sin(np.radians(g.lat_deg[1])))))
    assert a1_midpoint_edge == pytest.approx(-82.49685126, abs=1e-6)
    our_edge = np.degrees(np.arcsin(mu_edges(g.area)[1]))
    assert our_edge == pytest.approx(-83.207689584, abs=1e-9)
    assert our_edge == pytest.approx(
        np.degrees(np.arcsin(-1.0 + 2.0 * g.area[0])), abs=1e-12)
    assert our_edge < a1_midpoint_edge - 0.5, "面积边必须按 cumsum(area) 而非 μ 中点"


# ---------------------------------------------------------------------------
# R1 常值场
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nlat,nlon", [(32, 64), (48, 96)])
def test_constant_field_preserved(nlat, nlon, world_small_grid):
    atmo = _gauss_grid(nlat, nlon)
    a2w = RemapPair.build(atmo, world_small_grid)
    w2a = RemapPair.build(world_small_grid, atmo)

    out = a2w.remap(np.ones((nlat, nlon)))
    assert np.max(np.abs(out - 1.0)) <= 1e-12
    out_w = w2a.remap(np.ones((world_small_grid.nlat, world_small_grid.nlon)))
    assert np.max(np.abs(out_w - 1.0)) <= 1e-12


# ---------------------------------------------------------------------------
# R2 全球面积权均值守恒
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nlat,nlon", [(32, 64), (48, 96)])
def test_global_mean_conserved_both_directions(nlat, nlon, world_small_grid):
    rng = np.random.default_rng(20261020)
    atmo = _gauss_grid(nlat, nlon)

    f_a = rng.normal(size=(nlat, nlon))
    out_w = RemapPair.build(atmo, world_small_grid).remap(f_a)
    assert global_mean(out_w, world_small_grid) == pytest.approx(
        global_mean(f_a, atmo), abs=1e-12, rel=0.0)

    f_w = rng.normal(size=(world_small_grid.nlat, world_small_grid.nlon))
    out_a = RemapPair.build(world_small_grid, atmo).remap(f_w)
    assert global_mean(out_a, atmo) == pytest.approx(
        global_mean(f_w, world_small_grid), abs=1e-12, rel=0.0)


# ---------------------------------------------------------------------------
# R3 接缝 / 极点
# ---------------------------------------------------------------------------
def test_pole_and_seam_no_nan(world_small_grid):
    atmo = _gauss_grid(32, 64)
    lat = np.repeat(world_small_grid.lat_deg[:, None], world_small_grid.nlon, axis=1)
    lon = np.repeat(world_small_grid.lon_deg[None, :], world_small_grid.nlat, axis=0)
    f = (np.sin(np.radians(3.0 * lat)) + np.cos(np.radians(2.0 * lon))
         + (lon > 180.0).astype(np.float64))

    out = RemapPair.build(world_small_grid, atmo).remap(f)
    assert np.all(np.isfinite(out))
    back = RemapPair.build(atmo, world_small_grid).remap(out)
    assert np.all(np.isfinite(back))

    # 纬向权重只依赖纬行 ⇒ 纬向常值源在极区也必须保持经向常数
    f_lat_only = np.repeat(np.sin(np.radians(3.0 * lat[:, :1])), world_small_grid.nlon,
                           axis=1)
    out_z = RemapPair.build(world_small_grid, atmo).remap(f_lat_only)
    assert np.allclose(out_z, out_z[:, :1], rtol=0.0, atol=1e-12)
    back_z = RemapPair.build(atmo, world_small_grid).remap(out_z)
    assert np.allclose(back_z[0], back_z[0, 0], rtol=0.0, atol=1e-12)
    assert np.allclose(back_z[-1], back_z[-1, 0], rtol=0.0, atol=1e-12)


def test_seam_shifted_source_feeds_both_edges():
    """源经度错开 1/4 格 ⇒ 源第 0 格跨接缝，权重必须落在目标首列/次列。"""
    nlat, nlon = 16, 120
    lat = -90.0 + (np.arange(nlat) + 0.5) * 180.0 / nlat
    step = 360.0 / nlon
    lon_shift = (np.arange(nlon, dtype=np.float64) + 0.25) * step
    src = regular_grid(lat, lon_shift, name="shifted")
    dst = regular_grid(lat, np.arange(nlon, dtype=np.float64) * step, name="dst")
    w = lon_weights(src, dst)
    # 接缝在首格边界：源第 0 格 [−0.75°, 2.25°] 的 0.75 落在目标首列、0.25 落次列
    assert w[0, 0] == pytest.approx(0.75, abs=1e-14)
    assert w[1, 0] == pytest.approx(0.25, abs=1e-14)
    assert w[-1, 0] == 0.0
    assert np.allclose(w.sum(axis=1), 1.0, rtol=0.0, atol=1e-14)
    assert np.allclose(w.sum(axis=0), 1.0, rtol=0.0, atol=1e-14), "源格必须被完整分配"


# ---------------------------------------------------------------------------
# R4 真 land_mask 的陆地比例
# ---------------------------------------------------------------------------
def test_world_land_mask_fraction_conserved():
    from simulation.config import EnvFieldConfig
    from world.env_field import EnvField
    from world.light_and_temperature import LightAndTemperature

    rows, cols = 60, 120
    w = SphereWorld(rows, cols)
    lt = LightAndTemperature(w)
    env = EnvField(w, lt, EnvFieldConfig(enabled=True, c0b_enabled=True))
    mask_w = env.land_mask.reshape(rows, cols).astype(np.float64)
    wg = world_grid(w)
    atmo = _gauss_grid(32, 64)

    mask_a = RemapPair.build(wg, atmo).remap(mask_w)
    assert np.all(np.isfinite(mask_a))
    assert mask_a.min() >= -1e-12 and mask_a.max() <= 1.0 + 1e-12
    assert np.any((mask_a > 0.0) & (mask_a < 1.0)), "保守重映射应出现海岸分数格"

    frac_src = global_mean(mask_w, wg)
    frac_a = global_mean(mask_a, atmo)
    assert frac_a == pytest.approx(frac_src, abs=1e-12, rel=0.0)

    back = RemapPair.build(atmo, wg).remap(mask_a)
    assert global_mean(back, wg) == pytest.approx(frac_src, abs=1e-12, rel=0.0)


# ---------------------------------------------------------------------------
# R5 北→南纬序翻转
# ---------------------------------------------------------------------------
def test_descending_latitude_flips_exactly(world_small_grid):
    asc = _gauss_grid(32, 64)
    desc = grid_from_atmo_spec(_gauss_spec(32, 64, reverse=True))
    assert desc.flipped_from_descending is True
    assert np.array_equal(desc.lat_deg, asc.lat_deg)
    assert np.array_equal(desc.area, asc.area)

    rng = np.random.default_rng(7)
    f_asc = rng.normal(size=(32, 64))
    f_desc = f_asc[::-1].copy()          # 与"北→南"输入对齐
    out_asc = RemapPair.build(asc, world_small_grid).remap(f_asc)
    out_desc = RemapPair.build(desc, world_small_grid).remap(f_desc)
    assert np.array_equal(out_asc, out_desc), "纬序翻转必须逐位一致（禁静默南北镜像）"


# ---------------------------------------------------------------------------
# R6 与 A17 规则经纬权重对拍
# ---------------------------------------------------------------------------
def test_weights_match_a17_on_regular_grids():
    from atmo_bridge.remap import lat_weights as a17_lat
    from atmo_bridge.remap import lon_weights as a17_lon

    rng = np.random.default_rng(99)
    for nlat_s, nlat_d in ((20, 33), (32, 24)):
        lat_s = np.sort(rng.uniform(-86.0, 86.0, nlat_s))
        lat_d = np.sort(rng.uniform(-88.0, 88.0, nlat_d))
        lat_s[0], lat_s[-1] = -86.0, 86.0
        lat_d[0], lat_d[-1] = -88.0, 88.0
        lon_s = np.sort(rng.uniform(0.0, 360.0, 12))
        lon_d = np.sort(rng.uniform(0.0, 360.0, 17))
        gs = regular_grid(lat_s, lon_s, name="s")
        gd = regular_grid(lat_d, lon_d, name="d")
        assert np.allclose(lat_weights(gs, gd), a17_lat(lat_s, lat_d),
                           rtol=0.0, atol=1e-13)
        assert np.allclose(lon_weights(gs, gd), a17_lon(lon_s, lon_d),
                           rtol=0.0, atol=1e-13)


# ---------------------------------------------------------------------------
# R7 fail-loud
# ---------------------------------------------------------------------------
def test_invalid_inputs_fail_loud():
    lat = np.array([-80.0, 0.0, 80.0])
    lon = np.array([0.0, 120.0, 240.0])
    with pytest.raises(ValueError):
        Grid(name="bad", lat_deg=lat, lon_deg=lon, area=np.array([0.2, 0.2, 0.2]))
    with pytest.raises(ValueError):
        Grid(name="bad", lat_deg=lat[::-1], lon_deg=lon, area=np.array([0.2, 0.6, 0.2]))
    spec = _gauss_spec(32, 64)
    spec["lat_deg"] = spec["lat_deg"].copy()
    spec["lat_deg"][10] = spec["lat_deg"][9]
    with pytest.raises(ValueError):
        grid_from_atmo_spec(spec)
    g = _gauss_grid(32, 64)
    wg = world_grid(SphereWorld(60, 120))
    pair = RemapPair.build(g, wg)
    with pytest.raises(ValueError):
        pair.remap(np.ones((31, 64)))
    with pytest.raises(ValueError):
        mu_edges(np.array([0.5, 0.0, 0.5]))


# ---------------------------------------------------------------------------
# R8 真 480×960 + 3D
# ---------------------------------------------------------------------------
def test_world_s2_shapes_and_3d(world_s2):
    wg = world_grid(world_s2)
    atmo = _gauss_grid(32, 64)
    pair = RemapPair.build(atmo, wg)
    assert pair.wlat.shape == (480, 32)
    assert pair.wlon.shape == (960, 64)

    rng = np.random.default_rng(480)
    f3 = rng.normal(size=(8, 32, 64))
    out = pair.remap(f3)
    assert out.shape == (8, 480, 960) and out.dtype == np.float64
    assert np.all(np.isfinite(out))
    for k in range(8):
        assert global_mean(out[k], wg) == pytest.approx(
            global_mean(f3[k], atmo), abs=1e-12, rel=0.0)


# ===========================================================================
# A3.3 时钟 + 状态桥
# ===========================================================================
_TINY = (24, 48)
_ORBIT = {
    "year_s": 31536000.0, "rotation_period_s": 86400.0,
    "obliquity_rad": 0.0, "eccentricity": 0.0167,
    "equinox_offset_s": 0.0, "solar_constant_w_m2": 1361.0,
    "radius_m": 6371000.0,
}


@pytest.fixture(scope="module")
def world_tiny():
    return SphereWorld(*_TINY)


def _tiny_boundary(sst_offset: float = 0.0) -> dict:
    rows, cols = _TINY
    lat = np.degrees(np.linspace(-90.0, 90.0, rows + 1)[:-1] + 90.0 / rows)
    land = np.where(np.abs(lat) < 40.0, 1.0, 0.0)[:, None]
    mask = np.broadcast_to(land, (rows, cols)).copy()
    sst = (300.0 - 40.0 * np.sin(np.radians(lat)) ** 2)[:, None] + sst_offset
    return {
        "h": np.zeros((rows, cols)),
        "mask": mask,
        "alb": np.where(mask > 0.5, 0.17, 0.06) * np.ones((rows, cols)),
        "sst": np.broadcast_to(sst, (rows, cols)).copy(),
        "ocean": np.broadcast_to(1.0 - land, (rows, cols)).copy(),
        "seaice": np.zeros((rows, cols)),
    }


def _bridge(world, *, tick_s: float = 36.0, seed: int = 7,
            sst_offset: float = 0.0, nlat: int = 16, nlon: int = 32) -> CouplingBridge:
    cfg = AtmoConfig(enabled=True, weather_source="atmo",
                     nlat=nlat, nlon=nlon, seed=seed)
    core = SyntheticCore(cfg)
    return CouplingBridge(
        world, cfg, core, tick_s,
        boundary_fn=lambda: _tiny_boundary(sst_offset), orbit=_ORBIT)


@pytest.mark.parametrize(
    "tick_s,dt_atmo,k,dt_step,n_year",
    [(36.0, 900.0, 25, 900.0, 35040),     # v1.0.0–v1.0.2 旧默认
     (360.0, 900.0, 2, 720.0, 43800),     # dt 非 tick 整数倍
     (36.0, 288.0, 8, 288.0, 109500)])    # v1.0.3 默认 dt_atmo=288
def test_c1_clock_m2prime_zero_drift(tick_s, dt_atmo, k, dt_step, n_year):
    clk = CouplingClock(tick_s, dt_atmo)
    assert (clk.k, clk.dt_step, clk.n_sub) == (k, dt_step, 1)
    assert clk.dt_remainder == pytest.approx(dt_atmo - dt_step)
    for t in range(0, 10001):
        assert clk.time_s(t) == float((t // k) * dt_step), "time_s 必须整数倍、零漂移"
        assert clk.is_exchange(t) is (t % k == 0)
    assert clk.exchanges_per_year(31536000.0) == 31536000 // dt_step
    assert clk.exchanges_per_year(31536000.0) == n_year


def test_c2_clock_fail_loud():
    with pytest.raises(ValueError):
        CouplingClock(900.0, 600.0)                     # dt_atmo < tick
    with pytest.raises(NotImplementedError):
        CouplingClock(36.0, 900.0, step_mode="tick_split")
    with pytest.raises(ValueError):
        CouplingClock(36.0, 900.0, step_mode="bogus")
    clk = CouplingClock(36.0, 900.0)
    with pytest.raises(ValueError):
        clk.exchanges_per_year(1000.0)                  # 非 dt_step 整数倍
    with pytest.raises(ValueError):
        clk.is_exchange(-1)


def test_b1_exchange_phase_idempotent_and_no_rewind(world_tiny):
    br = _bridge(world_tiny)
    for t in range(0, 51):
        br.on_tick(t)
    assert br.exchanges == 3                            # t = 0, 25, 50
    assert br.last_tick == 50
    assert br.on_tick(50) is False                      # 同 tick 幂等
    assert br.on_tick(51) is False                      # 非交换相位
    with pytest.raises(ValueError):
        br.on_tick(49)                                  # 倒带
    with pytest.raises(ValueError):
        br.exchange(7)                                  # 非交换相位

    br2 = _bridge(world_tiny, tick_s=360.0)
    for t in range(0, 51):
        br2.on_tick(t)
    assert br2.exchanges == 26                          # k=2 ⇒ t = 0,2,...,50


def test_b2_same_seed_bitwise_and_seed_moves_precip(world_tiny):
    runs = []
    for seed in (7, 7, 8):
        br = _bridge(world_tiny, seed=seed)
        for t in range(0, 51):
            br.on_tick(t)
        runs.append(br)
    d = [r.core.digest(r.state) for r in runs]
    assert d[0] == d[1]
    assert np.array_equal(runs[0].world_t2m_c(), runs[1].world_t2m_c())
    assert np.array_equal(runs[0].world_precip_mm_yr(), runs[1].world_precip_mm_yr())
    assert not np.array_equal(runs[0].world_precip_mm_yr(), runs[2].world_precip_mm_yr())


def test_b3_state_in_place_and_export_cache_is_copy(world_tiny):
    br = _bridge(world_tiny)
    st = br.state
    d0 = br.core.digest(st)
    br.exchange(0)
    assert br.state is st, "OPEN-5：core.step 原地更新，桥不得换对象"
    assert br.core.digest(st) != d0
    e0 = br.last_world_exports["t2m_c"]
    v0 = e0.copy()
    br.exchange(25)
    assert np.array_equal(e0, v0), "旧出口缓存不得被下一步原地改写"
    assert br.last_world_exports["t2m_c"] is not e0


def test_b4_stub_raises_not_implemented(world_tiny):
    stub = AtmoCoreStub()
    with pytest.raises(NotImplementedError):
        stub.grid_spec()
    with pytest.raises(NotImplementedError):
        stub.make_state()
    with pytest.raises(NotImplementedError):
        stub.step({}, {}, 900.0)
    with pytest.raises(NotImplementedError):
        stub.export({})
    with pytest.raises(NotImplementedError):
        stub.digest({})
    cfg = AtmoConfig(enabled=True, weather_source="atmo", nlat=16, nlon=32)
    with pytest.raises(NotImplementedError):
        CouplingBridge(world_tiny, cfg, stub, 36.0,
                       boundary_fn=lambda: _tiny_boundary(), orbit=_ORBIT)


def test_b5_world_sst_enters_atmo_t2m(world_tiny):
    cold = _bridge(world_tiny, sst_offset=-20.0)
    warm = _bridge(world_tiny, sst_offset=+20.0)
    cold.exchange(0)
    warm.exchange(0)
    # 只有海洋格有权重 ⇒ 海温 ±20 K 必须让世界 t2m 朝同向移动
    assert float(warm.world_t2m_c().mean()) > float(cold.world_t2m_c().mean())
    assert float(warm.last_forcing_summary["sst_mean_k"]) > \
        float(cold.last_forcing_summary["sst_mean_k"]) + 5.0


# ===========================================================================
# A3.4 出口场 + 世界消费
# ===========================================================================
def _weather_world_cfg() -> SimConfig:
    cfg = SimConfig(seed=42)
    cfg.simulation.use_sim_core = False
    cfg.env_field.enabled = True
    cfg.env_field.c0b_enabled = True
    return cfg


def test_w1_atmo_config_default_off_and_fail_loud():
    cfg = AtmoConfig()
    assert cfg.enabled is False and cfg.weather_source == "off"
    assert cfg.manifest()["enabled"] is False
    with pytest.raises(ValueError):
        AtmoConfig(enabled=True, weather_source="off")
    with pytest.raises(ValueError):
        AtmoConfig(enabled=False, weather_source="atmo")
    with pytest.raises(NotImplementedError):
        AtmoConfig(nlev=12)


def test_w2_weather_mode_guards():
    good = _weather_world_cfg()
    g = validate_weather_config(good)
    assert g["P_ANCHOR_KG_M2_S"].startswith("not_active")
    assert g["c1_enabled"] is False and g["c0b_enabled"] is True

    bad = _weather_world_cfg()
    bad.atmosphere.enabled = True
    with pytest.raises(ValueError, match="P̄=990|_P_ANCHOR"):
        validate_weather_config(bad)

    bad = _weather_world_cfg()
    bad.env_field.c1_enabled = True
    with pytest.raises(ValueError, match="C1"):
        validate_weather_config(bad)

    bad = _weather_world_cfg()
    bad.env_field.c0b_enabled = False
    with pytest.raises(ValueError, match="c0b_enabled"):
        validate_weather_config(bad)

    bad = _weather_world_cfg()
    bad.env_field.water_sensitivity = 0.5
    with pytest.raises(NotImplementedError, match="water_sensitivity"):
        validate_weather_config(bad)

    bad = _weather_world_cfg()
    bad.simulation.use_sim_core = True
    with pytest.raises(NotImplementedError, match="use_sim_core"):
        validate_weather_config(bad)


def test_w3_w4_weather_engine_consumes_atmo_fields():
    cfg = _weather_world_cfg()
    acfg = AtmoConfig(enabled=True, weather_source="atmo",
                      nlat=16, nlon=32, seed=7)
    eng, src = make_weather_engine(cfg, acfg, SyntheticCore(acfg))

    # 构造期：三条口径全部来自 atmo，P̄ 锚与 C1 band 都不在
    assert eng.atmosphere is None
    assert eng.env_field.pr_source == "atmo"
    assert eng.env_field.band is None
    assert eng.climate is src and src.name == "atmo"
    assert eng.env_field.pr_proxy is not None, "bind 后应记录换源前 proxy"
    m = src.manifest()
    assert m["guards"]["world_atmosphere_enabled"] is False
    assert m["time"]["k_ticks"] == 25 and m["time"]["exchanges_per_year"] == 35040
    assert m["time_scale"]["tick_seconds"] == 36
    assert m["exports"]["precip_policy"] == "keep_total"

    for _ in range(30):
        src.on_tick(eng.tick + 1)
        eng.step()

    p = src.probe()
    assert p["bound"] and p["synthetic"] is True
    assert p["t2m_calls"] >= 30, "生物层每 tick 必须走 climate.temperature_c"
    assert p["bridge"]["exchanges"] == 2 and p["bridge"]["last_tick"] == 25
    assert p["precip_refresh_count"] >= 2
    assert p["pr_source"] == "atmo"

    flat = np.array([0, eng.world.n_cells - 1], dtype=np.int64)
    cached = src._bridge.world_t2m_c().ravel()[flat]
    assert np.array_equal(src.temperature_c(flat, eng.tick), cached)
    # t2m 确实替换了世界光照温度（不是同一条链）
    light_t = np.asarray(eng.light.temperature(flat, eng.tick), dtype=np.float64)
    assert not np.allclose(cached, light_t, rtol=0.0, atol=1e-9)

    diag = eng.env_field.pr_diag
    assert abs(float(diag["mean_before"]) - float(diag["mean_after"])) < 1e-5
    assert float(eng.env_field.pr.mean()) > 0.0


# ---------------------------------------------------------------------------
# P 真实扩展适配器（INTERFACE v1.0.2 §8.7）
# ---------------------------------------------------------------------------
class _FakeAtmoModule:
    INTERFACE_VERSION = "1.0.2"

    def __init__(self, *, with_export: bool = True) -> None:
        self.export_args = None
        if not with_export:
            self.export = None

    def grid_spec(self, nlat, nlon, nlev, kind):
        return {"nlat": nlat, "nlon": nlon, "nlev": nlev, "kind": kind,
                "lat_deg": np.zeros(nlat), "lon_deg": np.zeros(nlon),
                "area": np.full(nlat, 1.0 / nlat)}

    def make_state(self, g, T0, ps0):
        shape = (g["nlev"], g["nlat"], g["nlon"])
        return {"u": np.zeros(shape), "v": np.zeros(shape), "w": np.zeros(shape),
                "T": np.full(shape, T0), "q": np.zeros(shape),
                "ps": np.full((g["nlat"], g["nlon"]), ps0)}

    def step(self, state, forcing, dt):
        return {"dt": dt}

    def digest(self, state):
        return "0123456789abcdef"

    def export(self, state, grid=None):
        self.export_args = (state, grid)
        nlat, nlon = grid["nlat"], grid["nlon"]
        return {"t2m": np.full((nlat, nlon), 300.0),
                "u10": np.zeros((nlat, nlon)), "v10": np.zeros((nlat, nlon))}


def test_p1_module_adapter_export_v102_passes_grid():
    mod = _FakeAtmoModule()
    adapter = ModuleAtmoCore(mod, AtmoConfig())
    st = adapter.make_state()
    out = adapter.export(st)
    assert set(out) == {"t2m", "u10", "v10"}
    assert mod.export_args is not None
    assert mod.export_args[0] is st
    assert mod.export_args[1] is adapter.grid_spec(), \
        "v1.0.2 §8.7：export(st, g) 必须带网格，不许让核自己猜默认网格"


def test_p2_module_adapter_fail_loud():
    class BadVersion(_FakeAtmoModule):
        INTERFACE_VERSION = "9.9.9"

    with pytest.raises(ValueError, match="INTERFACE_VERSION"):
        ModuleAtmoCore(BadVersion(), AtmoConfig())

    adapter = ModuleAtmoCore(_FakeAtmoModule(with_export=False), AtmoConfig())
    with pytest.raises(NotImplementedError, match="export"):
        adapter.export(adapter.make_state())
