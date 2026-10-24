"""B7：真实世界字段适配测试（只读导入 world/simulation；不改它们）。

- 小引擎（60×120，env+C0b 开，ticks=0）构造真实对象 ⇒
  `adapt_real_world` 全字段适配 + 不变量；
- fail-loud：缺 EnvField / 缺 surface_type ⇒ NotImplementedError（不静默退化）；
- 合成小世界不变量（形状/斑块/资源初值/季节相位）；
- 真实引擎季节相位读数（tilt 0.41 / 周期 400）。
"""
import numpy as np
import pytest

from biosphere.world_adapter import (
    WorldView, adapt_real_resource, adapt_real_world, synthetic_world,
)

TIER = "mainline"

#: 小引擎参数（测试用；s2 口径见 world_smoke CLI）
ROWS, COLS, PATCHES = 60, 120, 10


@pytest.fixture(scope="module")
def real_engine():
    from biosphere.world_smoke import build_real_engine
    return build_real_engine(rows=ROWS, cols=COLS, patches=PATCHES,
                             seed=42, ticks=0, season=True)


@pytest.fixture(scope="module")
def real_view(real_engine) -> WorldView:
    return adapt_real_world(real_engine.world, real_engine.env_field,
                            real_engine.resources, real_engine.light)


class TestAdaptRealWorld:
    def test_shapes(self, real_view):
        n = ROWS * COLS
        assert real_view.n_cells == n
        assert real_view.rows == ROWS and real_view.cols == COLS
        for name in ("surface_type", "elevation_m", "depth_m", "patch_id"):
            assert len(getattr(real_view, name)) == n
        assert len(real_view.neighbors) == n

    def test_depth_water_invariant(self, real_view):
        water = real_view.water_mask()
        assert water.any() and (~water).any(), "小引擎应同时含陆海"
        assert bool(np.all(real_view.depth_m >= 0.0))
        assert bool(np.all(real_view.depth_m[~water] == 0.0))
        deep = float(np.mean(real_view.depth_m[water] > 0.0))
        assert deep > 0.99

    def test_land_elevation_positive(self, real_view):
        water = real_view.water_mask()
        assert bool(np.all(real_view.elevation_m[~water] > 0.0))

    def test_patch_labels(self, real_view):
        pid = real_view.patch_id
        assert int(pid.min()) >= -1
        n_patches = int(pid.max()) + 1
        assert n_patches >= 1
        assert int((pid >= 0).sum()) > 0

    def test_season_phase(self, real_view):
        assert real_view.tilt_rad > 0.0
        assert real_view.season_period == 400
        assert real_view.season_phase(0) == 0.0
        assert real_view.season_phase(100) == pytest.approx(0.25)
        assert real_view.season_phase(400) == pytest.approx(0.0)

    def test_fail_loud_missing_env_field(self, real_engine):
        with pytest.raises(NotImplementedError):
            adapt_real_world(real_engine.world)

    def test_fail_loud_missing_surface_type(self, real_engine):
        class _Stub:
            pass  # 无 surface_type/h_m/sea_level_m
        with pytest.raises(NotImplementedError):
            adapt_real_world(real_engine.world, env_field=_Stub())


class TestAdaptRealResource:
    """B17.3：真实资源场 → SyntheticResource 只读快照。"""

    def test_snapshot_matches_engine(self, real_engine):
        res = adapt_real_resource(real_engine.resources)
        assert np.array_equal(res.amount, real_engine.resources._grid)
        assert np.array_equal(res.capacity, real_engine.resources._capacity)
        assert res.regrow_rate == pytest.approx(
            real_engine.resources.regrowth_rate)

    def test_snapshot_is_copy(self, real_engine):
        res = adapt_real_resource(real_engine.resources)
        before = float(real_engine.resources._grid[0])
        res.amount[0] += 123.0
        assert float(real_engine.resources._grid[0]) == before

    def test_fail_loud_missing_fields(self):
        with pytest.raises(NotImplementedError):
            adapt_real_resource(object())


class TestSimulatorInjection:
    """B17.3：world/resource 注入与合成默认路径逐位一致。"""

    @staticmethod
    def _cfg():
        from biosphere.simulator import BioMVPConfig
        return BioMVPConfig(rows=8, cols=16, seed=11, ticks=12,
                            init_pop=10, patch_count=2, patch_radius=1)

    @staticmethod
    def _world_of(cfg):
        return synthetic_world(
            rows=cfg.rows, cols=cfg.cols, seed=cfg.seed,
            land_frac=cfg.land_frac, patch_count=cfg.patch_count,
            patch_radius=cfg.patch_radius,
            patch_capacity=cfg.patch_capacity, bg_capacity=cfg.bg_capacity,
            regrow_rate=cfg.regrow_rate, tilt_rad=cfg.tilt_rad,
            season_period=cfg.season_period)

    def test_injection_matches_default_path(self):
        from biosphere.simulator import BiosphereSimulator
        cfg = self._cfg()
        view, res = self._world_of(cfg)
        base = BiosphereSimulator(cfg)
        inj = BiosphereSimulator(cfg, world=view, resource=res)
        base.run()
        inj.run()
        assert len(base.log.frames) == len(inj.log.frames) == cfg.ticks
        for fa, fb in zip(base.log.frames, inj.log.frames):
            assert np.array_equal(fa.uids, fb.uids)
            assert np.array_equal(fa.cells, fb.cells)
            assert np.array_equal(fa.energies, fb.energies)
        assert base.log.deaths == inj.log.deaths
        assert base.log.births == inj.log.births

    def test_world_requires_resource(self):
        from biosphere.simulator import BiosphereSimulator
        cfg = self._cfg()
        view, _res = self._world_of(cfg)
        with pytest.raises(ValueError):
            BiosphereSimulator(cfg, world=view)

    def test_size_mismatch_fails_loud(self):
        from biosphere.simulator import BiosphereSimulator
        cfg = self._cfg()
        view, res = synthetic_world(rows=16, cols=16, seed=1,
                                    patch_count=2, patch_radius=1)
        with pytest.raises(ValueError):
            BiosphereSimulator(cfg, world=view, resource=res)


class TestSyntheticWorld:
    def test_shapes_and_initial_resource(self):
        view, res = synthetic_world(rows=16, cols=32, seed=1)
        n = 16 * 32
        assert view.n_cells == n
        assert len(view.neighbors) == n
        assert res.amount.shape == (n,)
        assert res.capacity.shape == (n,)
        assert np.array_equal(res.amount, res.capacity)  # 初值 = 容量

    def test_patches_on_land_only(self):
        view, _res = synthetic_world(rows=16, cols=32, seed=2)
        land = view.surface_type == 0
        on_patch = view.patch_id >= 0
        assert on_patch.any()
        assert bool(np.all(on_patch <= land)), "斑块只能落在陆地"

    def test_water_depth_consistency(self):
        view, _res = synthetic_world(rows=16, cols=32, seed=3)
        water = view.water_mask()
        assert bool(np.all(view.depth_m[water] > 0.0))
        assert bool(np.all(view.depth_m[~water] == 0.0))

    def test_season_phase_default_on(self):
        view, _res = synthetic_world(rows=16, cols=32, seed=4)
        assert view.season_period > 1
        assert view.season_phase(0) == 0.0
        assert view.season_phase(view.season_period // 2) == pytest.approx(0.5)

    def test_determinism(self):
        v1, r1 = synthetic_world(rows=16, cols=32, seed=5)
        v2, r2 = synthetic_world(rows=16, cols=32, seed=5)
        assert np.array_equal(v1.surface_type, v2.surface_type)
        assert np.array_equal(v1.elevation_m, v2.elevation_m)
        assert np.array_equal(v1.patch_id, v2.patch_id)
        assert np.array_equal(r1.amount, r2.amount)
