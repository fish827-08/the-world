"""快照机制测试（v2，适配 L10a + 统计数组）。

核心：save_snapshot → load_snapshot → 继续跑，结果与"不保存连续跑"逐位一致。
覆盖：基本保存/恢复、续跑逐位一致、版本/配置/gene_count 校验、L10a 数组、RNG 状态。
"""
import os
import tempfile

import numpy as np
import pytest

from simulation.config import SimConfig
from simulation.sphere_engine import SphereEngine
from simulation.genes import Gene


def _make_config(seed=42, fruit_enabled=False):
    cfg = SimConfig(seed=seed)
    cfg.world.rows = 60
    cfg.world.cols = 120
    cfg.population.initial_count = 100
    cfg.resources.distribution = "patchy"
    cfg.simulation.use_sim_core = True
    cfg.fruit.enabled = fruit_enabled
    return cfg


class TestSnapshotBasic:
    """基本保存/恢复功能。"""

    def test_save_and_load_basic(self):
        cfg = _make_config()
        e1 = SphereEngine(cfg)
        for _ in range(200):
            e1.step()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            e1.save_snapshot(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

            e2 = SphereEngine.load_snapshot(path)
            P = len(e1._id)
            assert P == len(e2._id)
            assert e1._tick == e2._tick
            assert e1._next_id == e2._next_id
            assert e1._max_generation == e2._max_generation
            np.testing.assert_array_equal(e1._id[:P], e2._id[:P])
            np.testing.assert_array_equal(e1._flat[:P], e2._flat[:P])
            np.testing.assert_array_equal(e1._energy[:P], e2._energy[:P])
            np.testing.assert_array_equal(e1._genes[:P], e2._genes[:P])
        finally:
            os.unlink(path)

    def test_l10a_arrays_saved(self):
        """L10a 果实场数组必须保存/恢复。"""
        cfg = _make_config(fruit_enabled=True)
        e1 = SphereEngine(cfg)
        for _ in range(500):
            e1.step()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            e1.save_snapshot(path)
            e2 = SphereEngine.load_snapshot(path)
            np.testing.assert_array_equal(e1._fruit_grid, e2._fruit_grid)
            P = len(e1._id)
            np.testing.assert_array_equal(e1._fruit_charge[:P], e2._fruit_charge[:P])
            np.testing.assert_array_equal(e1._seed_carried[:P], e2._seed_carried[:P])
        finally:
            os.unlink(path)

    def test_run_stats_saved(self):
        """运行统计（_run_born/_run_died/_run_deaths）必须保存/恢复。"""
        cfg = _make_config()
        e1 = SphereEngine(cfg)
        for _ in range(500):
            e1.step()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            e1.save_snapshot(path)
            e2 = SphereEngine.load_snapshot(path)
            assert e1._run_born == e2._run_born
            assert e1._run_died == e2._run_died
            assert dict(e1._run_deaths) == dict(e2._run_deaths)
        finally:
            os.unlink(path)

    def test_rng_state_saved(self):
        """RNG 状态必须保存/恢复（可复现的关键）。"""
        cfg = _make_config()
        e1 = SphereEngine(cfg)
        for _ in range(100):
            e1.step()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            e1.save_snapshot(path)
            e2 = SphereEngine.load_snapshot(path)
            assert e1.rng.bit_generator.state == e2.rng.bit_generator.state
        finally:
            os.unlink(path)

    def test_resource_state_saved(self):
        """资源场状态必须保存/恢复。"""
        cfg = _make_config()
        e1 = SphereEngine(cfg)
        for _ in range(300):
            e1.step()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            e1.save_snapshot(path)
            e2 = SphereEngine.load_snapshot(path)
            np.testing.assert_array_equal(e1.resources._grid, e2.resources._grid)
            np.testing.assert_array_equal(e1.resources._capacity, e2.resources._capacity)
            np.testing.assert_array_equal(e1.resources._patch_mask, e2.resources._patch_mask)
        finally:
            os.unlink(path)


class TestSnapshotReproducibility:
    """核心：恢复后续跑与不保存连续跑逐位一致。"""

    def test_continue_after_load_bitwise_equal(self):
        """保存→恢复→跑100tick，与不保存连续跑100tick逐位一致。"""
        cfg = _make_config()
        e_continuous = SphereEngine(cfg)
        e_snapshot = SphereEngine(cfg)

        # 两个引擎用相同配置，前 300 tick 应该完全一致
        for _ in range(300):
            e_continuous.step()
            e_snapshot.step()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            e_snapshot.save_snapshot(path)
            e_loaded = SphereEngine.load_snapshot(path)

            # 继续跑 100 tick
            for _ in range(100):
                e_continuous.step()
                e_loaded.step()

            P = len(e_continuous._id)
            assert P == len(e_loaded._id)
            np.testing.assert_array_equal(e_continuous._energy[:P], e_loaded._energy[:P])
            np.testing.assert_array_equal(e_continuous._flat[:P], e_loaded._flat[:P])
            np.testing.assert_array_equal(e_continuous._genes[:P], e_loaded._genes[:P])
            np.testing.assert_array_equal(e_continuous._age[:P], e_loaded._age[:P])
            np.testing.assert_array_equal(e_continuous._valence[:P], e_loaded._valence[:P])
            np.testing.assert_array_equal(e_continuous._trust[:P], e_loaded._trust[:P])
        finally:
            os.unlink(path)

    def test_continue_with_l10a_bitwise_equal(self):
        """L10a 开启时，恢复后续跑逐位一致。"""
        cfg = _make_config(fruit_enabled=True)
        e_continuous = SphereEngine(cfg)
        e_snapshot = SphereEngine(cfg)

        for _ in range(300):
            e_continuous.step()
            e_snapshot.step()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            e_snapshot.save_snapshot(path)
            e_loaded = SphereEngine.load_snapshot(path)

            for _ in range(100):
                e_continuous.step()
                e_loaded.step()

            P = len(e_continuous._id)
            np.testing.assert_array_equal(e_continuous._energy[:P], e_loaded._energy[:P])
            np.testing.assert_array_equal(e_continuous._fruit_grid, e_loaded._fruit_grid)
            np.testing.assert_array_equal(e_continuous._fruit_charge[:P], e_loaded._fruit_charge[:P])
        finally:
            os.unlink(path)

    def test_multiple_save_load_cycles(self):
        """多次保存/恢复循环后仍逐位一致。"""
        cfg = _make_config()
        e_continuous = SphereEngine(cfg)
        e_loaded = SphereEngine(cfg)

        for _ in range(100):
            e_continuous.step()
            e_loaded.step()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            # 3 次保存/恢复循环，每次循环后跑 50 tick
            for cycle in range(3):
                e_loaded.save_snapshot(path)
                e_loaded = SphereEngine.load_snapshot(path)
                for _ in range(50):
                    e_continuous.step()
                    e_loaded.step()

            P = len(e_continuous._id)
            np.testing.assert_array_equal(e_continuous._energy[:P], e_loaded._energy[:P])
            np.testing.assert_array_equal(e_continuous._flat[:P], e_loaded._flat[:P])
        finally:
            os.unlink(path)


class TestSnapshotValidation:
    """版本/配置/gene_count 校验。"""

    def test_version_mismatch_raises(self):
        """快照版本不兼容时必须报错。"""
        cfg = _make_config()
        e1 = SphereEngine(cfg)
        e1.step()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            e1.save_snapshot(path)
            # 手动修改版本号
            data = dict(np.load(path, allow_pickle=True))
            data["snapshot_version"] = np.array(999)
            np.savez_compressed(path, **data)

            with pytest.raises(ValueError, match="版本"):
                SphereEngine.load_snapshot(path)
        finally:
            os.unlink(path)

    def test_config_fingerprint_mismatch_raises(self):
        """提供 config 时，指纹不匹配必须报错。"""
        cfg1 = _make_config(seed=42)
        cfg2 = _make_config(seed=999)  # 不同 seed → 不同指纹
        e1 = SphereEngine(cfg1)
        e1.step()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            e1.save_snapshot(path)
            with pytest.raises(ValueError, match="指纹"):
                SphereEngine.load_snapshot(path, config=cfg2)
        finally:
            os.unlink(path)

    def test_load_without_config_recovers_config(self):
        """不提供 config 时，从快照中恢复配置。"""
        cfg = _make_config(seed=42)
        e1 = SphereEngine(cfg)
        for _ in range(50):
            e1.step()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            e1.save_snapshot(path)
            e2 = SphereEngine.load_snapshot(path)  # config=None
            assert e2.config.seed == 42
            assert e2._tick == e1._tick
        finally:
            os.unlink(path)

    def test_snapshot_file_is_compressed(self):
        """快照文件使用 npz 压缩，大小应合理。"""
        cfg = _make_config()
        cfg.population.initial_count = 500
        e1 = SphereEngine(cfg)
        for _ in range(100):
            e1.step()

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            e1.save_snapshot(path)
            size = os.path.getsize(path)
            # N=500, expectation (500,120) = 60000 floats = 480KB，压缩后应 < 1MB
            assert size < 5 * 1024 * 1024, f"快照过大: {size} bytes"
            assert size > 1024, f"快照过小: {size} bytes"
        finally:
            os.unlink(path)


class TestSnapshotIdLedgerLengths:
    """R215 §七-1 守卫：恢复后 _id 键控终身账本长度必须 == `_next_id`。

    被守护缺陷（修复前）：`load_snapshot` 把"账本补齐"放在 `_next_id` 赋值**之前** ⇒
    补齐长度按**构造期**的 `_next_id`（= initial_count，`__init__` 建的初始种群）算：

    * 旧快照 / 默认档（无探针键，`measure_resp` 与 `oracle` 均关）⇒
      `need = initial_count − initial_count = 0` ⇒ 12 个账本长度**停在 initial_count**，
      而 `_next_id` 已是快照值 ⇒ 续跑首次由"第二世代"个体繁衍时
      `self._rs_children[self._id[ri]] += 1` 越界（IndexError）；
      开启 measure/oracle 时 `self._rs_observed[alive_ids]` 同样越界。
    * 缺 delta 键的新快照 ⇒ 兜底数组按 stale 长度建，与其余账本不等长。

    ⇒ 直接打的是**默认档的快照续跑**（14.10 长跑复活流程）。
    修法 = R215 §七-1 的"补齐改到赋值之后"（等价：元数据整块前移）。
    """

    # 全部 _id 键控账本（`_grow_id_arrays` 的 12 个 + 3 个 delta 兜底）
    LEDGERS = (
        "_rs_children", "_rs_observed", "_rs_g15", "_rs_age", "_rs_energy",
        "_rs_cc0", "_rs_cohort", "_emit_count", "_oracle_gain",
        "_delta_full_by_id", "_delta_content_by_id", "_delta_tick_by_id",
    )

    @staticmethod
    def _rounded(cfg, ticks):
        e = SphereEngine(cfg)
        for _ in range(ticks):
            e.step()
        return e

    def _save_load(self, e, cfg=None):
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            e.save_snapshot(path)
            return SphereEngine.load_snapshot(path, config=cfg)
        finally:
            os.unlink(path)

    def test_no_probe_keys_grows_ledgers_to_snapshot_next_id(self):
        """无探针键（默认档）：补齐长度用**快照** next_id，不是构造期 initial_count。"""
        cfg = _make_config()
        e1 = self._rounded(cfg, 5)
        assert e1._next_id == cfg.population.initial_count
        # 等价于"已出生 20 个"的合法状态：真实出生路径同序（先 _grow_id_arrays 再 concat id）
        e1._grow_id_arrays(20)
        e1._next_id += 20

        e2 = self._save_load(e1, cfg)
        assert e2._next_id == e1._next_id == cfg.population.initial_count + 20
        for name in self.LEDGERS:
            assert len(getattr(e2, name)) == e2._next_id, (
                f"{name} 长度 {len(getattr(e2, name))} != _next_id {e2._next_id}"
                "（R215 §七-1：补齐须在 _next_id 赋值之后）")

    def test_real_births_ledgers_match_next_id_after_load(self):
        """真路径：多世代跑 → 存 → 载 ⇒ 账本长度 == 快照 `_next_id`（守卫本体）。"""
        cfg = _make_config()
        cfg.organisms.maturity_fraction = 1e-6   # 一出生即成熟 ⇒ 快速到多世代
        e1 = self._rounded(cfg, 150)
        assert e1._next_id > len(e1._id), "前提：须已发生出生（否则本测试退化）"

        e2 = self._save_load(e1, cfg)
        assert e2._next_id == e1._next_id
        for name in self.LEDGERS:
            assert len(getattr(e2, name)) == e2._next_id, (
                f"{name} 长度 {len(getattr(e2, name))} != _next_id {e2._next_id}"
                "（R215 §七-1：补齐须在 _next_id 赋值之后）")
        # 出生路径的索引形态（`self._rs_children[self._id[ri]] += 1`）在恢复后不得越界
        e2._rs_children[e2._next_id - 1] += 1
        assert e2._rs_children[e2._next_id - 1] == 1

    @pytest.mark.xfail(strict=True, reason=(
        "R215-b（修 §七-1 时新发现，待裁）：14.9 ARS 六数组（_out_taken/_feed_fast/"
        "_feed_slow/_ars_extensive/_heading/_giveup_ct）**未进快照键** ⇒ 恢复后长度停在构造期 "
        "initial_count，首次死亡压缩即 IndexError（且即便等长也丢 ARS 状态 ⇒ 续跑轨迹不逐位）。"
        "修法待裁（保存恢复 vs 载入重置）；修好后请删本 xfail。"))
    def test_all_slot_arrays_have_alive_length_after_load(self):
        """全量普查：恢复后**所有**按个体数组长度必须 == 存活数（本批新发现的 ARS 族会红）。"""
        cfg = _make_config()
        cfg.organisms.maturity_fraction = 1e-6
        e1 = self._rounded(cfg, 150)
        P = len(e1._id)
        assert e1._next_id > P

        e2 = self._save_load(e1, cfg)
        assert len(e2._id) == P
        skip = set(self.LEDGERS)          # _id 键控：长度 = _next_id（另一套契约）
        bad = []
        for name in sorted(set(SphereEngine.__slots__)):
            if name in skip or name.startswith("__"):
                continue
            v1 = getattr(e1, name, None)
            if not isinstance(v1, np.ndarray) or v1.ndim < 1 or v1.shape[0] != P:
                continue                  # 非按个体数组（格域 (n_cells,) 等）不查
            v2 = getattr(e2, name, None)
            if not isinstance(v2, np.ndarray) or v2.shape[0] != P:
                bad.append((name, None if not isinstance(v2, np.ndarray) else v2.shape[0]))
        assert not bad, f"恢复后按个体数组长度 != 存活数 {P}：{bad}"
