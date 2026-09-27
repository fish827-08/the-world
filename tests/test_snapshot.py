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


def _make_ars_config(seed=42):
    """14.9 ARS 开档（斑块臂；`use_sim_core=False` —— ARS ∧ sim_core 构造期硬错，H3 A1）。"""
    cfg = _make_config(seed=seed)
    cfg.simulation.use_sim_core = False
    c = cfg.resources
    c.bg_production_zero = True
    c.patch_count = 30
    c.patch_radius = 2
    c.patch_regrowth_mult = 1.195
    o = cfg.organisms
    o.max_energy = 600.0
    o.initial_energy = 300.0
    o.starve_frac = 0.30
    o.exhaust_frac = 0.17
    o.eat_efficiency = 7.5
    o.assim_herb = 0.4
    o.assim_carn = 0.8
    o.stomach_cap_mass = 25.0
    o.eat_threshold_frac = 0.6
    o.photo_max = 0.0
    cfg.ars.enabled = True
    cfg.ars.gain = 1.0
    cfg.ars.fast_tau = 5.0
    cfg.ars.slow_tau = 50.0
    return cfg


def _make_subpos_config(seed=42):
    """13.4 波 1 亚格连续坐标开档（`use_sim_core=False`：subpos ∧ Rust 路径构造期硬错，H3）。"""
    cfg = _make_config(seed=seed)
    cfg.simulation.use_sim_core = False
    cfg.subpos.enabled = True
    cfg.subpos.speed_max = 0.125
    cfg.subpos.speed_gain = 0.125
    cfg.subpos.subdiv = 80
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

    ARS_STATE = ("_out_taken", "_feed_fast", "_feed_slow",
                 "_ars_extensive", "_heading", "_giveup_ct")

    def test_ars_state_roundtrip_and_resume_bitwise(self):
        """R216 §四 裁定 A：14.9 ARS 六数组随快照**存 + 恢复** ⇒ 续跑逐位一致。

        修前：六数组不入快照键 ⇒ 恢复后停在构造期初值（长度 initial_count、状态丢失）
        ⇒ 续跑轨迹不逐位，且首次死亡压缩即 IndexError（R215-b）。
        长度普查见 `TestSnapshotIdLedgerLengths.test_all_slot_arrays_have_alive_length_after_load`。
        """
        cfg = _make_ars_config()
        e_cont = SphereEngine(cfg)
        e_snap = SphereEngine(cfg)
        for _ in range(60):
            e_cont.step()
            e_snap.step()
        P = len(e_cont._id)
        assert e_cont._feed_fast[:P].sum() > 0.0, "前提：ARS 期待状态须非平凡（否则测试退化）"

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            e_snap.save_snapshot(path)
            e_load = SphereEngine.load_snapshot(path, config=cfg)
            assert len(e_load._id) == P
            for name in self.ARS_STATE:
                a, b = getattr(e_cont, name)[:P], getattr(e_load, name)
                assert b.dtype == a.dtype, f"{name} dtype {b.dtype} != {a.dtype}"
                np.testing.assert_array_equal(a, b, err_msg=name)

            for _ in range(60):
                e_cont.step()
                e_load.step()
            P2 = len(e_cont._id)
            assert P2 == len(e_load._id)
            for name in self.ARS_STATE:
                np.testing.assert_array_equal(
                    getattr(e_cont, name)[:P2], getattr(e_load, name), err_msg=name)
            np.testing.assert_array_equal(e_cont._energy[:P2], e_load._energy[:P2])
            np.testing.assert_array_equal(e_cont._flat[:P2], e_load._flat[:P2])
        finally:
            os.unlink(path)

    def test_subpos_config_roundtrip_and_resume_bitwise(self):
        """R233 T-F：subpos 开档的「存档 → 续跑」必须逐位 == 连续跑。

        被守护的缺陷：`SimConfig.from_dict` 曾**整段漏传 `subpos`** ⇒ `load_snapshot(path)`
        （config=None ⇒ 走 from_dict 还原配置）把 subpos 静默退回默认（enabled=False，
        subdiv=4 / speed_max=2.0 / gain=4.0）⇒ 续跑**换掉运动模型**（非逐位且不报错）。
        ⚠️ 既有测试为何没抓住：其他往返测试要么显式 `config=cfg` 传入（跳过 from_dict），
        要么用 subpos 默认（关）档；本测试刻意走 **config=None + subpos 开档**。
        """
        cfg = _make_subpos_config()
        assert SimConfig.from_dict(cfg.to_dict()) == cfg, "from_dict 必须忠实回放 subpos 段"
        e_cont = SphereEngine(cfg)
        e_snap = SphereEngine(cfg)
        for _ in range(60):
            e_cont.step()
            e_snap.step()
        P = len(e_cont._id)
        assert P > 0 and e_cont._sub_r.min() != e_cont._sub_r.max(), \
            "前提：亚格坐标须非平凡（否则测试退化）"

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            e_snap.save_snapshot(path)
            e_load = SphereEngine.load_snapshot(path)      # config=None ⇒ from_dict 路径
            assert e_load.config.subpos.enabled is True
            assert e_load.config.subpos.subdiv == 80
            assert float(e_load.config.subpos.speed_gain) == 0.125
            assert e_load.config.fingerprint() == e_cont.config.fingerprint()
            np.testing.assert_array_equal(e_cont._sub_r[:P], e_load._sub_r[:P])
            np.testing.assert_array_equal(e_cont._sub_c[:P], e_load._sub_c[:P])

            for _ in range(60):
                e_cont.step()
                e_load.step()
            P2 = len(e_cont._id)
            assert P2 == len(e_load._id)
            np.testing.assert_array_equal(e_cont._flat[:P2], e_load._flat[:P2])
            np.testing.assert_array_equal(e_cont._energy[:P2], e_load._energy[:P2])
            np.testing.assert_array_equal(e_cont._sub_r[:P2], e_load._sub_r[:P2])
            np.testing.assert_array_equal(e_cont._sub_c[:P2], e_load._sub_c[:P2])
            np.testing.assert_array_equal(e_cont._id, e_load._id)
        finally:
            os.unlink(path)

    def test_ars_missing_keys_fall_back_to_init_lengths(self):
        """旧快照（无 ARS 六键）仍可载入：回退初值 + 长度按**恢复后存活数**建（S1 兼容承诺）。"""
        cfg = _make_ars_config()
        e = SphereEngine(cfg)
        for _ in range(30):
            e.step()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            e.save_snapshot(path)
            d = dict(np.load(path, allow_pickle=True))
            for k in ("out_taken", "feed_fast", "feed_slow",
                      "ars_extensive", "heading", "giveup_ct"):
                d.pop(k)
            np.savez_compressed(path, **d)

            e2 = SphereEngine.load_snapshot(path, config=cfg)
            P = len(e2._id)
            assert len(e2._out_taken) == P and e2._out_taken.dtype == np.float64
            assert len(e2._feed_fast) == P and e2._feed_fast.dtype == np.float64
            assert len(e2._feed_slow) == P and e2._feed_slow.dtype == np.float64
            assert len(e2._ars_extensive) == P and e2._ars_extensive.dtype == bool
            assert len(e2._heading) == P and e2._heading.dtype == np.int64
            assert len(e2._giveup_ct) == P and e2._giveup_ct.dtype == np.int64
            np.testing.assert_array_equal(e2._heading, np.full(P, -1))
            np.testing.assert_array_equal(e2._ars_extensive, np.ones(P, dtype=bool))
            np.testing.assert_array_equal(e2._giveup_ct, np.zeros(P, dtype=np.int64))
            e2.step()   # 载入即可续跑（旧快照的最低契约）
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

    def test_all_slot_arrays_have_alive_length_after_load(self):
        """全量普查：恢复后**所有**按个体数组长度必须 == 存活数。

        R215-b 发现 ARS 族（`_out_taken/_feed_fast/_feed_slow/_ars_extensive/_heading/
        `_giveup_ct`）未进快照键 ⇒ 恢复后长度停在构造期 initial_count（本测试当时红）。
        R216 §四 裁定 A（存 + 恢复）修完 ⇒ 转真测试（xfail 摘除，见提交说明）。
        """
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
