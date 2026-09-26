"""P2② 对拍：移动决策整块向量化（``SphereEngine._move_decide_batch``）。

参考实现 = 同文件的原逐个体循环（`_batch_move_on=False` 强制走它，逐位基线）。
覆盖：
* 整引擎锁步对拍（vanilla + subpos，每 tick 逐数组比较，含 RNG 流形状）；
* 极点个体的整带候选（`_pole_nb` 广播分支）与普通行/极点行混编；
* 平局分支（`rand_choice % len(nb)`）与 argmax 分支的**规格**断言；
* 记忆项 / 解读项的**行级条件**（没有的那些个体不得被 `+0.0` 扰动）；
* 门控：任一可选机制开启 ⇒ 快路径必须**不启用**（且基线确实启用，排除空过）。
"""

import numpy as np
import pytest

from simulation.config import SimConfig
from simulation.genes import Gene
from simulation.sphere_engine import SphereEngine

# 锁步比较的数组清单：覆盖位置/能量/基因/账本/信任/解读表/记忆/信号/资源
_KEYS = (
    "_flat", "_energy", "_stomach", "_genes", "_age", "_id", "_trust",
    "_interpret", "_work_memory", "_valence", "_arousal", "_expectation",
    "_out_taken", "_feed_fast", "_feed_slow", "_heading", "_giveup_ct",
)
_LEDGERS = ("_rs_children", "_rs_observed", "_rs_g15", "_emit_count")


def _make(seed=42, n=200, rows=60, cols=120, subpos=False):
    cfg = SimConfig(seed=seed)
    cfg.world.rows, cfg.world.cols = rows, cols
    cfg.simulation.use_sim_core = False
    cfg.population.initial_count = n
    if subpos:
        cfg.subpos.enabled = True
    return SphereEngine(cfg)


def _state(e):
    d = {k: np.array(getattr(e, k), copy=True) for k in _KEYS if hasattr(e, k)}
    for k in _LEDGERS:
        d[k] = np.array(getattr(e, k), copy=True)
    d["marks"] = np.array(e.signals._marks, copy=True)
    d["grid"] = np.array(e.resources._grid, copy=True)
    d["scalars"] = np.array([e._tick, e._next_id, e._mem_ptr, len(e._id)],
                            dtype=np.int64)
    return d


def _assert_same(a, b, tick):
    for k in a:
        va, vb = a[k], b[k]
        assert va.shape == vb.shape, f"t={tick} 形状差异 {k}: {va.shape} vs {vb.shape}"
        assert va.dtype == vb.dtype, f"t={tick} dtype 差异 {k}"
        assert np.array_equal(va, vb), (
            f"t={tick} 数值差异 {k}: "
            f"{(va != vb).sum()} 处（首处 {np.flatnonzero(va.ravel() != vb.ravel())[:3]}）"
        )


def _lockstep(make_kw, ticks, pin_energy=None, pin_cells=(), preseed=None):
    """同种子两引擎锁步（on/off），每 tick 逐数组比较；返回 (两引擎, 快路径调用数)。

    `pin_cells` = ((个体下标, 目标格) …)：把个体钉在指定格（用于极点用例）。
    `preseed(e, t)`：每 tick 步进前对**两个引擎**做同一注入（逼出稀有分支）。
    """
    e_on, e_off = _make(**make_kw), _make(**make_kw)
    e_off._batch_move_on = False
    for e in (e_on, e_off):
        for _i, _cell in pin_cells:
            e._flat[_i] = _cell
        if pin_cells and pin_energy is not None:
            e._energy[:] = np.maximum(e._energy, pin_energy)
    calls = [0]
    orig = SphereEngine._move_decide_batch

    def counting(self, *a, **k):
        calls[0] += 1
        return orig(self, *a, **k)

    SphereEngine._move_decide_batch = counting
    try:
        for t in range(1, ticks + 1):
            if preseed is not None:
                preseed(e_on, t)
                preseed(e_off, t)
            e_on.step()
            e_off.step()
            if pin_energy is not None:
                for e in (e_on, e_off):
                    e._energy[:] = np.maximum(e._energy, pin_energy)
            _assert_same(_state(e_on), _state(e_off), t)
    finally:
        SphereEngine._move_decide_batch = orig
    return e_on, e_off, calls[0]


# ---------------------------------------------------------------------------
# 整引擎锁步：逐位一致
# ---------------------------------------------------------------------------

def test_fast_path_matches_reference_vanilla_subpos():
    """T4 同款（Python 路径 + subpos）：120 tick 全状态逐位一致，且快路径确实启用。"""
    e_on, _e_off, calls = _lockstep(
        dict(seed=42, n=200, subpos=True), 120, pin_energy=50.0)
    assert calls > 0, "快路径一次都没调用 ⇒ 对拍空过"
    assert e_on.rng.draws > 0


def test_fast_path_matches_reference_poles_pinned():
    """极点个体（整带 cols 邻）：钉在极点跑 60 tick，逐位一致。"""
    south = 60 * 120 - 1          # 默认世界 60×120 的南极格
    _lockstep(dict(seed=7, n=80, subpos=False), 60, pin_energy=300.0,
              pin_cells=((0, 0), (1, south), (2, 0), (3, south)))


def test_fast_path_matches_reference_small_world():
    """小世界（含极点行与重复邻居的退化候选）：逐位一致。"""
    _lockstep(dict(seed=11, n=40, rows=6, cols=4, subpos=True), 30,
              pin_energy=300.0)


def test_fast_path_matches_reference_with_signals_and_memory():
    """每 tick 重撒标记 + 预写记忆 ⇒ 逼出解读项/记忆项分支，整引擎仍逐位一致。"""

    def preseed(e, t):
        e.signals._marks[::5] = np.uint8(1 + (t % 15))
        e._work_memory[:, t % 4] = e._flat

    e_on, _e_off, calls = _lockstep(
        dict(seed=13, n=150, subpos=True), 60, pin_energy=100.0, preseed=preseed)
    assert calls == 60, "快路径未逐 tick 启用 ⇒ 对拍空过"
    assert np.count_nonzero(e_on.signals._marks) > 0


def test_rng_stream_shape_unchanged():
    """快路径**不新增也不跳过**任何随机抽取（draws 与 bit_generator 状态同）。"""
    e_on, e_off = _make(seed=3, n=120, subpos=True), _make(seed=3, n=120, subpos=True)
    e_off._batch_move_on = False
    for _ in range(25):
        e_on.step()
        e_off.step()
    assert e_on.rng.draws == e_off.rng.draws
    s_on = e_on.rng.bit_generator.state["state"]
    s_off = e_off.rng.bit_generator.state["state"]
    assert s_on["state"] == s_off["state"]
    assert s_on["inc"] == s_off["inc"]


# ---------------------------------------------------------------------------
# 规格断言：直接调 `_move_decide_batch`（手工构造输入）
# ---------------------------------------------------------------------------

def _spec_engine(n=3, rows=6, cols=8):
    e = _make(seed=1, n=n, rows=rows, cols=cols)
    e._genes[:] = 0.0                      # perc=0 / soc=−1（除显式设定外）
    e._genes[:, Gene.SOCIABILITY] = 0.5     # soc=0
    return e


def _zeros(e):
    nc = e.world.n_cells
    return (np.zeros(nc, dtype=np.float64), np.zeros(nc, dtype=np.float64),
            np.zeros(nc, dtype=np.float64))


def test_batch_tie_break_uses_rand_choice():
    """平局（score 恒 0）⇒ 取 `nb[rand_choice % len(nb)]`，极点行走整带。"""
    e = _spec_engine()
    n_cells = e.world.n_cells
    c0 = e.world.cols + 5                 # 非极点格（row 1；row 0 是极点行）
    e._flat[:] = (0, n_cells - 1, c0)
    fr, sp, dn = _zeros(e)
    rc = np.array([3, 5, 1], dtype=np.int64)
    out = e._move_decide_batch(np.arange(3, dtype=np.int64), fr, sp, dn, rc, 0.0)
    exp = np.array([
        e._pole_nb[0][3],
        e._pole_nb[1][5],
        e._nb_table[c0][1],
    ], dtype=np.int64)
    assert np.array_equal(out, exp), f"{out} != {exp}"


def test_batch_argmax_picks_densest_neighbor():
    """非平局（群居项）⇒ 取 densities 最大的候选格（普通行）。"""
    e = _spec_engine()
    e._genes[:, Gene.SOCIABILITY] = 1.0     # soc = +1.0
    c0 = e.world.cols + 5
    e._flat[:] = c0
    fr, sp, dn = _zeros(e)
    nb = e._nb_table[c0]
    dn[nb[5]] = 2.0                          # 唯一最大
    out = e._move_decide_batch(np.arange(1, dtype=np.int64), fr, sp, dn,
                               np.array([0], dtype=np.int64), 0.0)
    assert out[0] == nb[5]


def test_batch_memory_term_is_row_conditional():
    """记忆项：有记忆的个体朝记忆格；无记忆（全 −1）的个体**不受 `+0.0` 扰动**。"""
    e = _spec_engine(n=2)
    e._genes[:, Gene.PERCEPTION] = 1.0
    c0 = e.world.cols + 5
    e._flat[:] = c0
    fr, sp, dn = _zeros(e)
    nb = e._nb_table[c0]
    e._work_memory[:] = -1
    e._work_memory[0, 0] = nb[7]             # 0 号个体记得 nb[7]
    rc = np.array([0, 0], dtype=np.int64)     # 平局时都会取 nb[0]
    out = e._move_decide_batch(np.arange(2, dtype=np.int64), fr, sp, dn, rc, 0.0)
    assert out[0] == nb[7], "记忆项未生效"
    assert out[1] == nb[0], "无记忆个体被扰动（应为平局 ⇒ rand_choice 分支）"


def test_batch_interp_term_only_on_marks_rows():
    """解读项：仅候选格**有标记**时相加；标记内容按 `_interpret[idx, s]` 取值。"""
    e = _spec_engine(n=2)
    e._genes[:, Gene.PERCEPTION] = 1.0
    c0, c1 = e.world.cols + 5, e.world.cols + 9
    e._flat[:] = (c0, c1)
    fr, sp, dn = _zeros(e)
    nb0, nb1 = e._nb_table[c0], e._nb_table[c1]
    e._interpret[0, 5] = 10.0                # 0 号个体对码 5 的解读 = +10
    e.signals._marks[nb0[2]] = 5
    rc = np.array([0, 6], dtype=np.int64)
    out = e._move_decide_batch(np.arange(2, dtype=np.int64), fr, sp, dn, rc, 0.0)
    assert out[0] == nb0[2], "解读项未生效"
    assert out[1] == nb1[6 % 8], "无标记个体被扰动"


# ---------------------------------------------------------------------------
# 门控：可选机制开启时快路径必须让路
# ---------------------------------------------------------------------------

def _count_batch_calls(engine, ticks=5):
    calls = [0]
    orig = SphereEngine._move_decide_batch

    def counting(self, *a, **k):
        calls[0] += 1
        return orig(self, *a, **k)

    SphereEngine._move_decide_batch = counting
    try:
        for _ in range(ticks):
            engine.step()
            engine._energy[:] = np.maximum(engine._energy, 300.0)
    finally:
        SphereEngine._move_decide_batch = orig
    return calls[0]


def test_gate_off_when_optional_mechanisms_on():
    """逐机制：开启后快路径必须停用；同时基线（同种子 vanilla）必须启用（防空过）。"""

    def setup(c):
        return c

    def set_wound(c):
        c.corpse_wound.wound_enabled = True      # w_fear_health 默认 0.5 > 0
        return c

    def set_mig(c):
        c.migration.enabled = True
        c.light.tilt_rad = 0.4
        c.light.season_period = 6000
        return c

    def set_d2_asym(c):
        c.info_structure.enabled = True
        c.info_structure.perception_radius = 4
        return c

    def set_mem_grad(c):
        c.info_structure.memory_gradient = "orientation"
        return c

    cases = {
        "span2": lambda c: (setattr(c.simulation, "perception_span", 2), c)[1],
        "cap": lambda c: (setattr(c.simulation, "cell_occupancy_cap_enabled", True), c)[1],
        "l2_dash": lambda c: (setattr(c.simulation, "l2_dash", True), c)[1],
        "ars": lambda c: (setattr(c.ars, "enabled", True), c)[1],
        "wound": set_wound,
        "migration": set_mig,
        "d2_asym": set_d2_asym,
        "mem_grad": set_mem_grad,
    }

    base = _make(seed=5, n=60)
    assert _count_batch_calls(base) > 0, "基线未启用快路径 ⇒ 门控测试空过"

    for name, mutate in cases.items():
        cfg = SimConfig(seed=5)
        cfg.world.rows, cfg.world.cols = 60, 120
        cfg.simulation.use_sim_core = False
        cfg.population.initial_count = 60
        mutate(cfg)
        n = _count_batch_calls(SphereEngine(cfg))
        assert n == 0, f"{name} 开启后快路径仍在跑（{n} 次）⇒ 半接线"


def test_gate_off_when_switch_disabled():
    """`_batch_move_on=False` ⇒ 强制参考循环（对拍/调试开关本身有效）。"""
    e = _make(seed=5, n=60)
    e._batch_move_on = False
    assert _count_batch_calls(e) == 0
