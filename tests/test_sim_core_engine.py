"""引擎级对拍：use_sim_core 开关（模块三 · 3.3 接入）。

同一配置、同一种子，一个引擎走纯 Python 数值管线，另一个把
第 1~3 步 / 第 5~8 步下沉到 Rust（sim_core.step_vectors_stage1/2）。

判定：逐 tick 的 TickStats 逐位相等（float 字段 = 位级比较），
且最终种群内部状态（能量/胃/年龄/基因/位置/冷却）也逐位相等。
这证明 Rust 接入后 RNG 消费顺序不变、数值精算一致——只是换个更快的算盘。
"""
import numpy as np
import pytest

from simulation.config import SimConfig
from simulation.sphere_engine import SphereEngine


def small_cfg(seed):
    c = SimConfig(seed=seed)
    c.world.rows = 8
    c.world.cols = 12
    c.light.rotation_period = 600          # 昼夜短 → 成熟/老死在小 tick 数内出现
    c.population.initial_count = 60
    c.simulation.ticks = 300
    return c


def make_pair(seed):
    py_cfg = small_cfg(seed)
    rs_cfg = small_cfg(seed)
    rs_cfg.simulation.use_sim_core = True
    return SphereEngine(py_cfg), SphereEngine(rs_cfg)


@pytest.mark.parametrize("seed", [3, 7, 42])
def test_engine_tickstats_bitwise_equal(seed):
    """逐 tick：Rust 引擎与 Python 引擎的 TickStats 完全相等（含死亡与出生）。"""
    py_e, rs_e = make_pair(seed)
    n_death_events = 0
    n_birth_events = 0
    for _ in range(500):
        s_py = py_e.step()
        s_rs = rs_e.step()
        assert s_rs == s_py, (
            f"tick {s_rs.tick} 不一致: py={s_py} rs={s_rs}"
        )
        n_birth_events += int(s_py.born > 0)
        n_death_events += int(s_py.died > 0)
    # 场景确实发生了死亡与出生（否则守不到"热"路径）
    assert n_death_events > 0
    assert n_birth_events > 0


def test_engine_internal_state_bitwise_equal():
    """跑满后种群内部数组逐位相等（能量/胃/年龄/基因/冷却/世代/位置）。"""
    py_e, rs_e = make_pair(7)
    for _ in range(500):
        py_e.step()
        rs_e.step()
    assert py_e.alive_count() == rs_e.alive_count()
    fields = [
        "_flat", "_energy", "_stomach", "_genes", "_age",
        "_generation", "_parent", "_repro_cooldown", "_id",
    ]
    for f in fields:
        a, b = getattr(py_e, f), getattr(rs_e, f)
        assert a.shape == b.shape, f"{f}: 形状 {a.shape} != {b.shape}"
        np.testing.assert_array_equal(b, a, err_msg=f"{f} 逐位不相等")
    # regrow 路径：资源网格也逐位一致（3.4 接入后 Rust 侧同步再生）
    np.testing.assert_array_equal(
        rs_e.resources._grid, py_e.resources._grid,
        err_msg="resources._grid 逐位不相等",
    )


# ---- L7e：patchy 模式引擎级双路径对拍 ------------------------------------

def make_patchy_pair(seed):
    """同 seed 的 patchy 引擎对：Python 再生 vs Rust regrow_patchy。"""
    py_cfg = small_cfg(seed)
    rs_cfg = small_cfg(seed)
    py_cfg.resources.distribution = "patchy"
    rs_cfg.resources.distribution = "patchy"
    # 小网格（8×12=96 格）：斑块数/倍率需保守，否则背景容量为负
    py_cfg.resources.patch_count = 3
    rs_cfg.resources.patch_count = 3
    py_cfg.resources.patch_radius = 1
    rs_cfg.resources.patch_radius = 1
    py_cfg.resources.patch_capacity_mult = 2.0
    rs_cfg.resources.patch_capacity_mult = 2.0
    rs_cfg.simulation.use_sim_core = True
    return SphereEngine(py_cfg), SphereEngine(rs_cfg)


@pytest.mark.parametrize("seed", [3, 7])
def test_patchy_engine_tickstats_bitwise_equal(seed):
    """patchy：逐 tick TickStats 逐位相等（Rust regrow_patchy 接入不改变行为）。"""
    py_e, rs_e = make_patchy_pair(seed)
    for _ in range(300):
        s_py = py_e.step()
        s_rs = rs_e.step()
        assert s_rs == s_py, (
            f"tick {s_rs.tick} 不一致: py={s_py} rs={s_rs}"
        )
    # 资源网格走的是 Rust regrow_patchy，应逐位一致
    np.testing.assert_array_equal(
        rs_e.resources._grid, py_e.resources._grid,
        err_msg="patchy 下 resources._grid 逐位不相等",
    )