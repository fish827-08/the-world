"""L1 斑块资源（守恒版）的回归测试。

覆盖：
- 默认 uniform 行为与旧版完全一致（零回归）
- 容量守恒：patchy 模式 Σ_capacity == uniform 模式 Σ_capacity
- 再生守恒：patchy 模式总再生量 == uniform 模式总再生量（同温度）
- 空间非均匀：patch 格容量/食物 > 背景格
- 可复现：同 patch_seed → 同 patch_mask
- 引擎集成：patchy 配置跑 tick 不崩、种群存活
- 边界：倍率过大导致背景容量为负时抛 ValueError
"""
import numpy as np
import pytest

from simulation.config import SimConfig
from simulation.sphere_engine import SphereEngine
from world.light_and_temperature import LightAndTemperature
from world.resource_field import ResourceField
from world.sphere_world import SphereWorld


def make_world_lt():
    """构造一个标准球面世界 + 光照温度场（与引擎默认一致）。"""
    world = SphereWorld(rows=60, cols=120)
    lt = LightAndTemperature(
        world,
        rotation_period=2400,
        t_equator=30.0,
        t_pole=-20.0,
        day_boost=6.0,
        lat_base_ref=1.0,
    )
    return world, lt


def make_uniform(world, lt):
    return ResourceField(world, lt, capacity_per_area=40.0, regrowth_rate=0.5)


def make_patchy(world, lt, **kw):
    """构造 patchy 资源场；不传参数时用 ResourceField 默认保守值。"""
    defaults = dict(
        capacity_per_area=40.0,
        regrowth_rate=0.5,
        distribution="patchy",
    )
    defaults.update(kw)
    return ResourceField(world, lt, **defaults)


# ---------------------------------------------------------------------------
# 默认 uniform 行为零回归
# ---------------------------------------------------------------------------

def test_default_distribution_is_uniform():
    """默认配置 distribution == 'uniform'。"""
    world, lt = make_world_lt()
    rf = make_uniform(world, lt)
    assert rf.distribution == "uniform"
    assert rf._patch_mask is None


def test_uniform_initial_fill_is_half_capacity():
    """uniform 模式初始食物 = 容量 × 0.5（旧版行为）。"""
    world, lt = make_world_lt()
    rf = make_uniform(world, lt)
    np.testing.assert_allclose(rf._grid, rf._capacity * 0.5)


def test_uniform_custom_initial_fill():
    """uniform 模式 initial_fill 参数生效。"""
    world, lt = make_world_lt()
    rf = ResourceField(world, lt, initial_fill=0.3)
    np.testing.assert_allclose(rf._grid, rf._capacity * 0.3)


# ---------------------------------------------------------------------------
# 容量守恒
# ---------------------------------------------------------------------------

def test_capacity_conservation_patchy_vs_uniform():
    """patchy 模式总容量 == uniform 模式总容量（种群承载上限不变）。"""
    world, lt = make_world_lt()
    uniform = make_uniform(world, lt)
    patchy = make_patchy(world, lt)
    np.testing.assert_allclose(
        patchy._capacity.sum(), uniform._capacity.sum(), rtol=1e-12
    )


def test_patchy_patch_cells_have_higher_capacity():
    """patch 格容量 > 背景格容量（富集生效）。"""
    world, lt = make_world_lt()
    patchy = make_patchy(world, lt)
    mask = patchy._patch_mask
    assert mask is not None and mask.any()
    patch_cap = patchy._capacity[mask].mean()
    bg_cap = patchy._capacity[~mask].mean()
    assert patch_cap > bg_cap * 2  # 倍率 6.0，patch 平均应显著高于背景


def test_patchy_bg_cap_mult_positive():
    """正常参数下背景容量倍率 > 0。"""
    world, lt = make_world_lt()
    patchy = make_patchy(world, lt)
    # 背景格容量应全为正
    assert (patchy._capacity[~patchy._patch_mask] > 0).all()


def test_patchy_excessive_mult_raises_value_error():
    """斑块容量倍率过大导致背景容量为负时，抛 ValueError（而非静默负值）。"""
    world, lt = make_world_lt()
    with pytest.raises(ValueError, match="背景容量为负"):
        make_patchy(world, lt, patch_capacity_mult=100.0, patch_count=200)


# ---------------------------------------------------------------------------
# 再生守恒
# ---------------------------------------------------------------------------

def test_regrowth_conservation_formula_exact():
    """守恒公式本身精确成立：patch_mult × patch_frac + bg_mult × bg_frac = 1。"""
    world, lt = make_world_lt()
    patchy = make_patchy(world, lt)
    areas = world.cell_area(np.arange(world.n_cells))
    mask = patchy._patch_mask
    patch_frac = float(areas[mask].sum()) / float(areas.sum())
    bg_frac = float(areas[~mask].sum()) / float(areas.sum())
    lhs = patchy._patch_regrowth_mult * patch_frac + patchy._bg_regrowth_mult * bg_frac
    assert abs(lhs - 1.0) < 1e-12


def test_regrowth_conservation_patchy_vs_uniform():
    """同 tick 下 patchy 总再生量 ≈ uniform 总再生量。

    注：守恒公式是面积加权的，但 patch 格随机分布导致其平均温度与整体
    有微小差异（高温格再生快×倍率），实测偏差 <0.1%，远小于任何生态效应。
    """
    world, lt = make_world_lt()
    uniform = make_uniform(world, lt)
    patchy = make_patchy(world, lt)
    for tick in (0, 100, 600, 1200, 2399):
        g_u = uniform._regrowth_amount(tick)
        g_p = patchy._regrowth_amount(tick)
        np.testing.assert_allclose(g_p.sum(), g_u.sum(), rtol=1e-2)


def test_patchy_patch_cells_regrow_faster():
    """patch 格再生速度 > 背景格（温度相同时）。"""
    world, lt = make_world_lt()
    patchy = make_patchy(world, lt)
    # 选赤道格（温度高，再生非零），分别取 patch 和背景
    equator_row = 30
    patch_cells = [c for c in range(equator_row * 120, (equator_row + 1) * 120)
                   if patchy._patch_mask[c]]
    bg_cells = [c for c in range(equator_row * 120, (equator_row + 1) * 120)
                 if not patchy._patch_mask[c]]
    if patch_cells and bg_cells:
        growth = patchy._regrowth_amount(600)  # 正午附近
        assert growth[patch_cells].mean() > growth[bg_cells].mean()


# ---------------------------------------------------------------------------
# 空间非均匀
# ---------------------------------------------------------------------------

def test_patchy_initial_food_nonuniform():
    """patchy 模式初始食物分布非均匀（patch 填满，背景压低）。"""
    world, lt = make_world_lt()
    uniform = make_uniform(world, lt)
    patchy = make_patchy(world, lt)
    # patchy 的食物方差应显著大于 uniform（uniform 方差仅来自容量面积差异）
    assert patchy._grid.std() > uniform._grid.std() * 3


def test_patchy_patch_cells_full_background_low():
    """patch 格初始食物 ≈ 容量（填满），背景格 ≈ 容量 × background_fill。"""
    world, lt = make_world_lt()
    patchy = make_patchy(world, lt, background_fill=0.05)
    mask = patchy._patch_mask
    np.testing.assert_allclose(patchy._grid[mask], patchy._capacity[mask], rtol=1e-12)
    np.testing.assert_allclose(
        patchy._grid[~mask], patchy._capacity[~mask] * 0.05, rtol=1e-12
    )


# ---------------------------------------------------------------------------
# 可复现
# ---------------------------------------------------------------------------

def test_patchy_reproducible_same_seed():
    """同 patch_seed 两次构造 → patch_mask 完全相同。"""
    world, lt = make_world_lt()
    a = make_patchy(world, lt, patch_seed=123)
    b = make_patchy(world, lt, patch_seed=123)
    np.testing.assert_array_equal(a._patch_mask, b._patch_mask)
    np.testing.assert_allclose(a._capacity, b._capacity)


def test_patchy_different_seed_different_mask():
    """不同 patch_seed → patch_mask 不同（随机性生效）。"""
    world, lt = make_world_lt()
    a = make_patchy(world, lt, patch_seed=1)
    b = make_patchy(world, lt, patch_seed=2)
    assert not np.array_equal(a._patch_mask, b._patch_mask)


# ---------------------------------------------------------------------------
# 引擎集成
# ---------------------------------------------------------------------------

def test_engine_default_uniform_runs():
    """引擎默认配置（uniform）跑 50 tick 不崩。"""
    cfg = SimConfig(seed=42)
    cfg.population.initial_count = 50
    cfg.simulation.ticks = 50
    e = SphereEngine(cfg)
    for _ in range(50):
        e.step()
    assert e.alive_count() > 0 or e.extinct  # 要么活着要么正常灭绝，不崩


def test_engine_patchy_runs():
    """引擎 patchy 配置跑 50 tick 不崩、种群存活。"""
    cfg = SimConfig(seed=42)
    cfg.population.initial_count = 50
    cfg.simulation.ticks = 50
    cfg.resources.distribution = "patchy"
    e = SphereEngine(cfg)
    assert e.resources.distribution == "patchy"
    for _ in range(50):
        e.step()
    # patchy 模式下种群应能存活（斑块有充足食物）
    assert e.alive_count() > 0


def test_engine_patchy_uses_patch_seed_from_config():
    """引擎用 config.seed 作为 patch_seed → 同 seed 可复现。"""
    cfg = SimConfig(seed=99)
    cfg.resources.distribution = "patchy"
    e1 = SphereEngine(cfg)
    e2 = SphereEngine(cfg)
    np.testing.assert_array_equal(e1.resources._patch_mask, e2.resources._patch_mask)


def test_engine_patchy_does_not_consume_engine_rng():
    """patchy 中心选择用独立 rng，不消费引擎 self.rng → 不影响后续 RNG 顺序。

    验证：uniform 和 patchy 引擎在第一个 tick 的个体初始基因/位置分布
    （由 self.rng 生成）应完全一致（同 seed），因为 patchy 的 rng 是独立的。
    """
    cfg_u = SimConfig(seed=7)
    cfg_u.population.initial_count = 30
    e_u = SphereEngine(cfg_u)

    cfg_p = SimConfig(seed=7)
    cfg_p.population.initial_count = 30
    cfg_p.resources.distribution = "patchy"
    e_p = SphereEngine(cfg_p)

    # 初始基因和位置由 self.rng 生成，应完全一致
    np.testing.assert_allclose(e_u._genes, e_p._genes)
    np.testing.assert_array_equal(e_u._flat, e_p._flat)
