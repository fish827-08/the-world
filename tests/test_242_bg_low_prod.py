"""R242 背景低产能带（bg_low_prod_frac / bg_cap_mult）回归测试。

背景：R241 根因 —— `bg_production_zero=True` 下生产格仅占 2.78%，斑块是零食物沙漠中的
孤岛 ⇒ 个体 97–100% 时间被钉在斑块格 ⇒ 跨斑块探索几何上不可能 ⇒ 门B 结构性失败。
R242 给背景一部分格极低产能，形成"绿洲链"续命带（续命不养活）。

本测试守三件事（对应 C9「死字段」家族与 C7「默认逐位等价」）：
  1. **默认逐位等价**：两个字段默认 0 ⇒ `_capacity` / `_grid` / 再生路径与原版完全相同；
  2. **透传不静默**：`ResourceConfig` 的字段必须真的到达 `ResourceField`
     （13.5① `bg_production_zero` 曾因少一行接线而静默无效）；
  3. **机制真生效**：低产能格拿到 >0 且 << patch 的容量；占比 = 请求比例；
     语义守卫（给了比例不给倍率 ⇒ 报错）。
"""
import numpy as np
import pytest

from simulation.config import ResourceConfig, SimConfig
from simulation.sphere_engine import SphereEngine
from world.resource_field import ResourceField


def _mk(**kw):
    """构造一个 480×960 patchy 引擎（走 Python 路径），便于取内部数组。"""
    from experiments.steady_k_probe import make_cfg, apply_post_build
    c, notes = make_cfg(42, 480, 960, 2000, 1700, True, 0.125, 0.125, 80, 2.5, 30000, 1.195)
    c.simulation.use_sim_core = False
    c.resource_dynamics.enabled = False
    for k, v in kw.items():
        setattr(c.resources, k, v)
    eng = SphereEngine(c)
    apply_post_build(eng, notes)
    return eng


# ---------------- 1) 默认逐位等价 ----------------

def test_defaults_are_zero():
    r = ResourceConfig()
    assert r.bg_low_prod_frac == 0.0
    assert r.bg_cap_mult == 0.0


def test_default_path_is_bitwise_identical():
    """默认（两字段为 0）⇒ 容量与初始存量与"显式全零产能"完全一致。"""
    a = _mk()                                   # 默认 0
    b = _mk(bg_low_prod_frac=0.0, bg_cap_mult=0.0)
    assert np.array_equal(a.resources._capacity, b.resources._capacity)
    assert np.array_equal(a.resources._grid, b.resources._grid)
    assert a.resources._bg_low_mask.sum() == 0
    # 默认档下 C7 digest 不该动（该断言由既有 test_a_continuous 等承担，此处仅守局部）

    # 更硬的：默认档与"不传该参数"的裸 ResourceField 逐位相同
    rf = ResourceField(a.world, a.light, distribution="patchy", patch_count=1700,
                       patch_radius=2, patch_capacity_mult=3.0,
                       patch_regrowth_mult=1.195, bg_production_zero=True,
                       patch_seed=42)
    assert np.array_equal(a.resources._capacity, rf._capacity)


# ---------------- 2) 透传不静默（C9 家族） ----------------

def test_config_field_reaches_resource_field():
    """`ResourceConfig.bg_low_prod_frac` 必须真的到达 `ResourceField`（防死字段）。"""
    c = SimConfig()
    c.resources.distribution = "patchy"
    c.resources.patch_count = 300
    c.resources.bg_production_zero = True
    c.resources.bg_low_prod_frac = 0.4
    c.resources.bg_cap_mult = 0.05
    c.simulation.use_sim_core = False
    eng = SphereEngine(c)
    rf = eng.resources
    assert rf._bg_low_mask is not None
    # 若透传断了 ⇒ mask 全 False ⇒ 本断言变红（正是要堵的"静默无变化"）
    assert rf._bg_low_mask.sum() > 0, "bg_low_prod_frac 未透传到 ResourceField（死字段）"
    assert rf._bg_low_cap_mult > 0.0, "bg_cap_mult 未透传到 ResourceField（死字段）"


# ---------------- 3) 机制真生效 ----------------

def test_low_prod_fraction_and_capacity():
    eng = _mk(bg_low_prod_frac=0.4, bg_cap_mult=0.05)
    rf = eng.resources
    pm = rf._patch_mask
    bm = rf._bg_low_mask
    n_bg = int((~pm).sum())
    n_low = int(bm.sum())
    assert abs(n_low / n_bg - 0.4) < 0.01, "低产能格占比应为 40%"
    cap = rf._capacity
    # 低产能格容量 > 0（否则等于没开）且 << patch 格（否则沙漠变宜居区，斑块信息价值消失）
    assert cap[bm].mean() > 0.0
    ratio = cap[bm].mean() / cap[pm].mean()
    assert ratio < 0.10, f"低产能/patch 容量比 {ratio:.4f} 过大 ⇒ 沙漠变宜居"
    # 其余背景格仍为 0（零食物墙还在）
    rest = (~pm) & (~bm)
    if rest.any():
        assert cap[rest].max() == 0.0, "非低产能背景格应保持零容量"


def test_low_prod_regrowth_is_low_but_positive():
    """低产能格的再生必须 >0（续命）且 << patch（不养活）。"""
    eng = _mk(bg_low_prod_frac=0.4, bg_cap_mult=0.05)
    rf = eng.resources
    g = rf._regrowth_amount(tick=0)
    pm, bm = rf._patch_mask, rf._bg_low_mask
    g_patch = float(g[pm].mean())
    g_low = float(g[bm].mean())
    g_zero = float(g[(~pm) & (~bm)].mean()) if ((~pm) & (~bm)).any() else 0.0
    assert g_low > 0.0, "低产能格无再生 ⇒ 绿洲链无意义"
    assert g_zero == 0.0, "零产能背景格再生应为 0"
    assert g_low < g_patch * 0.10


def test_independent_rng_does_not_touch_engine_rng():
    """低产能格的选取必须用独立 rng ⇒ 不改变引擎 RNG 消费顺序。

    验证：同一 seed 下，开/关低产能带，引擎自身 rng 的**首个**随机数应相同。
    """
    a = _mk()
    b = _mk(bg_low_prod_frac=0.4, bg_cap_mult=0.05)
    # 两个引擎各自的 self.rng 状态在构造后应一致（低产能带选取未消费它）
    sa = a.rng.bit_generator.state["state"]["state"]
    sb = b.rng.bit_generator.state["state"]["state"]
    assert sa == sb, "低产能带消费了引擎 RNG ⇒ 会改变同 seed 结果（违反独立 rng 纪律）"


def test_guard_frac_without_mult_raises():
    with pytest.raises(AssertionError):
        ResourceConfig(distribution="patchy", bg_production_zero=True,
                       bg_low_prod_frac=0.4, bg_cap_mult=0.0)


def test_guard_frac_without_bg_zero_raises():
    """背景未归零时开低产能带 ⇒ 语义冲突（守恒摊均值会覆盖）⇒ 构造期报错。"""
    from experiments.steady_k_probe import make_cfg, apply_post_build
    c, notes = make_cfg(42, 240, 480, 500, 100, True, 0.125, 0.125, 80, 2.5, 30000, 1.195)
    c.simulation.use_sim_core = False
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = False     # 背景未归零
    c.resources.bg_low_prod_frac = 0.4
    c.resources.bg_cap_mult = 0.05
    with pytest.raises(ValueError):
        SphereEngine(c)
