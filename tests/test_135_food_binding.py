"""13.5 ①② 食物绑定 —— 背景产能归零 + 斑块再生校准（2026-09-24 派工单）。

覆盖：
  ① **默认不变**（`bg_production_zero=False`）⇒ 背景照常生产、两条守恒式仍成立
     ⇒ 既有 patchy 批（13.4 各批）**仍可复现**
  ② **背景归零生效**：背景格容量 / 再生 / 初始存量**三者都 0**；斑块侧不受影响
  ③ **斑块再生校准 ×1.5**（`patch_regrowth_mult` 2.0→3.0）：Σ_斑块 名义再生 680 → **1 020**
  ④ 🔴 **K 目标对账（唯一硬判据的解析形式，零机时）**：
     13.5 配置（背景归零 + 倍率 3.0 + `assim_herb=0.4`）⇒ `K_pred = Σ_斑块再生 ÷ 人均需求`
     必须落进 **[1200, 1900]** 且 **< 软顶 1944**（R186 §一）
  ⑤ `deposit()`（③ 的回流接口）：正常放回 / 容量封顶 / 空输入 / 与 `consume_many` 往返守恒
  ⑥ ③ 的 7 个新字段**默认值 = 现状行为**（这是给接线方的契约：默认不许改行为）
  ⑦ `patch_regrowth_mult` **两处默认值一致**（`ResourceConfig` 与 `ResourceField` 签名）
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.config import OrganismConfig, ResourceConfig, SimConfig
from world.light_and_temperature import LightAndTemperature
from world.resource_field import ResourceField
from world.sphere_world import SphereWorld

# 派工单 §二：人均支出（实测，R185）+ 现状 eat_efficiency
PER_CAPITA_SPEND_E = 0.762
EAT_EFFICIENCY = 3.0
#: 软顶（`soft_cap_target=0.6 × max_count 3240`）—— K 必须低于它，食物才算绑定
SOFT_CAP_LOCAL = 1944.0


def make_world_lt():
    world = SphereWorld(rows=60, cols=120)
    lt = LightAndTemperature(world, rotation_period=2400, t_equator=30.0,
                             t_pole=-20.0, day_boost=6.0, lat_base_ref=1.0)
    return world, lt


def make_patchy(**kw):
    world, lt = make_world_lt()
    defaults = dict(capacity_per_area=40.0, regrowth_rate=0.5,
                    distribution="patchy", patch_count=30, patch_radius=2,
                    patch_capacity_mult=3.0)
    defaults.update(kw)
    return ResourceField(world, lt, **defaults)


# --------------------------------------------------------- ① 默认不变（可复现性）

def test_default_keeps_background_producing():
    """`bg_production_zero` 默认 False ⇒ 背景照常（否则 13.4 各批不可复现）。"""
    rf = make_patchy()
    assert rf.bg_production_zero is False
    assert (rf._capacity[~rf._patch_mask] > 0).all()
    assert rf._bg_regrowth_mult > 0.0
    g = rf._regrowth_amount(0)
    assert g[~rf._patch_mask].sum() > 0.0


def test_default_regrowth_conservation_identity_holds():
    """默认档下「再生守恒式」仍成立（背景归零时它才被有意打破）。"""
    rf = make_patchy()
    area = rf.world.cell_area(np.arange(rf.world.n_cells))
    pa = float(area[rf._patch_mask].sum())
    ba = float(area[~rf._patch_mask].sum())
    lhs = (rf._patch_regrowth_mult * pa + rf._bg_regrowth_mult * ba) / (pa + ba)
    assert abs(lhs - 1.0) < 1e-12


# --------------------------------------------------------- ② 背景产能归零

def test_bg_production_zero_kills_capacity_regrowth_and_stock():
    """背景归零 ⇒ 容量 / 再生 / 初始存量**三者都为 0**；斑块侧完好。"""
    rf = make_patchy(bg_production_zero=True)
    m = rf._patch_mask
    assert rf.bg_production_zero is True
    assert (rf._capacity[~m] == 0.0).all(), "背景容量未归零"
    assert (rf._grid[~m] == 0.0).all(), "背景初始存量未归零（容量为 0 时应自动为 0）"
    assert rf._bg_regrowth_mult == 0.0, "背景再生倍率未归零"
    assert (rf._regrowth_amount(0)[~m] == 0.0).all(), "背景仍在生产"
    # 斑块侧不受影响
    assert (rf._capacity[m] > 0).all()
    assert rf._regrowth_amount(0)[m].sum() > 0.0


def test_bg_zero_regrow_keeps_background_empty():
    """归零后跑多 tick 的再生 ⇒ 背景**永远长不出来**（这是"食物绑定"的载体）。"""
    rf = make_patchy(bg_production_zero=True)
    m = rf._patch_mask
    for t in range(5):
        rf.regrow(t * 100)
    assert float(rf._grid[~m].sum()) == 0.0
    assert float(rf._grid[m].sum()) > 0.0


def test_bg_zero_breaks_conservation_on_purpose():
    """🔴 归零**有意打破**两条构造期守恒 ⇒ 必须能读出这个事实（防后人误判为缺陷）。"""
    normal = make_patchy()
    zeroed = make_patchy(bg_production_zero=True)
    cap_n = float(normal._capacity.sum())
    cap_z = float(zeroed._capacity.sum())
    area = normal.world.cell_area(np.arange(normal.world.n_cells))
    m = normal._patch_mask
    assert cap_z < cap_n
    # 剩下的正是"斑块部分"（≈32.3%，R182）
    assert abs(cap_z / cap_n - 0.323) < 0.01, f"斑块容量占比 {cap_z / cap_n:.3f} 与 R182 不符"
    # 斑块面积占比 ≈10.8%，容量倍率 3.0 ⇒ 3.0×10.8% ≈ 32.3%
    assert float(area[m].sum() / area.sum()) == pytest.approx(0.1076, abs=0.005)


# --------------------------------------------------------- ③ 斑块再生校准

def test_patch_regrowth_calibration_is_x1_5():
    """`patch_regrowth_mult` 2.0→3.0 ⇒ Σ_斑块 名义再生 **680 → 1020**（= ×1.5）。"""
    a = make_patchy(patch_regrowth_mult=2.0)
    b = make_patchy(patch_regrowth_mult=3.0)
    m = a._patch_mask
    ga = float(a._regrowth_amount(0)[m].sum())
    gb = float(b._regrowth_amount(0)[m].sum())
    assert ga == pytest.approx(680.1, abs=0.5), f"mult=2.0 的斑块再生 {ga}（R185 报 680）"
    assert gb == pytest.approx(1020.1, abs=0.5), f"mult=3.0 的斑块再生 {gb}"
    assert gb / ga == pytest.approx(1.5, rel=1e-9)


# --------------------------------------------------------- ④ K 目标对账（唯一硬判据）

def _sigma_patch(mult: float = 3.0, bg_zero: bool = True) -> float:
    """Σ_斑块名义再生（质量/tick）。"""
    rf = make_patchy(bg_production_zero=bg_zero, patch_regrowth_mult=mult)
    m = rf._patch_mask
    return float(rf._regrowth_amount(0)[m].sum()) if bg_zero \
        else float(rf._regrowth_amount(0).sum())


def _k(sigma: float, eat_eff: float, assim: float) -> float:
    """`K = Σ ÷ 人均需求`；`人均需求 = 支出 ÷ (eat_efficiency × 吸收率)`。"""
    return sigma / (PER_CAPITA_SPEND_E / (eat_eff * assim))


def test_sigma_patch_is_linear_in_mult():
    """Σ_斑块再生与倍率**严格线性**（K 校准据此外推）。"""
    assert _sigma_patch(1.0) == pytest.approx(340.04, abs=0.5)
    assert _sigma_patch(2.0) == pytest.approx(680.1, abs=0.5)     # 与 R185 逐值一致
    assert _sigma_patch(3.0) == pytest.approx(1020.1, abs=0.5)


def test_k_depends_only_on_net_absorption():
    """🔴 **K 只由"净吸收" `eat_efficiency × 吸收率` 决定**（2026-09-24 的关键耦合）。

    派工单 §二 的算术隐含「`assim_herb=0.4` 把人均需求从 0.254 抬到 0.635」⇒ K ≈ 1 606；
    但 R187 为修"吸收率致灭绝"把 `eat_efficiency` 3.0 → **7.5**（语义 = 完全燃烧值）⇒
    `7.5 × 0.4 = 3.0` = **现状净吸收** ⇒ 人均需求**没变** ⇒ K 变 **≈4 016**（≥ 软顶 ⇒ 不达标）。
    ⇒ 本测试把这条耦合钉死：**改了 ③ 的净吸收就必须重算 ② 的生产校准**。
    """
    sig = _sigma_patch(3.0)
    k_paper = _k(sig, 3.0, 0.4)      # 派工单 §二 口径（净吸收 1.2）
    k_r187 = _k(sig, 7.5, 0.4)       # R187 口径（净吸收 3.0 = 现状）
    assert k_paper == pytest.approx(1606, abs=2)
    assert k_r187 == pytest.approx(4016, abs=3)
    assert k_r187 / k_paper == pytest.approx(2.5, rel=1e-6), "净吸收 3.0/1.2 = 2.5 ⇒ K 同比"
    # K 只看净吸收 ⇒ 两支"净吸收相同"的组合必须给出同一个 K
    assert _k(sig, 3.0, 1.0) == pytest.approx(_k(sig, 7.5, 0.4), rel=1e-12)


def test_required_mult_for_target_k():
    """两个口径各自需要的 `patch_regrowth_mult`（**方向相反**：派工单要 ×1.5，R187 要下调）。

    目标 K = 1600 ⇒ 需要 `Σ = 1600 × 人均需求`。
    """
    need_paper = PER_CAPITA_SPEND_E / (3.0 * 0.4)      # 0.6350
    need_r187 = PER_CAPITA_SPEND_E / (7.5 * 0.4)       # 0.2540
    coef = 340.04
    assert 1600.0 * need_paper / coef == pytest.approx(2.99, abs=0.02)   # ≈ 派工单的 3.0 ✓
    assert 1600.0 * need_r187 / coef == pytest.approx(1.195, abs=0.02)   # R187 口径要**下调**
    assert 1600.0 * need_paper / coef > 1600.0 * need_r187 / coef, "净吸收越低 ⇒ 需求越大 ⇒ 需要更多产能"


def test_current_default_caliber_k_is_far_above_softcap():
    """反证（两个都钉住，防"绝对值漂移"）：

    * **背景不归零**（现状）⇒ Σ=3 154 ⇒ K ≈ 12 400（> 6× 软顶）
    * **背景归零 + mult 3.0 + R187 净吸收** ⇒ Σ=1 020 ⇒ K ≈ 4 016（仍 ≥ 软顶 ⇒ 仍不达标）
    ⇒ 后者说明：**只做 ①②、不重算净吸收，K 达不到 [1200,1900]** —— 必须与 ③ 联动定稿。
    """
    k_bg_on = _k(_sigma_patch(3.0, bg_zero=False), 3.0, 1.0)
    assert k_bg_on > 12_000.0, f"背景未归零时 K 只有 {k_bg_on:.0f}（预期 ≈12 400）"
    k_bg_off_r187 = _k(_sigma_patch(3.0), 7.5, 0.4)
    assert k_bg_off_r187 > SOFT_CAP_LOCAL, (
        f"背景归零后 K={k_bg_off_r187:.0f} 已 < 软顶？与 2026-09-24 实测算不符 ⇒ 重算"
    )


# --------------------------------------------------------- ⑤ deposit（回流接口）

def test_deposit_returns_actual_amount_and_caps_at_capacity():
    rf = make_patchy()
    cell = int(np.flatnonzero(rf._patch_mask)[0])
    rf._grid[:] = 0.0
    cap = float(rf._capacity[cell])
    got = rf.deposit(cell, cap * 3.0)              # 想放 3 倍容量
    assert float(got) == pytest.approx(cap, rel=1e-12), "应按容量封顶"
    assert float(rf._grid[cell]) == pytest.approx(cap, rel=1e-12)
    assert float(rf.deposit(cell, 5.0)) == 0.0, "已满的格不应再进"


def test_deposit_empty_and_vector_forms():
    rf = make_patchy()
    assert rf.deposit(np.zeros(0, dtype=np.int64), np.zeros(0)).shape == (0,)
    cells = np.flatnonzero(rf._patch_mask)[:3]
    rf._grid[cells] = 0.0
    out = rf.deposit(cells, 1.0)
    assert out.shape == (3,)
    np.testing.assert_allclose(out, 1.0)


def test_deposit_is_inverse_of_consume():
    """`consume_many` 吃掉的量，`deposit` 能原样放回（闭环 ⇒ 守恒可审计）。"""
    rf = make_patchy()
    cells = np.flatnonzero(rf._patch_mask)[:4]
    before = rf._grid[cells].copy()
    taken = rf.consume_many(cells, 0.5)
    assert float(taken.sum()) > 0.0
    back = rf.deposit(cells, taken)
    np.testing.assert_allclose(rf._grid[cells], before, rtol=1e-12)


# --------------------------------------------------------- ⑥ ③ 字段契约（默认 = 现状）

def test_new_energy_fields_exist_with_current_behavior_defaults():
    """③ 的字段必须存在，且**默认值 = 现状行为**（接线方不许靠改默认来开机制）。"""
    o = OrganismConfig()
    assert o.stomach_cap_mass == 0.0, "0 = 用旧派生式（现状）"
    assert o.eat_threshold_frac == 0.0, "0 = 永远进食（现状）"
    assert o.starve_frac == 0.0, "0 = 无饿死阈（现状）"
    assert o.exhaust_frac == 0.0, "0 = 无力竭阈（现状）"
    assert o.assim_herb == 1.0 and o.assim_carn == 1.0, "1.0 = 无吸收损失（现状）"
    assert o.assim_return_frac == 1.0, "1.0 = 全回流；assim=1 时该字段不起作用"
    assert o.eat_amount == 0.9, "eat_amount 保持 13.4 的值（本批不动）"
    assert o.photo_max == 0.1, "photo_max 默认保持 0.1（13.5 由预设置 0）"


def test_dual_threshold_order_assertion():
    """双阈值语义：`exhaust_frac ≤ starve_frac`（力竭先触发）；单独给值不受约束。"""
    OrganismConfig(exhaust_frac=0.17, starve_frac=0.30)      # ✓ 13.5 拟值
    OrganismConfig(exhaust_frac=0.0, starve_frac=0.30)       # ✓ 只开饿死
    OrganismConfig(exhaust_frac=0.5, starve_frac=0.0)        # ✓ 只开力竭（无比较对象）
    with pytest.raises(AssertionError):
        OrganismConfig(exhaust_frac=0.5, starve_frac=0.3)    # ❌ 力竭晚于饿死 ⇒ 报错


def test_resource_config_defaults_and_field_alignment():
    """`bg_production_zero` 默认 False；`patch_regrowth_mult` 两处默认值一致（防漂移）。"""
    r = ResourceConfig()
    assert r.bg_production_zero is False
    assert r.patch_regrowth_mult == 3.0
    rf = make_patchy()          # 不传该参数 ⇒ 用 ResourceField 签名默认
    assert rf._patch_regrowth_mult == r.patch_regrowth_mult, (
        "ResourceField 签名默认与 ResourceConfig 默认漂移 ⇒ 直接构造的测试/工具会得到不同世界"
    )


def test_config_fingerprint_tracks_new_fields():
    """新字段都必须进配置指纹（纪元可自证；否则跨档续跑拦不住）。"""
    base = SimConfig().fingerprint()
    c1 = SimConfig(); c1.resources.bg_production_zero = True
    c2 = SimConfig(); c2.organisms.assim_herb = 0.4
    assert c1.fingerprint() != base, "bg_production_zero 未进指纹"
    assert c2.fingerprint() != base, "assim_herb 未进指纹"
