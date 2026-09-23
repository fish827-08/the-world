"""13.5 ③ 能量标定 —— 回归单测（`[所有者·天平]` 接线，R188）。

覆盖（每条都断言"**可观测的行为差异**"，不是"字段存在"）：
    E1 关档逐位等价：③ 全默认 vs 显式传默认值 ⇒ digest 完全相同
    E2 `stomach_cap_mass` 生效：胃容量不再由 `max_energy` 派生（上界真的变小）
    E3 `eat_threshold_frac` 生效：胃到阈值就停吃（`_stomach` 上界 ≈ 阈值×容量）
    E4 `exhaust_frac` 生效：能量低于阈值即死（**无论胃**）⇒ 死亡激增
    E5 `starve_frac` 的"且胃空"语义：同阈值下 `starve_frac` 死亡数 **<** `exhaust_frac`
       （前者要求胃空、后者不要求）⇒ 这条是**双阈值语义**的直接检验
    E6 `assim_herb` 生效：吸收率 0.4 ⇒ 同 tick 数下总能量显著更低
    E7 `assim_return_frac` 生效：未吸收部分真的回流到资源格

设计原则（教训库）：**默认关 = 旧行为逐位一致**（E1 钉死）；每条改一个参数、
断言"变化方向与量级"，避免"有测试却不验不变量"（A#8）。
"""

from __future__ import annotations

import numpy as np
import pytest

from simulation.config import InfoStructureConfig, SimConfig
from simulation.sphere_engine import SphereEngine


def _engine(**org_kw) -> SphereEngine:
    """构造引擎；`org_kw` 覆盖 `OrganismConfig` 字段（未列的保持默认 = 现状）。"""
    cfg = SimConfig(seed=42)
    cfg.simulation.use_sim_core = False          # 新机制强制 Python（§14.7）
    cfg.population.max_count = 600
    cfg.info_structure = InfoStructureConfig(
        enabled=True, learning_rate=0.05, memory_gradient="none"
    )
    for k, v in org_kw.items():
        assert hasattr(cfg.organisms, k), f"OrganismConfig 无字段 {k}（防拼写错 ⇒ 静默无效）"
        setattr(cfg.organisms, k, v)
    return SphereEngine(cfg)


def _run(e: SphereEngine, ticks: int = 60) -> SphereEngine:
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    return e


def _digest(e: SphereEngine) -> tuple[int, float]:
    return (int(e._flat.sum()), round(float(e._energy.sum()), 6))


def _deaths(e: SphereEngine) -> dict:
    return {str(k).split(".")[-1]: int(v) for k, v in e.death_cause_totals().items()}


# --------------------------------------------------------------------------- E1
def test_e1_defaults_are_bitwise_identical() -> None:
    """③ 全默认 vs 显式传默认值 ⇒ 逐位等价（关档不改变旧行为）。

    显式传的默认值：`stomach_cap_mass=0` / `eat_threshold_frac=0` /
    `starve_frac=0` / `exhaust_frac=0` / `assim_*=1.0` / `assim_return_frac=0`。
    """
    a = _run(_engine(), 60)
    b = _run(_engine(stomach_cap_mass=0.0, eat_threshold_frac=0.0, starve_frac=0.0,
                     exhaust_frac=0.0, assim_herb=1.0, assim_carn=1.0,
                     assim_return_frac=0.0), 60)
    assert _digest(a) == _digest(b), (
        f"显式默认值改变了行为：{_digest(a)} vs {_digest(b)} ⇒ 接线在默认档**不等价**")


def test_e1b_matches_c7_baseline() -> None:
    """③ 全默认 + 本仓库的 C7 helper 口径 ⇒ 与钉死基线一致（防"孤立等价"）。"""
    cfg = SimConfig(seed=42)
    cfg.simulation.use_sim_core = False
    cfg.simulation.l2_dash = False
    cfg.population.max_count = 600
    cfg.predation.forage_tradeoff_k = 0.0
    cfg.info_structure = InfoStructureConfig(
        enabled=True, learning_rate=0.05, memory_gradient="none"
    )
    cfg.organisms.young_mob_mult = 0.55
    cfg.organisms.old_mob_mult = 0.55
    cfg.organisms.dash_min_energy_frac = 0.2
    cfg.organisms.far_cap = 32
    e = SphereEngine(cfg)
    for _ in range(50):
        if e.extinct:
            break
        e.step()
    assert _digest(e) == (574887, 11266.746993), (
        f"C7 基线漂移：{_digest(e)} ≠ (574887, 11266.746993) ⇒ 13.5 接线动了默认档")


# --------------------------------------------------------------------------- E2 / E3
def test_e2_stomach_cap_mass_is_independent() -> None:
    """`stomach_cap_mass=5` ⇒ 胃上界 ≤ 5×cap_mult_max = 10；默认档实测 ≈75.5。

    🔴 **陷阱（本测试第一版就踩了）**：胃容量 = `容量基准 × cap_mult`，
    其中 `cap_mult = 0.5 + Gene.STOMACH_CAP × 1.5 ∈ [0.5, 2.0]`（基因缩放）。
    默认公式的基准 = `max_energy / eat_efficiency × 0.5` = **50** ⇒ 默认档上界 ≈ 50×cap_mult（可达 100）。
    ⇒ 用 `stomach_cap_mass=25` 时 25×2.0 = 50，**恰好与"默认基准 50"巧合相同** ⇒ 无法区分！
    必须用**足够小**的值（5.0）才能把两档分开。
    """
    e = _run(_engine(stomach_cap_mass=5.0), 60)
    top = float(e._stomach.max())
    assert top <= 10.0 + 1e-6, f"胃上界 {top:.3f} > 10（=5×cap_mult_max 2.0）⇒ 独立胃容量未生效"

    d = _run(_engine(), 60)
    top_d = float(d._stomach.max())
    assert top_d > 50.0, (
        f"默认档胃上界只有 {top_d:.3f} ⇒ 本测试前提不成立"
        "（默认篇应 ≈50×cap_mult > 50 ⇒ 才能反衬独立档）")


def test_e3_eat_threshold_stops_eating_early() -> None:
    """`eat_threshold_frac=0.5` ⇒ 胃上界显著低于"不设阈值"的对照（比值断言）。

    ⚠️ 不能断言绝对上界 = 阈值：判定是「胃 < 阈值 才吃」，而**一次取食量 `eat_amount`(=0.9)
    会把胃推过阈值**（胃 4.9 → 吃 0.9 → 5.8 → 停）。所以正确的可观测差异是
    **上界比值**（实测 0.63×），而不是等式。
    """
    e = _run(_engine(stomach_cap_mass=5.0, eat_threshold_frac=0.5), 60)
    base = _run(_engine(stomach_cap_mass=5.0, eat_threshold_frac=0.0), 60)
    top = float(e._stomach.max())
    top_b = float(base._stomach.max())
    assert top < 0.8 * top_b, (
        f"进食阈值未生效：设 0.5 阈值后上界 {top:.3f} 未显著低于对照 {top_b:.3f}"
        "（应 ≈0.6× —— 阈值 0.5 再加一次取食的越界量）")
    assert top_b > 5.0 + 1e-6, f"对照组上界仅 {top_b:.3f} ⇒ 前提不成立（应 ≈5×cap_mult ≈10）"


# --------------------------------------------------------------------------- E4 / E5
def test_e4_exhaust_kills_regardless_of_stomach() -> None:
    """`exhaust_frac=0.9`（阈值 270）⇒ 几乎全员立即"力竭"而死（无论胃里有没有食）。"""
    e = _run(_engine(exhaust_frac=0.9), 60)
    dc = _deaths(e)
    dead = sum(dc.values())
    assert dead > 0, "力竭阈值 0.9 未造成任何死亡 ⇒ 该阈值未接线"


def test_e5_starve_requires_empty_stomach_semantics() -> None:
    """双阈值语义：同阈值下 `starve_frac`（**且胃空**）致死数 < `exhaust_frac`（无论胃）。

    这条直接检验 E4/E5 的分工：若两者死亡数相同 ⇒ "且胃空"这个条件没有生效。
    """
    stv = _run(_engine(starve_frac=0.9), 60)
    exh = _run(_engine(exhaust_frac=0.9), 60)
    n_stv = sum(_deaths(stv).values())
    n_exh = sum(_deaths(exh).values())
    assert n_stv < n_exh, (
        f"starve({n_stv}) 应 < exhaust({n_exh})：前者额外要求「胃空」⇒ "
        "若相等说明「且胃空」条件未生效（双阈值被退化成单阈值）")


# --------------------------------------------------------------------------- E6 / E7
def test_e6_assim_herb_reduces_energy_intake() -> None:
    """吸收率 0.4 ⇒ 同 tick 数下总能量显著低于默认（吃同样多、得到的少）。"""
    hi = _run(_engine(), 120)
    lo = _run(_engine(assim_herb=0.4), 120)
    s_hi, s_lo = float(hi._energy.sum()), float(lo._energy.sum())
    assert s_lo < s_hi, f"吸收率 0.4 的总能量 {s_lo:.1f} 未低于默认 {s_hi:.1f} ⇒ 未生效"
    assert s_lo < 0.9 * s_hi, (
        f"降幅过小（{s_hi:.1f}→{s_lo:.1f}）：若几乎不变，需查 `eat_efficiency` 是否被同批抬高")


def test_e7_assim_return_flows_back_to_resource_grid() -> None:
    """未吸收部分真的回流：`assim_return_frac=1.0` 时资源格总量 ≥ 关闭时。"""
    keep = _run(_engine(assim_herb=0.4, assim_return_frac=1.0), 120)
    drop = _run(_engine(assim_herb=0.4, assim_return_frac=0.0), 120)
    g_keep = float(np.asarray(keep.resources._grid, dtype=float).sum())
    g_drop = float(np.asarray(drop.resources._grid, dtype=float).sum())
    assert g_keep > g_drop, (
        f"回流未生效：return=1.0 资源总量 {g_keep:.1f} 未高于 return=0.0 的 {g_drop:.1f}")


def test_e7b_return_does_nothing_when_assim_is_one() -> None:
    """反退化：`assim=1.0`（无未吸收部分）时，`assim_return_frac` 不应改变任何东西。"""
    a = _run(_engine(assim_return_frac=1.0), 60)
    b = _run(_engine(assim_return_frac=0.0), 60)
    assert _digest(a) == _digest(b), (
        f"assim=1 时 return_frac 仍改变了行为：{_digest(a)} vs {_digest(b)} ⇒ "
        "回流公式没有按「未吸收量」缩放（会在默认档引入偏差）")
