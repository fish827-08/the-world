"""14.9 ARS 双模式觅食 —— 回归单测。

照 13.8 的教训（R197 V2：交付时零单测 ⇒ DEL-8 结构性无法执行），本文件从一开始就把
"关档等价 / 真生效 / H3 fail-loud / 变异敏感性 / 反退化"写成常驻断言。

覆盖：
    S1 关档逐位等价：全默认 vs 显式传默认 ⇒ digest 完全相同
    S2 C7 基线不动：默认档 vs 钉死基线 `(574887, 11266.746993)`
    S3 `ArsConfig` 字段校验（`__post_init__`，fail-loud）
    S4 **H3 A1**：`ars.enabled ∧ use_sim_core=True` ⇒ NotImplementedError
    S5 `gain` 真生效：惯性项读数随 gain 变（不是"字段存在"）
    S6 **DEL-8 变异敏感性（逐机制）**：开档 ≠ 关档 digest
    S7 反退化读数存在且真跑了（`ars_dec_n > 0`）
    S8 关档 ⇒ `ars_probe()` **None**（未适用），不是 0（R120/§五.12）
    S9 P7：`Gene.PERSISTENCE == 20`、`Gene.GIVE_UP == 21` 且语义已改名
    S10 模式切换真的发生：`sw_ie_n`/`sw_ei_n` > 0（两腿都活着）
"""

from __future__ import annotations

import numpy as np
import pytest

from simulation.config import ArsConfig, InfoStructureConfig, SimConfig
from simulation.genes import GENE_SEMANTICS, Gene
from simulation.sphere_engine import SphereEngine


def _engine(*, ars: bool = False, gain: float = 1.0, theta: float = 0.5,
            kappa: float = 0.0, giveup: int = 20,
            fast_tau: float = 50.0, slow_tau: float = 500.0,
            patchy: bool = False,
            use_sim_core: bool = False, max_count: int = 600) -> SphereEngine:
    cfg = SimConfig(seed=42)
    cfg.simulation.use_sim_core = use_sim_core
    cfg.simulation.l2_dash = False
    cfg.population.max_count = max_count
    cfg.predation.forage_tradeoff_k = 0.0
    cfg.info_structure = InfoStructureConfig(
        enabled=True, learning_rate=0.05, memory_gradient="none"
    )
    cfg.organisms.young_mob_mult = 0.55
    cfg.organisms.old_mob_mult = 0.55
    cfg.organisms.dash_min_energy_frac = 0.2
    cfg.organisms.far_cap = 32
    if patchy:
        # 13.5 B 臂（食物绑定）：斑块世界才有"荒漠出生 ⇒ 需要赶路"的场景
        c_ = cfg.resources
        c_.distribution = "patchy"
        c_.bg_production_zero = True
        c_.patch_count = 30
        c_.patch_radius = 2
        c_.patch_regrowth_mult = 1.195
        o_ = cfg.organisms
        o_.max_energy = 600.0
        o_.initial_energy = 300.0
        o_.starve_frac = 0.30
        o_.exhaust_frac = 0.17
        o_.eat_efficiency = 7.5
        o_.assim_herb = 0.4
        o_.assim_carn = 0.8
        o_.stomach_cap_mass = 25.0
        o_.eat_threshold_frac = 0.6
        o_.photo_max = 0.0
    cfg.ars.enabled = bool(ars)
    cfg.ars.gain = float(gain)
    cfg.ars.theta = float(theta)
    cfg.ars.kappa = float(kappa)
    cfg.ars.giveup = int(giveup)
    cfg.ars.fast_tau = float(fast_tau)
    cfg.ars.slow_tau = float(slow_tau)
    return SphereEngine(cfg)


def _run(e: SphereEngine, ticks: int = 50):
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    return e


def _digest(e: SphereEngine) -> tuple[int, float]:
    """与 13.8 同一口径（_flat 索引和 + 能量和）⇒ 与既有基线可比。"""
    return (int(e._flat.sum()), round(float(e._energy.sum()), 6))


def test_s1_defaults_are_bitwise_identical() -> None:
    """全默认 vs **显式传默认值** ⇒ 逐位等价（关档不改变旧行为）。"""
    a = _run(_engine(ars=False), 50)
    b = _run(_engine(ars=False, gain=1.0, theta=0.5, kappa=0.0, giveup=20), 50)
    assert _digest(a) == _digest(b), (
        f"显式默认值改变了行为：{_digest(a)} vs {_digest(b)} ⇒ 接线在默认档**不等价**")


def test_s2_matches_c7_baseline() -> None:
    """默认档 = 仓库钉死的 C7 基线（防"孤立等价"）。"""
    d = _digest(_run(_engine(ars=False), 50))
    assert d == (574887, 11266.746993), f"C7 基线漂移：{d}"


def test_s3_config_validates() -> None:
    """`ArsConfig.__post_init__` fail-loud。"""
    with pytest.raises(AssertionError):
        ArsConfig(kappa=0.9)          # κ 上限 0.5（fish D3：总格数不超太多）
    with pytest.raises(AssertionError):
        ArsConfig(fast_tau=500.0, slow_tau=50.0)   # slow 必须 > fast
    with pytest.raises(AssertionError):
        ArsConfig(giveup=0)
    with pytest.raises(AssertionError):
        ArsConfig(theta=0.0)


def test_s4_h3_a1_fail_loud_on_sim_core() -> None:
    """H3 A1：`ars.enabled ∧ use_sim_core=True` ⇒ 构造期硬报错。"""
    with pytest.raises(NotImplementedError):
        _engine(ars=True, use_sim_core=True)


def test_s5_gain_really_takes_effect() -> None:
    """`gain` 真生效：惯性项随 gain 变（读数块数值不同 = 不是空壳）。"""
    p1 = _run(_engine(ars=True, gain=1.0, fast_tau=5.0, slow_tau=50.0, patchy=True), ticks=300).ars_probe()
    p3 = _run(_engine(ars=True, gain=3.0, fast_tau=5.0, slow_tau=50.0, patchy=True), ticks=300).ars_probe()
    assert p1["ars_dec_n"] > 0 and p3["ars_dec_n"] > 0
    assert p1["ars_inertia_sum"] != p3["ars_inertia_sum"]


def test_s6_mutation_sensitivity_open_vs_off() -> None:
    """🔴 DEL-8：开档 ≠ 关档 digest（拆接线 ⇒ 本测试立刻变红）。"""
    off = _digest(_run(_engine(ars=False, patchy=True), 300))
    on = _digest(_run(_engine(ars=True, gain=2.0, fast_tau=5.0, slow_tau=50.0,
                              patchy=True), 300))
    assert on != off, "开档与关档 digest 相同 ⇒ ARS 未接进引擎（no-op）"


def test_s7_antidegeneration_counters_present_and_sane() -> None:
    """反退化读数存在、可解释，且真跑了（`ars_dec_n > 0`）。"""
    p = _run(_engine(ars=True, gain=1.0, fast_tau=5.0, slow_tau=50.0, patchy=True), ticks=300).ars_probe()
    assert p["ars_dec_n"] > 0, "ARS 一次都没求值（空壳接线）"
    assert p["ars_sw_ei_n"] > 0, "从未发生 赶路→驻留 切换 ⇒ 赶路腿没接进进食链"


def test_s8_probe_is_none_when_disabled() -> None:
    """关档 ⇒ `ars_probe()` None（未适用），不是 0。"""
    assert _engine(ars=False).ars_probe() is None


def test_s9_gene_slots_registered() -> None:
    """P7：g20/g21 位号正确且语义已改名（不再是「预留」）。"""
    assert int(Gene.PERSISTENCE) == 20
    assert int(Gene.GIVE_UP) == 21
    assert GENE_SEMANTICS[20] == "赶路惯性（ARS）"
    assert GENE_SEMANTICS[21] == "失望阈值（ARS）"


def test_s10_mode_switching_happens_both_ways() -> None:
    """两腿都活着：驻留→赶路 与 赶路→驻留 都发生过。"""
    p = _run(_engine(ars=True, gain=1.0, fast_tau=5.0, slow_tau=50.0, patchy=True), ticks=300).ars_probe()
    assert p["ars_sw_ie_n"] > 0, "从未 驻留→赶路"
    assert p["ars_sw_ei_n"] > 0, "从未 赶路→驻留"
