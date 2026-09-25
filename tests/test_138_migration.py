"""13.8 日历—罗盘式定向迁徙（g23）—— 回归单测。

⚠️ 本文件的**存在理由**（照 R197 V2 的教训）：13.7 交付时零单测 ⇒ DEL-8
（变异测试：改坏接线 ⇒ 测试必须变红）**结构性无法执行**。13.8 一开始就把
"关档等价 / 真生效 / H3 双守卫 / 变异敏感性"四条写成常驻断言。

覆盖：
    S1 关档逐位等价：全默认 vs 显式传默认 ⇒ digest 完全相同
    S2 C7 基线不动：默认档 vs 钉死基线 `(574887, 11266.746993)`
    S3 `photoperiod` 解析式**真生效**：纬度结构（N/S 异号）+ 季节翻转
    S4 `MigrationConfig` 字段校验（`__post_init__`，fail-loud）
    S5 **H3 M1**：`migration.enabled ∧ use_sim_core=True` ⇒ NotImplementedError
    S6 **H3 M2**：`migration.enabled ∧ 无季节` ⇒ ValueError（无季节 ⇒ A≡0 ⇒ 假阴性）
    S7 `gain` 与 `min_abs_anomaly` **真生效**：读数块数值随之变（不是"字段存在"）
    S8 🔴 **DEL-8 变异敏感性（逐机制）**：开档 ≠ 关档 digest
    S9 反退化读数存在且**可解释**：`mig_flat_frac`/`mig_zero_frac`/`mig_skip_frac`
       —— 且 `mig_dec_n > 0`（真跑了，不是空壳）
    S10 P7：`Gene.MIGRATE_BIAS == 23` 且 `GENE_SEMANTICS[23]` 已改名（不再是「预留」）

设计原则（教训库）：**默认关 = 旧行为逐位一致**（S1/S2 钉死）；每条只改一个参数、
断言**可观测的行为差异**（不是"字段存在"）——避免"有测试却不验不变量"（A#8）。
"""

from __future__ import annotations

import numpy as np
import pytest

from simulation.config import InfoStructureConfig, MigrationConfig, SimConfig
from simulation.genes import GENE_SEMANTICS, Gene
from simulation.sphere_engine import SphereEngine

ROT = 2400
SEASON = 4 * ROT
TILT = float(np.deg2rad(23.44))


def _engine(*, tilt_rad: float = 0.0, season_period: int = 0,
            migration: bool = False, gain: float = 50.0, min_abs: float = 0.0,
            use_sim_core: bool = False, max_count: int = 600) -> SphereEngine:
    """构造引擎；迁徙走**配置字段**（与 a4 的 CLI 同一条路径）。"""
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
    cfg.light.tilt_rad = float(tilt_rad)
    cfg.light.season_period = int(season_period)
    cfg.migration.enabled = bool(migration)
    cfg.migration.gain = float(gain)
    cfg.migration.min_abs_anomaly = float(min_abs)
    return SphereEngine(cfg)


def _run(e: SphereEngine, ticks: int = 50) -> SphereEngine:
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    return e


def _digest(e: SphereEngine) -> tuple[int, float]:
    return (int(e._flat.sum()), round(float(e._energy.sum()), 6))


def _hemi_pp(e: SphereEngine, t: int) -> tuple[float, float]:
    """t 时刻的 (北半球平均日长, 南半球平均日长)。

    ⚠️ `world.latitude_of()` 吃的是**纬度行号**（不是平铺索引）⇒ 走 `flat_to_rc`。
    """
    flat = np.arange(e.world.n_cells)
    rows_idx, _ = e.world.flat_to_rc(flat)
    lat = np.asarray(e.world.latitude_of(rows_idx), dtype=float)
    pp = np.asarray(e.light.photoperiod(flat, t), dtype=float)
    north = lat > 0.0
    return float(pp[north].mean()), float(pp[~north].mean())


# --------------------------------------------------------------------------- S1
def test_s1_defaults_are_bitwise_identical() -> None:
    """全默认 vs **显式传默认值** ⇒ 逐位等价（关档不改变旧行为）。"""
    a = _run(_engine(), 50)
    b = _run(_engine(tilt_rad=0.0, season_period=0, migration=False, gain=50.0), 50)
    assert _digest(a) == _digest(b), (
        f"显式默认值改变了行为：{_digest(a)} vs {_digest(b)} ⇒ 接线在默认档**不等价**")


def test_s1b_matches_c7_baseline() -> None:
    """默认档 = 仓库钉死的 C7 基线（防"孤立等价"）。"""
    d = _digest(_run(_engine(), 50))
    assert d == (574887, 11266.746993), (
        f"C7 基线漂移：{d} ≠ (574887, 11266.746993) ⇒ 13.8 接线动了默认档")


# --------------------------------------------------------------------------- S3
def test_s3_photoperiod_has_lat_structure_and_flips() -> None:
    """`photoperiod` **真生效**：纬度结构 + 季节翻转。

    判据（强于"数组不同"）：
      · 关档：P ≡ 0.5（处处、处处）——**无纬度结构**；
      · 开档 t=P/4（北半球夏）：北 > 0.5 > 南；
      · 开档 t=3P/4（南半球夏）：南 > 0.5 > 北 ⇒ **符号翻转**。
    且南北均值关于 0.5 **严格对称**（解析式的必然结果，可当不变量）。
    """
    off = _engine()
    n0, s0 = _hemi_pp(off, SEASON // 4)
    assert abs(n0 - 0.5) < 1e-12 and abs(s0 - 0.5) < 1e-12, (
        f"关档时日长应恒 0.5（无季节 ⇒ δ≡0），实得 北{n0} 南{s0}")

    on = _engine(tilt_rad=TILT, season_period=SEASON)
    nA, sA = _hemi_pp(on, SEASON // 4)
    nB, sB = _hemi_pp(on, 3 * SEASON // 4)
    assert nA > 0.5 > sA, f"t=P/4 应北长南短：北{nA:.4f} 南{sA:.4f}"
    assert sB > 0.5 > nB, f"t=3P/4 应南长北短：北{nB:.4f} 南{sB:.4f}"
    # 南北对称（解析式的硬不变量）
    assert abs((nA + sA) - 1.0) < 1e-9, f"南北日长应关于 0.5 对称：{nA + sA}"


def test_s3b_polar_day_and_night() -> None:
    """极区在夏季应出现**极昼（P=1）/极夜（P=0）** —— 解析式的边界行为。"""
    on = _engine(tilt_rad=TILT, season_period=SEASON)
    flat = np.arange(on.world.n_cells)
    pp = np.asarray(on.light.photoperiod(flat, SEASON // 4), dtype=float)
    assert pp.max() > 0.999, f"应有极昼（P→1），实得 max={pp.max():.6f}"
    assert pp.min() < 0.001, f"应有极夜（P→0），实得 min={pp.min():.6f}"


def test_s3c_no_nan_or_out_of_range_over_full_cycle() -> None:
    """P3：全周期抽样 ⇒ 无 NaN、无越界（极点 `tan` 守卫真在起作用）。"""
    on = _engine(tilt_rad=TILT, season_period=SEASON)
    flat = np.arange(on.world.n_cells)
    for t in range(0, SEASON + 1, max(1, SEASON // 24)):
        pp = np.asarray(on.light.photoperiod(flat, t), dtype=float)
        assert np.isfinite(pp).all(), f"t={t} 出现非有限值"
        assert (pp >= 0.0).all() and (pp <= 1.0).all(), (
            f"t={t} 日长越界：[{pp.min()}, {pp.max()}]")


# --------------------------------------------------------------------------- S4
def test_s4_migration_config_validates() -> None:
    """`MigrationConfig.__post_init__` **fail-loud**（构造即校验）。"""
    with pytest.raises(AssertionError):
        MigrationConfig(gain=-1.0)
    with pytest.raises(AssertionError):
        MigrationConfig(min_abs_anomaly=0.6)      # A∈[−0.5,0.5]
    with pytest.raises(AssertionError):
        MigrationConfig(min_abs_anomaly=-0.1)
    ok = MigrationConfig(enabled=True, gain=20.0, min_abs_anomaly=0.05)
    assert ok.enabled and ok.gain == 20.0


# --------------------------------------------------------------------------- S5
def test_s5_h3_m1_fail_loud_on_sim_core() -> None:
    """H3 **M1**：`migration ∧ use_sim_core=True` ⇒ 构造期硬报错。"""
    with pytest.raises(NotImplementedError) as ei:
        _engine(tilt_rad=TILT, season_period=SEASON, migration=True,
                use_sim_core=True)
    assert "migration" in str(ei.value), f"报错信息未指明 migration：{ei.value}"


# --------------------------------------------------------------------------- S6
def test_s6_h3_m2_fail_loud_without_season() -> None:
    """H3 **M2**：`migration ∧ 无季节` ⇒ ValueError。

    🔴 为什么必须单独拦（比 M1 更隐蔽）：无季节 ⇒ δ(t)≡0 ⇒ P≡0.5（**与纬度无关**）
    ⇒ 日历轴异常 A≡0 ⇒ 迁移项逐候选恒 0 ⇒ argmax 逐位不变 ⇒
    实验会读出「迁徙无效」的**假阴性**（错在实验设计，不在机制）。
    """
    with pytest.raises(ValueError) as ei:
        _engine(migration=True)                       # 无季节
    assert "season" in str(ei.value).lower(), f"报错信息未指明 season：{ei.value}"
    # 只给一半（有 tilt 无 period）也算"无季节"⇒ 同样拦
    with pytest.raises(ValueError):
        _engine(tilt_rad=TILT, season_period=0, migration=True)


# --------------------------------------------------------------------------- S7
def test_s7_gain_and_min_abs_really_take_effect() -> None:
    """`gain` 与 `min_abs_anomaly` **真生效**：读数块数值随之变。"""
    def probe(**kw):
        e = _run(_engine(tilt_rad=TILT, season_period=SEASON, migration=True, **kw), 60)
        return e.migration_probe()

    lo = probe(gain=2.0)
    hi = probe(gain=40.0)
    assert lo["mig_term_abs_mean"] is not None
    assert hi["mig_term_abs_mean"] > lo["mig_term_abs_mean"], (
        f"gain 未生效：gain=40 的量级 {hi['mig_term_abs_mean']} "
        f"不大于 gain=2 的 {lo['mig_term_abs_mean']}")

    # min_abs_anomaly 抬高 ⇒ 更多个体被跳过
    gated = probe(gain=20.0, min_abs=0.45)
    assert gated["mig_skip_frac"] > 0.0, (
        "min_abs_anomaly=0.45 应跳过绝大多数个体（|A|≤0.5）⇒ skip_frac 应为正")
    assert gated["mig_skip_frac"] > probe(gain=20.0, min_abs=0.0)["mig_skip_frac"]


# --------------------------------------------------------------------------- S8
def test_s8_mutation_sensitivity_open_vs_off() -> None:
    """🔴 **DEL-8 变异测试**：开档 ≠ 关档 digest（按 **gain 逐档**各测一条）。

    🔴 **为什么逐档**（13.7 的实测教训）：若只测"开档 vs 关档"，则把 `gain` 的
    透传写死成某个非零常数**仍会变红**（因为 term 还在），但把 gain 写成
    **另一个非零值**就测不出来。⇒ 用 **两个不同 gain 的 digest 互不相同**
     来守护 `gain` 真的从 config 走进了热路径。
    """
    off = _digest(_run(_engine(tilt_rad=TILT, season_period=SEASON), 60))
    on_a = _digest(_run(_engine(tilt_rad=TILT, season_period=SEASON,
                                migration=True, gain=2.0), 60))
    on_b = _digest(_run(_engine(tilt_rad=TILT, season_period=SEASON,
                                migration=True, gain=40.0), 60))
    assert on_a != off, (
        f"**开迁徙**时 digest 与关档相同（{off}）⇒ 机制**未接进引擎**（no-op）："
        "检查 `sphere_engine._step_population` 的迁移项与 `__slots__` 登记")
    assert on_a != on_b, (
        f"**不同 gain** 的 digest 相同（{on_a}）⇒ `gain` 未真正进入热路径"
        "（被写死成常数）")


# --------------------------------------------------------------------------- S9
def test_s9_antidegeneration_counters_present_and_sane() -> None:
    """反退化读数（R148-1）存在、可解释，且**真跑了**（`mig_dec_n > 0`）。"""
    e = _run(_engine(tilt_rad=TILT, season_period=SEASON,
                     migration=True, gain=20.0), 80)
    p = e.migration_probe()
    assert p is not None, "开档时 migration_probe 不应为 None（'没测'≠'测出零'）"
    for k in ("mig_flat_frac", "mig_zero_frac", "mig_skip_frac",
              "mig_term_abs_mean", "mig_dec_n"):
        assert k in p, f"读数块缺字段 {k}"
    assert p["mig_dec_n"] > 0, "mig_dec_n=0 ⇒ 迁移项一次都没求值（空壳接线）"
    # 反退化占比必须是**比例**（∈[0,1]），且不能恒等于 1（= 跨候选全同值 = 逐位 no-op）
    assert 0.0 <= p["mig_flat_frac"] <= 1.0
    assert p["mig_flat_frac"] < 1.0, (
        "mig_flat_frac=1.0 ⇒ 项对**所有**个体都跨候选取同值 ⇒ 逐位 no-op（R148-1）")
    assert p["mig_zero_frac"] < 0.5, "mig_zero_frac 过高 ⇒ 项形同自熄（P4）"


def test_s9b_probe_is_none_when_disabled() -> None:
    """关档 ⇒ `migration_probe()` **None**（未适用），**不是 0**（R120/§五.12）。"""
    assert _run(_engine(), 10).migration_probe() is None


# --------------------------------------------------------------------------- S10
def test_s10_gene_slot_registered() -> None:
    """P7：g23 位号 = 23 且语义已改名（不再是「预留」）。"""
    assert int(Gene.MIGRATE_BIAS) == 23
    assert "预留" not in GENE_SEMANTICS[23], (
        f"GENE_SEMANTICS[23] 仍是预留：{GENE_SEMANTICS[23]!r}")
    assert "迁徙" in GENE_SEMANTICS[23]
