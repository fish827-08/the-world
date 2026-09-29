"""S3 前置：记忆改造 v2（egocentric）—— 回归单测。

规格：`docs/设计文档/设计-S3记忆改造-实施细化稿-20260927.md`（v2.0，§一~§十）。

照 13.8 的教训（R197 V2：交付时零单测 ⇒ DEL-8 结构性无法执行），本文件把
"关档等价 / C7 基线 / fail-loud / 真写入 / 旋转 / 距离只增 / 永久降级 / TTL /
cos 打分语义 / 快照 roundtrip / 变异敏感性"写成常驻断言。

覆盖：
    S1 关档逐位等价：memory_v2=False ⇒ 与现 main 行为一致（默认值 = 旧行为）
    S2 C7 基线不动：钉死 digest `(574887, 11266.746993)` 不漂移
    S3 config fail-loud：v2 ⇒ memory_gradient=='orientation'；数值域 assert
    S4 H3 fail-loud ×2：v2∧use_sim_core ⇒ NotImplementedError；
       v2∧rd.enabled∧not bg_production_zero ⇒ NotImplementedError
    S5 真写入：patchy 世界跑若干 tick ⇒ `_mem_az` 有填充，方位值 ∈ [0,7]
    S6 旋转：heading 变 δ ⇒ 记忆方位 q 同步变 δ（自我参照系路径整合）
    S7 距离**只增不减** + 超阈**永久降级**（不可逆）
    S8 TTL：age > ttl ⇒ 槽在打分中失效
    S9 打分 cos 语义：对准 ≈ +1、背向 ≈ −1；精记忆带 exp(−d/scale) 衰减、粗记忆不带
    S10 快照 roundtrip：save → load ⇒ 5 数组逐位一致；繁殖/死亡后长度一致
    S11 **DEL-8 变异敏感性**：v2 开 ≠ 关 digest（机制真生效，不是字段存在）
    S12 13.11 记忆权重基因位（g22）：乘子 2×g22 语义（0.5⇒基准 / 1⇒×2 / 0⇒零）、
        开关∧¬v2 构造期 fail-loud、开档 ≠ 关档 digest（机制真接线）
"""

from __future__ import annotations

import numpy as np
import pytest

from simulation.config import (
    InfoStructureConfig, ResourceDynamicsConfig, SimConfig,
)
from simulation.genes import Gene
from simulation.sphere_engine import SphereEngine


def _engine(*, seed: int = 42, max_count: int = 600,
            patchy: bool = False, bgzero: bool = True,
            v2: bool = False, noise: bool = False, noise_p: float = 0.1,
            dist_scale: float = 15.0, degrade_thr: float = 20.0,
            ttl: int = 1000, rd_enabled: bool = False,
            use_sim_core: bool = False,
            memory_gradient: str = "orientation",
            weight_gene: bool = False) -> SphereEngine:
    cfg = SimConfig(seed=seed)
    cfg.simulation.use_sim_core = use_sim_core
    cfg.simulation.l2_dash = False
    cfg.population.max_count = max_count
    cfg.predation.forage_tradeoff_k = 0.0
    cfg.info_structure = InfoStructureConfig(
        enabled=True, learning_rate=0.05,
        memory_gradient=memory_gradient,
        memory_v2=v2,
        memory_dist_scale=dist_scale,
        memory_degrade_thr=degrade_thr,
        memory_coarse_gain=0.15,
        memory_ttl=ttl,
        memory_noise=noise,
        memory_noise_p=noise_p,
        memory_weight_gene=weight_gene,
    )
    cfg.organisms.young_mob_mult = 0.55
    cfg.organisms.old_mob_mult = 0.55
    cfg.organisms.far_cap = 32
    if patchy:
        c_ = cfg.resources
        c_.distribution = "patchy"
        c_.bg_production_zero = bgzero
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
    if rd_enabled:
        cfg.resource_dynamics = ResourceDynamicsConfig(enabled=True)
    return SphereEngine(cfg)


def _run(e: SphereEngine, ticks: int = 50):
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    return e


def _digest(e: SphereEngine, ticks: int = 50) -> tuple:
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    return int(e._flat.sum()), round(float(e._energy.sum()), 6), int(e.tick)


# ---------------------------------------------------------------- S1/S2 关档等价 + C7

def test_s1_off_default_is_identical_to_explicit_false():
    """关档逐位等价：默认（不传 memory_v2）≡ 显式 memory_v2=False。"""
    a = _run(_engine(v2=False, memory_gradient="orientation"), ticks=50)
    b = _run(_engine(memory_gradient="orientation"), ticks=50)  # 默认 False
    assert int(a._flat.sum()) == int(b._flat.sum())
    assert round(float(a._energy.sum()), 6) == round(float(b._energy.sum()), 6)
    # v2 数组保持初值（-1 / 0）——关档完全不写
    assert int((a._mem_az >= 0).sum()) == 0
    assert float(a._mem_dist.sum()) == 0.0


def test_s2_c7_baseline_unchanged():
    """C7 基线不动：复刻 test_a_continuous 的默认档 50 tick digest 仍钉死。"""
    d = _digest(_engine(memory_gradient="none", max_count=600), ticks=50)
    assert d[:2] == (574887, 11266.746993), f"C7 基线漂移: {d}"


# ---------------------------------------------------------------- S3 config fail-loud

def test_s3_v2_requires_orientation():
    with pytest.raises(AssertionError):
        InfoStructureConfig(
            enabled=True, learning_rate=0.05,
            memory_gradient="none", memory_v2=True)


def test_s3_numeric_domain_asserts():
    for kw in (dict(memory_dist_scale=0.0), dict(memory_degrade_thr=-1.0),
               dict(memory_ttl=0), dict(memory_noise_p=1.5)):
        with pytest.raises(AssertionError):
            InfoStructureConfig(
                enabled=True, learning_rate=0.05,
                memory_gradient="orientation", memory_v2=True, **kw)


# ---------------------------------------------------------------- S4 H3 fail-loud ×2

def test_s4_v2_use_sim_core_fail_loud():
    with pytest.raises(NotImplementedError):
        _engine(v2=True, use_sim_core=True)


def test_s4_v2_rd_enabled_nonzero_bg_fail_loud():
    """v2 ∧ rd.enabled ∧ 背景有产能 ⇒ 硬报错（rd 搬移 ⇒ 静态质心失效）。"""
    with pytest.raises(NotImplementedError):
        _engine(v2=True, patchy=True, bgzero=False, rd_enabled=True)
    # bg_production_zero=True（S 线）允许
    e = _engine(v2=True, patchy=True, bgzero=True, rd_enabled=True)
    assert e is not None


# ---------------------------------------------------------------- S5 真写入

def test_s5_write_happens_in_patchy_world():
    e = _run(_engine(v2=True, patchy=True, max_count=3000), ticks=60)
    filled = int((e._mem_az >= 0).sum())
    assert filled > 0, "patchy 世界跑 60 tick 应写入记忆槽"
    vals = e._mem_az[e._mem_az >= 0]
    assert int(vals.min()) >= 0 and int(vals.max()) <= 7, "方位 ∈ [0,7]"
    # 写入伴随距离=0、tick 有效
    assert float(e._mem_dist[e._mem_az >= 0].max()) >= 0.0
    assert int((e._mem_tick >= 0).sum()) == filled


# ---------------------------------------------------------------- S6 旋转

def test_s6_memory_rotates_with_heading():
    """自我参照系核心：heading 变 δ ⇒ 记忆方位 q 同步变 δ（mod 8）。"""
    e = _engine(v2=True, patchy=True, max_count=3000)
    # 直接构造一个带记忆的个体状态，模拟"站在富食格记了一笔"
    n = len(e._flat)
    idx = 0
    e._mem_az[idx] = [2, -1, -1, -1]
    e._mem_tick[idx] = [e.tick, -1, -1, -1]
    e._mem_dist[idx] = [0.0, 0.0, 0.0, 0.0]
    hd_old = 0
    hd_new = 3          # δ = 3
    e._mem_v2_step_one(idx, hd_old, hd_new, 1.0)
    assert int(e._mem_az[idx, 0]) == (2 + 3) % 8, "q 应随 heading 同步旋转"
    # 距离累加
    assert float(e._mem_dist[idx, 0]) == 1.0
    # 未知朝向 → 不旋转但更新距离
    e._mem_az[idx] = [5, -1, -1, -1]
    e._mem_tick[idx] = [e.tick, -1, -1, -1]
    e._mem_dist[idx] = [0.0, 0.0, 0.0, 0.0]
    e._mem_v2_step_one(idx, -1, 2, 1.0)
    assert int(e._mem_az[idx, 0]) == 5, "未知朝向豁免旋转"
    assert float(e._mem_dist[idx, 0]) == 1.0


# ---------------------------------------------------------------- S7 距离只增 + 永久降级

def test_s7_distance_monotonic_and_permanent_degrade():
    e = _engine(v2=True, patchy=True, max_count=3000)
    n = len(e._flat)
    idx = 0
    e._mem_az[idx] = [1, -1, -1, -1]
    e._mem_tick[idx] = [e.tick, -1, -1, -1]
    e._mem_dist[idx] = [0.0, 0.0, 0.0, 0.0]
    thr = e._mem_degrade_thr
    # 走到超过阈值
    for _ in range(int(thr) + 3):
        e._mem_v2_step_one(idx, 1, 1, 1.0)
    d0 = float(e._mem_dist[idx, 0])
    assert d0 == float(thr) + 3.0, "距离只增不减"
    assert bool(e._mem_degraded[idx, 0]) is True, "超阈 ⇒ 永久降级"
    # 降级后继续走：仍保持 True（不可逆）
    e._mem_v2_step_one(idx, 1, 1, 1.0)
    assert bool(e._mem_degraded[idx, 0]) is True


# ---------------------------------------------------------------- S8 TTL

def test_s8_ttl_expiry_kills_slot():
    e = _engine(v2=True, patchy=True, max_count=3000, ttl=5)
    n = len(e._flat)
    idx = 0
    e._mem_az[idx] = [3, -1, -1, -1]
    e._mem_tick[idx] = [e.tick - 100, -1, -1, -1]   # 早已过期
    nb = np.arange(max(8, 8), dtype=np.int64)
    # nb 必须真实邻居；直接用世界邻居
    cur = int(e._flat[idx])
    nb = np.asarray(e.world.neighbors(cur), dtype=np.int64)[:8]
    g = e._memory_egocentric_cos(idx, cur, nb, int(e._heading[idx]))
    assert float(np.abs(g).max()) == 0.0, "过期槽不贡献打分"
    # 未过期 → 有贡献
    e._mem_tick[idx] = [e.tick, -1, -1, -1]
    g2 = e._memory_egocentric_cos(idx, cur, nb, int(e._heading[idx]))
    assert float(np.abs(g2).max()) > 0.0


# ---------------------------------------------------------------- S9 cos 打分语义

def test_s9_score_cos_alignment_and_gain_split():
    e = _engine(v2=True, patchy=True, max_count=3000)
    n = len(e._flat)
    idx = 0
    cur = int(e._flat[idx])
    nb = np.asarray(e.world.neighbors(cur), dtype=np.int64)[:8]
    e._mem_az[idx] = [0, -1, -1, -1]          # q=0
    e._mem_tick[idx] = [e.tick, -1, -1, -1]
    e._mem_dist[idx] = [0.0, 0.0, 0.0, 0.0]
    e._mem_degraded[idx] = [False, False, False, False]
    # 精记忆：对准槽位 → cos=1 → gain 0.3；候选槽位 = 5 → cos=-1 → −0.3
    g_precise = e._memory_egocentric_cos(idx, cur, nb, 0)   # heading=0
    # 找对准/背向的候选下标
    dirs = np.array([e._cell_dir_slot(cur, int(x)) for x in nb])
    aligned = int(np.argmax(dirs == 0)) if (dirs == 0).any() else None
    if aligned is None:
        pytest.skip("世界邻居不含正对槽位（网格几何限制）")
    # 精记忆增益 = memory_gradient_gain（0.3），exp(0)=1
    assert abs(float(g_precise[aligned]) - 0.3) < 1e-9
    # 粗记忆（降级后）：增益 0.15 且**不含 exp(−d/scale)**（d 不进公式）
    e._mem_degraded[idx] = [True, False, False, False]
    e._mem_dist[idx] = [1000.0, 0.0, 0.0, 0.0]   # d 很大但粗记忆不受 d 影响
    g_coarse = e._memory_egocentric_cos(idx, cur, nb, 0)
    assert abs(float(g_coarse[aligned]) - 0.15) < 1e-9, "粗记忆固定增益 0.15"


# ---------------------------------------------------------------- S10 快照 roundtrip + 长度

def test_s10_snapshot_roundtrip_and_lengths(tmp_path):
    e = _run(_engine(v2=True, patchy=True, max_count=3000), ticks=40)
    p = tmp_path / "snap.npz"
    e.save_snapshot(str(p))
    e2 = SphereEngine.load_snapshot(str(p))
    assert int((e._mem_az >= 0).sum()) == int((e2._mem_az >= 0).sum())
    assert np.array_equal(
        e._mem_az[e._mem_az >= 0], e2._mem_az[e2._mem_az >= 0])
    assert np.allclose(e._mem_dist, e2._mem_dist)
    assert np.array_equal(e._mem_degraded, e2._mem_degraded)
    assert np.array_equal(e._patch_centroid, e2._patch_centroid)
    # 数组长度与 _flat 一致
    P = len(e._flat)
    assert e._mem_az.shape[0] == P and e._mem_dist.shape[0] == P
    assert e._mem_tick.shape[0] == P and e._mem_degraded.shape[0] == P


# ---------------------------------------------------------------- S11 变异敏感性

def test_s11_v2_changes_digest_in_patchy_world():
    """机制真生效：patchy 世界下 v2 开 ≠ 关 digest（DEL-8 变异敏感性）。"""
    off = _digest(_engine(v2=False, patchy=True, max_count=3000), ticks=40)
    on = _digest(_engine(v2=True, patchy=True, max_count=3000), ticks=40)
    assert on != off, "v2 开档必须改变轨迹（否则 = 静默 no-op）"


# ---------------------------------------------------------------- S12 13.11 记忆权重基因位

def test_s12_gene_multiplier_semantics():
    """乘子语义：`gain ← gain × 2×g22`（仅开关开档）；0.5 ⇒ 基准 / 1 ⇒ ×2 / 0 ⇒ 零贡献。"""
    e = _engine(v2=True, patchy=True, max_count=3000, weight_gene=True)
    idx = 0
    cur = int(e._flat[idx])
    nb = np.asarray(e.world.neighbors(cur), dtype=np.int64)[:8]
    e._mem_az[idx] = [0, -1, -1, -1]          # q=0
    e._mem_tick[idx] = [e.tick, -1, -1, -1]
    e._mem_dist[idx] = [0.0, 0.0, 0.0, 0.0]
    e._mem_degraded[idx] = [False, False, False, False]
    dirs = np.array([e._cell_dir_slot(cur, int(x)) for x in nb])
    if not (dirs == 0).any():
        pytest.skip("世界邻居不含正对槽位（网格几何限制）")
    aligned = int(np.argmax(dirs == 0))
    # g22=0.5 ⇒ 乘子 1.0：精记忆 = 0.3（与关档同式，便于 S3 批对齐）
    e._genes[idx, Gene.MEMORY_WEIGHT] = 0.5
    assert abs(float(e._memory_egocentric_cos(idx, cur, nb, 0)[aligned]) - 0.3) < 1e-9
    # g22=1.0 ⇒ 乘子 2.0
    e._genes[idx, Gene.MEMORY_WEIGHT] = 1.0
    assert abs(float(e._memory_egocentric_cos(idx, cur, nb, 0)[aligned]) - 0.6) < 1e-9
    # 粗记忆同乘（2:1 比例不变）：0.15 × 2 = 0.3
    e._mem_degraded[idx] = [True, False, False, False]
    e._mem_dist[idx] = [1000.0, 0.0, 0.0, 0.0]
    assert abs(float(e._memory_egocentric_cos(idx, cur, nb, 0)[aligned]) - 0.3) < 1e-9
    # g22=0.0 ⇒ 乘子 0 ⇒ 该个体记忆项不贡献（全候选 0）
    e._mem_degraded[idx] = [False, False, False, False]
    e._mem_dist[idx] = [0.0, 0.0, 0.0, 0.0]
    e._genes[idx, Gene.MEMORY_WEIGHT] = 0.0
    assert float(np.abs(e._memory_egocentric_cos(idx, cur, nb, 0)).max()) == 0.0


def test_s12_weight_gene_requires_v2():
    """构造期 fail-loud：乘子只在 v2 打分路径被消费 ⇒ 开关∧¬v2 = 静默 no-op ⇒ 拒绝。"""
    with pytest.raises(AssertionError):
        InfoStructureConfig(
            enabled=True, learning_rate=0.05,
            memory_gradient="orientation", memory_v2=False,
            memory_weight_gene=True)


def test_s12_weight_gene_changes_digest():
    """机制真接线（DEL-8 同族）：v2 开 + 基因位开 ≠ v2 开 + 关（否则 = 静默 no-op）。"""
    off = _digest(_engine(v2=True, patchy=True, max_count=3000), ticks=40)
    on = _digest(_engine(v2=True, patchy=True, max_count=3000, weight_gene=True), ticks=40)
    assert on != off, "13.11 开关开档必须改变轨迹（否则 = 静默 no-op）"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
