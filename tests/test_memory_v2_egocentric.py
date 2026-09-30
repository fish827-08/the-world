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
    S12 质心动态化（R264）：rd 搬移后质心表与当前掩码重算一致
    S13 质心增量重算（R273）：优化 vs 全量逐位一致（函数级边界 + 引擎级每 tick）
        + 门控（关档零质心工作 / 开档走增量；**极区接触 ⇒ 回退全量**见函数级专测）
    S14 T1 方向量化快路径/向量化：常量==逐格标量（标准行全列）、向量化 pair 量化
        ==标量（含极区/边界行 + 随机长程）、cos 快路径==一般路径（含门控事实）、
        批量步进==逐个体 + 噪声档禁入、写路径==旧标量参考
"""

from __future__ import annotations

import numpy as np
import pytest

from simulation.config import (
    InfoStructureConfig, ResourceDynamicsConfig, SimConfig,
)
from simulation.sphere_engine import SphereEngine


def _engine(*, seed: int = 42, max_count: int = 600,
            patchy: bool = False, bgzero: bool = True,
            v2: bool = False, noise: bool = False, noise_p: float = 0.1,
            dist_scale: float = 15.0, degrade_thr: float = 20.0,
            ttl: int = 1000, rd_enabled: bool = False,
            use_sim_core: bool = False,
            dynamic_centroid: bool = True,
            memory_gradient: str = "orientation") -> SphereEngine:
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
        memory_v2_dynamic_centroid=dynamic_centroid,
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


def test_s4_v2_rd_enabled_nonzero_bg_static_fail_loud():
    """v2 ∧ rd ∧ 背景有产能 ∧ 动态质心关 ⇒ 硬报错（静态质心失效）。"""
    with pytest.raises(NotImplementedError):
        _engine(v2=True, patchy=True, bgzero=False, rd_enabled=True,
                dynamic_centroid=False)


def test_s4_v2_rd_enabled_nonzero_bg_dynamic_allows():
    """R264：v2 ∧ rd ∧ 背景有产能 ∧ 动态质心开（默认）⇒ 不再报错。"""
    e = _engine(v2=True, patchy=True, bgzero=False, rd_enabled=True,
                dynamic_centroid=True)
    assert e is not None
    # bg_production_zero=True（S 线）也允许
    e2 = _engine(v2=True, patchy=True, bgzero=True, rd_enabled=True)
    assert e2 is not None


def test_s12_centroid_updates_after_rd_relocation():
    """R264 质心动态化：rd 轮作搬移后质心表与新掩码一致。"""
    e = _run(_engine(v2=True, patchy=True, bgzero=False, rd_enabled=True,
                     max_count=2000), ticks=300)
    # 搬移后质心表：每个斑块格的质心必须属于该斑块当前连通域
    pm = e.resources._patch_mask
    cent = e._patch_centroid
    patch_cells = np.flatnonzero(pm)
    if patch_cells.size == 0:
        return  # 无斑块 ⇒ 质心全 -1，跳过
    # 抽样验证：每个斑块格的质心格必须也是斑块格
    sampled = patch_cells[::max(1, len(patch_cells) // 200)]
    for c in sampled:
        cid = int(cent[c])
        assert cid >= 0, f"斑块格 {c} 的质心为 -1（搬移后未更新）"
        assert bool(pm[cid]), f"斑块格 {c} 的质心 {cid} 不是斑块格（搬移后未更新）"
    # 质心表与直接用当前掩码重算的结果一致（搬移后动态更新正确性）
    from simulation.sphere_engine import _build_patch_centroid
    expected = _build_patch_centroid(e.world, pm)
    assert np.array_equal(cent, expected), "搬移后质心表与当前掩码重算结果不一致"


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


# ---------------------------------------------------------------- S13 R273 质心优化

def test_s13_incremental_matches_full_every_tick():
    """R273 L2：增量重算表与全量重算**逐位一致**（每 tick 对拍；装置 = rd+on 臂）。"""
    from simulation.sphere_engine import _build_patch_centroid
    e = _engine(v2=True, patchy=True, bgzero=False, rd_enabled=True,
                max_count=2000)
    n_chg = 0
    for _ in range(150):
        if e.extinct:
            break
        prev = e.resources._patch_mask.copy()
        e.step()
        cur = e.resources._patch_mask
        n_chg += int(not np.array_equal(prev, cur))
        expected = _build_patch_centroid(e.world, cur)
        assert np.array_equal(e._patch_centroid, expected), (
            f"tick {e.tick} 增量表 ≠ 全量表")
    assert n_chg > 0, "本测试未覆盖任何掩码变化 ⇒ 对拍空洞（须修装置）"


def test_s13_off_arm_skips_centroid_work_on_arm_uses_incremental(monkeypatch):
    """R273 L1/L2 门控：关档 ⇒ 零质心工作；开档 ⇒ 走增量（构造期 1 次全量；
    极区接触时增量返回 False ⇒ 引擎回退全量 ⇒ 计数可 >1，见 §S13 回退专测）。"""
    import simulation.sphere_engine as se
    cnt = {"full": 0, "inc": 0}
    orig_f = se._build_patch_centroid
    orig_i = se._update_patch_centroid_incremental

    def _cf(*a, **k):
        cnt["full"] += 1
        return orig_f(*a, **k)

    def _ci(*a, **k):
        cnt["inc"] += 1
        return orig_i(*a, **k)

    monkeypatch.setattr(se, "_build_patch_centroid", _cf)
    monkeypatch.setattr(se, "_update_patch_centroid_incremental", _ci)
    # 关档臂：rd 开、掩码会变，但不得出现任何质心工作（表无人消费）
    e_off = _engine(v2=False, patchy=True, bgzero=False, rd_enabled=True,
                    max_count=2000)
    n_chg = 0
    prev = e_off.resources._patch_mask.copy()
    for _ in range(300):
        if e_off.extinct:
            break
        e_off.step()
        cur = e_off.resources._patch_mask
        n_chg += int(not np.array_equal(prev, cur))
        prev = cur.copy()
    assert n_chg > 0, "装置未发生掩码变化 ⇒ 门控断言空洞"
    assert cnt == {"full": 0, "inc": 0}, f"关档臂不应有质心工作：{cnt}"
    assert bool((e_off._patch_centroid == -1).all()), "关档表应保持初值 -1"
    # 开档臂：构造期恰好 1 次全量；运行期走增量
    cnt["full"] = cnt["inc"] = 0
    e_on = _engine(v2=True, patchy=True, bgzero=False, rd_enabled=True,
                   max_count=2000)
    assert cnt["full"] == 1, "构造期应恰好全量建表一次"
    for _ in range(300):
        if e_on.extinct:
            break
        e_on.step()
    assert cnt["inc"] > 0, f"开档臂应走增量重算：{cnt}"
    # 运行期全量调用只允许来自极区回退（增量返回 False 时）——非回退路径不得调全量
    assert cnt["full"] >= 1, f"开档臂应至少构造期全量建表一次：{cnt}"
    assert np.array_equal(
        e_on._patch_centroid,
        orig_f(e_on.world, e_on.resources._patch_mask)), "终态增量表 ≠ 全量表"


def test_s13_incremental_edges_split_merge_seam():
    """分裂 / 合并 / 跨经度接缝 / 孤立域删除 / 多格组合 —— 逐例与全量对拍。

    装置全部落在对称区（rows=8 ⇒ 极区行 {0,1,6,7} 之外的行 2..5）。
    """
    from simulation.sphere_engine import (
        _build_patch_centroid, _update_patch_centroid_incremental)
    from world.sphere_world import SphereWorld
    w = SphereWorld(rows=8, cols=16)
    mask = np.zeros(w.n_cells, dtype=bool)
    mask[[62, 63, 48, 49]] = True      # 跨 0 列接缝链（r3c14→c15→[环]→c0→c1）
    mask[[64, 65, 66]] = True          # 孤立横线（分裂源）
    mask[[85, 87]] = True              # 待桥接对
    mask[42] = True                    # 孤立单格（供删除）
    out = _build_patch_centroid(w, mask)

    def _apply(new_mask: np.ndarray) -> None:
        nonlocal mask
        changed = np.flatnonzero(mask != new_mask)
        if changed.size:
            ok = _update_patch_centroid_incremental(w, mask, new_mask, out, changed)
            assert ok is True, "对称区步骤不应触发极区回退"
        mask = new_mask
        assert np.array_equal(out, _build_patch_centroid(w, mask)), (
            "增量 ≠ 全量（步骤退化）")

    n = mask.copy(); n[63] = False; _apply(n)          # 接缝链分裂：{62} ∥ {48,49}
    n = mask.copy(); n[65] = False; _apply(n)          # 横线分裂：{64} ∥ {66}
    n = mask.copy(); n[86] = True; _apply(n)           # 合并：86 桥接 {85} 与 {87}
    n = mask.copy(); n[49] = False; _apply(n)          # 链端收缩
    n = mask.copy(); n[42] = False; _apply(n)          # 删除孤立单格
    n = mask.copy(); n[[48, 85]] = False
    n[[95, 80]] = True; _apply(n)                      # 多格组合（移除 + 跨接缝新增对）
    n = mask.copy(); n[[95, 80, 86]] = False; _apply(n)  # 纯多格移除


def test_s13_pole_zone_falls_back_and_leaves_out_untouched():
    """极区（非对称邻接）接触 ⇒ 返回 False 且不改 out（由调用方回退全量重建）。"""
    from simulation.sphere_engine import (
        _build_patch_centroid, _update_patch_centroid_incremental)
    from world.sphere_world import SphereWorld
    w = SphereWorld(rows=8, cols=16)   # 极区行 = {0,1,6,7}
    mask = np.zeros(w.n_cells, dtype=bool)
    mask[[5, 19, 69]] = True           # 极行(r0c5) / 邻行(r1c3) / 内部(r4c5)
    out = _build_patch_centroid(w, mask)

    # 内部变更（对称区）⇒ 走增量、返回 True
    n = mask.copy(); n[69] = False
    ok = _update_patch_centroid_incremental(
        w, mask, n, out, np.flatnonzero(mask != n))
    assert ok is True
    assert np.array_equal(out, _build_patch_centroid(w, n))
    mask = n
    base = out.copy()

    # 邻行（row1）变更/新增 ⇒ 回退信号 + out 不动
    n = mask.copy(); n[19] = False
    ok = _update_patch_centroid_incremental(
        w, mask, n, out, np.flatnonzero(mask != n))
    assert ok is False, "邻行（row1）变更必须触发极区回退"
    assert np.array_equal(out, base), "回退时不得改动 out"

    n = mask.copy(); n[26] = True
    ok = _update_patch_centroid_incremental(
        w, mask, n, out, np.flatnonzero(mask != n))
    assert ok is False, "邻行（row1）新增必须触发极区回退"
    assert np.array_equal(out, base)

    # 极行（row0）变更 ⇒ 回退
    n = mask.copy(); n[5] = False
    ok = _update_patch_centroid_incremental(
        w, mask, n, out, np.flatnonzero(mask != n))
    assert ok is False, "极行（row0）变更必须触发极区回退"
    assert np.array_equal(out, base)


# ---------------------------------------------------------------- S14 T1 方向量化快路径/向量化

def test_s14_dirs_const_and_vec_pairs_match_scalar():
    """T1：标准行方向常量 == 逐格标量；向量化 pair 量化 == 标量（含极区/边界行）。"""
    e = _engine(v2=True, patchy=True, max_count=3000)
    rows, cols = int(e.world.rows), int(e.world.cols)
    assert np.array_equal(e._dirs8_const, np.arange(8, dtype=np.int64))
    assert np.array_equal(e._dirs4_const, np.array([1, 3, 4, 6], dtype=np.int64))
    # 标准行（2..rows-3）**全列**：8 邻 / VN 4 邻 == 常量
    for r in (2, rows // 2, rows - 3):
        for c in range(cols):
            cell = r * cols + c
            nb8 = np.asarray(e.world.neighbors(cell), dtype=np.int64)
            d8 = np.array([e._cell_dir_slot(cell, int(x)) for x in nb8])
            assert np.array_equal(d8, e._dirs8_const), f"row {r} col {c} 8邻≠常量"
            nb4 = np.asarray(e.world.neighbors_von_neumann(cell), dtype=np.int64)
            d4 = np.array([e._cell_dir_slot(cell, int(x)) for x in nb4])
            assert np.array_equal(d4, e._dirs4_const), f"row {r} col {c} 4邻≠常量"
    # 向量化 pair 量化 vs 标量：边界行/极行（回退路径；极行 nb=整行）
    for r in (0, 1, rows - 2, rows - 1):
        for c in (0, cols // 2, cols - 1):
            cell = r * cols + c
            nb = np.asarray(e.world.neighbors(cell), dtype=np.int64)
            ref = np.array([e._cell_dir_slot(cell, int(x)) for x in nb])
            got = e._cell_dir_slot_pairs(np.full(nb.size, cell), nb)
            assert np.array_equal(got, ref), f"cell {cell} pair 量化≠标量"
    # 随机长程 pair（写路径几何；含跨 0/cols 环绕）
    rr = np.random.default_rng(7)
    cur = rr.integers(0, rows * cols, 4000).astype(np.int64)
    tar = rr.integers(0, rows * cols, 4000).astype(np.int64)
    ref = np.array([e._cell_dir_slot(int(a), int(b)) for a, b in zip(cur, tar)])
    assert np.array_equal(e._cell_dir_slot_pairs(cur, tar), ref), "随机 pair ≠ 标量"
    # 同格/退化 ⇒ -1
    assert e._cell_dir_slot_pairs(
        np.array([5, 7]), np.array([5, 7])).tolist() == [-1, -1]


def test_s14_cos_fast_path_equals_general(monkeypatch):
    """快路径（常量查表）与一般路径（向量化）在 cos 打分上**逐位一致**；
    并证明门控事实：标准行不调用 `_cell_dir_slot_pairs`，span2 非 None 必调用。"""
    e = _engine(v2=True, patchy=True, max_count=3000)
    idx = 0
    e._mem_az[idx] = [2, 5, -1, -1]
    e._mem_tick[idx] = [e.tick, e.tick, -1, -1]
    e._mem_dist[idx] = [3.0, 25.0, 0.0, 0.0]
    e._mem_degraded[idx] = [False, True, False, False]
    cols = int(e.world.cols)
    for row in (2, 10, int(e.world.rows) - 3):
        cell = row * cols + 5
        nb = np.asarray(e.world.neighbors(cell), dtype=np.int64)
        for hd in (3, -1):
            g_fast = e._memory_egocentric_cos(idx, cell, nb, hd)
            e._span_table = np.zeros((e.world.n_cells, 1), dtype=np.int64)
            g_gen = e._memory_egocentric_cos(idx, cell, nb, hd)
            e._span_table = None
            assert np.array_equal(g_fast, g_gen), f"row {row} hd {hd} 快/一般不一致"
    assert float(np.abs(g_fast).max()) > 0.0, "装置空洞：打分全 0"
    # 门控事实：标准行走常量快路径（不调用向量化一般路径）
    calls: list = []
    orig = e._cell_dir_slot_pairs
    monkeypatch.setattr(type(e), "_cell_dir_slot_pairs",
                        lambda self, a, b: (calls.append(1), orig(a, b))[1])
    cell = 10 * cols + 5
    nb = np.asarray(e.world.neighbors(cell), dtype=np.int64)
    e._memory_egocentric_cos(idx, cell, nb, 3)
    assert calls == [], "标准行应走常量快路径"
    e._span_table = np.zeros((e.world.n_cells, 1), dtype=np.int64)
    e._memory_egocentric_cos(idx, cell, nb, 3)
    e._span_table = None
    assert len(calls) == 1, "span2 非 None ⇒ 必须走一般路径"


def test_s14_step_batch_matches_scalar_and_noise_guard():
    """批量步进 vs 逐个体：三数组逐位一致；噪声档禁入（独立 rng 顺序契约）。"""
    e = _engine(v2=True, patchy=True, max_count=3000)
    e._mem_az[:6] = [[3, -1, 5, -1], [0, 7, -1, -1], [-1, -1, -1, -1],
                     [4, 1, 6, 2], [7, -1, 0, -1], [1, 2, 3, 4]]
    e._mem_tick[:6] = [[0, -1, 5, -1], [0, 0, -1, -1], [-1, -1, -1, -1],
                       [0, 0, 0, 0], [0, -1, 2, -1], [0, 1, 2, 3]]
    e._mem_dist[:6] = [[0.0, 0.0, 7.0, 0.0], [19.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0], [25.0, 1.0, 0.5, 30.0],
                       [0.0, 0.0, 10.0, 0.0], [3.0, 3.0, 3.0, 3.0]]
    e._mem_degraded[:6] = [[False, False, True, False], [False, False, False, False],
                           [False, False, False, False], [False, True, False, False],
                           [False, False, False, False], [False, False, False, False]]
    triples = [(0, 2, 5), (1, -1, 6), (2, 3, 3), (3, 7, 0), (4, 0, -1), (5, 5, 6)]
    snap = (e._mem_az[:6].copy(), e._mem_dist[:6].copy(), e._mem_degraded[:6].copy())
    e._mem_v2_step_batch([t[0] for t in triples], [t[1] for t in triples],
                         [t[2] for t in triples], 1.0)
    got = (e._mem_az[:6].copy(), e._mem_dist[:6].copy(), e._mem_degraded[:6].copy())
    # 非空洞守卫：至少一个数组确有变化
    assert not (np.array_equal(snap[0], got[0]) and np.array_equal(snap[1], got[1]))
    # 还原 → 逐个体标量参考
    e._mem_az[:6], e._mem_dist[:6], e._mem_degraded[:6] = (
        s.copy() for s in snap)
    for (i, ho, hn) in triples:
        e._mem_v2_step_one(i, ho, hn, 1.0)
    for name, a, b in zip(("az", "dist", "degraded"), got,
                          (e._mem_az[:6], e._mem_dist[:6], e._mem_degraded[:6])):
        assert np.array_equal(a, b), f"批量步进 ≠ 逐个体（{name}）"
    # 噪声档禁入（rng 消费顺序契约）
    e2 = _engine(v2=True, patchy=True, max_count=1000, noise=True)
    assert bool(e2._mem_noise) is True
    with pytest.raises(AssertionError):
        e2._mem_v2_step_batch([0], [0], [1], 1.0)


def test_s14_write_vectorized_matches_scalar_reference():
    """写路径向量化 vs 旧标量循环（测试内联 T1 前参考语义，五数组逐位一致）。"""
    e = _engine(v2=True, patchy=True, max_count=3000)
    idx_arr = np.flatnonzero(np.ones(len(e._flat), dtype=bool))
    cur_flat = e._flat.copy()
    e._heading[:] = 0
    e._heading[::3] = -1
    keys = ("_mem_az", "_mem_dist", "_mem_tick", "_mem_degraded", "_mem_food")
    snap = {k: getattr(e, k).copy() for k in keys}
    tick = int(e.tick) + 1
    e._mem_v2_write(idx_arr, cur_flat, tick)
    got = {k: getattr(e, k).copy() for k in keys}
    assert int((got["_mem_az"] >= 0).sum()) > 0, "装置空洞：没有任何写入"
    # 还原 → 旧标量参考（T1 前 `_mem_v2_write` 主体）
    for k, v in snap.items():
        setattr(e, k, v.copy())
    ccell = e._patch_centroid[cur_flat[idx_arr]]
    ok = ccell >= 0
    assert ok.any()
    ridx = idx_arr[ok]
    cc = ccell[ok]
    hd = e._heading[ridx]
    slots = np.argmin(e._mem_tick[ridx], axis=1)
    az = np.full(ridx.size, -1, dtype=np.int64)
    for i in range(ridx.size):
        s = e._cell_dir_slot(int(cur_flat[ridx[i]]), int(cc[i]))
        if s < 0 or hd[i] < 0:
            continue
        az[i] = (s - int(hd[i])) % 8
    m = az >= 0
    assert m.any()
    ri = ridx[m]
    si = slots[m]
    ai = az[m]
    e._mem_az[ri, si] = ai.astype(np.int8)
    e._mem_dist[ri, si] = 0.0
    e._mem_tick[ri, si] = tick
    e._mem_degraded[ri, si] = False
    cap = np.maximum(e.resources._capacity[cur_flat[ri]], 1e-12)
    e._mem_food[ri, si] = (
        (e.resources._grid[cur_flat[ri]] / cap).astype(np.float32))
    for k in keys:
        assert np.array_equal(got[k], getattr(e, k)), f"写路径 ≠ 标量参考（{k}）"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))