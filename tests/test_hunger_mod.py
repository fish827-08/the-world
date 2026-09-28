"""R247 饥饿调制（HM）—— 实施规格 `docs/设计文档/实施规格-移动决策三件-20260928.md` §一。

覆盖（规格 §1.4 五例 + 交付检查单）：
  ① **默认关逐位不变**：C7 基线 digest `(574887, 11266.746993)` 不动（关档零接触）；
  ② **单调性**：hunger↑ ⇒ 移动倾向↑（人工能量分布 + 同 seed，比**跨格个体数** ——
     规格原话"直接调函数"在实现里对应"经真实走停接线可观测"，避免自证式复读公式）；
  ③ **边界**：hunger=0/1 两端的 h_norm（−h_mid/(1−h_mid) / 1）经接线可观测；
  ④ **变异检查（机械化，DEL-8 家法）**：① / ② 两条接线**各自**必须改变 digest
     （alpha-only / beta-only 各一条），且 alpha=beta=0 必须**不**改变（数值控制，
     ⇒ 例 ② 的判据不是空过）；
  ⑤ **双路径**：Python vs Rust 同 seed 逐位一致（HM 开档 + 关档各一）；
  ⑥ **快路径**：HM 开档**不失** `_move_decide_batch`，且与参考循环逐 tick 逐位一致
     （vanilla + subpos 两态）；
  ⑦ **配置四件套**：asdict 往返 / 缺键回退 / 未知键忽略 / 断言（alpha,beta∈[0,2]、h_mid∈[0,1]）。
"""

from __future__ import annotations

import numpy as np
import pytest

from simulation.config import HungerModConfig, InfoStructureConfig, SimConfig
from simulation.genes import Gene
from simulation.sphere_engine import SphereEngine

_U = object()   # "未传"哨兵：区分"用 config 默认值"与"显式置 0"


def _engine(*, seed: int = 42, n: int = 200, alpha=_U, beta=_U, h_mid=_U,
            subpos: bool = False, use_sim_core: bool = False) -> SphereEngine:
    cfg = SimConfig(seed=seed)
    cfg.simulation.use_sim_core = use_sim_core
    cfg.population.initial_count = n
    if subpos:
        cfg.subpos.enabled = True
    if alpha is not _U or beta is not _U or h_mid is not _U:
        cfg.hunger_mod.enabled = True
        if alpha is not _U:
            cfg.hunger_mod.alpha = alpha
        if beta is not _U:
            cfg.hunger_mod.beta = beta
        if h_mid is not _U:
            cfg.hunger_mod.h_mid = h_mid
    return SphereEngine(cfg)


def _run(e: SphereEngine, ticks: int = 50) -> SphereEngine:
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    return e


def _digest(e: SphereEngine) -> tuple[int, float]:
    return (int(e._flat.sum()), round(float(e._energy.sum()), 6))


def _movers(seed: int, energy: float, *, alpha: float = 0.6, n: int = 400,
            h_mid=_U) -> tuple[int, int]:
    """1 tick 后按 **_id 对齐**统计"换了格"的个体数（抗死亡压缩重排）。

    走停基因钉死为"必走"（MOVE_PROB=1, ROOTING=0）⇒ `moved = u < move_prob_eff`
    两档的区别**只**来自 HM ① 对 `move_prob` 的调制。
    """
    e = _engine(seed=seed, n=n, alpha=alpha, beta=0.0, h_mid=h_mid)
    e._genes[:, Gene.MOVE_PROB] = 1.0
    e._genes[:, Gene.ROOTING] = 0.0
    e._energy[:] = energy
    before = dict(zip(e._id.tolist(), e._flat.tolist()))
    e.step()
    after = dict(zip(e._id.tolist(), e._flat.tolist()))
    moved = sum(1 for i, c in before.items() if i in after and after[i] != c)
    return moved, len(after)


# --------------------------------------------------------------------------- ①
def test_default_off_c7_baseline_unchanged() -> None:
    """默认关：与 R244 之后钉死的 C7 基线逐位一致（HM 关档零接触）。"""
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
    assert cfg.hunger_mod.enabled is False, "HM 必须默认关（回滚点）"
    e = _run(SphereEngine(cfg), 50)
    assert _digest(e) == (574887, 11266.746993), (
        f"C7 基线漂移：{_digest(e)} ⇒ HM 关档动了默认路径"
    )


# --------------------------------------------------------------------------- ②③
def test_hunger_monotone_increases_movement() -> None:
    """② 单调性：饿（energy=30 ⇒ h_norm≈0.8）比饱（energy=300 ⇒ h_norm=−1）**更常走**。

    定量：alpha=0.6、h_mid=0.5 时 —— 饿 ⇒ move_prob_eff = clip(1×1.48) = **1.0**（全员走）；
    饱 ⇒ 1×(1−0.6) = **0.4**（约四成走；seed 31 实测 177/400=0.443，含 ~1% 非闸位移偏置，
    见边界例 docstring）。两档同 seed、同基因，唯一差别是能量。
    """
    hungry, n_h = _movers(31, 30.0)
    full, n_f = _movers(31, 300.0)
    assert hungry >= 0.95 * n_h, f"饿档未接近全员移动：{hungry}/{n_h}"
    assert full <= 0.55 * n_f, f"饱档移动比例过高（调制未生效？）：{full}/{n_f}"
    assert hungry > full


def test_boundary_h_norm_endpoints_engine_observed() -> None:
    """③ 边界（规格 §1.4-3 的引擎侧对应）：hunger 两端 ⇒ h_norm 端点值**经接线可观测**。

    `h_norm = (hunger − h_mid)/max(1e-9, 1−h_mid)`，h_mid=0.5、alpha=0.6、base=1.0：
      * hunger=0（energy=300）⇒ h_norm = −h_mid/(1−h_mid) = **−1.0** ⇒ eff = **0.4**；
      * hunger=0.9（energy=30）⇒ h_norm = 0.8 ⇒ eff = 1.48 ⇒ **clip → 1.0**（全员走）。
    ⚠️ hunger=1（energy=0）**经引擎不可观测**：`moved &= energy ≥ 移动成本` 先把
    "付不起"者全滤掉 ⇒ 该端点由 clip 上端（1.48→1.0）间接覆盖（clip 落在 [0,1] 的体现）。
    观测偏置：同 tick 还有 ~1% 非闸位移（攻击接触等，seed 7 实测 miss=4/extra=0）
    ⇒ 实测分数略高于 0.4（seed 31：0.443）⇒ 判据用窗口而非点值。
    """
    full, n_f = _movers(31, 300.0)
    hungry, n_h = _movers(31, 30.0)
    assert hungry == n_h, (
        f"hunger=0.9（eff=1.48 被 clip 到 1.0）应全员移动：{hungry}/{n_h} ⇒ clip/公式失守"
    )
    frac = full / n_f
    assert 0.30 <= frac <= 0.52, (
        f"hunger=0 的移动分数 {frac:.3f} 偏离期望 0.4（±~1% 其他位移偏置）"
        "⇒ 检查 h_norm 端点公式（−h_mid/(1−h_mid)）或 clip"
    )


def test_h_mid_boundary_shifts_cutoff() -> None:
    """③ 边界：h_mid=0.2 ⇒ 饱档 h_norm = −0.2/0.8 = −0.25（eff=0.85），饿档仍被 clip 到 1.0。

    这条同时钉住"h_mid 真的进了公式"：h_mid 若被写死 0.5，h_mid=0.2 的饱档
    eff 会从 0.85 掉到 0.4 ⇒ 绝对值窗口（[0.75, 0.95]）直接破下界；
    再叠加**两档之差**判定：h_mid=0.2 的 (饿−饱) 差应显著**小于** h_mid=0.5 的差。
    """
    h_hungry, n1 = _movers(31, 30.0, h_mid=0.2)
    h_full, n_hf = _movers(31, 300.0, h_mid=0.2)
    d02 = h_hungry - h_full

    m_hungry, _ = _movers(31, 30.0, h_mid=0.5)
    m_full, n2 = _movers(31, 300.0, h_mid=0.5)
    d05 = m_hungry - m_full

    assert h_hungry >= 0.95 * n1, f"h_mid=0.2 饿档未接近全走：{h_hungry}/{n1}"
    assert d02 < d05, (
        f"h_mid 未生效：h_mid=0.2 的（饿−饱）差 {d02} 未低于 h_mid=0.5 的 {d05}"
    )
    assert h_full > m_full, "h_mid 抬高门槛后，饱档应更常走（eff 0.85 vs 0.4）"
    # 绝对值窗口：h_mid=0.2 ⇒ 饱档 eff = 1+0.6×(−0.25) = **0.85**（实测 seed 31：0.877）
    # ⇒ 若 h_mid 未进公式（写死 0.5）此处 ≈0.44 会跌破下界，被本行抓住。
    frac02 = h_full / n_hf
    assert 0.75 <= frac02 <= 0.95, (
        f"h_mid=0.2 饱档移动分数 {frac02:.3f} 偏离期望 0.85 ⇒ h_mid 未进 h_norm 公式？"
    )


# --------------------------------------------------------------------------- ④
def test_zero_strength_is_bitwise_noop() -> None:
    """数值控制：alpha=beta=0（开档）⇒ digest 与关档**逐位相同**（公式无非参数副作用）。"""
    off = _digest(_run(_engine(seed=42, alpha=0.0, beta=0.0), 50))
    assert off == _digest(_run(_engine(seed=42), 50)), (
        "alpha=beta=0 的「开档」改变了行为 ⇒ 调制含隐藏副作用（或 clip 非恒等）"
    )


def test_mutation_sensitivity_per_channel() -> None:
    """🔴 DEL-8 变异测试（机械化）：**逐通道**开档必须各自可检出。

    * ① 只开（alpha=0.6, beta=0）≠ 关档 ⇒ 守护走停接线；
    * ② 只开（alpha=0, beta=0.8）≠ 关档 ⇒ 守护 perc 调制接线（Python 参考 + 快路径）；
    * 两者同开 ≠ 关档 ⇒ 守护组合。
    任一条接线被改死（"少一行 ⇒ 死参数"家族）⇒ 对应条立刻变红。
    """
    off = _digest(_run(_engine(seed=42), 50))
    a_only = _digest(_run(_engine(seed=42, alpha=0.6, beta=0.0), 50))
    b_only = _digest(_run(_engine(seed=42, alpha=0.0, beta=0.8), 50))
    both = _digest(_run(_engine(seed=42, alpha=0.6, beta=0.8), 50))
    assert a_only != off, f"只开 ① 与关档相同（{off}）⇒ 走停调制未接线"
    assert b_only != off, f"只开 ② 与关档相同（{off}）⇒ perc 调制未接线（或未进快路径）"
    assert both != off


def test_subpos_path_hm_active() -> None:
    """① 的 subpos 叠加分支（p_move = 1 − stay_eff）也真接线。"""
    off = _digest(_run(_engine(seed=42, subpos=True), 50))
    on = _digest(_run(_engine(seed=42, alpha=0.6, beta=0.8, subpos=True), 50))
    assert on != off, "subpos 路径上 HM 为 no-op ⇒ `_p_move_eff` 分支未接线"


# --------------------------------------------------------------------------- ⑤
def test_dual_path_bitwise_identical() -> None:
    """⑤ 双路径：Python(use_sim_core=False) vs Rust(True) 同 seed 50 tick 逐位一致。"""
    kw = dict(seed=7, n=150)
    py0 = _digest(_run(_engine(**kw, use_sim_core=False), 50))
    rs0 = _digest(_run(_engine(**kw, use_sim_core=True), 50))
    assert py0 == rs0, f"HM 关档双路径不一致：py={py0} rust={rs0}"

    py1 = _digest(_run(_engine(**kw, alpha=0.6, beta=0.8, use_sim_core=False), 50))
    rs1 = _digest(_run(_engine(**kw, alpha=0.6, beta=0.8, use_sim_core=True), 50))
    assert py1 == rs1, f"HM 开档双路径不一致：py={py1} rust={rs1}"
    assert py1 != py0, "HM 开档 == 关档 ⇒ 双路径对拍空过（机制未生效）"


# --------------------------------------------------------------------------- ⑥
@pytest.mark.parametrize("subpos", [False, True])
def test_batch_fast_path_lockstep_with_hm(subpos: bool) -> None:
    """⑥ HM 开档：`_move_decide_batch` 仍启用，且与参考循环**逐 tick 逐位**一致。"""
    e_on = _engine(seed=13, n=150, alpha=0.6, beta=0.8, subpos=subpos)
    e_off = _engine(seed=13, n=150, alpha=0.6, beta=0.8, subpos=subpos)
    e_off._batch_move_on = False
    calls = [0]
    orig = SphereEngine._move_decide_batch

    def counting(self, *a, **k):
        calls[0] += 1
        return orig(self, *a, **k)

    SphereEngine._move_decide_batch = counting
    try:
        for t in range(60):
            e_on.step()
            e_off.step()
            assert np.array_equal(e_on._flat, e_off._flat), f"t={t} _flat 分岔"
            assert np.array_equal(e_on._energy, e_off._energy), f"t={t} _energy 分岔"
            assert np.array_equal(e_on._genes, e_off._genes), f"t={t} _genes 分岔"
            assert np.array_equal(e_on._id, e_off._id), f"t={t} _id 分岔"
    finally:
        SphereEngine._move_decide_batch = orig
    assert calls[0] > 0, "HM 开档把快路径挤掉了 ⇒ 本对拍空过（门表被误加）"


# --------------------------------------------------------------------------- ⑦
def test_config_four_piece() -> None:
    """⑦ 配置四件套：往返 / 缺键回退 / 未知键忽略 / 断言 / 进 fingerprint。"""
    h = HungerModConfig(enabled=True, alpha=0.6, beta=0.7, h_mid=0.4)
    cfg = SimConfig(seed=1)
    cfg.hunger_mod = h
    d = cfg.to_dict()
    assert d["hunger_mod"] == {
        "enabled": True, "alpha": 0.6, "beta": 0.7, "h_mid": 0.4, "stay_gain": 0.0}

    back = SimConfig.from_dict(d)
    assert back.hunger_mod == h

    d2 = dict(d)
    d2.pop("hunger_mod")
    assert SimConfig.from_dict(d2).hunger_mod == HungerModConfig(), (
        "旧存档缺 `hunger_mod` 键必须回退默认关（R237 教训：漏传 = 续跑静默换配置）"
    )

    d3 = dict(d)
    d3["hunger_mod"] = {**d["hunger_mod"], "bogus_key": 123}
    assert SimConfig.from_dict(d3).hunger_mod == h, "未知键必须忽略而非报错"

    c2 = SimConfig(seed=1)
    c2.hunger_mod.alpha = 0.9
    assert cfg.fingerprint() != c2.fingerprint(), "hunger_mod 必须进 fingerprint"

    with pytest.raises(AssertionError):
        HungerModConfig(alpha=-0.1)
    with pytest.raises(AssertionError):
        HungerModConfig(alpha=2.1)
    with pytest.raises(AssertionError):
        HungerModConfig(beta=2.1)
    with pytest.raises(AssertionError):
        HungerModConfig(h_mid=1.1)
    with pytest.raises(AssertionError):
        HungerModConfig(stay_gain=-0.1)
