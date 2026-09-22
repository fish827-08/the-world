"""尸体—食腐 + 血条 S3 交互测试（设计稿 §5.4；本段 = `[云端·开发]` 线）。

覆盖（S3 完成判据 = 单测绿 + 争夺可单独关闭 + 二分预测两条可分离 + 反退化断言不触发）：
  ① **H1 关档逐位等价**：contest/wound 全关 ⇒ C7 digest 不变（S1 钉死值）
  ② **争夺战**（#6，contest_enabled）：同格/邻格有取食者 ⇒ 高 g16 者可驱逐；
     RHP = health×(0.3+g16)×(energy/max_energy)，取食方 ×(1+holder_adv)；
     H4 升级阈值（|ΔRHP|>escalation_gap ⇒ 弱方撤退）；败者扣血 + 胜者付能量
  ③ **争夺可单独关闭**：contest_enabled=False ⇒ contest_n=0（即使 wound 开）
  ④ **血条恐惧项**（#7，wound_enabled ∧ w_fear_health>0）：低血条 ⇒ score 扣减；
     与 L1 恐惧项独立（L1 关时仍生效）；反退化计数 fear_health_flat_frac ≈0
  ⑤ **饥饿激进项**（#7，wound_enabled ∧ need_aggression_k>0）：能量低 ⇒ 攻击概率↑
     （二分预测：血条↓恐惧 vs 能量↓激进，方向相反、可分离）
  ⑥ **读数可回**：contest_win_by_holder_frac / fear_health_flat_frac 可读
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.config import CorpseWoundConfig, InfoStructureConfig, SimConfig
from simulation.sphere_engine import SphereEngine, Gene


def _engine(ticks: int = 0, *, seed: int = 42,
            corpse: bool = False, wound: bool = False, contest: bool = False,
            cwc: CorpseWoundConfig | None = None) -> SphereEngine:
    cfg = SimConfig(seed=seed)
    cfg.population.max_count = 600
    cfg.predation.forage_tradeoff_k = 0.0
    cfg.info_structure = InfoStructureConfig(
        enabled=True, learning_rate=0.05, memory_gradient="none",
    )
    cfg.corpse_wound = cwc or CorpseWoundConfig(
        corpse_enabled=corpse, wound_enabled=wound, contest_enabled=contest,
    )
    e = SphereEngine(cfg)
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    return e


def _digest(e: SphereEngine) -> tuple[int, float]:
    return (int(e._flat.sum()), round(float(e._energy.sum()), 6))


# --------------------------------------------------------------- ① H1 / C7

def test_s3_default_off_is_bit_identical():
    """C7：contest/wound 全关必须与**改动前**逐位一致（S1 钉死值不变）。"""
    assert _digest(_engine(ticks=50)) == (573985, 8171.692943), (
        "S3 关档改变了轨迹 ⇒ 破坏 H1（默认关 = 逐位等价）")


# --------------------------------------------------------------- ②③ 争夺战

def _setup_contest(e: SphereEngine) -> None:
    """构造"同格两人、一取食一挑战"的最小争夺局。

    holder(0)：energy 高、g16 中、health 满（取食方，×holder_adv）
    challenger(1)：energy 中、g16 高（可驱逐）
    """
    P = len(e._id)
    cols = int(e.world.cols)
    cell = 30 * cols + 10
    e._flat[:] = cell                    # 全体同格
    e._genes[:, Gene.AGGRESSION] = 0.3   # 默认低攻击（holder 不驱逐别人）
    e._genes[0, Gene.AGGRESSION] = 0.3   # holder：中等
    e._genes[1, Gene.AGGRESSION] = 0.9   # challenger：高
    e._genes[:, 0] = 1.0
    e._genes[:, 19] = 0.0
    e._energy[:] = 60.0
    e._energy[0] = 100.0                 # holder 能量高 ⇒ RHP 高
    e._health[:] = 1.0


def test_contest_occurs_when_on():
    """争夺战：contest_enabled ⇒ contest_n>0 且记录持有者胜率。"""
    e = _engine(wound=True, contest=True, ticks=0)
    _setup_contest(e)
    # 让个体 0 "正在取食"：直接调用 _step_contest（等价于取食段内挂点）
    e._step_contest(len(e._id), np.array([0], dtype=np.int64), e._genes, e._energy[:len(e._id)])
    assert e._contest_n > 0, "同格高 g16 挑战者应触发争夺战"
    # 持有者（0）RHP = 100×(0.3+0.3)×1 ×1.3 ≈ 78；挑战者（1）= 60×(0.3+0.9)×1 = 72
    # ⇒ holder 胜，胜率分子 +1
    assert e._contest_holder_win_n == 1, "持有者优势下 holder 应胜"
    pr = e.wound_probe()
    assert pr["contest_win_by_holder_frac"] == pytest.approx(1.0, abs=1e-9), (
        "持有者胜率应可读回")


def test_contest_can_be_turned_off():
    """争夺可单独关闭：contest_enabled=False ⇒ contest_n=0（即使 wound 开）。

    通过完整 tick 验证（开关在调用点守卫，直接调方法会绕过守卫 ⇒ 用 step 路径）。
    """
    e = _engine(wound=True, contest=False, ticks=1)
    _setup_contest(e)
    e._stomach[:] = 0.0                    # 保证"正在取食"分支可进入
    e._energy[:] = 100.0
    e.step()
    assert e._contest_n == 0, "contest 关时不应发生争夺战"
    assert e._contest_holder_win_n == 0
    # 对照：开档应发生（同布局）
    e2 = _engine(wound=True, contest=True, ticks=1)
    _setup_contest(e2)
    e2._stomach[:] = 0.0
    e2._energy[:] = 100.0
    e2.step()
    assert e2._contest_n > 0, "contest 开时应发生争夺战（对照）"


def test_contest_escalation_gap_retreat():
    """H4：RHP 差距大 ⇒ 弱方立即撤退（败者扣血，但不算胜/负局数外事件）。"""
    e = _engine(wound=True, contest=True, ticks=0)
    _setup_contest(e)
    # 把挑战者能量压到极低 ⇒ RHP 差距 >> escalation_gap(0.25) ⇒ 弱方撤退
    e._energy[1] = 1.0
    e._step_contest(len(e._id), np.array([0], dtype=np.int64), e._genes, e._energy[:len(e._id)])
    assert e._contest_n > 0, "差距大也应算一次争夺（撤退）"
    assert e._health[1] == pytest.approx(1.0 - 0.35, rel=1e-9), (
        "撤退方（弱方）应扣血条 wound_base")


# --------------------------------------------------------------- ④ 血条恐惧项

def _fear_health_setup(e: SphereEngine, health: float) -> None:
    """构造"威胁邻格 + 低血条"布局：验证血条恐惧项对 score 的作用（反退化）。"""
    P = len(e._id)
    cols = int(e.world.cols)
    # 个体 0 在 (30,10)；邻居 (30,9) 放高 g16 威胁者
    e._flat[0] = 30 * cols + 10
    e._flat[1] = 30 * cols + 9
    e._genes[:, Gene.AGGRESSION] = 0.9    # 邻居都是威胁（含自己所在格邻域）
    e._genes[:, 0] = 1.0
    e._genes[:, 19] = 0.0
    e._energy[:] = 100.0
    e._health[:] = health


def test_fear_health_flat_frac_approx_zero():
    """血条恐惧项按候选求值 ⇒ 反退化占比应≈0（与 L1 恐惧同规格，R148-1 形态）。"""
    e = _engine(wound=True, ticks=0)
    _fear_health_setup(e, health=0.2)
    e.step()
    pr = e.wound_probe()
    assert pr["fear_health_flat_frac"] is not None
    assert pr["fear_health_flat_frac"] == pytest.approx(0.0, abs=0.1), (
        f"fear_health_flat_frac={pr['fear_health_flat_frac']} 应≈0"
        "（血条恐惧项逐候选求值，不应退化为常量）")


def test_fear_health_independent_of_l1():
    """血条恐惧项与 L1 恐惧项独立：L1 全关时血条项仍生效（可单独关闭）。"""
    cfg = SimConfig(seed=42)
    cfg.population.max_count = 600
    cfg.predation.forage_tradeoff_k = 0.0
    cfg.info_structure = InfoStructureConfig(
        enabled=True, learning_rate=0.05, memory_gradient="none",
    )
    cfg.corpse_wound = CorpseWoundConfig(wound_enabled=True, w_fear_health=0.5)
    # L1 默认全关
    e = SphereEngine(cfg)
    _fear_health_setup(e, health=0.2)
    e.step()
    assert e.l1_probe() is None, "L1 应仍关（血条项独立）"
    assert e._fearh_dec_n > 0, "血条恐惧项应被求值（独立于 L1）"
    # w_fear_health=0 ⇒ 关闭
    cfg2 = SimConfig(seed=42)
    cfg2.population.max_count = 600
    cfg2.predation.forage_tradeoff_k = 0.0
    cfg2.info_structure = InfoStructureConfig(
        enabled=True, learning_rate=0.05, memory_gradient="none",
    )
    cfg2.corpse_wound = CorpseWoundConfig(wound_enabled=True, w_fear_health=0.0)
    e2 = SphereEngine(cfg2)
    _fear_health_setup(e2, health=0.2)
    e2.step()
    assert e2._fearh_dec_n == 0, "w_fear_health=0 ⇒ 血条恐惧项应关闭"


# --------------------------------------------------------------- ⑤ 饥饿激进项

def test_need_aggression_separates_binary_prediction():
    """二分预测可分离：need_aggression_k=0（D 臂）⇒ 攻击概率与 k>0 不同；
    能量低 ⇒ k>0 时更激进（攻击者更多），k=0 时无此放大。"""
    def _attackers_frac(k: float, energy_frac: float) -> float:
        cfg = SimConfig(seed=7)
        cfg.population.max_count = 600
        cfg.predation.forage_tradeoff_k = 0.0
        cfg.info_structure = InfoStructureConfig(
            enabled=True, learning_rate=0.05, memory_gradient="none",
        )
        cfg.corpse_wound = CorpseWoundConfig(
            wound_enabled=True, need_aggression_k=k,
        )
        e = SphereEngine(cfg)
        e._genes[:, Gene.AGGRESSION] = 1.0
        e._genes[:, 0] = 1.0
        e._energy[:] = cfg.organisms.max_energy * energy_frac
        e.step()
        # 用 _duel 计数推断攻击者（real_attempts ≥ 攻击者数；这里取 nominal 近似）
        return float(e._duel["real_attempts"] + e._duel["skip_no_energy"]
                     + e._duel["skip_no_prey"] + e._duel["skip_already_eaten"])

    # 饥饿（energy=30%）：k=0.5 应比 k=0 攻击更多（饥饿激进放大）
    hungry_k0 = _attackers_frac(0.0, 0.3)
    hungry_k05 = _attackers_frac(0.5, 0.3)
    assert hungry_k05 > hungry_k0, (
        f"饥饿时 k=0.5 攻击者({hungry_k05})应 > k=0({hungry_k0}) —— 二分预测的"
        "'能量↓激进'在 k>0 时才成立")
    # 饱食（energy=95%）：饥饿项 ≈0 ⇒ 两档应接近（方向分离的对照）
    full_k0 = _attackers_frac(0.0, 0.95)
    full_k05 = _attackers_frac(0.5, 0.95)
    assert full_k05 <= full_k0 * 1.2 + 1, (
        f"饱食时不应有饥饿激进放大（k0={full_k0}, k05={full_k05}）")


# --------------------------------------------------------------- ⑥ 读数回

def test_s3_probes_readable():
    """S3 读数可回：开档后 contest 胜率 / fear-health 反退化 可读（None 或值）。"""
    e = _engine(corpse=True, wound=True, contest=True, ticks=30)
    pr = e.wound_probe()
    assert "contest_win_by_holder_frac" in pr, "争夺胜率键应存在"
    assert "fear_health_flat_frac" in pr, "血条恐惧反退化键应存在"
    cp = e.corpse_probe()
    assert cp["corpse_total"] >= 0.0
