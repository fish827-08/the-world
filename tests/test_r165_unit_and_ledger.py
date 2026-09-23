"""R165 波 0 回归测试：尸体池单位统一（能量）+ `intake_scav` 通道 + 两 cost 入账。

**为什么单开一个文件**：这三项是 13.4 纪元的**构造级前提**（R165 §四），
需要长期钉死；且它们都属"口径"类改动 —— 最容易在后续重构里被静默改回去。

覆盖：
  1. 单位标记可读回（C4）
  2. 池收支**闭账**（`corpse_balance_residual` 恒 0）—— 池只有四条流
  3. Σ取（质量）× `eat_efficiency` ≡ 池减（能量）—— 精确
  4. 旧口径（×3 放大）不会回归：存量 1.0 + 同格多体 ⇒ 取量×eff ≤ 存量
  5. `intake_scav` 与 `intake_forage` 可分离，且 `intake_scav ≤ corpse_scav_e`
  6. 归因数组 `_stomach_scav` 与胃同步（繁殖/死亡位点不漏改）
  7. 争夺代价 → `cost_attack`、愈合耗能 → `cost_meta`（微场景直调）
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.config import InfoStructureConfig, SimConfig
from simulation.sphere_engine import (
    EC_ATTACK, EC_META, SphereEngine,
)
from simulation.genes import Gene


def _engine(corpse=False, wound=False, contest=False, ticks=0, seed=42,
            max_count=600):
    cfg = SimConfig(seed=seed)
    cfg.simulation.use_sim_core = False
    cfg.population.max_count = max_count
    cfg.info_structure = InfoStructureConfig(enabled=False)
    cw = cfg.corpse_wound
    cw.corpse_enabled = corpse
    cw.wound_enabled = wound
    cw.contest_enabled = contest
    cw.corpse_cap_per_cell = 200
    e = SphereEngine(cfg)
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    return e


# --------------------------------------------------------------- 1. 单位标记
def test_pool_unit_marker_readback():
    """C4：产物必须能自证"这批用的是哪套单位"。"""
    cp = _engine(corpse=True, ticks=5).corpse_probe()
    assert cp["corpse_pool_unit"] == "energy"
    assert cp["scav_to_energy_divisor"] == pytest.approx(3.0)


def test_pool_flows_present_and_residual_named():
    cp = _engine(corpse=True, ticks=5).corpse_probe()
    for k in ("corpse_deposited_e", "corpse_overflow_e", "corpse_scav_e",
              "corpse_decayed_e", "corpse_balance_residual"):
        assert k in cp, f"缺 {k}（池收支闭账字段）"


# --------------------------------------------------------------- 2. 池闭账
def test_pool_balance_closes_over_run():
    """池收支只有四条流（投放/溢出/食腐/腐烂）⇒ 残差恒 0。

    残差非 0 ⇒ 有第五条流（漏记），或某条流量化口径与池不同（单位又错了一次）。
    """
    e = _engine(corpse=True, wound=True, contest=True, ticks=300)
    cp = e.corpse_probe()
    assert cp["corpse_deposited_e"] > 0.0, "300 tick 内应产生尸体"
    assert abs(cp["corpse_balance_residual"]) < 1e-6, (
        f"池收支不闭合，残差 {cp['corpse_balance_residual']}（应有第五条流被漏记）")


# --------------------------------------------------------------- 3. 单点换算
def test_mass_times_efficiency_equals_pool_draw():
    """R165 0-1：取量（质量）× `eat_efficiency` ≡ 池减（能量）—— 精确到浮点级。"""
    e = _engine(corpse=True)
    P = len(e._id)
    cell = 100
    e._flat[:P] = 777
    e._flat[0] = cell
    e._stomach[:P] = 0.0
    e._stomach_scav[:P] = 0.0
    e._corpse_energy[:] = 0.0
    e._corpse_energy[cell] = 100.0
    e._genes[0, Gene.AGGRESSION] = 0.99
    b = float(e._corpse_energy[cell])
    e._step_scavenging(P, e._stomach[:P], np.full(P, 1e9), e._genes[:P])
    take = float(e._stomach[0])
    drawn = b - float(e._corpse_energy[cell])
    eff = float(e.config.organisms.eat_efficiency)
    assert take > 0.0, "g16=0.99 时应当能吃"
    assert abs(take * eff - drawn) < 1e-9, (
        f"池减 {drawn} ≠ 取量 {take} × {eff}（单点换算失效）")


def test_no_triple_inflation_regression():
    """旧口径（×3 放大）不得回归：存量 1.0 + 同格多体 ⇒ 取量×eff ≤ 存量、格上 ≥ 0。"""
    e = _engine(corpse=True)
    P = len(e._id)
    assert P >= 12
    cell = 3610
    e._flat[:P] = cell
    e._genes[:P, Gene.AGGRESSION] = 0.99
    e._corpse_energy[:] = 0.0
    e._corpse_energy[cell] = 1.0
    e._stomach[:P] = 0.0
    e._stomach_scav[:P] = 0.0
    e._step_scavenging(P, e._stomach[:P], np.full(P, 1e9), e._genes[:P])
    take = float(e._stomach[:P].sum())
    after = float(e._corpse_energy[cell])
    eff = float(e.config.organisms.eat_efficiency)
    assert after >= -1e-12, f"格上被扣成负值：{after}"
    assert take * eff <= 1.0 + 1e-9, (
        f"取量×eff = {take * eff} 超过格上存量 1.0（超发回归）")


# --------------------------------------------------------------- 4. 账本可分离
def test_intake_scav_separated_when_corpse_on():
    e = _engine(corpse=True, ticks=400)
    g = e.energy_ledger()["global"]
    cp = e.corpse_probe()
    assert cp["corpse_scav_e"] > 0.0, "400 tick 内应有食腐"
    assert g["intake_scav_sum"] > 0.0, "尸体收入必须单列（不得混进 intake_forage）"
    # 入账 ≤ 取走：差额 = 仍在胃里的 + 随死亡/捕食损失的
    assert g["intake_scav_sum"] <= cp["corpse_scav_e"] + 1e-6, (
        "intake_scav 超过池取走量 ⇒ 记账凭空放大")


def test_intake_scav_zero_when_corpse_off():
    """关档：新通道恒 0，且既有通道不受影响（零轨迹影响的门面）。"""
    e = _engine(corpse=False, ticks=200)
    g = e.energy_ledger()["global"]
    assert g["intake_scav_sum"] == 0.0
    assert g["intake_forage_sum"] > 0.0


def test_stomach_scav_attribution_stays_in_sync():
    """归因数组有 4 个维护位点（食腐/繁殖/新生儿/死亡压缩）⇒ 漏改会静默脱节。"""
    e = _engine(corpse=True, wound=True, contest=True, ticks=500)
    au = e.stomach_scav_audit()
    assert au["over"] == 0, f"_stomach_scav 超过胃（{au['over']} 个个体）"
    assert au["negative"] == 0, f"_stomach_scav 出现负值（{au['negative']} 个）"


# --------------------------------------------------------------- 5. 两 cost 入账
def test_contest_cost_booked_into_cost_attack():
    """争夺胜者代价必须进 `cost_attack`（R165 0-2 折进现有通道，不新开）。

    微场景直调：同格两人、g16 相近、RHP 差 < `escalation_gap` ⇒ 进入战斗分支 ⇒
    持有者（eaters[0]）胜 ⇒ 付出 `contest_cost_energy`。
    """
    e = _engine(contest=True, ticks=0)
    P = len(e._id)
    cell = 2000
    e._flat[:P] = 777
    e._flat[0] = cell
    e._flat[1] = cell
    e._genes[:P, Gene.AGGRESSION] = 0.5
    e._genes[1, Gene.AGGRESSION] = 0.6           # 挑战者攻击性更高 ⇒ 才发起
    e._health[:P] = 1.0
    e._energy[:P] = 50.0                         # RHP 差 < gap ⇒ 进入战斗
    e._ec_pt[:] = 0.0
    e._ec_ensure(P)          # ⚠️ 记账缓冲**懒扩容**（按 idx 上界）⇒ 读 index 1 前须先确保够长
    cost = float(e.config.corpse_wound.contest_cost_energy)
    e._step_contest(
        P, [0], e._genes[:P], e._energy[:P],
    )
    assert float(e._ec_pt[EC_ATTACK, 0]) == pytest.approx(cost), (
        "争夺代价未记入 cost_attack")
    assert float(e._ec_pt[EC_ATTACK, 1]) == pytest.approx(0.0), (
        "败者不应付争夺代价")


def test_heal_cost_booked_into_cost_meta():
    """愈合耗能必须进 `cost_meta`（维持类）。

    两引擎同 seed、同初值（health=0.5），唯一差别是 `wound_enabled`
    ⇒ 1 tick 后 `cost_meta_sum` 之差应**包含**愈合耗能。
    ⚠️ 2026-09-23（T4 反转）：wound 开/关的捕食轨迹**不再逐位一致**（反转只影响
    wound_on：成功即死→尸体 vs 关档转移）⇒ 差值 ≠ 恰愈合成本（实测 7.92 vs 10，
    偏离 21%）。断言改为**方向 + 量级**：差值 > 0（愈合确实扣能入账）。
    """
    n = 200
    on = _engine(wound=True, ticks=0, max_count=n)
    off = _engine(wound=False, ticks=0, max_count=n)
    off._health[:] = 0.5                          # 同初值（否则对比无意义）
    on._health[:] = 0.5
    p = len(on._id)
    on.step()
    off.step()
    d = (on.energy_ledger()["global"]["cost_meta_sum"]
         - off.energy_ledger()["global"]["cost_meta_sum"])
    expect = float(on.config.corpse_wound.wound_heal_energy_cost) * p
    assert d > 0, f"cost_meta 之差 {d:.3f} 应为正（愈合耗能确实入账）"
    assert d > expect * 0.5, (
        f"cost_meta 之差 {d:.3f} 应含大部分愈合耗能（期望 ~{expect:.3f}）")
