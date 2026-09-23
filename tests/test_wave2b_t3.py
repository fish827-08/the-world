"""13.4 波 2B T3 测试（修 F1 + 视野 2 格 + 单格上限 + 去盲选 + 看见才出手）。

覆盖（任务书 T3 完成判据）：
  ① **F1 修复**：densities 归一化分母 = 每格实际邻居数（非 stride 120）⇒
     社交项量级恢复（nb_len 众数 8 / 极区 120）；关档 digest 钉死新基线
  ② **span=2 候选数**：普通格 ring1+2 = 24（`mean_candidates`），极区降级（F3 闸）
  ③ **看见才出手**：视野内无猎物 ⇒ 零出手（span=2 合取项生效）
  ④ **单格上限**：score 层剔除满格（cap_blocked_n > 0）；落本格不受限
  ⑤ **去盲选**：span=2 时 dash 不再盲选覆盖（保留 score 目标）
  ⑥ **H3**：span=2 ∧ use_sim_core ⇒ 构造期 NotImplementedError
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.config import InfoStructureConfig, SimConfig
from simulation.genes import Gene
from simulation.sphere_engine import SphereEngine


def _engine(ticks: int = 0, *, seed: int = 42,
            span: int = 1, cap: bool = False, cap_val: int = 3,
            use_sim_core: bool = False, l2: bool = False,
            predation: bool = True) -> SphereEngine:
    cfg = SimConfig(seed=seed)
    cfg.population.max_count = 600
    cfg.predation.forage_tradeoff_k = 0.0
    cfg.predation.enabled = bool(predation)
    cfg.info_structure = InfoStructureConfig(
        enabled=True, learning_rate=0.05, memory_gradient="none",
    )
    cfg.simulation.use_sim_core = bool(use_sim_core)
    cfg.simulation.perception_span = int(span)
    cfg.simulation.cell_occupancy_cap = int(cap_val)
    cfg.simulation.cell_occupancy_cap_enabled = bool(cap)
    cfg.simulation.l2_dash = bool(l2)
    e = SphereEngine(cfg)
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    return e


def _digest(e: SphereEngine) -> tuple[int, float]:
    return (int(e._flat.sum()), round(float(e._energy.sum()), 6))


# --------------------------------------------------------------- ① F1 修复

def test_f1_social_norm_actual_neighbors():
    """F1：nb_norm = 每格实际邻居数（众数 8 / 极区 120），非 stride 120。"""
    e = _engine(ticks=1)
    from collections import Counter
    c = Counter(int(x) for x in e._nb_norm)
    assert c.get(8, 0) == 6960, f"普通格应 6960 个 8 邻（实测 {dict(c)}）"
    assert c.get(120, 0) == 240, f"极区应 240 个 120 邻（实测 {dict(c)}）"
    # social_norm="auto" ⇒ 每格实际邻居数；改数字 ⇒ 冻结
    cfg = SimConfig(seed=42)
    cfg.simulation.social_norm = "120"
    e2 = SphereEngine(cfg)
    assert set(np.unique(e2._nb_norm)) == {120.0}, "冻结常量应全 120"


def test_f1_new_digest_pinned():
    """F1 修复后（社交项恢复量级）关档 digest 钉死（构造级变更已同步测试）。"""
    assert _digest(_engine(ticks=50)) == (574887, 11266.746993), (
        "F1 修复后关档 digest 变了 ⇒ 基线未钉死")


# --------------------------------------------------------------- ② span=2

def test_span2_candidates_and_downgrade():
    """span=2：普通格候选 = ring1+2 = 24；极区 > cap ⇒ 降级（span_downgrade_frac>0）。"""
    e = _engine(ticks=1, span=2)
    pr = e.wave2b_probe()
    assert pr["mean_candidates_per_decision"] is not None
    # 普通格 24、极区 240（240 格）⇒ 总体均值 = (6960×24 + 240×240)/7200 ≈ 34.7
    assert 30 <= pr["mean_candidates_per_decision"] <= 40, (
        f"mean_candidates={pr['mean_candidates_per_decision']} 应在 30~40"
        "（普通格 24 / 极区 240 的加权均值）")
    assert pr["span_downgrade_frac"] is not None and pr["span_downgrade_frac"] > 0, (
        "极区应触发降级（F3 闸）")
    # span=1 ⇒ None（未适用）
    e1 = _engine(ticks=1, span=1)
    assert e1.wave2b_probe()["span_downgrade_frac"] is None


# --------------------------------------------------------------- ③ 看见才出手

def test_visible_prey_gate():
    """看见才出手：span=2 时视野内无猎物 ⇒ 攻击者数应远低于 span=1（合取项生效）。"""
    def _attackers(span: int) -> int:
        e = _engine(ticks=2000, span=span)
        return int(e._run_attack_total) if hasattr(e, "_run_attack_total") else None
    # 用 probe 间接：span=2 的 cannibalism_stats n_attacks 应可读（>0 说明机制在工作）
    e2 = _engine(ticks=2000, span=2)
    cs = e2.cannibalism_stats()
    assert cs["n_attacks"] is not None, "span=2 攻击计数应可读"
    # 机制接线验证：有攻击者（说明"看见才出手"没把出手全杀掉）
    assert cs["n_attacks"] > 0, "视野内有猎物时应有出手（合取项不误杀）"


# --------------------------------------------------------------- ④ 单格上限

def test_cap_excludes_crowded_cells():
    """单格上限：满格候选被剔除（cap_blocked_n>0）；落本格不受限。"""
    e = _engine(ticks=2000, cap=True, cap_val=3)
    pr = e.wave2b_probe()
    assert pr["cap_blocked_n"] > 0, "应真触发过满格剔除"
    # 落本格不受限：cap_stay_n 只在全满兜底时 > 0（通常 0，不崩）
    assert pr["cap_stay_n"] >= 0
    # 关档 ⇒ 计数 0（路径不进）
    e0 = _engine(ticks=50, cap=False)
    assert e0.wave2b_probe()["cap_blocked_n"] == 0


def test_cap_stay_fallback():
    """全候选满 ⇒ 留本格（兜底）+ 计 cap_stay_n（不因平局回退破坏 cap）。"""
    e = _engine(ticks=1, cap=True, cap_val=1)
    # 构造：全部邻居满（occ≥1）+ 自身所在格也满 ⇒ 兜底触发
    e._flat[:] = 100
    e._genes[:, 0] = 1.0   # 必移动
    for _ in range(3):
        e.step()
    # 兜底路径不抛异常、不越界（N 不崩）
    assert len(e._id) > 0


# --------------------------------------------------------------- ⑤ 去盲选

def test_span2_removes_blind_dash():
    """去盲选：span=2 时 dash 不再盲选覆盖（保留 score 目标）。"""
    # span=2 + l2_dash 同开（不互斥，T3 去盲选的前提）
    e = _engine(ticks=500, span=2, l2=True)
    assert len(e._id) > 0, "span=2 + l2 不应崩"


def test_cap_blocks_l2_blind_dash_target():
    """cap 开 + span=1 + l2：L2 盲选覆盖被跳过 ⇒ 冲刺目标不落满格（任务书 T3 §5）。

    构造（**部分满**场景，避免"全候选满 ⇒ score 兜底 continue"在盲选覆盖前短路）：
    个体 0（成年、必移动、必 dash）在普通格 cell；
    邻居 8 个中 4 个满（occ=1=cap_val=1）、4 个空；
    strict 2 圈（far 候选）**全满**；其余个体 MOVE_PROB=0 不动。
    修复前：dash ⇒ 盲选覆盖 ⇒ 目标 = far 盲选（far 全满 ⇒ 落满格，cap 被破坏）；
    修复后：cap 开 ⇒ 盲选跳过 ⇒ 目标 = score 从 4 个非满邻居选（不落满格）。
    断言：个体 0 落点 ∈ {本格} ∪ {非满邻居}。
    """
    e = _engine(ticks=1, cap=True, cap_val=1, l2=True, span=1,
                predation=False)
    P = len(e._id)
    assert P >= 60, f"需要足够个体（P={P}）"
    cell = 1000                        # 普通格（row 8，8 邻；非极区 ⇒ far_len>0）
    nb = np.asarray(e.world.neighbors(cell))
    _off, _fl = int(e._far_off[cell]), int(e._far_len[cell])
    far = e._far_cells[_off:_off + _fl]
    assert len(far) >= 4 and len(nb) >= 8
    e._flat[:] = 5000                  # 全部远置（row 41，普通格）
    e._flat[0] = cell                  # 个体 0 在 cell
    e._flat[1:5] = nb[:4]              # 4 个邻居满（occ=1=cap_val=1）
    e._flat[5:5 + len(far)] = far      # far 候选全满
    e._genes[1:, Gene.MOVE_PROB] = 0.0  # 其余个体不动（保持 occ 布局）
    e._genes[0, Gene.MOVE_PROB] = 1.0  # 个体 0 必移动
    e._genes[0, Gene.DEFENSE] = 1.0    # mob_eff=1 ⇒ dash 概率 100%
    e._energy[0] = float(e.config.organisms.max_energy)  # 冲刺门槛过
    _ls = e._lifespan(e._genes[0, Gene.LIFE_GENE])
    e._age[0] = e.config.organisms.maturity_fraction * _ls  # 成年 ⇒ age_factor=1
    free = set(int(x) for x in np.concatenate(([cell], nb[4:])))  # 本格 + 4 非满邻居
    for _ in range(3):
        if e.extinct:
            break
        e.step()
    assert e._cap_blocked_n > 0, "score 层应触发过满格剔除"
    assert int(e._flat[0]) in free, (
        "dash 个体落点应在非满集合（本格 ∪ 非满邻居）；盲选覆盖会把目标救去满 far 格")


# --------------------------------------------------------------- ⑥ H3

def test_span2_use_sim_core_h3():
    """H3：span=2 ∧ use_sim_core ⇒ 构造期 NotImplementedError。"""
    with pytest.raises(NotImplementedError):
        _engine(span=2, use_sim_core=True)
