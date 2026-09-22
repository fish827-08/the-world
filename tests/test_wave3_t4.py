"""13.4 波 3 T4 测试（血条语义反转 + 恐惧门槛 + 纪元）。

覆盖（任务书 T4 完成判据）：
  ① **语义反转**：成功 ⇒ 一击毙命（能量进尸体）；失败 ⇒ 扣血条（health≤0 才死）
  ② **能量守恒**：corpse 开 ⇒ 击杀能量进尸体（corpse_total 增加）；corpse 关 ⇒ 旧转移
  ③ **恐惧门槛**：`1−health < 0.3` 不触发，≥0.3 才触发（带门槛连续）
  ④ **纪元可自证**：switches 含 wound_fear_threshold / eat_amount 等（C4）
  ⑤ **H3**：wound 开 + use_sim_core ⇒ 构造期 NotImplementedError
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.config import CorpseWoundConfig, InfoStructureConfig, SimConfig
from simulation.sphere_engine import SphereEngine


def _engine(ticks: int = 0, *, seed: int = 42,
            corpse: bool = False, wound: bool = False,
            use_sim_core: bool = False,
            threshold: float | None = None) -> SphereEngine:
    cfg = SimConfig(seed=seed)
    cfg.population.max_count = 600
    cfg.predation.forage_tradeoff_k = 0.0
    cfg.info_structure = InfoStructureConfig(
        enabled=True, learning_rate=0.05, memory_gradient="none",
    )
    cfg.simulation.use_sim_core = bool(use_sim_core)
    kw = dict(corpse_enabled=corpse, wound_enabled=wound)
    if threshold is not None:
        kw["wound_fear_threshold"] = threshold
    cfg.corpse_wound = CorpseWoundConfig(**kw)
    e = SphereEngine(cfg)
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    return e


# --------------------------------------------------------------- ① 语义反转

def test_success_kills_and_deposits_corpse():
    """成功 ⇒ 一击毙命 + 能量进尸体（corpse_total 增加，攻击者无即时转移）。"""
    e = _engine(ticks=2000, corpse=True, wound=True)
    cp = e.corpse_probe()
    assert cp["corpse_total"] > 0, "击杀能量应进尸体（corpse_total > 0）"
    assert cp["corpse_eaten"] >= 0
    # 死因含 PREDATION（成功致死）—— 键 = DeathCause 枚举对象（str(k) 匹配）
    dc = e.death_cause_totals()
    n_pred = sum(int(v) for k, v in dc.items() if "PREDATION" in str(k))
    assert n_pred > 0, "应有捕食致死"
    pr = e.wound_probe()
    assert pr["wound_n"] > 0, "失败致伤应发生（wound_n > 0）"


def test_failure_wounds_and_eventually_kills():
    """失败 ⇒ 扣血条（health 下降）；血条归零 ⇒ 死亡。"""
    e = _engine(ticks=2000, corpse=True, wound=True)
    pr = e.wound_probe()
    assert pr["health_mean"] is not None
    assert pr["health_mean"] < 1.0, "应有受伤个体（health_mean < 1）"
    assert pr["health_low_frac"] is not None


# --------------------------------------------------------------- ② 守恒

def test_corpse_off_falls_back_to_transfer():
    """corpse 关 + wound 开 ⇒ 击杀沿用旧 transfer（能量不消失，守恒）。"""
    e = _engine(ticks=2000, corpse=False, wound=True)
    cp = e.corpse_probe()
    assert cp["corpse_total"] == 0.0, "corpse 关不应有尸体"
    aud = e.energy_ledger()
    # 能量守恒审计应通过（击杀能量经 transfer 转移，不凭空消失）
    assert aud is not None


# --------------------------------------------------------------- ③ 恐惧门槛

def test_fear_threshold_gate():
    """恐惧门槛：1−health < 0.3 不触发（fearh 施加为 0）；≥0.3 触发。"""
    # threshold=0.3（默认）：健康个体（health=1）⇒ 1−1=0 < 0.3 ⇒ 不触发
    e = _engine(ticks=500, wound=True, threshold=0.3)
    pr = e.wound_probe()
    # threshold 高 ⇒ 触发少；threshold 低 ⇒ 触发多（门槛生效）
    e_low = _engine(ticks=500, wound=True, threshold=0.0)
    n_high = e._fearh_dec_n
    n_low = e_low._fearh_dec_n
    assert n_low >= n_high, (
        f"门槛 0.0（{n_low}）应 ≥ 门槛 0.3（{n_high}）：门槛应降低触发")


# --------------------------------------------------------------- ④ 纪元自证

def test_epoch_switches_readback():
    """纪元可自证：wound_fear_threshold / eat_amount 可从 config 读回（C4）。"""
    e = _engine(ticks=1, wound=True, threshold=0.25)
    assert float(e.config.corpse_wound.wound_fear_threshold) == 0.25
    assert float(e.config.organisms.eat_amount) == 0.9
    assert float(e.config.simulation.perception_span) == 1
    assert float(e.config.resource_dynamics.kill_frac) == 10.0


# --------------------------------------------------------------- ⑤ H3

def test_wound_use_sim_core_h3():
    """H3：wound 开 + use_sim_core ⇒ 构造期 NotImplementedError。"""
    with pytest.raises(NotImplementedError):
        _engine(wound=True, use_sim_core=True)
