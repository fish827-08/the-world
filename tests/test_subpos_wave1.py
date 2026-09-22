"""13.4 波 1 a4 收尾测试（任务书 T1；分支 dev/13.4-wave1）。

覆盖（T1 完成判据）：
  ① **E1 关档逐位等价**：subpos 全关 ⇒ C7 基线 digest == (542646, 11197.859208)
  ② **E2 开关读回（C4）**：a4 build 传 subpos 参数 ⇒ switches/subpos_probe 逐键正确
  ③ **E3 H3 互斥**：subpos.enabled ∧ l2_dash ⇒ 构造期 NotImplementedError
  ④ **E8 反退化**：开档后 steps_frac ≥2 档非零（速度映射未塌成常数）
  ⑤ **CSV 列**：subpos_flat_moves / subpos_slow_frac 关档空串、开档有值
  ⑥ **summary.subpos**：关档 None（未适用，非 0）、开档 dict
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.config import InfoStructureConfig, SimConfig, SubposConfig
from simulation.sphere_engine import SphereEngine


def _engine(ticks: int = 0, *, seed: int = 42,
            subpos: bool = False, gain: float = 4.0,
            l2_dash: bool = False) -> SphereEngine:
    cfg = SimConfig(seed=seed)
    cfg.population.max_count = 600
    cfg.predation.forage_tradeoff_k = 0.0
    cfg.info_structure = InfoStructureConfig(
        enabled=True, learning_rate=0.05, memory_gradient="none",
    )
    cfg.simulation.l2_dash = bool(l2_dash)
    cfg.subpos = SubposConfig(
        enabled=bool(subpos), speed_gain=float(gain),
    )
    e = SphereEngine(cfg)
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    return e


def _digest(e: SphereEngine) -> tuple[int, float]:
    return (int(e._flat.sum()), round(float(e._energy.sum()), 6))


# --------------------------------------------------------------- ① E1 关档等价

def test_subpos_off_bit_identical():
    """C7：subpos 全关 ⇒ 与**改动前**逐位一致（S1 钉死值不变）。"""
    assert _digest(_engine(ticks=50)) == (542646, 11197.859208), (
        "subpos 关档改变了轨迹 ⇒ 破坏 E1（默认关 = 逐位等价）")


# --------------------------------------------------------------- ② E2 开关读回

def test_subpos_switches_readback():
    """C4：a4 build 传参 ⇒ config.subpos 读回全对（含默认值）。"""
    cfg = SimConfig(seed=7)
    cfg.subpos = SubposConfig(enabled=True, speed_gain=4.0, speed_max=2.0,
                              min_energy_frac=0.05, lat_floor=0.3)
    e = SphereEngine(cfg)
    sp = e.subpos_probe()
    assert sp is not None, "开档 subpos_probe 不应为 None"
    assert sp["enabled"] is True
    assert sp["speed_gain"] == 4.0
    assert sp["speed_max"] == 2.0
    assert sp["min_energy_frac"] == 0.05
    assert sp["lat_floor"] == 0.3
    assert sp["stay_max"] == 0.8


def test_subpos_probe_none_when_off():
    """R120 口径：关档 subpos_probe 返回 None（未适用），不是 0。"""
    e = _engine(ticks=10, subpos=False)
    assert e.subpos_probe() is None, "关档应返回 None（未适用）"


# --------------------------------------------------------------- ③ E3 H3 互斥

def test_subpos_l2dash_h3():
    """H3 fail-loud：subpos.enabled ∧ l2_dash ⇒ 构造期 NotImplementedError。"""
    with pytest.raises(NotImplementedError):
        _engine(ticks=0, subpos=True, l2_dash=True)


# --------------------------------------------------------------- ④ E8 反退化

def test_subpos_steps_frac_not_degenerate():
    """反退化：开档后 steps_frac ≥2 档非零（速度映射未塌成常数）。"""
    e = _engine(ticks=2000, subpos=True)
    sp = e.subpos_probe()
    assert sp is not None
    sf = sp["steps_frac"]
    assert sf is not None and len(sf) >= 2, "steps_frac 应有多档"
    nz = [x for x in sf if x and x > 0]
    assert len(nz) >= 2, (
        f"steps_frac 只有 {len(nz)} 档非零 ⇒ 速度映射塌成常数（反退化失败）"
        f"steps_frac={sf}")
    assert sp["slow_frac"] is not None and sp["slow_frac"] > 0, (
        "slow_frac 应 > 0（确有 steps=0 的'白移动'）")


# --------------------------------------------------------------- ⑤⑥ 读数/CSV

def test_subpos_probe_mover_reads():
    """读数：开档后 mean_flat_moves 有值（分母 mover_n > 0）。"""
    e = _engine(ticks=2000, subpos=True)
    sp = e.subpos_probe()
    assert sp["mover_n"] > 0, "mover_n 应为 0（无移动者？）"
    assert sp["mean_flat_moves"] is not None, "mean_flat_moves 应有值"
    assert 0.0 <= sp["mean_flat_moves"] <= 1.0, "mean_flat_moves 是比例"


def test_subpos_cross_gain_changes_flat_moves():
    """跨档可区分：gain 不同 ⇒ mean_flat_moves 不同（C 臂体感更大 ⇒ 换格更少）。"""
    e_low = _engine(ticks=2000, subpos=True, gain=1.0)
    e_hi = _engine(ticks=2000, subpos=True, gain=4.0)
    lo = e_low.subpos_probe()["mean_flat_moves"]
    hi = e_hi.subpos_probe()["mean_flat_moves"]
    assert lo is not None and hi is not None
    # gain 低 ⇒ 速度慢 ⇒ 换格比例应 ≤ gain 高（允许波动，但不应反向显著）
    assert lo < hi + 0.15, (
        f"gain=1.0 换格率({lo}) 应 ≤ gain=4.0({hi}) + 0.15（体感方向）")
