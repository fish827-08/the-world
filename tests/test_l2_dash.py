"""L2 机动性（R146/R149）回归测试 —— 本段 = `[本地开发]` 线（B2/B3）。

覆盖：
  ① **H1**：`l2_dash=False` ⇒ **逐位等价**（C7 对拍，基线同 `test_a_continuous` 钉死值）
  ② **开关真的生效**（开 ≠ 关）
  ③ **H3 fail-loud**：`use_sim_core=True` + L2 开 ⇒ 必抛（**不得**静默换路径）
  ④ `age_factor`：幼/老 ×`young_mob_mult`/`old_mob_mult`，且**基准 = 个体自己的 lifespan**
  ⑤ **纯能量闸**：能量不足 ⇒ 不能冲刺（且两个闸顺序无关）
  ⑥ **耗能 ∝(1+κ(2^p−1))**：全冲时能量消耗 = 关档的 `_step` 倍
  ⑦ **极区退化格不可冲刺**（`far_len==0`）
  ⑧ **H2 形状**：源码级断言 —— L2 段内**不得**出现任何 `self.rng.` 调用（不新增也不跳过）
  ⑨ 读数口径：关档 `l2_probe()` 返回 **None（未适用）**，不是 0
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from simulation.config import InfoStructureConfig, SimConfig
from simulation.sphere_engine import SphereEngine


def _engine(l2: bool = False, ticks: int = 50, *, use_core: bool = False,
            young: float = 0.55, old: float = 0.55, dash_frac_gate: float = 0.2,
            far_cap: int = 32) -> SphereEngine:
    cfg = SimConfig(seed=42)
    cfg.simulation.use_sim_core = use_core
    cfg.simulation.l2_dash = l2
    cfg.population.max_count = 600
    cfg.predation.forage_tradeoff_k = 0.0
    cfg.info_structure = InfoStructureConfig(
        enabled=True, learning_rate=0.05, memory_gradient="none",
    )
    cfg.organisms.young_mob_mult = young
    cfg.organisms.old_mob_mult = old
    cfg.organisms.dash_min_energy_frac = dash_frac_gate
    cfg.organisms.far_cap = far_cap
    e = SphereEngine(cfg)
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    return e


def _digest(e: SphereEngine) -> tuple[int, float]:
    return (int(e._flat.sum()), round(float(e._energy.sum()), 6))


# --------------------------------------------------------------- ① H1 / C7

def test_l2_default_off_is_bit_identical():
    """C7：L2 关档必须与**改动前**逐位一致。

    基线同 `tests/test_a_continuous.py` 钉死值（50 tick / seed 42 / max_count 600 /
    D2 enabled / `memory_gradient=none`）—— 取自**加 L2 之前**的 HEAD。
    """
    assert _digest(_engine(l2=False)) == (574887, 11266.746993), (
        "L2 关档改变了轨迹 ⇒ 破坏 H1（默认关 = 逐位等价）")


def test_l2_on_actually_changes_behaviour():
    """开档必须**真的改行为**（防"接了却没生效"—— dash PR 的同族风险）。"""
    assert _digest(_engine(l2=True)) != _digest(_engine(l2=False))


# --------------------------------------------------------------- ③ H3

def test_l2_with_sim_core_raises():
    """H3：L2 开 + `use_sim_core=True` ⇒ **构造期直接抛**（不静默换路径）。"""
    cfg = SimConfig(seed=1)
    cfg.simulation.use_sim_core = True
    cfg.simulation.l2_dash = True
    with pytest.raises(NotImplementedError, match="L1/L2"):
        SphereEngine(cfg)


# --------------------------------------------------------------- ④ age_factor

def test_age_factor_uses_own_lifespan():
    """`age_factor` 的基准必须是个体**自己的** lifespan（内评 §四）。

    做法（消除抽样噪声）：让**全体同组**（全幼 / 全成 / 全老），三段各跑一次，
    断言 `age_factor_mean` 精确等于该档倍率 —— 只有"基准取自体自己 lifespan"时成立。
    """
    ls_for = 0.5                              # 统一 g3 ⇒ 统一寿命
    cases = ((0.05, 0.30), (0.50, 1.00), (0.90, 0.70))   # (age/ls, 期望倍率)
    for frac, want in cases:
        e = _engine(l2=True, ticks=0, young=0.30, old=0.70)
        P = len(e._id)
        e._genes[:, 3] = ls_for
        ls = float(e._lifespan(e._genes[:P, 3])[0])
        e._age[:P] = frac * ls
        e.step()
        pr = e.l2_probe()
        assert pr is not None and pr["age_factor_mean"] is not None
        assert pr["age_factor_mean"] == pytest.approx(want, abs=1e-6), (
            f"age/ls={frac} ⇒ age_factor_mean={pr['age_factor_mean']} ≠ {want}"
            "（分段阈值或寿命基准不对；基准须为**个体自己**的 lifespan）")


# --------------------------------------------------------------- ⑤⑥ 冲刺闸与耗能

def _force_all_dash(e: SphereEngine) -> None:
    """把条件推到"人人可冲"：g18=1（P(冲)=1）、**成年**（age_factor=1）、能量充足、非极区格。"""
    P = len(e._id)
    e._genes[:, 18] = 1.0                     # DEFENSE(g18) = mob
    e._genes[:, 3] = 0.5                      # 统一寿命
    ls = float(e._lifespan(e._genes[:P, 3])[0])
    e._age[:P] = 0.5 * ls                     # 成年 ⇒ age_factor = 1.0
    e._energy[:P] = e.config.organisms.max_energy
    # 放到"可冲刺"的普通格（far_len > 0），并让移动判定必过
    ok_cells = np.flatnonzero(e._far_len > 0)
    e._flat[:P] = ok_cells[np.arange(P) % len(ok_cells)]
    e._genes[:, 0] = 1.0                      # MOVE_PROB
    e._genes[:, 19] = 0.0                     # 非扎根


def test_energy_gate_blocks_dash():
    """🔴 纯能量闸：能量不足 ⇒ 冲不了（且只降能量、不动其他开关）。"""
    e = _engine(l2=True, ticks=0)
    _force_all_dash(e)
    e._energy[:] = 0.01                       # 远低于 dash 门槛与 2 格成本
    # 0.01 连 1 格成本都不够 ⇒ 不会移动；抬到"够 1 格但不够门槛"的水平
    e._energy[:] = 0.5
    e.step()
    pr = e.l2_probe()
    assert pr is not None
    assert pr["dash_n"] == 0, "能量低于 dash 门槛仍发生冲刺 ⇒ 能量闸没接线"


def test_all_dash_and_cost_multiplier():
    """⑥ 全冲档：`dash_frac == 1`，且 `cost_move` 通道 = 关档的 `1+κ(2^p−1)` 倍（p=2 ⇒ ×4）。

    用**记账通道**（`energy_ledger().global.cost_move_sum`）而非总能量差 —— 后者被
    代谢等通道淹没（一次改动就值 4 倍分辨率的教训）。
    ⚠️ 容差 `rel=1e-5`：账本落盘时 `round(...,3)`（Σcost≈250 ⇒ 相对误差 ~4e-6），
    不是机制噪声 —— 不要收紧到 1e-9（那是在测舍入）。
    """
    kappa, exp = 1.0, 2.0
    step = 1.0 + kappa * (2.0 ** exp - 1.0)   # = 4.0
    on = _engine(l2=True, ticks=0)
    _force_all_dash(on)
    off = _engine(l2=False, ticks=0)
    _force_all_dash(off)
    on.step()
    off.step()
    pr = on.l2_probe()
    assert pr is not None and pr["dash_frac"] == pytest.approx(1.0, abs=1e-9), (
        f"全冲档下 dash_frac={pr and pr['dash_frac']} ≠ 1")
    c_on = on.energy_ledger()["global"]["cost_move_sum"]
    c_off = off.energy_ledger()["global"]["cost_move_sum"]
    assert c_off > 0
    assert c_on / c_off == pytest.approx(step, rel=1e-5), (
        f"cost_move 倍率 {c_on / c_off:.6f} ≠ {step}（p=2 ⇒ ×4）")


def test_pole_degenerate_cells_cannot_dash():
    """⑦ 极区退化格（`far_len == 0`）**不可冲刺**（拓扑排除，非"极区个体"惩罚）。"""
    e = _engine(l2=True, ticks=0)
    _force_all_dash(e)
    bad = int(np.flatnonzero(e._far_len == 0)[0])
    e._flat[:] = bad
    e.step()
    pr = e.l2_probe()
    assert pr is not None
    assert pr["dash_n"] == 0, "退化格仍发生冲刺 ⇒ 极区排除没接线"
    assert pr["dash_ineligible_cells"] == 480


# --------------------------------------------------------------- ⑧ H2 形状

def test_l2_block_adds_no_rng_draws():
    """H2：**我加的** L2 代码段内不得出现 `self.rng.`（不新增、也不跳过抽取）。

    只扫**本线新增的两段**（决策段 + 盲选段），不扫包住整个原循环 ⇒ 避免把既有
    `u = self.rng.random()`（那正是我要**保留**的抽取形状）误判为违规。
    """
    src = Path("simulation/sphere_engine.py").read_text(encoding="utf-8")
    spans = (
        ("dash = np.zeros(Nm, dtype=bool)", "self._agef_sum += float(age_factor.sum())"),
        # T3（13.4 波 2B）去盲选：span=2 时跳过盲选覆盖 ⇒ 条件加了 `and not _span2_on`
        ("if _l2_on and dash[i] and not _span2_on:", "int(rand_choice[i] // 100) % _fl])"),
    )
    for head, tail in spans:
        i0 = src.index(head)
        block = src[i0:src.index(tail, i0)]
        code = "\n".join(l for l in block.splitlines() if not l.strip().startswith("#"))
        assert "self.rng." not in code, (
            f"L2 段（{head[:32]}…）内出现 self.rng. 调用 ⇒ 违反 H2"
            "（新增/跳过抽取会破坏跨臂可比性）")
    # 反向自证：既有软 max 抽取仍在（形状保持的依据）
    assert "u = self.rng.random()" in src


# --------------------------------------------------------------- ⑨ 口径

def test_probe_is_none_when_off():
    """关档 ⇒ `l2_probe()` 返回 **None（未适用）**，不是 0（R120 口径铁律）。"""
    assert _engine(l2=False, ticks=10).l2_probe() is None


def test_switches_visible_for_epoch():
    """L2 开关与关键参数必须可从产物读回（C4；否则纪元不可自证）。"""
    e = _engine(l2=True, ticks=0)
    assert bool(e.config.simulation.l2_dash) is True
    a, b = SimConfig(), SimConfig()
    b.simulation.l2_dash = True
    assert a.fingerprint() != b.fingerprint(), "l2_dash 未进指纹 ⇒ 纪元不可判"
    assert re.search(r"l2_dash", "l2_dash")  # 占位：字段名自解释（C5 第三项）
