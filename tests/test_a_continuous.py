"""R135 第 3 步 **A-连续**（凸 trade-off）回归。

设计（裁定 checklist）：
  * 一次只动**曲率**一项：`forage_mult(g16) = (1 − g16) ** k`，其余机制全关；
  * `k = 0`（默认）⇒ 恒 1 ⇒ **与旧版逐位一致**（C7）；
  * `k > 1` ⇒ **凸（加速下降）** ⇒ 中间态杂食者两边都不精（Geritz et al. 1998 的分支条件）。
    云端开发者 §三 的陡峭表最接近 k≈1.74；本批取 **k=2.0** 以保证凸度足够。
    对照（g16 → 取食倍率）：
        g16:   0.0    0.2    0.5    0.8    1.0
        表:   1.00   0.80   0.30   0.05   0.00
        k=2:  1.00   0.64   0.25   0.04   0.00

⚠️ 只动**取食侧**：捕猎成功率侧（`energy_ratio × (0.5 + g16×0.5)`）在 Rust
`predation.rs:99`，按裁定**不动** ⇒ 无需重编、双路径天然一致。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import InfoStructureConfig, SimConfig  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402


def _engine(*, k: float = 0.0, seed: int = 42, max_count: int = 600,
            d2: bool = False, ticks_run: int = 0) -> SphereEngine:
    cfg = SimConfig(seed=seed)
    cfg.simulation.use_sim_core = False
    cfg.population.max_count = max_count
    cfg.predation.forage_tradeoff_k = float(k)
    cfg.info_structure = InfoStructureConfig(
        enabled=d2, learning_rate=0.05, memory_gradient="none",
    )
    e = SphereEngine(cfg)
    for _ in range(ticks_run):
        if e.extinct:
            break
        e.step()
    return e


def _digest(e: SphereEngine, ticks: int = 80) -> tuple:
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    return int(e._flat.sum()), round(float(e._energy.sum()), 6), int(e.tick)


# ---------------------------------------------------------------- ① 形状（纯数学）

def test_tradeoff_shape_is_convex_for_k_gt_1():
    """凸的定义：**中间点低于线性插值** ⇒ 杂食者吃亏（分支的前提）。"""
    g = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    lin = 1.0 - g                                  # k=1 的线性基准
    for k in (1.74, 2.0):
        conv = np.power(1.0 - g, k)
        mid = conv[2]
        assert mid < lin[2], f"k={k} 中间点未低于线性 ⇒ 不凸，分支条件不成立"
        # 单调不增 + 端点锚定
        assert np.all(np.diff(conv) <= 1e-12)
        assert abs(conv[0] - 1.0) < 1e-12 and abs(conv[-1]) < 1e-12


def test_k_zero_is_identity():
    """k=0 ⇒ 倍率恒 1（这是"默认关闭"的语义，不是"取食为 0"）。"""
    g = np.linspace(0.0, 1.0, 11)
    assert np.allclose(np.power(1.0 - g, 0.0), 1.0)


def test_negative_k_rejected():
    """负曲率会让高 g16 **吃得更多**（反向选择）⇒ 配置期硬失败，不能静默。"""
    import pytest

    from simulation.config import PredationConfig
    with pytest.raises(AssertionError):
        PredationConfig(forage_tradeoff_k=-0.5)


# ---------------------------------------------------------------- ② 行为活性

def test_k_gt0_changes_trajectory():
    """防静默 no-op（教训 11）：开了曲率，轨迹**必须**不同。"""
    base = _digest(_engine(k=0.0))
    on = _digest(_engine(k=2.0))
    assert base != on, "k=2 与 k=0 轨迹逐位相同 ⇒ 开关没接进食（静默 no-op）"


def test_high_g16_eats_less():
    """极性（F-R18：方向是契约的一部分）：同样的胃容量缺口下，g16 高 ⇒ 吃得少。"""
    cfg = SimConfig(seed=42)
    cfg.simulation.use_sim_core = False
    cfg.population.max_count = 600
    cfg.predation.forage_tradeoff_k = 2.0
    cfg.info_structure = InfoStructureConfig(enabled=False)
    e = SphereEngine(cfg)
    # 人为把 g16 设成两个极端组（其余基因不动），跑相同 tick 后比较胃粮
    P = e._genes.shape[0]
    half = P // 2
    e._genes[:half, 16] = 0.05        # 低攻击（食草型）
    e._genes[half:, 16] = 0.95        # 高攻击（捕食型）
    for _ in range(60):
        if e.extinct:
            break
        e.step()
        # 每 tick 重置 g16，防止选择把它洗掉（本测试只验**当刻**的取食极性）
        # ⚠️ P 会随生死变化 ⇒ 索引必须**每 tick 重算**（首版用固定 idx ⇒ IndexError）。
        P_now = int(e._genes.shape[0])
        h = P_now // 2
        e._genes[:h, 16] = 0.05
        e._genes[h:, 16] = 0.95
    P_now = int(e._genes.shape[0])
    h = P_now // 2
    low_idx = np.arange(h)
    high_idx = np.arange(h, P_now)
    diff = float(e._stomach[high_idx].mean()) - float(e._stomach[low_idx].mean())
    # 高 g16 组胃粮更少（或相等）：因取食倍率被压低
    assert diff <= 1e-9, (
        f"高 g16 组胃粮反而更多（Δ={diff:+.4f}）⇒ 极性反了，trade-off 装反了方向"
    )


# ---------------------------------------------------------------- ③ C7 中立性

def test_k0_is_bitwise_equivalent_to_baseline():
    """🔴 C7：默认（k=0）必须与**未启用该机制**时逐位一致，否则历史批不可比。

    用"同一 seed 跑同一段"对拍：`forage_tradeoff_k=0.0` 与显式构造默认配置等价。
    更强的保证由 `test_memory_gradient.py` / `test_pc1_switches.py` 里既有的
    digest 钉死测试提供（它们跑的是 k=0 的默认路径）。
    """
    a = _digest(_engine(k=0.0), ticks=50)
    b = _digest(_engine(k=0.0), ticks=50)
    assert a == b, "同配置两次跑不一致 ⇒ RNG 或状态不确定（不应发生）"
    # 与既有钉死 digest 同源（test_memory_gradient 用 D2 开、max_count=600、50 tick）
    cfg = SimConfig(seed=42)
    cfg.simulation.use_sim_core = False
    cfg.population.max_count = 600
    cfg.predation.forage_tradeoff_k = 0.0
    cfg.info_structure = InfoStructureConfig(
        enabled=True, learning_rate=0.05, memory_gradient="none",
    )
    e = SphereEngine(cfg)
    for _ in range(50):
        if e.extinct:
            break
        e.step()
    d = (int(e._flat.sum()), round(float(e._energy.sum()), 6))
    # 基线取自 **改动前的 HEAD**（`git stash` 实测）：50 tick / seed 42 / max_count 600 /
    # D2 enabled / memory_gradient=none。
    # ⚠️ 勿与 `test_memory_gradient.py` 的 (532906, 7954.475077) 混淆——那条是
    # `enabled=False + orientation` 的配置，与本文不同（我第一版就抄错了，测试当场挂）。
    assert d == (573985, 8171.692943), (
        f"k=0 的 50 tick digest 变了（{d}）⇒ **破坏了与历史批的逐位可比性**，"
        "默认分支必须严格保持原式"
    )
