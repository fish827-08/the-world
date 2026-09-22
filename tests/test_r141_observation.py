"""R141 / R138 观测列回归：分通道能量记账 + 真决斗三级拆分 + g16 初始投放。

派工来源（2026-09-21 所有者）：
  * R141 §三.1：**分通道能量记账**（能量校准预实验的前提）——五通道 + g16 三分箱。
  * R141 §三.2：`init_g16_clusters` 初始投放（默认关 = 旧行为）。
  * R138 §二 问答 1 / R141：**真决斗三级拆分**——旧口径把"根本没出手"也算进分母。
  * R139/R140 P0：`g16_hist_0..9` / `mean_energy` / 死因时间序列 / `g3`。

⭐ 本模块钉死的最重要一条：**名义成功率 ≠ 真实出手成功率**。
   实测（patchy/16 码/软顶 0.6/N≈1944/8k tick）：名义 7.93% 而**真实出手 34.64%**——
   因为 **76% 的"攻击"其邻格根本没有猎物**（`skip_no_prey`）。
   在低密度世界（max_count=600 无软顶）差距更极端：3.1% vs 33.9%（**低估 11 倍**）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import InfoStructureConfig, SimConfig  # noqa: E402
from simulation.sphere_engine import (  # noqa: E402
    EC_NAMES, EC_N, SphereEngine,
)


def _engine(*, d2: bool = True, ticks: int = 300, seed: int = 42,
            use_core: bool = False, max_count: int = 600) -> SphereEngine:
    cfg = SimConfig(seed=seed)
    cfg.simulation.use_sim_core = use_core
    cfg.population.max_count = max_count
    cfg.info_structure = InfoStructureConfig(enabled=d2, learning_rate=0.05)
    e = SphereEngine(cfg)
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    return e


# ---------------------------------------------------------------- ① 分通道记账

def test_channel_accounting_has_seven_channels_and_boxes():
    """R165 0-2（2026-09-23）：通道 **6 → 7**，新增 `intake_scav`（尸体收入单列）。

    加它的原因：食腐质量在胃里与植物质量不可区分 ⇒ 消化时整笔记进 `intake_forage`
    ⇒ 账本无法分离"尸体收入 vs 植物收入"（R163 复核发现 3）。
    """
    el = _engine().energy_ledger()
    assert el["path"] == "python"
    # 派工单 §1.3 锁定的结构（段名/键名勿改——calib_solve.py 按此消费）
    assert set(el["groups"].keys()) == {"lo", "mid", "hi"}
    assert len(EC_NAMES) == EC_N == 7
    assert "intake_photo" in EC_NAMES          # 光合是独立收入通道（不经过胃）
    assert "intake_scav" in EC_NAMES           # R165 0-2：尸体收入单列
    for b in ("lo", "mid", "hi"):
        assert "intake_scav_sum" in el["groups"][b]
    for k in ("obs_count", "global", "groups", "prey", "attack"):
        assert k in el


def test_channel_accounting_is_additive():
    """收入通道必须为正、支出通道必须为正（符号约定：全部记"量"而非"带符号量"）。"""
    g = _engine(ticks=400).energy_ledger()["global"]
    for k in ("intake_forage_sum", "intake_photo_sum"):
        assert g[k] > 0, f"{k} 应 > 0（收入通道）"
    for k in ("cost_meta_sum", "cost_move_sum"):
        assert g[k] > 0, f"{k} 应 > 0（支出量）"


def test_channel_accounting_reports_none_on_rust_path():
    """Rust 路径 ⇒ **None（未观测）**，不是 0。把"没测到"写成"测到 0"是本项目老坑。"""
    el = _engine(use_core=True, ticks=50).energy_ledger()
    assert el["path"] == "rust"
    assert "global" not in el
    assert "None" in el["note"] or "未观测" in el["note"]


# ---------------------------------------------------------------- ② 三级拆分

def test_duel_split_is_consistent():
    """`名义 = Σskip + 真出手` 必须**精确闭合**（否则计数器漏事件）。"""
    d = _engine(ticks=500).cannibalism_stats()["duel"]
    assert d is not None
    total = (d["skip_no_energy"] + d["skip_no_prey"]
             + d["skip_already_eaten"] + d["real_attempts"])
    assert total == d["nominal_attackers"], (
        f"三级拆分不闭合：skip+real={total} vs nominal={d['nominal_attackers']}"
    )


def test_real_success_rate_exceeds_nominal():
    """🔴 核心性质：**真实出手成功率 > 名义成功率**（旧口径把未出手算进分母 ⇒ 低估）。

    这条一旦不成立，说明"未出手"的分类错了（或攻击成本被算到了未出手上）。
    """
    d = _engine(ticks=600).cannibalism_stats()["duel"]
    assert d["real_attempts"] > 0, "600 tick 内应至少有一次真出手（否则场景设计失败）"
    assert d["success_rate_real"] > d["success_rate_nominal"], (
        f"真实 {d['success_rate_real']} 未高于名义 {d['success_rate_nominal']} ⇒ 拆分语义错"
    )
    assert d["real_attempts"] < d["nominal_attackers"], "真出手应严格少于名义攻击者"


def test_skip_reasons_are_reported_separately():
    """三类 skip 必须**分开**报（合并了就丢掉了"为什么没人可打"的信息）。"""
    d = _engine(ticks=500).cannibalism_stats()["duel"]
    for k in ("skip_no_energy", "skip_no_prey", "skip_already_eaten"):
        assert k in d and isinstance(d[k], int)


# ---------------------------------------------------------------- ③ 初始投放

def test_init_g16_clusters_splits_evenly():
    cfg = SimConfig(seed=42)
    cfg.simulation.use_sim_core = False
    cfg.population.max_count = 600
    cfg.genome.init_g16_clusters = "0.05,0.5,0.9"
    e = SphereEngine(cfg)
    g = e._genes[:, 16]
    vals, cnts = np.unique(np.round(g, 6), return_counts=True)
    assert list(np.round(vals, 3)) == [0.05, 0.5, 0.9]
    assert max(cnts) - min(cnts) <= 2, f"三组未均分：{cnts}"
    # 摘要必须与**实际**基因一致——注意三组大小可差 1（200/3 ⇒ 67/67/66），
    # 所以**不能**假设算术平均 (0.05+0.5+0.9)/3（首版就栽在这，差 0.00208）。
    expected = float(sum(v * c for v, c in zip(vals, cnts)) / cnts.sum())
    assert abs(e.genome_t0_stats()["g16_mean"] - expected) < 1e-9


def test_init_g16_clusters_default_is_old_behavior():
    """默认空串 ⇒ **不触碰** g16（与旧行为一致）。"""
    cfg = SimConfig(seed=42)
    cfg.simulation.use_sim_core = False
    cfg.population.max_count = 600
    assert cfg.genome.init_g16_clusters == ""
    e = SphereEngine(cfg)
    e2 = SphereEngine(SimConfig(seed=42))
    assert np.allclose(e._genes[:, 16], e2._genes[:, 16])


# ---------------------------------------------------------------- ④ P0 观测列

def test_hist10_normalizes_to_n():
    from experiments.a4_verify_capacity import G16_BINS, _hist10
    assert G16_BINS == 10
    x = np.array([0.0, 0.05, 0.5, 0.99, 1.0, 0.5])
    h = _hist10(x)
    assert len(h) == 10 and sum(h) == x.size
    assert _hist10(np.zeros(0)) == [0] * 10      # 空 ⇒ 报零向量（不是崩溃）


def test_hist10_clips_out_of_range():
    """越界值夹到两端（基因理论上在 [0,1]，但数值路径可能越界 ⇒ 不能崩、不能丢计数）。"""
    from experiments.a4_verify_capacity import _hist10
    h = _hist10(np.array([-0.5, 1.5, 2.0]))
    assert sum(h) == 3
    assert h[0] == 1 and h[-1] == 2
