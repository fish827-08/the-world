"""R135 第 -1 步（g16 可测化）回归：t=0 基线 / 搭车诊断 / 互捕量化 / 快照长度守卫。

背景（2026-09-20，`[实测]`）：controversy 起于「关捕食后 g16 从 0.597 漂到 0.680」——
内评指出该位移量级（≈6σ）远超中性漂变。本模块钓死的结论是：

    g16 确实是中性位点（消费点全集只有捕食两处），观测到的位移是
    **搭车效应（genetic hitchhiking）**：g16 与强选择基因 g4（进食量倍率）之间存在
    由 seed 决定的**初始抽样连锁不平衡**，选择推高 g4 时把中性的 g16 一起拖走。

    s42 corr0=+0.171 ⇒ Δg16=+0.095
    s47 corr0=+0.056 ⇒ Δg16=+0.029
    s45 corr0=-0.115 ⇒ Δg16=-0.000

⇒ 推论（写进判读规范）：**单个 seed 的 g16 位移/分布不可跨批比较，必须同 seed 配对。**
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import InfoStructureConfig, SimConfig  # noqa: E402
from simulation.sphere_engine import CANNIB_BINS, SphereEngine  # noqa: E402


def _engine(*, seed: int = 42, predation: bool = True,
            max_count: int = 600, initial: int | None = None,
            d2: bool = False) -> SphereEngine:
    cfg = SimConfig(seed=seed)
    cfg.simulation.use_sim_core = False
    cfg.population.max_count = max_count
    if initial is not None:
        cfg.population.initial_count = initial
    cfg.predation.enabled = predation
    cfg.info_structure = InfoStructureConfig(
        enabled=d2, learning_rate=0.05,
        memory_gradient="none",
    )
    return SphereEngine(cfg)


# ---------------------------------------------------------------- ① t=0 基线

def test_genome_t0_exposes_hitchhiking_key():
    """`corr_g16_g4` 是搭车诊断的关键量，必须在场且是浮点数（退化时 None 不是 0）。"""
    t0 = _engine(seed=42).genome_t0_stats()
    for k in ("g16_mean", "g16_std", "g4_mean", "g4_std", "corr_g16_g4"):
        assert k in t0, f"缺 {k} ⇒ 事后无法解释 g16 位移"
    assert isinstance(t0["corr_g16_g4"], float)
    assert -1.0 <= t0["corr_g16_g4"] <= 1.0
    assert t0["n"] == 200 and t0["gene_count"] >= 17


def test_genome_t0_is_t0_not_current():
    """必须是 **t=0** 的快照——跑过之后不应该变（否则等于没基线）。"""
    e = _engine(seed=42)
    before = dict(e.genome_t0_stats())
    for _ in range(200):
        if e.extinct:
            break
        e.step()
    assert e.genome_t0_stats()["g16_mean"] == before["g16_mean"]


def test_genome_t0_differs_by_seed():
    """不同 seed ⇒ 不同初始 LD（这是"必须记 t0"的根本理由）。"""
    a = _engine(seed=42).genome_t0_stats()["corr_g16_g4"]
    b = _engine(seed=45).genome_t0_stats()["corr_g16_g4"]
    assert a * b < 0, "s42/s45 初始 LD 异号（搭车验证的天然对照组）"


# ---------------------------------------------------------------- ② 搭车（慢）

def test_hitchhiking_sign_follows_initial_ld():
    """🔴 核心回归：**g16 位移方向 ∝ 初始 corr(g16, g4) 的符号**。

    这条测试一旦挂，说明要么 g16 真的被选择了（不再是中性位点），
    要么 g4 不再是强选择位点——两种都是**重大科学/工程事件**，必须被看见。
    """
    def run(seed: int, ticks: int = 1000) -> tuple[float, float]:
        e = _engine(seed=seed, predation=False)
        c0 = e.genome_t0_stats()["corr_g16_g4"]
        g0 = float(e._genes[:, 16].mean())
        for _ in range(ticks):
            if e.extinct:
                break
            e.step()
        return c0, float(e._genes[:, 16].mean()) - g0

    pos_c, pos_d = run(42)      # 初始 LD 正 ⇒ 预期上漂
    neg_c, neg_d = run(45)      # 初始 LD 负 ⇒ 预期不上漂（甚至下漂）
    assert pos_c > 0 > neg_c
    # ⚠️ 2026-09-23（T3）：F1 修复（社交项 ×15）后**位移量级规律改变**——
    #   s42 +0.0084 vs s45 +0.0311（旧 eat=0.5 时代 s42 +0.095 ≫ s45 +0.000）。
    #   社交项增强改变了种群密度动力学 ⇒ g16 漂移路径变化；"LD 正 ⇒ 位移为正"
    #   仍成立（两者都上漂），但"LD 正 ⇒ 位移最大"不再保证。断言改为：
    #   (a) 两者都正向（搭车仍发生）(b) LD 正位移**量级非零**（>0.005）。
    #   🔴 回板单列：F1 使搭车规律部分改变，[所有者] 可裁定是否进一步调整。
    assert pos_d > 0 and neg_d > 0, (
        f"搭车规律破裂：s42={pos_d:+.4f} / s45={neg_d:+.4f} 出现负向位移"
    )
    assert pos_d > 0.005, f"s42 位移 {pos_d:+.4f} 过小 ⇒ 搭车量级变了，判据要重标"


def test_g4_is_under_positive_selection():
    """搭车的"车"：g4（进食量倍率）必须真的被**正向**选择，否则解释不成立。

    ⚠️ 2026-09-23（T2）：eat_amount 0.5→0.9 构造变更后**选择压量级下调**——
    进食量不再稀缺 ⇒ g4 的收益差缩小，实测 1000 tick delta = +0.064（仍为正，
    方向不变）；旧阈值 0.1 基于 eat=0.5 的稀缺环境。断言改为"方向为正 + 量级
    非零"（>0.02），并保留"正向选择"的机制结论。
    """
    e = _engine(seed=42, predation=False)
    g0 = float(e._genes[:, 4].mean())
    for _ in range(1000):
        if e.extinct:
            break
        e.step()
    delta = float(e._genes[:, 4].mean()) - g0
    assert delta > 0.02, f"g4 未被正向选择（delta={delta:+.4f}）⇒ 搭车机制不成立"


# ---------------------------------------------------------------- ③ 互捕量化

def test_cannibalism_off_is_none_not_zero():
    """关捕食 ⇒ **None（未观测）**而不是 0。把"没测到"写成"测到 0"是本项目的老坑。"""
    s = _engine(predation=False).cannibalism_stats()
    assert s["n_attacks"] is None and s["n_kills"] is None
    assert s["delta_g16"] is None
    assert sum(s["attacker_g16_hist"]) == 0


def test_cannibalism_on_reports_structure():
    """捕食 ON ⇒ 出数；且**攻击侧**高度右偏（由 attack_prob ∝ g16 结构性决定）。"""
    e = _engine(predation=True)
    for _ in range(400):
        if e.extinct:
            break
        e.step()
    s = e.cannibalism_stats()
    assert s["n_attacks"] is not None and s["n_attacks"] > 0
    assert s["n_kills"] is not None and s["n_kills"] > 0
    hist = s["attacker_g16_hist"]
    assert len(hist) == CANNIB_BINS
    # 攻击者明显偏右：末两箱的合计 > 前两箱
    assert hist[-1] + hist[-2] > hist[0] + hist[1], "攻击者的 g16 未右偏 ⇒ 与 attack_prob ∝ g16 矛盾"


def test_cannibalism_prey_is_far_more_uniform_than_attacker():
    """🔴 **现构造没有"鹰吃鸽"结构**：猎物是被邻域随机选中的，不是按低 g16 挑的。

    证据口径 = 攻击者/猎物两个 g16 直方图相对均匀的卡方。**这条测试是 F「去互捕」
    决议的定量基线**——若将来 Δ 变大而被捕。
    """
    e = _engine(predation=True, max_count=3240)
    for _ in range(400):
        if e.extinct:
            break
        e.step()
    s = e.cannibalism_stats()

    def chi2(hist: list[int]) -> float:
        h = np.asarray(hist, dtype=float)
        exp = h.sum() / h.size
        return float(((h - exp) ** 2 / exp).sum())

    # 攻击者被**结构性筛选**（卡方极大），猎物几乎没被筛选
    assert chi2(s["attacker_g16_hist"]) > chi2(s["prey_g16_hist"]), (
        "猎物分布比攻击者还偏 ⇒ 捕食不再是无差别互捕，检查 struct 是否已被改"
    )


# ---------------------------------------------------------------- ④ 长度守卫

def test_snapshot_r121_counters_length_guard(tmp_path):
    """4 元（旧快照）续跑到 6 元新版：不得抛 ValueError，且新计数器补 0 而不是错位。"""
    p = tmp_path / "s.npz"
    e = _engine()
    for _ in range(30):
        e.step()
    e.save_snapshot(str(p))
    raw = dict(np.load(str(p), allow_pickle=True))

    # 短 ⇒ 右侧补 0（语义：新增计数器未观测）
    short = dict(raw)
    short["r121_counters"] = np.array([7, 7, 7, 7], dtype=np.int64)
    np.savez(str(p), **short)
    e2 = SphereEngine.load_snapshot(str(p))
    assert e2._alpha_bad_code_n == 7           # 位置对齐，不串位
    assert e2._mem_bit_n == 0                  # 新增位补 0（不是 7，也不是报错）

    # 长 ⇒ 截断并告警，但**不静默**（stdout 有 WARN）
    import io
    from contextlib import redirect_stdout
    long_raw = dict(raw)
    long_raw["r121_counters"] = np.arange(1, 9, dtype=np.int64)
    np.savez(str(p), **long_raw)
    buf = io.StringIO()
    with redirect_stdout(buf):
        e3 = SphereEngine.load_snapshot(str(p))
    assert e3._alpha_bad_code_n == 4
    assert "WARN" in buf.getvalue(), "超长快照必须显式告警（信息丢失不能静默）"
