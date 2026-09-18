"""gate 批收尾分析工具的回归测试（纯函数 + 一处 F-R23 家族守卫）。

覆盖：
  · `sign_test_two_sided` 的**精确值**（含 F-R23 口径：`k` = 实际正向数，不是 `n`）
  · `regime` 的两套划法（R38③ 划法 A / 内评 §四.2 划法 B）
  · `pairs_of` 的 **按 seed 显式配对**（缺一侧 ⇒ 跳过；顺序不得错位）
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import gate_pair_analysis as gp  # noqa: E402


# ------------------------------------------------------------ 符号检验精确值
def test_sign_test_exact_values():
    """双侧精确符号检验：`p = 2·P(X≥k)`（上限 1），`k` = max(正,负)。"""
    assert gp.sign_test_two_sided(6, 0) == pytest.approx(2 * (1 / 64))     # 0.03125
    assert gp.sign_test_two_sided(5, 1) == pytest.approx(2 * (7 / 64))     # 0.21875
    assert gp.sign_test_two_sided(4, 2) == pytest.approx(2 * (22 / 64))    # 0.6875
    assert gp.sign_test_two_sided(3, 3) == pytest.approx(1.0)
    assert gp.sign_test_two_sided(0, 0) == pytest.approx(1.0)              # 空样本 ⇒ 不判


def test_sign_test_k_is_actual_positive_count_not_n():
    """🔴 F-R23 家族守卫：`k` 必须随**实际正向数**变化，不得退化常数。

    （原缺陷型：调用点写 `sign_test_pvalue(n, n)` ⇒ n=6 时**恒 0.0156**，与数据无关。）
    """
    assert gp.sign_test_two_sided(6, 0) != pytest.approx(gp.sign_test_two_sided(5, 1))
    assert gp.sign_test_two_sided(6, 0) == pytest.approx(0.03125)
    assert gp.sign_test_two_sided(5, 1) == pytest.approx(0.21875)          # ≠ 1/64
    # 单调：同 n 下，正向越多 ⇒ p 越小
    assert gp.sign_test_two_sided(6, 0) < gp.sign_test_two_sided(5, 1) < gp.sign_test_two_sided(4, 2)


# ------------------------------------------------------------ 区制标签
def test_regime_two_schemes():
    """两套划法并列输出（内评 §四.2：不得只报一种）。"""
    assert gp.regime(0.95, 200) == "PRED/PRED"        # A: pred≥0.9；B: N<1000
    assert gp.regime(0.30, 3240) == "SAT/SAT"         # A: pred<0.9；B: N≥3000
    assert gp.regime(0.50, 2000) == "SAT/过渡"        # A: 非捕食；B: 1000≤N<3000
    assert gp.regime(None, 100) == "?"


# ------------------------------------------------------------ 按 seed 显式配对
def _row(seed: int, **kw) -> dict:
    base = dict(arm="x", seed=seed)
    base.update(kw)
    return base


def test_pairs_of_aligns_by_seed_and_skips_missing():
    """配对必须**按 seed 对齐**；缺一侧 ⇒ 跳过（不得按位置错配）。"""
    rows = {
        "gated_s42": _row(42, rho=0.10),
        "ungated_s42": _row(42, rho=0.05),
        "gated_s43": _row(43, rho=0.20),
        "ungated_s44": _row(44, rho=0.00),      # 缺 gated_s44 ⇒ 该对跳过
        "gated_s45": _row(45, rho=0.30),
        "ungated_s45": _row(45, rho=0.40),
    }
    pr = gp.pairs_of(rows, "gated", "ungated", "rho")
    assert [p[0] for p in pr] == [42, 45], "必须只保留两臂齐备的 seed"
    assert pr[0][3] == pytest.approx(0.05)      # 42: 0.10 − 0.05
    assert pr[1][3] == pytest.approx(-0.10)     # 45: 0.30 − 0.40
    assert len(pr) == 2


def test_pairs_of_skips_none_values():
    """任一侧为 None ⇒ 跳过（不得当 0 处理）。"""
    rows = {
        "gated_s42": _row(42, N=100),
        "ungated_s42": _row(42, N=None),
    }
    assert gp.pairs_of(rows, "gated", "ungated", "N") == []


def test_regime_threshold_constants_are_declared():
    """口径常量必须在模块里显式声明（禁散落字面量 ⇒ 便于审阅对照）。"""
    assert gp.PRED_DOMAIN_MAX == 0.9
    assert gp.N_SAT == 3000
    assert gp.TRANSITION_LO == 1000


def test_oracle_only_labels_exist_and_are_labels_not_field_keys():
    """守卫：`ORACLE_ONLY_LABELS` 每个名字必须是 `FIELDS` 的**标签**（第 1 元），且**不得**
    与内部**字段键**（第 2 元）撞名。

    🔴 2026-09-19 真实事故：我把集合写成标签名、而循环里比对的是**字段键**（如 `ratio`
    vs `oracle_ratio`）⇒ 「非 oracle 臂略去 oracle 指标」**静默失效**（报告照出
    「不显著 n_eff=0」的误导行）。本测试把约定钉死：集合必须 ⊆ 标签集，且 ∩ 键集 = ∅。
    """
    labels = {f[0] for f in gp.FIELDS}
    keys = {f[1] for f in gp.FIELDS}
    missing = gp.ORACLE_ONLY_LABELS - labels
    assert not missing, f"ORACLE_ONLY_LABELS 里有 FIELDS 中不存在的标签：{missing}"
    collide = gp.ORACLE_ONLY_LABELS & keys
    assert not collide, f"ORACLE_ONLY_LABELS 与字段键撞名（集合应为**标签**）：{collide}"


def test_make_name_re_matches_only_given_arms():
    """臂名正则由参数决定（通用化后不得写死 gated/ungated）。"""
    rex = gp.make_name_re("a8", "b4")
    assert rex.match("a8_s42") and rex.match("b4_s47")
    assert not rex.match("gated_s42") and not rex.match("a16_s42")
