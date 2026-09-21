"""R107 判据（校准达成口径）的回归测试 —— `experiments/r97_calibration_judge.py` v4。

覆盖：
- R107 合取判定的三处**必备守卫**（缺配对/空域/反向 都不得判通过）；
- **F-R22**（`rand_` 前缀臂必须被自动识别，原 glob `m*_s*` 漏统）；
- 划法与域标注的一致性（A/B 两种独立划法）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import r97_calibration_judge as j  # noqa: E402


def _row(seed: int, m: float, ratio, N, pred, rand: bool = False) -> dict:
    name = f"{'rand_' if rand else ''}m{m}_s{seed}"
    return {"name": name, "m": m, "seed": seed, "is_rand": rand, "ratio": ratio,
            "N": N, "pred_frac": pred, "rho": 0.0, "resp_a": None, "resp_b": None,
            "resp_tri": None, "rs": {}, "ledger": {}, "flag": True, "sw_m": m,
            "signal_mode": "random" if rand else "state"}


def _good_batch() -> dict[str, dict]:
    """构造一批**符合 R107 判据**的数据：m1.3 配对全正且域内全达（两划法一致）。"""
    rows: dict[str, dict] = {}
    # 基线 m1.0：域内 seed 43/45（N≥3000 且 pred<0.9）ratio 0.80/0.88
    rows["m1.0_s42"] = _row(42, 1.0, 0.50, 685, 0.92)
    rows["m1.0_s43"] = _row(43, 1.0, 0.80, 3240, 0.30)
    rows["m1.0_s45"] = _row(45, 1.0, 0.88, 3240, 0.41)
    # 处理臂 m1.3：域内全落 [1.2,1.5]，且逐 seed 相对基线为正
    rows["m1.3_s42"] = _row(42, 1.3, 1.23, 3240, 0.95)     # 划法 B 域内（N≥3000）
    rows["m1.3_s43"] = _row(43, 1.3, 1.21, 3239, 0.29)
    rows["m1.3_s45"] = _row(45, 1.3, 1.26, 3239, 0.75)
    return rows


# ------------------------------------------------------------ F-R22：rand_ 前缀

def test_f_r22_load_finds_rand_prefixed_arms(tmp_path):
    """🔴 F-R22 回归：`rand_m1.3_s42` 必须被 `load()` 识别。

    原实现 glob `m*_s*.summary.json` ⇒ `rand_` 前缀文件**被静默漏统**（云端只得手工补算）。
    这里同时放置 `m*` 与 `rand_m*` 文件，断言两类都被载入且 `is_rand` 标注正确。
    """
    for name, is_rand in (("m1.3_s42", False), ("rand_m1.3_s42", True)):
        (tmp_path / f"{name}.summary.json").write_text(json.dumps({
            "switches": {"is_calibration_arm": True, "oracle_gain_multiplier": 1.3,
                         "signal_mode": "random" if is_rand else "state"},
            "result": {"final_N": 3000, "final_pred_frac": 0.5,
                       "oracle": {"oracle_return_ratio": 1.25,
                                  "receiver_side": {}, "ledger": {}},
                       "selection_gradient": {"non_sat": {"spearman_rho": 0.01}},
                       "signal_response": {}},
        }), encoding="utf-8")
    rows = j.load(tmp_path)
    assert set(rows) == {"m1.3_s42", "rand_m1.3_s42"}
    assert rows["m1.3_s42"]["is_rand"] is False
    assert rows["rand_m1.3_s42"]["is_rand"] is True
    assert rows["rand_m1.3_s42"]["signal_mode"] == "random"


def test_load_ignores_unrelated_summary_files(tmp_path):
    """非本批命名的 summary（如 `oracle_s42`）不得被误纳。"""
    (tmp_path / "oracle_s42.summary.json").write_text('{"switches": {}, "result": {}}',
                                                      encoding="utf-8")
    assert j.load(tmp_path) == {}


# ------------------------------------------------------------ 域与划法

def test_in_domain_two_methods_are_independent():
    """划法 A（pred<0.9）与 B（N≥3000）**相互独立**：本例 A 真 B 假 ⇒ 域标 `域A`。"""
    r = _row(42, 1.3, 1.23, 500, 0.5)          # N 小 ⇒ B 否；pred 低 ⇒ A 是
    assert j.in_domain(r, "A") is True and j.in_domain(r, "B") is False
    assert j.domain_tag(r) == "域A"
    r2 = _row(42, 1.3, 0.6, 3240, 0.95)        # N 大、pred 高 ⇒ 只属 B
    assert j.domain_tag(r2) == "域B"
    r3 = _row(42, 1.3, 0.6, 1500, 0.95)        # 两者皆否、N 在过渡带
    assert j.domain_tag(r3) == "过渡带"
    r4 = _row(42, 1.3, 0.6, 300, 0.95)
    assert j.domain_tag(r4) == "域外"


# ------------------------------------------------------------ R107 合取判定的三处守卫

def test_paired_all_positive_passes_on_matching_seeds():
    p = j.paired_all_positive(_good_batch(), 1.3)
    assert p["all_positive"] is True
    assert p["n_pairs"] == 3 and p["n_missing"] == 0
    assert min(p["deltas"]) > 0


def test_reversed_pair_must_fail():
    """🔴 反向条件公示：任一 seed 的 Δ ≤ 0 ⇒ **不得**判「校准达成」。"""
    rows = _good_batch()
    rows["m1.3_s43"]["ratio"] = 0.70          # Δ = 0.70 − 0.80 < 0（反向）
    p = j.paired_all_positive(rows, 1.3)
    assert p["all_positive"] is False
    assert p["reversed_seeds"] == [43]
    assert j.calibration_verdict(rows, 1.3)["pass"] is False


def test_missing_baseline_pair_must_fail_not_pass():
    """🔴 **缺配对 ≠ 通过**：基线缺失时 `paired_all_positive` 必须为假。"""
    rows = _good_batch()
    del rows["m1.0_s43"]                       # 去掉一个基线
    p = j.paired_all_positive(rows, 1.3)
    assert p["n_missing"] == 1 and p["all_positive"] is False
    assert j.calibration_verdict(rows, 1.3)["pass"] is False


def test_empty_domain_must_not_pass():
    """🔴 **空域 ⇒ 不得判通过**（`all([])` 为真的经典陷阱）。"""
    rows = _good_batch()
    for r in rows.values():
        r["N"] = 100                            # 全部落域外
        r["pred_frac"] = 0.99
    c = j.domain_coverage(rows, 1.3, "B")
    assert c["n"] == 0 and c["all_inside"] is False
    assert j.calibration_verdict(rows, 1.3)["pass"] is False


def test_domain_value_below_floor_fails_even_if_paired_positive():
    """配对全正但域内值 <1.2 ⇒ 仍不通过（R107 是**合取**）。"""
    rows = _good_batch()
    rows["m1.3_s43"]["ratio"] = 1.05            # >基线 0.80（配对为正）但 <1.2
    v = j.calibration_verdict(rows, 1.3)
    assert v["pair"]["all_positive"] is True
    assert v["coverage"]["A"]["all_inside"] is False
    assert v["pass"] is False
    assert "域内未全达" in v["reason"]


def test_value_above_ceiling_also_fails():
    """域内值 >1.5（超出增益档区间）同样判未达（上界也是判据的一部分）。"""
    rows = _good_batch()
    rows["m1.3_s45"]["ratio"] = 1.62
    c = j.domain_coverage(rows, 1.3, "A")
    assert c["seeds_above"] == [45] and c["all_inside"] is False


# ------------------------------------------------------------ 判定与锁 m

def test_calibration_verdict_passes_and_reports_consistency():
    v = j.calibration_verdict(_good_batch(), 1.3)
    assert v["pass"] is True
    assert v["methods_consistent"] is True
    assert v["coverage"]["A"]["n"] == 2 and v["coverage"]["B"]["n"] == 3
    assert "配对全正" in v["reason"]


def test_pick_locked_m_takes_minimal_satisfier():
    """R106/R108「**最小满足者**」：m1.3 与 m1.5 都满足 ⇒ 锁 1.3（不是最大）。"""
    v13 = {"m": 1.3, "pass": True}
    v15 = {"m": 1.5, "pass": True}
    v = j.pick_locked_m([v15, v13])
    assert v["locked_m"] == 1.3
    assert v["candidates"] == [1.3, 1.5]
    assert j.pick_locked_m([{"m": 1.5, "pass": False}])["locked_m"] is None


# ------------------------------------------------------------ 端到端（真实产物）

def test_report_runs_end_to_end_on_synthetic_batch(tmp_path):
    """整报告在合成批上可跑通，且**判定一节**与结构化结果一致（防"报告与 JSON 两张皮"）。"""
    rows = _good_batch()
    # 追加 rand 臂（**新键**，不得覆盖 state 臂组）
    rows.update({f"rand_{k}": dict(v, is_rand=True, signal_mode="random")
                 for k, v in _good_batch().items()})
    text, res = j.report(rows)
    assert "R107" in text and "⑧ 边界声明" in text
    assert res["locked_m"] == 1.3
    assert any(v["pass"] for v in res["verdicts"])
    assert res["n_rand_runs"] == len(_good_batch())
    # 边界声明四句必须在（R108 §二/§三 的措辞纪律）
    assert len(res["boundaries"]) == 4
    assert any("不含 L1" in b for b in res["boundaries"])


# ------------------------------------------------------------ 内评 21:38 §三 三条加固

def test_hardening2_threshold_is_ratio_of_max_count():
    """加固 2：域 B 阈值 = `0.9 × max_count`（**写死比值**，不许手填绝对值）。"""
    assert j.n_threshold({"max_count": 3240}) == pytest.approx(2916.0)
    assert j.n_threshold({"max_count": 5000}) == pytest.approx(4500.0)
    # max_count 缺失（旧批）⇒ 显式回退常量（不得静默变 0）
    assert j.n_threshold({}) == pytest.approx(float(j.N_DOMAIN_FALLBACK))


def test_hardening2_domain_b_boundary_is_derived_not_hardcoded():
    """阈值边界必须**由 max_count 推导**：N=2915 不入域、N=2916 入域（max_count=3240）。"""
    below = dict(_row(42, 1.3, 1.25, 2915, 0.5), max_count=3240)
    at = dict(_row(42, 1.3, 1.25, 2916, 0.5), max_count=3240)
    assert j.in_domain(below, "B") is False
    assert j.in_domain(at, "B") is True
    # 同一条数据在 max_count=5000 的批里则不入域 ⇒ 证明阈值确实随批变化（非硬编码）
    assert j.in_domain(at, "B") is True and j.in_domain(
        dict(at, max_count=5000), "B") is False


def test_sign_test_pvalue_matches_exact_binomial():
    """内评 21:38 §二 补的量化：`P(X≥6|n=6)=1/64=0.015625`（精确二项，非近似）。"""
    assert j.sign_test_pvalue(6, 6) == pytest.approx(0.015625)
    assert j.sign_test_pvalue(5, 6) == pytest.approx(7 / 64)          # 0.109375
    assert j.sign_test_pvalue(0, 6) == pytest.approx(1.0)
    # 单调：要求越多全正 ⇒ p 越小
    assert j.sign_test_pvalue(6, 8) > j.sign_test_pvalue(8, 8)


def test_hardening1_and_3_are_present_in_report_and_json():
    """加固 1（口径版本 + 预注册声明）与加固 3（表述额度）必须**可机器读取**。"""
    text, res = j.report(_good_batch())
    assert res["criterion_version"].startswith("R107-v1")
    assert "唯一" in res["preregistered_as"]
    assert res["allowed_statement"] == j.ALLOWED_STATEMENT
    assert len(res["forbidden_statements"]) == 4
    # 报告正文必须含三条加固的落点
    for needle in ("口径版本", "表述额度", "口径冻结", "阈值写死", "为判定前置"):
        assert needle in text, f"报告缺「{needle}」（加固未落码）"
    # 禁用清单必须含「有信息者优于随机者」（加固 3 的核心禁令）
    assert any("有信息者优于随机者" in f for f in res["forbidden_statements"])


def test_report_does_not_name_removed_constant():
    """回归：`N_TRANSITION`（旧常量）已被比值阈值取代 ⇒ 报告生成不得再引用它。"""
    assert not hasattr(j, "N_TRANSITION")
    assert hasattr(j, "N_TRANSITION_LO") and hasattr(j, "N_DOMAIN_RATIO")
    j.report(_good_batch())          # 不得抛 NameError


# ------------------------------------------------------------ 内评 21:38 §三 三条加固

def test_hardening2_threshold_is_ratio_of_max_count():
    """加固 2：域 B 阈值 = `0.9 × max_count`（**写死比值**，不许手填绝对值）。"""
    assert j.n_threshold({"max_count": 3240}) == pytest.approx(2916.0)
    assert j.n_threshold({"max_count": 5000}) == pytest.approx(4500.0)
    # max_count 缺失（旧批）⇒ 显式回退常量（不得静默变 0）
    assert j.n_threshold({}) == pytest.approx(float(j.N_DOMAIN_FALLBACK))


def test_hardening2_domain_b_boundary_is_derived_not_hardcoded():
    """阈值边界必须**由 max_count 推导**：N=2915 不入域、N=2916 入域（max_count=3240）。"""
    below = dict(_row(42, 1.3, 1.25, 2915, 0.5), max_count=3240)
    at = dict(_row(42, 1.3, 1.25, 2916, 0.5), max_count=3240)
    assert j.in_domain(below, "B") is False
    assert j.in_domain(at, "B") is True
    # 同一条数据在 max_count=5000 的批里则不入域 ⇒ 证明阈值确实随批变化（非硬编码）
    assert j.in_domain(at, "B") is True and j.in_domain(
        dict(at, max_count=5000), "B") is False


def test_sign_test_pvalue_matches_exact_binomial():
    """内评 21:38 §二 补的量化：`P(X≥6|n=6)=1/64=0.015625`（精确二项，非近似）。"""
    assert j.sign_test_pvalue(6, 6) == pytest.approx(0.015625)
    assert j.sign_test_pvalue(5, 6) == pytest.approx(7 / 64)          # 0.109375
    assert j.sign_test_pvalue(0, 6) == pytest.approx(1.0)
    # 单调：要求越多全正 ⇒ p 越小
    assert j.sign_test_pvalue(6, 8) > j.sign_test_pvalue(8, 8)


def test_hardening1_and_3_are_present_in_report_and_json():
    """加固 1（口径版本 + 预注册声明）与加固 3（表述额度）必须**可机器读取**。"""
    text, res = j.report(_good_batch())
    assert res["criterion_version"].startswith("R107-v1")
    assert "唯一" in res["preregistered_as"]
    assert res["allowed_statement"] == j.ALLOWED_STATEMENT
    assert len(res["forbidden_statements"]) == 4
    # 报告正文必须含三条加固的落点
    for needle in ("口径版本", "表述额度", "口径冻结", "阈值写死", "为判定前置"):
        assert needle in text, f"报告缺「{needle}」（加固未落码）"
    # 禁用清单必须含「有信息者优于随机者」（加固 3 的核心禁令）
    assert any("有信息者优于随机者" in f for f in res["forbidden_statements"])


def test_report_does_not_name_removed_constant():
    """回归：`N_TRANSITION`（旧常量）已被比值阈值取代 ⇒ 报告生成不得再引用它。"""
    assert not hasattr(j, "N_TRANSITION")
    assert hasattr(j, "N_TRANSITION_LO") and hasattr(j, "N_DOMAIN_RATIO")
    j.report(_good_batch())          # 不得抛 NameError


# ------------------------------------------------------------ F-R23（2026-09-17 内评抽核）
def test_f_r23_sign_test_uses_actual_positive_count_not_n_pairs():
    """🔴 回归：`P(X≥k|n)` 的 `k` 必须是**实际为正的配对数**，不是 `n_pairs`。

    原缺陷（F-R23）：调用点写 `sign_test_pvalue(n_pairs, n_pairs)` ⇒ k 恒等 n
    ⇒ 检验退化为"全正假设"下的**常数**（n=6 恒 0.0156，与数据里几个为正**无关**）。
    `[实测]` 真实数据（C1a）是 5/6 为正 ⇒ 正确值 = P(X≥5|6) = 7/64 = **0.1094**（不显著）。
    """
    rows: dict[str, dict] = {}
    seeds = (42, 43, 44, 45, 46, 47)
    for s in seeds:
        rows[f"m1.0_s{s}"] = _row(s, 1.0, 0.50, 3240, 0.30)
        # 5 个为正、1 个（s46）为负
        rows[f"m1.3_s{s}"] = _row(s, 1.3, 1.25 if s != 46 else 0.40, 3240, 0.30)
    res = j.report(rows)[1]
    pv = res["sign_test"]["1.3"]
    assert pv == pytest.approx(0.109375), "F-R23：5/6 为正时应为 P(X>=5|6)=0.1094"
    assert pv != pytest.approx(0.015625), "仍在使用全正假设常数（F-R23 未修）"


def test_f_r23_sign_test_still_all_positive_when_six_of_six():
    """对照：真全正（6/6）时仍应得 0.0156 —— 修复不得把正确情形改坏。"""
    rows: dict[str, dict] = {}
    for s in (42, 43, 44, 45, 46, 47):
        rows[f"m1.0_s{s}"] = _row(s, 1.0, 0.50, 3240, 0.30)
        rows[f"m1.3_s{s}"] = _row(s, 1.3, 1.25, 3240, 0.30)
    assert j.report(rows)[1]["sign_test"]["1.3"] == pytest.approx(0.015625)
