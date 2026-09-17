"""C 步判读（`experiments/cstep_judge.py`）的回归测试 —— 用**合成批次**驱动。

覆盖：Q1（R107 合取 ∧ 适用域=生态门）、Q2（G-A 仪器门 + 前置）、Q3/Q4（科学侧）、
以及 **R109 §4.1 的机器强制**：科学集绝不含校准臂、仪器段强制打标。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import cstep_judge as cj  # noqa: E402

SEEDS = (42, 43, 44, 45, 46, 47)


def _row(name: str, *, ratio=None, N=3240, pred=0.4, eco=True, rho=0.0,
         resp_tri=0.02, is_cal=False, max_count=3240, rand=False) -> dict:
    return {"name": name, "kind": ("oracle" if name.startswith(("oracle_", "rand_oracle"))
                                  else ("rand" if rand else name.split("_")[0])),
            "m": (float(name.split("_m")[1].split("_")[0])
                  if "_m" in name else None),
            "is_rand": rand, "seed": int(name.rsplit("_s", 1)[1]),
            "ratio": ratio, "N": N, "pred_frac": pred, "eco_gate": eco,
            "eco_applicable": True, "rho": rho, "resp_a": 0.9, "resp_b": 0.02,
            "resp_tri": resp_tri, "rs": {}, "ledger": {"identity_ok": True},
            "is_cal": is_cal, "sw_m": 1.3, "signal_mode": "random" if rand else "state",
            "max_count": max_count, "dir": "synth"}


def _synth(*, m13_ratio=1.25, m10_ratio=0.80, eco=lambda s: True,
           m13_rho=(0.10, 0.12, 0.11, 0.13, 0.09, 0.12),
           zero_rho=(-0.01, 0.02, -0.02, 0.01, 0.00, -0.01),
           main_rho=(0.05, 0.07, 0.04, 0.06, 0.05, 0.06)) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for i, s in enumerate(SEEDS):
        rows[f"oracle_m1.3_s{s}"] = _row(f"oracle_m1.3_s{s}", ratio=m13_ratio,
                                         eco=eco(s), rho=m13_rho[i], is_cal=True)
        rows[f"oracle_m1.0_s{s}"] = _row(f"oracle_m1.0_s{s}", ratio=m10_ratio,
                                         eco=eco(s), rho=0.01, is_cal=True)
        rows[f"zero_s{s}"] = _row(f"zero_s{s}", rho=zero_rho[i])
        rows[f"main_s{s}"] = _row(f"main_s{s}", rho=main_rho[i])
    return rows


# ------------------------------------------------------------ 命名解析与分集

def test_parse_name_variants():
    assert cj.parse_name("oracle_m1.3_s42") == {"name": "oracle_m1.3_s42", "kind": "oracle",
                                                "m": 1.3, "is_rand": False, "seed": 42}
    assert cj.parse_name("rand_m1.3_s47")["is_rand"] is True
    assert cj.parse_name("zero_s43")["kind"] == "zero"
    assert cj.parse_name("main_s44")["m"] is None
    assert cj.parse_name("oracle_s42") is None      # 无 m 的裸 oracle 名不属本批


def test_split_sets_is_exclusive_and_science_has_no_calibration_arm():
    """🔴 R109 §4.1：仪器集/科学集**互斥**；科学集**绝不含**校准臂。"""
    inst, sci = cj.split_sets(_synth())
    assert {v["kind"] for v in inst.values()} == {"oracle"}
    assert {v["kind"] for v in sci.values()} == {"zero", "main"}
    assert all(not v["is_cal"] for v in sci.values())


# ------------------------------------------------------------ Q1

def test_q1_passes_on_clean_synthetic_batch():
    v = cj.q1_verdict(_synth(), 1.3, 1.0)
    assert v["pair_all_positive"] is True
    assert v["sign_test_p"] == pytest.approx(0.015625)
    assert v["coverage_in_domain"]["A"]["n"] == 6
    assert v["coverage_in_domain"]["B"]["n"] == 6
    assert v["pass"] is True


def test_q1_fails_when_domain_value_below_floor():
    """域内 ratio < 1.2 ⇒ 不通过（R107 合取）。"""
    assert cj.q1_verdict(_synth(m13_ratio=1.05), 1.3, 1.0)["pass"] is False


def test_q1_fails_when_ratio_above_ceiling():
    assert cj.q1_verdict(_synth(m13_ratio=1.62), 1.3, 1.0)["pass"] is False


def test_q1_domain_is_eco_gate_only_that_seed_excluded():
    """**适用域 = 生态门**：eco 不通过的 seed 被排除在域内覆盖之外（但配对仍按 seed 对）。"""
    v = cj.q1_verdict(_synth(eco=lambda s: s != 44), 1.3, 1.0)
    assert v["coverage_in_domain"]["A"]["seeds"] == [42, 43, 45, 46, 47]
    assert v["pair_all_positive"] is True          # 配对不看生态门（条件 6 只看 seed 配对）
    assert v["pass"] is True


def test_q1_fails_when_paired_delta_reversed():
    rows = _synth()
    rows["oracle_m1.0_s43"]["ratio"] = 1.40        # Δ = 1.25 − 1.40 < 0
    v = cj.q1_verdict(rows, 1.3, 1.0)
    assert v["pair_all_positive"] is False
    assert v["pass"] is False


# ------------------------------------------------------------ Q2（仪器门）

def test_q2_detects_positive_and_is_marked_as_instrument():
    q2 = cj.q2_ga(_synth(), 1.3)
    assert q2["pass"] is True
    assert "仪器门" in q2["marked"] and "不构成科学阳性" in q2["marked"]
    assert q2["n_pairs"] == 6


def test_q2_not_detected_when_oracle_equals_zero():
    """阳性对照若不优于零模型 ⇒ 未检出（这才是 G-A 该失败的形态）。"""
    rows = _synth(m13_rho=(0.0,) * 6, zero_rho=(0.0,) * 6)
    assert cj.q2_ga(rows, 1.3)["pass"] is False


# ------------------------------------------------------------ Q3/Q4（科学侧）

def test_q3_uses_only_science_arms_and_reports_cluster():
    q3 = cj.q3_science(_synth())
    assert q3["main_cluster"]["n_seeds"] == 6
    assert q3["pass_rho_positive"] is True
    assert q3["n_eco_pass_main"] == 6


# ------------------------------------------------------------ 报告端到端

def test_report_marks_q2_skipped_when_q1_fails():
    """Q1 未过 ⇒ 按 R109 **不判 Q2**（判定点不得落在中性边界，教训 18）。"""
    text, res = cj.report(_synth(m13_ratio=0.9))
    assert res["q1_pass"] is False
    assert res["q2"]["skipped"] is True
    assert "Q1 未过" in text


def test_report_end_to_end_and_boundaries():
    text, res = cj.report(_synth())
    assert res["q1_pass"] is True
    assert res["n_instrument"] == 12 and res["n_science"] == 12
    assert len(res["boundaries"]) == 4
    for needle in ("Q1", "Q2", "Q3/Q4", "边界声明", "仪器门"):
        assert needle in text
    # 校准臂只在仪器集：报告须写明分组依据
    assert "is_calibration_arm" in text or "校准臂" in text


def test_load_reads_only_recognized_names(tmp_path):
    (tmp_path / "oracle_m1.3_s42.summary.json").write_text(json.dumps({
        "switches": {"is_calibration_arm": True, "oracle_gain_multiplier": 1.3,
                     "max_count": 3240},
        "result": {"final_N": 3240, "final_pred_frac": 0.4, "eco_gate_pass": True,
                   "eco_gate_applicable": True, "oracle": {"oracle_return_ratio": 1.25},
                   "selection_gradient": {"non_sat": {"spearman_rho": 0.1}},
                   "signal_response": {}},
    }), encoding="utf-8")
    (tmp_path / "not_a_run.summary.json").write_text('{"switches": {}, "result": {}}',
                                                     encoding="utf-8")
    rows = cj.load([tmp_path])
    assert set(rows) == {"oracle_m1.3_s42"}
    assert rows["oracle_m1.3_s42"]["is_cal"] is True
    assert rows["oracle_m1.3_s42"]["ratio"] == 1.25


# ------------------------------------------------------------ F-R23（2026-09-17）
def test_f_r23_q1_sign_test_k_is_positive_count():
    """🔴 F-R23 回归（C 步判读侧）：k = 正向配对数；5 正 1 负 ⇒ p = 0.1094。"""
    rows = _synth(m13_ratio=1.25, m10_ratio=0.80)
    rows["oracle_m1.3_s46"] = dict(rows["oracle_m1.3_s46"], ratio=0.70)   # 造一个反向
    v = cj.q1_verdict(rows, 1.3, 1.0)
    assert v["n_positive"] == 5
    assert v["sign_test_p"] == pytest.approx(0.109375)
    assert v["pair_all_positive"] is False
    assert "正向配对数" in v["sign_test_k_definition"]


def test_f_r23_q1_sign_test_all_positive_unchanged():
    """对照：真全正（6/6）仍为 1/64 = 0.0156。"""
    v = cj.q1_verdict(_synth(m13_ratio=1.25, m10_ratio=0.80), 1.3, 1.0)
    assert v["n_positive"] == 6 and v["sign_test_p"] == pytest.approx(0.015625)
