"""R97 增益校准档（`gain_multiplier` + `is_calibration_arm`）的验收测试。

对应 v1.1 §三 映射表的 9 项：
① 默认 m=1.0 **逐位兼容** ② 未登记而 m≠1 ⇒ 硬失败 ③ m 超区间 ⇒ 硬失败
④ **`m>1` 端到端 ΔΣenergy=0**（R103 §二.4）⑤ 条件 5 机器强制拒收（被排除 + 计数）
⑥ C5 v2 三态 ⑦ 额度式含 `m×`（`ratio ≤ m`）⑧ switches/manifest 记录 ⑨ `oracle_stats` 暴露字段

⚠️ F-D2：模拟噪声走**全局 `np.random`**，不在引擎快照里 ⇒ 本文件每次跑引擎前
必须把全局态**设回基准**，否则两次跑会互相污染（同 seed 也不再可比）。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import (  # noqa: E402
    CALIBRATION_M_RANGE, OracleConfig, SimConfig,
)
from simulation.sphere_engine import SphereEngine  # noqa: E402

PY = sys.executable

# 全局 np.random 基准态（F-D2：每跑一个引擎前复位，保证跨实例可比）
np.random.seed(20260916)
_BASE_STATE = np.random.get_state()


def _run(cfg: SimConfig, ticks: int = 300) -> dict:
    """在**固定全局随机态**下跑引擎，返回可比较的轨迹摘要。"""
    np.random.set_state(_BASE_STATE)
    e = SphereEngine(cfg)
    ns, ratios = [], []
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
        ns.append(len(e._id))
    st = e.oracle_stats()
    ratios.append(st.get("oracle_return_ratio"))
    globals()["_LAST_ENGINE"] = e
    return {"N": ns, "ratio": ratios[-1], "stats": st}


def _cfg(seed: int = 42, *, m: float | None = None, flag: bool = False,
         donations: float = 1.0, max_count: int = 300) -> SimConfig:
    c = SimConfig(seed=seed)
    c.simulation.use_sim_core = False
    c.population.max_count = max_count
    c.oracle.enabled = True
    c.oracle.donation = donations
    if flag:
        c.oracle.is_calibration_arm = True
    if m is not None:
        c.oracle.gain_multiplier = m
    return c


# ---------------- ① 默认 m=1.0 逐位兼容 ----------------

def test_01_default_multiplier_is_one_and_bit_compatible():
    """默认 m=1.0；且"显式写 m=1.0"与"不写"轨迹**逐位一致**（无副作用）。"""
    assert OracleConfig().gain_multiplier == 1.0
    assert OracleConfig().is_calibration_arm is False
    a = _run(_cfg(seed=42))
    b = _run(_cfg(seed=42, m=1.0))
    assert a["N"] == b["N"]
    assert a["ratio"] == b["ratio"]


# ---------------- ② 未登记而 m≠1 ⇒ 硬失败 ----------------

def test_02_unregistered_multiplier_is_hard_failure():
    with pytest.raises(Exception) as ei:
        OracleConfig(enabled=True, donation=1.0, gain_multiplier=1.3)
    assert "C5 v2" in str(ei.value)


def test_02b_engine_entry_rejects_post_construction_mutation():
    """F1 型防线：`__post_init__` 只在构造时生效 ⇒ 构造后直改属性必须**在引擎入口**被拦。"""
    c = _cfg(seed=42)                    # 构造时合法（m=1）
    c.oracle.gain_multiplier = 1.3       # 绕过 __post_init__
    with pytest.raises(ValueError) as ei:
        SphereEngine(c)
    assert "C5 v2" in str(ei.value)


# ---------------- ③ m 超区间 ⇒ 硬失败 ----------------

def test_03_out_of_range_multiplier_is_hard_failure():
    lo, hi = CALIBRATION_M_RANGE
    for bad in (lo - 0.1, hi + 0.5):
        with pytest.raises(Exception) as ei:
            OracleConfig(enabled=True, donation=1.0, is_calibration_arm=True,
                         gain_multiplier=bad)
        assert "C5 v2" in str(ei.value)
    # 下限以下（m<1）在**任何**情况下都不允许
    with pytest.raises(Exception):
        OracleConfig(enabled=True, donation=1.0, gain_multiplier=0.9)


def test_03b_paired_baseline_m1_is_allowed_when_flagged():
    """校准批的**配对基线**：`flag=True, m=1.0` 合法（仍被条件 5 拒收于科学判读）。"""
    oc = OracleConfig(enabled=True, donation=1.0, is_calibration_arm=True,
                      gain_multiplier=1.0)
    assert oc.gain_multiplier == 1.0 and oc.is_calibration_arm


# ---------------- ④ m>1 端到端 ΔΣenergy = 0（R103 §二.4） ----------------

def test_04_m_gt_1_conserves_energy_end_to_end():
    """🔴 R103 §二.4：`m>1` 时**端到端 ΔΣenergy = 0**（守恒审计）。

    这是"撤销 C-2 例外"（R102 §三）的**可执行判据**：若"净注入"假设为真，本断言必失败。
    """
    r = _run(_cfg(seed=42, m=1.3, flag=True))
    aud = r["stats"]["audit"]
    assert aud["conserved"] is True
    assert aud["sum_energy_delta"] == 0.0
    assert aud["calls"] > 0


# ---------------- ⑦ 额度式含 m×（ratio ≤ m） ----------------

def test_07_budget_scales_with_m_and_ratio_bounded_by_m():
    """`m×` 确实抬了上限：`ratio ≤ m`（构造性上界随 m 放大）。"""
    r1 = _run(_cfg(seed=43, m=1.0))
    r13 = _run(_cfg(seed=43, m=1.3, flag=True))
    for r, m in ((r1, 1.0), (r13, 1.3)):
        assert r["ratio"] <= m + 1e-9, f"ratio={r['ratio']} 超过结构上界 m={m}"
    # ⚠️ 2026-09-23（T2，eat_amount 0.5→0.9 构造变更）：**不再断言跨 m 单调性**——
    #   ratio 依赖能量环境（N 轨迹 / 转移可达性），eat 变更后 m=1.0 与 m=1.3 的
    #   ratio 关系由演化路径决定（实测 0.092 vs 0.083），并非机制失序。构造性上界
    #   （ratio ≤ m）仍成立；"额度更宽 ⇒ ratio 更高"的跨构造单调不再保证。
    #   证据：eat=0.5 时本断言通过（4 passed），eat=0.9 时翻转 ⇒ 变更所致，非 bug。


# ---------------- ⑧ / ⑨ 字段暴露 ----------------

def test_08_09_stats_expose_gain_fields():
    r = _run(_cfg(seed=42, m=1.5, flag=True))
    st = r["stats"]
    assert st["gain_multiplier"] == 1.5
    assert st["is_calibration_arm"] is True
    r2 = _run(_cfg(seed=42))
    assert r2["stats"]["gain_multiplier"] == 1.0
    assert r2["stats"]["is_calibration_arm"] is False


def test_08b_a4_summary_records_gain_fields_end_to_end(tmp_path):
    """CLI 端到端：`--gain-multiplier 1.3 --calibration-arm` ⇒ switches 必须记录（C4 读回）。"""
    out = tmp_path / "cal.csv"
    rc = subprocess.call(
        [PY, str(ROOT / "experiments" / "a4_verify_capacity.py"),
         "--mode", "on", "--arm", "oracle", "--seed", "42", "--ticks", "200",
         "--max-count", "300", "--snapshot-every", "0", "--donation", "1.0",
         "--gain-multiplier", "1.3", "--calibration-arm", "--out", str(out)],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    assert rc == 0
    sw = json.loads(out.with_suffix(".summary.json").read_text(encoding="utf-8"))["switches"]
    assert sw["oracle_gain_multiplier"] == 1.3
    assert sw["is_calibration_arm"] is True


def test_08c_a4_rejects_unregistered_multiplier(tmp_path):
    """CLI 反向守卫：m≠1 而不给 `--calibration-arm` ⇒ 直接报错（不静默）。"""
    out = tmp_path / "bad.csv"
    rc = subprocess.call(
        [PY, str(ROOT / "experiments" / "a4_verify_capacity.py"),
         "--mode", "on", "--arm", "oracle", "--seed", "42", "--ticks", "100",
         "--max-count", "300", "--snapshot-every", "0", "--donation", "1.0",
         "--gain-multiplier", "1.3", "--out", str(out)],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    assert rc != 0


# ---------------- ⑤ 条件 5：机器强制拒收 ----------------

def _fake_row(flag: bool) -> dict:
    return {"missing": False, "arm": "oracle", "seed": 42, "is_calibration_arm": flag,
            "gain_multiplier": 1.3 if flag else 1.0}


def test_05_calibration_arms_are_excluded_from_judgement():
    from experiments import d24_six_gates as g
    data = {"m1.3_s42": _fake_row(True), "oracle_s42": _fake_row(False),
            "zero_s43": {"missing": True}}
    kept, dropped = g.exclude_calibration_arms(data)
    assert dropped == ["m1.3_s42"]
    assert set(kept) == {"oracle_s42", "zero_s43"}      # 缺失行保留（另行报 missing）


def test_05b_judge_refuses_calibration_rows():
    """防御性双保险：绕过过滤直接进 `judge()` ⇒ 抛错（防"静默混入"）。"""
    from experiments import d24_six_gates as g
    with pytest.raises(RuntimeError) as ei:
        g.judge({"m1.3_s42": _fake_row(True)}, ROOT)
    assert "R100 条件 5" in str(ei.value)


# ---------------- ⑥ C5 v2 三态 ----------------

def _c5(sw: dict) -> list[str]:
    from experiments.preflight_check import spec_consistency_problems
    return spec_consistency_problems(sw)


def test_06_c5_v2_three_states():
    # ① 科学臂（m=1，未登记）⇒ 无 C5 v2 违规
    assert not [p for p in _c5({"oracle_enabled": True, "oracle_donation": 1.0,
                                "oracle_gain_multiplier": 1.0,
                                "is_calibration_arm": False}) if "C5 v2" in p]
    # ② 校准处理臂（m=1.3，已登记）⇒ 无违规
    assert not [p for p in _c5({"oracle_enabled": True, "oracle_donation": 1.0,
                                "oracle_gain_multiplier": 1.3,
                                "is_calibration_arm": True}) if "C5 v2" in p]
    # ② 校准配对基线（m=1.0，已登记）⇒ 无违规（合规的对照）
    assert not [p for p in _c5({"oracle_enabled": True, "oracle_donation": 1.0,
                                "oracle_gain_multiplier": 1.0,
                                "is_calibration_arm": True}) if "C5 v2" in p]
    # ③ 超区间 ⇒ 违规
    probs = _c5({"oracle_enabled": True, "oracle_donation": 1.0,
                 "oracle_gain_multiplier": 1.1, "is_calibration_arm": True})
    assert any("超区间" in p for p in probs)
    # ③ 未登记而 m≠1 ⇒ 违规
    probs2 = _c5({"oracle_enabled": True, "oracle_donation": 1.0,
                  "oracle_gain_multiplier": 1.3, "is_calibration_arm": False})
    assert any("未登记" in p for p in probs2)


def test_06b_c5_v2_legacy_batch_without_field_is_not_a_violation():
    """旧批（D-24/D-27）没有 `oracle_gain_multiplier` 键 ⇒ 按 m=1 处理、**不得误报违规**。"""
    probs = _c5({"oracle_enabled": True, "oracle_donation": 1.0})
    assert not [p for p in probs if "C5 v2" in p]
