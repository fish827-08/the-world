"""D-25 并行跑批包装（experiments/batch_runner.py）的单测。

只测纯逻辑（并发决策 / 网格展开 / summary 合法性判定），**不真跑仿真**——
真跑由端到端手测覆盖（3 run × 1500 tick，见 D-25 提交说明）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from batch_runner import (  # noqa: E402
    Run, cpu_count, expand, memory_gb, pick_concurrency, summary_ok,
)


# ---------------- 内存与并发（R48：上限按内存定，须留 ≥1 GB） ----------------

def test_memory_gb_returns_sane_values():
    avail, total, load = memory_gb()
    assert total > 0, "总内存应可读"
    assert 0 <= avail <= total + 0.01
    assert 0 <= load <= 100


def test_pick_concurrency_respects_cap_and_cpu():
    # 极小 per_run ⇒ 受 cpu-1 与 cap 限制
    n = pick_concurrency(per_run_gb=0.001, reserve_gb=1.0, cap=3)
    assert n <= 3
    assert n <= max(1, cpu_count() - 1)
    assert n >= 1, "至少 1（不能算出 0 或负）"


def test_pick_concurrency_is_at_least_one_when_memory_tight():
    # 内存极紧（per_run 巨大）⇒ 不能返回 0/负
    assert pick_concurrency(per_run_gb=999.0, reserve_gb=1.0, cap=None) >= 1


# ---------------- 网格展开 ----------------

def test_expand_cartesian_product_and_template():
    runs = expand(
        grid=["codebook=0,1", "seed=42,43"],
        fixed=["mode=on", "ticks=60000", "fresh"],
        template="_rerun_logs/a4_fix/asym_on_cb{codebook}_s{seed}.csv",
        script="experiments/a4_verify_capacity.py",
        workdir=ROOT,
    )
    assert len(runs) == 4
    names = {r.name for r in runs}
    assert names == {"asym_on_cb0_s42", "asym_on_cb0_s43",
                     "asym_on_cb1_s42", "asym_on_cb1_s43"}


def test_expand_passes_grid_params_and_flag_style_fixed():
    runs = expand(
        grid=["seed=7"],
        fixed=["mode=on", "fresh"],           # fresh 无值 ⇒ 只传开关
        template="_t/run_{seed}.csv",
        script="s.py",
        workdir=ROOT,
    )
    cmd = runs[0].cmd
    assert "--out" in cmd and "_t/run_7.csv" in cmd
    assert cmd[cmd.index("--seed") + 1] == "7"
    assert "--mode" in cmd and cmd[cmd.index("--mode") + 1] == "on"
    assert "--fresh" in cmd
    # fresh 后不能再跟参数值（它是布尔开关）
    assert cmd[-1] == "--fresh"


def test_expand_out_and_summary_paths_derive_from_template():
    runs = expand(["seed=1"], [], "_t/run_{seed}.csv", "s.py", workdir=ROOT)
    r: Run = runs[0]
    assert r.out.name == "run_1.csv"
    assert r.summary.name == "run_1.summary.json"


# ---------------- summary 合法性判定 ----------------

def _write(path: Path, obj: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return path


def test_summary_ok_short_run_that_reached_target(tmp_path):
    """跑满目标 tick 的**短跑**必须判 ok（否则小规格批会被无限重试）。"""
    p = _write(tmp_path / "a.summary.json", {
        "switches": {"ticks_target": 1500, "seed": 1},
        "result": {"final_tick": 1500, "final_N": 103, "eco_gate_pass": False},
    })
    assert summary_ok(p) is True


def test_summary_ok_rejects_early_stop_before_target(tmp_path):
    """没跑满目标就结束（灭绝）⇒ 不合法。"""
    p = _write(tmp_path / "b.summary.json", {
        "switches": {"ticks_target": 60000, "seed": 2},
        "result": {"final_tick": 12345, "final_N": 0, "eco_gate_pass": False},
    })
    assert summary_ok(p) is False


def test_summary_ok_rejects_missing_or_broken(tmp_path):
    assert summary_ok(tmp_path / "nope.summary.json") is False       # 不存在
    bad = tmp_path / "bad.summary.json"
    bad.write_text("{not json", encoding="utf-8")                    # 坏 JSON
    assert summary_ok(bad) is False
    noN = _write(tmp_path / "noN.summary.json", {
        "switches": {"ticks_target": 100},
        "result": {"final_tick": 100},                               # 缺 final_N
    })
    assert summary_ok(noN) is False


def test_summary_ok_60k_run_still_ok(tmp_path):
    """60k 长跑跑满 ⇒ ok（回归保护：不能因改判据而放过/误杀 60k）。"""
    p = _write(tmp_path / "c.summary.json", {
        "switches": {"ticks_target": 60000, "seed": 3},
        "result": {"final_tick": 60000, "final_N": 4999, "eco_gate_pass": True},
    })
    assert summary_ok(p) is True
