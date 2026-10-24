"""R110 §一（Q1 k-of-n 三态）与 §三（原子锁并发）的回归测试。

R110（2026-09-16 22:37 裁定，C1a 完成前锁定 ⇒ 属预注册）：
> **Q1 通过 ⟺** ① 域内可用 seed ≥ 5/6；∧ ② 至多 1 个例外（域内落带数 ≥ 域内数 − 1）。
> **域内 < 5 ⇒「样本不足、不判」**（既非通过也非失败 ⇒ 停下上板；与"值不达"**分开记录**）。
> 两划法在 60k 下**已退化为一** ⇒ 降**诊断列**，不作独立护栏。
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import batch_runner as br  # noqa: E402
from experiments import cstep_judge as cj  # noqa: E402
from tests._synth_batch import synth_batch as _synth  # noqa: E402
# ↑ W0-T（2026-10-18）：原 `from tests.test_cstep_judge import _synth`；
#   该文件是 legacy 且已归档到 tests/archive/ ⇒ 数据构造提到 tests/_synth_batch.py。


def _with_m13(ratio: float, *seeds: int) -> dict[str, dict]:
    rows = _synth()
    for s in seeds:
        rows[f"oracle_m1.3_s{s}"]["ratio"] = ratio
    return rows


# ------------------------------------------------------------ k-of-n 三态

def test_kofn_passes_with_exactly_one_exception():
    """5/6 落带 + **1 个例外** ⇒ 通过（这正是 R110 放宽掉的那类"单点噪声"）。"""
    v = cj.q1_verdict(_with_m13(1.05, 44), 1.3, 1.0)   # s44 = 1.05（<1.2）
    assert v["domain"]["n_seeds"] == 6
    assert v["domain"]["n_inside"] == 5
    assert v["domain"]["exceptions"] == [44]
    assert v["state"] == "pass" and v["pass"] is True


def test_kofn_fails_with_two_exceptions():
    """2 个例外 ⇒ 不通过（放宽到 1，不放到 2）。"""
    v = cj.q1_verdict(_with_m13(1.05, 44, 45), 1.3, 1.0)
    assert v["domain"]["exceptions"] == [44, 45]
    assert v["state"] == "fail" and v["pass"] is False
    assert "例外" in v["reason"]


def test_insufficient_is_neither_pass_nor_fail():
    """🔴 域内 < 5 ⇒ **「样本不足、不判」** —— 三态必须可分（R110 §一 明示）。"""
    v = cj.q1_verdict(_synth(eco=lambda s: s not in (44, 45)), 1.3, 1.0)
    assert v["domain"]["n_seeds"] == 4
    assert v["state"] == "insufficient"
    assert v["pass"] is False
    assert "样本不足" in v["reason"] and "不判" in v["reason"]
    # 与"值不达"的措辞必须不同（否则事后无法区分迁移失败 vs 域内种子太少）
    v_fail = cj.q1_verdict(_with_m13(1.05, 44, 45), 1.3, 1.0)
    assert v_fail["reason"] != v["reason"] and "样本不足" not in v_fail["reason"]


def test_domain_uses_seed_level_eco_gate():
    """域 = **seed 级生态门**（逐 run 判定，不看批次整体）。"""
    rows = _synth(eco=lambda s: s == 42)          # 仅 1 个 seed 过门
    v = cj.q1_verdict(rows, 1.3, 1.0)
    assert v["domain"]["seeds"] == [42]
    assert v["state"] == "insufficient"
    # 配对不受域影响：仍按 seed 配对（6 对，全正 ⇒ 配对侧成立）
    assert v["pair_all_positive"] is True and len(v["pairs"]) == 6


def test_two_methods_are_marked_diagnostic_only():
    """两划法在 60k 下**降诊断列**（不再作"两法一致"独立护栏）。"""
    v = cj.q1_verdict(_synth(), 1.3, 1.0)
    assert "诊断" in v["methods_role"]
    assert set(v["coverage_in_domain"]) == {"A", "B"}
    # 判定**不依赖**两法一致性：即便把 A 诊断列造出不一致，k-of-n 达标即通过
    rows = _synth()
    rows["oracle_m1.3_s42"]["pred_frac"] = 0.95        # 划法 A 少一个 seed（B 仍有）
    v2 = cj.q1_verdict(rows, 1.3, 1.0)
    assert v2["coverage_in_domain"]["A"]["n"] != v2["coverage_in_domain"]["B"]["n"]
    assert v2["state"] == "pass"


def test_report_shows_three_states_and_degeneracy():
    rows = _synth(eco=lambda s: s not in (44, 45))
    text, res = cj.report(rows)
    assert res["q1_states"] == ["insufficient"]
    assert res["q1_pass"] is False
    assert "样本不足" in text and "诊断列" in text and "退化" in text or "退化" in text
    # 口径版本必须带 k-of-n 与退化标注（R110 §一 要求）
    assert "k-of-n" in res["criterion_version"]
    assert "退化" in res["criterion_version"]
    assert res["criterion_version_8k"].startswith("R107-v1")   # 8k 口径保留、不回改


def test_q2_not_judged_when_q1_insufficient():
    """Q1 未过（含样本不足）⇒ 不判 Q2（R109 预授权：停 + 上板）。"""
    text, res = cj.report(_synth(eco=lambda s: s not in (44, 45)))
    assert res["q2"]["skipped"] is True


# ------------------------------------------------------------ 原子锁：同时启动

def test_concurrent_start_exactly_one_wins(monkeypatch, tmp_path):
    """🔴 R110 §三：**同一秒启动两个实例**时，必须**恰好一个**拿到锁。

    原实现"先查后写"（exists → 读 → 判活 → 才写）⇒ 两个实例可同时通过检查；
    改用 `os.open(..., O_CREAT|O_EXCL)` 原子抢占后，竞争由内核裁决。
    """
    monkeypatch.setattr(br, "LOCK_PATH", tmp_path / ".batch_runner.lock")
    barrier = threading.Barrier(2)
    results: list[tuple[str, str | None]] = []

    def worker(tag: str) -> None:
        barrier.wait()                       # 让两个线程尽量同时进入
        results.append((tag, br.acquire_lock(tag)))

    ts = [threading.Thread(target=worker, args=(t,)) for t in ("a", "b")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    winners = [r for r in results if r[1] is None]
    assert len(winners) == 1, f"应恰好一个实例拿到锁，实得 {results}"
    loser = next(r for r in results if r[1] is not None)
    assert "已有实例在跑" in loser[1]
    br.release_lock()


def test_lock_holder_recorded_is_single_process(monkeypatch, tmp_path):
    monkeypatch.setattr(br, "LOCK_PATH", tmp_path / ".batch_runner.lock")
    assert br.acquire_lock("only") is None
    import json
    info = json.loads(br.LOCK_PATH.read_text(encoding="utf-8"))
    assert info["preset"] == "only" and info["pid"] > 0
    br.release_lock()


def test_stale_lock_taken_over_atomically(monkeypatch, tmp_path):
    """陈旧锁（pid 已死）⇒ 原子接管（先 unlink 再 O_EXCL 建）。"""
    import json
    monkeypatch.setattr(br, "LOCK_PATH", tmp_path / ".batch_runner.lock")
    br.LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    br.LOCK_PATH.write_text(json.dumps({"pid": 999999, "preset": "dead"}),
                            encoding="utf-8")
    assert br.acquire_lock("fresh") is None
    assert json.loads(br.LOCK_PATH.read_text(encoding="utf-8"))["preset"] == "fresh"
    br.release_lock()


def test_lock_note_declares_threat_model():
    """锁注释须**声明威胁模型**（R110 §三：防同时启动；不防陈旧锁）—— 防被误读成万能锁。"""
    assert hasattr(br, "LOCK_NOTE")
    assert "同时启动" in br.LOCK_NOTE
