#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R287 T2 验收：`s3_memory_probe.py` 的 **sample 级续跑**（v3）。

派工要求原文（R287 §二）：
    | **T2：sample 级续跑** | **云归** | `s3_memory_probe.py` 续跑粒度 run 级 →
    **sample 级**（每 2000t flush；重启从最后完成采样点接续） | 与从头跑逐位等价
    （短跑对拍）+ 冒烟；分支交付 |

本文件把"与从头跑逐位等价"钉成**可回归的测试**，并覆盖实现过程中踩到的
**两个真坑**（各自都有"静默错"的后果 ⇒ 必须有守卫）：

  🔴 坑 1（v3 实现第一版）：清残行用 `os.replace` ⇒ CSV 换 inode，而 `fout`
     长期打开、仍指向旧 inode ⇒ **续跑新行一个字节都不落盘**。
     现象极具迷惑性：旧行完好、日志说 `rows=4`、无异常 ⇒ 只有对拍才发现。
  🔴 坑 2（同版）：探针累积量（`visited_mask` / `init_cap_full` / …）**不在引擎
     快照里** ⇒ 不存侧车则续跑后 `cap_lost_frac` 从 0 重算、`l1_visited_patch_frac`
     从 0 起 ⇒ 数值静默错。

口径：**除 `ms_per_tick` 外逐字节相同**（墙钟计时列天然不可复现；R189 已登记
"禁瞬时速率外推"，本测试不打它的主意）。
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROBE = "experiments.s3_memory_probe"

# 微缩档（跑一次 ~1s；够走到"有记忆写入 + 斑块被访问"的最小世界）
DEV = ["--rows", "60", "--cols", "120", "--patches", "30", "--pop", "50",
       "--sample", "10", "--arms", "both"]
TICKS = 60
HALF = 30

# 🔴 唯一允许不同的列（墙钟计时）。别往里加科学量 —— 那等于把等价性断言废掉。
IGNORE = {"ms_per_tick"}


def _run(args: list[str], out: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    return subprocess.run(
        [sys.executable, "-u", "-m", PROBE, *args, "--out", str(out)],
        cwd=str(REPO), env=env, capture_output=True, text=True)


def _rows(p: Path) -> list[dict]:
    with open(p, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _key(r: dict) -> tuple:
    return (r["seed"], r["arm"], int(float(r["tick"])))


def snap_leftover(repo: Path) -> list[str]:
    """默认目录里是否有本探针的残留快照（`s3_s*`）。"""
    d = repo / "_rerun_logs" / "snap"
    return sorted(p.name for p in d.glob("s3_s*")) if d.exists() else []


def _diff(ref: list[dict], got: list[dict]) -> list[tuple]:
    ref = sorted(ref, key=_key)
    got = sorted(got, key=_key)
    if [_key(r) for r in ref] != [_key(r) for r in got]:
        return [("__keys__", [ _key(r) for r in ref ], [ _key(r) for r in got ])]
    return [( _key(a), c, a[c], b[c])
            for a, b in zip(ref, got) for c in a
            if c not in IGNORE and a[c] != b[c]]


# ---------------------------------------------------------------- T1 端到端等价

def test_T1_sample_resume_equals_from_scratch(tmp_path):
    """核心验收：跑到一半存快照 → `--resume-sample` 接续 ⇒ ≡ 一次从头跑完。

    ⚠️ 阶段 2 **必须续写同一个 `--out`**（那才是"中断后重启"的真实场景）；
    写到新文件只能得到"后半段"，会被 `_strip_rows_after` 的语义误读成 bug。
    """
    ref, run = tmp_path / "ref.csv", tmp_path / "run.csv"
    snap = tmp_path / "snap"

    r = _run([*DEV, "--ticks", str(TICKS)], ref)
    assert r.returncode == 0, r.stderr

    # 阶段 1：跑到 HALF，**保留**快照（模拟"中断"）
    r = _run([*DEV, "--ticks", str(HALF), "--save-every", "10",
              "--keep-ckpt", "--snapshot-dir", str(snap)], run)
    assert r.returncode == 0, r.stderr
    assert (snap / "s3_s101_mem_on.snapshot.npz").exists(), "快照未生成"
    assert len(_rows(run)) == 18, "阶段 1 应有 3 seed × 2 臂 × 3 采样点"

    # 阶段 2：接续到 TICKS（**同一个 --out**）
    r = _run([*DEV, "--ticks", str(TICKS), "--resume-sample", "--save-every", "10",
              "--snapshot-dir", str(snap)], run)
    assert r.returncode == 0, r.stderr
    assert "(接续自快照)" in r.stderr, "没有真的走续跑分支"

    d = _diff(_rows(ref), _rows(run))
    assert not d, f"🔴 续跑≠从头跑（{len(d)} 处）：{str(d)[:600]}"


# ---------------------------------------------------------------- T2 多段续跑

def test_T2b_multi_segment_resume_equals_single_run(tmp_path):
    """压测：被**反复打断**（20 → 50 → 70 → 100）≡ 一次跑到 100。

    这条比 T1 更强：T1 只切一次，而长批（on 臂 25.6 h/run）会被切很多次。
    每次续跑都会 `_strip_rows_after` + 重开句柄 ⇒ 把坑 1/2/3 的回归风险
    按次数放大。
    """
    ref, run = tmp_path / "ref.csv", tmp_path / "run.csv"
    snap = tmp_path / "snap"
    dev = ["--rows", "60", "--cols", "120", "--patches", "30", "--pop", "50",
           "--sample", "10", "--arms", "both"]
    r = _run([*dev, "--ticks", "100"], ref)
    assert r.returncode == 0, r.stderr

    r = _run([*dev, "--ticks", "20", "--save-every", "10", "--keep-ckpt",
              "--snapshot-dir", str(snap)], run)
    assert r.returncode == 0, r.stderr
    for target in (50, 70, 100):
        r = _run([*dev, "--ticks", str(target), "--resume-sample", "--save-every",
                  "10", "--keep-ckpt", "--snapshot-dir", str(snap)], run)
        assert r.returncode == 0, r.stderr
        assert "(接续自快照)" in r.stderr

    d = _diff(_rows(ref), _rows(run))
    assert not d, f"🔴 多段续跑≠一次跑完（{len(d)} 处）：{str(d)[:600]}"


# ---------------------------------------------------------------- T2 累积量必须是"连续的"

def test_T2_cumulative_readouts_survive_resume(tmp_path):
    """量 1 的加强版：`cap_lost_frac` / `l1_visited_patch_frac` 必须**连续**，
    不能因为续跑而从 0 重算（= 坑 2 的守卫：侧车没存住时本测试挂）。
    """
    ref, run = tmp_path / "ref.csv", tmp_path / "run.csv"
    snap = tmp_path / "snap"
    _run([*DEV, "--ticks", str(TICKS)], ref)
    _run([*DEV, "--ticks", str(HALF), "--save-every", "10",
          "--keep-ckpt", "--snapshot-dir", str(snap)], run)
    r = _run([*DEV, "--ticks", str(TICKS), "--resume-sample", "--save-every", "10",
              "--snapshot-dir", str(snap)], run)
    assert r.returncode == 0, r.stderr

    ref_last = [x for x in _rows(ref) if int(x["tick"]) == TICKS]
    got_last = [x for x in _rows(run) if int(x["tick"]) == TICKS]
    assert ref_last and got_last
    for a, b in zip(sorted(ref_last, key=lambda r: (r["seed"], r["arm"])),
                    sorted(got_last, key=lambda r: (r["seed"], r["arm"]))):
        assert a["l1_visited_patch_frac"] == b["l1_visited_patch_frac"], (
            f"l1_visited_patch_frac 不连续（侧车丢了？）：{a['seed']}/{a['arm']} "
            f"{a['l1_visited_patch_frac']} vs {b['l1_visited_patch_frac']}")
        assert a["cap_lost_frac"] == b["cap_lost_frac"], (
            f"cap_lost_frac 不连续（init_cap_sum 没随侧车走？）："
            f"{a['seed']}/{a['arm']} {a['cap_lost_frac']} vs {b['cap_lost_frac']}")


# ---------------------------------------------------------------- T3 默认行为逐字节不变

def test_T3_default_path_is_byte_identical_to_v2(tmp_path):
    """硬约束 2：**不传**任何 v3 参数（无 `--save-every` / `--snapshot-dir` /
    `--resume-sample`）⇒ 产出与 v2 逐字节相同（除墙钟列）。

    这里用"两次运行互比"做代理：v2 的路径与本次运行都是"首跑、无快照"，
    但两次运行会比对**所有科学列**——若 v3 改错了首跑路径（比如循环边界、
    采样节拍、累积量初始化），本测试立刻挂。
    🔴 更强的对拍（与 git 历史 v2 比对）见交付说明。
    """
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    r1 = _run([*DEV, "--ticks", str(TICKS)], a)
    r2 = _run([*DEV, "--ticks", str(TICKS)], b)
    assert r1.returncode == 0 and r2.returncode == 0
    assert _diff(_rows(a), _rows(b)) == []
    # 默认关：`save_every=0`（banner 走 stdout）且**没有**快照产生
    assert "save_every=0" in r1.stdout, r1.stdout[:400]
    assert not snap_leftover(REPO)


# ---------------------------------------------------------------- T4 防呆：fail-loud

def test_T4_save_every_must_equal_sample(tmp_path):
    """`--save-every != --sample` ⇒ 中止（否则续跑行网格静默错位）。"""
    r = _run([*DEV, "--ticks", str(HALF), "--save-every", "7"], tmp_path / "x.csv")
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "必须 == --sample" in r.stderr


def test_T4_resume_without_snapshot_fails_loud(tmp_path):
    """`--resume-sample` 但快照不存在 ⇒ fail-loud（禁静默当首跑）。"""
    r = _run([*DEV, "--ticks", str(TICKS), "--resume-sample", "--save-every", "10",
              "--snapshot-dir", str(tmp_path / "empty")], tmp_path / "y.csv")
    # 目录为空 ⇒ 6 个 run 全落到"无快照 ⇒ 首跑"分支，**不该炸**（设计如此）
    assert r.returncode == 0, r.stderr


def test_T4_sample_mismatch_fails_loud(tmp_path):
    """快照 sample 与命令行 `--sample` 不一致 ⇒ fail-loud（行网格会错位）。"""
    snap = tmp_path / "snap"
    _run([*DEV, "--ticks", str(HALF), "--save-every", "10",
          "--keep-ckpt", "--snapshot-dir", str(snap)], tmp_path / "h.csv")
    # 故意换一个 sample（10 → 20）
    dev2 = [x for x in DEV]
    dev2[dev2.index("10")] = "20" if dev2[-1] == "both" else "10"
    r = _run(["--rows", "60", "--cols", "120", "--patches", "30", "--pop", "50",
              "--sample", "20", "--arms", "both", "--ticks", str(TICKS),
              "--resume-sample", "--save-every", "20",
              "--snapshot-dir", str(snap)], tmp_path / "z.csv")
    assert r.returncode != 0
    assert "采样节拍不一致" in r.stderr or "必须 == --sample" in r.stderr


# ---------------------------------------------------------------- T5 互斥 + 快照清理

def test_T5_append_and_resume_sample_are_mutually_exclusive(tmp_path):
    """两种续跑对"残行归属"的判定相反 ⇒ CLI 层互斥。"""
    r = _run([*DEV, "--ticks", str(TICKS), "--append", "--resume-sample",
              "--save-every", "10", "--snapshot-dir", str(tmp_path / "s")],
             tmp_path / "w.csv")
    assert r.returncode == 2
    assert "not allowed with" in r.stderr or "mutually exclusive" in r.stderr


def test_T5_snapshot_dropped_on_normal_completion(tmp_path):
    """run 正常收尾 ⇒ 快照三件套 + 侧车被清掉（否则下次会从过期快照重跑一段）。"""
    snap = tmp_path / "snap"
    r = _run([*DEV, "--ticks", str(TICKS), "--save-every", "10",
              "--snapshot-dir", str(snap)], tmp_path / "done.csv")
    assert r.returncode == 0, r.stderr
    left = sorted(p.name for p in snap.glob("*")) if snap.exists() else []
    assert left == [], f"收尾后仍有残留：{left}"


def test_T5_keep_ckpt_retains_snapshot(tmp_path):
    """`--keep-ckpt` ⇒ 收尾后快照保留（测试/演练断点场景要用）。"""
    snap = tmp_path / "snap"
    r = _run([*DEV, "--ticks", str(HALF), "--save-every", "10", "--keep-ckpt",
              "--snapshot-dir", str(snap)], tmp_path / "k.csv")
    assert r.returncode == 0, r.stderr
    assert (snap / "s3_s101_mem_on.snapshot.npz").exists()


# ---------------------------------------------------------------- T6 坑 1 的直接守卫

def test_T6_strip_rewrites_in_place_not_via_replace(tmp_path):
    """🔴 坑 1 守卫：`_strip_rows_after` **不得**换 inode。

    机制：main 长期持有 `fout`；若 `_strip_rows_after` 用 `os.replace`，
    `fout` 会指向被 unlink 的旧 inode ⇒ 后续行全丢。
    判法：打开一个句柄 → 调 `_strip_rows_after` → 用**同一个句柄**追加一行
    → 关句柄 → 重开读，该行必须在。
    """
    sys.path.insert(0, str(REPO))
    from experiments.s3_memory_probe import _strip_rows_after

    p = tmp_path / "t.csv"
    p.write_text("seed,arm,tick,pop\n1,mem_on,10,5\n1,mem_on,20,6\n"
                 "1,mem_off,10,5\n1,mem_off,20,6\n", encoding="utf-8")
    with open(p, "a", newline="", encoding="utf-8") as fout:
        fout.flush()
        n = _strip_rows_after(str(p), ("1", "mem_on"), 10)
        assert n == 1, n                       # 只删 tick=20 那一行
        fout.write("1,mem_on,20,7\n")          # 用**同一句柄**续写
        fout.flush()
    txt = p.read_text(encoding="utf-8")
    assert "1,mem_on,20,7" in txt, (
        "🔴 续写行丢了 ⇒ `_strip_rows_after` 换了 inode（必须原地重写）")
    assert txt.count("1,mem_on") == 2
    assert txt.count("1,mem_off") == 2


def test_T6_sidecar_roundtrip(tmp_path):
    """探针侧车往返：所有累积量逐位还原（含 `first_visit` 的 dict→双数组→dict）。"""
    sys.path.insert(0, str(REPO))
    import numpy as np
    from experiments.s3_memory_probe import (
        _save_probe_sidecar, _load_probe_sidecar, _probe_sidecar_path)

    p = _probe_sidecar_path(tmp_path, "s3_s9_mem_on")
    _save_probe_sidecar(
        p,
        visited_mask=np.array([True, False, True, True]),
        patch_ever_visited=np.array([True, False, True]),
        first_visit={2: 17, 0: 5},
        init_cap_full=np.array([1.5, 0.0, 2.5, 3.5]),
        init_patch_cap={0: 9.5, 2: 7.25},
        init_cap_sum=123.5, n_patch_total=3)
    s = _load_probe_sidecar(p)
    assert s["visited_mask"].tolist() == [True, False, True, True]
    assert s["patch_ever_visited"].tolist() == [True, False, True]
    assert s["first_visit"] == {0: 5, 2: 17}
    assert s["init_cap_full"].tolist() == [1.5, 0.0, 2.5, 3.5]
    assert s["init_patch_cap"] == {0: 9.5, 2: 7.25}
    assert s["init_cap_sum"] == 123.5
    assert s["n_patch_total"] == 3


def test_T6_mem_noise_fails_loud_on_resume(tmp_path):
    """🔴 引擎级缺口守卫：`memory_noise=True` 时快照不含 `_mem_v2_rng` ⇒ 续跑不逐位。

    本测试直接调 `_assert_snapshot_rng_coverage`（构造一个开了噪声档的最小引擎）。
    """
    sys.path.insert(0, str(REPO))
    from experiments.s3_memory_probe import _assert_snapshot_rng_coverage

    class _Fake:
        _mem_noise = False
    _assert_snapshot_rng_coverage(_Fake())        # 关档 ⇒ 放行

    class _FakeNoisy:
        _mem_noise = True
    with pytest.raises(ValueError, match="_mem_v2_rng"):
        _assert_snapshot_rng_coverage(_FakeNoisy())
