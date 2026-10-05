"""S3-RDSWITCH 验收：`s3_memory_probe` 的 `--rd-mode` / `--alphabet` 两开关。

口径（M1 装置互斥消解，轻舟 2026-10-05；卡 S3-RDSWITCH）：
  ① 默认（不传两参）与显式默认（`--rd-mode on --alphabet 16`）输出等价
     —— CSV 除 `ms_per_tick` 外逐行一致，summary meta 两键**均被摘除**（`_summary_meta`）；
  ② choices 收紧 ⇒ 非法档 argparse rc≠0（fail-loud，不静默降级）；
  ③ 档位真落地：`_build_fresh_run` 出参引擎 config 即所选档（rd off / alphabet "4"）；
  ④ 续跑指纹：rd/alphabet 进 `_config_fingerprint_check` 逐字段核对 ⇒ 跨档续跑硬报错；
  ⑤ `_summary_meta`：默认摘 / 非默认留 / **不改 namespace**（拷贝语义，防后续 a.rd_mode 炸）；
  ⑥ `--alphabet 8` ∧ mem_on ⇒ 「记忆位恒 0」警告（rc=0，**告警不拦**——属可测性质，
     见 R345 v1.2 P1-a：memv2 下 `_work_memory` 不写 ⇒ mem_bit≡0）。

运行成本：全部走微缩档（60×120），T1 两条 30t、T6 两条 10t、T3/T4 各建一次引擎。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROBE = "experiments.s3_memory_probe"

# 微缩档（与 test_r287 同源）
DEV = ["--rows", "60", "--cols", "120", "--patches", "30", "--pop", "50",
       "--sample", "10", "--arms", "both"]

# 🔴 唯一允许不同的列（墙钟计时）。别往里加科学量。
IGNORE = {"ms_per_tick"}


def _run(args: list[str], out: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    # PYTHONIOENCODING 钉死 UTF-8：告警行含 ⚠️（stderr），若父进程默认 GBK 解码
    # 会在 reader 线程炸 UnicodeDecodeError（本机 T6 首跑实测）。两侧同钉 ⇒ 与
    # 调用者 shell 环境无关。
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               PYTHONIOENCODING="utf-8")
    return subprocess.run(
        [sys.executable, "-u", "-m", PROBE, *args, "--out", str(out)],
        cwd=str(REPO), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace")


def _rows(p: Path) -> list[dict]:
    with open(p, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _diff(ref: list[dict], got: list[dict]) -> list[tuple]:
    ref = sorted(ref, key=lambda r: (r["seed"], r["arm"], int(float(r["tick"]))))
    got = sorted(got, key=lambda r: (r["seed"], r["arm"], int(float(r["tick"]))))
    if len(ref) != len(got):
        return [("__n__", len(ref), len(got))]
    return [(r["seed"], r["arm"], r["tick"], c, a[c], b[c])
            for a, b in zip(ref, got) for c in a
            if c not in IGNORE and a[c] != b[c]]


def _meta(out: Path) -> dict:
    with open(out.with_suffix(".summary.json"), "r", encoding="utf-8") as jf:
        return json.load(jf)["meta"]


# ---------------------------------------------------------------- ① 默认 = 显式默认

def test_T1_default_equals_explicit_default(tmp_path):
    """不传两参 ≡ 显式 `on`/`16`（行先于 meta 判：行不等则 meta 同也没意义）。"""
    ref, run = tmp_path / "ref.csv", tmp_path / "run.csv"
    r = _run([*DEV, "--ticks", "30"], ref)
    assert r.returncode == 0, r.stderr
    r = _run([*DEV, "--ticks", "30", "--rd-mode", "on", "--alphabet", "16"], run)
    assert r.returncode == 0, r.stderr
    d = _diff(_rows(ref), _rows(run))
    assert not d, f"🔴 默认≠显式默认（{len(d)} 处）：{str(d)[:600]}"
    # meta：两条都不得含 rd_mode/alphabet（默认摘除 ⇒ 与历史批逐字节可比）
    m_ref, m_run = _meta(ref), _meta(run)
    assert "rd_mode" not in m_ref and "alphabet" not in m_ref
    assert "rd_mode" not in m_run and "alphabet" not in m_run
    # out 是唯一合法差异（两条命令的落盘路径不同）；其余全等
    m_ref.pop("out"), m_run.pop("out")
    assert m_ref == m_run, "🔴 两条默认运行的 meta（除 out）不一致"


# ---------------------------------------------------------------- ② choices fail-loud

def test_T2_choices_reject_bogus(tmp_path):
    r = _run([*DEV, "--ticks", "10", "--rd-mode", "maybe"], tmp_path / "a.csv")
    assert r.returncode != 0, "非法 --rd-mode 竟然跑通了（静默降级？）"
    assert "invalid choice" in r.stderr
    r = _run([*DEV, "--ticks", "10", "--alphabet", "32"], tmp_path / "b.csv")
    assert r.returncode != 0, "非法 --alphabet 竟然跑通了（静默降级？）"
    assert "invalid choice" in r.stderr


# ---------------------------------------------------------------- ③ 档位真落地

def test_T3_switch_lands_in_engine_config():
    from experiments.s3_memory_probe import _build_fresh_run
    (_, _, _, _, _, _, _, _, _), eng = _build_fresh_run(
        901, 60, 120, 50, 30, 1.195, False, 0.0, 0.0, False,
        rd_on=False, alphabet="4")
    assert bool(eng.config.resource_dynamics.enabled) is False, "rd off 未落地"
    assert str(eng.config.signal_alphabet) == "4", "alphabet 4 未落地"
    # 反向：默认档
    (_, _, _, _, _, _, _, _, _), eng2 = _build_fresh_run(
        902, 60, 120, 50, 30, 1.195, False, 0.0, 0.0, False)
    assert bool(eng2.config.resource_dynamics.enabled) is True, "默认 rd on 未落地"
    assert str(eng2.config.signal_alphabet) == "16", "默认 alphabet 16 未落地"


# ---------------------------------------------------------------- ④ 跨档续跑硬报错

def test_T4_fingerprint_check_blocks_cross_switch():
    from experiments.s3_memory_probe import (
        _build_fresh_run, _config_fingerprint_check)
    (_, _, _, _, _, _, _, _, _), eng = _build_fresh_run(
        903, 60, 120, 50, 30, 1.195, False, 0.0, 0.0, False)
    _base = (eng, 903, 60, 120, 50, 30, 1.195, False, 0.0, 0.0, False)
    _config_fingerprint_check(*_base, True, "16")      # 同档 ⇒ 通过
    with pytest.raises(ValueError, match="rd_enabled"):
        _config_fingerprint_check(*_base, False, "16")  # rd 跨档 ⇒ 炸且指名字段
    with pytest.raises(ValueError, match="signal_alphabet"):
        _config_fingerprint_check(*_base, True, "4")    # alphabet 跨档 ⇒ 炸且指名字段


# ---------------------------------------------------------------- ⑤ _summary_meta 单元

def test_T5_summary_meta_strip_keep_and_no_mutation():
    from experiments.s3_memory_probe import _summary_meta
    ns = argparse.Namespace(rd_mode="on", alphabet="16", seeds="101")
    meta = _summary_meta(ns)
    assert "rd_mode" not in meta and "alphabet" not in meta, "默认值未摘除"
    assert meta == {"seeds": "101"}
    # 拷贝语义：原 namespace 不被改（直接 pop vars(a) 会让后续 a.rd_mode 炸）
    assert ns.rd_mode == "on" and ns.alphabet == "16", "namespace 被就地修改"
    ns2 = argparse.Namespace(rd_mode="off", alphabet="8", seeds="101")
    meta2 = _summary_meta(ns2)
    assert meta2["rd_mode"] == "off" and meta2["alphabet"] == "8", "非默认值被误摘"


# ---------------------------------------------------------------- ⑥ alphabet 8 告警

def test_T6_alphabet8_mem_on_warns_mem_off_silent(tmp_path):
    """M1 组合档：rd off + alphabet 8 + m0 仪表 + mem_on ⇒ rc=0 且出警告。"""
    r = _run([*DEV, "--ticks", "10", "--arms", "on", "--rd-mode", "off",
              "--alphabet", "8", "--m0-instruments"], tmp_path / "c.csv")
    assert r.returncode == 0, r.stderr
    assert "记忆位恒 0" in r.stderr, "mem_on 臂未出记忆位告警"
    # 非默认档进 meta（可追溯）
    assert _meta(tmp_path / "c.csv")["rd_mode"] == "off"
    assert _meta(tmp_path / "c.csv")["alphabet"] == "8"
    r2 = _run([*DEV, "--ticks", "10", "--arms", "off", "--alphabet", "8"],
              tmp_path / "d.csv")
    assert r2.returncode == 0, r2.stderr
    assert "记忆位恒 0" not in r2.stderr, "mem_off 臂不应出该告警（v1 记忆位非恒 0）"
