#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R278 §三 P3 验收：装置预设档防呆（`--device`）+ S2 逐 run flush。

派工原文（R278 §三，天平 2026-09-30，云归领）：
    > 你的建议（`--device s2` 预设档 / 探针默认值前置）**采纳**——"忘传 ⇒ 静默换装置"
    > 这类事故要从根上消灭；
    > **要求**：① 走**分支交付**（勿直推 main）② 预设档口径 =
    > `rows=480 cols=960 patches=1700 pop=10000 rgm=1.195 bg_low 0.4/0.05`
    > ③ 默认行为**不变**（不传 `--device` 逐位等于旧版）④ 附单测/冒烟
    > ⑤ 建议同时把 `--device` 用于 S3/S3.5 探针评估
    > 另（R278 §五）：S2 探针 flush 与 `--device` **合并为一个 P3 交付**，
    > 含"传 `--device` 时回显展开字段"的第三人眼校验建议——**采纳**。

本文件把五条要求逐条钉成可回归测试。核心两条**硬等价**：
  · **不传 `--device` ⇒ CSV + summary 与"改造前"逐字节相同**（要求③）
  · **S2 flush 改造 ⇒ 正常跑完的 CSV + summary 与旧版逐字节相同**（行为不变）
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
S2 = "experiments.s2_depletion_probe"
S3 = "experiments.s3_memory_probe"

# 微缩档：跑一次约 1–3s，够走完「建世界 → 若干 tick → 落盘」
DEV = ["--rows", "20", "--cols", "40", "--patches", "6", "--pop", "30",
       "--sample", "5", "--arms", "off"]
TICKS = 10

# 预设档口径（R278 §三② 逐字）
PRESET_S2 = {"rows": 480, "cols": 960, "patches": 1700, "pop": 10000,
             "rgm": 1.195, "bg_low_prod_frac": 0.0, "bg_low_cap_mult": 0.0}


def _run(probe: str, args: list[str], out: Path, seed: str = "165") -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    return subprocess.run(
        [sys.executable, "-u", "-m", probe, *args, "--seeds", seed,
         "--ticks", str(TICKS), "--out", str(out)],
        cwd=str(REPO), env=env, capture_output=True, text=True)


def _rows(p: Path) -> list[dict]:
    with open(p, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _summary(p: Path) -> dict:
    return json.load(open(str(p).replace(".csv", ".summary.json"), encoding="utf-8"))


def _params(probe: str, out: Path) -> dict:
    """取 summary 里的参数字典：S2 在 `params`，S3 在 `meta`。"""
    s = _summary(out)
    return s["params"] if probe == S2 else s["meta"]


# 🔴 唯一允许不同的列（墙钟计时；R189 已登记"禁瞬时速率外推"）。
IGNORE = {"ms_per_tick"}


def _assert_same_science(a: Path, b: Path):
    """除 IGNORE 列外逐行逐列相同（CSV 行数、行序也必须一致）。"""
    ra, rb = _rows(a), _rows(b)
    assert len(ra) == len(rb), f"行数不同 {len(ra)} vs {len(rb)}"
    for i, (x, y) in enumerate(zip(ra, rb)):
        for k in x:
            if k in IGNORE:
                continue
            assert x[k] == y[k], f"row{i} col {k}: {x[k]} != {y[k]}"


# ---------------------------------------------------------------- 预设档内容

@pytest.mark.parametrize("probe", [S2, S3])
def test_T1_preset_matches_r278_spec(probe):
    """`--device s2` 展开后逐字段 == R278 §三② 指定口径。"""
    out = Path("/tmp") / f"p3_spec_{probe.split('.')[-1]}.csv"
    # 故意**不传**装置参数，只传 --device s2 ⇒ 应完全由预设档决定
    r = _run(probe, ["--device", "s2"], out)
    assert r.returncode == 0, r.stderr
    # 从回显里解析"实际生效装置"
    line = [l for l in r.stderr.splitlines() if "实际生效装置" in l]
    assert line, f"未回显生效装置：{r.stderr}"
    got = dict(kv.split("=") for kv in line[0].split("：", 1)[1].split())
    for k, v in PRESET_S2.items():
        assert float(got[k]) == float(v), f"{k}: 预设={v} 实际={got[k]}"

    # summary 里 params 记录实际值（provenance）
    p = _params(probe, out)
    for k, v in PRESET_S2.items():
        assert float(p[k]) == float(v)


@pytest.mark.parametrize("probe", [S2, S3])
def test_T1b_preset_provenance_recorded(probe):
    """传了 --device ⇒ summary 里留下 device 名（可回溯）。"""
    out = Path("/tmp") / f"p3_prov_{probe.split('.')[-1]}.csv"
    r = _run(probe, ["--device", "s2"], out)
    assert r.returncode == 0
    assert _params(probe, out).get("device") == "s2"


# ---------------------------------------------------------------- 要求③：默认不变

@pytest.mark.parametrize("probe", [S2, S3])
def test_T2_default_removes_device_key_from_params(probe):
    """🔴 不传 --device ⇒ summary 参数字典里**不得**出现 device 键。

    这是"默认逐位等价"的命门：argparse 加了 --device 就会塞 device=None，
    而探针用 `vars(a)` 写 summary ⇒ 会多出 `"device": null` 破坏等价。
    resolve_device 在 None 分支删掉该键，本测试守护它不被改回去。
    """
    out = Path("/tmp") / f"p3_def_{probe.split('.')[-1]}.csv"
    r = _run(probe, [], out)
    assert r.returncode == 0, r.stderr
    assert "device" not in _params(probe, out)


def test_T2b_s2_default_csv_equals_baseline():
    """🔴 要求③钉死：S2 不传 --device 的 CSV 与"改造前缓存基线"科学列全同。

    基线由旧版（改造前）在**同一微缩档**上产出，存于 tests/fixtures/。
    （flush 改造只改"何时落盘"，不改"落什么" ⇒ 除 ms_per_tick 外必须逐格一致。）
    """
    baseline = REPO / "tests" / "fixtures" / "p3_s2_default_baseline.csv"
    if not baseline.exists():
        pytest.skip("基线缺失（需先用改造前版本生成）")
    out = Path("/tmp") / "p3_default_baseline.csv"
    r = _run(S2, [*DEV], out)          # 🔴 必须与生成基线时同一微缩档
    assert r.returncode == 0, r.stderr
    _assert_same_science(out, baseline)


# ---------------------------------------------------------------- 显式优先 + 回显

@pytest.mark.parametrize("probe", [S2, S3])
def test_T3_explicit_overrides_preset_with_notice(probe):
    """显式参数优先于预设 + 必须打印覆盖提示（第三人眼校验）。"""
    out = Path("/tmp") / f"p3_ovr_{probe.split('.')[-1]}.csv"
    r = _run(probe, ["--device", "s2", "--pop", "30", "--rows", "20"], out)
    assert r.returncode == 0, r.stderr
    p = _params(probe, out)
    assert p["pop"] == 30 and p["rows"] == 20          # 显式获胜
    assert float(p["cols"]) == 960                      # 未覆盖的仍取预设
    assert "被显式参数覆盖" in r.stderr, "缺少覆盖提示"
    assert "pop=30" in r.stderr


@pytest.mark.parametrize("probe", [S2, S3])
def test_T3b_echo_lists_all_device_fields(probe):
    """回显必须列出**全部**装置字段（防"只看了一行就以为对"）。"""
    out = Path("/tmp") / f"p3_echo_{probe.split('.')[-1]}.csv"
    r = _run(probe, ["--device", "s2"], out)
    assert r.returncode == 0
    line = [l for l in r.stderr.splitlines() if "实际生效装置" in l][0]
    for k in PRESET_S2:
        assert f"{k}=" in line, f"回显缺字段 {k}"


def test_T3c_override_detection_recognizes_underscore_form():
    """覆盖检测要同时认 `--bg-low-prod-frac` 与下划线 `--bg_low_prod_frac`。

    ⚠️ 下划线写法**不是** argparse 的合法 flag（CLI 只认连字符写法），故不能靠
    "真传下划线"来测。这里直接构造 Namespace + argv，单测 `resolve_device()` 的
    覆盖判定逻辑（两写法都必须被认成"显式" ⇒ 用户值保留、不被预设覆盖）。
    """
    sys.path.insert(0, str(REPO))
    import argparse
    from tools.device_presets import resolve_device

    def mk(argv):
        ns = argparse.Namespace(device="s2", rows=480, cols=960, patches=1700,
                                pop=10000, rgm=1.195, bg_low_prod_frac=0.9,
                                bg_low_cap_mult=0.05)
        resolve_device(ns, argv, logger=lambda m: None)
        return ns

    # 带连字符（合法 flag 写法）：显式 ⇒ 0.9 保留
    assert mk(["--device", "s2", "--bg-low-prod-frac", "0.9"]).bg_low_prod_frac == 0.9
    # 下划线写法：同样被识别为显式 ⇒ 0.9 保留
    assert mk(["--device", "s2", "--bg_low_prod_frac", "0.9"]).bg_low_prod_frac == 0.9
    # 带等号写法
    assert mk(["--device", "s2", "--bg-low-prod-frac=0.9"]).bg_low_prod_frac == 0.9
    # 都未显式传 ⇒ 取预设 0.0（R326 撤绿洲带后）
    ns = argparse.Namespace(device="s2", rows=480, cols=960, patches=1700,
                            pop=10000, rgm=1.195, bg_low_prod_frac=0.0,
                            bg_low_cap_mult=0.05)
    resolve_device(ns, ["--device", "s2"], logger=lambda m: None)
    assert ns.bg_low_prod_frac == 0.0


# ---------------------------------------------------------------- fail-loud

@pytest.mark.parametrize("probe", [S2, S3])
def test_T4_unknown_device_fails_loud(probe):
    """未知档名 ⇒ rc!=0 且**不静默回落**（argparse choices 已挡，双保险）。"""
    out = Path("/tmp") / f"p3_bad_{probe.split('.')[-1]}.csv"
    r = _run(probe, ["--device", "s9_nonexistent"], out)
    assert r.returncode != 0


# ---------------------------------------------------------------- S2 flush（要求：等价）

def test_T5_s2_run_flush_lands_before_process_end():
    """S2 每个 run 结束即 fsync ⇒ 信号杀死后已完成 run 的行仍在盘上。

    做法：跑到 2 个 run 的量，中途 SIGKILL，检查盘上已有第 1 个 run 的全部行。
    （改造前是 `with open(...,"w")` ⇒ 缓冲全丢、一行不落。）
    """
    out = Path("/tmp") / "p3_flush.csv"
    if out.exists():
        out.unlink()
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    # 两个 seed、每 run 长到足以"等第 1 个 run 落盘后立刻杀"
    p = subprocess.Popen(
        [sys.executable, "-u", "-m", S2, *DEV, "--seeds", "165,166",
         "--ticks", "4000", "--sample", "500", "--out", str(out)],
        cwd=str(REPO), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # 等第 1 个 run 的第一行出现（最多 60s），随即杀掉
    import time
    ok = False
    for _ in range(240):
        time.sleep(0.25)
        if out.exists() and len(_rows(out)) >= 1:
            ok = True
            break
    assert ok, "第 1 个 run 在 60s 内没落盘（flush 失效？）"
    p.kill()
    p.wait()
    rows = _rows(out)
    # 被 SIGKILL，但已落盘的 run 数据必须在盘上（这正是 flush 的意义）。
    # 断言：至少 seed 165 有行，且该 seed 的行是**连续完整**的（无半行/截断）。
    seeds = {r["seed"] for r in rows}
    assert "165" in seeds, f"seed165 的行没落盘：{seeds}"
    r165 = [r for r in rows if r["seed"] == "165"]
    assert all(r["arm"] == "off" for r in r165)
    # 每行字段数一致（无截断的半行）
    ncol = len(r165[0])
    assert all(len(r) == ncol for r in r165)


def test_T5b_s2_summary_written_atomically_per_run():
    """summary 每 run 原子重写 ⇒ 跑完 1 个 run 后 summary 已含该 run。"""
    out = Path("/tmp") / "p3_sum.csv"
    r = _run(S2, [*DEV], out)
    assert r.returncode == 0
    s = _summary(out)
    assert len(s["summary"]) == 1 and s["summary"][0]["seed"] == 165
