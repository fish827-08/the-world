"""E-057 批编排（`experiments/e057_repw_batch.py`）靶向测试 —— 卡 BATCH-SCRIPT。

零跑批：全程只拼 argv / 造假 summary，**不 import 引擎不消费 RNG**（t11 有 import 守卫）。
变异目标：档位矩阵、solo 在场、装置锁字段、回显键核对、断点续跑判据、平台门。
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import e057_repw_batch as P  # noqa: E402


# ---------------- 造具 ----------------

def _plan(tmp: Path) -> list[dict]:
    return P.plan_runs(tmp)


def _find(runs: list[dict], tier: float, seed: int) -> dict:
    return next(r for r in runs if r["tier"] == tier and r["seed"] == seed)


def _fake_run_outs(tmp: Path, tier: float, seed: int, *, enabled: bool = True,
                   eff: float | None = None, meta_rep_w: float | None = ...,
                   meta_solo: bool = True, arm: str = "mem_off", n_runs: int = 1,
                   real_seed: int | None = ..., rd: bool = False,
                   alphabet: str = "8", csv_lines: int = 21) -> dict:
    """按探针 summary 形状造假产物（键名/层级与 s3_memory_probe 一致）。"""
    runs = _plan(tmp)
    run = copy.deepcopy(_find(runs, tier, seed))
    run["csv"].parent.mkdir(parents=True, exist_ok=True)
    run["summary"].parent.mkdir(parents=True, exist_ok=True)
    run["csv"].write_text("tick,pop\n" + "".join(f"{i},100\n" for i in range(csv_lines - 1)),
                          encoding="utf-8")
    body = []
    for _ in range(n_runs):
        body.append({
            "seed": seed if real_seed is ... else real_seed,
            "arm": arm,
            "stop": "ticks", "wall_s": 238.6,
            "info_structure_enabled_effective": enabled,
            "reputation_weight_effective": tier if eff is None else eff,
            "rd_enabled_effective": rd,
            "signal_alphabet_effective": alphabet,
        })
    meta = {"rows": 480, "cols": 960, "patches": 1700, "pop": 10000,
            "rep_w": tier if meta_rep_w is ... else meta_rep_w,
            "rep_w_solo": meta_solo}
    run["summary"].write_text(json.dumps({"meta": meta, "runs": body}), encoding="utf-8")
    return run


# ---------------- 一、矩阵与装置锁 ----------------

def test_t1_matrix_is_32_and_all_solo(tmp_path):
    runs = _plan(tmp_path)
    assert len(runs) == P.EXPECTED_RUNS == 32
    assert sorted({r["tier"] for r in runs}) == [0.0, 0.05, 0.2, 0.5]
    assert sorted({r["seed"] for r in runs}) == list(range(207, 215))
    assert len({str(r["csv"]) for r in runs}) == 32          # 输出不互相覆盖
    for r in runs:
        assert "--rep-w-solo" in r["argv"], f"{r['name']} 缺 solo ⇒ 裸 rep_w 空转"
        assert r["argv"][r["argv"].index("--rep-w") + 1] == r["tier_declared"]


def test_t2_golden_argv(tmp_path):
    """锁定档全 argv 金样本：任何装置常量漂移即红（PI 会签 @2306949 的装置面）。"""
    argv = P.build_argv(0.05, 207, tmp_path)
    assert argv == [
        "-m", "experiments.s3_memory_probe", "--device", "s2",
        "--rows", "480", "--cols", "960", "--patches", "1700", "--pop", "10000",
        "--rgm", "1.195", "--bg-low-prod-frac", "0.0", "--bg-low-cap-mult", "0.0",
        "--seeds", "207", "--ticks", "2000", "--sample", "100",
        "--arms", "off", "--rd-mode", "off", "--alphabet", "8",
        "--m0-instruments", "--m0-emit-window-frac", "0.25",
        "--rep-w", "0.05", "--rep-w-solo",
        "--out", str(tmp_path / "e057_repw" / "e057_rpw5_d1700_s207_t2000.csv"),
    ]


def test_t3_anchor_tier_zero_is_named_rpw0(tmp_path):
    r = _find(_plan(tmp_path), 0.0, 210)
    assert r["name"] == "e057_rpw0_d1700_s210_t2000"
    assert "--rep-w-solo" in r["argv"]                       # 锚臂也 solo（同指纹配对）


def test_t4_unknown_tier_rejected(tmp_path):
    with pytest.raises(P.PlanError, match="不在锁定矩阵"):
        P.build_argv(0.1, 207, tmp_path)


@pytest.mark.parametrize("mutate,why", [
    (lambda a: a.pop(a.index("--rep-w-solo")), "缺 --rep-w-solo"),
    (lambda a: a.__setitem__(a.index("--alphabet") + 1, "4"), "alphabet"),
    (lambda a: a.__setitem__(a.index("--patches") + 1, "1500"), "patches"),
    (lambda a: a.__setitem__(a.index("--rd-mode") + 1, "on"), "rd-mode"),
    (lambda a: a.__setitem__(a.index("--sample") + 1, "250"), "sample"),
    (lambda a: a.pop(a.index("--m0-instruments")), "m0-instruments"),
])
def test_t5_check_argv_catches_device_drift(tmp_path, mutate, why):
    runs = _plan(tmp_path)
    r = _find(runs, 0.2, 209)
    bad = list(r["argv"])
    mutate(bad)
    with pytest.raises(P.PlanError):
        P.check_argv(0.2, 209, bad)


def test_t6_plan_runs_rejects_tier_seed_mismatch(tmp_path, monkeypatch):
    """tier 与 argv 里的 --rep-w 不同步 ⇒ 静态守当场红（三档配错值=判据作废）。"""
    orig = P.build_argv

    def sabotage(tier, seed, data_dir):
        argv = orig(tier, seed, data_dir)
        if tier == 0.5:
            argv[argv.index("--rep-w") + 1] = "0.05"
        return argv

    monkeypatch.setattr(P, "build_argv", sabotage)
    with pytest.raises(P.PlanError, match="应为 '0.5'"):
        P.plan_runs(tmp_path)


# ---------------- 二、回显键核对（跑前/跑中硬门） ----------------

@pytest.mark.parametrize("tier", [0.0, 0.05, 0.2, 0.5])
def test_t7_verify_ok_for_all_four_arms(tmp_path, tier):
    run = _fake_run_outs(tmp_path, tier, 207)
    echo = P.verify_summary(run)
    assert echo["reputation_weight_effective"] == tier
    assert echo["info_structure_enabled_effective"] is True
    assert echo["meta_rep_w_solo"] is True
    assert echo["csv_rows"] == 20


def test_t8_inert_tier_is_red(tmp_path):
    """声明 0.5 而生效 0.0 = 引擎 :3999 空转 ⇒ 必红（否则三档逐位相同=白跑）。"""
    run = _fake_run_outs(tmp_path, 0.5, 208, eff=0.0)
    with pytest.raises(P.EchoError, match="reputation_weight_effective"):
        P.verify_summary(run)


def test_t9_enabled_false_is_red_even_for_anchor(tmp_path):
    run = _fake_run_outs(tmp_path, 0.0, 209, enabled=False)
    with pytest.raises(P.EchoError, match="四臂全 solo"):
        P.verify_summary(run)


@pytest.mark.parametrize("kw,why", [
    ({"meta_rep_w": 0.2}, "meta 回显"),
    ({"meta_solo": False}, "meta 回显"),
    ({"real_seed": 210}, "seed 回显"),
    ({"arm": "mem_on"}, "mem_off"),
    ({"n_runs": 2}, "单 seed 单臂"),
    ({"rd": True}, "rd_enabled_effective"),
    ({"alphabet": "4"}, "signal_alphabet_effective"),
])
def test_t10_other_echo_gates_red(tmp_path, kw, why):
    run = _fake_run_outs(tmp_path, 0.05, 207, **kw)
    with pytest.raises(P.EchoError, match=why):
        P.verify_summary(run)


def test_t11_missing_artifacts_red(tmp_path):
    r = _find(_plan(tmp_path), 0.2, 211)
    with pytest.raises(P.EchoError, match="summary 缺失"):
        P.verify_summary(r)
    run = _fake_run_outs(tmp_path, 0.2, 211)
    run["csv"].write_text("", encoding="utf-8")
    with pytest.raises(P.EchoError, match="CSV 缺失或空"):
        P.verify_summary(run)


# ---------------- 三、断点续跑判据 ----------------

def test_t12_resume_semantics(tmp_path):
    r = _find(_plan(tmp_path), 0.05, 212)
    assert P.complete(r, False) is False                      # 无产物 ⇒ 跑
    run = _fake_run_outs(tmp_path, 0.05, 212)
    assert P.complete(run, False) is True                     # 产物+核得过 ⇒ 跳
    bad = _fake_run_outs(tmp_path, 0.2, 212, eff=0.0)          # 产物在场但空转
    assert P.complete(bad, False) is False                    # ⇒ 自愈重跑
    assert P.complete(bad, True) is True                      # strict 才只看在场


# ---------------- 四、执行门（平台 / 解释器 / 零副作用） ----------------

def test_t13_run_refused_off_cloud(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(P.platform, "system", lambda: "Windows")
    rc = P.main(["--run", "--data-dir", str(tmp_path)])
    err = capsys.readouterr().err
    assert rc == 4 and "R117/R398" in err                      # 平台门本体，非别处报错凑数
    assert not (tmp_path / "e057_logs" / "status.txt").exists()


def test_t14_missing_interpreter_fails_loud(tmp_path, monkeypatch):
    monkeypatch.setattr(P.platform, "system", lambda: "Linux")
    monkeypatch.setattr(P, "avail_gb", lambda: 1.5)
    monkeypatch.setattr(P, "carrier_head", lambda: ("deadend", P.CARRIER_BASE))
    called = []
    monkeypatch.setattr(P.subprocess, "call",
                        lambda *a, **k: called.append(a) or 0)
    rc = P.main(["--run", "--data-dir", str(tmp_path), "--allow-non-cloud",
                 "--python", str(tmp_path / "nope" / "python")])
    assert rc == 4 and called == []                           # 解释器缺失 ⇒ 零子进程


def test_t15_missing_data_dir_fails_loud(tmp_path, monkeypatch):
    monkeypatch.setattr(P.platform, "system", lambda: "Linux")
    monkeypatch.setattr(P, "avail_gb", lambda: 1.5)
    rc = P.main(["--run", "--allow-non-cloud", "--python", sys.executable,
                 "--data-dir", str(tmp_path / "不在的地方")])
    assert rc == 4


def test_t16_plan_has_zero_side_effects(tmp_path, monkeypatch, capsys):
    """--plan 是 dry-run：不许起子进程、不许写盘（本机不跑批 R117 的机器面保证）。"""
    def _boom(*a, **k):
        raise AssertionError("--plan 不得起子进程")

    monkeypatch.setattr(P.subprocess, "call", _boom)
    monkeypatch.setattr(P.subprocess, "run", _boom)
    monkeypatch.setattr(P.subprocess, "Popen", _boom)
    rc = P.main(["--plan", "--data-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.count("python -m experiments.s3_memory_probe") == 32
    lines = [ln for ln in out.splitlines() if ln.startswith("[")]
    assert len(lines) == 32
    assert all("--rep-w-solo" in ln for ln in lines)          # 32 行全带 solo
    assert list(tmp_path.rglob("*.csv")) == []
    assert list(tmp_path.rglob("*.json")) == []


def test_t17_missing_mode_is_usage_error(tmp_path):
    with pytest.raises(SystemExit) as e:
        P.main(["--data-dir", str(tmp_path)])
    assert e.value.code == 2                                  # argparse 互斥必选


# ---------------- 五、manifest / 载体指纹 ----------------

def test_t18_manifest_row_has_four_elements(tmp_path):
    run = _fake_run_outs(tmp_path, 0.5, 214)
    row = P.manifest_row(run, 0, P.verify_summary(run), "abc1234")
    for k in ("rows", "cols", "patches", "pop"):              # R277 四要素逐字段
        assert row[k] == int(P.FIELD_LOCK[k])
    assert row["carrier_base"] == "7faca82" and row["carrier_head"] == "abc1234"
    assert row["argv"][0] == "python" and "--rep-w-solo" in row["argv"]
    assert row["echo"]["reputation_weight_effective"] == 0.5
    assert "error" not in row
    bad = P.manifest_row(run, 1, None, "abc1234", "炸了")
    assert bad["error"] == "炸了"


def test_t20_default_python_resolution(tmp_path, monkeypatch):
    """解释器候选：云机首次起跑撞过「拿 data-dir 父目录猜 .venv」⇒ 逐候选验存在才可用。"""
    import os

    home = tmp_path / "home"
    exe = "python.exe" if os.name == "nt" else "python"
    sub = "Scripts" if os.name == "nt" else "bin"
    good = home / "world" / "the-world" / ".venv" / sub / exe
    good.parent.mkdir(parents=True)
    good.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.delenv("E057_PY", raising=False)
    monkeypatch.setattr(P, "ROOT", tmp_path / "无树的根")
    monkeypatch.setattr(P, "_main_repo_root", lambda: None)
    monkeypatch.setattr(P.Path, "home", staticmethod(lambda: home))
    assert P.default_python() == str(good)
    good.unlink()
    with pytest.raises(P.EchoError, match="找不到可用解释器"):
        P.default_python()


def test_t19_no_engine_import_in_orchestrator():
    """不消费 RNG 的结构证明：编排模块的 import 面里没有引擎/仿真。"""
    import re

    src = (ROOT / "experiments" / "e057_repw_batch.py").read_text(encoding="utf-8")
    imports = [ln for ln in src.splitlines() if re.match(r"^\s*(from|import)\s", ln)]
    assert imports, "import 行应能解析出来（防正则失效变成空断言）"
    assert not any(("simulation" in ln) or ("sphere_engine" in ln)
                   or ("numpy" in ln) for ln in imports), imports
