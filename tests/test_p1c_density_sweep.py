# -*- coding: utf-8 -*-
"""N2 密度批载体靶向测试（卡 N2-DENSITY-CARRIER，R373-①）。

覆盖：
 ① 矩阵冻结：四档×四seed=16 行、新增 12、命名/命令逐字（含显式 --device s2）——
    改锁定常量 ⇒ 必红（变异检查）；
 ② verify_readback 正/负例（R277 四要素-②）；
 ③ judge_band 三重锁 S1/S2/S3 + S4 NaN fail-loud + §四 生态门/止步；
 ④ cmd_aggregate 端到端（假 summary 数据 + 缺数据 rc=4 / S4 rc=3）；
 ⑤ smoke-gate 拒覆盖/缺存档门（不真跑）；run-matrix 无授权拒起跑 + dry-run 零子进程；
 ⑥ 探针 --home-range：默认关⇒表头/summary 键逐字节不变；开启⇒hr 列存在且口径
    逐式 = s3 R308（`_HomeRangeCore` 合成几何锚 + 经度环绕变异检查）。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments import p1c_density_sweep as sweep
from experiments.p1c_erasure_probe import _HomeRangeCore, _HR_COLS


def _ns(**kw):
    ns = {k: v for k, v in kw.items()}
    return type("NS", (), ns)()


# ---------------- ① 矩阵冻结 ----------------

def test_matrix_shape_and_naming():
    rows = sweep.plan_rows()
    assert len(rows) == 16
    assert [r["patches"] for r in rows] == [p for p in sweep.PATCHES_LADDER for _ in sweep.SEEDS]
    new = [r for r in rows if r["new_run"]]
    assert len(new) == 12 and all(r["patches"] != 1700 for r in new)
    assert sweep.out_name(850, 207) == "p1c_density_d850_s207_t3000"


def test_build_cmd_frozen_string():
    """逐字冻结 D1×207 命令：任何锁定常量漂移（device/patches/ticks/sample/ring/hr/smoke）⇒ 红。"""
    cmd = sweep.build_cmd("PY", 850, 207, Path("D/p1c_density_d850_s207_t3000.csv"))
    joined = " ".join(cmd)
    assert ("--device s2 --patches 850 --rows 480 --cols 960 --pop 10000 --rgm 1.195 "
            "--bg-low-prod-frac 0.0 --bg-low-cap-mult 0.0 --seed 207 --ticks 3000 "
            "--sample 100 --ring-depth 64 --home-range --smoke") in joined
    assert cmd[-2:] == ["--out", str(Path("D/p1c_density_d850_s207_t3000.csv"))]
    # 变异检查：R5.6 红线——s2 必须显式，patches 必须逐档显式（不吃默认 60×120/1700）
    assert "--device" in cmd and cmd[cmd.index("--device") + 1] == "s2"
    assert cmd[cmd.index("--patches") + 1] == "850"
    # D-DENSITY-SMOKE：D0（判别档）命令逐字不变 ⇒ 复现门/dm-guard 路径不受影响
    cmd0 = sweep.build_cmd("PY", 1700, 207, Path("D/d0.csv"))
    assert "--smoke" not in cmd0


def test_smoke_flag_matches_probe_guard(tmp_path, monkeypatch):
    """贯通测试（D-DENSITY-SMOKE 防复发）：载体造的命令必须真能过探针 R278 守卫。

    取真实 build_cmd 产物（仅把装置缩到微型 + ticks 缩短以省机时，不改参数**结构**），
    在进程内跑探针 main：
      ① 带 `--smoke` ⇒ 正常返回 0；
      ② 去掉 `--smoke` ⇒ 必须 SystemExit（守卫拒跑）——若载体修复被摘除，①变红。
    """
    from experiments import p1c_erasure_probe as probe
    monkeypatch.setitem(sweep.LOCKED_DEVICE, "rows", 24)
    monkeypatch.setitem(sweep.LOCKED_DEVICE, "cols", 60)
    monkeypatch.setitem(sweep.LOCKED_DEVICE, "pop", 50)
    monkeypatch.setattr(sweep, "TICKS", 40)
    monkeypatch.setattr(sweep, "SAMPLE", 20)

    out = tmp_path / "d850_s207.csv"
    cmd = sweep.build_cmd(sys.executable, 850, 207, out)
    assert "--smoke" in cmd and "--out" == cmd[-2]
    # cmd = [python, -u, -m, module, *探针参数] ⇒ 进程内喂 argv 须剥前 4 项
    monkeypatch.setattr(sweep.sys, "argv", ["p1c_erasure_probe"] + cmd[4:])
    assert probe.main() == 0
    assert out.exists()

    cmd_nosmoke = ["p1c_erasure_probe"] + [c for c in cmd[4:] if c != "--smoke"]
    monkeypatch.setattr(sweep.sys, "argv", cmd_nosmoke)
    with pytest.raises(SystemExit) as ei:
        probe.main()
    assert "--smoke" in str(ei.value)


def test_ladder_locked():
    assert sweep.PATCHES_LADDER == [1700, 850, 425, 213]
    assert sweep.SEEDS == [207, 208, 209, 210]
    assert (sweep.TICKS, sweep.SAMPLE, sweep.RING_DEPTH) == (3000, 100, 64)
    assert sweep.S0_THRESH == 0.50 and sweep.S2_MARGIN_MULT == 2.0
    assert sweep.S3_PEP_MAX == 0.80 and sweep.S3_POS_DMIN == 2.0 and sweep.MIN_ALIVE == 3


# ---------------- ② verify_readback ----------------

def _summary(patches=850, ticks=3000, sample=100, **rb_over):
    rb = {"signal_alphabet": "8", "memory_v2": False, "info_structure_enabled": False,
          "rd_enabled": False, "use_sim_core": False, "subpos_enabled": True,
          "rows": 480, "cols": 960, "initial_pop": 10000, "rgm": 1.195,
          "ring_depth": 64, "patches": patches}
    rb.update(rb_over)
    return {"readback": rb, "ticks": ticks, "sample": sample, "argv": [],
            "extinct_at": None, "pooled": {}, "per_sample": []}


def test_verify_readback_positive_and_negative():
    assert sweep.verify_readback(_summary(), expect_patches=850) == []
    mism = sweep.verify_readback(_summary(memory_v2=True), expect_patches=850)
    assert any("memory_v2" in m for m in mism)
    mism2 = sweep.verify_readback(_summary(patches=425), expect_patches=850)
    assert any("≠请求档 850" in m for m in mism2)
    mism3 = sweep.verify_readback(_summary(ticks=2000), expect_patches=850)
    assert any("ticks/sample" in m for m in mism3)
    mism4 = sweep.verify_readback(_summary(patches=999), expect_patches=999)
    assert any("不在四档阶梯" in m for m in mism4)


# ---------------- ③ judge_band 三重锁 ----------------

def _run(patches, seed, pos, n_pairs=1000, pep=0.45, d_med=3.5, extinct=None):
    return {"band": sweep.BAND_NAMES[patches], "patches": patches, "seed": seed,
            "src": "t", "reused": False, "extinct_at": extinct,
            "pos_frac": pos, "pos_n_pairs": n_pairs, "pos_d_med": d_med,
            "pep_frac": pep, "pep_n": 800, "pep_d_med": 1.0,
            "delta_r": (pos - pep) if (pos is not None and pep is not None) else None,
            "hr_run": None, "ms_tick_med": 100.0, "pop_end": 5000,
            "window_mean_pos_frac": pos}


def test_judge_band_positive():
    runs = [_run(850, s, 0.40 + 0.01 * i) for i, s in enumerate(sweep.SEEDS)]
    b = sweep.judge_band(850, runs)
    assert b["verdict"].startswith("阳性")
    assert b["s1_pass"] and b["s2_pass"] and b["s3_pass"]
    # 逐对池化（§3.1 主口径）：Σ(v×n)/Σn = 均值（n 相等时）
    assert abs(b["band_pooled_pos_frac"] - (0.40 + 0.43) / 2) < 1e-12


def test_judge_band_s1_fail_and_gray():
    runs = [_run(850, 207, 0.49), _run(850, 208, 0.49),
            _run(850, 209, 0.62), _run(850, 210, 0.70)]
    b = sweep.judge_band(850, runs)
    assert b["verdict"].startswith("不定") and not b["s1_pass"]


def test_judge_band_s2_margin():
    runs = [_run(850, s, v) for s, v in zip(sweep.SEEDS, [0.49, 0.48, 0.51, 0.44])]
    b = sweep.judge_band(850, runs)
    assert b["s1_pass"]                      # 3/4 ≤0.50
    assert not b["s2_pass"]                  # 档值离界余量 < 2×SD
    assert b["verdict"].startswith("不定")


def test_judge_band_s3_reverse_dynamics():
    runs = [_run(850, s, 0.40) for s in sweep.SEEDS]
    runs = [dict(r, pep=0.90) for r in runs]          # 擦除动力学反向（落 H1 带）
    for r in runs:
        r["pep_frac"] = 0.90
    b = sweep.judge_band(850, runs)
    assert not b["s3_pass"] and b["verdict"].startswith("不定")
    runs2 = [dict(r, pep=0.45) for r in runs]
    for r in runs2:
        r["pep_frac"], r["pos_d_med"] = 0.45, 1.5     # pos_d_med < 2.0 ⇒ S3 亦不过
    b2 = sweep.judge_band(850, runs2)
    assert not b2["s3_pass"]


def test_judge_band_s4_nan_fail_loud():
    """镜 N4 唯一修订条：任一存活 run NaN ⇒ 档判不定 + s4_nan 记录（不得静默跳过）。"""
    runs = [_run(850, 207, 0.40), _run(850, 208, float("nan")),
            _run(850, 209, 0.41), _run(850, 210, 0.42)]
    b = sweep.judge_band(850, runs)
    assert "S4 NaN" in b["verdict"] and b["s4_nan"] == [208]


def test_judge_band_eco_gate_and_partial():
    runs = [_run(850, 207, 0.40, extinct=1800), _run(850, 208, 0.41, extinct=2500),
            _run(850, 209, 0.42), _run(850, 210, 0.43)]
    b = sweep.judge_band(850, runs)
    assert "崩溃" in b["verdict"] and b["gate"] is False
    assert b["crashed_seeds"] == {"207": 1800, "208": 2500}


# ---------------- ④ cmd_aggregate 端到端 ----------------

def _write_fake_summary(dir: Path, patches, seed, pos, extinct=None):
    dir.mkdir(parents=True, exist_ok=True)
    sp = dir / f"{sweep.out_name(patches, seed)}.summary.json"
    sp.write_text(json.dumps({
        "probe": "p1c_erasure_probe", "argv": ["--fake"], "seed": seed,
        "ticks": 3000, "sample": 100, "extinct_at": extinct,
        "readback": {}, "per_sample": [{"n_agents": 4000, "pos_frac_read_moore": 0.4}],
        "pooled": {"pos_frac_read_moore": pos, "pos_n_dist_pairs": 1000,
                   "pos_d_med": 3.5,
                   "pep": {"frac_read_moore_pos": 0.45, "n_pos": 800, "d_med_pos": 1.0}},
    }), encoding="utf-8")


def test_aggregate_rc_and_json(tmp_path, capsys):
    d1 = tmp_path / "p1c_density"
    for s, v in zip(sweep.SEEDS, [0.40, 0.41, 0.42, 0.43]):
        _write_fake_summary(d1, 850, s, v)
    a = _ns(data_dir=str(d1), erase_dir=str(tmp_path / "p1c_erase"),
            out_md=str(tmp_path / "t.md"), json_out=str(tmp_path / "t.json"))
    assert sweep.cmd_aggregate(a) == 0
    js = json.loads((tmp_path / "t.json").read_text(encoding="utf-8"))
    assert js["candidate_band"] == "D1" and js["first_le_050_band"] == "D1"
    assert any("缺数据" in b["verdict"] for b in js["bands"])     # D2/D3 未到位

    # S4 fail-loud ⇒ rc=3（上板，不得静默）
    _write_fake_summary(d1, 850, 207, float("nan"))
    assert sweep.cmd_aggregate(a) == 3

    # 🔴 未跑档缺数据 ⇒ rc=4（fail-loud，不得以复用/假阳性放行）
    d2 = tmp_path / "empty" / "p1c_density"
    a2 = _ns(data_dir=str(d2), erase_dir=str(tmp_path / "nope"),
             out_md=None, json_out=str(tmp_path / "t2.json"))
    assert sweep.cmd_aggregate(a2) == 4


# ---------------- ⑤ 起跑/复现门安全门 ----------------

def test_run_matrix_requires_authorization(monkeypatch):
    monkeypatch.setattr(sweep.sys, "argv", ["x", "--run-matrix"])
    with pytest.raises(SystemExit, match="authorized-by"):
        sweep.main()


def test_run_matrix_dry_run_no_spawn(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("dry-run 不得起子进程")
    monkeypatch.setattr(sweep.subprocess, "call", _boom)
    a = _ns(python="PY", data_dir=str(tmp_path / "pd"), only_patches=None,
            only_seeds=None, dry_run=True, force=False, allow_dirty=True,
            authorized_by=None)
    assert sweep.cmd_run_matrix(a) == 0


def test_smoke_gate_missing_archive(tmp_path):
    a = _ns(python="PY", erase_dir=str(tmp_path / "nope"),
            gate_out=str(tmp_path / "g.csv"), force=False)
    with pytest.raises(SystemExit, match="复现门"):
        sweep.cmd_smoke_gate(a)


def test_smoke_gate_refuses_overwrite(tmp_path):
    (tmp_path / "p1c_erase").mkdir()
    (tmp_path / "p1c_erase" / "p1c_erase_s207_t3000.csv").write_text("tick\n", encoding="utf-8")
    g = tmp_path / "g.csv"
    g.write_text("x", encoding="utf-8")
    a = _ns(python="PY", erase_dir=str(tmp_path / "p1c_erase"),
            gate_out=str(g), force=False)
    with pytest.raises(SystemExit, match="不覆盖"):
        sweep.cmd_smoke_gate(a)


# ---------------- ⑥ 探针 home-range：默认关不变 + 开启口径 ----------------

class _FakeWorld:
    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols

    def flat_to_rc(self, flat):
        f = np.asarray(flat)
        return f // self.cols, f % self.cols


class _FakeEng:
    def __init__(self, world, ids, flat):
        self.world = world
        self._id = np.asarray(ids, np.int64)
        self._flat = np.asarray(flat, np.int64)


def _hr_dist(w, r1, c1, r2, c2):
    t = _HomeRangeCore(w)
    return t._dist(r1, c1, r2, c2)


def test_home_range_core_geometry_anchors():
    """口径锚（s3 R308 逐式）：同列跨 10 行 ⇒ 10.0 格；经度环绕 119→0 ⇒ ≈1 格。

    变异检查：换成平面列差（忘环绕）⇒ 119 ⇒ 断言必红。
    """
    w = _FakeWorld(60, 120)
    d = _hr_dist(w, 30, 60, 40, 60)
    assert abs(d - 10.0) < 1e-9, f"经线 10 行应=10.0 格，实得 {d}"
    d2 = _hr_dist(w, 30, 119, 30, 0)
    assert 0.9 < d2 <= 1.0 + 1e-6, f"接缝单格应≈1 格宽，实得 {d2}"
    naive = abs(119 - 0)                      # 平面列差的错误实现
    assert not (0.9 < naive <= 1.0 + 1e-6)    # 冻结断言对错误实现必红


def test_home_range_core_tracks_move_patch_away():
    w = _FakeWorld(60, 120)
    eng = _FakeEng(w, [1, 2], [0, 60 * 30 + 119])
    labels = np.full(w.rows * w.cols, -1, np.int64)
    labels[0] = 0                                # 个体1 出生在斑块 0
    t = _HomeRangeCore(w)
    t.update(eng, labels)
    s = t.stats()
    assert s["hr_n"] == 2 and s["hr_max_dist_median"] == 0.0
    assert s["hr_patch_median"] == 0.5           # 个体1 见过 1 斑，个体2 见过 0
    assert abs(s["hr_away_median"] - 0.5) < 1e-9
    # 个体1 移到 (30, 0)、个体2 死亡 ⇒ 只留活体，max_dist 记 30 行距离，move 累计
    eng2 = _FakeEng(w, [1], [30 * 120 + 0])
    t.update(eng2, labels)
    s2 = t.stats()
    assert s2["hr_n"] == 1
    expect = _hr_dist(w, 0, 0, 30, 0)
    assert abs(s2["hr_max_dist_median"] - expect) < 1e-9
    assert abs(s2["hr_move_median"] - expect) < 1e-9   # 单跳：max=move（直线累计）


def _run_probe_main(tmp_path, extra: list[str], monkeypatch) -> Path:
    """微缩装置跑探针 main（--smoke 显式声明非判别档）。"""
    out = tmp_path / f"p_{'-'.join(extra).replace('--', '')}.csv"
    argv = ["p1c", "--rows", "60", "--cols", "120", "--patches", "8",
            "--pop", "50", "--ticks", "120", "--sample", "30", "--smoke",
            "--out", str(out)] + extra
    monkeypatch.setattr(sweep.sys, "argv", argv)
    from experiments import p1c_erasure_probe as probe
    assert probe.main() == 0
    return out


def test_probe_home_range_off_byte_identical_columns(tmp_path, monkeypatch):
    out = _run_probe_main(tmp_path, [], monkeypatch)
    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert "hr_" not in header
    summary = json.loads(out.with_name(out.stem + ".summary.json")
                         .read_text(encoding="utf-8"))
    assert "home_range" not in summary["readback"]
    assert "hr_run_median" not in summary["pooled"]
    assert not any(k.startswith("hr_") for k in summary["per_sample"][0])


def test_probe_home_range_on_adds_columns(tmp_path, monkeypatch):
    out = _run_probe_main(tmp_path, ["--home-range"], monkeypatch)
    lines = out.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    assert header[-len(_HR_COLS):] == _HR_COLS          # 追加在表尾，不打乱既有列序
    summary = json.loads(out.with_name(out.stem + ".summary.json")
                         .read_text(encoding="utf-8"))
    assert summary["readback"]["home_range"] is True
    hr = summary["pooled"]["hr_run_median"]
    assert hr["hr_max_dist_median"] >= 0.0              # run 级=逐窗中位
    assert any(k.startswith("hr_") for k in summary["per_sample"][0])
