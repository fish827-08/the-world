"""RPW-KNOB 验收：`s3_memory_probe` 的 `--rep-w` / `--rep-w-solo` 两开关。

口径（PI 10-06 派工①／鱼令③；卡 RPW-KNOB）：
  ① 默认档（不传 / `--rep-w 0.0`）⇒ **逐字节等于旧版**：CSV 除计时列零格差，
     summary `meta` 两键被摘除（`_summary_meta` 第五条件），banner 不追加；
  ② 非负校验：`--rep-w < 0` ⇒ CLI exit 2 **且** `_build_fresh_run` ValueError
     （构造后赋值不触发 `__post_init__` 的 `assert`（config.py:660）⇒ 复刻，F1 同型）；
  ③ 🔴 空转禁令：`--rep-w > 0` ∧ 无 `--rep-w-solo` ⇒ CLI exit 2 + 装配期 RuntimeError。
     根因 = `sphere_engine.py:3999` `rep_w = reputation_weight if enabled else 0.0`，
     而本探针装置是 A0 口径（`info_structure.enabled=False`）⇒ 只设档位跑出三组同值 run；
  ④ 读回守：生效值取自**构造后的引擎 config**（不看 CLI／入参）⇒ 删接线／回退总开关当场炸；
  ⑤ solo 形制装配校验：`enabled=True` ＋ 六项中性字段（`_SOLO_OFF_FIELDS`）逐项回读；
  ⑥ 续跑指纹：`reputation_weight` + `info_structure_enabled` 进 `_config_fingerprint_check`
     ⇒ 换档／换形制续跑硬报错；
  ⑦ R277 回显：改档 ⇒ summary run 级四键在场（`rep_w_declared` /
     **`reputation_weight_effective`**（键名照砚预注册-E057 锁定文本）/
     `info_structure_enabled_effective` / `rep_w_solo`）＋ banner 追加；默认档一律不在场；
  ⑧ 档差真实存在（微型实证）：solo 下 `rep_w=0.0` 与 `0.5` 引擎状态摘要**不同**；
     而 solo∧`rep_w=0.0` 与 s3 现装置摘要**相同** ⇒ solo 形制零扰动（对照臂可与 E-056 配对）。

运行成本：②③④⑤⑥⑧ 全走 in-process 建引擎（不步进或只步进 60t 微型档）；
①⑦ 各两条 subprocess 微缩档（40×80 / 30t）。
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

sys.path.insert(0, str(REPO))
from experiments import s3_memory_probe as P  # noqa: E402

# 微缩档（同 `_rerun_logs/lh_rpw/tier_diff*.py` 形制；不跑批 R117）
DEV = ["--rows", "40", "--cols", "80", "--patches", "20", "--pop", "60",
       "--sample", "20", "--arms", "off"]
IGNORE = {"ms_per_tick"}


def _run(args: list[str], out: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    return subprocess.run(
        [sys.executable, "-u", "-m", PROBE, *args, "--out", str(out)],
        cwd=str(REPO), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace")


def _rows(p: Path) -> list[dict]:
    with open(p, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _diff(ref: list[dict], got: list[dict]) -> list[tuple]:
    key = lambda r: (r["seed"], r["arm"], int(float(r["tick"])))  # noqa: E731
    ref, got = sorted(ref, key=key), sorted(got, key=key)
    if len(ref) != len(got):
        return [("__n__", len(ref), len(got))]
    return [(r["seed"], r["arm"], r["tick"], c, a[c], b[c])
            for a, b in zip(ref, got) for c in a
            if c not in IGNORE and a[c] != b[c]]


def _summary(out: Path) -> dict:
    with open(out.with_suffix(".summary.json"), "r", encoding="utf-8") as jf:
        return json.load(jf)


def _build(seed=902, ticks=None, rep_w=P._REP_W_DEFAULT, rep_w_solo=False, **kw):
    """走探针官方构建路径；`ticks` 非 None ⇒ 顺手步进（微型档）。"""
    _state, eng = P._build_fresh_run(seed, 40, 80, 60, 20, 1.195, False,
                                     0.0, 0.0, False,
                                     rep_w=rep_w, rep_w_solo=rep_w_solo, **kw)
    for _ in range(ticks or 0):
        eng.step()
    return eng


def _digest(eng) -> tuple:
    n = int(len(eng._id))
    return (n, int(eng._tick), int(eng._flat[:n].sum()),
            round(float(eng._energy[:n].sum()), 9),
            round(float(eng._stomach[:n].sum()), 9),
            round(float(eng._genes[:n].sum()), 9),
            round(float(eng.resources._capacity.sum()), 9))


# ------------------------------------------------------------------ ① 默认档逐字节等价
def test_t1_default_csv_byte_equivalent(tmp_path):
    """不传 rep 参数 vs 显式 `--rep-w 0.0` ⇒ CSV 零格差 + meta 无两键 + banner 不追加。"""
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    ra = _run([*DEV, "--ticks", "30", "--seeds", "902"], a)
    rb = _run([*DEV, "--ticks", "30", "--seeds", "902", "--rep-w", "0.0"], b)
    assert ra.returncode == 0 and rb.returncode == 0, ra.stderr + rb.stderr
    assert _diff(_rows(a), _rows(b)) == []
    for out in (a, b):
        meta = _summary(out)["meta"]
        assert "rep_w" not in meta and "rep_w_solo" not in meta
        assert "reputation_weight_effective" not in _summary(out)["runs"][0]
    # banner 只在**改档**时追加；`--rep-w 0.0` = 默认值 ⇒ 两条日志逐字节同形
    assert "rep_w=" not in ra.stdout
    assert "rep_w=" not in rb.stdout


# ------------------------------------------------------------------ ② 非负校验（双路径）
def test_t1b_summary_meta_fifth_condition():
    """`_summary_meta` 第五摘除条件：默认档两键**同生同灭**；改档 ⇒ 两键在场；不改 namespace。"""
    def _ns(**kw):
        d = dict(rd_mode=P._RD_MODE_DEFAULT, alphabet=P._ALPHABET_DEFAULT,
                 m0_instruments=False, m0_emit_window_frac=0.25,
                 rep_w=P._REP_W_DEFAULT, rep_w_solo=False)
        d.update(kw)
        return argparse.Namespace(**d)

    ns = _ns()
    meta = P._summary_meta(ns)
    assert "rep_w" not in meta and "rep_w_solo" not in meta
    assert getattr(ns, "rep_w") == P._REP_W_DEFAULT, "🔴 pop 到了原 namespace（拷贝语义破了）"
    m2 = P._summary_meta(_ns(rep_w=0.05))
    assert m2["rep_w"] == 0.05 and m2["rep_w_solo"] is False
    m3 = P._summary_meta(_ns(rep_w_solo=True))
    assert m3["rep_w"] == P._REP_W_DEFAULT and m3["rep_w_solo"] is True


def test_t2_negative_tier_rejected():
    with pytest.raises(ValueError, match="必须 ≥ 0"):
        _build(rep_w=-0.1)


def test_t2b_negative_tier_cli_exit2(tmp_path):
    out = tmp_path / "n.csv"
    r = _run([*DEV, "--ticks", "10", "--seeds", "902", "--rep-w", "-1"], out)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "必须 ≥ 0" in r.stderr
    assert not out.exists(), "🔴 前置校验必须**开 CSV 之前**拦（不留半成品）"


# ------------------------------------------------------------------ ③ 空转禁令（双路径）
def test_t3_inert_tier_raises_at_build():
    """档位 >0 而无 solo ⇒ 引擎生效值 0 ⇒ 装配期 RuntimeError（绝不空跑出同值三档）。"""
    with pytest.raises(RuntimeError, match="空转禁令"):
        _build(rep_w=0.5)


def test_t3b_inert_tier_cli_exit2(tmp_path):
    out = tmp_path / "i.csv"
    r = _run([*DEV, "--ticks", "10", "--seeds", "902", "--rep-w", "0.05"], out)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "必须配 --rep-w-solo" in r.stderr
    assert not out.exists()


# ------------------------------------------------------------------ ④⑤ solo 形制落地
def test_t4_solo_assembly_lands():
    eng = _build(rep_w=0.2, rep_w_solo=True)
    iscfg = eng.config.info_structure
    assert bool(iscfg.enabled) is True
    assert float(iscfg.reputation_weight) == pytest.approx(0.2)
    for k, v in P._SOLO_OFF_FIELDS:
        assert getattr(iscfg, k) == v, f"solo 形制字段 {k} 未关到中性"


def test_t5_readback_mismatch_raises(monkeypatch):
    """引擎构造期把档位改值 ⇒ 读回守（生效值 vs 声明值）当场炸，不静默出混档数据。"""
    real = P.SphereEngine

    def _factory(c):
        eng = real(c)
        eng.config.info_structure.reputation_weight = 0.3   # 声明 0.2 ⇒ 不符
        return eng

    monkeypatch.setattr(P, "SphereEngine", _factory)
    with pytest.raises(RuntimeError, match="读回不符"):
        _build(rep_w=0.2, rep_w_solo=True)


def test_t6_solo_assembly_failure_raises(monkeypatch):
    """带 `rep_w_solo` 但总开关／中性字段没装配上 ⇒ 独立第三守拦下（`rep_w=0` 时前两条不触发）。"""
    real = P.SphereEngine

    def _factory(c):
        eng = real(c)
        eng.config.info_structure.enabled = False            # solo 装配被回退
        return eng

    monkeypatch.setattr(P, "SphereEngine", _factory)
    with pytest.raises(RuntimeError, match="solo 形制装配失败"):
        _build(rep_w=0.0, rep_w_solo=True)


# ------------------------------------------------------------------ ⑥ 续跑指纹
def test_t7_fingerprint_blocks_tier_change():
    eng = _build(seed=207, rep_w=0.2, rep_w_solo=True)
    kw = dict(mem_on=False, bg_low_prod_frac=0.0, bg_low_cap_mult=0.0,
              weight_gene=False, rd_on=True, alphabet="16")
    # 同档同形制 ⇒ 不报（默认参数即旧口径）
    P._config_fingerprint_check(eng, 207, 40, 80, 60, 20, 1.195,
                                rep_w=0.2, rep_w_solo=True, **kw)
    with pytest.raises(ValueError, match="reputation_weight"):
        P._config_fingerprint_check(eng, 207, 40, 80, 60, 20, 1.195,
                                    rep_w=0.05, rep_w_solo=True, **kw)
    with pytest.raises(ValueError, match="info_structure_enabled"):
        P._config_fingerprint_check(eng, 207, 40, 80, 60, 20, 1.195,
                                    rep_w=0.2, rep_w_solo=False, **kw)


# ------------------------------------------------------------------ ⑦ R277 回显
def test_t8_summary_echo_keys_when_tier_changed(tmp_path):
    out = tmp_path / "s.csv"
    r = _run([*DEV, "--ticks", "20", "--seeds", "902",
              "--rep-w", "0.05", "--rep-w-solo"], out)
    assert r.returncode == 0, r.stdout + r.stderr
    s = _summary(out)
    for row in s["runs"]:
        assert row["rep_w_declared"] == pytest.approx(0.05)
        assert row["reputation_weight_effective"] == pytest.approx(0.05)
        assert row["info_structure_enabled_effective"] is True
        assert row["rep_w_solo"] is True
    assert s["meta"]["rep_w"] == pytest.approx(0.05)
    assert s["meta"]["rep_w_solo"] is True
    assert "rep_w=0.05" in r.stdout and "reputation_weight_effective=0.05" in r.stdout


# ------------------------------------------------------------------ ⑧ 档差真实存在
@pytest.mark.parametrize("seed", [903])
def test_t9_tier_changes_dynamics(seed):
    solo0 = _digest(_build(seed=seed, ticks=60, rep_w=0.0, rep_w_solo=True))
    solo5 = _digest(_build(seed=seed, ticks=60, rep_w=0.5, rep_w_solo=True))
    legacy = _digest(_build(seed=seed, ticks=60))          # s3 现装置（A0 口径）
    assert solo0 != solo5, "🔴 solo 下三档跑出同值 ⇒ 旋钮没接线（档差消失）"
    assert solo0 == legacy, "🔴 solo∧rep_w=0 应逐位等于现装置（否则对照臂不可比）"
