"""14.10 T2/T3 定档接线单测（`[云端开发·云启]`，2026-09-26；派工单 T2/T3）。

覆盖（对应派工单验收）：
  T2-① `rescale_config` **默认 k = DEFAULT_K = 2.5**（不传 k ⇒ 与显式 2.5 逐字段一致）
  T2-② `cap_kp_report` 报满「每 tick 概率」**六字段**；k=2.5 下 CAP 类 `k·p ≤ 0.5`（k=5 ⇒ 越线可判）
  T2-③ 定档常量自洽（`DEFAULT_D == round(2400 / DEFAULT_K)`）
  T3-① `apply_speed_std` ⇒ `gain = v_max`、`subdiv = SUBDIV_STD`（**80**，R213 §四）；**顶格% ≡ 0 是构造保证**（gain = cap）
  T3-② `R = v_max × subdiv` 返回值 + `ladder_warn` 边界（R<5 报警 / R≥5 静默）
  T3-③ **端到端默认值**（DEL-3 精神，防「写了没生效」）：两个调用点脚本真跑一次，断言新规则
        出现在**实际输出**里（`time_compress_check --ds` 默认 2400,960；`steady_k_probe` gain=subdiv 定档）
  C7   : 本模块**不改任何默认档**（默认 digest 由 `self_audit digest` 的 12 个文件钉死，此处不重复抄）

🔴 DEL-8 变异的**敏感断言** = 本文件（改坏接线 ⇒ 必红）；变异证据见 `_rerun_logs/t2t3_smoke/`。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.scaling_rescale import (  # noqa: E402
    CAP_FIELDS,
    DEFAULT_D,
    DEFAULT_K,
    SUBDIV_STD,
    apply_post_build,
    apply_speed_std,
    cap_kp_report,
    ladder_warn,
    rescale_config,
)
from simulation.config import SimConfig  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402

PY = sys.executable


def _snapshot(c: SimConfig) -> dict:
    """取一组「重标会碰到」的字段快照（用于 k 等价/恒等比较）。"""
    return {
        "rotation_period": c.light.rotation_period,
        "regrowth_rate": c.resources.regrowth_rate,
        "base_metabolism": c.organisms.base_metabolism,
        "attack_prob_coef": c.predation.attack_prob_coef,
        "baseline_rate": c.pleasure.baseline_rate,
        "valence_decay": c.pleasure.valence_decay,
        "duration_ticks": c.signals.duration_ticks,
        "repro_cooldown_gene_scale": c.organisms.repro_cooldown_gene_scale,
        "giveup": c.ars.giveup,
        "subdiv": c.subpos.subdiv,
        "speed_gain": c.subpos.speed_gain,
        "speed_max": c.subpos.speed_max,
    }


# ─────────────────────────── T2 ───────────────────────────

def test_default_k_is_25_and_matches_explicit():
    """不传 k ⇒ 定档 2.5，且与显式 DEFAUT_K 逐字段一致（自证：昼夜 960）。"""
    a = SimConfig(seed=1)
    notes_a = rescale_config(a)
    b = SimConfig(seed=1)
    notes_b = rescale_config(b, DEFAULT_K)
    assert notes_a == notes_b
    assert notes_a["k"] == DEFAULT_K == 2.5
    assert DEFAULT_D == round(2400 / DEFAULT_K) == 960
    assert _snapshot(a) == _snapshot(b)
    # 定档可自证的两点：昼夜 = 960 tick；CAP 概率恰落在 0.5 线上（无余量）
    assert a.light.rotation_period == 960
    assert a.predation.attack_prob_coef == 0.5
    assert a.signals.duration_ticks == 20          # 50 / 2.5


def test_k1_is_identity():
    """k=1 ⇒ 不改（= 配置默认值）——「默认 = 现状」纪律。"""
    ref = _snapshot(SimConfig(seed=1))
    c = SimConfig(seed=1)
    notes = rescale_config(c, 1.0)
    assert notes["k"] == 1.0
    assert _snapshot(c) == ref


def test_cap_report_six_fields_and_cap_line():
    """六字段报满；k=2.5 无 CAP 饱和；k=5 时 CAP 越线（可判性）。"""
    rows = cap_kp_report()
    assert [r["field"] for r in rows] == [f"{g}.{n}" for g, n in CAP_FIELDS]
    assert len(rows) == 6
    cap_rows = [r for r in rows if r["rule"] == "CAP"]
    assert len(cap_rows) == 1 and cap_rows[0]["field"] == "predation.attack_prob_coef"
    assert cap_rows[0]["kp"] == 0.5                # 定档 k=2.5 恰在线上 ⇒ k>2.5 即饱和
    assert all(not r["sat"] for r in rows)
    # 越线可判（否则判据形同虚设）
    rows5 = cap_kp_report(k=5.0)
    assert [r["sat"] for r in rows5 if r["rule"] == "CAP"] == [True]


# ─────────────────────────── T3 ───────────────────────────

def test_apply_speed_std_is_constructively_clip_free():
    """gain = v_max、subdiv = SUBDIV_STD（80）；R 返回值；R<5 可判。"""
    c = SimConfig(seed=1)
    r = apply_speed_std(c.subpos, 0.25)
    assert (c.subpos.speed_max, c.subpos.speed_gain, c.subpos.subdiv) == (
        0.25, 0.25, SUBDIV_STD)
    assert r == 0.25 * SUBDIV_STD == 20.0          # SUBDIV_STD = 80（R213 §四 合并裁定）
    # 构造保证：gain = cap ⇒ speed = clip(mob_eff×gain, 0, cap) = mob_eff×cap ≤ cap ⇒ 永不撞顶
    assert c.subpos.speed_gain <= c.subpos.speed_max
    # 定档耦合：subdiv 固定 SUBDIV_STD ⇒ R = SUBDIV_STD·v_max；R<5 ⇒ 档数不足（报警可判）
    assert ladder_warn(4.9) is not None
    assert ladder_warn(5.0) is None


def test_dual_path_bitwise_at_k25():
    """T2-① 直译「k=2.5 下 C7 对拍绿」：**重标后**双路径（Python vs Rust）仍逐位一致。

    小世界 + 短昼夜（8×12 / 240 tick 昼夜）⇒ 300 tick 内出现死亡与出生（守住热路径）。
    """
    def _mk(seed: int) -> tuple[SimConfig, dict]:
        c = SimConfig(seed=seed)
        c.world.rows, c.world.cols = 8, 12
        c.light.rotation_period = 600          # 重标后 = 240（÷2.5）
        c.population.initial_count = 60
        return c, rescale_config(c)            # 不传 k ⇒ 定档 2.5

    py, n_py = _mk(7)
    rs, n_rs = _mk(7)
    rs.simulation.use_sim_core = True
    e_py, e_rs = SphereEngine(py), SphereEngine(rs)
    apply_post_build(e_py, n_py)
    apply_post_build(e_rs, n_rs)
    for _ in range(300):
        s_py, s_rs = e_py.step(), e_rs.step()
        assert s_rs == s_py, f"tick {s_rs.tick} 不一致：py={s_py} rs={s_rs}"
    assert e_py.alive_count() == e_rs.alive_count()
    for f in ("_flat", "_energy", "_stomach", "_genes", "_age",
              "_generation", "_parent", "_repro_cooldown", "_id"):
        np.testing.assert_array_equal(
            getattr(e_rs, f), getattr(e_py, f), err_msg=f"{f} 逐位不相等")
    np.testing.assert_array_equal(
        e_rs.resources._grid, e_py.resources._grid, err_msg="resources._grid 逐位不相等")


def test_obsolete_helpers_removed():
    """R204 §二 否决物不得回流（`gain = v_max/ā`）。"""
    import experiments.scaling_rescale as sr
    assert not hasattr(sr, "gain_for")
    assert not hasattr(sr, "subdiv_for")


def test_call_site_defaults_end_to_end():
    """DEL-3 精神：脚本**真跑**，新规则要出现在实际输出里（防「写了没生效」）。

    用最小世界（60×120 / 1 tick / pop 50）压成本；定档一致性看打印的速度行。
    """
    # ① time_compress_check：默认对照臂 = k=1 与 定档 k=2.5（D=2400,960）
    #    🔴 R98 族：子进程按 R98 纪律把 stdout 钉成 UTF-8 ⇒ 本机（GBK 控制台）父进程
    #       必须显式 decoding ⇒ 否则 reader 线程 UnicodeDecodeError ⇒ stdout=None（合并时补）。
    r1 = subprocess.run(
        [PY, "experiments/time_compress_check.py", "--days", "0.05",
         "--pop", "50", "--subpos", "on"],
        cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=600)
    assert r1.returncode == 0, r1.stderr[-800:]
    assert "[2400, 960]" in r1.stdout, r1.stdout[-500:]
    assert "k = [1.0, 2.5]" in r1.stdout, r1.stdout[-500:]

    # ② steady_k_probe：gain 默认 = speed_max、subdiv 默认 = SUBDIV_STD（都出现在速度行）
    r2 = subprocess.run(
        [PY, "experiments/steady_k_probe.py", "--rows", "60", "--cols", "120",
         "--patches", "30", "--pop", "50", "--ticks", "1", "--sample", "1",
         "--seeds", "42", "--subpos", "on", "--out", ""],
        cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=600)
    assert r2.returncode == 0, r2.stderr[-800:]
    assert f"gain=0.25 subdiv={SUBDIV_STD}" in r2.stdout, r2.stdout[-500:]