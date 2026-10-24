"""13.6 S2 四读数 —— 食物利用率 / 斑块访问率 / 在斑块占比 / GUD 方差（2026-09-24）。

依据：R193 §四（S2 定义）+ R194 §S2（判据「从产物可自证（R190/R191 的数字可复算）」）
实现：`experiments/a4_verify_capacity.py` 的 `SpatialReadings` + `SPATIAL_NOTE`（纯观测）

覆盖（S2 判据 + 协议 DEL-3/4/6/7/8）：
  ① **纯观测**（DEL-4 家族）：带读数跑 50 tick ⇒ 与不带的**逐位一致**（零 RNG、零行为改变）
  ② **口径正确**（R190/R191 可复算）：三项口径各自对着**权威实现**核算
     · 利用率 = 取食质量 ÷（Σ名义再生 × tick）—— 与 `tools/p135_verdict.py` 同式
     · 访问率 / 在斑块占比 —— 与 `tools/spatial_diag.py` 同式（同一输入 ⇒ 同值）
  ③ **GUD 定义**（本线首次落码）：手工构造存量/容量 ⇒ 逐值核对 mean/var（ddof=1）
  ④ **DEL-7 缺失值**：灭绝（P=0）⇒ `on_patch_frac` / `gud_*` = **None**（禁写 0）；
     单格占据 ⇒ `gud_var` = None 而 `gud_mean` 有值（样本 < 2）
  ⑤ **DEL-6 运行期不变量（t>0）**：`patch_visit_frac` **单调不减** + 首采样点即 > 0
     （t=0 ⇒ 利用率 None；t>0 ⇒ 有值）
  ⑥ **DEL-8 变异敏感**：`sr.observe` 不调用 ⇒ ①③④ 的对应断言必须变红
     （见 `_rerun_logs/terrain_s2_smoke/del8_s2_*.txt`）
  ⑦ **侧车续跑**：visited 与取食能量**跨段结转**；侧车缺失 ⇒ `spatial_carry_ok=False` 自曝
  ⑧ **端到端**：a4 子进程 ⇒ `result.spatial` 键齐 + CSV 六列在册且 t>0 非空
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.a4_verify_capacity import (  # noqa: E402
    SPATIAL_NOTE, SpatialReadings, build,
)

PY = sys.executable
KEYS = ("food_util_frac", "patch_visit_frac", "on_patch_frac", "gud_var")


def _engine(seed: int = 42, *, distribution: str = "patchy", patch_mult=None,
            bgzero: bool = True):
    return build("on", True, seed, 100, max_count=3240, distribution=distribution,
                 patch_regrowth_mult=patch_mult, bg_production_zero=bgzero,
                 photo_max=0.0 if bgzero else None)


def _digest(e) -> tuple[int, float]:
    return (int(e._flat.sum()), round(float(e._energy.sum()), 6))


def _advance(e, ticks: int, *, sr: "SpatialReadings | None" = None) -> None:
    """推进引擎；`sr` 非空 ⇒ 同时驱动读数（证明读数不改行为）。"""
    for t in range(1, ticks + 1):
        e.step()
        if sr is not None:
            sr.observe(e)
            sr.sample(e, t)


# ----------------------------------------------------- ① 纯观测（逐位一致）

def test_readings_are_pure_observation_bit_identical():
    """带读数 vs 不带读数：50 tick 后逐位一致（读数必须零副作用）。

    ⚠️ **F-D2 陷阱**：D2 的感知噪声走**进程级共享**的全局 `np.random`
    ⇒ 必须"先跑完 a **再**构造 b"（构造时 `np.random.seed(seed)` 重置起点）；
    同时构造两个引擎会互相消耗全局流 ⇒ 假失败（`test_d18_response.py` 已记录同一坑）。
    """
    a = _engine()
    _advance(a, 50)
    b = _engine()                     # 此刻 np.random 被重新 seed ⇒ 与 a 起点一致
    sr = SpatialReadings(b)
    _advance(b, 50, sr=sr)
    assert _digest(a) == _digest(b)
    np.testing.assert_array_equal(a._flat, b._flat)
    np.testing.assert_array_equal(a._energy, b._energy)
    assert a.rng_draws == b.rng_draws, "读数吃掉了 RNG（必须零新增抽取）"


# ----------------------------------------------------- ② 口径正确（可复算）

def test_food_util_matches_p135_verdict_formula():
    """利用率 =（账本取食能量 ÷ 净吸收）÷（Σ名义再生 × tick）—— 与 R190 工具同式。"""
    e = _engine()
    sr = SpatialReadings(e)
    for _ in range(300):
        e.step()
        sr.observe(e)
    t = int(e._tick)
    led = e.energy_ledger()["global"]
    eff = float(e.config.organisms.eat_efficiency)
    ah = float(e.config.organisms.assim_herb)
    manual = (float(led["intake_forage_sum"]) / (eff * ah)) / (sr.sigma_regen * t)
    assert sr.food_util_frac(e, t) == pytest.approx(manual, rel=1e-12)
    # t=0 ⇒ None（分母为 0 ⇒ 未适用，不是 0）
    assert sr.food_util_frac(e, 0) is None


def test_visit_and_on_patch_match_spatial_diag_formula():
    """访问率/在斑块占比：与 `tools/spatial_diag.py` 同式（同输入 ⇒ 同值）。"""
    e = _engine()
    sr = SpatialReadings(e)
    n = e.world.n_cells
    patch = np.asarray(e.resources._patch_mask, dtype=bool)
    visited = np.zeros(n, dtype=bool)
    for _ in range(60):
        e.step()
        sr.observe(e)
        P = len(e._flat)
        if P:
            visited[e._flat[:P]] = True          # 参照实现（逐字照抄 spatial_diag 第 64 行）
    P = len(e._flat)
    ref_vis = float(visited[patch].sum()) / int(patch.sum())
    ref_on = float(np.mean(patch[e._flat[:P]])) if P else None
    got = sr.sample(e, int(e._tick))
    assert got["patch_visit_frac"] == pytest.approx(ref_vis, rel=1e-12)
    assert got["on_patch_frac"] == pytest.approx(ref_on, rel=1e-12)
    # 口径源必须随产物自证（C4/E 类：不留"这数是什么口径"的悬空）
    assert "R190" in SPATIAL_NOTE and "R191" in SPATIAL_NOTE and "gud_var" in SPATIAL_NOTE


# ----------------------------------------------------- ③ GUD 定义（手工核对）

def test_gud_mean_and_var_by_hand():
    """手工构造：3 只个体分别落在 stock/cap = 0.2 / 0.5 / 1.0 的**斑块格**
    ⇒ mean=0.5667、var=0.16333（ddof=1）。"""
    e = _engine()
    sr = SpatialReadings(e)
    assert len(e._id) >= 3
    e._id = e._id[:3]                      # 只留 3 只（读数以 len(_id) 为准）
    patch = np.asarray(e.resources._patch_mask, dtype=bool)
    cells = np.asarray(np.flatnonzero(patch)[:3], dtype=np.int64)   # 容量 > 0 的格
    e._flat[:3] = cells
    for c, frac in zip(cells, (0.2, 0.5, 1.0)):
        e.resources._grid[c] = float(e.resources._capacity[c]) * frac
    got = sr.sample(e, 10)
    vals = np.array([0.2, 0.5, 1.0])
    assert got["gud_mean"] == pytest.approx(float(vals.mean()), rel=1e-9)
    assert got["gud_var"] == pytest.approx(float(vals.var(ddof=1)), rel=1e-9)
    assert got["gud_var"] == pytest.approx(0.1633333, rel=1e-6)


def test_gud_excludes_zero_capacity_cells():
    """bgzero 世界：背景格容量 0 ⇒ 其 GUD 无定义 ⇒ **剔除**（不是记 0）。"""
    e = _engine()
    sr = SpatialReadings(e)
    patch = np.asarray(e.resources._patch_mask, dtype=bool)
    bg = int(np.flatnonzero(~patch)[0])
    pc = int(np.flatnonzero(patch)[0])
    assert float(e.resources._capacity[bg]) == 0.0
    e._id = e._id[:2]
    e._flat[:2] = np.array([bg, pc], dtype=np.int64)
    e.resources._grid[pc] = float(e.resources._capacity[pc]) * 0.4
    got = sr.sample(e, 10)
    assert got["gud_mean"] == pytest.approx(0.4, rel=1e-9), "背景格（cap=0）必须被剔除"


# ----------------------------------------------------- ④ DEL-7 缺失值语义

def test_extinct_gives_none_not_zero():
    """灭绝（P=0）⇒ on_patch/gud = **None**（"没测" ≠ "测出零"；R120/DEL-7）。"""
    e = _engine()
    sr = SpatialReadings(e)
    for _ in range(5):
        e.step()
        sr.observe(e)
    e._id = e._id[:0]                 # 模拟灭绝（读数只看 len(_id)）
    got = sr.sample(e, 10)
    assert got["on_patch_frac"] is None
    assert got["gud_var"] is None and got["gud_mean"] is None


def test_single_occupied_cell_gives_none_variance():
    """样本 < 2 ⇒ 方差未定义（None），但均值仍有值（不是 0）。"""
    e = _engine()
    sr = SpatialReadings(e)
    patch = np.asarray(e.resources._patch_mask, dtype=bool)
    pc = int(np.flatnonzero(patch)[0])
    e._id = e._id[:1]
    e._flat[:1] = np.array([pc], dtype=np.int64)
    e.resources._grid[pc] = float(e.resources._capacity[pc]) * 0.7
    got = sr.sample(e, 10)
    assert got["gud_mean"] == pytest.approx(0.7, rel=1e-9)
    assert got["gud_var"] is None, "单格 ⇒ 方差未定义（None，禁写 0）"


# ----------------------------------------------------- ⑤ DEL-6 运行期不变量

def test_visit_frac_monotone_and_positive_at_t_gt_0():
    """运行期不变量（t>0 采样）：访问率**单调不减**且**早就 > 0**（不是恒 0）。

    🔴 本条即 **DEL-8 的敏感断言**：`sr.observe()` 若不调用 ⇒ 访问率恒 0 ⇒ 立刻红。
    """
    e = _engine()
    sr = SpatialReadings(e)
    seq = []
    for t in range(1, 121):
        e.step()
        sr.observe(e)
        if t % 30 == 0:
            seq.append(float(sr.sample(e, t)["patch_visit_frac"]))
    assert all(b >= a - 1e-12 for a, b in zip(seq, seq[1:])), f"访问率非单调：{seq}"
    assert seq[0] > 0.0, "t=30 时访问率仍为 0 ⇒ observe 未接线（或个体从不落斑块格）"
    assert seq[-1] >= seq[0]


def test_uniform_world_patch_readings_are_none_not_crash():
    """🔴 **uniform 世界**（`_patch_mask is None`）⇒ 斑块类读数 = **None（未适用）**。

    事故原型（2026-09-24 S2 实施中实测）：`_patch_mask` 在 uniform 下是 **None**
    ⇒ `np.asarray(None)` 是 0 维 ⇒ `patch[cells]` 抛 IndexError ⇒ **a4 直接崩**
    （`test_f_r10_f_r12` 的 F-R13 集成 6 例变红）。修法 = 归一成"无斑块"并让
    斑块类读数返回 None（**不是 0**：否则 uniform 批被读成"从不访问斑块"的假阴性）。
    """
    e = _engine(distribution="uniform", bgzero=False)
    sr = SpatialReadings(e)
    assert sr.has_patches is False and sr.area_patch == 0
    for _ in range(10):
        e.step()
        sr.observe(e)
    got = sr.sample(e, 10)
    assert got["on_patch_frac"] is None
    assert got["patch_visit_frac"] is None
    assert got["patch_stock_frac"] is None
    # 利用率/ GUD **仍适用**（不依赖斑块掩码）
    assert got["food_util_frac"] is not None and got["food_util_frac"] > 0.0
    assert got["gud_var"] is not None
    sp = sr.summary(e, 10)
    assert sp["has_patches"] is False and sp["patch_cells"] == 0


# ----------------------------------------------------- ⑦ 侧车续跑

def test_sidecar_carries_visited_and_intake(tmp_path):
    """续跑：visited 与取食能量跨段结转（引擎快照不含账本 ⇒ 必须自带侧车）。"""
    e = _engine()
    sr = SpatialReadings(e, sidecar=tmp_path / "a.spatial.npz")
    for _ in range(80):
        e.step()
        sr.observe(e)
    vis_before = int(sr.visited.sum())
    led_before = float(e.energy_ledger()["global"]["intake_forage_sum"])
    assert vis_before > 0 and led_before > 0.0
    path = sr.save_sidecar(e)
    assert path is not None and path.exists()

    e2 = _engine()                     # 新进程语义：账本从 0 起
    sr2 = SpatialReadings.load_sidecar(tmp_path / "a.spatial.npz", e2)
    assert int(sr2.visited.sum()) == vis_before, "visited 未跨段结转"
    assert sr2.carried_e == pytest.approx(led_before, rel=1e-9), "取食能量未跨段结转"
    assert sr2.carry_ok is True
    # 续跑段的利用率分子 = 结转 + 本段 ⇒ 必须 ≥ 结转本身
    for _ in range(20):
        e2.step()
        sr2.observe(e2)
    assert sr2._intake_e_total(e2) >= led_before


def test_missing_sidecar_self_exposes_carry_false():
    """侧车缺失却在续跑 ⇒ `spatial_carry_ok=False`（**自曝**，不假装全程覆盖）。"""
    e = _engine()
    sr = SpatialReadings(e, sidecar=None, carry_ok=False)
    assert sr.summary(e, 10)["spatial_carry_ok"] is False
    assert "spatial_carry_ok" in sr.summary(e, 10)


# ----------------------------------------------------- ⑧ 端到端

def test_end_to_end_summary_and_csv(tmp_path):
    """a4 子进程 ⇒ `result.spatial` 键齐 + CSV 六列在册（t>0 非空，DEL-6 采样点）。"""
    out = tmp_path / "t.csv"
    cmd = [PY, str(ROOT / "experiments" / "a4_verify_capacity.py"),
           "--mode", "on", "--arm", "main", "--seed", "42", "--ticks", "300",
           "--max-count", "300", "--log-interval", "100", "--snapshot-every", "0",
           "--snapshot-dir", str(tmp_path / "snap"), "--distribution", "patchy",
           "--bgzero", "--patch-mult", "1.195", "--photo-max", "0",
           "--out", str(out)]
    rc = subprocess.call(cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    assert rc == 0
    d = json.loads(out.with_suffix(".summary.json").read_text(encoding="utf-8"))
    sp = d["result"]["spatial"]
    for k in (*KEYS, "sigma_regen_nominal", "patch_cells", "visited_patch_cells",
              "spatial_carry_ok", "note"):
        assert k in sp, f"result.spatial 缺键 {k}"
    # R190/R191 口径的量级抽查（宽区间，只防"接线错成别的量"）
    assert sp["sigma_regen_nominal"] == pytest.approx(406.34, abs=0.5)
    assert sp["patch_cells"] == 817
    assert 0.0 < sp["food_util_frac"] < 1.0
    assert 0.0 <= sp["on_patch_frac"] <= 1.0
    assert sp["patch_stock_frac"] is not None and sp["patch_stock_frac"] > 0.5
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert rows, "CSV 无数据行"
    last = rows[-1]
    for k in (*KEYS, "gud_mean", "patch_stock_frac"):
        assert k in last, f"CSV 缺列 {k}"
    assert last["food_util_frac"] != "", "t>0 的利用率不应为空（DEL-6 采样点）"
    assert float(last["patch_visit_frac"]) > 0.0