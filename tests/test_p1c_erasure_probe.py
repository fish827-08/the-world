# -*- coding: utf-8 -*-
"""P1-c 擦除探针最小测试（R358-T7 = T1 设计 §六 + R363 派工 ①②③④）。

 ① 合成日志确定性单测：已知环深 ⇒ 擦除窗逐位可算（K16=最近16除末4；K64 同）；
 ② 对角邻居 ⇒ 口径 B 可直读=1（与 P1-c `measure()` 同式，接缝守卫锁"不得把索引差当距离"）；
 ③ **变异检查**：把球面距离换成跳数/平面差 ⇒ 冻结断言必须变红（断言有判别力）；
 ④ dm-guard 正/负例：共享列一致通过；一处不等即报（含 tick/列名）。
 ＋ `_id` 非降序 fail-loud（T1 §2.2-3 引擎不变量）。

装置口径与探针同链（make_cfg + apply_post_build + use_sim_core=False）；
世界缩到 60×120（与 P1-c 测试同几何锚：(30,60)→(31,61) 对角、col119↔col0 接缝）。
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from experiments import p1c_erasure_probe as probe
from experiments.p1c_distance_probe import measure as measure_p1c
from experiments.steady_k_probe import make_cfg, apply_post_build
from simulation.sphere_engine import SphereEngine
from world.sphere_world import SphereWorld
from world.subpos import sphere_dist_rows

ROWS, COLS = 60, 120
DIAG_CUR = (30, 60)
DIAG_MEM = (31, 61)
SEAM_CUR = (30, 119)
SEAM_MEM = (30, 0)


# ---------------- ① 合成日志确定性 ----------------

def test_ring_window_deterministic():
    """已知事件序 ⇒ 擦除窗逐位可算：k=16 ⇒ 最近16除末4；k=64 ⇒ 全环除末4。"""
    log = probe._RingLog(depth=64)
    for k in range(20):
        log.add(7, 100 + k)
    assert log.n_events(7) == 20
    assert log.window(7, 16) == list(range(104, 116))   # 环[-16:-4]
    assert log.window(7, 64) == list(range(100, 116))   # 环[-64:-4]
    # 恰 4 条 ⇒ 全被"最近 4"覆盖 ⇒ 擦除集空；5 条 ⇒ 只剩最旧 1 条
    for k in range(5):
        log.add(8, 200 + k)
    assert log.window(8, 16) == [200]
    for _ in range(4):
        log.add(9, 300)
    assert log.window(9, 16) == []
    assert log.window(9, 64) == []
    # 不存在 & 未知 agent ⇒ 空（不炸）
    assert log.window(12345, 16) == []


def test_ring_depth_is_bounded():
    """deque(maxlen) ⇒ 深超环深丢最旧；环深必须 > 4。"""
    log = probe._RingLog(depth=5)
    for k in range(9):
        log.add(1, k)
    assert log.n_events(1) == 5
    assert log.window(1, 64) == [4]          # 环=[4..8]，除末 4 ⇒ 只剩 4
    with pytest.raises(ValueError):
        probe._RingLog(depth=4)


# ---------------- ② 对角邻居：口径 B 可直读=1 ----------------

@pytest.fixture(scope="module")
def eng():
    """微缩引擎（探针同链；纯读，全模块复用）。"""
    c, notes = make_cfg(42, ROWS, COLS, 10, 5, True, 0.125, 0.125, 80, 2.5, 30000, 1.195)
    c.simulation.use_sim_core = False
    c.resource_dynamics.enabled = False
    e = SphereEngine(c)
    apply_post_build(e, notes)
    return e


def _flat(rc):
    return int(np.int64(rc[0]) * COLS + rc[1])


def test_read_moore_diag_and_seam_and_pole(eng):
    """口径 B：对角/接缝/极点带可直读；隔一行不可直读（与 P1-c measure 对照）。"""
    w = eng.world
    cur = np.array([_flat(DIAG_CUR), _flat(SEAM_CUR), 0, COLS - 1], np.int64)
    mem = np.array([_flat(DIAG_MEM), _flat(SEAM_MEM), COLS, 2 * COLS - 1], np.int64)
    got = probe._read_moore(w, cur, mem)
    assert got.tolist() == [True, True, True, True], (
        "对角/接缝邻应判可直读；极点格按相邻纬度带整行处理")
    # 隔一行（同列 rows+2）⇒ 不可直读
    far = probe._read_moore(w, np.array([_flat((30, 60))]),
                            np.array([_flat((32, 60))]))
    assert far.tolist() == [False]
    # 同格也算可直读（mem==cur 分支）
    same = probe._read_moore(w, np.array([_flat((30, 60))]),
                             np.array([_flat((30, 60))]))
    assert same.tolist() == [True]


# ---------------- ③ 变异检查：换实现必须变红 ----------------

def _hop_moore(rf1, cf1, rf2, cf2, world):
    dr = np.abs(np.asarray(rf1, np.float64) - np.asarray(rf2, np.float64))
    dc = np.abs(np.asarray(cf1, np.float64) - np.asarray(cf2, np.float64))
    return np.maximum(dr, dc)


def _dc_no_wrap(rf1, cf1, rf2, cf2, world):
    """只按原始列差（忘经度环绕）—— 典型缺陷实现。"""
    return np.abs(np.asarray(cf1, np.float64) - np.asarray(cf2, np.float64))


def _assert_seam_geometry(d: float) -> None:
    """冻结断言：接缝同行单格（col119↔col0）在球面口径下 ≈1 格宽。"""
    assert 0.9 < d <= 1.0 + 1e-9, f"接缝单格距离应 ≈1 格宽，实得 {d!r}"


def test_mutation_hopcount_goes_red(eng, monkeypatch):
    """变异检查：球面距离换成跳数/平面差 ⇒ 冻结断言必须变红（否则测试恒真）。"""
    w = eng.world
    d_true = float(sphere_dist_rows(30.5, 119.5, 30.5, 0.5, w))
    _assert_seam_geometry(d_true)                       # 正确实现在此绿
    # 探针内实际使用的通路（_dist_and_readable）同样必须满足冻结断言
    cur = np.array([_flat(SEAM_CUR)], np.int64)
    mem = np.array([_flat(SEAM_MEM)], np.int64)
    d_impl, _rd = probe._dist_and_readable(eng, mem, cur)
    _assert_seam_geometry(float(d_impl[0]))

    monkeypatch.setattr(probe, "sphere_dist_rows", _hop_moore)
    d_mut, _ = probe._dist_and_readable(eng, mem, cur)
    with pytest.raises(AssertionError):
        _assert_seam_geometry(float(d_mut[0]))          # Moore 跳数=119 ⇒ 必红

    monkeypatch.setattr(probe, "sphere_dist_rows", _dc_no_wrap)
    d_mut2, _ = probe._dist_and_readable(eng, mem, cur)
    with pytest.raises(AssertionError):
        _assert_seam_geometry(float(d_mut2[0]))         # 忘环绕列差=119 ⇒ 必红
    monkeypatch.undo()

    # 对角锚（与 P1-c 测试同几何）：真实现 ≈√2；跳数 Moore=1 ⇒ 必红
    cur2 = np.array([_flat(DIAG_CUR)], np.int64)
    mem2 = np.array([_flat(DIAG_MEM)], np.int64)
    d_diag, _ = probe._dist_and_readable(eng, mem2, cur2)
    assert abs(float(d_diag[0]) - math.sqrt(2)) < 0.02
    monkeypatch.setattr(probe, "sphere_dist_rows", _hop_moore)
    d_diag_mut, _ = probe._dist_and_readable(eng, mem2, cur2)
    assert not (1.3 < float(d_diag_mut[0]) < 1.5), "跳数=1 不应落在 √2 锚带"


# ---------------- ④ dm-guard 正/负例 ----------------

_DM_COLS = ["tick", "n_agents", "n_slots", "frac_empty", "d_med", "d_p90",
            "d_p99", "d_mean", "d_max", "frac_d0", "frac_dle1", "frac_dle15",
            "frac_read_moore", "ms_per_tick_window", "pos_n_slots",
            "pos_d_med", "pos_d_p90", "pos_frac_dle1", "pos_frac_read_moore"]


def _write_stored(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_DM_COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _stored_row(tick: int, read_moore: float) -> dict:
    return {"tick": tick, "n_agents": 100, "n_slots": 300, "frac_empty": 0.25,
            "d_med": 0.5, "d_p90": 2.0, "d_p99": 4.0, "d_mean": 0.75,
            "d_max": 9.0, "frac_d0": 0.5, "frac_dle1": 0.9, "frac_dle15": 0.95,
            "frac_read_moore": 0.99, "ms_per_tick_window": 12.3,
            "pos_n_slots": 150, "pos_d_med": 1.0, "pos_d_p90": 2.5,
            "pos_frac_dle1": 0.8, "pos_frac_read_moore": read_moore}


def _new_row(tick: int, read_moore: float) -> dict:
    r = _stored_row(tick, read_moore)
    r.pop("ms_per_tick_window")                    # 新产物逐行含 ms 列（被 guard 排除）
    r["ms_per_tick_window"] = 11.1
    r["pea_d_med"] = 1.2                           # 探针额外列（guard 不比对）
    return r


def test_dm_guard_positive_and_negative(tmp_path):
    stored = tmp_path / "stored.csv"
    _write_stored(stored, [_stored_row(100, 0.8621), _stored_row(200, 0.8610)])
    ok_rows = [_new_row(100, 0.8621), _new_row(200, 0.8610)]
    assert probe._dm_guard_mismatches(ok_rows, stored) == []      # 正例：通过
    # round(,4) 口径：差异小于末位 ⇒ 视为相等（0.86214 → 0.8621）
    near = [_new_row(100, 0.8621400001), _new_row(200, 0.8610)]
    assert probe._dm_guard_mismatches(near, stored) == []

    # 负例 a：一处数值不等 ⇒ 报 tick+列名
    bad = [_new_row(100, 0.8621), _new_row(200, 0.9000)]
    mism = probe._dm_guard_mismatches(bad, stored)
    assert len(mism) == 1 and "tick=200" in mism[0] and "pos_frac_read_moore" in mism[0]

    # 负例 b：tick 缺失
    mism2 = probe._dm_guard_mismatches([_new_row(100, 0.8621)], stored)
    assert any("缺失" in m or "行数" in m for m in mism2)

    # 负例 c：n_agents 不等（整数列）
    bad3 = [_new_row(100, 0.8621), _new_row(200, 0.8610)]
    bad3[1]["n_agents"] = 101
    mism3 = probe._dm_guard_mismatches(bad3, stored)
    assert len(mism3) == 1 and "n_agents" in mism3[0]


# ---------------- ⑤ _id 非降序 fail-loud ----------------

def test_ids_must_be_strictly_ascending():
    probe._assert_ids_ascending(np.array([1, 2, 3], np.int64))     # 正例不炸
    with pytest.raises(RuntimeError, match="_id"):
        probe._assert_ids_ascending(np.array([1, 3, 3], np.int64))  # 重复 ⇒ 炸
    with pytest.raises(RuntimeError, match="_id"):
        probe._assert_ids_ascending(np.array([2, 1], np.int64))    # 降序 ⇒ 炸
