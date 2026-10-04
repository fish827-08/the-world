"""P1-c 距离/可达性门探针回归测试（R358 T2 = 镜 D1 审计阻塞项）。

镜 D1 裁定（2026-10-03）：读数独立复核全对，但探针**零测试** ⇒ 冻结以下最小集：

  ① **同格严格 0**：`sphere_dist_rows` 同坐标 ⇒ **恰好 0.0**。这不是吹毛求疵——
     它是 atan2 形式（而非 `arccos(dot)`）存在的唯一理由：arccos 版本在同点会给
     ~3e-7 格伪残差（`world/subpos.py` 注释自陈的坑）。
  ② **对角邻居**：60×120 网格上 (30,60)→(31,61) 对角 = 球面 √2 格（口径 A：**不可**直读），
     但引擎 Moore 邻表判**可直读**（口径 B）⇒ 一条测试同时锁两口径及其差异。
  ③ **变异检查**：把 `sphere_dist_rows` 换成"跳数"实现（Moore / Manhattan / 忘环绕列差）
     ⇒ 冻结断言必须**变红**（证明断言有判别力，不是"恒真"）。
  ＋ **接缝守卫**：同行 col 119 ↔ col 0（flat 索引差 119）⇒ 仍 ≈1 格且可直读。
  ＋ **独立重跑确定性**（镜建议②）：微缩装置连跑两次探针 ⇒ CSV 逐字节相同
     （排除计时列）。此前 v1/v2 对拍是同码两版本，不算独立重跑。

装置口径：与 `experiments/p1c_distance_probe.py` 同链（make_cfg + apply_post_build +
`c.simulation.use_sim_core=False`），世界缩到 60×120 / patches=5（fixture pop=10
即可覆盖 measure 纯读场景；⑤ 子进程用 **pop=1000** —— 记忆槽只在个体站上富食格
时写入，pop=10 时 10 tick 内零槽 ⇒ 走不到池化主路径，且会触发探针**空池汇总打印
崩溃**（已作为独立缺陷上报，此处先绕开，不在本任务改探针本体）。
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.steady_k_probe import make_cfg, apply_post_build
from experiments import p1c_distance_probe as probe
from simulation.sphere_engine import SphereEngine
from world.sphere_world import SphereWorld
from world.subpos import sphere_dist_rows

ROWS, COLS = 60, 120

# 60×120 网格：dlat == dlon == 3°。取 (30,60) 格（中心 φ≈+1.5°，近赤道带）。
DIAG_CUR = (30, 60)
DIAG_MEM = (31, 61)          # 对角邻居（drow=+1, dcol=+1）
SEAM_CUR = (30, 119)
SEAM_MEM = (30, 0)           # 经度环绕的单格邻

@pytest.fixture(scope="module")
def eng():
    """微缩引擎（探针同链；measure 纯读，可全模块复用）。"""
    c, notes = make_cfg(42, ROWS, COLS, 10, 5, True, 0.125, 0.125, 80, 2.5, 30000, 1.195)
    c.simulation.use_sim_core = False
    c.resource_dynamics.enabled = False
    e = SphereEngine(c)
    apply_post_build(e, notes)
    return e


def _measure_slot(eng, cur_rc: tuple[int, int], mem_rc: tuple[int, int]) -> dict:
    """把 0 号个体的当前格/记忆槽钉到指定格（其余槽全清），再走探针 measure。"""
    w = eng.world
    cur = int(np.int64(cur_rc[0]) * w.cols + cur_rc[1])
    mem = int(np.int64(mem_rc[0]) * w.cols + mem_rc[1])
    eng._flat[0] = cur
    eng._work_memory[0, :] = -1
    eng._work_memory[0, 0] = mem
    return probe.measure(eng)


# ---------------- ① 同格严格 0 ----------------

@pytest.mark.parametrize("rf,cf", [
    (30.5, 60.5),    # 近赤道
    (30.5, 0.5),     # 接缝
    (1.5, 119.5),
    (0.5, 3.5),      # 上极行
    (59.5, 100.5),   # 下极行
])
def test_dist_same_coords_is_exactly_zero(rf, cf):
    """直接函数级：同坐标 ⇒ **恰好** 0.0（不是 < eps）。"""
    w = SphereWorld(ROWS, COLS)
    d = float(sphere_dist_rows(rf, cf, rf, cf, w))
    assert d == 0.0, f"同点距离应为严格 0，实得 {d!r}（疑似回退到 arccos 形式）"


@pytest.mark.parametrize("rc", [(30, 60), (0, 0), (1, 60)])
def test_probe_same_cell_slot_reads_zero(eng, rc):
    """探针通路：mem==cur ⇒ d==0、frac_d0==1、不驱动 mem_bit（pos_n_slots==0）。"""
    rec = _measure_slot(eng, rc, rc)
    assert rec["n_slots"] == 1
    assert np.all(rec["_d_pool"] == 0.0)
    assert rec["frac_d0"] == 1.0
    assert rec["pos_n_slots"] == 0
    # mem==cur 或相邻均判可直读（口径 B 的 `readable = mem==cur` 分支）
    assert rec["frac_read_moore"] == 1.0


# ---------------- ② 对角邻居：口径 A vs B ----------------

def _assert_diag_geometry(rec: dict) -> None:
    """冻结断言：60×120 上 (30,60)→(31,61) 对角。"""
    assert rec["pos_n_slots"] == 1
    assert rec["frac_read_moore"] == 1.0, "口径 B：Moore 8 邻应含对角"
    assert rec["pos_frac_read_moore"] == 1.0
    assert rec["frac_dle1"] == 0.0, "口径 A：对角 √2>1 格，不应算 <=1"
    assert rec["frac_dle15"] == 1.0
    d = rec["pos_d_med"]
    assert 1.3 < d < 1.5, f"对角距离应 ≈√2 格，实得 {d!r}"
    assert abs(d - math.sqrt(2)) < 0.005, f"赤道带对角应≈√2（实得 {d!r}）"


def test_diagonal_neighbor_moore_readable_but_not_dle1(eng):
    """对角：口径 B 可直读；口径 A 下 d≈√2>1（两口径差异被同时锁定）。"""
    _assert_diag_geometry(_measure_slot(eng, DIAG_CUR, DIAG_MEM))


# ---------------- ④ 接缝守卫 ----------------

def _assert_seam_geometry(rec: dict) -> None:
    """冻结断言：同行 col 119 ↔ col 0（经度环绕单格）。"""
    assert rec["pos_n_slots"] == 1
    assert rec["frac_read_moore"] == 1.0, "经度环绕邻应判可直读"
    assert rec["frac_dle1"] == 1.0, "接缝单格应 <=1 格宽"
    assert rec["frac_d0"] == 0.0
    d = rec["pos_d_med"]
    assert 0.9 < d <= 1.0 + 1e-9, f"接缝单格距离应 ≈1 格宽，实得 {d!r}"


def test_cross_seam_same_row_is_one_cell(eng):
    """col 119 ↔ col 0 是物理相邻；flat 索引差 119 ⇒ 不得把索引差当距离。"""
    _assert_seam_geometry(_measure_slot(eng, SEAM_CUR, SEAM_MEM))


# ---------------- ③ 变异检查：换实现必须变红 ----------------

def _hop_moore(rf1, cf1, rf2, cf2, world):
    dr = np.abs(np.asarray(rf1, np.float64) - np.asarray(rf2, np.float64))
    dc = np.abs(np.asarray(cf1, np.float64) - np.asarray(cf2, np.float64))
    return np.maximum(dr, dc)


def _hop_manhattan(rf1, cf1, rf2, cf2, world):
    dr = np.abs(np.asarray(rf1, np.float64) - np.asarray(rf2, np.float64))
    dc = np.abs(np.asarray(cf1, np.float64) - np.asarray(cf2, np.float64))
    return dr + dc


def _dc_no_wrap(rf1, cf1, rf2, cf2, world):
    """只按原始列差（忘经度环绕）—— 典型缺陷实现。"""
    return np.abs(np.asarray(cf1, np.float64) - np.asarray(cf2, np.float64))


def test_mutation_hopcount_goes_red(eng, monkeypatch):
    """变异检查：把球面距离换成跳数 ⇒ 冻结断言必须变红（否则测试是恒真的）。"""
    # 先证冻结断言在正确实现下是绿的
    _assert_diag_geometry(_measure_slot(eng, DIAG_CUR, DIAG_MEM))
    _assert_seam_geometry(_measure_slot(eng, SEAM_CUR, SEAM_MEM))

    # 对角：两种常见跳数口径都必须让 ② 变红
    for fake in (_hop_moore, _hop_manhattan):
        monkeypatch.setattr(probe, "sphere_dist_rows", fake)
        rec = _measure_slot(eng, DIAG_CUR, DIAG_MEM)
        with pytest.raises(AssertionError):
            _assert_diag_geometry(rec)
    monkeypatch.undo()

    # 接缝：把列差当距离（跳过环绕）必须让 ④ 变红
    monkeypatch.setattr(probe, "sphere_dist_rows", _dc_no_wrap)
    rec = _measure_slot(eng, SEAM_CUR, SEAM_MEM)
    with pytest.raises(AssertionError):
        _assert_seam_geometry(rec)


# ---------------- ⑤ 独立重跑确定性（镜建议②） ----------------

def test_probe_rerun_is_byte_identical(tmp_path):
    """同参数连跑两次探针（微缩装置）⇒ CSV 逐字节相同（排除计时列）。

    走**子进程独立重跑**（同码同档两次），而不是 v1/v2 那种"同码两版本"对拍。
    若引擎任何环节吃全局未播种随机源（F-D2 类回归），本测试会变红。

    `--pop 1000`：记忆槽只在个体站上富食格时写入；pop 太小 ⇒ 全采样零槽
    （探针空池路径，另有缺陷记录）——本测试要求走池化主路径。
    """
    root = Path(__file__).resolve().parents[1]
    out_a = tmp_path / "a.csv"
    out_b = tmp_path / "b.csv"
    base_cmd = [
        sys.executable, "-m", "experiments.p1c_distance_probe",
        "--seed", "42", "--ticks", "10", "--sample", "5",
        "--rows", str(ROWS), "--cols", str(COLS), "--patches", "5",
        "--pop", "1000", "--rgm", "1.195",
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    for out in (out_a, out_b):
        r = subprocess.run(base_cmd + ["--out", str(out)], cwd=str(root),
                           env=env, capture_output=True, timeout=600)
        assert r.returncode == 0, (
            f"探针重跑失败 rc={r.returncode}：\n"
            f"stdout={r.stdout.decode('utf-8', 'replace')[-2000:]}\n"
            f"stderr={r.stderr.decode('utf-8', 'replace')[-2000:]}")

    def _rows(path: Path) -> list[list[str]]:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        header = lines[0].split(",")
        drop = header.index("ms_per_tick_window")
        return [[f for i, f in enumerate(ln.split(",")) if i != drop]
                for ln in lines[1:]]

    rows_a, rows_b = _rows(out_a), _rows(out_b)
    assert len(rows_a) >= 2, "微缩装置 10 tick / 每 5 采样应至少出 2 行"
    assert rows_a == rows_b, "同参数独立重跑 CSV 不一致 ⇒ 探针/引擎有非确定性"
