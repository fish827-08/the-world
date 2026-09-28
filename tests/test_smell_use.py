"""R244 §二 气味场**消费端**单测（`[云端开发·云启]`，2026-09-28）。

对实施规格《移动决策三件》§2.3 测试清单 **六条** + 三条防呆：
  A 默认关               —— `use_in_move=False` ⇒ 与"场全关"**逐位一致**（+ C7 由 digest 管）
  B 方向正确性（构造性）  —— 单侧高浓度 ⇒ 个体**朝浓度高侧**移动（非统计断言）
  C 风险负权重           —— `w_risk<0` ⇒ 朝**浓度低侧**走（回避）
  D 归一化（解析上界）    —— `Ŝ = clip(S/S_max,0,1) ≤ 1`；`S = S_max ⇒ Ŝ = 1`
  E 双路径（Rust vs 参考）—— 开档逐位一致（含移动与能量）
  F 防呆                 —— `norm_mode="window"`（未实现）⇒ 炸；`use_in_move=True` 但通道空 ⇒ 炸；
                            拼错通道 ⇒ 炸
（第 6 条"变异检查：`w_food` 置 0 ⇒ 例 B 必红"由 `_rerun_logs/smell_v1_smoke/del8_mutations_use.py` 执行。）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import SimConfig  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402
from simulation.genes import Gene  # noqa: E402


def _cfg(use: bool, rust: bool = True, seed: int = 7, **kw) -> SimConfig:
    """小世界：**无食无信号**（⇒ 打分里唯一非均匀项 = 气味）⇒ 方向性可构造断言。"""
    c = SimConfig(seed=seed)
    c.world.rows, c.world.cols = 8, 12
    c.population.initial_count = 1
    c.resources.distribution = "uniform"
    c.resources.bg_production_zero = True          # 全场无食物（food_ratio 逐格同值 ⇒ 不参与方向）
    c.resources.initial_fill = 0.0
    c.simulation.use_sim_core = rust
    c.smell.channels = ("food", "risk")
    c.smell.use_in_move = use
    for k, v in kw.items():
        setattr(c.smell, k, v)
    return c


def _state(e: SphereEngine) -> tuple:
    P = len(e._id)
    return (tuple(int(x) for x in e._flat[:P]),
            tuple(round(float(x), 12) for x in e._energy[:P]))


def _single_at(eng: SphereEngine, row: int, col: int) -> int:
    """把唯一那个个体钉在 (row, col)，并给足能量/移动意愿/感知（构造确定性场景）。

    `perc > 0` 是必须的：嗅觉项与基础项都乘 `perc` ⇒ `perc=0` 会让所有候选**同分**
    （退化为随机平局 ⇒ 测试会 flaky）。
    """
    cell = int(row * eng.world.cols + col)
    eng._flat[0] = cell
    eng._energy[0] = 1e6
    eng._genes[0, int(Gene.MOVE_PROB)] = 1.0
    eng._genes[0, int(Gene.ROOTING)] = 0.0
    eng._genes[0, int(Gene.PERCEPTION)] = 1.0
    return cell


def _hot_box(eng: SphereEngine, channel: str, cells, value: float) -> None:
    """把指定格的 `channel` 近场打到高浓度（**只写近场**）。

    ⚠️ 不要同时把粗网格置成同值：粗网格是**块均值**（≈value/s²），置成同值会让**整块**
    乃至远处候选都超上界被 clip 到 1 ⇒ 梯度消失（方向性断言会退化成随机平局）。
    """
    ci = eng.smell._idx[channel]
    eng.smell._S[ci, cells] = value


# ─────────────────────── A：默认关 ───────────────────────

def test_default_off_bitwise():
    """`use_in_move=False` ⇒ 与"场本体全关"逐位一致（默认关 = 回滚点）。"""
    on_field_only = SphereEngine(_cfg(False, True, 11))
    off_all = SphereEngine(_cfg(False, True, 11))
    off_all.smell = None                      # 场本体也关
    off_all._smell_on = False
    for _ in range(25):
        on_field_only.step()
        off_all.step()
    assert _state(on_field_only) == _state(off_all), "消费端关档却改动了轨迹"


# ─────────────────────── B/C：方向性（构造性）───────────────────────

def test_direction_toward_high_food():
    """①食物通道：单侧高浓度 ⇒ 个体**朝浓度高侧**（东）移动 —— 构造性断言。"""
    eng = SphereEngine(_cfg(True, True, 7, w_food=0.5, w_risk=0.0))
    row, col = eng.world.rows // 2, 3
    cell = _single_at(eng, row, col)
    east = int(row * eng.world.cols + (col + 1))
    for _ in range(3):                        # 场需要 k=4 tick 才更新一次；这里直接写场，再走 1 tick
        eng.step()
    _single_at(eng, row, col)
    _hot_box(eng, "food", [east], 10.0)
    eng.step()
    new_cell = int(eng._flat[0])
    new_col = new_cell % eng.world.cols
    assert new_col == col + 1 and (new_cell // eng.world.cols) == row, (
        f"应朝东（浓度高侧）走：cell {cell} → {new_cell}（列 {col} → {new_col}）")


def test_direction_away_from_high_risk():
    """③风险通道负权重 ⇒ 朝**浓度低侧**走（回避）。"""
    eng = SphereEngine(_cfg(True, True, 7, w_food=0.0, w_risk=-0.5))
    row, col = eng.world.rows // 2, 3
    _single_at(eng, row, col)
    for _ in range(3):
        eng.step()
    _single_at(eng, row, col)
    west = int(row * eng.world.cols + (col - 1))
    _hot_box(eng, "risk", [west], 10.0)
    eng.step()
    new_cell = int(eng._flat[0])
    new_col = new_cell % eng.world.cols
    assert new_col != col - 1, f"应避开高浓度（西）侧：cell {new_col}（原列 {col}）"


# ─────────────────────── D：归一化 ───────────────────────

def test_normalization_analytic_bound():
    """`Ŝ = clip(S/S_max, 0, 1)`：`S=S_max ⇒ Ŝ=1`；超界 ⇒ clip 到 1（并有计数上报）。"""
    eng = SphereEngine(_cfg(True, True, 7, w_food=0.5, w_risk=0.0))
    sm = eng.smell
    c = np.array([5], dtype=np.int64)
    smax = sm.s_max("food")
    assert smax > 0.0
    sm._S[0, 5] = 0.0
    sm._Sc[0, 0, 0] = 0.0
    assert float(sm.combined(c)[0]) == 0.0
    sm._S[0, 5] = smax                        # 恰在上界 ⇒ Ŝ = 1
    val = float(sm.combined(c)[0])
    assert val == pytest.approx(0.5, rel=1e-12), f"Ŝ=1 ⇒ 0.5×1 = 0.5，实得 {val}"
    sm._S[0, 5] = 10.0 * smax                 # 超界 ⇒ clip + 计数
    before = sm.probe()["hat_clip_n"]
    assert float(sm.combined(c)[0]) == pytest.approx(0.5, rel=1e-12), "超界必须 clip 到 1"
    assert sm.probe()["hat_clip_n"] > before, "截断必须在 probe 里报数（不静默）"


# ─────────────────────── E：双路径 ───────────────────────

def test_dual_path_bitwise_with_use_on():
    """开消费端：**三处同式** —— Rust 热核 / 向量化快路径 / Python 参考循环 **逐位一致**。"""
    kw = dict(w_food=0.5, w_risk=-0.5)
    r = SphereEngine(_cfg(True, True, 13, **kw))          # ① Rust
    v = SphereEngine(_cfg(True, False, 13, **kw))         # ② 向量化快路径（_batch_move_on 默认 True）
    p = SphereEngine(_cfg(True, False, 13, **kw))         # ③ Python 参考循环
    p._batch_move_on = False
    for _ in range(30):
        r.step()
        v.step()
        p.step()
    assert _state(r) == _state(v), "Rust vs 向量化快路径 不逐位"
    assert _state(v) == _state(p), "向量化快路径 vs Python 参考循环 不逐位"
    assert len(r._flat) == len(p._flat)


# ─────────────────────── F：防呆 ───────────────────────

def test_fail_loud_guards():
    """未实现的归一化模式 / 通道空却开消费端 / 拼错通道 —— 三者都必须**炸**。"""
    with pytest.raises(AssertionError):
        SimConfig(seed=1).smell.__class__(channels=("food",), norm_mode="window")
    with pytest.raises(AssertionError):
        SimConfig(seed=1).smell.__class__(use_in_move=True, channels=())
    with pytest.raises(AssertionError):
        SimConfig(seed=1).smell.__class__(channels=("foods",))
    # 构造后改配置不走 __post_init__ ⇒ 模块侧/引擎侧自查
    c = _cfg(True)
    c.smell.norm_mode = "window"
    with pytest.raises(AssertionError):
        SphereEngine(c)