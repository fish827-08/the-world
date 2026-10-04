"""R358 T4（A1/A2/A3）：M0 仪表 `_M0Instruments` 纯单元测试（stub 最小引擎状态）。

覆盖（不跑真引擎 ⇒ 秒级）：
  ① 装配 fail-loud（缺属性即炸，不许静默 NaN）；
  ② G3 连串：连续写 k tick ⇒ 断串时入桶 `hist[k]`；空档 ⇒ 断串；
  ③ G3 审计：同格双写者 ⇒ ambig；新写格无候选 ⇒ orphan；
  ④ G5：方向分布 TV 距离 / 静止率差（手算锚）+ 单侧为空 ⇒ NaN；
  ⑤ G1/G0/G4 采样列数值（手算锚）。
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.s3_memory_probe import _M0_COLS, _M0Instruments


class _Sig:
    def __init__(self, n, duration=5):
        self._marks = np.zeros(n, dtype=np.uint8)
        self._age = np.zeros(n, dtype=np.int32)
        self.duration = int(duration)

    def write(self, cell, pat):
        self._marks[cell] = np.uint8(pat)
        self._age[cell] = self.duration

    def decay(self):
        """模拟引擎步内 `signals.tick()`：先衰减、过期清零（emission 写在其后）。"""
        a = self._age
        a[a > 0] -= 1
        ex = (a <= 0) & (self._marks != 0)
        self._marks[ex] = 0
        a[ex] = 0


class _World:
    def __init__(self, rows=4, cols=8):
        self.rows, self.cols = rows, cols
        self._pole_top, self._pole_bottom = 0, rows - 1
        n = rows * cols
        self._nb_table = np.tile(
            np.arange(n, dtype=np.int64)[:, None], (1, 8))          # 邻居=自身（可手算）
        self._pole_nb = np.array([np.arange(cols), np.full(cols, cols)],
                                 dtype=np.int64)


class _Eng:
    def __init__(self, rows=4, cols=8):
        n = rows * cols
        self.signals = _Sig(n)
        self.world = _World(rows, cols)
        self._nb_table = self.world._nb_table
        self._pole_nb = self.world._pole_nb
        self._flat = np.array([1, 2], dtype=np.int64)
        self._id = np.array([10, 11], dtype=np.int64)
        self._genes = np.zeros((2, 32), dtype=np.float64)
        self._genes[:, 15] = 0.5
        self._mem_bit_on = 0
        self._mem_bit_n = 0
        self._alpha = "16"


def _advance(eng, writes):
    """一步：先 tick() 衰减，再 emission 写（与引擎步内顺序一致）。"""
    eng.signals.decay()
    for cell, pat in writes:
        eng.signals.write(cell, pat)


def test_attach_fail_loud_on_missing_attr():
    eng = _Eng()
    m0 = _M0Instruments()
    m0.attach(eng)                                # 装配成功路径
    eng2 = _Eng()
    del eng2._mem_bit_on                          # 抽掉一个必需属性
    with pytest.raises(RuntimeError, match="mem_bit_on"):
        _M0Instruments().attach(eng2)


def test_g3_run_len_consecutive_and_gap():
    eng = _Eng()                                  # id10@cell1, id11@cell2
    m0 = _M0Instruments()
    m0.attach(eng)
    for t in (1, 2, 3):                           # 连续 3 tick 写 ⇒ 串长 3
        _advance(eng, [(1, 1)])
        m0.after_step(eng, t, sampled=False)
    assert m0.hist_run[3] == 0                    # 串未断 ⇒ 未入桶
    _advance(eng, [])                             # t=4 空档（衰减，无写）
    m0.after_step(eng, 4, sampled=False)
    _advance(eng, [(1, 1)])                       # t=5 复写 ⇒ 断串入桶 hist[3]
    m0.after_step(eng, 5, sampled=False)
    assert m0.hist_run[3] == 1
    assert m0.fresh_n == 4                        # 3 + 1 次新写格
    assert m0.ambig_n == 0 and m0.orphan_n == 0


def test_g3_ambig_and_orphan():
    eng = _Eng()
    eng._flat = np.array([1, 1], dtype=np.int64)  # 两体同格 cell1
    eng._id = np.array([10, 11], dtype=np.int64)
    m0 = _M0Instruments()
    m0.attach(eng)
    _advance(eng, [(1, 1)])
    m0.after_step(eng, 1, sampled=False)
    assert m0.ambig_n == 1                        # 同格双写者 ⇒ 歧义（取小 id 兜底）
    _advance(eng, [(5, 1)])                       # cell5 无人占 ⇒ orphan
    m0.after_step(eng, 2, sampled=False)
    assert m0.orphan_n == 1


def test_g5_dir_and_move_diff_hand_anchored():
    eng = _Eng()                                  # rows=4, cols=8
    # t-1 末：id10 在 cell9（row1,普通行）、id11 在 cell10。marks[9] 亮 ⇒ 仅 id10 有信号。
    m0 = _M0Instruments()
    m0.last_ids = np.array([10, 11], dtype=np.int64)
    m0.last_flat = np.array([9, 10], dtype=np.int64)
    eng.signals._marks[9] = 1                     # 决策时点图 = 本步末 marks（含本步写）
    eng.signals._age[9] = eng.signals.duration
    # t 末：id10 右移 9→10（code=5），id11 静止（code=4）。
    eng._id = np.array([10, 11], dtype=np.int64)
    eng._flat = np.array([10, 10], dtype=np.int64)
    g5 = m0._compute_g5(eng)
    assert g5["g5_n_sig"] == 1 and g5["g5_n_nosig"] == 1
    assert g5["g5_dir_diff"] == pytest.approx(1.0)   # 两分布各占一不同槽 ⇒ TV=1
    assert g5["g5_move_diff"] == pytest.approx(1.0)  # 有信号全动 / 无信号全静


def test_g5_single_side_empty_gives_nan():
    eng = _Eng()
    m0 = _M0Instruments()
    m0.last_ids = np.array([10, 11], dtype=np.int64)
    m0.last_flat = np.array([9, 10], dtype=np.int64)
    # 全场无 marks ⇒ n_sig=0（marks/duration 由构造处全零）
    eng._id = np.array([10, 11], dtype=np.int64)
    eng._flat = np.array([10, 10], dtype=np.int64)
    g5 = m0._compute_g5(eng)
    assert g5["g5_n_sig"] == 0 and g5["g5_n_nosig"] == 2
    assert np.isnan(g5["g5_dir_diff"]) and np.isnan(g5["g5_move_diff"])


def test_sample_cols_entropy_g15_g4():
    eng = _Eng()
    m0 = _M0Instruments()
    m0.attach(eng)
    eng.signals._marks[:] = 0
    eng.signals._marks[2] = 1                     # pattern 1 一次
    eng.signals._marks[3] = 3                     # pattern 3 一次
    eng._mem_bit_on, eng._mem_bit_n = 3, 4
    d = m0.sample_cols(eng)
    assert d["marks_hist_1"] == 1 and d["marks_hist_3"] == 1
    assert d["marks_hist_0"] == 30
    assert d["g1_entropy"] == pytest.approx(1.0 / 4.0)   # 两符号各半 ⇒ H=1 bit ÷ 4
    assert d["g15_mean"] == pytest.approx(0.5)
    assert d["g4_mem_bit_frac"] == pytest.approx(0.75)
    assert d["g4_mem_bit_on"] == 3 and d["g4_mem_bit_n"] == 4
    assert set(_M0_COLS) <= set(d.keys())        # 列齐全（缺失即失败）


def test_sample_cols_empty_marks_nan():
    eng = _Eng()
    m0 = _M0Instruments()
    m0.attach(eng)
    eng._mem_bit_on = eng._mem_bit_n = 0
    d = m0.sample_cols(eng)
    assert np.isnan(d["g1_entropy"])              # 无标记 ⇒ NaN（非 0）
    assert np.isnan(d["g4_mem_bit_frac"])         # 0/0 ⇒ NaN
    assert np.isnan(d["g5_dir_diff"])
