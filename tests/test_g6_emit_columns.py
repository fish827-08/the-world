"""R396 三拍②（G6 发射计量列）靶向测试 —— 全部用 stub 信号场，不跑真引擎（秒级）。

对应 `_archive/2026-10-10-退役团队-归档/docs-旧档/tasks/需求单-GAP-A-EMIT-发射率专列-20261004.md` §3.3 第 4 条测试清单：

| # | 钉住的事 | 变异注入（改坏 ⇒ 本测试必须变红） |
|---|---|---|
| T1 | 已知发射日程 ⇒ `g6_emit_events_cum` 逐采样点**精确** | 包装器转调前漏计数 ⇒ 列恒 0 |
| T2 | 同格两体同 tick ⇒ 计 **2**（G3 归因只记 1 ⇒ 两列不同轴） | 改用 `fresh` 格数当分子 ⇒ 计 1 |
| T3 | 付不起成本（引擎 `can_afford` 已滤）⇒ 不进 `write_many` ⇒ 不计 | 按活体数估 ⇒ 多计 |
| T4 | 全程零发射 ⇒ rate = **0** 而非 NaN；`Δperson_ticks=0` ⇒ **NaN 不补 0** | 分母 0 时补 0 ⇒ 假"发射率归零" |
| T5 | 🔴 有写入格却零调用 ⇒ **停跑**（不许静默出零列） | 去掉 fail-loud ⇒ GAP-A §二那类失效无声通过 |
| T6 | `emit_pooled` 分母 = **人·tick**（首/末段合并比） | 误用采样点数 ⇒ 首末段比值必偏 |
| T7 | 挂钩装/卸**成对**：`detach` 后类方法回到原函数且不再计数 | 卸不干净 ⇒ 泄漏到同进程后续 run |
| T8 | Rust 路径（`_use_sim_core=True`）⇒ `g6_*` 全 NaN（禁当真值） | 默认当真值 ⇒ Rust 批出假读数 |
| T9 | 列集完整 + 无重复（表头/行/summary 三处同一套列名） | 列名漂移 ⇒ 续跑列错位 |
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import experiments.s3_memory_probe as P
from experiments.s3_memory_probe import _G6_NUM_COLS, _M0_COLS, _M0Instruments


class _Sig:
    """stub 信号场：`write_many` 语义照 `world/signal_field.py:104`（含 pattern=0 清格）。"""

    def __init__(self, n, duration=5):
        self._marks = np.zeros(n, dtype=np.uint8)
        self._age = np.zeros(n, dtype=np.int32)
        self.duration = int(duration)

    def write_many(self, cells, patterns):
        cells = np.asarray(cells, dtype=np.int64)
        pats = np.asarray(patterns, dtype=np.uint8)
        self._marks[cells] = pats
        self._age[cells] = np.where(pats > 0, self.duration, 0).astype(np.int32)

    def decay(self):
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
        self._nb_table = np.tile(np.arange(n, dtype=np.int64)[:, None], (1, 8))
        self._pole_nb = np.array([np.arange(cols), np.full(cols, cols)], dtype=np.int64)


class _Eng:
    def __init__(self, rows=4, cols=8, n_ind=6, use_sim_core=False):
        self.signals = _Sig(rows * cols)
        self.world = _World(rows, cols)
        self._nb_table = self.world._nb_table
        self._pole_nb = self.world._pole_nb
        cells = np.arange(1, n_ind + 1, dtype=np.int64)
        self._flat = cells.copy()
        self._id = (10 + np.arange(n_ind, dtype=np.int64))
        self._genes = np.zeros((n_ind, 32), dtype=np.float64)
        self._genes[:, 15] = 0.5
        self._mem_bit_on = 0
        self._mem_bit_n = 0
        self._alpha = "16"
        if use_sim_core:
            self._use_sim_core = True


def _emit(eng, cells, pats):
    eng.signals.decay()
    if len(cells):
        eng.signals.write_many(np.asarray(cells, dtype=np.int64),
                               np.asarray(pats, dtype=np.uint8))


def _run_sched(m0, eng, sched, sample, ticks):
    """按日程推进；返回 {tick: sample_cols} （只在采样点取）。"""
    out = {}
    try:
        for t in range(1, ticks + 1):
            sampled = (t % sample == 0) or (t == ticks)
            _emit(eng, *sched.get(t, ([], [])))
            m0.after_step(eng, t, sampled)
            if sampled:
                out[t] = m0.sample_cols(eng)
    finally:
        m0.detach()
    return out


# ---------------- T1 ----------------

def test_t1_events_cum_exact_per_sample_point():
    eng = _Eng(n_ind=6)                       # 6 体 @ cell1..6，恒定 N=6
    m0 = _M0Instruments()
    m0.attach(eng)
    sched = {t: ([1, 2], [3, 4]) for t in (1, 2, 3, 4)}     # 4 tick × 2 发射者 = 8 人·次
    got = _run_sched(m0, eng, sched, sample=4, ticks=12)
    assert got[4]["g6_emit_events_cum"] == 8
    assert got[4]["g6_emit_person_ticks_cum"] == 24          # 6 体 × 4 tick
    assert got[8]["g6_emit_events_cum"] == 8                 # 5..8 无发射
    assert got[12]["g6_emit_events_cum"] == 8
    assert got[4]["g6_emit_events_window"] == 8              # 首窗 = 整段
    assert got[8]["g6_emit_events_window"] == 0
    assert got[4]["g6_emit_rate_window"] == pytest.approx(8 / 24)
    assert got[12]["g6_emit_rate_window"] == 0.0             # 零发射窗 ⇒ 0（非 NaN）
    assert got[12]["g6_emit_rate_cum"] == pytest.approx(8 / 72)


# ---------------- T2 ----------------

def test_t2_same_cell_two_emitters_counts_two_not_one():
    eng = _Eng(n_ind=2)
    eng._flat = np.array([1, 1], dtype=np.int64)             # 两体同格 cell1
    eng._id = np.array([10, 11], dtype=np.int64)
    m0 = _M0Instruments()
    m0.attach(eng)
    try:
        _emit(eng, [1, 1], [3, 5])                           # 同格同 tick 各发一次
        m0.after_step(eng, 1, sampled=False)
    finally:
        m0.detach()
    assert m0.g6_events == 2                                 # G6 = 人·次
    assert m0.fresh_n == 1 and m0.ambig_n == 1               # G3 侧：一格 + 归因歧义
    assert sum(m0.hist_run) == 0                             # 串未断 ⇒ G3 一条不入
    assert m0.g6_zero_patterns == 0


# ---------------- T3 ----------------

def test_t3_cant_afford_not_counted():
    eng = _Eng(n_ind=3)
    m0 = _M0Instruments()
    m0.attach(eng)
    try:
        _emit(eng, [2], [7])                                 # 3 体里只有 1 体付得起
        m0.after_step(eng, 1, sampled=False)
    finally:
        m0.detach()
    assert m0.g6_events == 1
    assert m0.g6_calls == 1
    assert m0.g6_person_ticks == 3                           # 分母仍是全体活体


# ---------------- T4 ----------------

def test_t4_zero_emission_gives_rate_zero_not_nan():
    eng = _Eng(n_ind=4)
    m0 = _M0Instruments()
    m0.attach(eng)
    got = _run_sched(m0, eng, {}, sample=2, ticks=4)
    assert got[2]["g6_emit_events_cum"] == 0
    assert got[2]["g6_emit_rate_window"] == 0.0
    assert got[4]["g6_emit_rate_cum"] == 0.0
    assert not math.isnan(got[4]["g6_emit_rate_window"])


def test_t4b_extinction_window_gives_nan_not_zero():
    eng = _Eng(n_ind=2)
    m0 = _M0Instruments()
    m0.attach(eng)
    try:
        eng._flat = np.array([], dtype=np.int64)             # 全段灭绝 ⇒ 分母 0
        eng._id = np.array([], dtype=np.int64)               # 引擎不变量：与 _flat 同长
        m0.after_step(eng, 1, sampled=True)                   # 不炸（明知无数可采）
        d = m0.sample_cols(eng)
    finally:
        m0.detach()
    assert d["g6_emit_person_ticks_cum"] == 0
    assert math.isnan(d["g6_emit_rate_window"])
    assert math.isnan(d["g6_emit_rate_cum"])


# ---------------- T5（变异必红①）----------------

def test_t5_silent_tap_bypass_fails_loud():
    """marks 有新增而挂钩零调用 ⇒ 必须**停跑**（防"一列静默零值"）。"""
    orig = _Sig.write_many                                   # 取挂钩前的原函数
    eng = _Eng(n_ind=3)
    m0 = _M0Instruments()
    m0.attach(eng)
    try:
        eng.signals.decay()
        orig(eng.signals, np.array([1], dtype=np.int64),
             np.array([3], dtype=np.uint8))                  # 绕过类方法直写
        assert m0.g6_calls == 0
        with pytest.raises(RuntimeError, match="发射抓手失效"):
            m0.after_step(eng, 1, sampled=False)
    finally:
        m0.detach()


def test_t5b_double_attach_fails_loud():
    eng = _Eng(n_ind=2)
    m0 = _M0Instruments()
    m0.attach(eng)
    try:
        with pytest.raises(RuntimeError, match="attach 被调了两次"):
            m0.attach(eng)
    finally:
        m0.detach()


def test_t5c_tick_must_advance():
    eng = _Eng(n_ind=2)
    m0 = _M0Instruments()
    m0.attach(eng)
    try:
        m0.after_step(eng, 5, sampled=False)
        with pytest.raises(RuntimeError, match="tick 未前进"):
            m0.after_step(eng, 5, sampled=False)
    finally:
        m0.detach()


# ---------------- T6（变异必红②：分母 = 人·tick）----------------

def test_t6_emit_pooled_uses_person_ticks_denominator():
    rows = [{"g6_emit_events_cum": e, "g6_emit_person_ticks_cum": pt}
            for e, pt in zip((10, 20, 30, 40, 50, 60, 70, 80),
                             (5, 10, 15, 20, 25, 30, 36, 42))]
    out = _M0Instruments().emit_pooled(rows, window_frac=0.25)
    # 首段 = 行 0..1 ⇒ Δev=20 / Δpt=10 = 2.0；末段 = 行 6..7 ⇒ 20 / (42-30) = 5/3
    assert out["emit_rate_first"] == pytest.approx(2.0)
    assert out["emit_rate_last"] == pytest.approx(5 / 3)
    assert out["emit_rate_ratio"] == pytest.approx(5 / 6)
    # 误用"采样点数"作分母 ⇒ 首末都得 10.0、ratio=1.0（与上面的 0.8333 不符 ⇒ 必红）
    assert out["emit_rate_ratio"] != pytest.approx(1.0)
    assert out["emit_rate_n_samples"] == 8
    assert out["emit_events_total"] == 80
    assert out["emit_person_ticks_total"] == 42
    assert out["emit_rate_window_frac"] == 0.25


def test_t6b_first_segment_zero_gives_nan_ratio_with_note():
    rows = [{"g6_emit_events_cum": 0, "g6_emit_person_ticks_cum": 10},
            {"g6_emit_events_cum": 0, "g6_emit_person_ticks_cum": 20},
            {"g6_emit_events_cum": 5, "g6_emit_person_ticks_cum": 30},
            {"g6_emit_events_cum": 9, "g6_emit_person_ticks_cum": 40}]
    out = _M0Instruments().emit_pooled(rows, window_frac=0.5)
    assert out["emit_rate_first"] == 0.0
    assert math.isnan(out["emit_rate_ratio"])                # 0 分母 ⇒ 不可算，不返回 inf
    assert out["emit_rate_note"] == "ratio_undef:first_segment_rate_zero"


def test_t6c_empty_rows_nan_with_note():
    out = _M0Instruments().emit_pooled([], window_frac=0.25)
    assert math.isnan(out["emit_rate_ratio"])
    assert out["emit_rate_note"] == "nan:no_sample_rows"


# ---------------- T6d/T6e（A1：窗式 = 已锁 index 式 floor，PI 复审裁定 21:4x）----------------

def _spike_rows(n, spikes):
    """行 i：events 增量恒 1；person_ticks 增量默认 1，`spikes` 指定的行改为 100。

    尖峰用来把"段边界落在哪一行"变成可读出的数：若边界把尖峰行**包含进段内**，
    该段分母多出 99 ⇒ 比值明显偏离 1.0。
    """
    ev = pt = 0
    rows = []
    for i in range(n):
        ev += 1
        pt += spikes.get(i, 1)
        rows.append({"g6_emit_events_cum": ev, "g6_emit_person_ticks_cum": pt})
    return rows


def test_t6d_window_is_floor_index_not_round():
    """n=30 · frac=0.25 ⇒ k=int(7.5)=**7**（floor），首段行 0..6、末段行 23..29。

    尖峰放在**边界外一行**（row 7 = k 的下一个下标）与**边界基线行**（row 22 =
    末段前一行）⇒ floor 式两段都干净（=1.0）；round 式 k=8 会把 row 7 吃进首段、
    把 row 22 吃进末段 ⇒ 两段同时变 8/107 ⇒ 必红。
    """
    rows = _spike_rows(30, {7: 100, 22: 100})
    out = _M0Instruments().emit_pooled(rows, window_frac=0.25)
    assert out["emit_rate_n_samples"] == 30
    assert out["emit_rate_first"] == pytest.approx(1.0)       # k=8 ⇒ 8/107=0.0748
    assert out["emit_rate_last"] == pytest.approx(1.0)        # k=8 ⇒ 8/107=0.0748
    assert out["emit_rate_ratio"] == pytest.approx(1.0)       # k=8 两段同偏 ⇒ ratio=1（假正常）
    assert out["emit_events_total"] == 30
    assert out["emit_person_ticks_total"] == 228              # 7+100+14+100+7


def test_t6e_window_below_one_sample_is_nan_not_k1():
    """n·frac < 1 ⇒ k=0 ⇒ 出 NaN，**不补 k=1**（补 1 即自造口径，与锁定文本分叉）。"""
    rows = [{"g6_emit_events_cum": e, "g6_emit_person_ticks_cum": pt}
            for e, pt in zip((2, 4, 6), (10, 20, 30))]          # n=3 ⇒ int(0.75)=0
    out = _M0Instruments().emit_pooled(rows, window_frac=0.25)
    assert out["emit_rate_n_samples"] == 3
    for key in ("emit_rate_first", "emit_rate_last", "emit_rate_ratio"):
        assert math.isnan(out[key]), key                       # k=1 时 first=0.2 ⇒ 必红
    assert out["emit_rate_note"] == "nan:window_below_one_sample"
    assert out["emit_events_total"] == 6                       # 总量键不丢（summary 键集稳定）
    assert out["emit_person_ticks_total"] == 30


# ---------------- T7 ----------------

def test_t7_tap_installs_and_restores():
    orig = _Sig.write_many
    assert _Sig not in P._G6_TAPS                             # 进测试时该类干净
    eng = _Eng(n_ind=2)
    m0 = _M0Instruments()
    m0.attach(eng)
    assert _Sig.write_many is not orig                        # 类级挂钩已装
    _emit(eng, [1], [4])
    assert m0.g6_events == 1
    m0.detach()
    assert _Sig.write_many is orig                            # 已还原
    assert _Sig not in P._G6_TAPS                             # 已还原到干净
    _emit(eng, [2], [4])                                      # 卸后再写 ⇒ 不计数
    assert m0.g6_events == 1
    assert m0.detach() is None                                # detach 幂等


def test_t7b_later_instance_supersedes_not_crashes():
    eng = _Eng(n_ind=2)
    a, b = _M0Instruments(), _M0Instruments()
    a.attach(eng)                                             # 故意不 detach（模拟泄漏）
    try:
        b.attach(eng)
        _emit(eng, [1], [4])
        assert b.g6_events == 1 and a.g6_events == 0
        assert a._g6_superseded is True
    finally:
        b.detach()
        a.detach()
    assert _Sig.write_many.__name__ == "write_many"


# ---------------- T8 ----------------

def test_t8_rust_path_columns_are_nan():
    eng = _Eng(n_ind=3, use_sim_core=True)
    m0 = _M0Instruments()
    m0.attach(eng)
    try:
        assert m0.g6_nan is True
        assert _Sig not in P._G6_TAPS                          # Rust 路径不装挂钩
        _emit(eng, [1, 2], [3, 4])
        m0.after_step(eng, 1, sampled=True)                   # 不炸（明知无数可采）
        d = m0.sample_cols(eng)
        for k in _G6_NUM_COLS:
            assert math.isnan(d[k]), k
        out = m0.emit_pooled([d], window_frac=0.25)
        assert out["emit_rate_note"] == "nan:g6_tap_unavailable"
    finally:
        m0.detach()


# ---------------- T9 ----------------

def test_t9_column_set_complete_and_unique():
    assert list(_G6_NUM_COLS) == [
        "g6_emit_events_cum", "g6_emit_person_ticks_cum",
        "g6_emit_events_window", "g6_emit_rate_window", "g6_emit_rate_cum"]
    assert len(set(_M0_COLS)) == len(_M0_COLS)                # 无重复列名
    assert _M0_COLS[-len(_G6_NUM_COLS):] == list(_G6_NUM_COLS)    # 追加在尾（不改旧列序）
    eng = _Eng(n_ind=2)
    m0 = _M0Instruments()
    m0.attach(eng)
    try:
        d = m0.sample_cols(eng)
    finally:
        m0.detach()
    assert set(_M0_COLS) <= set(d.keys())                     # 每列都有值（缺列即失败）
