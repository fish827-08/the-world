"""B10：攻击窗口分布仪器测试（chase_gain / flee_gain distribution）。

覆盖 metrics.py 的两个永久仪器（B6/B10 诊断读数，进 metrics_json）：
- chase_gain_distribution 的四个分桶（送上门/目标走近/攻击者逼近/其他）
  与 drop 统计；
- flee_gain_distribution 只计致伤（致死无窗口——B6 标签修复的证据口径）；
- metrics_json 输出含两个分布键；
- 空日志不崩。
"""
import numpy as np
import pytest

from biosphere.metrics import (
    Frame, SimLog, chase_gain_distribution, flee_gain_distribution,
    metrics_json,
)
from biosphere.resolve import ResolveEvent
from biosphere.world_adapter import WorldView, _von_neumann_table

ROWS, COLS = 4, 6


def _world() -> WorldView:
    n = ROWS * COLS
    return WorldView(
        rows=ROWS, cols=COLS, n_cells=n,
        surface_type=np.zeros(n, dtype=np.uint8),
        elevation_m=np.zeros(n), depth_m=np.zeros(n),
        patch_id=np.full(n, -1, dtype=np.int32),
        neighbors=_von_neumann_table(ROWS, COLS),
    )


def _frame(tick, positions: dict) -> Frame:
    """positions: {uid: cell} ⇒ 一帧（能量 50、nn 2.5）。"""
    uids = np.array(sorted(positions), dtype=np.int64)
    cells = np.array([positions[u] for u in sorted(positions)], dtype=np.int64)
    return Frame(tick, uids, cells, np.full(uids.size, 50.0),
                 np.full(uids.size, 2.5))


def _attack(tick, actor, target, status="attack_survived"):
    return ResolveEvent(tick, actor, "attack", status, -1, target, -0.4)


class TestChaseGainDistribution:
    def test_attacker_closed_bucket(self):
        """攻击者 3 tick 内逼近 ≥1 格 ⇒ attacker_closed、drop≥0.5。"""
        log = SimLog()
        # t-3..t-1：攻击者在 8、目标在 16；t：攻击者 14、目标 16
        for t in range(15, 20):
            pos = {1: 8, 2: 16}
            if t == 20 - 1 and False:
                pass
            log.frames.append(_frame(t, pos))
        # t=19 帧：攻击者已到 14（位移在 t=19 生效）
        log.frames[-1] = _frame(19, {1: 14, 2: 16})
        log.frames.append(_frame(20, {1: 14, 2: 16}))
        log.events.append(_attack(20, 1, 2))
        d = chase_gain_distribution(log, _world())
        assert d["n_hits"] == 1 and d["n_window_available"] == 1
        assert d["frac_attacker_closed"] == 1.0
        assert d["frac_already_adjacent_at_t3"] == 0.0
        assert d["frac_drop_ge_0p5"] == 1.0
        assert d["drop_median"] == 1.0

    def test_already_adjacent_bucket(self):
        """3 tick 前已贴贴 ⇒ already_adjacent（机会主义，非追击）。"""
        log = SimLog()
        for t in range(15, 21):
            log.frames.append(_frame(t, {1: 15, 2: 16}))
        log.events.append(_attack(20, 1, 2))
        d = chase_gain_distribution(log, _world())
        assert d["frac_already_adjacent_at_t3"] == 1.0
        assert d["frac_attacker_closed"] == 0.0

    def test_target_closed_bucket(self):
        """目标走到攻击者旧位 ⇒ target_closed（送上门）。"""
        log = SimLog()
        for t in range(15, 20):
            log.frames.append(_frame(t, {1: 8, 2: 16}))
        log.frames.append(_frame(20, {1: 8, 2: 14}))  # 目标送到攻击者邻位
        log.events.append(_attack(20, 1, 2))
        d = chase_gain_distribution(log, _world())
        assert d["frac_target_closed"] == 1.0
        assert d["frac_already_adjacent_at_t3"] == 0.0
        assert d["frac_drop_ge_0p5"] == 0.0

    def test_lethal_hit_counted_in_chase(self):
        """追击分布统计全部命中（致死也计——追击是攻击者行为）。"""
        log = SimLog()
        for t in range(15, 21):
            log.frames.append(_frame(t, {1: 15, 2: 16}))
        log.events.append(_attack(20, 1, 2, status="ok"))
        d = chase_gain_distribution(log, _world())
        assert d["n_hits"] == 1

    def test_missed_not_counted(self):
        log = SimLog()
        for t in range(15, 21):
            log.frames.append(_frame(t, {1: 15, 2: 16}))
        log.events.append(_attack(20, 1, 2, status="attack_no_target"))
        d = chase_gain_distribution(log, _world())
        assert d["n_hits"] == 0
        assert d["n_window_available"] == 0

    def test_empty_log(self):
        d = chase_gain_distribution(SimLog(), _world())
        assert d["n_hits"] == 0 and d["n_window_available"] == 0
        assert d["frac_drop_ge_0p5"] == 0.0


class TestFleeGainDistribution:
    def test_wounded_moving_away(self):
        log = SimLog()
        for t in range(15, 21):
            log.frames.append(_frame(t, {1: 15, 2: 16}))
        for t in range(21, 24):
            log.frames.append(_frame(t, {1: 15, 2: 18 if t == 23 else 17}))
        log.events.append(_attack(20, 1, 2, status="attack_survived"))
        d = flee_gain_distribution(log, _world())
        assert d["n_wounded"] == 1 and d["n_lethal"] == 0
        assert d["n_window_available"] == 1
        # dist(18,15)=1+min(3,3)=4 − dist(16,15)=1 ⇒ gain 3（6 列环绕）
        assert d["gain_median"] == 3.0
        assert d["frac_gain_ge_0p5"] == 1.0

    def test_lethal_not_counted(self):
        """致死受害者无帧 ⇒ 不进逃避分布（B6 证据口径）。"""
        log = SimLog()
        for t in range(15, 21):
            log.frames.append(_frame(t, {1: 15, 2: 16}))
        log.frames.append(_frame(21, {1: 15}))     # 受害者死亡消失
        log.frames.append(_frame(23, {1: 15}))
        log.events.append(_attack(20, 1, 2, status="ok"))
        d = flee_gain_distribution(log, _world())
        assert d["n_lethal"] == 1 and d["n_wounded"] == 0
        assert d["n_window_available"] == 0

    def test_empty_log(self):
        d = flee_gain_distribution(SimLog(), _world())
        assert d["n_lethal"] == 0 and d["n_window_available"] == 0


class TestMetricsJsonIncludesDistributions:
    def test_keys_present(self):
        log = SimLog()
        for t in range(6):
            log.frames.append(_frame(t, {1: 15, 2: 16}))
        log.events.append(_attack(5, 1, 2))
        m = metrics_json(log, _world(), seed=1, ticks=6)
        assert "flee_gain_distribution" in m
        assert "chase_gain_distribution" in m
        assert m["chase_gain_distribution"]["n_hits"] == 1
