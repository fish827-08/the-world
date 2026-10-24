"""B4：度量模块测试（五类行为标签可判别 + 输出形状 + 谱系/交互）。

覆盖 B1 metrics.py：
- 用手工帧 + 事件流水**逐一证伪**五类标签可检出（觅食/逃避/追击/结群/信号），
  且不相关个体不误报；
- metrics_json 定键形状（schema/种群/能量/行为/谱系/交互/动作计数）；
- lineage_json 世代递推与根系数；
- interaction_json 边排序与计数。

标签阈值来自预注册常量（metrics.py 头部），本测试不重算物理常数，
只验证"规则 + 阈值"在构造数据上的行为符合定义。
"""
import json

import numpy as np

from biosphere.metrics import (
    BEHAVIOR_LABELS, SCHEMA, BehaviorLabeler, SimLog,
    interaction_json, lineage_json, metrics_json,
)
from biosphere.resolve import ResolveEvent

# 小世界（6 列，避开环绕歧义）
ROWS, COLS = 4, 6


def _world():
    from biosphere.world_adapter import WorldView, _von_neumann_table
    n = ROWS * COLS
    return WorldView(
        rows=ROWS, cols=COLS, n_cells=n,
        surface_type=np.zeros(n, dtype=np.uint8),
        elevation_m=np.zeros(n), depth_m=np.zeros(n),
        patch_id=np.full(n, -1, dtype=np.int32),
        neighbors=_von_neumann_table(ROWS, COLS),
    )


def _craft_log() -> SimLog:
    """六个体 60 tick 的手工日志（位置/事件逐 tick 写死）。"""
    log = SimLog()
    T = 60
    # 位置表：uid → tick → cell
    pos = {
        1: {t: (0 if t % 2 == 0 else 1) for t in range(T)},          # 觅食者往返
        2: {t: (8 if t <= 17 else (14 if t == 18 else 15)) for t in range(T)},  # 追击者
        3: {t: (16 if t <= 20 else (17 if t == 21 else 18)) for t in range(T)},  # 逃亡者
        4: {t: 3 for t in range(T)},                                 # 结群 A
        5: {t: 4 for t in range(T)},                                 # 结群 B
        6: {t: 20 for t in range(T)},                                # 信号者
    }
    for t in range(T):
        uids = np.array(sorted(pos), dtype=np.int64)
        cells = np.array([pos[u][t] for u in sorted(pos)], dtype=np.int64)
        energies = np.full(len(uids), 50.0)
        # 最近邻：4-5 恒相邻；2-3 仅 tick 20 相邻；6 在 tick 18-19 邻 2
        nn = []
        for u in sorted(pos):
            if u in (4, 5):
                nn.append(1.0)
            elif u == 2:
                nn.append(1.0 if t == 20 else 2.5)
            elif u == 3:
                nn.append(1.0 if t == 20 else 2.5)
            elif u == 6:
                nn.append(1.0 if t in (18, 19) else 2.5)
            else:
                nn.append(2.5)
        log.frames.append(_frame(t, uids, cells, energies,
                                 np.array(nn, dtype=np.float64)))
    # 事件：1 每 tick 吃（净增益 0.5）；2 在 tick 20 攻击 3；6 每 tick 发信号
    for t in range(T):
        log.events.append(ResolveEvent(
            t, 1, "eat", "ok", pos[1][t], -1, 0.45))
        log.events.append(ResolveEvent(
            t, 6, "signal", "signal_ok", pos[6][t], -1, -0.15))
    log.events.append(ResolveEvent(20, 2, "attack", "ok", 15, 3, -0.4))
    return log


def _frame(tick, uids, cells, energies, nn):
    from biosphere.metrics import Frame
    return Frame(tick=tick, uids=uids, cells=cells, energies=energies,
                 nn_dist=nn)


class TestBehaviorLabels:
    def test_all_five_labels_detectable(self):
        log = _craft_log()
        labeler = BehaviorLabeler(log)
        labels = labeler.label_all(_world())
        expect = {
            1: {"foraging": True, "fleeing": False, "chasing": False,
                "grouping": False, "signaling": False},
            2: {"foraging": False, "fleeing": False, "chasing": True,
                "grouping": False, "signaling": False},
            3: {"foraging": False, "fleeing": True, "chasing": False,
                "grouping": False, "signaling": False},
            4: {"foraging": False, "fleeing": False, "chasing": False,
                "grouping": True, "signaling": False},
            5: {"foraging": False, "fleeing": False, "chasing": False,
                "grouping": True, "signaling": False},
            6: {"foraging": False, "fleeing": False, "chasing": False,
                "grouping": False, "signaling": True},
        }
        for uid, exp in expect.items():
            assert labels[uid] == exp, f"uid={uid} 标签不符: {labels[uid]}"

    def test_label_keys_fixed(self):
        labels = BehaviorLabeler(_craft_log()).label_all(_world())
        for lab_map in labels.values():
            assert tuple(sorted(lab_map)) == tuple(sorted(BEHAVIOR_LABELS))

    def test_window_restriction(self):
        """窗口收窄到前 10 tick：追击（tick 20）与逃避都消失。"""
        log = _craft_log()
        labeler = BehaviorLabeler(log)
        labels = labeler.label_individual(2, _world(), 0, 9)
        assert labels["chasing"] is False
        labels3 = labeler.label_individual(3, _world(), 0, 9)
        assert labels3["fleeing"] is False

    def test_block_series_shape(self):
        log = _craft_log()
        series = BehaviorLabeler(log).block_series(_world(), block=30)
        assert set(series) == set(BEHAVIOR_LABELS)
        assert all(len(v) == 2 for v in series.values())
        assert series["signaling"] == [1, 1]


class TestMetricsJson:
    def test_shape_and_consistency(self):
        log = _craft_log()
        m = metrics_json(log, _world(), seed=42, ticks=60)
        assert m["schema"] == SCHEMA
        assert m["seed"] == 42 and m["ticks"] == 60
        pop = m["population"]
        assert pop["final_alive"] == 6
        assert pop["peak_alive"] == 6
        assert len(pop["alive_series"]) == 60
        assert len(pop["energy_series"]) == 60
        assert m["world"]["rows"] == ROWS and m["world"]["cols"] == COLS
        beh = m["behaviors"]["counts"]
        assert beh == {"foraging": 1, "fleeing": 1, "chasing": 1,
                       "grouping": 2, "signaling": 1}
        # 动作计数含两类事件 + 攻击
        assert m["action_counts"]["eat:ok"] == 60
        assert m["action_counts"]["signal:signal_ok"] == 60
        assert m["action_counts"]["attack:ok"] == 1

    def test_json_serializable_deterministic(self):
        log = _craft_log()
        m = metrics_json(log, _world(), seed=42, ticks=60)
        s1 = json.dumps(m, sort_keys=True)
        s2 = json.dumps(metrics_json(_craft_log(), _world(), seed=42,
                                     ticks=60), sort_keys=True)
        assert s1 == s2  # 同构造数据 ⇒ JSON 逐位一致


class TestLineageJson:
    def test_generation_chain(self):
        log = SimLog()
        log.births = [(0, 1, 5), (1, 2, 6), (0, 3, 7)]
        out = lineage_json(log)
        assert out["n_individuals"] == 4
        assert out["n_roots"] == 1
        assert out["max_generation"] == 2
        assert out["nodes"]["0"]["generation"] == 0
        assert out["nodes"]["1"]["generation"] == 1
        assert out["nodes"]["2"]["generation"] == 2
        assert out["nodes"]["3"]["generation"] == 1

    def test_death_recorded(self):
        log = SimLog()
        log.deaths = [(1, "starved", 100)]
        out = lineage_json(log)
        assert out["nodes"]["1"]["death_tick"] == 100
        assert out["nodes"]["1"]["death_cause"] == "starved"
        assert out["nodes"]["1"]["alive"] is False


class TestInteractionJson:
    def test_edges_and_counts(self):
        log = SimLog()
        log.events = [
            ResolveEvent(0, 1, "attack", "ok", 0, 2, -0.4),
            ResolveEvent(1, 1, "attack", "ok", 0, 2, -0.4),
            ResolveEvent(2, 3, "attack", "ok", 5, 4, -0.4),
            ResolveEvent(3, 5, "share", "ok", 0, 6, -5.0),
            ResolveEvent(4, 7, "signal", "signal_ok", 0, -1, -0.15),
            ResolveEvent(5, 7, "signal", "signal_ok", 0, -1, -0.15),
        ]
        out = interaction_json(log)
        assert out["attack_edges"] == [[1, 2, 2], [3, 4, 1]]
        assert out["share_edges"] == [[5, 6, 1]]
        assert out["signal_counts"] == [[7, 2]]
        assert out["n_attack_pairs"] == 2
        assert out["n_signalers"] == 1

    def test_failed_attacks_not_counted(self):
        log = SimLog()
        log.events = [ResolveEvent(0, 1, "attack", "attack_no_target", 0, 2, -0.4)]
        out = interaction_json(log)
        assert out["attack_edges"] == []
        assert out["n_attack_pairs"] == 0
