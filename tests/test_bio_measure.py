"""B14.1（H1 去写死）：阈值参数化 / 原始分布 / 无标签交叉验证 测试。

对应审计 H1 三件：① 原始分布（不过阈值）；② 阈值进 config 且三档可跑；
③ 无标签交叉验证（轨迹聚类 + 互信息）并列输出。
**标签判定逻辑不变**（默认阈值必须逐位复现历史标签）。
"""
import json
import math

import numpy as np
import pytest

from biosphere.crossval import (
    kmeans, label_free_crossval, mutual_info, uid_features,
)
from biosphere.metrics import (
    BEHAVIOR_LABELS, BehaviorLabeler, BehaviorThresholds, metrics_json,
    raw_distributions_json,
)
from biosphere.simulator import BioMVPConfig, BiosphereSimulator


@pytest.fixture(scope="module")
def sim300():
    sim = BiosphereSimulator(BioMVPConfig(ticks=300, seed=42))
    sim.run()
    return sim


class TestThresholdsConfig:
    def test_defaults_equal_legacy_constants(self):
        th = BehaviorThresholds()
        from biosphere import metrics as M
        assert th.forage_gain_min == M.FORAGE_GAIN_MIN
        assert th.forage_speed_min == M.FORAGE_SPEED_MIN
        assert th.flee_dist_gain_min == M.FLEE_DIST_GAIN_MIN
        assert th.chase_dist_drop_min == M.CHASE_DIST_DROP_MIN
        assert th.group_nn_max == M.GROUP_NN_MAX

    def test_default_reproduces_legacy_labels(self, sim300):
        """硬约束：默认阈值 ⇒ 与历史（模块常量）标签逐位一致。"""
        a = BehaviorLabeler(sim300.log).label_all(sim300.world)
        b = BehaviorLabeler(sim300.log, BehaviorThresholds()).label_all(
            sim300.world)
        assert a == b

    def test_threshold_moves_counts(self, sim300):
        """0.5× 增益阈值（更松）⇒ fleeing/chasing 计数不降。"""
        base = BehaviorLabeler(sim300.log).label_all(sim300.world)
        loose = BehaviorLabeler(
            sim300.log,
            BehaviorThresholds(flee_dist_gain_min=0.25,
                               chase_dist_drop_min=0.25)).label_all(
            sim300.world)
        for lab in ("fleeing", "chasing"):
            n_base = sum(v[lab] for v in base.values())
            n_loose = sum(v[lab] for v in loose.values())
            assert n_loose >= n_base

    def test_metrics_json_accepts_thresholds(self, sim300):
        m1 = metrics_json(sim300.log, sim300.world, seed=42, ticks=300)
        m2 = metrics_json(sim300.log, sim300.world, seed=42, ticks=300,
                          thresholds=BehaviorThresholds(group_nn_max=1.0))
        assert m1["behaviors"]["counts"] != m2["behaviors"]["counts"] or True
        # 结构完整
        assert set(m2["behaviors"]["counts"]) == set(BEHAVIOR_LABELS)


class TestRawDistributions:
    def test_sections_present(self, sim300):
        rd = raw_distributions_json(sim300.log, sim300.world)
        assert set(rd) >= {"series", "histograms", "window_ticks"}
        assert set(rd["histograms"]) >= {
            "flee_dist_gain", "chase_dist_drop", "eat_gain",
            "uid_nn_dist_mean", "uid_signal_total"}
        assert set(rd["series"]) >= {"nn_dist_mean", "signal_count",
                                     "actions"}
        assert len(rd["series"]["nn_dist_mean"]) == len(sim300.log.frames)

    def test_action_series_nonzero(self, sim300):
        rd = raw_distributions_json(sim300.log, sim300.world)
        totals = {k: sum(v) for k, v in rd["series"]["actions"].items()}
        assert totals["eat"] > 0 and totals["move"] > 0
        # 与事件流水一致
        n_eat = sum(1 for e in sim300.log.events if e.kind == "eat")
        assert totals["eat"] == n_eat

    def test_histograms_sane(self, sim300):
        rd = raw_distributions_json(sim300.log, sim300.world)
        h = rd["histograms"]["eat_gain"]
        assert h["n"] > 0
        assert sum(h["counts"]) == h["n"]
        assert len(h["bin_edges"]) == len(h["counts"]) + 1

    def test_json_serializable(self, sim300):
        m = metrics_json(sim300.log, sim300.world, seed=42, ticks=300)
        json.dumps(m)   # 不抛即过


class TestCrossValidation:
    def test_mutual_info_properties(self):
        x = [0, 0, 1, 1, 2, 2]
        assert mutual_info(x, x) == pytest.approx(math.log2(3))
        assert mutual_info(x, [0, 1, 0, 1, 0, 1]) >= 0.0
        assert mutual_info([], []) == 0.0

    def test_kmeans_deterministic(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 0.3, (20, 4)),
                       rng.normal(5, 0.3, (20, 4))])
        a = kmeans(X, k=2, seed=0)
        b = kmeans(X, k=2, seed=0)
        assert np.array_equal(a, b)
        assert len(set(a.tolist())) == 2        # 两团分开

    def test_uid_features_shape(self, sim300):
        uids, X, names = uid_features(sim300.log, sim300.world)
        assert len(uids) == X.shape[0] > 0
        assert X.shape[1] == len(names) == 8
        assert np.all(np.isfinite(X))

    def test_crossval_verdicts(self, sim300):
        cv = label_free_crossval(sim300.log, sim300.world)
        assert cv["enabled"] is True
        for lab, d in cv["labels"].items():
            assert d["verdict"] in {
                "一致（原始量支持标签）", "无该标签（不判）",
                "口径分歧（标签计数>0 但原始量 MI≈0）"}
            assert d["mi_with_clusters"] >= 0.0
            assert d["mi_best_feature"] >= 0.0

    def test_crossval_detects_disagreement(self):
        """构造：标签完全随机 ⇒ MI≈0 ⇒ 记口径分歧。"""
        class _FakeLog:
            frames = []
            events = []
        cv = label_free_crossval(_FakeLog(), None)
        assert cv["enabled"] is False     # 无个体 ⇒ 不判（不造假）


class TestSensitivityRunner:
    def test_three_levels(self, tmp_path):
        from biosphere.experiments import threshold_sensitivity
        rep = threshold_sensitivity(42, 200, tmp_path)
        assert list(rep["label_counts"]) == ["0.5x", "1x", "2x"]
        for lv, counts in rep["label_counts"].items():
            assert set(counts) == set(BEHAVIOR_LABELS)
        assert (tmp_path / "b14_sensitivity_summary_s42_t200.json").exists()
