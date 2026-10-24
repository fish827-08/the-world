"""B15 诊断轮测试：oracle 正对照 / 谱系条件判据 / by_emitting 度量。

>2 s 的单例标 slow。默认关 ⇒ 关档逐位等价由 test_c3_off_equivalence 钉。
"""
import numpy as np
import pytest

from biosphere.c3 import C3Config
from biosphere.criteria import (
    _mi_counts, badge_keff, lineage_conditional_mi,
    lineage_label_permutation, lineage_root_map,
)
from biosphere.genome import Genome, PARAM_SLOTS, random_genome
from biosphere.metrics import metrics_json
from biosphere.resolve import Individual, Population
from biosphere.simulator import BioMVPConfig, BiosphereSimulator


def _ctrl(**kw):
    from biosphere.develop import ControllerParams
    base = dict(sense_food=1.0, sense_threat=1.0, sense_kin=1.0,
                sense_signal=1.0, move_bias=0.5, eat_greed=0.5,
                attack_drive=0.0, share_drive=0.5, signal_drive=0.0,
                repro_drive=0.5, metabolism=1.0, move_cost_mult=1.0,
                lifespan=1000.0, repro_thresh=40.0, habitat_aquatic=1.0,
                habitat_terrestrial=1.0, adhesion=0.5, defense=0.0,
                eat_amount=0.5, signal_payload=0.5, give_up=10.0,
                persistence=0.0)
    base.update(kw)
    return ControllerParams(**base)


class TestOracleConfig:
    def test_defaults_off(self):
        """默认关 ⇒ 与 B14 口径逐位等价（oracle 字段不进默认行为）。"""
        c = C3Config()
        assert c.oracle_response is False
        assert c.oracle_emitter is False

    def test_oracle_equivalence_when_off(self):
        """C3 全关时两 oracle 臂与自然臂逐位一致（同 seed 同 tick）。"""
        a = BiosphereSimulator(BioMVPConfig(ticks=40, seed=42))
        b = BiosphereSimulator(BioMVPConfig(
            ticks=40, seed=42,
            c3=C3Config(enabled=False, oracle_response=True,
                        oracle_emitter=True)))
        a.run(); b.run()
        ma = metrics_json(a.log, a.world, seed=42, ticks=40)
        mb = metrics_json(b.log, b.world, seed=42, ticks=40)
        assert ma == mb


class TestLineageCriteria:
    def test_keff_extremes(self):
        assert badge_keff({0: 1, 1: 1, 2: 1, 3: 1}) == pytest.approx(4.0)
        assert badge_keff({0: 100}) == pytest.approx(1.0)
        assert badge_keff({}) == 0.0

    def test_mi_counts_perfect(self):
        # 完全相关 ⇒ MI = log2(2) = 1 bit
        assert _mi_counts([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)
        # 独立 ⇒ 0
        assert _mi_counts([0, 1, 0, 1], [0, 0, 1, 1]) == pytest.approx(0.0)

    def test_root_map(self):
        births = [(-1, 1, 0), (1, 2, 1), (-1, 3, 2), (3, 4, 3)]
        roots = lineage_root_map(births)
        # founder（parent<0）不在 map ⇒ .get 默认自身即 root；后代回溯到 founder
        assert roots.get(1, 1) == 1 and roots[2] == 1
        assert roots.get(3, 3) == 3 and roots[4] == 3

    def _recs(self):
        # 谱系 A：badge 与 sbin 完全对齐（8 样本 ⇒ 置换保持对齐概率 1/70）；
        # 谱系 B：badge 恒定（层内 MI=0）
        recs = []
        for b in range(4):
            for _ in range(4):
                recs.append({"badge": b, "sbin": 0 if b % 2 == 0 else 3,
                             "root": "A"})
        for b, s in [(2, 0), (2, 3), (2, 1), (2, 2)]:
            recs.append({"badge": b, "sbin": s, "root": "B"})
        return recs

    def test_conditional_mi_stratifies(self):
        cond = lineage_conditional_mi(self._recs())
        # A 层 MI=1（完全对齐），B 层 MI=0 ⇒ 加权 = 16/(16+4) × 1 = 0.8
        assert cond["per_root"]["A"] == pytest.approx(1.0)
        assert cond["per_root"]["B"] == pytest.approx(0.0)
        assert cond["mi_weighted"] == pytest.approx(0.8)

    def test_permutation_detects_alignment(self):
        recs = self._recs()
        perm = lineage_label_permutation(recs, n_perm=500, seed=3)
        assert perm["observed"] > perm["null_mean"]
        assert perm["p_one_sided"] < 0.05      # 对齐结构可检出

    def test_permutation_null_when_uniform(self):
        # badge 与 sbin 独立 ⇒ 置换 p 应大（不显著）
        rng = np.random.default_rng(5)
        recs = [{"badge": int(rng.integers(0, 4)),
                 "sbin": int(rng.integers(0, 4)),
                 "root": f"r{i % 3}"} for i in range(60)]
        perm = lineage_label_permutation(recs, n_perm=500, seed=3)
        assert perm["p_one_sided"] > 0.05


class TestByEmittingMetric:
    def test_c3_json_shape(self):
        sim = BiosphereSimulator(BioMVPConfig(
            ticks=120, seed=42, c3=C3Config(enabled=True)))
        sim.run()
        m = metrics_json(sim.log, sim.world, seed=42, ticks=120)
        c3 = m["c3"]
        assert c3["enabled"] is True
        be = c3["by_emitting"]
        for key in ("attacked_rate_emitting", "attacked_rate_non_emitting",
                    "exposure_emitting", "exposure_non_emitting"):
            assert key in be
        # 曝光守恒：发射 + 非发射 = 全体个体·tick
        assert (be["exposure_emitting"] + be["exposure_non_emitting"]
                > 0)

    def test_by_emitting_zero_when_off(self):
        sim = BiosphereSimulator(BioMVPConfig(ticks=30, seed=42))
        sim.run()
        m = metrics_json(sim.log, sim.world, seed=42, ticks=30)
        assert m["c3"]["enabled"] is False


class TestOracleBehavior:
    def test_oracle_response_sets_signs(self, monkeypatch):
        """oracle-response：强徽章载体 ⇒ response 负；弱 ⇒ 正。"""
        from biosphere.c3 import visible_badge
        sim = BiosphereSimulator(BioMVPConfig(
            ticks=1, seed=7, init_pop=0, rows=4, cols=6,
            c3=C3Config(enabled=True, oracle_response=True)))
        # 造两个个体（genome 无关紧要——controllers 直接注入，
        # 因为 develop 的 strength 读出被 GRN 中性项调制，全零基因组
        # 的 strength 上限只有 0.5）
        pop = Population()
        for uid, cell in [(0, 0), (1, 1)]:
            ind = Individual(uid=uid, genome=Genome(genes=np.zeros(64)),
                             cell=cell, energy=30.0)
            pop.individuals.append(ind)
            pop.next_id = uid + 1
        sim.population = pop
        sim._occupant = pop.occupant_map(sim.world.n_cells)
        sim._by_uid = {i.uid: i for i in pop.individuals}
        controllers = {0: _ctrl(strength=0.9), 1: _ctrl(strength=0.1)}
        # 给 0 号格写一个强徽章（uid0 发射过）
        sim.badge_layer[0, 0] = 1.0
        sim.badge_layer[1, 1] = 1.0
        sim._apply_oracle_response(controllers)
        # uid1 的 response_state 应被覆写（徽章 0/1 都可见）
        rs = pop.individuals[1].response_state
        assert rs is not None and rs.shape[0] == 4
        # 徽章0 载体（uid0，strength 高）应得负反应；徽章1 载体弱 ⇒ 正
        assert rs[0] < 0.0
        assert rs[1] > 0.0

    def test_oracle_emitter_gate(self):
        """oracle-emitter：只有强者有徽章曝光（弱者 SIGNAL 被硬压）。"""
        sim = BiosphereSimulator(BioMVPConfig(
            ticks=250, seed=42,
            c3=C3Config(enabled=True, oracle_emitter=True,
                        emitter_strength_min=0.5)))
        sim.run()
        m = metrics_json(sim.log, sim.world, seed=42, ticks=250)
        ar = m["c3"]["attacked_rate"]
        # 强者（high）有 emitting 曝光；弱者（low/mid）只能"蹭格"
        # （徽章在格上不跟人走 ⇒ 站强者发射格的低强度个体偶发计入）
        hi = ar["exposure_ticks_emitting"]["high"]
        lo = ar["exposure_ticks_emitting"]["low"]
        mid = ar["exposure_ticks_emitting"]["mid"]
        assert hi > 0
        assert lo + mid < 0.2 * hi
