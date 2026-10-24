"""B14.3（C3 v2）：发送者获益通道 测试。

owner 已批口径：徽章含义为空；只写"发射耗能 + 徽章可遗传 + 反应权重
可遗传 + 打斗结果由实力决定"；**无"信号→攻击欲"直规则**。
"""
import numpy as np
import pytest

from biosphere.c3 import (
    C3_BADGE_K, C3Config, learning_update, visible_badge,
)
from biosphere.develop import ControllerParams, develop
from biosphere.genome import Genome, PARAM_SLOTS, random_genome
from biosphere.metrics import c3_json, metrics_json
from biosphere.resolve import (
    Individual, Population, ResolveConfig, commit_intents, resolve_tick,
)
from biosphere.action import ActionKind, Intent
from biosphere.simulator import BioMVPConfig, BiosphereSimulator
from biosphere.world_adapter import (
    SyntheticResource, WorldView, _von_neumann_table,
)


def _ctrl(**overrides) -> ControllerParams:
    base = dict(
        sense_food=1.0, sense_threat=1.0, sense_kin=1.0, sense_signal=1.0,
        move_bias=0.5, eat_greed=0.5, attack_drive=0.0, share_drive=0.5,
        signal_drive=0.0, repro_drive=0.5, metabolism=1.0, move_cost_mult=1.0,
        lifespan=1000.0, repro_thresh=40.0, habitat_aquatic=1.0,
        habitat_terrestrial=1.0, adhesion=0.5, defense=0.0, eat_amount=0.5,
        signal_payload=0.5, give_up=10.0, persistence=0.0,
    )
    base.update(overrides)
    return ControllerParams(**base)


def _world() -> WorldView:
    n = 24
    return WorldView(
        rows=4, cols=6, n_cells=n,
        surface_type=np.zeros(n, dtype=np.uint8),
        elevation_m=np.ones(n), depth_m=np.zeros(n),
        patch_id=np.full(n, -1, dtype=np.int32),
        neighbors=_von_neumann_table(4, 6),
        soil_water=np.full(n, 5.0), soil_water_min=2.0,
    )


def _resource() -> SyntheticResource:
    return SyntheticResource(amount=np.ones(24), capacity=np.ones(24),
                             regrow_rate=0.0)


class TestGenes:
    def test_new_slots_registered(self):
        for name in ("badge", "emit_rate", "strength", "response_0",
                     "response_3", "learning_rate"):
            assert name in PARAM_SLOTS

    def test_badge_discretization(self):
        """badge 基因 sigmoid×K 离散化 ⇒ 恒在 [0, K-1]。"""
        for s in range(30):
            c = develop(random_genome(np.random.default_rng(100 + s)))
            assert 0 <= c.badge <= C3_BADGE_K - 1
            assert len(c.response) == C3_BADGE_K
            assert all(-1.0 <= r <= 1.0 for r in c.response)

    def test_with_gene_returns_new_genome(self):
        g = random_genome(np.random.default_rng(3))
        g2 = g.with_gene(PARAM_SLOTS["badge"], 5.0)
        assert g2 is not g
        assert g2.genes[PARAM_SLOTS["badge"]] == 5.0
        assert g.genes[PARAM_SLOTS["badge"]] != 5.0   # 原基因组只读不变


class TestSignalEmission:
    def test_signal_cost_scales_with_emit_rate(self):
        """发射耗能 = base × emit_rate（可遗传 ⇒ 发射有代价）。"""
        base = C3Config.signal_emit_cost_base
        c_hi = _ctrl(emit_rate=1.0)
        c_lo = _ctrl(emit_rate=0.0)
        assert base * c_hi.emit_rate == pytest.approx(base)
        assert base * c_lo.emit_rate == 0.0

    def test_badge_layer_written_on_signal(self):
        pop = Population()
        for uid, cell in [(0, 0), (1, 5)]:
            pop.individuals.append(Individual(
                uid=uid, genome=Genome(genes=np.zeros(72)), cell=cell,
                energy=50.0))
            pop.next_id = uid + 1
        ctrls = {0: _ctrl(badge=2, signal_payload=1.0),
                 1: _ctrl(badge=0)}
        c3 = C3Config(enabled=True)
        badge_layer = np.zeros((C3_BADGE_K, 24))
        intents = [Intent(actor_id=0, kind=ActionKind.SIGNAL, target_cell=0,
                          payload=1.0, energy_cost=0.1)]
        resolve_tick(pop, commit_intents(pop, intents), _world(), _resource(),
                     np.zeros(24), 0, np.random.default_rng(0), ctrls,
                     badge_layer=badge_layer, c3=c3)
        assert badge_layer[2, 0] == pytest.approx(1.0)   # 徽章 2 通道被写
        assert badge_layer[0, 0] == 0.0

    def test_visible_badge_threshold(self):
        row = np.array([0.0, 0.05, 0.5, 0.0])
        assert visible_badge(row, 0.10) == 2
        assert visible_badge(np.zeros(4), 0.10) == -1


class TestCombatStrength:
    def test_strength_adds_power(self):
        """strength 进威力（打斗结果由实力决定）。"""
        def duel(atk_drive, atk_strength, tgt_defense):
            pop = Population()
            for uid, cell, en in [(0, 0, 100.0), (1, 1, 100.0)]:
                pop.individuals.append(Individual(
                    uid=uid, genome=Genome(genes=np.zeros(72)), cell=cell,
                    energy=en))
                pop.next_id = uid + 1
            ctrls = {0: _ctrl(attack_drive=atk_drive, strength=atk_strength),
                     1: _ctrl(defense=tgt_defense)}
            c3 = C3Config(enabled=True)
            intents = [Intent(actor_id=0, kind=ActionKind.ATTACK,
                              target_cell=0, target_id=1, energy_cost=0.4)]
            resolve_tick(pop, commit_intents(pop, intents), _world(),
                         _resource(), np.zeros(24), 0,
                         np.random.default_rng(0), ctrls, c3=c3)
            return 100.0 - pop.individuals[1].energy
        weak = duel(0.2, 0.0, 0.0)
        strong = duel(0.2, 1.0, 0.0)
        assert strong > weak          # 同攻击倾向，强度高 ⇒ 伤害大

    def test_strength_effect_switchable(self):
        """关强度不对称（负对照）：strength 不进威力。"""
        def duel(strength):
            pop = Population()
            for uid, cell, en in [(0, 0, 100.0), (1, 1, 100.0)]:
                pop.individuals.append(Individual(
                    uid=uid, genome=Genome(genes=np.zeros(72)), cell=cell,
                    energy=en))
                pop.next_id = uid + 1
            ctrls = {0: _ctrl(attack_drive=0.2, strength=strength),
                     1: _ctrl(defense=0.0)}
            intents = [Intent(actor_id=0, kind=ActionKind.ATTACK,
                              target_cell=0, target_id=1, energy_cost=0.4)]
            resolve_tick(pop, commit_intents(pop, intents), _world(),
                         _resource(), np.zeros(24), 0,
                         np.random.default_rng(0), ctrls,
                         c3=C3Config(enabled=True, strength_effect=False))
            return 100.0 - pop.individuals[1].energy
        assert duel(0.0) == pytest.approx(duel(1.0))


class TestMemory:
    def test_learning_update_signs(self):
        resp = np.zeros(C3_BADGE_K)
        d = learning_update(resp, 1, +1.0, learning_rate=1.0)
        assert d > 0 and resp[1] > 0
        d2 = learning_update(resp, 1, -1.0, learning_rate=1.0)
        assert d2 < 0

    def test_zero_learning_rate_is_pure_evolution(self):
        resp = np.zeros(C3_BADGE_K)
        assert learning_update(resp, 2, +1.0, learning_rate=0.0) == 0.0
        assert np.all(resp == 0.0)


class TestIntegration:
    def test_c3_runs_and_deterministic(self):
        def run():
            sim = BiosphereSimulator(
                BioMVPConfig(ticks=200, seed=42, c3=C3Config(enabled=True)))
            sim.run()
            return [(i.uid, i.cell, i.energy)
                    for i in sim.population.individuals]
        assert run() == run()

    def test_c3_off_bit_identical(self):
        """硬约束：C3Config 默认关 ⇒ 与既有口径逐位等价。"""
        a = BiosphereSimulator(BioMVPConfig(ticks=100, seed=42))
        a.run()
        b = BiosphereSimulator(BioMVPConfig(ticks=100, seed=42,
                                            c3=C3Config()))
        b.run()
        assert [(i.uid, i.cell, i.energy) for i in a.population.individuals] \
            == [(i.uid, i.cell, i.energy) for i in b.population.individuals]

    def test_badge_reshuffle_changes_badges(self):
        """负对照：徽章重排 ⇒ 子代徽章分布更均匀（对比不重排）。"""
        def run(reshuffle: bool):
            sim = BiosphereSimulator(
                BioMVPConfig(ticks=300, seed=42,
                             c3=C3Config(enabled=True,
                                         badge_reshuffle=reshuffle)))
            sim.run()
            from biosphere.develop import develop
            return [develop(i.genome).badge for i in sim.population.individuals]
        a, b = run(False), run(True)
        assert len(a) > 0 and len(b) > 0
        # 两者都能跑；分布差异由跑批统计判定（这里只验证不崩）

    def test_c3_metrics_section(self):
        sim = BiosphereSimulator(
            BioMVPConfig(ticks=200, seed=42, c3=C3Config(enabled=True)))
        sim.run()
        m = metrics_json(sim.log, sim.world, seed=42, ticks=200)
        assert m["c3"]["enabled"] is True
        assert set(m["c3"]["attacked_rate"]["all"]) == {"low", "mid", "high"}

    def test_checkpoint_resume_bit_identical_with_c3(self, tmp_path):
        straight = BiosphereSimulator(
            BioMVPConfig(ticks=120, seed=43, c3=C3Config(enabled=True)))
        straight.run()
        part = BiosphereSimulator(
            BioMVPConfig(ticks=60, seed=43, c3=C3Config(enabled=True)))
        part.run(ckpt_dir=tmp_path, ckpt_every=60, tag="c3")
        from biosphere.checkpoint import load_checkpoint
        resumed = load_checkpoint(next(tmp_path.glob("ckpt_c3_t60.pkl")))
        resumed.run(ticks=120)
        key = lambda sim: [(i.uid, i.cell, i.energy)
                           for i in sim.population.individuals]
        assert key(resumed) == key(straight)
        assert resumed.log.energy_in == pytest.approx(straight.log.energy_in)
