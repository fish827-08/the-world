"""B14.2（H5）：打斗反击风险测试。

派工要求：①弱者不可无损白打强者；②默认档"打同强度"仍可行；
③系数进 config；④默认档旧测试仍绿（由 tests/test_bio_resolve.py 等
既有套件覆盖——本文件只测新增行为）。
"""
import numpy as np
import pytest

from biosphere.action import ActionKind, Intent
from biosphere.develop import ControllerParams
from biosphere.genome import Genome
from biosphere.resolve import (
    Individual, Population, ResolveConfig, ResolveStatus, commit_intents,
    resolve_tick,
)
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


def _duel(atk_drive: float, tgt_defense: float, cfg: ResolveConfig = None,
          attacker_energy: float = 100.0,
          target_energy: float = 100.0):
    pop = Population()
    for uid, cell, en in [(0, 0, attacker_energy), (1, 1, target_energy)]:
        pop.individuals.append(Individual(
            uid=uid, genome=Genome(genes=np.zeros(64)), cell=cell, energy=en))
        pop.next_id = uid + 1
    ctrls = {0: _ctrl(attack_drive=atk_drive),
             1: _ctrl(defense=tgt_defense)}
    intents = [Intent(actor_id=0, kind=ActionKind.ATTACK, target_cell=0,
                      target_id=1, energy_cost=0.4)]
    rep = resolve_tick(pop, commit_intents(pop, intents), _world(),
                       _resource(), np.zeros(24), 0,
                       np.random.default_rng(0), ctrls, cfg=cfg)
    ev = [e for e in rep.events if e.kind == "attack"][0]
    return pop, ev


class TestCounterRisk:
    def test_same_strength_no_counter(self):
        """同强度（attack_drive=defense）⇒ excess=−4 ⇒ 无反击、攻击生效。"""
        pop, ev = _duel(0.5, 0.5)
        assert ev.status == ResolveStatus.ATTACK_SURVIVED
        assert ev.counter_damage == 0.0
        assert pop.individuals[0].energy == pytest.approx(99.6)   # 只付成本

    def test_weak_cannot_farm_strong(self):
        """弱者打强者：反击伤害 > 0 ⇒ 不是无损白打。"""
        pop, ev = _duel(0.05, 1.0)
        assert ev.counter_damage > 0.0
        # 攻击者净付 = 意图成本 0.4 + 反击 0.8 ⇒ 掉得比成本多
        assert pop.individuals[0].energy < 100.0 - 0.4 - 1e-9
        assert pop.individuals[0].energy == pytest.approx(98.8)

    def test_counter_transfers_to_defender(self):
        """反击能量攻击者付、防御方得（伤害本身是沉洞，pre-existing 口径）。"""
        pop, ev = _duel(0.05, 1.0)
        # 攻击者净付 = 意图成本 0.4 + 反击
        assert pop.individuals[0].energy == pytest.approx(
            100.0 - 0.4 - ev.counter_damage)
        # 防御方净收 = −伤害 + 反击（damage = power×(1−0.6·defense)）
        damage = (4.0 + 8.0 * 0.05) * (1.0 - 0.6 * 1.0)
        assert pop.individuals[1].energy == pytest.approx(
            100.0 - damage + ev.counter_damage)

    def test_repeated_attacks_drain_weak_attacker(self):
        """连续白打强者 ⇒ 弱者能量单调下降（不能刷）。"""
        cfg = ResolveConfig()
        pop = Population()
        for uid, cell, en in [(0, 0, 100.0), (1, 1, 100.0)]:
            pop.individuals.append(Individual(
                uid=uid, genome=Genome(genes=np.zeros(64)), cell=cell,
                energy=en))
            pop.next_id = uid + 1
        ctrls = {0: _ctrl(attack_drive=0.05), 1: _ctrl(defense=1.0)}
        energies = []
        for tick in range(6):
            intents = [Intent(actor_id=0, kind=ActionKind.ATTACK,
                              target_cell=0, target_id=1, energy_cost=0.4)]
            resolve_tick(pop, commit_intents(pop, intents), _world(),
                         _resource(), np.zeros(24), tick,
                         np.random.default_rng(tick), ctrls, cfg=cfg)
            energies.append(pop.individuals[0].energy)
        assert all(b < a for a, b in zip(energies, energies[1:]))

    def test_fail_threshold_configurable(self):
        """fail 门槛调低 ⇒ attack_failed：目标无伤、无掠食。"""
        pop, ev = _duel(0.0, 1.0,
                        cfg=ResolveConfig(counter_fail_threshold=3.0))
        assert ev.status == ResolveStatus.ATTACK_FAILED
        assert pop.individuals[1].energy == pytest.approx(101.0)  # 只得反击
        assert ev.counter_damage == pytest.approx(1.0)

    def test_counter_off_when_threshold_high(self):
        """counter_threshold 抬极高 ⇒ 反击关闭（参数化可回退）。"""
        pop, ev = _duel(0.0, 1.0,
                        cfg=ResolveConfig(counter_threshold=1e9))
        assert ev.counter_damage == 0.0
        assert ev.status == ResolveStatus.ATTACK_SURVIVED

    def test_defaults_in_config(self):
        cfg = ResolveConfig()
        assert cfg.counter_defense_scale == 8.0
        assert cfg.counter_threshold == 2.0
        assert cfg.counter_slope == 0.5
        assert cfg.counter_fail_threshold == 6.0


class TestCombatIntegration:
    def test_full_sim_deterministic_with_counter(self):
        def run():
            sim = BiosphereSimulator(BioMVPConfig(ticks=200, seed=42))
            sim.run()
            return [(i.uid, i.cell, i.energy)
                    for i in sim.population.individuals]
        assert run() == run()

    def test_attack_failed_events_recorded(self):
        """全模拟里出现 attack_failed 时其 counter_damage 也入账。"""
        sim = BiosphereSimulator(BioMVPConfig(ticks=400, seed=42))
        log = sim.run()
        failed = [e for e in log.events
                  if e.kind == "attack"
                  and e.status == ResolveStatus.ATTACK_FAILED]
        for e in failed:
            assert e.counter_damage > 0.0
