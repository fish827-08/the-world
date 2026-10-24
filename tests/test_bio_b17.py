"""B17：生态门诊断的 uid 修复 + B17.2 命中即转移（contact_loot）。

钉住两件事：
1. ``Genome.with_gene`` 必须保留 uid（否则 ControllerCache 碰撞 ⇒ 整群同表型）；
2. ``contact_loot`` 默认关闭逐位等价；开启时命中存活转移
   ``attack_loot_frac × min(damage, 命中前能量)``，致死分支不变。
"""
import numpy as np
import pytest

from biosphere.action import ActionKind, Intent
from biosphere.develop import ControllerParams
from biosphere.experiments import build_b16_config
from biosphere.genome import Genome, PARAM_SLOTS, random_genome
from biosphere.metrics import run_digest
from biosphere.resolve import (
    Individual, Population, ResolveConfig, ResolveStatus, commit_intents,
    resolve_tick,
)
from biosphere.simulator import BioMVPConfig, BiosphereSimulator
from biosphere.world_adapter import (
    SyntheticResource, WorldView, _von_neumann_table,
)

TIER = "mainline"


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


def _duel(atk_drive: float, tgt_defense: float, cfg: ResolveConfig,
          target_energy: float = 100.0):
    pop = Population()
    for uid, cell, en in [(0, 0, 100.0), (1, 1, float(target_energy))]:
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


class TestWithGeneUid:
    def test_with_gene_preserves_uid(self):
        g = random_genome(np.random.default_rng(0), uid=17)
        g2 = g.with_gene(PARAM_SLOTS["badge"], 1.0)
        assert g2.uid == 17

    def test_spread_founders_have_distinct_controllers(self):
        """uid 修复回归：spread>0 的初代不得退化为同一表型。"""
        cfg = build_b16_config("b16_full", 44, 10, spread=2.0)
        sim = BiosphereSimulator(cfg)
        atk = np.array([sim.cache.get(ind.genome).attack_drive
                        for ind in sim.population.individuals])
        dfn = np.array([sim.cache.get(ind.genome).defense
                        for ind in sim.population.individuals])
        assert atk.std() > 0.1
        assert dfn.std() > 0.1

    def test_cache_keys_differ_after_with_gene(self):
        rng = np.random.default_rng(1)
        g1 = random_genome(rng, uid=0)
        g2 = random_genome(rng, uid=1)
        g1 = g1.with_gene(PARAM_SLOTS["attack_drive"], 3.0)
        g2 = g2.with_gene(PARAM_SLOTS["attack_drive"], -3.0)
        sim = BiosphereSimulator(BioMVPConfig(ticks=1, seed=0))
        c1 = sim.cache.get(g1)
        c2 = sim.cache.get(g2)
        assert c1.attack_drive != c2.attack_drive


class TestContactLoot:
    def test_default_off(self):
        assert ResolveConfig().contact_loot is False

    def test_off_no_loot_on_nonlethal(self):
        pop, ev = _duel(0.5, 0.5, ResolveConfig())
        assert ev.status == ResolveStatus.ATTACK_SURVIVED
        assert ev.loot == 0.0
        assert pop.individuals[0].energy == pytest.approx(99.6)
        assert pop.individuals[1].energy == pytest.approx(100.0 - 5.6)

    def test_on_transfers_fraction_of_damage(self):
        pop, ev = _duel(0.5, 0.5, ResolveConfig(contact_loot=True))
        assert ev.status == ResolveStatus.ATTACK_SURVIVED
        assert ev.loot == pytest.approx(0.5 * 5.6)
        assert pop.individuals[0].energy == pytest.approx(99.6 + 2.8)
        assert pop.individuals[1].energy == pytest.approx(100.0 - 5.6)

    def test_lethal_branch_unchanged(self):
        off_pop, off_ev = _duel(0.5, 0.5, ResolveConfig(),
                                target_energy=5.0)
        on_pop, on_ev = _duel(0.5, 0.5, ResolveConfig(contact_loot=True),
                              target_energy=5.0)
        assert off_ev.status == on_ev.status == ResolveStatus.OK
        assert off_ev.loot == pytest.approx(2.5)
        assert on_ev.loot == pytest.approx(2.5)
        assert on_pop.individuals[0].energy == pytest.approx(
            off_pop.individuals[0].energy)

    def test_zero_loot_frac_disables_pathway(self):
        pop, ev = _duel(
            0.5, 0.5,
            ResolveConfig(contact_loot=True, attack_loot_frac=0.0))
        assert ev.loot == 0.0
        assert pop.individuals[0].energy == pytest.approx(99.6)

    def test_default_config_bit_identical_sim(self):
        """默认关 vs 显式关：同一 seed 的整条轨迹逐位一致。"""
        def _digest(cfg):
            sim = BiosphereSimulator(cfg)
            return run_digest(sim.run())

        base = BioMVPConfig(ticks=200, seed=42)
        explicit = BioMVPConfig(ticks=200, seed=42,
                                resolve=ResolveConfig(contact_loot=False))
        assert _digest(base) == _digest(explicit)


class TestB17Arms:
    def test_arm_flags(self):
        full = build_b16_config("b17_full", 41, 10)
        off = build_b16_config("b17_off", 41, 10)
        no_loot = build_b16_config("b17_no_loot", 41, 10)
        no_def = build_b16_config("b17_no_defense", 41, 10)
        assert full.resolve.contact_loot is True
        assert off.resolve.contact_loot is False
        assert no_loot.resolve.contact_loot is True
        assert no_loot.resolve.attack_loot_frac == 0.0
        assert no_def.resolve.defense_mit == 0.0
        assert no_def.resolve.counter_defense_scale == 0.0
        assert full.predation.init_trait_spread == 2.0
