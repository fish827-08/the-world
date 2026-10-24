"""B13（W4c）：三旋钮机制修正测试（预注册 §13.12；每组只改一个旋钮）。

- G1 组繁殖门槛参数化（{40,80,160}）：低门槛 ⇒ 组繁殖事件更多；
- G2 亲缘结构（kin_cohesion）：计数加权贴团奖励 + 粘附个体降扩散；
- G3 粘附收益（adhesion_benefit）：群体觅食分红，记账进 energy_in；
- 硬约束回归：W4Config 默认关 ⇒ 关档逐位等价；旋钮开时确定性；
  checkpoint 续跑 = 直跑逐位。
"""
import numpy as np
import pytest

from biosphere.develop import ControllerParams
from biosphere.genome import Genome
from biosphere.groups import (
    GROUP_FORAGE_BASE, W4Config, build_adhesion_chain_index,
    group_forage_dividend, reproduce_groups,
)
from biosphere.metrics import metrics_json
from biosphere.resolve import Individual, Population, ResolveReport
from biosphere.simulator import BioMVPConfig, BiosphereSimulator
from biosphere.world_adapter import WorldView, _von_neumann_table


def _ctrl(**overrides) -> ControllerParams:
    base = dict(
        sense_food=1.0, sense_threat=1.0, sense_kin=1.0, sense_signal=1.0,
        move_bias=0.5, eat_greed=0.5, attack_drive=0.0, share_drive=0.5,
        signal_drive=0.0, repro_drive=0.5, metabolism=1.0, move_cost_mult=1.0,
        lifespan=1000.0, repro_thresh=40.0, habitat_aquatic=1.0,
        habitat_terrestrial=1.0, adhesion=0.9, defense=0.0, eat_amount=0.5,
        signal_payload=0.5, give_up=10.0, persistence=0.0,
    )
    base.update(overrides)
    return ControllerParams(**base)


def _world(rows=4, cols=6) -> WorldView:
    n = rows * cols
    return WorldView(
        rows=rows, cols=cols, n_cells=n,
        surface_type=np.zeros(n, dtype=np.uint8),
        elevation_m=np.full(n, 5.0), depth_m=np.zeros(n),
        patch_id=np.full(n, -1, dtype=np.int32),
        neighbors=_von_neumann_table(rows, cols),
        soil_water=np.full(n, 5.0), soil_water_min=2.0,
    )


def _pop(pairs, diff_states=None) -> Population:
    pop = Population()
    for uid, cell, energy in pairs:
        ind = Individual(uid=uid, genome=Genome(genes=np.zeros(64)),
                         cell=cell, energy=energy)
        if diff_states:
            ind.diff_state = diff_states.get(uid, 0)
        pop.individuals.append(ind)
        pop.next_id = max(pop.next_id, uid + 1)
    return pop


def _repro_events(pop, w, ctrls, thresh):
    events = []
    reproduce_groups(pop, build_adhesion_chain_index(pop, w, ctrls), w,
                     ctrls, np.random.default_rng(0), 0,
                     ResolveReport(tick=0), max_pop=512,
                     events_out=events, thresh_per_capita=thresh)
    return events


# ── G1 组繁殖门槛扫描 ────────────────────────────────────────────────────


class TestGroupReproThreshold:
    def test_lower_threshold_more_events(self):
        """同一局面：门槛 40 ⇒ 可繁殖；160 ⇒ 不可（假设方向）。"""
        w = _world()
        pop = _pop([(0, 0, 90.0), (1, 1, 90.0)], diff_states={0: 1, 1: 1})
        ctrls = {0: _ctrl(), 1: _ctrl()}
        # 总 180；人均门槛 40 ⇒ 180 ≥ 80 ✓；160 ⇒ 180 < 320 ✗
        assert len(_repro_events(pop, w, ctrls, 40.0)) == 2
        pop2 = _pop([(0, 0, 90.0), (1, 1, 90.0)], diff_states={0: 1, 1: 1})
        assert len(_repro_events(pop2, w, ctrls, 160.0)) == 0

    def test_config_knob_reaches_engine(self):
        """W4Config.group_repro_thresh_per_capita 真的进引擎。"""
        from biosphere.experiments import W4C_ARMS
        assert W4C_ARMS["thr40_full"]["w4"]["group_repro_thresh_per_capita"] \
            == 40.0
        assert W4C_ARMS["thr160_z2"]["w4"]["group_repro_thresh_per_capita"] \
            == 160.0
        assert W4C_ARMS["thr80_z2"]["w4"].get("group_reproduction") is False


# ── G2 亲缘结构 ──────────────────────────────────────────────────────────


class TestKinCohesion:
    def test_knob_runs_and_deterministic(self):
        def run():
            sim = BiosphereSimulator(
                BioMVPConfig(ticks=200, seed=42,
                             w4=W4Config(enabled=True, kin_cohesion=True)))
            sim.run()
            return ([(i.uid, i.cell, i.energy, i.diff_state)
                     for i in sim.population.individuals],
                    [(f.n_multi_chain, f.in_multi_chain)
                     for f in sim.log.chain_frames])
        assert run() == run()

    def test_sticky_prefers_denser_kin_cell(self):
        """计数加权：粘附个体应偏好链团邻居更多的候选格（对照二值版）。"""
        from biosphere.genome import PARAM_SLOTS
        slot = PARAM_SLOTS["adhesion"]

        def _sticky_genome() -> Genome:
            genes = np.zeros(64)
            genes[slot] = 5.0            # ⇒ 粘附参数 ≥ 0.5（必粘）
            genes[PARAM_SLOTS["attack_drive"]] = -5.0   # 隔离：不让 ATTACK 抢
            return Genome(genes=genes)

        # 团成员：uid1/2/3 在 0/1/2（一行三连，互相相邻）；被测 uid0 在 6
        # （0 的正下方）。uid0 的候选格：0（上方，与 1/2/3 都相邻 ⇒ 链团
        # 邻居最多）、5/7（各只碰一个）、12（不碰，被硬过滤）。
        pop = Population()
        for uid, cell in [(0, 6), (1, 0), (2, 1), (3, 2)]:
            ind = Individual(uid=uid, genome=_sticky_genome(), cell=cell,
                             energy=30.0)
            ind.diff_state = 1          # 体细胞态：全额表达
            pop.individuals.append(ind)
            pop.next_id = uid + 1
        sim = BiosphereSimulator(
            BioMVPConfig(ticks=1, seed=7, init_pop=0, rows=4, cols=6,
                         w4=W4Config(enabled=True, kin_cohesion=True)))
        sim.resource.amount[:] = 0.0            # 无食物 ⇒ EAT 不抢 argmax
        sim.population = pop
        sim._occupant = pop.occupant_map(sim.world.n_cells)
        sim._by_uid = {i.uid: i for i in pop.individuals}
        ctrls = {i.uid: sim.cache.get(i.genome) for i in pop.individuals}
        sim._group_index = __import__(
            "biosphere.groups", fromlist=["build_group_index"]
        ).build_group_index(pop, sim.world)
        sim._chain_index = build_adhesion_chain_index(pop, sim.world, ctrls)
        assert sim._chain_index.size(0) == 4      # 四个都粘 ⇒ 一团
        intent = sim._decide(pop.individuals[0])
        assert intent is not None and intent.kind.name == "MOVE"
        assert intent.target_cell == 0


# ── G3 粘附收益（群体觅食分红） ──────────────────────────────────────────


class TestAdhesionBenefit:
    def test_dividend_math_and_accounting(self):
        """分红 = strength × BASE × Σn（多体链团）；总额守恒可验。"""
        w = _world()
        pop = _pop([(0, 0, 10.0), (1, 1, 10.0), (2, 12, 10.0)],
                   diff_states={0: 1, 1: 1, 2: 1})
        ctrls = {i: _ctrl() for i in range(3)}
        index = build_adhesion_chain_index(pop, w, ctrls)
        assert index.size(0) == 2          # 0-1 成链；2 单体
        before = sum(i.energy for i in pop.individuals)
        gained = group_forage_dividend(pop, index, 1.0)
        after = sum(i.energy for i in pop.individuals)
        assert gained == pytest.approx(1.0 * GROUP_FORAGE_BASE * 2)
        assert after - before == pytest.approx(gained)
        # 只给多体链团成员：uid2 不涨
        assert pop.individuals[2].energy == pytest.approx(10.0)

    def test_zero_strength_no_dividend(self):
        w = _world()
        pop = _pop([(0, 0, 10.0), (1, 1, 10.0)], diff_states={0: 1, 1: 1})
        ctrls = {i: _ctrl() for i in range(2)}
        index = build_adhesion_chain_index(pop, w, ctrls)
        assert group_forage_dividend(pop, index, 0.0) == 0.0
        assert all(i.energy == 10.0 for i in pop.individuals)

    def test_dividend_hits_energy_in(self):
        """高粘附创始者成链 ⇒ 分红立即计入 energy_in（世界侧流入）。"""
        from biosphere.genome import PARAM_SLOTS
        slot = PARAM_SLOTS["adhesion"]

        def run(benefit: float) -> float:
            sim = BiosphereSimulator(
                BioMVPConfig(ticks=60, seed=42, init_pop=0,
                             w4=W4Config(enabled=True,
                                         adhesion_benefit=benefit)))
            # 6 个高粘附创始者放相邻格 ⇒ tick 0 即成链
            cells = [0, 1, 2, 6, 7, 8]
            for i, cell in enumerate(cells):
                sim.spawn_founder(cell, 60.0, {slot: 3.0})
            for ind in sim.population.individuals:
                ind.diff_state = 1              # 体细胞态：全额表达粘附
            sim.run()
            return sim.log.energy_in

        assert run(1.0) > run(0.0)


# ── 硬约束回归 ───────────────────────────────────────────────────────────


class TestW4cHardConstraints:
    def test_w4_off_bit_identical(self):
        """W4Config 新旋钮默认关 ⇒ 与 W0-B 逐位等价。"""
        a = BiosphereSimulator(BioMVPConfig(ticks=80, seed=42))
        a.run()
        b = BiosphereSimulator(BioMVPConfig(ticks=80, seed=42,
                                            w4=W4Config()))
        b.run()
        assert [(i.uid, i.cell, i.energy) for i in a.population.individuals] \
            == [(i.uid, i.cell, i.energy) for i in b.population.individuals]

    def test_baseline_knobs_equal_w4b_default(self):
        """W4c 默认旋钮值 = W4b baseline ⇒ thr80_full 应与 W4b full 等价。"""
        a = BiosphereSimulator(BioMVPConfig(ticks=100, seed=42,
                                            w4=W4Config(enabled=True)))
        a.run()
        b = BiosphereSimulator(
            BioMVPConfig(ticks=100, seed=42,
                         w4=W4Config(enabled=True,
                                     group_repro_thresh_per_capita=80.0,
                                     kin_cohesion=False,
                                     adhesion_benefit=0.0)))
        b.run()
        assert [(i.uid, i.cell, i.energy) for i in a.population.individuals] \
            == [(i.uid, i.cell, i.energy) for i in b.population.individuals]

    def test_checkpoint_resume_bit_identical_with_knobs(self, tmp_path):
        straight = BiosphereSimulator(
            BioMVPConfig(ticks=120, seed=43,
                         w4=W4Config(enabled=True, kin_cohesion=True,
                                     adhesion_benefit=0.5,
                                     group_repro_thresh_per_capita=40.0)))
        straight.run()
        part = BiosphereSimulator(
            BioMVPConfig(ticks=60, seed=43,
                         w4=W4Config(enabled=True, kin_cohesion=True,
                                     adhesion_benefit=0.5,
                                     group_repro_thresh_per_capita=40.0)))
        part.run(ckpt_dir=tmp_path, ckpt_every=60, tag="w4c")
        from biosphere.checkpoint import load_checkpoint
        resumed = load_checkpoint(next(tmp_path.glob("ckpt_w4c_t60.pkl")))
        resumed.run(ticks=120)
        key = lambda sim: [(i.uid, i.cell, i.energy, i.diff_state)
                           for i in sim.population.individuals]
        assert key(resumed) == key(straight)
        assert resumed.log.energy_in == pytest.approx(straight.log.energy_in)

    def test_metrics_shape_with_knobs(self):
        sim = BiosphereSimulator(
            BioMVPConfig(ticks=150, seed=42,
                         w4=W4Config(enabled=True, kin_cohesion=True,
                                     adhesion_benefit=0.5)))
        sim.run()
        m = metrics_json(sim.log, sim.world, seed=42, ticks=150)
        chain = m["groups"]["adhesion_chain"]
        assert chain["enabled"] is True
        assert "adhesion_trajectory" in chain
        assert "repro_composition" in chain
        assert m["energy"]["world_inflow"] > 0.0
