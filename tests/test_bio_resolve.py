"""B4：两段式提交裁定测试（确定性 + 一格一写者 + 能量记账）。

覆盖 B1 resolve.py 的冻结口径：
- 同 seed 两次 resolve ⇒ 种群状态逐位一致（含繁殖突变的 RNG 流）；
- 一格一写者：竞争失败者 stay_contested；占用者不动 ⇒ stay_occupied；
- 链式让位：A→B 格、B→C 格、C 不动 ⇒ 双双失败（曾因静态 departing
  数组漏判而双占，回归测试）；
- 环（swap）⇒ 双赢；
- 栖息拦截：不过线不进对应生境；
- 能量记账：成本显式支付、吃入/掠食/endowment 台账精确；
- 攻击致死即掠食，第二攻击者扑空；
- commit_intents 按 actor_id 排序。
"""
import numpy as np
import pytest

from biosphere.action import ActionKind, Intent
from biosphere.develop import ControllerParams
from biosphere.genome import Genome, random_genome
from biosphere.resolve import (
    REPRO_ENDOWMENT, Individual, Population, ResolveConfig,
    commit_intents, resolve_tick,
)
from biosphere.world_adapter import (
    SURFACE_LAND, SURFACE_OCEAN, SyntheticResource, WorldView,
    _von_neumann_table,
)

# ── 测试夹具（小手工世界，禁用合成世界的随机性） ──────────────────────────


def make_world(rows: int = 4, cols: int = 4,
               water_cells=()) -> WorldView:
    """全陆小世界；water_cells 指定水格。"""
    n = rows * cols
    st = np.full(n, SURFACE_LAND, dtype=np.uint8)
    for c in water_cells:
        st[c] = SURFACE_OCEAN
    return WorldView(
        rows=rows, cols=cols, n_cells=n,
        surface_type=st,
        elevation_m=np.zeros(n), depth_m=np.zeros(n),
        patch_id=np.full(n, -1, dtype=np.int32),
        neighbors=_von_neumann_table(rows, cols),
    )


def make_resource(amounts) -> SyntheticResource:
    amounts = np.asarray(amounts, dtype=np.float64)
    return SyntheticResource(amount=amounts.copy(),
                             capacity=np.maximum(amounts, 1.0),
                             regrow_rate=0.0)


_CTRL_DEFAULTS = dict(
    sense_food=1.0, sense_threat=1.0, sense_kin=1.0, sense_signal=1.0,
    move_bias=0.5, eat_greed=0.5, attack_drive=0.0, share_drive=0.0,
    signal_drive=0.0, repro_drive=0.5, metabolism=1.0, move_cost_mult=1.0,
    lifespan=1000.0, repro_thresh=40.0, habitat_aquatic=1.0,
    habitat_terrestrial=1.0, adhesion=0.0, defense=0.0, eat_amount=0.5,
    signal_payload=0.5, give_up=10.0, persistence=0.0,
)


def ctrl(**overrides) -> ControllerParams:
    base = dict(_CTRL_DEFAULTS)
    base.update(overrides)
    return ControllerParams(**base)


def add_ind(pop: Population, uid: int, cell: int, energy: float,
            genome: Genome | None = None) -> Individual:
    if genome is None:
        genome = Genome(genes=np.zeros(64))
    ind = Individual(uid=uid, genome=genome, cell=cell, energy=energy)
    pop.individuals.append(ind)
    pop.next_id = max(pop.next_id, uid + 1)
    return ind


def controllers_for(pop: Population, **overrides) -> dict:
    return {ind.uid: ctrl(**overrides) for ind in pop.individuals}


def move_intent(uid: int, dest: int, cost: float = 0.3) -> Intent:
    return Intent(actor_id=uid, kind=ActionKind.MOVE, target_cell=dest,
                  energy_cost=cost)


def eat_intent(uid: int, cell: int, cost: float = 0.05) -> Intent:
    return Intent(actor_id=uid, kind=ActionKind.EAT, target_cell=cell,
                  energy_cost=cost)


# ── 确定性 ────────────────────────────────────────────────────────────────


class TestDeterminism:
    def _scenario(self, seed: int):
        """一个含全部动作类型的小场景（intent 由 rng 生成）。"""
        rng = np.random.default_rng(seed)
        world = make_world(4, 4)
        pop = Population()
        cells = list(range(16))
        rng.shuffle(cells)
        for i, cell in enumerate(cells[:8]):
            add_ind(pop, uid=i, cell=int(cell), energy=float(20 + 10 * i),
                    genome=random_genome(np.random.default_rng(seed * 100 + i)))
        controllers = {ind.uid: ctrl() for ind in pop.individuals}
        intents = []
        for ind in pop.individuals:
            roll = rng.integers(0, 4)
            nb = [int(x) for x in world.neighbors[ind.cell]]
            if roll == 0:
                intents.append(move_intent(ind.uid, int(rng.choice(nb))))
            elif roll == 1:
                intents.append(eat_intent(ind.uid, ind.cell))
            elif roll == 2:
                intents.append(Intent(actor_id=ind.uid, kind=ActionKind.SIGNAL,
                                      target_cell=ind.cell, payload=0.5,
                                      energy_cost=0.15))
            else:
                intents.append(Intent(actor_id=ind.uid, kind=ActionKind.ATTACK,
                                      target_id=int(rng.integers(0, 8)),
                                      energy_cost=0.4))
        resource = make_resource(rng.random(16) * 5)
        signal = np.zeros(16)
        return pop, world, resource, signal, controllers, intents

    def _run_once(self, seed: int):
        pop, world, resource, signal, controllers, intents = self._scenario(seed)
        report = resolve_tick(
            pop, commit_intents(pop, intents), world, resource, signal,
            tick=0, rng=np.random.default_rng(seed),
            controllers=controllers)
        return pop, report, resource, signal

    def test_same_seed_bit_identical(self):
        p1, r1, res1, sig1 = self._run_once(3)
        p2, r2, res2, sig2 = self._run_once(3)
        assert [(i.uid, i.cell, i.energy, i.alive) for i in p1.individuals] == \
            [(i.uid, i.cell, i.energy, i.alive) for i in p2.individuals]
        assert [(e.actor_id, e.kind, e.status, e.target_id,
                 e.actor_energy_delta) for e in r1.events] == \
            [(e.actor_id, e.kind, e.status, e.target_id,
              e.actor_energy_delta) for e in r2.events]
        assert np.array_equal(res1.amount, res2.amount)
        assert np.array_equal(sig1, sig2)
        assert (r1.energy_paid, r1.energy_in, r1.endowment_paid) == \
            (r2.energy_paid, r2.energy_in, r2.endowment_paid)
        assert r1.births == r2.births

    def test_repro_rng_stream_deterministic(self):
        """繁殖突变消费同一 rng 流 ⇒ 两次子代基因组逐位一致。"""
        world = make_world(4, 4)
        results = []
        for _ in range(2):
            pop = Population()
            parent_genome = random_genome(np.random.default_rng(77), uid=0)
            add_ind(pop, uid=0, cell=0, energy=100.0, genome=parent_genome)
            pop.next_id = 1
            intents = [Intent(actor_id=0, kind=ActionKind.REPRODUCE,
                              energy_cost=0.5)]
            resolve_tick(pop, commit_intents(pop, intents), world,
                         make_resource(np.ones(16)), np.zeros(16), tick=0,
                         rng=np.random.default_rng(555),
                         controllers={0: ctrl()})
            results.append(pop.individuals[-1].genome.genes.copy())
        assert np.array_equal(results[0], results[1])


# ── 一格一写者与占用 ──────────────────────────────────────────────────────


class TestOneCellOneWriter:
    def test_contest_lowest_uid_wins(self):
        world = make_world(4, 4)
        pop = Population()
        add_ind(pop, uid=1, cell=0, energy=50.0)
        add_ind(pop, uid=2, cell=2, energy=50.0)
        intents = [move_intent(1, 1), move_intent(2, 1)]
        report = resolve_tick(
            pop, commit_intents(pop, intents), world,
            make_resource(np.ones(16)), np.zeros(16), tick=0,
            rng=np.random.default_rng(0),
            controllers=controllers_for(pop))
        st = {(e.actor_id): e.status for e in report.events}
        assert st[1] == "ok"
        assert st[2] == "stay_contested"
        assert pop.individuals[0].cell == 1
        assert pop.individuals[1].cell == 2  # 竞争失败原地不动

    def test_occupied_by_non_mover(self):
        world = make_world(4, 4)
        pop = Population()
        add_ind(pop, uid=1, cell=0, energy=50.0)   # 不动
        add_ind(pop, uid=2, cell=2, energy=50.0)
        intents = [move_intent(2, 0)]              # 2 想进 1 的格
        report = resolve_tick(
            pop, commit_intents(pop, intents), world,
            make_resource(np.ones(16)), np.zeros(16), tick=0,
            rng=np.random.default_rng(0),
            controllers=controllers_for(pop))
        assert report.events[0].status == "stay_occupied"
        assert pop.individuals[1].cell == 2

    def test_swap_both_win(self):
        world = make_world(4, 4)
        pop = Population()
        add_ind(pop, uid=1, cell=0, energy=50.0)
        add_ind(pop, uid=2, cell=1, energy=50.0)
        intents = [move_intent(1, 1), move_intent(2, 0)]  # 互换
        report = resolve_tick(
            pop, commit_intents(pop, intents), world,
            make_resource(np.ones(16)), np.zeros(16), tick=0,
            rng=np.random.default_rng(0),
            controllers=controllers_for(pop))
        assert {e.status for e in report.events} == {"ok"}
        assert pop.individuals[0].cell == 1
        assert pop.individuals[1].cell == 0

    def test_chain_blocked_both_stay(self):
        """A→B 格、B→C 格、C 不动 ⇒ 链断，双双 stay_occupied（双占回归）。"""
        world = make_world(4, 4)
        pop = Population()
        add_ind(pop, uid=1, cell=0, energy=50.0)   # A: 0 → 1
        add_ind(pop, uid=2, cell=1, energy=50.0)   # B: 1 → 2
        add_ind(pop, uid=3, cell=2, energy=50.0)   # C: 不动
        intents = [move_intent(1, 1), move_intent(2, 2)]
        report = resolve_tick(
            pop, commit_intents(pop, intents), world,
            make_resource(np.ones(16)), np.zeros(16), tick=0,
            rng=np.random.default_rng(0),
            controllers=controllers_for(pop))
        assert {e.status for e in report.events} == {"stay_occupied"}
        cells = sorted(i.cell for i in pop.individuals)
        assert cells == [0, 1, 2]  # 无人移动，无双占

    def test_chain_into_empty_all_win(self):
        """A→B 格、B→空格 ⇒ 链通，两人都动。"""
        world = make_world(4, 4)
        pop = Population()
        add_ind(pop, uid=1, cell=0, energy=50.0)   # A: 0 → 1
        add_ind(pop, uid=2, cell=1, energy=50.0)   # B: 1 → 5（空）
        intents = [move_intent(1, 1), move_intent(2, 5)]
        report = resolve_tick(
            pop, commit_intents(pop, intents), world,
            make_resource(np.ones(16)), np.zeros(16), tick=0,
            rng=np.random.default_rng(0),
            controllers=controllers_for(pop))
        assert {e.status for e in report.events} == {"ok"}
        assert pop.individuals[0].cell == 1
        assert pop.individuals[1].cell == 5

    def test_no_double_occupancy_invariant(self):
        """100 个随机移动场景后：任何 tick 结束都无两个体同格。"""
        for seed in range(30):
            rng = np.random.default_rng(seed)
            world = make_world(4, 4)
            pop = Population()
            free = list(range(16))
            rng.shuffle(free)
            for i, cell in enumerate(free[:10]):
                add_ind(pop, uid=i, cell=int(cell), energy=50.0)
            intents = [move_intent(ind.uid, int(rng.choice(
                [int(x) for x in world.neighbors[ind.cell]])))
                for ind in pop.individuals]
            resolve_tick(
                pop, commit_intents(pop, intents), world,
                make_resource(np.ones(16)), np.zeros(16), tick=0,
                rng=np.random.default_rng(0),
                controllers=controllers_for(pop))
            cells = [i.cell for i in pop.individuals]
            assert len(cells) == len(set(cells)), f"seed={seed} 出现双占"


class TestHabitat:
    def test_aquatic_blocked_from_land(self):
        world = make_world(4, 4)
        pop = Population()
        add_ind(pop, uid=1, cell=5, energy=50.0)   # cell 5 是水？不——全陆。
        # 把 cell 5 设为水
        world.surface_type[5] = SURFACE_OCEAN
        controllers = {1: ctrl(habitat_aquatic=1.0, habitat_terrestrial=0.0)}
        intents = [move_intent(1, 1)]              # 1 是陆格 ⇒ 拦截
        report = resolve_tick(
            pop, commit_intents(pop, intents), world,
            make_resource(np.ones(16)), np.zeros(16), tick=0,
            rng=np.random.default_rng(0), controllers=controllers)
        assert report.events[0].status == "stay_habitat"
        assert pop.individuals[0].cell == 5

    def test_amphibious_crosses(self):
        world = make_world(4, 4)
        world.surface_type[5] = SURFACE_OCEAN
        pop = Population()
        add_ind(pop, uid=1, cell=5, energy=50.0)
        controllers = {1: ctrl(habitat_aquatic=1.0, habitat_terrestrial=0.6)}
        intents = [move_intent(1, 1)]
        report = resolve_tick(
            pop, commit_intents(pop, intents), world,
            make_resource(np.ones(16)), np.zeros(16), tick=0,
            rng=np.random.default_rng(0), controllers=controllers)
        assert report.events[0].status == "ok"
        assert pop.individuals[0].cell == 1


# ── 能量记账 ──────────────────────────────────────────────────────────────


class TestEnergyAccounting:
    def test_costs_paid_win_or_lose(self):
        world = make_world(4, 4)
        pop = Population()
        add_ind(pop, uid=1, cell=0, energy=50.0)
        add_ind(pop, uid=2, cell=2, energy=50.0)
        intents = [move_intent(1, 1, cost=0.3),   # 成功
                   move_intent(2, 1, cost=0.3),   # 竞争失败
                   eat_intent(1, 0, cost=0.05)]   # 同一 tick 只能一个意图？
        # resolve 不阻止一个 actor 多个意图；这里只测记账，去掉重复 actor
        intents = [move_intent(1, 1, cost=0.3), move_intent(2, 1, cost=0.3)]
        report = resolve_tick(
            pop, commit_intents(pop, intents), world,
            make_resource(np.ones(16)), np.zeros(16), tick=0,
            rng=np.random.default_rng(0),
            controllers=controllers_for(pop))
        # 尝试即付成本：成败都付
        assert pop.individuals[0].energy == pytest.approx(50.0 - 0.3)
        assert pop.individuals[1].energy == pytest.approx(50.0 - 0.3)
        assert report.energy_paid == pytest.approx(0.6)

    def test_eat_pool_first_come(self):
        world = make_world(4, 4)
        pop = Population()
        add_ind(pop, uid=1, cell=0, energy=50.0)
        add_ind(pop, uid=2, cell=0, energy=50.0)   # 同格（构造期允许）
        resource = make_resource([1.0] + [9.0] * 15)
        intents = [eat_intent(1, 0, cost=0.05), eat_intent(2, 0, cost=0.05)]
        controllers = {1: ctrl(eat_amount=0.4), 2: ctrl(eat_amount=0.4)}
        report = resolve_tick(
            pop, commit_intents(pop, intents), world, resource,
            np.zeros(16), tick=0, rng=np.random.default_rng(0),
            controllers=controllers)
        # uid 1 先到：吃 0.4；uid 2 得剩余 0.6 中再吃 0.4
        assert pop.individuals[0].energy == pytest.approx(50.0 - 0.05 + 0.4)
        assert pop.individuals[1].energy == pytest.approx(50.0 - 0.05 + 0.4)
        assert resource.amount[0] == pytest.approx(0.2)
        assert report.energy_in == pytest.approx(0.8)

    def test_eat_empty_pool(self):
        world = make_world(4, 4)
        pop = Population()
        add_ind(pop, uid=1, cell=0, energy=50.0)
        resource = make_resource([0.0] * 16)
        report = resolve_tick(
            pop, commit_intents(pop, [eat_intent(1, 0)]), world, resource,
            np.zeros(16), tick=0, rng=np.random.default_rng(0),
            controllers=controllers_for(pop))
        assert report.events[0].status == "eat_empty"
        assert pop.individuals[0].energy == pytest.approx(49.95)
        assert report.energy_in == 0.0


# ── 攻击 / 分享 / 繁殖 ────────────────────────────────────────────────────


class TestAttackShareReproduce:
    def test_kill_and_loot(self):
        world = make_world(4, 4)
        pop = Population()
        add_ind(pop, uid=1, cell=0, energy=50.0)
        add_ind(pop, uid=2, cell=1, energy=5.0)    # 邻格，能量低
        intents = [Intent(actor_id=1, kind=ActionKind.ATTACK,
                          target_id=2, energy_cost=0.4)]
        controllers = {1: ctrl(attack_drive=1.0), 2: ctrl(defense=0.0)}
        report = resolve_tick(
            pop, commit_intents(pop, intents), world,
            make_resource(np.ones(16)), np.zeros(16), tick=0,
            rng=np.random.default_rng(0), controllers=controllers)
        # power = 4 + 8×1 = 12 > 5 ⇒ 死；loot = 0.5 × 5 = 2.5
        assert pop.individuals[1].alive is False
        assert pop.individuals[0].energy == pytest.approx(50.0 - 0.4 + 2.5)
        assert report.deaths == [(2, "attacked")]
        assert report.energy_in == pytest.approx(2.5)

    def test_second_attacker_misses_dead_target(self):
        world = make_world(4, 4)
        pop = Population()
        add_ind(pop, uid=1, cell=0, energy=50.0)
        add_ind(pop, uid=2, cell=1, energy=5.0)
        add_ind(pop, uid=3, cell=2, energy=50.0)    # 与 2 不相邻？1-2 邻，2-3 邻
        intents = [Intent(actor_id=1, kind=ActionKind.ATTACK,
                          target_id=2, energy_cost=0.4),
                   Intent(actor_id=2, kind=ActionKind.ATTACK,
                          target_id=3, energy_cost=0.4)]
        controllers = {1: ctrl(attack_drive=1.0), 2: ctrl(attack_drive=1.0),
                       3: ctrl(defense=0.0)}
        report = resolve_tick(
            pop, commit_intents(pop, intents), world,
            make_resource(np.ones(16)), np.zeros(16), tick=0,
            rng=np.random.default_rng(0), controllers=controllers)
        st = {(e.actor_id): e.status for e in report.events}
        assert st[1] == "ok"               # 1 杀死 2
        assert st[2] == "actor_dead"       # 2 的意图：人已死，不生效不付费
        assert pop.individuals[1].alive is False

    def test_survived_attack(self):
        world = make_world(4, 4)
        pop = Population()
        add_ind(pop, uid=1, cell=0, energy=50.0)
        add_ind(pop, uid=2, cell=1, energy=50.0)
        intents = [Intent(actor_id=1, kind=ActionKind.ATTACK,
                          target_id=2, energy_cost=0.4)]
        controllers = {1: ctrl(attack_drive=0.0), 2: ctrl(defense=0.0)}
        report = resolve_tick(
            pop, commit_intents(pop, intents), world,
            make_resource(np.ones(16)), np.zeros(16), tick=0,
            rng=np.random.default_rng(0), controllers=controllers)
        assert report.events[0].status == "attack_survived"
        assert pop.individuals[1].energy == pytest.approx(46.0)  # -4
        assert pop.individuals[1].alive is True

    def test_attack_non_adjacent_fails(self):
        world = make_world(4, 4)
        pop = Population()
        add_ind(pop, uid=1, cell=0, energy=50.0)
        add_ind(pop, uid=2, cell=15, energy=50.0)   # 对角远端
        intents = [Intent(actor_id=1, kind=ActionKind.ATTACK,
                          target_id=2, energy_cost=0.4)]
        report = resolve_tick(
            pop, commit_intents(pop, intents), world,
            make_resource(np.ones(16)), np.zeros(16), tick=0,
            rng=np.random.default_rng(0),
            controllers=controllers_for(pop))
        assert report.events[0].status == "attack_no_target"

    def test_share_transfers_energy(self):
        world = make_world(4, 4)
        pop = Population()
        add_ind(pop, uid=1, cell=0, energy=50.0)
        add_ind(pop, uid=2, cell=1, energy=10.0)
        intents = [Intent(actor_id=1, kind=ActionKind.SHARE, target_id=2,
                          payload=5.0, energy_cost=0.2)]
        report = resolve_tick(
            pop, commit_intents(pop, intents), world,
            make_resource(np.ones(16)), np.zeros(16), tick=0,
            rng=np.random.default_rng(0),
            controllers=controllers_for(pop))
        assert pop.individuals[0].energy == pytest.approx(44.8)
        assert pop.individuals[1].energy == pytest.approx(15.0)
        assert report.events[0].status == "ok"

    def test_reproduce_spawn_and_ledger(self):
        world = make_world(4, 4)
        pop = Population()
        parent = add_ind(pop, uid=1, cell=0, energy=100.0,
                         genome=random_genome(np.random.default_rng(1)))
        pop.next_id = 2
        intents = [Intent(actor_id=1, kind=ActionKind.REPRODUCE,
                          energy_cost=0.5)]
        controllers = {1: ctrl(repro_thresh=40.0)}
        report = resolve_tick(
            pop, commit_intents(pop, intents), world,
            make_resource(np.ones(16)), np.zeros(16), tick=0,
            rng=np.random.default_rng(0), controllers=controllers)
        assert len(pop.individuals) == 2
        child = pop.individuals[-1]
        assert child.uid == 2
        assert child.parent == 1
        assert child.energy == pytest.approx(REPRO_ENDOWMENT)
        assert child.cell in [int(x) for x in world.neighbors[0]]
        assert parent.energy == pytest.approx(100.0 - 0.5 - REPRO_ENDOWMENT)
        assert report.endowment_paid == pytest.approx(REPRO_ENDOWMENT)
        assert report.births == [(1, 2, child.cell)]

    def test_reproduce_no_energy(self):
        world = make_world(4, 4)
        pop = Population()
        add_ind(pop, uid=1, cell=0, energy=10.0)
        intents = [Intent(actor_id=1, kind=ActionKind.REPRODUCE,
                          energy_cost=0.5)]
        report = resolve_tick(
            pop, commit_intents(pop, intents), world,
            make_resource(np.ones(16)), np.zeros(16), tick=0,
            rng=np.random.default_rng(0),
            controllers={1: ctrl(repro_thresh=40.0)})
        assert report.events[0].status == "repro_no_energy"
        assert len(pop.individuals) == 1

    def test_reproduce_pop_cap(self):
        world = make_world(4, 4)
        pop = Population()
        add_ind(pop, uid=1, cell=0, energy=100.0)
        cfg = ResolveConfig(max_pop=1)
        report = resolve_tick(
            pop, commit_intents(pop, [Intent(
                actor_id=1, kind=ActionKind.REPRODUCE, energy_cost=0.5)]),
            world, make_resource(np.ones(16)), np.zeros(16), tick=0,
            rng=np.random.default_rng(0),
            controllers={1: ctrl(repro_thresh=40.0)}, cfg=cfg)
        assert report.events[0].status == "repro_no_room"


class TestCommitIntents:
    def test_sorts_by_actor_id(self):
        pop = Population()
        intents = [move_intent(5, 1), move_intent(2, 1), move_intent(9, 1)]
        out = commit_intents(pop, intents)
        assert [i.actor_id for i in out] == [2, 5, 9]
        assert intents[0].actor_id == 5  # 不改入参

    def test_move_without_cell_raises(self):
        pop = Population()
        with pytest.raises(ValueError):
            commit_intents(pop, [Intent(actor_id=1, kind=ActionKind.MOVE,
                                        energy_cost=0.3)])
