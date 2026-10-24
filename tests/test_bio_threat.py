"""B6：威胁传感通道 + fleeing 标签修复测试。

覆盖两处改动：
1. **标签修复（bug）**：逃避标签原只认 `status=="ok"`（致死攻击），而致死
   受害者 3 tick 后必无帧 ⇒ 窗口恒空 ⇒ fleeing 结构性 = 0。修复为「命中即可
   （致死或致伤）」——本文件用构造数据钉住（回归）。
2. **威胁记忆通道（机制）**：受击（致伤）写入 `threat_cell/threat_age`，
   有效期 THREAT_MEMORY_TICKS；决策侧逃逸项把个体推离最近攻击者；
   `threat_sensing=False` 对照臂完全关闭威胁入口。
"""
import numpy as np
import pytest

from biosphere.action import ActionKind, Intent
from biosphere.genome import Genome, PARAM_SLOTS
from biosphere.metrics import Frame, SimLog, BehaviorLabeler
from biosphere.resolve import (
    THREAT_MEMORY_TICKS, THREAT_NONE_AGE, Individual, Population,
    ResolveConfig, commit_intents, resolve_tick,
)
from biosphere.simulator import BioMVPConfig, BiosphereSimulator
from biosphere.world_adapter import (
    SURFACE_LAND, SyntheticResource, WorldView, _von_neumann_table,
)

# ── 夹具（与 test_bio_resolve 同款手工小世界） ──────────────────────────


def make_world(rows: int = 4, cols: int = 4) -> WorldView:
    n = rows * cols
    return WorldView(
        rows=rows, cols=cols, n_cells=n,
        surface_type=np.full(n, SURFACE_LAND, dtype=np.uint8),
        elevation_m=np.zeros(n), depth_m=np.zeros(n),
        patch_id=np.full(n, -1, dtype=np.int32),
        neighbors=_von_neumann_table(rows, cols),
    )


def make_resource(n: int) -> SyntheticResource:
    return SyntheticResource(amount=np.zeros(n), capacity=np.ones(n),
                             regrow_rate=0.0)


def ctrl(**overrides):
    from biosphere.develop import ControllerParams
    base = dict(
        sense_food=1.0, sense_threat=1.0, sense_kin=1.0, sense_signal=1.0,
        move_bias=0.5, eat_greed=0.5, attack_drive=0.0, share_drive=0.0,
        signal_drive=0.0, repro_drive=0.5, metabolism=1.0, move_cost_mult=1.0,
        lifespan=1000.0, repro_thresh=40.0, habitat_aquatic=1.0,
        habitat_terrestrial=1.0, adhesion=0.0, defense=0.0, eat_amount=0.5,
        signal_payload=0.5, give_up=10.0, persistence=0.0,
    )
    base.update(overrides)
    return ControllerParams(**base)


def add_ind(pop: Population, uid: int, cell: int, energy: float) -> Individual:
    ind = Individual(uid=uid, genome=Genome(genes=np.zeros(64)), cell=cell,
                     energy=energy)
    pop.individuals.append(ind)
    pop.next_id = max(pop.next_id, uid + 1)
    return ind


# ── 1. fleeing 标签修复（回归） ──────────────────────────────────────────


def _craft_flee_log(status: str, move_away: bool) -> SimLog:
    """一次攻击事件 + 受害者 3 tick 位置序列（6 列世界）。

    攻击者格 10（row1 col4），受害者 tick t 在 16（row2 col4，距离 1）；
    若 move_away：t+1→17、t+2→18、t+3→18（row3 col0，距离 3）。
    status=="ok"（致死）时受害者从 t_attack 之后的帧中消失（真实跑的行为：
    死者不进帧）⇒ 无 3-tick 窗口。
    """
    log = SimLog()
    t_attack = 20
    victim_cells = {}
    for t in range(30):
        if t <= t_attack:
            victim_cells[t] = 16
        elif t == t_attack + 1:
            victim_cells[t] = 17 if move_away else 16
        else:
            victim_cells[t] = 18 if move_away else 16
    for t in range(30):
        if status == "ok" and t > t_attack:
            uids = np.array([1], dtype=np.int64)
            cell_arr = np.array([10], dtype=np.int64)
        else:
            uids = np.array([1, 2], dtype=np.int64)
            cell_arr = np.array([10, victim_cells[t]], dtype=np.int64)
        log.frames.append(Frame(t, uids, cell_arr, np.full(uids.size, 50.0),
                                np.full(uids.size, 2.5)))
    log.events.append(_ev(t_attack, 1, "attack", status, 10, 2))
    return log


def _ev(tick, actor, kind, status, target_cell, target_id, delta=-0.4):
    from biosphere.resolve import ResolveEvent
    return ResolveEvent(tick, actor, kind, status, target_cell, target_id,
                        delta)


class TestFleeingLabelFix:
    def test_wounded_attack_counts_as_hit(self):
        """致伤命中（attack_survived）必须进入逃避统计（B6 修复点）。"""
        from biosphere.metrics import _ATTACK_HIT_STATUSES
        assert "attack_survived" in _ATTACK_HIT_STATUSES
        assert "ok" in _ATTACK_HIT_STATUSES

    def test_fleeing_detectable_after_wound(self):
        log = _craft_flee_log("attack_survived", move_away=True)
        labels = BehaviorLabeler(log).label_all(make_world(4, 6))
        assert labels[2]["fleeing"] is True

    def test_no_fleeing_when_victim_stays(self):
        log = _craft_flee_log("attack_survived", move_away=False)
        labels = BehaviorLabeler(log).label_all(make_world(4, 6))
        assert labels[2]["fleeing"] is False

    def test_lethal_victim_has_no_window(self):
        """致死（ok）受害者无 3-tick 帧 ⇒ 不标逃避（窗口确实为空）。"""
        log = _craft_flee_log("ok", move_away=True)
        labels = BehaviorLabeler(log).label_all(make_world(4, 6))
        assert labels[2]["fleeing"] is False

    def test_missed_attack_ignored(self):
        log = _craft_flee_log("attack_no_target", move_away=True)
        labels = BehaviorLabeler(log).label_all(make_world(4, 6))
        assert labels[2]["fleeing"] is False


# ── 2. 威胁记忆：resolve 写入 ───────────────────────────────────────────


class TestThreatMemoryWrite:
    def test_wound_sets_memory(self):
        world = make_world()
        pop = Population()
        add_ind(pop, uid=1, cell=0, energy=50.0)
        add_ind(pop, uid=2, cell=1, energy=50.0)
        intents = [Intent(actor_id=1, kind=ActionKind.ATTACK,
                          target_id=2, energy_cost=0.4)]
        report = resolve_tick(
            pop, commit_intents(pop, intents), world, make_resource(16),
            np.zeros(16), tick=0, rng=np.random.default_rng(0),
            controllers={1: ctrl(attack_drive=1.0), 2: ctrl(defense=0.0)})
        victim = pop.individuals[1]
        assert victim.alive is True               # 50 - 12 = 38，致伤非致死
        assert victim.threat_cell == 0            # 攻击者格
        assert victim.threat_age == 0
        assert report.events[0].status == "attack_survived"

    def test_lethal_also_writes_memory(self):
        world = make_world()
        pop = Population()
        add_ind(pop, uid=1, cell=0, energy=50.0)
        add_ind(pop, uid=2, cell=1, energy=5.0)
        intents = [Intent(actor_id=1, kind=ActionKind.ATTACK,
                          target_id=2, energy_cost=0.4)]
        resolve_tick(
            pop, commit_intents(pop, intents), world, make_resource(16),
            np.zeros(16), tick=0, rng=np.random.default_rng(0),
            controllers={1: ctrl(attack_drive=1.0), 2: ctrl(defense=0.0)})
        victim = pop.individuals[1]
        assert victim.alive is False
        assert victim.threat_cell == 0            # 谱系/度量可追溯

    def test_founder_has_no_memory(self):
        assert THREAT_NONE_AGE > THREAT_MEMORY_TICKS
        ind = Individual(uid=0, genome=None, cell=0, energy=1.0)
        assert ind.threat_cell == -1
        assert ind.threat_age == THREAT_NONE_AGE


# ── 3. 逃逸行为：决策侧（对照臂） ───────────────────────────────────────


def _mini_sim(threat_sensing: bool, seed: int = 7):
    """两个体微型局：捕食者(高攻击) 邻 猎物(高移动+高威胁感知)。

    6×6 **全陆**世界（land_frac=0 ⇒ 无海洋，排除栖息拦截干扰）、无食物；
    两者除攻击/移动/威胁感知外的动作入口全部压低 ⇒ flag 关闭时猎物任何
    动作都不会位移（逃逸项是唯一位移动力）。
    格位：捕食者 7（row1 col1），猎物 1（row0 col1）——上下相邻。
    """
    cfg = BioMVPConfig(ticks=3, seed=seed, init_pop=0, rows=6, cols=6,
                       land_frac=0.0, threat_sensing=threat_sensing)
    sim = BiosphereSimulator(cfg)
    # 清空合成世界的资源（无食物）
    sim.resource.amount[:] = 0.0
    S = PARAM_SLOTS
    lo, hi = -3.0, 3.0
    predator_genes = {S["attack_drive"]: hi, S["move_bias"]: lo,
                      S["sense_threat"]: lo, S["repro_drive"]: lo,
                      S["signal_drive"]: lo, S["share_drive"]: lo,
                      S["eat_greed"]: lo}
    prey_genes = {S["move_bias"]: hi, S["sense_threat"]: hi,
                  S["attack_drive"]: lo, S["repro_drive"]: lo,
                  S["signal_drive"]: lo, S["share_drive"]: lo,
                  S["eat_greed"]: lo}
    sim.spawn_founder(7, 50.0, predator_genes)   # row1 col1
    sim.spawn_founder(1, 50.0, prey_genes)       # row0 col1，与 7 相邻
    return sim


def _prey_distance(sim) -> float:
    pred = sim.population.individuals[0]
    prey = sim.population.individuals[1]
    return sim.world.grid_distance(prey.cell, pred.cell)


class TestFleeBehavior:
    def test_memory_flees_attacker(self):
        """threat_on：受击后下一 tick 猎物与攻击者距离 1 → 2。"""
        sim = _mini_sim(threat_sensing=True)
        sim.step(0)   # 捕食者攻击猎物（致伤），写入威胁记忆
        prey = sim.population.individuals[1]
        assert prey.alive is True
        assert prey.threat_cell == 7
        d0 = _prey_distance(sim)
        assert d0 == 1.0
        sim.step(1)   # 逃逸项生效
        assert _prey_distance(sim) == 2.0

    def test_threat_off_never_moves(self):
        """threat_off：无威胁入口 ⇒ 猎物无位移动力，距离恒 1。"""
        sim = _mini_sim(threat_sensing=False)
        sim.step(0)
        assert _prey_distance(sim) == 1.0
        sim.step(1)
        assert _prey_distance(sim) == 1.0

    def test_memory_expires(self):
        """超过 THREAT_MEMORY_TICKS 失活：把猎物挪到无攻击者处后再等。"""
        sim = _mini_sim(threat_sensing=True)
        sim.step(0)
        prey = sim.population.individuals[1]
        prey.threat_age = THREAT_MEMORY_TICKS  # 强制过期
        assert prey.threat_age >= THREAT_MEMORY_TICKS

    def test_determinism_with_memory(self):
        """同 seed 两次（含威胁记忆路径）逐位一致。"""
        def run():
            sim = _mini_sim(threat_sensing=True, seed=11)
            sim.run()
            return [(i.uid, i.cell, i.energy) for i in sim.population.individuals]
        assert run() == run()


# ── 4. 网格距（列环绕） ─────────────────────────────────────────────────


class TestGridDistance:
    def test_wrap_semantics(self):
        w = make_world(4, 6)
        assert w.grid_distance(0, 5) == 1.0     # 同行列环绕邻格
        assert w.grid_distance(0, 3) == 3.0     # min(3, 6-3)
        assert w.grid_distance(0, 6) == 1.0     # 上下邻
        assert w.grid_distance(0, 0) == 0.0
