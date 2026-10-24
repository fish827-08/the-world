"""biosphere.resolve：两段式提交 + 种群容器（B 线 MVP，W0-B）。

通俗理解
--------
所有个体同一 tick 内"同时"行动，但计算机只能一个个算。两段式提交就是：

- **第一段（并行写意图）**：每个个体只往自己的缓冲区写一个 :class:`Intent`
  （想干什么、带多少成本），**不碰任何共享状态**——意图之间无先后。
- **第二段（按 id 固定顺序裁定）**：从最小 uid 到大 uid 逐个结算，
  共享冲突按 id 序裁决——**一格一写者**（MOVE 的目标格先声明先得），
  资源池先到先得（按 id 序），攻击/分享只认裁定瞬间仍存活的目标。

⇒ 同 seed 跑两次，结果**逐位一致**（B4 有测试钉住）：全程无并发、无
dict/set 迭代序依赖、无并行归约；唯一随机源是调用方传入的
``numpy.random.Generator``，且消费顺序写死（决策按 id 序、繁殖突变按 id 序）。

能量记账口径（与规格页 §四 一致）：
- 每个意图**裁定时支付显式成本**（成败都付——尝试即耗能）；
- EAT/ATTACK(致死掠食)/SHARE(收方) 是能量流入；
- REPRODUCE 额外支付子代 endowment（父出子进，系统内转移）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from biosphere.action import ActionKind, ACTION_NAMES, ResolveStatus, Intent
from biosphere.c3 import visible_badge
from biosphere.cell import TYPE_GATE, HabitatConfig, can_occupy, cell_type
from biosphere.genome import Genome, mutate
from biosphere.world_adapter import WorldView, SyntheticResource

# ── MVP 裁定常量（冻结；调参须改规格页并重跑预注册） ──────────────────────

#: 栖息适配下限（控制器 habitat_* < 此值 ⇒ 不许进对应生境）
HABITAT_MIN: float = 0.30
#: 攻击基础威力与攻击倾向加成
ATTACK_POWER_BASE: float = 4.0
ATTACK_POWER_DRIVE: float = 8.0
#: 防御减伤系数（damage × (1 - DEFENSE_MIT × defense)）
DEFENSE_MIT: float = 0.60
#: 致死掠食分成（掠食者取死前能量 × 此比例）
ATTACK_LOOT_FRAC: float = 0.50
# ── B14.2 反击风险（H5：打强者不再免费） ──
#: 防御折算出"反击威力"的系数（defense∈[0,1] ⇒ 反击威力 ∈ [0,8]）
COUNTER_DEFENSE_SCALE: float = 8.0
#: 反击门槛：目标反击威力 − 攻击威力 超过此值才开始算反击伤害
COUNTER_THRESHOLD: float = 2.0
#: 反击斜率：每单位超额的反击伤害（攻击者付、防御方得——守恒）
COUNTER_SLOPE: float = 0.5
#: 失败门槛：超过此值 ⇒ 攻击直接失败（目标无伤、无掠食）
COUNTER_FAIL_THRESHOLD: float = 6.0
#: 分享转移后发送方保留底线
SHARE_RESERVE: float = 1.0
#: 分享 payload 缩放（payload = SHARE_SCALE × share_drive，决策侧算）
SHARE_SCALE: float = 10.0
#: 信号层写入缩放（signal_layer += SIGNAL_SCALE × signal_payload）
SIGNAL_SCALE: float = 5.0
#: 繁殖子代 endowment（父支付、子获得）
REPRO_ENDOWMENT: float = 20.0
#: 种群上限（达到后繁殖失败）
MAX_POP: int = 512
#: 威胁记忆有效期（tick；B6：受击后这么久内"最近攻击者方位"有效）
THREAT_MEMORY_TICKS: int = 12
#: 威胁记忆初值的"无记忆"哨兵（age 大于有效期即失活）
THREAT_NONE_AGE: int = 10 ** 9


@dataclass
class ResolveConfig:
    """裁定参数（默认 = MVP 冻结值；注入用，便于测试小档）。"""

    # M7：栖息类型门限（默认 = 细胞尺度 TYPE_GATE；世界族可调）
    habitat_min: float = TYPE_GATE
    # M7：代谢罚值（两栖/未定准/错配）
    penalty_mismatch: float = 1.0
    penalty_amphibious: float = 0.5
    penalty_unadapted: float = 1.0
    attack_power_base: float = ATTACK_POWER_BASE
    attack_power_drive: float = ATTACK_POWER_DRIVE
    defense_mit: float = DEFENSE_MIT
    attack_loot_frac: float = ATTACK_LOOT_FRAC
    # B17.2：命中即转移（default-off；关时与旧行为逐位一致）
    contact_loot: bool = False
    # B14.2 反击风险（默认档：打同强度 excess=−4 ⇒ 无反击，旧口径兼容）
    counter_defense_scale: float = COUNTER_DEFENSE_SCALE
    counter_threshold: float = COUNTER_THRESHOLD
    counter_slope: float = COUNTER_SLOPE
    counter_fail_threshold: float = COUNTER_FAIL_THRESHOLD
    share_reserve: float = SHARE_RESERVE
    share_scale: float = SHARE_SCALE
    signal_scale: float = SIGNAL_SCALE
    repro_endowment: float = REPRO_ENDOWMENT
    max_pop: int = MAX_POP


# ── 种群容器 ──────────────────────────────────────────────────────────────


@dataclass
class Individual:
    """一个个体（可变状态；基因组是不可变值对象）。

    Attributes:
        threat_cell: 最近攻击者所在格（-1 = 无记忆；B6 威胁传感通道）。
        threat_age: 距上次受击的 tick 数（THREAT_NONE_AGE = 从未受击；
            每 tick +1，≥ THREAT_MEMORY_TICKS 失活）。
    """

    uid: int
    genome: Genome
    cell: int
    energy: float
    age: int = 0
    parent: int = -1          # -1 = 初代
    birth_tick: int = 0
    alive: bool = True
    death_tick: int = -1
    death_cause: str = ""
    threat_cell: int = -1
    threat_age: int = THREAT_NONE_AGE
    # W4 细胞尺度字段（见 biosphere/cell.py：细胞 = agent，与个体尺度分开）
    diff_state: int = 0             # 双稳态分化状态（0/1）
    # C3 v2：对各族徽章的反应状态（出生 = 基因版；可被打斗结果改写）
    response_state: np.ndarray = None  # type: ignore[assignment]
    # W4 团公共池流水（作弊者判据数据源）
    pool_in: float = 0.0
    pool_out: float = 0.0


@dataclass
class Population:
    """存活种群（按 uid 升序；死者进 graveyard 供谱系用）。"""

    individuals: List[Individual] = field(default_factory=list)
    graveyard: List[Individual] = field(default_factory=list)
    next_id: int = 0
    #: B19b light 长跑：``False`` = 死者不进 graveyard。graveyard 只被
    #: ``biosphere/diagnose.py`` 读（谱系诊断），**不参与任何状态转移**；
    #: 6,000 tick 下会累积约 20 万对象 ⇒ 长跑关掉它，短跑/完整路径默认保留。
    retain_dead: bool = True

    def alive_ids(self) -> List[int]:
        return [ind.uid for ind in self.individuals]

    def occupant_map(self, n_cells: int) -> np.ndarray:
        """cell → uid（-1 = 空）。个体已按 uid 排序 ⇒ 无写冲突。"""
        occ = np.full(n_cells, -1, dtype=np.int64)
        for ind in self.individuals:
            occ[ind.cell] = ind.uid
        return occ

    def total_energy(self) -> float:
        return math.fsum(ind.energy for ind in self.individuals)

    def spawn(self, genome: Genome, cell: int, energy: float,
              tick: int, parent: int,
              response_state=None) -> Individual:
        ind = Individual(
            uid=self.next_id, genome=genome, cell=cell, energy=energy,
            age=0, parent=parent, birth_tick=tick,
        )
        if response_state is not None:
            ind.response_state = np.asarray(response_state, dtype=np.float64)
        self.next_id += 1
        self.individuals.append(ind)
        return ind


# ── 裁定事件与报告 ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResolveEvent:
    """单条意图的裁定记录（进度量日志）。"""

    tick: int
    actor_id: int
    kind: str           # 动作名（ACTION_NAMES）
    status: str         # ResolveStatus
    target_cell: int
    target_id: int
    actor_energy_delta: float  # 该意图给作者带来的净能量变化（含成本）
    #: B14.2 反击伤害（攻击者因本次攻击付出的反击能量；0 = 无反击）
    counter_damage: float = 0.0
    #: C3 v2（B14.3）：当事双方的徽章/强度/目标是否在发射（判据数据源）
    actor_badge: int = -1
    target_badge: int = -1
    target_strength: float = 0.0
    target_emitting: int = 0
    #: B16：致死掠食取走的能量（0 = 未致死；军备竞赛"收益"的唯一通道）
    loot: float = 0.0


@dataclass
class ResolveReport:
    """一个 tick 的裁定报告（计数 + 台账 + 生卒）。"""

    tick: int
    events: List[ResolveEvent] = field(default_factory=list)
    births: List[Tuple[int, int, int]] = field(default_factory=list)   # (父, 子, 格)
    deaths: List[Tuple[int, str]] = field(default_factory=list)        # (uid, 原因)
    energy_paid: float = 0.0     # 意图成本总支付
    energy_in: float = 0.0       # 系统外/世界侧流入（吃资源、掠食）
    endowment_paid: float = 0.0  # 繁殖 endowment（系统内转移，另计）
    # W4 团机制台账（groups.py 写）
    group_share_events: int = 0
    group_births: List[Tuple[int, int, int, int]] = field(default_factory=list)

    def status_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for ev in self.events:
            key = f"{ev.kind}:{ev.status}"
            out[key] = out.get(key, 0) + 1
        return out


# ── 第一段：意图提交（并行写缓冲，仅排序与自检） ──────────────────────────


def commit_intents(population: Population, intents: List[Intent]) -> List[Intent]:
    """第一段：收集意图缓冲，按 actor_id 稳定排序（裁定顺序）。

    只做排序与结构性自检，**不碰种群/世界状态** ⇒ 意图写入无共享、
    无先后依赖（"并行"的含义）。
    """
    for it in intents:
        if it.kind == ActionKind.MOVE and it.target_cell < 0:
            raise ValueError(f"MOVE 意图缺 target_cell: {it}")
    return sorted(intents, key=lambda it: it.actor_id)


# ── 第二段：裁定 ──────────────────────────────────────────────────────────


def _adjacent(world: WorldView, a: int, b: int) -> bool:
    """同格或 von Neumann 邻接。"""
    if a == b:
        return True
    return bool(np.any(world.neighbors[a] == b))


def _habitat_ok(world: WorldView, cell: int, controller,
                 habitat_min: float = None) -> bool:
    """栖息准入（B11 补充指令：栖息 = 世界字段约束，不写标签）。

    类型由两个栖息基因定（cell.cell_type），准入按世界字段判：
    水生 ⇒ depth>0；陆生 ⇒ soil_water ≥ world.soil_water_min（阈值从
    world_adapter 读）；两栖/未定准 ⇒ 物理可待（代价在代谢侧）。
    """
    return can_occupy(cell_type(controller), world, cell)


def resolve_tick(
    population: Population,
    intents: List[Intent],
    world: WorldView,
    resource: SyntheticResource,
    signal_layer: np.ndarray,
    tick: int,
    rng: np.random.Generator,
    controllers: Dict[int, object],   # uid → ControllerParams（缓存）
    cfg: Optional[ResolveConfig] = None,
    badge_layer: Optional[np.ndarray] = None,   # C3 v2 (K, n) 徽章通道层
    c3=None,                                     # C3Config（None = 关）
    evo=None,                                    # EvolutionConfig（M4）
) -> ResolveReport:
    """第二段：按 uid 升序结算全部意图（就地修改种群/资源/信号层）。

    Args:
        intents: 已经 :func:`commit_intents` 排序的意图列表。
        controllers: uid → ControllerParams（决策侧缓存，裁定只读）。
        rng: 唯一随机源；本函数内**只**在繁殖突变处消费（按 uid 序）。

    Returns:
        ResolveReport（事件流水 + 台账 + 生卒）。
    """
    cfg = cfg or ResolveConfig()
    report = ResolveReport(tick=tick)
    n_cells = world.n_cells

    occupant = population.occupant_map(n_cells)
    by_uid = {ind.uid: ind for ind in population.individuals}

    # ── 2a. MOVE 声明（一格一写者：dest 先到先得；栖息不过线不声明） ──
    writer_of_cell = np.full(n_cells, -1, dtype=np.int64)
    claims: Dict[int, int] = {}       # actor → dest（dest ≠ 当前格）
    for it in intents:
        ind = by_uid.get(it.actor_id)
        if ind is None or not ind.alive or it.kind != ActionKind.MOVE:
            continue
        dest = it.target_cell
        if dest == ind.cell:
            continue                  # 原地不动：无需声明，直接算成功
        if writer_of_cell[dest] != -1:
            continue                  # 一格一写者竞争失败 ⇒ 不声明
        controller = controllers[it.actor_id]
        if not _habitat_ok(world, dest, controller, cfg.habitat_min):
            continue                  # 栖息拦截 ⇒ 不声明
        writer_of_cell[dest] = it.actor_id
        claims[it.actor_id] = dest

    # ── 2b. 位移链/环分解：沿"目标格占用者是否也撤离"递归定性 ──
    #   occupant_of_dest 无主 ⇒ 链通；成环 ⇒ 环通；撞上不动的占用者 ⇒ 链断。
    #   （静态 departing 数组会漏判"让位者自己后来失败"⇒ 双占 bug；链分解
    #     以占用者最终是否真撤离为准，是充要条件。）
    mover_of_cell = np.full(n_cells, -1, dtype=np.int64)  # 格 → 撤离者 uid
    for actor, dest in claims.items():
        mover_of_cell[by_uid[actor].cell] = actor
    chain_ok: Dict[int, bool] = {}    # 格 → 该格作为链/环节点是否最终让位

    def _chain_ok_from(cell: int) -> bool:
        """从目标格出发走占用链：空/环 ⇒ True；撞不动占用者 ⇒ False。"""
        stack: List[int] = []
        cur = cell
        verdict: Optional[bool] = None
        while True:
            if cur in chain_ok:
                verdict = chain_ok[cur]
                break
            if cur in stack:           # 成环（swap / 多环）⇒ 环上全部让位
                verdict = True
                break
            occ = int(occupant[cur])
            if occ == -1:              # 空格 ⇒ 链通
                verdict = True
                break
            mover = int(mover_of_cell[cur])
            if mover == -1:            # 占用者不撤离 ⇒ 链断
                verdict = False
                break
            stack.append(cur)
            cur = claims[mover]        # 沿占用者的目标格继续
        for c in stack:
            chain_ok[c] = verdict
        chain_ok[cell] = verdict
        return verdict

    for actor, dest in claims.items():
        _chain_ok_from(dest)

    # ── 2c. 按 uid 序生效：先位移（裁决后的位置），后其它效果 ──
    for it in intents:
        ind = by_uid.get(it.actor_id)
        if ind is None or not ind.alive:
            report.events.append(ResolveEvent(
                tick, it.actor_id, ACTION_NAMES[int(it.kind)],
                "actor_dead", it.target_cell, it.target_id, 0.0))
            continue
        if it.kind != ActionKind.MOVE:
            continue
        controller = controllers[it.actor_id]
        delta = -it.energy_cost          # 尝试即付成本
        ind.energy += delta
        report.energy_paid += it.energy_cost
        dest = claims.get(it.actor_id, -1)
        if dest < 0:
            if it.target_cell == ind.cell:
                status = ResolveStatus.OK
            elif not _habitat_ok(world, it.target_cell, controller,
                                cfg.habitat_min):
                status = ResolveStatus.STAY_HABITAT
            else:
                status = ResolveStatus.STAY_CONTESTED
        elif chain_ok.get(dest, False):
            occupant[ind.cell] = -1
            ind.cell = dest
            occupant[dest] = it.actor_id
            status = ResolveStatus.OK
        else:
            status = ResolveStatus.STAY_OCCUPIED
        report.events.append(ResolveEvent(
            tick, it.actor_id, ACTION_NAMES[int(ActionKind.MOVE)], status,
            it.target_cell, it.target_id, delta))

    # ── 2d. 按 uid 序生效：EAT / ATTACK / SHARE / SIGNAL / REPRODUCE ──
    for it in intents:
        ind = by_uid.get(it.actor_id)
        if ind is None or not ind.alive:
            report.events.append(ResolveEvent(
                tick, it.actor_id, ACTION_NAMES[int(it.kind)],
                "actor_dead", it.target_cell, it.target_id, 0.0))
            continue

        controller = controllers[it.actor_id]
        kind = it.kind
        if kind == ActionKind.MOVE:
            continue                  # 位移已在 2c 结算
        delta = -it.energy_cost          # 尝试即付成本
        ind.energy += delta
        report.energy_paid += it.energy_cost

        if kind == ActionKind.EAT:
            avail = float(resource.amount[ind.cell])
            if avail <= 0.0:
                status = ResolveStatus.EAT_EMPTY
            else:
                gain = min(controller.eat_amount, avail)
                resource.amount[ind.cell] = avail - gain
                ind.energy += gain
                delta += gain
                report.energy_in += gain
                status = ResolveStatus.OK
            report.events.append(ResolveEvent(
                tick, it.actor_id, ACTION_NAMES[int(kind)], status,
                ind.cell, -1, delta))

        elif kind == ActionKind.ATTACK:
            target = by_uid.get(it.target_id)
            counter = 0.0          # B14.2：无反击默认 0（无目标/失败早退）
            loot_gained = 0.0      # B16：致死掠食所得（未致死 = 0）
            if (target is None or not target.alive
                    or not _adjacent(world, ind.cell, target.cell)):
                status = ResolveStatus.ATTACK_NO_TARGET
            else:
                power = (cfg.attack_power_base
                         + cfg.attack_power_drive * controller.attack_drive)
                if c3 is not None and c3.enabled and c3.strength_effect:
                    power += c3.strength_power_coef * controller.strength
                # B14.2 反击风险：目标 defense 折成反击威力；超额越大攻击者
                # 越疼（弱者不可无损白打强者；打同强度 excess=−4 无反击）
                excess = (cfg.counter_defense_scale
                          * controllers[target.uid].defense - power)
                counter = (cfg.counter_slope * (excess - cfg.counter_threshold)
                           if excess > cfg.counter_threshold else 0.0)
                if excess > cfg.counter_fail_threshold:
                    # 攻击失败：目标无伤、无掠食；反击能量转移给防御方
                    ind.energy -= counter
                    target.energy += counter
                    delta -= counter
                    report.events.append(ResolveEvent(
                        tick, it.actor_id, ACTION_NAMES[int(kind)],
                        ResolveStatus.ATTACK_FAILED, ind.cell, it.target_id,
                        delta, counter))
                    continue
                damage = power * (1.0 - cfg.defense_mit * controllers[target.uid].defense)
                before = target.energy
                target.energy -= damage
                # B6：命中即写威胁记忆（致死也写——谱系/度量可追溯；
                # 活着的个体才用得上：致伤者的逃逸窗口由此打开）
                target.threat_cell = ind.cell
                target.threat_age = 0
                if target.energy <= 0.0:
                    target.alive = False
                    target.death_tick = tick
                    target.death_cause = "attacked"
                    occupant[target.cell] = -1   # 尸格当场释放
                    loot = cfg.attack_loot_frac * before
                    ind.energy += loot
                    delta += loot
                    report.energy_in += loot
                    report.deaths.append((target.uid, "attacked"))
                    loot_gained = loot
                    status = ResolveStatus.OK
                else:
                    status = ResolveStatus.ATTACK_SURVIVED
                    if cfg.contact_loot:
                        # B17.2：命中即转移（伤害不再纯销毁；不额外造能）
                        transferred = min(damage, max(0.0, before))
                        loot = cfg.attack_loot_frac * transferred
                        ind.energy += loot
                        delta += loot
                        report.energy_in += loot
                        loot_gained = loot
                if counter > 0.0:
                    # 反击仍生效：攻击者付、防御方得（守恒）
                    ind.energy -= counter
                    target.energy += counter
                    delta -= counter
            report.events.append(ResolveEvent(
                tick, it.actor_id, ACTION_NAMES[int(kind)], status,
                ind.cell, it.target_id, delta, counter,
                actor_badge=int(controller.badge),
                target_badge=(int(controllers[target.uid].badge)
                              if target is not None else -1),
                target_strength=(float(controllers[target.uid].strength)
                                 if target is not None else 0.0),
                target_emitting=(
                    int(visible_badge(badge_layer[:, target.cell],
                                      c3.emit_visible_min) >= 0)
                    if (badge_layer is not None and target is not None)
                    else 0),
                loot=loot_gained))

        elif kind == ActionKind.SHARE:
            target = by_uid.get(it.target_id)
            if (target is None or not target.alive
                    or not _adjacent(world, ind.cell, target.cell)):
                status = ResolveStatus.SHARE_NO_TARGET
            else:
                transfer = min(it.payload, ind.energy - cfg.share_reserve)
                if transfer <= 0.0:
                    status = ResolveStatus.SHARE_NO_TARGET
                else:
                    ind.energy -= transfer
                    target.energy += transfer
                    delta -= transfer
                    status = ResolveStatus.OK
            report.events.append(ResolveEvent(
                tick, it.actor_id, ACTION_NAMES[int(kind)], status,
                ind.cell, it.target_id, delta))

        elif kind == ActionKind.SIGNAL:
            signal_layer[ind.cell] += cfg.signal_scale * it.payload
            # C3 v2：同时写徽章通道（含义为空；强度 = payload）
            if badge_layer is not None and c3 is not None and c3.enabled:
                badge_layer[int(controller.badge), ind.cell] += it.payload
            status = ResolveStatus.SIGNAL_OK
            report.events.append(ResolveEvent(
                tick, it.actor_id, ACTION_NAMES[int(kind)], status,
                ind.cell, -1, delta, 0.0,
                actor_badge=int(controller.badge)))

        elif kind == ActionKind.REPRODUCE:
            if ind.energy < controller.repro_thresh:
                status = ResolveStatus.REPRO_NO_ENERGY
            elif len(population.individuals) >= cfg.max_pop:
                status = ResolveStatus.REPRO_NO_ROOM
            else:
                free_cell = -1
                for nb in world.neighbors[ind.cell]:
                    if occupant[int(nb)] == -1:
                        free_cell = int(nb)
                        break
                if free_cell < 0:
                    status = ResolveStatus.REPRO_NO_ROOM
                else:
                    child_genome, _rec = mutate(ind.genome, rng, uid=population.next_id)
                    ind.energy -= cfg.repro_endowment
                    delta -= cfg.repro_endowment
                    report.endowment_paid += cfg.repro_endowment
                    child = population.spawn(
                        child_genome, free_cell, cfg.repro_endowment, tick, ind.uid)
                    occupant[free_cell] = child.uid
                    by_uid[child.uid] = child
                    report.births.append((ind.uid, child.uid, free_cell))
                    status = ResolveStatus.OK
            report.events.append(ResolveEvent(
                tick, it.actor_id, ACTION_NAMES[int(kind)], status,
                ind.cell, -1, delta))

        else:  # pragma: no cover - ActionKind 穷尽性护栏
            raise ValueError(f"未知动作类型: {kind}")

    return report


__all__ = [
    "HABITAT_MIN", "ATTACK_POWER_BASE", "ATTACK_POWER_DRIVE", "DEFENSE_MIT",
    "ATTACK_LOOT_FRAC", "SHARE_RESERVE", "SHARE_SCALE", "SIGNAL_SCALE",
    "REPRO_ENDOWMENT", "MAX_POP", "THREAT_MEMORY_TICKS", "THREAT_NONE_AGE",
    "ResolveConfig",
    "Individual", "Population", "ResolveEvent", "ResolveReport",
    "commit_intents", "resolve_tick",
]
