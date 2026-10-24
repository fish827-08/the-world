"""biosphere.simulator：MVP 主循环（CPU 参考实现，W0-B）。

通俗理解
--------
一个 tick 的顺序（写死，保确定性）：

1. **世界更新**：资源向容量回复（季节调制）、信号层衰减；
2. **感知**：每个个体读自身格 + 4 邻域（食物/同类/威胁；信号层 MVP 期
   只进度量与交互网络，**不入决策**——发送者获益通道是 W2 接线点）；
3. **决策**（第一段·并行写意图）：按 uid 序为每个个体算六类动作得分，
   加同一 rng 的固定顺序抖动，取 argmax 写一个 :class:`Intent`；
4. **提交 + 裁定**（第二段）：``commit_intents`` → ``resolve_tick``；
5. **代谢与生死**：按 uid 序扣基础代谢（生境错配加罚）、年龄 +1、
   判饿毙/寿终；死者进墓地，种群按 uid 重排；
6. **记录**：观测台记一帧。

确定性：唯一随机源是构造时注入的 ``np.random.Generator``；消费顺序 =
uid 序（决策抖动）→ uid 序（繁殖突变）。同 seed 两次逐位一致。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np

from biosphere.action import ActionKind, Intent, intent_cost
from biosphere.c3 import C3Config, learning_update, visible_badge
from biosphere.cell import HabitatConfig
from biosphere.genome import EvolutionConfig
from biosphere.cell import (
    Differentiation, adhesion_expression, cell_type, occupancy_penalty,
    update_differentiation,
)
from biosphere.develop import ControllerCache, ControllerParams
from biosphere.genome import Genome, random_genome
from biosphere.groups import (
    ADHESION_COHESION, ADHESION_STICKY, GroupIndex, W4Config,
    build_adhesion_chain_index, build_group_index,
    group_expression_variance, group_forage_dividend, reproduce_groups,
    share_group_energy, KIN_COHESION_WEIGHT, STICKY_JITTER_SCALE,
)
from biosphere.metrics import SimLog
from biosphere.predation import (
    PredationConfig, apply_oracle_traits, spread_founder_traits,
)
from biosphere.resolve import (
    THREAT_MEMORY_TICKS, THREAT_NONE_AGE, Individual, Population,
    ResolveConfig, commit_intents, resolve_tick,
)
from biosphere.world_adapter import synthetic_world

# ── MVP 常量（冻结；调参须改规格页并重走预注册） ──────────────────────────

#: 基础代谢缩放（每 tick 能耗 = METABOLISM_SCALE × controller.metabolism）
METABOLISM_SCALE: float = 0.035
#: 信号层每 tick 衰减系数
SIGNAL_DECAY: float = 0.90
#: 季节对资源回复的调制幅度（回复率 ×(1+SEASON_AMP·sin(2π·phase))）
SEASON_AMP: float = 0.30
#: 决策得分抖动幅度（ε-探索；同 seed 下是确定性噪声）
JITTER: float = 0.15
#: SHARE payload 缩放（payload = 此值 × share_drive，决策侧算定）
SHARE_PAYLOAD_SCALE: float = 10.0
#: 威胁记忆逃逸项权重（B6：受击后 MOVE 评分加
#: sense_threat × 此值 × 记忆衰减 × (候选格-当前格) 的威胁距离差）
THREAT_MEMORY_WEIGHT: float = 3.0
#: 初始个体数 / 初始能量
INIT_POP: int = 48
INIT_ENERGY: float = 60.0

#: 动作得分计算顺序（argmax 平手取小索引；顺序即 ActionKind 序）
_ACTION_ORDER = (
    ActionKind.MOVE, ActionKind.EAT, ActionKind.ATTACK,
    ActionKind.SHARE, ActionKind.SIGNAL, ActionKind.REPRODUCE,
)


def _food01(amount: float) -> float:
    """资源量归一化到 [0,1]（4 单位 = 满格；防 MOVE 食物项压垮其它动作）。"""
    a = float(amount)
    if a <= 0.0:
        return 0.0
    return a / 4.0 if a < 4.0 else 1.0


@dataclass
class BioMVPConfig:
    """MVP 模拟配置（默认 = 规格页冻结值）。"""

    rows: int = 32
    cols: int = 64
    seed: int = 0
    ticks: int = 1000
    init_pop: int = INIT_POP
    init_energy: float = INIT_ENERGY
    # B6 对照臂开关：威胁传感（邻格攻击倾向排斥 + 受击记忆逃逸项）。
    # False = 个体完全看不到威胁（"无 threat 传感"对照臂）。
    threat_sensing: bool = True
    # W4 多细胞/合作机制（默认全关 ⇒ 与 W0-B 逐位等价； experiments w4 显式开）
    w4: W4Config = None  # type: ignore[assignment]
    # C3 v2 发送者获益通道（默认全关；徽章含义为空，只写物理）
    c3: C3Config = None  # type: ignore[assignment]
    # B14.4 参数化（默认 = 历史常量 ⇒ 逐位等价）
    habitat: HabitatConfig = None  # type: ignore[assignment]
    evolution: EvolutionConfig = None  # type: ignore[assignment]
    # B16 捕食-猎物军备竞赛（默认全关 ⇒ 与既有口径逐位等价）
    predation: PredationConfig = None  # type: ignore[assignment]
    # 合成世界参数（透传 synthetic_world）
    land_frac: float = 0.45
    patch_count: int = 10
    patch_radius: int = 2
    patch_capacity: float = 12.0
    bg_capacity: float = 1.0
    regrow_rate: float = 0.08
    tilt_rad: float = 0.20
    season_period: int = 400
    # 裁定参数
    resolve: ResolveConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.resolve is None:
            self.resolve = ResolveConfig()
        if self.w4 is None:
            self.w4 = W4Config()
        if self.c3 is None:
            self.c3 = C3Config()
        if self.habitat is None:
            self.habitat = HabitatConfig()
        if self.evolution is None:
            self.evolution = EvolutionConfig()
        if self.predation is None:
            self.predation = PredationConfig()


class BiosphereSimulator:
    """MVP 模拟器（CPU 参考实现；GPU 版 W3 另行对拍）。"""

    def __init__(self, config: Optional[BioMVPConfig] = None,
                 world=None, resource=None) -> None:
        """构造模拟器。

        ``world`` / ``resource`` 可注入真实世界适配结果（B17.3；须同时给，
        且 ``world.rows/cols`` 与配置一致）。不注入时走 ``synthetic_world``。
        """
        self.cfg = config or BioMVPConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        if (world is None) != (resource is None):
            raise ValueError("world 与 resource 必须同时注入（或都不注入）")
        if world is not None:
            if (int(world.rows), int(world.cols)) != (
                    int(self.cfg.rows), int(self.cfg.cols)):
                raise ValueError(
                    f"注入世界 {world.rows}×{world.cols} 与配置 "
                    f"{self.cfg.rows}×{self.cfg.cols} 不一致")
            self.world, self.resource = world, resource
        else:
            self.world, self.resource = synthetic_world(
                rows=self.cfg.rows, cols=self.cfg.cols, seed=self.cfg.seed,
                land_frac=self.cfg.land_frac,
                patch_count=self.cfg.patch_count,
                patch_radius=self.cfg.patch_radius,
                patch_capacity=self.cfg.patch_capacity,
                bg_capacity=self.cfg.bg_capacity,
                regrow_rate=self.cfg.regrow_rate,
                tilt_rad=self.cfg.tilt_rad,
                season_period=self.cfg.season_period,
            )
        self.signal_layer = np.zeros(self.world.n_cells, dtype=np.float64)
        # C3 v2：K 通道徽章层（关时也建但零写入/零衰减 ⇒ 无行为差异）
        self.badge_layer = np.zeros((self.cfg.c3.badge_k,
                                     self.world.n_cells), dtype=np.float64)
        self.population = Population()
        self.cache = ControllerCache()
        self._ctrl: Dict[int, ControllerParams] = {}   # 当前 tick 的控制器快照
        self.log = SimLog()
        self.tick_count = 0
        self._group_index = GroupIndex()   # 邻接团（对照镜头；step 重建）
        self._chain_index = GroupIndex()   # 粘附链团（主张镜头；step 重建）
        self._spawn_initial()

    # ── 初始化 ──

    def _spawn_initial(self) -> None:
        """初代个体：随机落格（只落陆地，保证起步可觅食）。"""
        land = np.flatnonzero(self.world.surface_type == 0)  # SURFACE_LAND
        if land.size < self.cfg.init_pop:
            raise RuntimeError("陆地格不足，放不下初代种群")
        picks = self.rng.choice(land, size=self.cfg.init_pop, replace=False)
        for cell in sorted(int(c) for c in picks):
            genome = random_genome(self.rng, uid=self.population.next_id)
            genome = spread_founder_traits(genome, self.rng,
                                           self.cfg.predation)
            self.population.spawn(
                genome, cell, self.cfg.init_energy, tick=0, parent=-1)

    def spawn_founder(self, cell: int, energy: float,
                      gene_overrides: Optional[Dict[int, float]] = None
                      ) -> Individual:
        """B8 双型对照：在指定格放一个初代（基因组随机 + 槽位覆盖）。

        Args:
            gene_overrides: {槽位: 原始基因值}，如 {6: +3.0} = 高攻击型。
                只覆盖给定槽位，其余仍随机（与常规初代同分布）。
        """
        genome = random_genome(self.rng, uid=self.population.next_id)
        if gene_overrides:
            genes = genome.genes.copy()
            for slot, value in gene_overrides.items():
                genes[int(slot)] = float(value)
            genome = Genome(genes=genes, uid=genome.uid)
        return self.population.spawn(
            genome, cell, energy, tick=0, parent=-1)

    def _controllers(self) -> Dict[int, ControllerParams]:
        """uid → 控制器参数快照（B16：oracle 极值在此钉上；关时 = 缓存原值）。

        所有需要"当前控制器"的地方（决策、裁定、记录）**统一走这里**，
        否则 oracle 只在一半路径生效（B15 同类教训）。
        """
        controllers = {ind.uid: self.cache.get(ind.genome)
                       for ind in self.population.individuals}
        if self.cfg.predation.enabled:
            controllers = apply_oracle_traits(controllers, self.cfg.predation)
        return controllers

    def _ctrl_of(self, uid: int, genome=None) -> ControllerParams:
        """取一个体的当前控制器（快照缺失时回退发育缓存）。

        快照缺失的情形：单测直接调 ``_decide``（没走 ``step``）、以及刚在
        resolve 里出生的个体 ⇒ 回退到纯发育值（**与 oracle 无关**的路径安全）。
        """
        c = self._ctrl.get(uid) if self._ctrl else None
        if c is None:
            c = self.cache.get(genome)
        return c

    # ── 决策（第一段） ──

    def _decide(self, ind: Individual) -> Optional[Intent]:
        """为一个个体写一个意图（或 None = 本 tick 不动）。

        单遍扫描 [自身格 + 4 邻域] 同时收集：食物、同类数、威胁（邻者攻击
        倾向之和）、最弱邻者（攻击目标）、饥饿邻者（分享目标）。
        """
        world = self.world
        c = self._ctrl_of(ind.uid, ind.genome)
        cell = ind.cell
        occ = self._occupant
        by_uid = self._by_uid
        threat_on = self.cfg.threat_sensing

        threat = 0.0
        target_attack = -1
        weakest = math.inf
        target_share = -1
        # C3 v2：可见徽章 ⇒ 反应权重（含义为空；正=倾向攻、负=倾向避）
        c3_on = self.cfg.c3.enabled
        resp = ind.response_state if c3_on else None
        attack_bias = 0.0
        badge_cells: List[int] = []      # 带负反应徽章的邻格（回避用）
        for nb in [cell] + [int(x) for x in world.neighbors[cell]]:
            uid = int(occ[nb])
            if uid == -1 or uid == ind.uid:
                continue
            other = by_uid[uid]
            if threat_on:
                threat += self._ctrl_of(uid, other.genome).attack_drive
            if other.energy < weakest:
                weakest, target_attack = other.energy, uid
            if target_share == -1 and other.energy < 0.5 * ind.energy:
                target_share = uid
            if c3_on and resp is not None:
                b = visible_badge(self.badge_layer[:, nb],
                                  self.cfg.c3.emit_visible_min)
                if b >= 0:
                    r = float(resp[b])
                    if r > 0.0:
                        # 正反应 ⇒ 优先把该徽章载体作为攻击目标
                        if r > attack_bias:
                            attack_bias = r
                            target_attack = uid
                    elif r < 0.0:
                        badge_cells.append(nb)

        # B6 威胁记忆（受击后 THREAT_MEMORY_TICKS 内有效）
        mem_active = (threat_on and ind.threat_cell >= 0
                      and ind.threat_age < THREAT_MEMORY_TICKS)
        mem_decay = (1.0 - ind.threat_age / THREAT_MEMORY_TICKS
                     if mem_active else 0.0)
        d_threat_cur = (world.grid_distance(cell, ind.threat_cell)
                        if mem_active else 0.0)

        food_here = float(self.resource.amount[cell])

        # 六类得分（顺序 = _ACTION_ORDER；全部 O(1) 量纲，抖动 ±JITTER 才有效）
        scores = np.zeros(6, dtype=np.float64)
        # MOVE：最好邻格得分（食物梯度 + 同类吸引 - 威胁排斥 + 逃逸项；
        # 食物项归一化）
        w4 = self.cfg.w4
        cohesion_on = w4.enabled and w4.adhesion
        # B11：粘附表达受双稳态分化调制（state=0 只表达一半 ⇒ 分工）
        adhesion_eff = (adhesion_expression(c.adhesion, ind.diff_state)
                        if cohesion_on else c.adhesion)
        sticky = (cohesion_on and adhesion_eff >= ADHESION_STICKY
                  and self._chain_index.size(ind.uid) >= 2)
        my_gid = self._chain_index.group_id(ind.uid) if cohesion_on else None
        best_nb, best_score = cell, 0.0
        for nb in world.neighbors[cell]:
            nb = int(nb)
            if sticky:
                # 粘附硬过滤：团在时只考虑与团成员相邻的格（无一符合则不动）
                nb_uid = int(occ[nb])
                touches_group = (
                    nb_uid != -1
                    and self._chain_index.group_id(nb_uid) == my_gid)
                if not touches_group:
                    for nb2 in world.neighbors[nb]:
                        u2 = int(occ[int(nb2)])
                        if u2 != -1 and self._chain_index.group_id(u2) == my_gid:
                            touches_group = True
                            break
                if not touches_group:
                    continue
            nb_food = _food01(self.resource.amount[nb])
            nb_kin = 1.0 if int(occ[nb]) != -1 else 0.0
            nb_threat = 0.0
            uid = int(occ[nb])
            if uid != -1 and threat_on:
                nb_threat = self._ctrl_of(uid, by_uid[uid].genome).attack_drive
            sc = (c.sense_food * nb_food
                  + c.sense_kin * adhesion_eff * nb_kin
                  - c.sense_threat * nb_threat)
            if c3_on and badge_cells:
                # 候选格贴着"被回避的徽章载体" ⇒ 扣分（梯度指向远离）
                for bc in badge_cells:
                    if world.grid_distance(nb, bc) <= 1.0:
                        sc -= self.cfg.c3.response_avoid_scale
            if cohesion_on and uid != -1 \
                    and self._chain_index.group_id(uid) == my_gid:
                if w4.kin_cohesion:
                    # W4c-G2：计数加权——候选格旁链团成员越多越值得去
                    n_kin = 1
                    for nb2 in world.neighbors[nb]:
                        u2 = int(occ[int(nb2)])
                        if u2 != -1 and u2 != ind.uid \
                                and self._chain_index.group_id(u2) == my_gid:
                            n_kin += 1
                    sc += KIN_COHESION_WEIGHT * adhesion_eff * n_kin
                else:
                    sc += ADHESION_COHESION * adhesion_eff   # 贴团奖励（二值）
            if mem_active:
                # 逃逸项：候选格比当前格远离最近攻击者 ⇒ 加分（线性于距离差）
                sc += (c.sense_threat * THREAT_MEMORY_WEIGHT * mem_decay
                       * (world.grid_distance(nb, ind.threat_cell)
                          - d_threat_cur))
            if sc > best_score:
                best_nb, best_score = nb, sc
        scores[0] = c.move_bias * best_score + (0.05 if best_nb != cell else 0.0)
        # EAT（站在食物上时强竞争）
        scores[1] = c.eat_greed * 2.0 * _food01(food_here)
        # ATTACK（C3 v2：正反应徽章载体加分销——无"信号→攻击欲"直规则，
        # 只有"对徽章载体的可遗传反应"）
        scores[2] = c.attack_drive * (1.0 if target_attack >= 0 else 0.0)
        if c3_on and attack_bias > 0.0:
            scores[2] += self.cfg.c3.response_attack_scale * attack_bias
        # SHARE
        scores[3] = c.share_drive * (1.0 if target_share >= 0 else 0.0)
        # SIGNAL：有威胁/有食物时略增强（W4 关信号零模型 ⇒ -inf 硬压，
        # 防 jitter 让 0 分动作胜出）
        signal_on = not (w4.enabled and not w4.signal)
        if signal_on:
            scores[4] = c.signal_drive * (0.5 + 0.25 * min(threat, 2.0)
                                          + 0.25 * min(_food01(food_here), 1.0))
            # B15.1 oracle：oracle-response 给保底发射（判据③要有分母）；
            # oracle-emitter 只允许强者发射（弱者的 SIGNAL 硬压 -inf）
            if c3_on and self.cfg.c3.oracle_response:
                scores[4] = max(scores[4],
                                self.cfg.c3.oracle_signal_floor)
            elif c3_on and self.cfg.c3.oracle_emitter:
                scores[4] = (10.0 if c.strength >= self.cfg.c3.emitter_strength_min
                             else -np.inf)
        elif c3_on and self.cfg.c3.oracle_emitter and \
                c.strength >= self.cfg.c3.emitter_strength_min:
            scores[4] = 10.0        # W4 关信号时 oracle-emitter 仍强制发射
        else:
            scores[4] = -np.inf
        # REPRODUCE（能量达标时强竞争）
        scores[5] = c.repro_drive * (
            2.0 if ind.energy >= c.repro_thresh else 0.0)

        # 确定性抖动 + argmax（平手取小索引）；W4c-G2：粘附个体降扩散
        jitter = self.rng.uniform(-JITTER, JITTER, size=6)
        if w4.enabled and w4.kin_cohesion and sticky:
            jitter = jitter * STICKY_JITTER_SCALE
        scores = scores + jitter
        k = int(np.argmax(scores))
        kind = _ACTION_ORDER[k]
        if scores[k] <= 0.0:
            return None

        if kind == ActionKind.MOVE:
            if best_nb == ind.cell:
                return None
            return Intent(actor_id=ind.uid, kind=kind, target_cell=best_nb,
                          energy_cost=intent_cost(kind, c))
        if kind == ActionKind.EAT:
            return Intent(actor_id=ind.uid, kind=kind, target_cell=ind.cell,
                          energy_cost=intent_cost(kind, c))
        if kind == ActionKind.ATTACK:
            return Intent(actor_id=ind.uid, kind=kind, target_id=target_attack,
                          energy_cost=intent_cost(kind, c))
        if kind == ActionKind.SHARE:
            payload = min(SHARE_PAYLOAD_SCALE * c.share_drive,
                          ind.energy - 1.0)
            if payload <= 0.0:
                return None
            return Intent(actor_id=ind.uid, kind=kind, target_id=target_share,
                          payload=payload, energy_cost=intent_cost(kind, c))
        if kind == ActionKind.SIGNAL:
            cost = intent_cost(kind, c)
            if c3_on:
                # C3 v2：发射耗能 = 基数 × emit_rate（可遗传 ⇒ 发射有代价）
                cost = self.cfg.c3.signal_emit_cost_base * c.emit_rate
            return Intent(actor_id=ind.uid, kind=kind, target_cell=ind.cell,
                          payload=c.signal_payload, energy_cost=cost)
        # REPRODUCE
        return Intent(actor_id=ind.uid, kind=kind,
                      energy_cost=intent_cost(kind, c))

    def _apply_oracle_response(self, controllers) -> None:
        """B15.1 oracle-response：按徽章载体真实 strength 设 response 符号。

        每 tick 统计各徽章通道“当前可见载体”的平均 strength，均值作分界：
        载体均强 ⇒ 该徽章 response = −1（回避）；弱 ⇒ +1（倾向攻）。
        覆写所有个体的 response_state（诊断正对照 = 完美接收器）。
        """
        c3 = self.cfg.c3
        k = c3.badge_k
        # 各徽章通道的可见载体 strength 列表
        vis: List[List[float]] = [[] for _ in range(k)]
        for ind in self.population.individuals:
            b = visible_badge(self.badge_layer[:, ind.cell],
                              c3.emit_visible_min)
            if b >= 0:
                vis[b].append(float(controllers[ind.uid].strength))
        means = [(math.fsum(v) / len(v)) if v else None for v in vis]
        known = [m for m in means if m is not None]
        if not known:
            return                      # 无人发射 ⇒ 无信号可反应（保持原状）
        mid = math.fsum(known) / len(known)
        signs = [0.0 if m is None else (-1.0 if m > mid else 1.0)
                 for m in means]
        for ind in self.population.individuals:
            if ind.response_state is None:
                ind.response_state = np.zeros(k, dtype=np.float64)
            else:
                ind.response_state = np.zeros(k, dtype=np.float64)
            for b in range(k):
                ind.response_state[b] = signs[b]

    def _c3_post_resolve(self, tick: int, report) -> None:
        """C3 v2 事后结算：新生儿反应状态、徽章重排（负对照）、记忆学习。"""
        c3 = self.cfg.c3
        by_uid = {ind.uid: ind for ind in self.population.individuals}
        # ① 新生儿：response_state = 发育版（基因遗传的起点）
        for parent, child, _cell in report.births:
            ind = by_uid.get(child)
            if ind is None or ind.response_state is not None:
                continue
            cc = self.cache.get(ind.genome)
            ind.response_state = np.array(cc.response, dtype=np.float64)
            # ② 徽章重排负对照：子代徽章基因随机化（断掉遗传）
            if c3.badge_reshuffle:
                from biosphere.genome import PARAM_SLOTS
                ind.genome = ind.genome.with_gene(
                    PARAM_SLOTS["badge"], float(self.rng.uniform(-4.0, 4.0)))
                cc2 = self.cache.get(ind.genome)
                ind.response_state = np.array(cc2.response, dtype=np.float64)
        # ③ 记忆学习（memory_enabled 才生效；首轮默认关）
        if not c3.memory_enabled:
            return
        for e in report.events:
            if e.kind != "attack" or e.target_id < 0:
                continue
            actor = by_uid.get(e.actor_id)
            target = by_uid.get(e.target_id)
            if actor is None or target is None:
                continue
            if actor.response_state is None or target.response_state is None:
                continue
            lr_a = self.cache.get(actor.genome).learning_rate
            lr_t = self.cache.get(target.genome).learning_rate
            countered = e.counter_damage > 0.0 or \
                e.status == "attack_failed"
            killed = e.status == "ok"
            # 攻击者：得手 ⇒ 学"攻该徽章"；被反击 ⇒ 学"避该徽章"
            learning_update(actor.response_state, e.target_badge,
                            1.0 if killed else (-1.0 if countered else 0.0),
                            lr_a, c3.memory_step)
            # 目标：被徽章 b 的攻击者打 ⇒ 学"避 b"；成功反击 ⇒ 学"攻 b"
            learning_update(target.response_state, e.actor_badge,
                            -1.0 if (killed or countered) else 0.0,
                            lr_t, c3.memory_step)

    # ── 世界更新 ──

    def _update_world(self, tick: int) -> None:
        """资源回复（季节调制）+ 信号衰减。"""
        phase = self.world.season_phase(tick)
        mult = 1.0 + SEASON_AMP * math.sin(2.0 * math.pi * phase)
        rate = self.resource.regrow_rate * mult
        cap = self.resource.capacity
        amt = self.resource.amount
        amt += rate * (cap - amt)
        np.clip(amt, 0.0, cap, out=amt)
        self.signal_layer *= SIGNAL_DECAY
        if self.cfg.c3.enabled:
            self.badge_layer *= SIGNAL_DECAY   # 徽章层同衰减

    # ── 代谢与生死 ──

    def _compact_dead(self) -> None:
        """把 resolve 阶段被杀者移出种群、进墓地（防代谢阶段重复计死）。"""
        survivors: List[Individual] = []
        for ind in self.population.individuals:
            if ind.alive:
                survivors.append(ind)
            elif self.population.retain_dead:
                self.population.graveyard.append(ind)
        self.population.individuals = survivors

    def _metabolize(self, tick: int, report) -> None:
        """按 uid 序扣代谢、加龄、判生死；死者进墓地。"""
        survivors: List[Individual] = []
        for ind in self.population.individuals:  # 已按 uid 排序
            c = self.cache.get(ind.genome)
            if ind.threat_age < THREAT_NONE_AGE:
                ind.threat_age += 1          # B6：威胁记忆老化
            # B11：栖息惩罚按细胞尺度字段判定（水生 depth>0 / 陆生
            # soil_water ≥ 世界阈值；两栖减半；未定准全付）
            penalty = occupancy_penalty(
                cell_type(c, self.cfg.resolve.habitat_min),
                self.world, ind.cell, self.cfg.habitat)
            cost = METABOLISM_SCALE * c.metabolism * (1.0 + penalty)
            ind.energy -= cost
            ind.age += 1
            if ind.energy <= 0.0:
                ind.alive = False
                ind.death_tick = tick
                ind.death_cause = "starved"
                if self.population.retain_dead:
                    self.population.graveyard.append(ind)
                report.deaths.append((ind.uid, "starved"))
                # B12：死者流水入账（修作弊者统计的生存偏差——只统计
                # 终存活者会把"蹭完就死"的作弊者洗掉）
                if self.cfg.w4.enabled:
                    self.log.dead_flows.append(
                        (ind.uid, ind.pool_in, ind.pool_out))
                continue
            if ind.age >= int(c.lifespan):
                ind.alive = False
                ind.death_tick = tick
                ind.death_cause = "old_age"
                if self.population.retain_dead:
                    self.population.graveyard.append(ind)
                report.deaths.append((ind.uid, "old_age"))
                if self.cfg.w4.enabled:
                    self.log.dead_flows.append(
                        (ind.uid, ind.pool_in, ind.pool_out))
                continue
            survivors.append(ind)
        self.population.individuals = survivors

    # ── 主循环 ──

    def _full_mixing(self) -> None:
        """W4 全混合零模型：tick 初把全部个体全局随机重排（空间结构归零）。"""
        n = self.world.n_cells
        k = len(self.population.individuals)
        if k == 0:
            return
        picks = self.rng.permutation(n)[:k]      # 互异的随机格
        for ind, cell in zip(self.population.individuals,
                             sorted(int(c) for c in picks)):
            ind.cell = cell                       # individuals 已按 uid 序

    def step(self, tick: int) -> None:
        """跑一个 tick。"""
        w4 = self.cfg.w4
        self._update_world(tick)
        if w4.enabled and w4.full_mixing:
            self._full_mixing()
        self._occupant = self.population.occupant_map(self.world.n_cells)
        self._by_uid = {ind.uid: ind for ind in self.population.individuals}
        # 两套团结构（B12）：邻接团=对照；粘附链团=机制与主张所用
        self._group_index = build_group_index(self.population, self.world)
        controllers = self._controllers()
        # 🔴 B16：决策侧也要读这份快照（含 oracle 钉死值）——否则
        #    attack_drive 的 oracle 只改威力、不改"想不想打"（半边生效）。
        self._ctrl = controllers
        self._chain_index = (build_adhesion_chain_index(
            self.population, self.world, controllers) if w4.enabled
            else GroupIndex())
        # B12：双稳态分化的输入 = **感知邻域**（邻接团大小 − 1），不是链团。
        # 🔴 死锁修复（2026-10-19）：先前用链团大小 ⇒ 非粘附细胞在链团里恒为
        #    单体 ⇒ 邻体数 0 ⇒ 永不翻到 state=1 ⇒ 粘附表达永远减半 ⇒ 链团
        #    永远不成形（自锁）。感知取邻接（细胞能看到什么），链团只作
        #    主张/机制镜头。
        if w4.enabled:
            for ind in self.population.individuals:
                ind.diff_state = update_differentiation(
                    ind.diff_state, self._group_index.size(ind.uid) - 1)

        # B15.1 oracle-response：每 tick 覆写 response_state 为“按徽章
        # 载体真实 strength 设符号”（诊断正对照：完美接收器）
        if self.cfg.c3.enabled and self.cfg.c3.oracle_response:
            self._apply_oracle_response(controllers)

        # 第一段：并行写意图（按 uid 序计算，但只写各自缓冲区）
        intents: List[Intent] = []
        for ind in self.population.individuals:
            it = self._decide(ind)
            if it is not None:
                intents.append(it)
        committed = commit_intents(self.population, intents)

        # 第二段：裁定（快照重建：resolve 内可能生子 ⇒ 覆盖 oracle 分组）
        controllers_now = self._controllers()
        report = resolve_tick(
            self.population, committed, self.world, self.resource,
            self.signal_layer, tick, self.rng, controllers_now,
            cfg=self.cfg.resolve,
            badge_layer=(self.badge_layer if self.cfg.c3.enabled else None),
            c3=(self.cfg.c3 if self.cfg.c3.enabled else None),
            evo=self.cfg.evolution,
        )
        if self.cfg.c3.enabled:
            self._c3_post_resolve(tick, report)

        # 代谢与生死（先压实 resolve 击杀者，防重复计死）
        self._compact_dead()
        self._metabolize(tick, report)

        # W4b：团共享 + 按团繁殖跑在**粘附链团**上（主张只看这套；
        # 邻接团仅作对照记录）
        if w4.enabled:
            # resolve 的个体繁殖也生子、resolve 也会杀人 ⇒ 结算与记录
            # 用**当刻重建**的两套索引 + 控制器快照
            controllers = self._controllers()
            index = build_adhesion_chain_index(self.population, self.world,
                                               controllers)
            self._group_index = build_group_index(self.population, self.world)
            self._chain_index = index
            if w4.group_sharing:
                share_group_energy(self.population, index, controllers, report)
            if w4.adhesion_benefit > 0.0:
                # W4c-G3：粘附团群体觅食分红（世界侧流入，计入 energy_in）
                gained = group_forage_dividend(self.population, index,
                                               w4.adhesion_benefit)
                report.energy_in += gained
            if w4.group_reproduction:
                n_before = len(self.log.group_repro_events)
                c_before = len(self.log.group_census)
                reproduce_groups(self.population, index, self.world,
                                 controllers, self.rng, tick, report,
                                 max_pop=self.cfg.resolve.max_pop,
                                 events_out=self.log.group_repro_events,
                                 census_out=self.log.group_census,
                                 thresh_per_capita=(
                                     w4.group_repro_thresh_per_capita))
            else:
                n_before = len(self.log.group_repro_events)
                c_before = len(self.log.group_census)
            # 繁殖新生个体不在 tick 开头的 controllers 快照里 ⇒ 记录前重建
            controllers = self._controllers()
            self.log.record_chain_frame(tick, self.population, index,
                                        controllers, self.world,
                                        events_tick=self.log
                                        .group_repro_events[n_before:],
                                        census_tick=self.log
                                        .group_census[c_before:])
            self.log.record_group_frame(tick, self.population,
                                        self._group_index, controllers,
                                        world=self.world)

        # 记录（帧 = tick 处理完的存活快照）
        c3_strengths = None
        if self.cfg.c3.enabled:
            # 直接查缓存（resolve 繁殖过 ⇒ tick 开头的 controllers 快照过期）
            c3_strengths = np.array(
                [self.cache.get(ind.genome).strength
                 for ind in self.population.individuals], dtype=np.float64)
        traits = None
        pred = self.cfg.predation
        if pred.enabled and pred.record_traits:
            end = self._controllers()
            traits = (
                np.array([end[ind.uid].defense
                          for ind in self.population.individuals],
                         dtype=np.float64),
                np.array([end[ind.uid].attack_drive
                          for ind in self.population.individuals],
                         dtype=np.float64),
            )
        self.log.record_frame(
            tick, self.population.individuals, self.world,
            c3=(self.cfg.c3 if self.cfg.c3.enabled else None),
            badge_layer=(self.badge_layer if self.cfg.c3.enabled else None),
            strengths_arr=c3_strengths, traits=traits)
        self.log.record_report(report)
        self.tick_count = tick + 1

    def run(self, ticks: Optional[int] = None,
            ckpt_dir=None, ckpt_every: int = 500, tag: str = "run",
            verbose_ckpt: bool = False) -> SimLog:
        """跑 ticks 个 tick（默认取配置值；支持 checkpoint 续跑）。

        Args:
            ticks: **总** tick 数（续跑时 = 目标终点，从当前 tick_count 继续）。
            ckpt_dir: checkpoint 目录（None = 不写；>1000 tick 跑批应传）。
            ckpt_every: 每多少 tick 写一个 checkpoint（默认 500）。
            tag: 文件/日志前缀。
        """
        n = self.cfg.ticks if ticks is None else ticks
        start = self.tick_count
        for tick in range(start, n):
            self.step(tick)
            if ckpt_dir is not None and (tick + 1) % ckpt_every == 0:
                from biosphere.checkpoint import append_run_log, save_checkpoint
                path = save_checkpoint(
                    self, Path(ckpt_dir) / f"ckpt_{tag}_t{tick + 1}.pkl",
                    meta={"tag": tag, "tick": tick + 1,
                          "seed": self.cfg.seed,
                          "alive": len(self.population.individuals)})
                append_run_log(
                    ckpt_dir, tag,
                    f"tick={tick + 1} alive={len(self.population.individuals)} "
                    f"born={len(self.log.births)} died={len(self.log.deaths)} "
                    f"ckpt={path.name} ts={time.strftime('%H:%M:%S')}")
                if verbose_ckpt:
                    print(f"[ckpt] {path}")
        return self.log


__all__ = [
    "METABOLISM_SCALE", "HABITAT_PENALTY", "SIGNAL_DECAY", "SEASON_AMP",
    "JITTER", "SHARE_PAYLOAD_SCALE", "THREAT_MEMORY_WEIGHT",
    "INIT_POP", "INIT_ENERGY",
    "BioMVPConfig", "BiosphereSimulator",
]
