"""biosphere.metrics：轨迹 → 行为标签 / 谱系树 / 交互网络 → JSON（W0-B）。

通俗理解
--------
这个模块是生物圈的"观测台"：模拟器每 tick 喂它三样东西——

1. **帧**（全部存活个体的 uid/格/能量 + 最近邻距离）；
2. **事件**（resolve 的意图裁定流水）；
3. **生卒**（繁殖/死亡）。

它据此产出预注册判据要用的全部读数：

- **行为标签**：觅食 / 逃避 / 追击 / 结群 / 信号（规则与阈值见
  :class:`BehaviorLabeler` 头部常量，全部显式预注册，禁隐式调参）；
- **谱系树**：父子边 + 出生/死亡 tick + 世代；
- **交互网络**：攻击/分享有向边（带计数）+ 信号活跃度；
- **JSON**：定键、sort_keys ⇒ 同 seed 两次输出逐位一致。

性能口径：最近邻距离用"占用格 + 4 邻域"扫描（O(N×5)），不搞 O(N²)；
标签器在构造时**一次建好**（tick→uid→格/最近邻、uid→事件）索引，
避免每标签一遍全表扫描。
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from biosphere.action import ACTION_BASE_COST, ACTION_NAMES, ActionKind
from biosphere.groups import CHEATER_MARGIN, group_expression_variance
from biosphere.resolve import ResolveEvent

# ── 行为标签阈值（预注册；改值 = 判据变更，须重走预注册） ──────────────────

#: 觅食：窗口内进食总增益下限（含 EAT 成本还原）
FORAGE_GAIN_MIN: float = 1.0
#: 觅食：窗口内平均步长下限（格/tick）
FORAGE_SPEED_MIN: float = 0.20
#: 逃避：受击后 3 tick 与攻击者距离增量下限（格）
FLEE_DIST_GAIN_MIN: float = 0.5
#: 追击：攻击前 3 tick 与目标距离减量下限（格）
CHASE_DIST_DROP_MIN: float = 0.5
#: 结群：窗口内最近邻距离均值上限（格）
GROUP_NN_MAX: float = 1.5
#: 结群：窗口内有效样本下限（有邻居的帧数）
GROUP_SAMPLES_MIN: int = 30
#: 信号：窗口内信号意图数下限
SIGNAL_COUNT_MIN: int = 1
#: 分块序列块长（tick）
BLOCK_TICKS: int = 100
#: 交互网络输出 top-K
TOP_EDGES: int = 20

#: 行为标签名（JSON 键；顺序固定）
BEHAVIOR_LABELS: Tuple[str, ...] = ("foraging", "fleeing", "chasing",
                                    "grouping", "signaling")

#: 攻击「命中」状态集合（致死 ok / 致伤 attack_survived）——逃避与追击
#: 两类标签共用；B6 修复前逃避只认 "ok"（致死受害者无存活窗口 ⇒ 恒 0）。
_ATTACK_HIT_STATUSES = ("ok", "attack_survived")

#: EAT 成本（还原进食净增益用；与 action.ACTION_BASE_COST 同源）
_EAT_COST: float = ACTION_BASE_COST[int(ActionKind.EAT)]

SCHEMA = "biosphere-metrics-v0.1"


# ── 观测日志（记录器） ────────────────────────────────────────────────────


@dataclass
class Frame:
    """一个 tick 的全种群快照。"""

    tick: int
    uids: np.ndarray      # int64 (k,)
    cells: np.ndarray     # int64 (k,)
    energies: np.ndarray  # float64 (k,)
    nn_dist: np.ndarray   # float64 (k,)，最近邻图距（0 同格 / 1 邻格 / 2.5 远）
    strengths: np.ndarray = None  # type: ignore[assignment]  # C3 v2 (k,)
    emitting: np.ndarray = None   # type: ignore[assignment]  # C3 v2 (k,) 0/1
    defense: np.ndarray = None    # type: ignore[assignment]  # B16 (k,)
    attack_drive: np.ndarray = None  # type: ignore[assignment]  # B16 (k,)


@dataclass
class GroupFrame:
    """一个 tick 的团结构快照（W4；flows 只留最终帧的个体公共池流水）。

    运动学（B11 补充指令 3：团速 = 质心位移 + 形态变化，**不用单细胞
    速度解释团速**）：焦点团 = 该 tick 的最大多体团（平手取最低 gid；
    团身份不跨 tick 稳定 ⇒ 质心序列是"最大团质心"序列，文档已注）。
    """

    tick: int
    n_groups: int
    n_multi: int                 # size ≥ 2 的团数
    n_in_multi: int              # 团内个体总数
    max_size: int
    mean_size_multi: float       # 多体团的平均大小
    within_var: float            # 团内表达方差（DIFF_PARAMS 均值）
    size_histogram: Dict[int, int]   # 大小 → 团数
    focal_size: int = 0          # 焦点团大小（0 = 无多体团）
    focal_centroid_row: float = 0.0
    focal_centroid_col: float = 0.0   # 弧度（环绕感知）
    focal_radius: float = 0.0    # 焦点团形态：成员到质心平均距
    mean_radius_multi: float = 0.0   # 全部多体团半径均值


@dataclass
class ChainFrame:
    """一个 tick 的**粘附链团**快照（W4b 主张镜头）+ 粘附基因读数。

    邻接团快照另存（GroupFrame，对照镜头）。运动学：焦点团 = 该 tick
    最大多体链团（团速 = 质心位移，不用单细胞速度解释）。
    """

    tick: int
    n_chain_groups: int
    n_multi_chain: int           # size ≥ 2 的链团数
    in_multi_chain: int          # 链团内个体总数
    max_chain_size: int
    mean_chain_size: float
    size_histogram: Dict[int, int]
    # 粘附基因（原始基因值；槽位 PARAM_SLOTS["adhesion"]）与表达
    adhesion_gene_mean: float
    adhesion_gene_var: float
    adhesion_expr_mean: float
    sticky_fraction: float
    # 运动学（焦点链团）
    focal_size: int
    focal_centroid_row: float
    focal_centroid_col: float
    focal_radius: float
    # 组繁殖构成普查（按共享桶）
    n_multi_high_share: int
    n_multi_low_share: int
    n_eligible_high: int
    n_eligible_low: int
    n_repro_high: int
    n_repro_low: int


@dataclass
class SimLog:
    """模拟全程观测日志（metrics 的唯一数据源）。"""

    frames: List[Frame] = field(default_factory=list)
    events: List[ResolveEvent] = field(default_factory=list)
    births: List[Tuple[int, int, int]] = field(default_factory=list)   # (父,子,格)
    deaths: List[Tuple[int, str, int]] = field(default_factory=list)   # (uid,因,tick)
    group_frames: List[GroupFrame] = field(default_factory=list)
    chain_frames: List[ChainFrame] = field(default_factory=list)
    #: W4b 组繁殖事件构成（逐事件：团大小/共享注入比/生殖系/子代）
    group_repro_events: List[Dict] = field(default_factory=list)
    #: W4b 组繁殖逐团普查（决策点、扣费前：合格/已繁殖/共享桶）
    group_census: List[Dict] = field(default_factory=list)
    #: 最终帧的个体公共池流水 [(uid, pool_in, pool_out)]（W4 作弊者判据）
    final_flows: List[Tuple[int, float, float]] = field(default_factory=list)
    #: 死亡个体的公共池流水（B12：修作弊者统计的生存偏差）
    dead_flows: List[Tuple[int, float, float]] = field(default_factory=list)
    #: 最终帧的粘附链团成员表（每团 uid 列表；每 tick 覆盖 ⇒ 内存 O(N)）
    final_chain_groups: List[List[int]] = field(default_factory=list)
    energy_paid: float = 0.0
    energy_in: float = 0.0
    endowment_paid: float = 0.0
    #: B19b light 长跑：只关观测累积，不改仿真（默认 True ⇒ 旧行为逐位不变）
    capture_frames: bool = True
    capture_events: bool = True

    def record_frame(self, tick: int, individuals, world, c3=None,
                     badge_layer=None, strengths_arr=None,
                     traits=None) -> None:
        """从种群快照 + 世界邻域算一帧（含最近邻距离）。

        c3/badge_layer 传入时附带 C3 v2 字段（强度、是否在发射可见徽章）；
        ``traits`` = (defense 数组, attack_drive 数组)（B16 军备竞赛开档时传）。
        """
        if not self.capture_frames:
            return
        uids = np.array([ind.uid for ind in individuals], dtype=np.int64)
        cells = np.array([ind.cell for ind in individuals], dtype=np.int64)
        energies = np.array([ind.energy for ind in individuals], dtype=np.float64)
        nn = _nearest_neighbor(cells, world)
        defense = attack = None
        if traits is not None:
            defense = np.asarray(traits[0], dtype=np.float64)
            attack = np.asarray(traits[1], dtype=np.float64)
        strengths = None
        emitting = None
        if c3 is not None and c3.enabled and strengths_arr is not None:
            from biosphere.c3 import visible_badge
            strengths = np.asarray(strengths_arr, dtype=np.float64)
            emitting = np.array(
                [1 if visible_badge(badge_layer[:, ind.cell],
                                    c3.emit_visible_min) >= 0 else 0
                 for ind in individuals], dtype=np.int8)
        self.frames.append(Frame(tick, uids, cells, energies, nn,
                                 strengths, emitting, defense, attack))

    def record_chain_frame(self, tick: int, population, index, controllers,
                           world, events_tick: Optional[List[Dict]] = None,
                           census_tick: Optional[List[Dict]] = None
                           ) -> None:
        """W4b 粘附链团帧（主张镜头）+ 粘附基因/组繁殖构成普查。

        ``events_tick`` = 本 tick 发生的组繁殖事件（用于高/低共享桶计数）。
        """
        if not self.capture_frames:
            return
        from biosphere.groups import (
            ADHESION_STICKY, GROUP_REPRO_THRESH_PER_CAPITA,
            is_sticky, share_injection_ratio,
        )
        from biosphere.genome import PARAM_SLOTS
        from biosphere.cell import adhesion_expression

        by_uid = {ind.uid: ind for ind in population.individuals}
        sizes = [len(index.members[g]) for g in sorted(index.members)]
        hist: Dict[int, int] = {}
        for s in sizes:
            hist[s] = hist.get(s, 0) + 1
        multi_gids = [g for g in sorted(index.members)
                      if len(index.members[g]) >= 2]
        multi = [len(index.members[g]) for g in multi_gids]

        # 粘附基因（原始值）与表达值（受分化调制）
        slot = PARAM_SLOTS["adhesion"]
        genes = [float(ind.genome.genes[slot]) for ind in
                 population.individuals]
        exprs = [adhesion_expression(controllers[ind.uid].adhesion,
                                     ind.diff_state) for ind in
                 population.individuals]
        gmean = math.fsum(genes) / len(genes) if genes else 0.0
        gvar = (math.fsum((g - gmean) ** 2 for g in genes) / len(genes)
                if genes else 0.0)
        sticky_frac = (math.fsum(1.0 for e in exprs if e >= ADHESION_STICKY)
                       / len(exprs)) if exprs else 0.0

        # 组繁殖构成普查：由 reproduce_groups 在**决策点**（扣费前）记的
        # 逐团行填充（事后按余额重算会漏掉净奉献的高共享团）
        rows = census_tick or []
        n_mh = sum(1 for r in rows if r["high_share"])
        n_ml = sum(1 for r in rows if not r["high_share"])
        n_eh = sum(1 for r in rows if r["high_share"] and r["eligible"])
        n_el = sum(1 for r in rows if (not r["high_share"]) and r["eligible"])
        n_rh = sum(1 for r in rows if r["high_share"] and r["reproduced"])
        n_rl = sum(1 for r in rows if (not r["high_share"]) and r["reproduced"])

        # 焦点链团运动学（质心 + 形态半径；环绕感知）
        fcr = fcc = frad = 0.0
        focal_size = 0
        if world is not None and multi_gids:
            focal_gid = max(multi_gids,
                            key=lambda g: (len(index.members[g]), -g))
            uids = index.members[focal_gid]
            rows = [float(by_uid[u].cell // world.cols) for u in uids]
            angs = [2.0 * math.pi * (by_uid[u].cell % world.cols) / world.cols
                    for u in uids]
            fcr = math.fsum(rows) / len(rows)
            fcc = math.atan2(math.fsum(math.sin(a) for a in angs),
                             math.fsum(math.cos(a) for a in angs))
            frad = math.fsum(
                math.hypot(by_uid[u].cell // world.cols - fcr,
                           _ang_delta(2.0 * math.pi * (by_uid[u].cell
                                                        % world.cols)
                                      / world.cols, fcc))
                for u in uids) / len(uids)
            focal_size = len(uids)

        self.chain_frames.append(ChainFrame(
            tick=tick, n_chain_groups=len(sizes), n_multi_chain=len(multi),
            in_multi_chain=sum(multi),
            max_chain_size=max(sizes) if sizes else 0,
            mean_chain_size=(math.fsum(multi) / len(multi)) if multi else 0.0,
            size_histogram=hist, adhesion_gene_mean=gmean,
            adhesion_gene_var=gvar,
            adhesion_expr_mean=(math.fsum(exprs) / len(exprs)
                                if exprs else 0.0),
            sticky_fraction=sticky_frac, focal_size=focal_size,
            focal_centroid_row=fcr, focal_centroid_col=fcc,
            focal_radius=frad, n_multi_high_share=n_mh,
            n_multi_low_share=n_ml, n_eligible_high=n_eh, n_eligible_low=n_el,
            n_repro_high=n_rh, n_repro_low=n_rl,
        ))
        # 最终帧的链团成员表（每 tick 覆盖 ⇒ 内存 O(N)）
        self.final_chain_groups = [list(index.members[g]) for g in multi_gids]

    def record_group_frame(self, tick: int, population, index,
                           controllers, world=None) -> None:
        """W4 团结构帧（simulator 在 W4 开启时每 tick 调一次）。

        world 传入时还算团运动学（焦点团质心 + 形态半径）。
        """
        if not self.capture_frames:
            return
        sizes = [len(index.members[g]) for g in sorted(index.members)]
        hist: Dict[int, int] = {}
        for s in sizes:
            hist[s] = hist.get(s, 0) + 1
        multi = [s for s in sizes if s >= 2]
        wv = group_expression_variance(index, controllers)

        by_uid = {ind.uid: ind for ind in population.individuals}

        def _centroid_radius(uids: List[int]) -> Tuple[float, float, float]:
            """环绕感知质心（row 均值 + col 角度均值）与平均半径。"""
            rows = []
            angs = []
            for u in uids:
                r, c = by_uid[u].cell // world.cols, by_uid[u].cell % world.cols
                rows.append(float(r))
                angs.append(2.0 * math.pi * c / world.cols)
            cr = math.fsum(rows) / len(rows)
            sc = math.fsum(math.sin(a) for a in angs)
            cc = math.fsum(math.cos(a) for a in angs)
            ang = math.atan2(sc, cc)
            rad = math.fsum(
                math.hypot(by_uid[u].cell // world.cols - cr,
                           _ang_delta(2.0 * math.pi * (by_uid[u].cell
                                                        % world.cols)
                                      / world.cols, ang))
                for u in uids) / len(uids)
            return cr, ang, rad

        focal_size = 0
        fcr = fcc = frad = 0.0
        radii: List[float] = []
        if world is not None:
            multi_gids = [g for g in sorted(index.members)
                          if len(index.members[g]) >= 2]
            for g in multi_gids:
                _cr, _ang, rad = _centroid_radius(index.members[g])
                radii.append(rad)
            if multi_gids:
                # 焦点团 = 最大多体团（平手取最低 gid）
                focal_gid = max(multi_gids,
                                key=lambda g: (len(index.members[g]), -g))
                fcr, fcc, frad = _centroid_radius(index.members[focal_gid])
                focal_size = len(index.members[focal_gid])

        self.group_frames.append(GroupFrame(
            tick=tick, n_groups=len(sizes), n_multi=len(multi),
            n_in_multi=sum(multi), max_size=max(sizes) if sizes else 0,
            mean_size_multi=(math.fsum(multi) / len(multi)) if multi else 0.0,
            within_var=wv["mean"], size_histogram=hist,
            focal_size=focal_size, focal_centroid_row=fcr,
            focal_centroid_col=fcc, focal_radius=frad,
            mean_radius_multi=(math.fsum(radii) / len(radii))
            if radii else 0.0,
        ))
        # 个体公共池流水（每 tick 覆盖 ⇒ 内存 O(N)，留最终值）
        self.final_flows = [(ind.uid, ind.pool_in, ind.pool_out)
                            for ind in population.individuals]

    def record_report(self, report) -> None:
        if self.capture_events:
            self.events.extend(report.events)
        for parent, child, cell in report.births:
            self.births.append((parent, child, cell))
        for uid, cause in report.deaths:
            self.deaths.append((uid, cause, report.tick))
        self.energy_paid += report.energy_paid
        self.energy_in += report.energy_in
        self.endowment_paid += report.endowment_paid
        for _gid, _parent, child, _cell in getattr(
                report, "group_births", []):
            self.births.append((_parent, child, _cell))


def _nearest_neighbor(cells: np.ndarray, world) -> np.ndarray:
    """最近邻图距：0 = 同格有他个，1 = 邻格有个，2.5 = 远（哨兵）。"""
    k = cells.shape[0]
    out = np.full(k, 2.5, dtype=np.float64)
    if k == 0:
        return out
    occ: Dict[int, List[int]] = {}
    for i, c in enumerate(cells):
        occ.setdefault(int(c), []).append(i)
    for i, c in enumerate(cells):
        c = int(c)
        if len(occ.get(c, ())) > 1:
            out[i] = 0.0
            continue
        for nb in world.neighbors[c]:
            if occ.get(int(nb)):
                out[i] = 1.0
                break
    return out


# ── 行为标签器（构造时建索引；阈值 = 可配置，默认 = 预注册常量） ──────────


@dataclass
class BehaviorThresholds:
    """行为标签阈值（H1 去写死：进 config，默认值 = 历史预注册常量）。

    ⚠️ 默认值与 B6–B13 各轮读数完全一致 ⇒ **标签判定逻辑不变**，
    只把"数字"从模块常量搬进 config（可跑三档敏感性）。
    """

    forage_gain_min: float = FORAGE_GAIN_MIN
    forage_speed_min: float = FORAGE_SPEED_MIN
    flee_dist_gain_min: float = FLEE_DIST_GAIN_MIN
    chase_dist_drop_min: float = CHASE_DIST_DROP_MIN
    group_nn_max: float = GROUP_NN_MAX
    window_ticks: int = 3


class BehaviorLabeler:
    """规则式行为分类。

    标签定义（预注册，全部阈值显式）：

    - **foraging 觅食**：窗口内进食净增益 ≥ FORAGE_GAIN_MIN 且平均步长
      ≥ FORAGE_SPEED_MIN；
    - **fleeing 逃避**：窗口内被攻击 ≥ 1 次，且受击后 3 tick 内与攻击者
      距离增量 ≥ FLEE_DIST_GAIN_MIN；
    - **chasing 追击**：窗口内发出 ≥ 1 次成功/命中攻击，且攻击前 3 tick
      内与目标距离减量 ≥ CHASE_DIST_DROP_MIN；
    - **grouping 结群**：窗口内最近邻距离均值 ≤ GROUP_NN_MAX（有效样本
      ≥ GROUP_SAMPLES_MIN）；
    - **signaling 信号**：窗口内信号意图数 ≥ SIGNAL_COUNT_MIN。

    距离 = 网格距（行差 + 列向环绕差），指标距。
    """

    def __init__(self, log: SimLog,
                 thresholds: Optional[BehaviorThresholds] = None) -> None:
        self.log = log
        self.th = thresholds or BehaviorThresholds()
        # tick → (uid → cell / nn / energy)（一次建好）
        self._cell: Dict[int, Dict[int, int]] = {}
        self._nn: Dict[int, Dict[int, float]] = {}
        self._energy: Dict[int, Dict[int, float]] = {}
        for f in log.frames:
            cm: Dict[int, int] = {}
            nm: Dict[int, float] = {}
            em: Dict[int, float] = {}
            for i in range(f.uids.shape[0]):
                uid = int(f.uids[i])
                cm[uid] = int(f.cells[i])
                nm[uid] = float(f.nn_dist[i])
                em[uid] = float(f.energies[i])
            self._cell[f.tick] = cm
            self._nn[f.tick] = nm
            self._energy[f.tick] = em
        self._ticks = sorted(self._cell)
        # uid → 该个体出现过的 tick 列表 / 事件列表
        self._ticks_of: Dict[int, List[int]] = {}
        for t in self._ticks:
            for uid in self._cell[t]:
                self._ticks_of.setdefault(uid, []).append(t)
        self._ev_actor: Dict[int, List[ResolveEvent]] = {}
        self._ev_target: Dict[int, List[ResolveEvent]] = {}
        for e in log.events:
            self._ev_actor.setdefault(e.actor_id, []).append(e)
            if e.target_id >= 0:
                self._ev_target.setdefault(e.target_id, []).append(e)

    # ── 查询原语 ──

    def cell_at(self, tick: int, uid: int) -> Optional[int]:
        m = self._cell.get(tick)
        return None if m is None else m.get(uid)

    @staticmethod
    def _wrapped_dist(world, c1: int, c2: int) -> float:
        r1, cc1 = int(c1) // world.cols, int(c1) % world.cols
        r2, cc2 = int(c2) // world.cols, int(c2) % world.cols
        dcol = min(abs(cc1 - cc2), world.cols - abs(cc1 - cc2))
        return math.hypot(float(abs(r1 - r2)), float(dcol))

    def _mean_speed(self, uid: int, t0: Optional[int],
                    t1: Optional[int]) -> float:
        """窗口内平均步长（相邻帧换格 = 1 步；帧不连续不计）。"""
        total, n = 0.0, 0
        prev_t: Optional[int] = None
        for t in self._ticks_of.get(uid, ()):
            if t0 is not None and t < t0:
                continue
            if t1 is not None and t > t1:
                continue
            if prev_t is not None and t == prev_t + 1:
                if self._cell[t][uid] != self._cell[prev_t][uid]:
                    total += 1.0
                    n += 1
                else:
                    n += 1  # 原地也计入样本（速度可以为 0）
            prev_t = t
        return total / n if n else 0.0

    def _nn_values(self, uid: int, t0: Optional[int],
                   t1: Optional[int]) -> List[float]:
        out: List[float] = []
        for t in self._ticks_of.get(uid, ()):
            if t0 is not None and t < t0:
                continue
            if t1 is not None and t > t1:
                continue
            out.append(self._nn[t][uid])
        return out

    # ── 标签 ──

    def label_individual(self, uid: int, world,
                         t0: Optional[int] = None,
                         t1: Optional[int] = None) -> Dict[str, bool]:
        """单个个体在 [t0, t1] 窗口内的五类标签（None = 不限）。"""
        evs = [e for e in self._ev_actor.get(uid, ())
               if (t0 is None or e.tick >= t0) and (t1 is None or e.tick <= t1)]

        # 觅食
        eat_gain = math.fsum(
            e.actor_energy_delta + _EAT_COST
            for e in evs if e.kind == "eat" and e.status == "ok"
        )
        speed = self._mean_speed(uid, t0, t1)
        foraging = (eat_gain >= self.th.forage_gain_min
                     and speed >= self.th.forage_speed_min)

        # 逃避（受击后 3 tick 距离增量）
        # 🔴 B6 修复（2026-10-18）：原过滤器只认 status=="ok"（= 致死攻击），
        # 而致死受害者 3 tick 后必然无帧 ⇒ 窗口恒空 ⇒ fleeing 结构性 = 0
        # （实测致死率仅 5–12%，幸存窗口 395–1485 个/run 全部被漏统计）。
        # 「命中」= 致死或致伤，两者都算受击。
        fleeing = False
        for e in self._ev_target.get(uid, ()):
            if e.kind != "attack" or e.status not in _ATTACK_HIT_STATUSES:
                continue
            if t0 is not None and e.tick < t0:
                continue
            if t1 is not None and e.tick > t1:
                continue
            a_cell = self.cell_at(e.tick, e.actor_id)
            v0 = self.cell_at(e.tick, uid)
            v3 = self.cell_at(e.tick + self.th.window_ticks, uid)
            if a_cell is None or v0 is None or v3 is None:
                continue
            if (self._wrapped_dist(world, v3, a_cell)
                    - self._wrapped_dist(world, v0, a_cell)
                    ) >= self.th.flee_dist_gain_min:
                fleeing = True
                break

        # 追击（攻击前 3 tick 距离减量）
        chasing = False
        for e in evs:
            if e.kind != "attack" or e.target_id < 0:
                continue
            if e.status not in _ATTACK_HIT_STATUSES:
                continue
            a0 = self.cell_at(e.tick, uid)
            a3 = self.cell_at(e.tick - self.th.window_ticks, uid)
            b = self.cell_at(e.tick, e.target_id)
            if a0 is None or a3 is None or b is None:
                continue
            if (self._wrapped_dist(world, a3, b)
                    - self._wrapped_dist(world, a0, b)
                    ) >= self.th.chase_dist_drop_min:
                chasing = True
                break

        # 结群（窗口内最近邻距离均值）
        nn_vals = [v for v in self._nn_values(uid, t0, t1) if v <= 2.0]
        grouping = (len(nn_vals) >= GROUP_SAMPLES_MIN
                    and math.fsum(nn_vals) / len(nn_vals) <= self.th.group_nn_max)

        # 信号
        n_signal = sum(1 for e in evs if e.kind == "signal")
        signaling = n_signal >= SIGNAL_COUNT_MIN

        return {
            "foraging": bool(foraging),
            "fleeing": bool(fleeing),
            "chasing": bool(chasing),
            "grouping": bool(grouping),
            "signaling": bool(signaling),
        }

    def label_all(self, world) -> Dict[int, Dict[str, bool]]:
        return {uid: self.label_individual(uid, world)
                for uid in sorted(self._ticks_of)}

    def block_series(self, world, block: int = BLOCK_TICKS
                     ) -> Dict[str, List[int]]:
        """分块标签计数序列（块内窗口 = 该块 tick 区间）。"""
        t_max = self._ticks[-1] if self._ticks else 0
        n_blocks = t_max // block + 1
        series: Dict[str, List[int]] = {lab: [0] * n_blocks
                                        for lab in BEHAVIOR_LABELS}
        for uid in sorted(self._ticks_of):
            for b in range(n_blocks):
                t0, t1 = b * block, (b + 1) * block - 1
                for lab, on in self.label_individual(uid, world, t0, t1).items():
                    if on:
                        series[lab][b] += 1
        return series


# ── 攻击窗口分布仪器（B6/B10：fleeing/chasing 诊断的永久读数） ───────────


def _hit_attack_events(log: SimLog):
    """命中攻击事件（致死或致伤）的 (tick, actor, target) 三元组。"""
    for e in log.events:
        if e.kind == "attack" and e.target_id >= 0 \
                and e.status in _ATTACK_HIT_STATUSES:
            yield e


def _cell_at_frame(log: SimLog, tick: int, uid: int) -> Optional[int]:
    for f in log.frames:
        if f.tick == tick:
            hit = np.nonzero(f.uids == uid)[0]
            return int(f.cells[int(hit[0])]) if hit.size else None
    return None


def flee_gain_distribution(log: SimLog, world) -> Dict:
    """逃避窗口分布：致伤命中后 3 tick 的距离增量（B6 仪器）。

    只统计致伤（attack_survived）——致死受害者无存活帧（窗口为空），
    这正是 B6 标签 bug 的证据读数。返回分位数 + 达阈值比例。
    """
    gains: List[float] = []
    n_lethal = n_wounded = n_window = 0
    for e in log.events:
        if e.kind != "attack" or e.target_id < 0:
            continue
        if e.status == "ok":
            n_lethal += 1
            continue
        if e.status != "attack_survived":
            continue
        n_wounded += 1
        f3 = _frame_at(log, e.tick + 3)
        f0 = _frame_at(log, e.tick)
        if f3 is None or f0 is None:
            continue
        u3 = {int(u): i for i, u in enumerate(f3.uids)}
        u0 = {int(u): i for i, u in enumerate(f0.uids)}
        if e.target_id not in u3 or e.target_id not in u0:
            continue
        v0 = int(f0.cells[u0[e.target_id]])
        v3 = int(f3.cells[u3[e.target_id]])
        a = int(f0.cells[u0[e.actor_id]]) if e.actor_id in u0 else None
        if a is None:
            continue
        n_window += 1
        gains.append(world.grid_distance(v3, a) - world.grid_distance(v0, a))
    arr = np.array(gains, dtype=np.float64) if gains else np.array([0.0])
    return {
        "n_lethal": n_lethal,
        "n_wounded": n_wounded,
        "n_window_available": n_window,
        "gain_min": float(arr.min()),
        "gain_p25": float(np.percentile(arr, 25)),
        "gain_median": float(np.median(arr)),
        "gain_p75": float(np.percentile(arr, 75)),
        "gain_max": float(arr.max()),
        "frac_gain_ge_0p5": float(np.mean(arr >= 0.5)),
    }


def _ang_delta(a: float, b: float) -> float:
    """两个角度（弧度）的最小差（环绕感知）。"""
    d = (a - b) % (2.0 * math.pi)
    if d > math.pi:
        d -= 2.0 * math.pi
    return d


def _frame_at(log: SimLog, tick: int) -> Optional[Frame]:
    for f in log.frames:
        if f.tick == tick:
            return f
    return None


def chase_gain_distribution(log: SimLog, world) -> Dict:
    """追击窗口分布（B10：对称于 flee_gain_distribution）。

    对每个命中攻击算 drop = dist(攻击者@t-3, 目标@t) −
    dist(攻击者@t, 目标@t)（攻击前 3 tick 的接近量；网格距整数 ⇒
    drop ≥ 0.5 ⟺ ≥ 1 格），并按"接近由谁完成"分桶：

    - already_adjacent@t-3：3 tick 前双方已贴贴 ⇒ 无追赶空间
      （机会主义出手，"猎物送上门"）；
    - target_closed：目标走到攻击者旧位（送上门）；
    - attacker_closed：攻击者主动接近 ≥1 格（真追击，标签来源）；
    - other：其余（含攻击者后撤）。

    chasing 计数低时，本分布区分"判据漏计"与"行为上无追击驱动"。
    """
    drops: List[float] = []
    n_adj = n_target = n_attacker = n_other = 0
    n_window_missing = 0
    n_hits = 0
    for e in _hit_attack_events(log):
        n_hits += 1
        f3 = _frame_at(log, e.tick - 3)
        f0 = _frame_at(log, e.tick)
        if f3 is None or f0 is None:
            n_window_missing += 1
            continue
        a3 = _cell_at_frame(log, e.tick - 3, e.actor_id)
        a0 = _cell_at_frame(log, e.tick, e.actor_id)
        b0 = _cell_at_frame(log, e.tick, e.target_id)
        b3 = _cell_at_frame(log, e.tick - 3, e.target_id)
        if a3 is None or a0 is None or b0 is None or b3 is None:
            n_window_missing += 1
            continue
        d0 = world.grid_distance(a0, b0)
        d3 = world.grid_distance(a3, b0)
        d_init = world.grid_distance(a3, b3)
        drops.append(d3 - d0)
        if d_init <= 1.0:
            n_adj += 1
        elif d3 <= 1.0:
            n_target += 1
        elif d3 - d0 >= 0.5:
            n_attacker += 1
        else:
            n_other += 1
    arr = np.array(drops, dtype=np.float64) if drops else np.array([0.0])
    denom = max(1, len(drops))
    return {
        "n_hits": n_hits,
        "n_window_available": len(drops),
        "n_window_missing": n_window_missing,
        "drop_min": float(arr.min()),
        "drop_median": float(np.median(arr)),
        "drop_max": float(arr.max()),
        "frac_drop_ge_0p5": float(np.mean(arr >= 0.5)),
        "frac_already_adjacent_at_t3": n_adj / denom,
        "frac_target_closed": n_target / denom,
        "frac_attacker_closed": n_attacker / denom,
        "frac_other": n_other / denom,
    }


# ── 谱系与交互网络 ────────────────────────────────────────────────────────


def lineage_json(log: SimLog) -> Dict:
    """谱系树 → JSON（节点 + 个体数 + 根系数 + 最大世代）。"""
    parent_of: Dict[int, int] = {}
    for parent, child, _cell in log.births:
        parent_of[child] = parent
    death_of: Dict[int, int] = {}
    cause_of: Dict[int, str] = {}
    for uid, cause, tick in log.deaths:
        death_of[uid] = tick
        cause_of[uid] = cause
    first_seen: Dict[int, int] = {}
    for f in log.frames:
        for uid in f.uids.tolist():
            uid = int(uid)
            if uid not in first_seen:
                first_seen[uid] = f.tick
    all_uids = set(parent_of) | set(death_of) | set(first_seen)
    all_uids.update(parent_of.values())

    gen_of: Dict[int, int] = {}

    def generation(uid: int) -> int:
        chain: List[int] = []
        cur = uid
        while cur != -1 and cur not in gen_of:
            chain.append(cur)
            cur = parent_of.get(cur, -1)
        base = gen_of[cur] if (cur != -1 and cur in gen_of) else -1
        for u in reversed(chain):
            base += 1
            gen_of[u] = base
        return gen_of[uid]

    nodes: Dict[str, Dict] = {}
    for uid in sorted(all_uids):
        nodes[str(uid)] = {
            "parent": parent_of.get(uid, -1),
            "birth_tick": first_seen.get(uid, -1),
            "death_tick": death_of.get(uid, -1),
            "death_cause": cause_of.get(uid, ""),
            "generation": generation(uid),
            "alive": uid not in death_of,
        }
    roots = sum(1 for u in all_uids if parent_of.get(u, -1) == -1)
    max_gen = max((v["generation"] for v in nodes.values()), default=0)
    return {"nodes": nodes, "n_individuals": len(nodes),
            "n_roots": roots, "max_generation": max_gen}


def interaction_json(log: SimLog, top_k: int = TOP_EDGES) -> Dict:
    """交互网络 → JSON（攻击/分享有向边 top-K + 信号活跃度 top-K）。"""
    attack: Dict[Tuple[int, int], int] = {}
    share: Dict[Tuple[int, int], int] = {}
    signal: Dict[int, int] = {}
    for e in log.events:
        if e.kind == "attack" and e.status == "ok" and e.target_id >= 0:
            key = (e.actor_id, e.target_id)
            attack[key] = attack.get(key, 0) + 1
        elif e.kind == "share" and e.status == "ok" and e.target_id >= 0:
            key = (e.actor_id, e.target_id)
            share[key] = share.get(key, 0) + 1
        elif e.kind == "signal" and e.status == "signal_ok":
            signal[e.actor_id] = signal.get(e.actor_id, 0) + 1

    def _top(d: Dict[Tuple[int, int], int]) -> List[List[int]]:
        return [[a, b, c] for (a, b), c in
                sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]]

    return {
        "attack_edges": _top(attack),
        "share_edges": _top(share),
        "signal_counts": [[u, c] for u, c in
                          sorted(signal.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]],
        "n_attack_pairs": len(attack),
        "n_share_pairs": len(share),
        "n_signalers": len(signal),
    }


# ── W4 团度量 ────────────────────────────────────────────────────────────


def adhesion_chain_json(log: SimLog, world=None) -> Dict:
    """**粘附链团**度量（W4b 主张镜头）：团大小分布 / 粘附基因频率轨迹 /
    组繁殖构成 / 作弊者 / 团运动学。

    🔴 所有"多细胞"主张只看本段；邻接团（密度混淆）见
    ``adjacency_control``。W4 未开时返回零化结构。
    """
    empty = {
        "enabled": False, "n_chain_groups_final": 0, "n_multi_chain_final": 0,
        "in_multi_chain_final": 0, "max_chain_size_final": 0,
        "mean_chain_size_final": 0.0, "size_histogram_final": {},
        "chain_ratio_series": [], "chain_ratio_mean": 0.0,
        "adhesion_trajectory": {}, "kinematics": {},
        "repro_composition": {}, "cheaters": {},
        "children_survival_by_bucket": {},
    }
    if not log.chain_frames:
        return empty

    frames = log.chain_frames
    last = frames[-1]

    # 粘附基因频率轨迹（起始 / 中点 / 终点 + 全程均值）
    def _snap(i: int) -> Dict[str, float]:
        cf = frames[i]
        return {"tick": cf.tick,
                "adhesion_gene_mean": cf.adhesion_gene_mean,
                "adhesion_gene_var": cf.adhesion_gene_var,
                "adhesion_expr_mean": cf.adhesion_expr_mean,
                "sticky_fraction": cf.sticky_fraction,
                "mean_chain_size": cf.mean_chain_size,
                "chain_ratio": (cf.in_multi_chain / len(log.frames[i].uids))
                if i < len(log.frames) and len(log.frames[i].uids)
                else 0.0}

    mid = len(frames) // 2
    traj = {"start": _snap(0), "mid": _snap(mid), "end": _snap(len(frames) - 1)}
    traj["whole_run_mean"] = {
        "adhesion_gene_mean": math.fsum(
            f.adhesion_gene_mean for f in frames) / len(frames),
        "adhesion_gene_var": math.fsum(
            f.adhesion_gene_var for f in frames) / len(frames),
        "adhesion_expr_mean": math.fsum(
            f.adhesion_expr_mean for f in frames) / len(frames),
        "sticky_fraction": math.fsum(
            f.sticky_fraction for f in frames) / len(frames),
    }

    # 粘附链团比例（时间平均）：在多体链团中的个体占比
    ratios: List[float] = []
    for i, cf in enumerate(frames):
        if i < len(log.frames):
            n_alive = len(log.frames[i].uids)
            if n_alive:
                ratios.append(cf.in_multi_chain / n_alive)

    # 团运动学（焦点链团质心位移 + 形态）
    focal = [f for f in frames if f.focal_size >= 2]
    speeds: List[float] = []
    for a, b in zip(focal, focal[1:]):
        dr = b.focal_centroid_row - a.focal_centroid_row
        dc = (_ang_delta(b.focal_centroid_col, a.focal_centroid_col)
              * (world.cols / (2.0 * math.pi))) if world is not None else 0.0
        speeds.append(math.hypot(dr, dc))

    # 组繁殖构成（高共享 vs 低共享桶）
    events = log.group_repro_events
    n_hi = sum(1 for e in events if e["high_share"])
    n_lo = len(events) - n_hi
    # 合格团·tick 计数取自决策点普查（group_census；扣费前）
    elig_hi = sum(1 for r in log.group_census
                  if r["high_share"] and r["eligible"])
    elig_lo = sum(1 for r in log.group_census
                  if (not r["high_share"]) and r["eligible"])
    rate_hi = n_hi / elig_hi if elig_hi else 0.0
    rate_lo = n_lo / elig_lo if elig_lo else 0.0
    repro = {
        "n_events": len(events),
        "n_high_share_events": n_hi,
        "n_low_share_events": n_lo,
        "eligible_group_ticks_high": elig_hi,
        "eligible_group_ticks_low": elig_lo,
        "repro_per_eligible_group_tick_high": rate_hi,
        "repro_per_eligible_group_tick_low": rate_lo,
        "rate_diff_high_minus_low": rate_hi - rate_lo,
        "germ_source": "团内最高能量者（平手取最低 uid）",
        "mean_share_drive_at_repro": (
            math.fsum(e["mean_share_drive"] for e in events) / len(events)
            if events else 0.0),
        "mean_ratio_net_at_repro": (
            math.fsum(e["ratio_net"] for e in events) / len(events)
            if events else 0.0),
    }

    # 作弊者（修生存偏差：终存活者 + 已死者流水全算）
    from biosphere.groups import CHEATER_MARGIN
    all_flows = list(log.final_flows) + list(log.dead_flows)
    members = [(u, i, o) for u, i, o in all_flows if (i > 0.0 or o > 0.0)]
    cheater_uids = {u for u, i, o in members
                    if (i - o) > CHEATER_MARGIN}
    # 每团作弊者占比（最终帧链团）
    per_group: List[Dict] = []
    for gidx, uids in enumerate(log.final_chain_groups):
        alive_in = [u for u in uids]
        n_ch = sum(1 for u in alive_in if u in cheater_uids)
        per_group.append({
            "group_index": gidx, "size": len(alive_in),
            "n_cheaters": n_ch,
            "cheater_fraction": n_ch / len(alive_in) if alive_in else 0.0,
        })
    cheaters = {
        "n_members": len(members),
        "n_cheaters": len(cheater_uids),
        "frequency": (len(cheater_uids) / len(members)) if members else 0.0,
        "margin": CHEATER_MARGIN,
        "n_dead_included": len(log.dead_flows),
        "per_final_group": per_group,
        "per_final_group_mean_fraction": (
            math.fsum(g["cheater_fraction"] for g in per_group) / len(per_group)
            if per_group else 0.0),
    }

    # 子代存活（按共享桶）：组繁殖的子代是否活到终局
    alive_end = set(int(u) for u in log.frames[-1].uids) if log.frames else set()
    surv = {}
    for name, sel in (("high_share", True), ("low_share", False)):
        kids = [e["child_uid"] for e in events if e["high_share"] == sel]
        surv[name] = {"n_children": len(kids),
                      "n_alive_at_end": sum(1 for c in kids
                                            if c in alive_end),
                      "survival_fraction": (sum(1 for c in kids
                                                if c in alive_end)
                                            / len(kids)) if kids else 0.0}

    return {
        "enabled": True,
        "n_chain_groups_final": last.n_chain_groups,
        "n_multi_chain_final": last.n_multi_chain,
        "in_multi_chain_final": last.in_multi_chain,
        "max_chain_size_final": last.max_chain_size,
        "mean_chain_size_final": last.mean_chain_size,
        "size_histogram_final": {str(k): v for k, v in
                                 sorted(last.size_histogram.items())},
        "chain_ratio_series": ratios,
        "chain_ratio_mean": (math.fsum(ratios) / len(ratios)
                             if ratios else 0.0),
        "adhesion_trajectory": traj,
        "kinematics": {
            "focal_centroid_speed_mean": (math.fsum(speeds) / len(speeds))
            if speeds else 0.0,
            "focal_centroid_speed_max": max(speeds) if speeds else 0.0,
            "focal_size_final": focal[-1].focal_size if focal else 0,
            "focal_radius_final": focal[-1].focal_radius if focal else 0.0,
            "note": "焦点团=每 tick 最大多体链团；团速=质心位移(格/tick)，"
                    "不用单细胞速度解释",
        },
        "repro_composition": repro,
        "cheaters": cheaters,
        "children_survival_by_bucket": surv,
    }


def _tercile_bounds(values: List[float]):
    """三分位边界（下 1/3、上 1/3；确定性排序）。"""
    s = sorted(values)
    if not s:
        return 0.0, 0.0
    return s[len(s) // 3], s[(2 * len(s)) // 3]


def c3_json(log: SimLog, world=None) -> Dict:
    """C3 v2 度量段（B14.3）：被攻击率表（强度三分位 × 是否在发射）。

    判据③的数据源：强者带徽章被攻击率↓、弱者带徽章被攻击率↑。
    emitting = 该 tick 目标格的徽章层可见（强度过 EMIT_VISIBLE_MIN）。
    被攻击率 = 攻击事件数 / 个体·tick（同格类曝光）。
    """
    from biosphere.c3 import C3_BADGE_K
    frames = [f for f in log.frames if f.strengths is not None]
    if not frames:
        return {"enabled": False}

    last = frames[-1]
    strengths = [float(v) for v in last.strengths]
    lo_b, hi_b = _tercile_bounds(strengths)

    exposure = {"low": 0, "mid": 0, "high": 0}
    exposure_emit = {"low": 0, "mid": 0, "high": 0}
    # B15.1 oracle 判据③：发射者 vs 非发射者（不分强度；oracle-emitter
    # 把“发射”与“强者”绑定 ⇒ 该差值即因果链是否通的直接读数）
    exp_emit_total = exp_all_total = 0
    att_emit_total = att_all_total = 0
    for f in frames:
        for i in range(f.uids.shape[0]):
            s = float(f.strengths[i])
            key = "low" if s <= lo_b else ("high" if s >= hi_b else "mid")
            exposure[key] += 1
            exp_all_total += 1
            if int(f.emitting[i]):
                exposure_emit[key] += 1
                exp_emit_total += 1
    attacked = {"low": 0, "mid": 0, "high": 0}
    attacked_emit = {"low": 0, "mid": 0, "high": 0}
    per_badge_attacks = {b: 0 for b in range(C3_BADGE_K)}
    for e in log.events:
        if e.kind != "attack" or e.target_id < 0:
            continue
        s = float(e.target_strength)
        key = "low" if s <= lo_b else ("high" if s >= hi_b else "mid")
        attacked[key] += 1
        att_all_total += 1
        if int(e.target_emitting):
            attacked_emit[key] += 1
            att_emit_total += 1
        if 0 <= e.target_badge < C3_BADGE_K:
            per_badge_attacks[e.target_badge] += 1

    def _rate(num, den):
        return (num / den) if den else 0.0

    return {
        "enabled": True,
        "n_final": int(last.uids.shape[0]),
        "strength_tercile_bounds": {"low_max": lo_b, "high_min": hi_b},
        "attacked_rate": {
            "all": {t: _rate(attacked[t], exposure[t])
                    for t in ("low", "mid", "high")},
            "emitting": {t: _rate(attacked_emit[t], exposure_emit[t])
                         for t in ("low", "mid", "high")},
            "n_attacks": attacked,
            "exposure_ticks": exposure,
            "exposure_ticks_emitting": exposure_emit,
        },
        "by_emitting": {
            "attacked_rate_emitting": _rate(att_emit_total, exp_emit_total),
            "attacked_rate_non_emitting": _rate(att_all_total - att_emit_total,
                                                exp_all_total - exp_emit_total),
            "n_attacks_emitting": att_emit_total,
            "n_attacks_non_emitting": att_all_total - att_emit_total,
            "exposure_emitting": exp_emit_total,
            "exposure_non_emitting": exp_all_total - exp_emit_total,
        },
        "attacks_by_target_badge": per_badge_attacks,
        "note": "emitting=该 tick 目标格徽章层可见；被攻击率=攻击事件/个体·tick",
    }


def groups_json(log: SimLog, world=None) -> Dict:
    """W4/W4b 团度量（两镜头）：粘附链团（主张）+ 邻接团（对照）。"""
    return {
        "adhesion_chain": adhesion_chain_json(log, world),
        "adjacency_control": _adjacency_groups_json(log, world),
    }


def _adjacency_groups_json(log: SimLog, world=None) -> Dict:
    """W4 团结构度量：大小分布 / 分化 / 作弊者频率 / 谱系存活 / 团运动学。

    W4 未开时（无 group_frames）返回零化结构（键齐全，JSON 形状稳定）。
    world 传入时团质心速度按格/ tick 计（列向环绕）。
    """
    if not log.group_frames:
        return {"enabled": False, "n_groups_final": 0, "n_multi_final": 0,
                "n_in_multi_final": 0, "max_size_final": 0,
                "mean_size_multi_final": 0.0,
                "size_histogram_final": {}, "within_var_final": 0.0,
                "n_multi_series": [], "within_var_series": [],
                "mean_radius_series": [],
                "kinematics": {
                    "focal_centroid_speed_mean": 0.0,
                    "focal_centroid_speed_max": 0.0,
                    "focal_size_final": 0, "focal_radius_final": 0.0,
                    "mean_radius_final": 0.0, "note":
                        "焦点团=每 tick 最大多体团；团速=质心位移(格/tick)，"
                        "不用单细胞速度解释"},
                "cheaters": {"n_members": 0, "n_cheaters": 0,
                             "frequency": 0.0, "margin": CHEATER_MARGIN},
                "lineage_survival": {"n_roots": 0, "n_surviving_roots": 0,
                                     "fraction": 0.0}}

    last = log.group_frames[-1]
    n_multi_series = [g.n_multi for g in log.group_frames]
    var_series = [g.within_var for g in log.group_frames]
    radius_series = [g.mean_radius_multi for g in log.group_frames]

    # 焦点团质心速度（相邻 tick 质心位移；不用单细胞速度解释团速）
    focal = [g for g in log.group_frames if g.focal_size >= 2]
    speeds: List[float] = []
    for a, b in zip(focal, focal[1:]):
        dr = b.focal_centroid_row - a.focal_centroid_row
        if world is not None:
            dc = _ang_delta(b.focal_centroid_col, a.focal_centroid_col) \
                * (world.cols / (2.0 * math.pi))
        else:
            dc = 0.0
        speeds.append(math.hypot(dr, dc))

    # 作弊者：公共池净蹭超过边际（final_flows = 最终帧个体流水）
    flows = log.final_flows
    members = [(u, i, o) for u, i, o in flows if (i > 0.0 or o > 0.0)]
    cheaters = [u for u, i, o in members if (i - o) > CHEATER_MARGIN]

    # 谱系存活：根（从未作为子代出现）中，有多少根有活着的后代（含自身）
    children_of: Dict[int, List[int]] = {}
    all_uids = set()
    for f in log.frames:
        all_uids.update(int(u) for u in f.uids)
    for parent, child, _cell in log.births:
        children_of.setdefault(parent, []).append(child)
        all_uids.add(child)
    child_set = {c for _p, c, _cell in log.births}
    roots = sorted(u for u in all_uids if u not in child_set)
    alive = set(int(u) for u in log.frames[-1].uids) if log.frames else set()

    def _has_alive_descendant(root: int) -> bool:
        stack = [root]
        seen = set()
        while stack:
            cur = stack.pop()
            if cur in alive:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(children_of.get(cur, ()))
        return False

    surviving = [r for r in roots if _has_alive_descendant(r)]

    return {
        "enabled": True,
        "n_groups_final": last.n_groups,
        "n_multi_final": last.n_multi,
        "n_in_multi_final": last.n_in_multi,
        "max_size_final": last.max_size,
        "mean_size_multi_final": last.mean_size_multi,
        "size_histogram_final": {str(k): v for k, v in
                                 sorted(last.size_histogram.items())},
        "within_var_final": last.within_var,
        "n_multi_series": n_multi_series,
        "within_var_series": var_series,
        "mean_radius_series": radius_series,
        "kinematics": {
            "focal_centroid_speed_mean": (math.fsum(speeds) / len(speeds))
            if speeds else 0.0,
            "focal_centroid_speed_max": max(speeds) if speeds else 0.0,
            "focal_size_final": focal[-1].focal_size if focal else 0,
            "focal_radius_final": focal[-1].focal_radius if focal else 0.0,
            "mean_radius_final": last.mean_radius_multi,
            "note": "焦点团=每 tick 最大多体团；团速=质心位移(格/tick)，"
                    "不用单细胞速度解释",
        },
        "cheaters": {
            "n_members": len(members),
            "n_cheaters": len(cheaters),
            "frequency": (len(cheaters) / len(members)) if members else 0.0,
            "margin": CHEATER_MARGIN,
        },
        "lineage_survival": {
            "n_roots": len(roots),
            "n_surviving_roots": len(surviving),
            "fraction": (len(surviving) / len(roots)) if roots else 0.0,
        },
    }


def _hist(values: List[float], bins: int = 12,
          vmax: Optional[float] = None) -> Dict:
    """等宽直方图（确定性；返回边界 + 计数）。"""
    if not values:
        return {"n": 0, "bin_edges": [], "counts": []}
    hi = float(max(values)) if vmax is None else float(vmax)
    lo = float(min(values))
    if hi <= lo:
        hi = lo + 1.0
    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    counts = [0] * bins
    for v in values:
        idx = int((v - lo) / (hi - lo) * bins)
        counts[min(max(idx, 0), bins - 1)] += 1
    return {"n": len(values), "bin_edges": edges, "counts": counts,
            "min": lo, "max": float(max(values)),
            "mean": math.fsum(values) / len(values)}


def raw_distributions_json(log: SimLog, world) -> Dict:
    """**行为原始分布**（H1 去写死：先给分布，再谈阈值标签）。

    四类原始量的时间序列 + 直方图（**不经过任何标签阈值**）：
    - 距离增量：逃避（受击后 window 距离增量）/ 追击（攻击前 window 距离减量）；
    - 觅食增益：每次 eat 的净能量增益；
    - 邻居密度：最近邻距离（逐帧）+ 个体均值直方图；
    - 信号计数：每 tick 信号意图数 + 个体总量直方图。
    另给逐 tick 动作计数序列（move/eat/attack/share/signal/repro）。
    """
    th = BehaviorThresholds()
    frames_by_tick = {f.tick: f for f in log.frames}

    # 距离增量（原始，逐事件）
    flee_gains: List[float] = []
    chase_drops: List[float] = []
    for e in log.events:
        if e.kind != "attack" or e.target_id < 0:
            continue
        if e.status not in _ATTACK_HIT_STATUSES:
            continue
        f0 = frames_by_tick.get(e.tick)
        if f0 is None:
            continue
        u0 = {int(u): i for i, u in enumerate(f0.uids)}
        if e.target_id not in u0:
            continue
        v0 = int(f0.cells[u0[e.target_id]])
        a_cell = int(f0.cells[u0[e.actor_id]]) if e.actor_id in u0 else None
        f3 = frames_by_tick.get(e.tick + th.window_ticks)
        if f3 is not None:
            u3 = {int(u): i for i, u in enumerate(f3.uids)}
            if e.target_id in u3 and a_cell is not None:
                v3 = int(f3.cells[u3[e.target_id]])
                flee_gains.append(world.grid_distance(v3, a_cell)
                                  - world.grid_distance(v0, a_cell))
        # 追击：攻击者 3 tick 前的位置
        fm = frames_by_tick.get(e.tick - th.window_ticks)
        if fm is not None:
            um = {int(u): i for i, u in enumerate(fm.uids)}
            if e.actor_id in um and e.target_id in u0:
                a3 = int(fm.cells[um[e.actor_id]])
                b = int(f0.cells[u0[e.target_id]])
                chase_drops.append(world.grid_distance(a3, b)
                                   - world.grid_distance(v0, b))

    # 觅食增益（原始，逐事件）
    eat_gains: List[float] = []
    for e in log.events:
        if e.kind == "eat" and e.status == "ok":
            eat_gains.append(float(e.actor_energy_delta + _EAT_COST))

    # 邻居密度：逐帧序列 + 个体均值
    nn_series: List[float] = []
    per_uid_nn: Dict[int, List[float]] = {}
    for f in log.frames:
        vals = [float(v) for v in f.nn_dist if np.isfinite(v)]
        nn_series.append(math.fsum(vals) / len(vals) if vals else 0.0)
        for i in range(f.uids.shape[0]):
            per_uid_nn.setdefault(int(f.uids[i]), []).append(
                float(f.nn_dist[i]))
    uid_nn_mean = [math.fsum(v) / len(v) for v in per_uid_nn.values() if v]

    # 信号计数：逐 tick 序列 + 个体总量
    signal_series = [0] * (len(log.frames))
    tick_index = {f.tick: i for i, f in enumerate(log.frames)}
    per_uid_signal: Dict[int, int] = {}
    for e in log.events:
        if e.kind != "signal":
            continue
        i = tick_index.get(e.tick)
        if i is not None:
            signal_series[i] += 1
        per_uid_signal[e.actor_id] = per_uid_signal.get(e.actor_id, 0) + 1

    # 逐 tick 动作计数序列
    action_series = {name: [0] * len(log.frames)
                     for name in ACTION_NAMES.values()}
    for e in log.events:
        i = tick_index.get(e.tick)
        name = e.kind if e.kind in action_series else None
        if i is not None and name:
            action_series[name][i] += 1

    return {
        "note": "H1：原始分布，不过任何标签阈值；标签计数见 behaviors 段",
        "window_ticks": th.window_ticks,
        "series": {
            "nn_dist_mean": nn_series,
            "signal_count": signal_series,
            "actions": action_series,
        },
        "histograms": {
            "flee_dist_gain": _hist(flee_gains, vmax=4.0),
            "chase_dist_drop": _hist(chase_drops, vmax=4.0),
            "eat_gain": _hist(eat_gains),
            "uid_nn_dist_mean": _hist(uid_nn_mean, vmax=4.0),
            "uid_signal_total": _hist([float(v) for v in
                                       per_uid_signal.values()]),
        },
    }


def run_digest(log: SimLog) -> str:
    """一次跑批的**逐位指纹**（sha256；关档等价护栏用）。

    把帧（tick/uid/格/能量）、事件流水、生卒、能量台账压成一个定序字符串
    再取 sha256 ⇒ 任何一处行为差异（含 1 ULP）都会换指纹。
    用途：新机制"默认关 ⇒ 与既有口径逐位等价"的可执行证明（B16 硬约束）。
    """
    h = hashlib.sha256()

    def _feed(text: str) -> None:
        h.update(text.encode("utf-8"))

    for f in log.frames:
        _feed("F%d|" % f.tick)
        for i in range(f.uids.shape[0]):
            _feed("%d,%d,%r|" % (int(f.uids[i]), int(f.cells[i]),
                                 float(f.energies[i])))
    for e in log.events:
        _feed("E%d,%d,%s,%s,%d,%d,%r,%r|" % (
            e.tick, e.actor_id, e.kind, e.status, e.target_cell,
            e.target_id, float(e.actor_energy_delta), float(e.counter_damage)))
    for p, c, cell in log.births:
        _feed("B%d,%d,%d|" % (p, c, cell))
    for uid, cause, tick in log.deaths:
        _feed("D%d,%s,%d|" % (uid, cause, tick))
    _feed("$%r,%r,%r" % (float(log.energy_paid), float(log.energy_in),
                         float(log.endowment_paid)))
    return h.hexdigest()


def bimodality_coefficient(values: Sequence[float]) -> float:
    """二型化指数 BC = (skew² + 1) / kurt（SAS 口径；>0.555 常作双峰判据）。

    n<4 或分布退化（kurt≈0）⇒ 返回 0.0（不可估）。
    """
    arr = np.asarray(list(values), dtype=np.float64)
    n = arr.size
    if n < 4:
        return 0.0
    sd = float(arr.std())
    if sd <= 1e-12:
        return 0.0
    z = (arr - float(arr.mean())) / sd
    g1 = float(np.sum(z ** 3)) / n
    g2 = float(np.sum(z ** 4)) / n - 3.0
    bc = (g1 * g1 + 1.0) / (g2 + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    return float(bc)


def predation_json(log: SimLog, world=None, trait_every: int = 500) -> Dict:
    """B16 捕食-猎物军备竞赛度量段（性状轨迹/攻击收益/存活/二型化）。

    数据源：Frame 的可选性状数组（predation 开档时每 tick 记 defense /
    attack_drive）+ 攻击事件流水（含 loot）。关档（数组为 None）⇒
    返回 ``{"enabled": False}``，形状稳定。
    """
    frames = [f for f in log.frames if getattr(f, "defense", None) is not None]
    if not frames:
        return {"enabled": False}

    # ── 攻击收益与受害（事件级）──
    n_attack = n_kill = n_wound = n_fail = 0
    loot_total = 0.0
    counter_total = 0.0
    attacked_of: Dict[int, int] = {}     # uid → 被攻击（命中）次数
    killed_of: Dict[int, int] = {}       # uid → 被打死次数
    loot_of: Dict[int, float] = {}       # uid（攻击者）→ 掠食所得
    net_of: Dict[int, float] = {}        # uid（攻击者）→ 攻击净能量（含成本）
    born_of: Dict[int, int] = {}         # uid（父）→ 子代数（间接收益读数）
    for e in log.events:
        if e.kind != "attack":
            continue
        n_attack += 1
        net_of[e.actor_id] = net_of.get(e.actor_id, 0.0) \
            + float(e.actor_energy_delta)
        counter_total += float(e.counter_damage)
        if e.status == "ok":
            n_kill += 1
            loot_total += float(e.loot)
            loot_of[e.actor_id] = loot_of.get(e.actor_id, 0.0) + float(e.loot)
        elif e.status == "attack_survived":
            n_wound += 1
        elif e.status == "attack_failed":
            n_fail += 1
        if e.status in _ATTACK_HIT_STATUSES and e.target_id >= 0:
            attacked_of[e.target_id] = attacked_of.get(e.target_id, 0) + 1
            if e.status == "ok":
                killed_of[e.target_id] = killed_of.get(e.target_id, 0) + 1

    # ── 繁殖产出（父 → 子代数；攻击的"清格子"间接收益读数）──
    for p, _c, _cell in log.births:
        if p >= 0:
            born_of[p] = born_of.get(p, 0) + 1

    # ── 个体·tick 曝光（被攻击率分母）──
    exposure: Dict[int, int] = {}
    for f in frames:
        for i in range(f.uids.shape[0]):
            uid = int(f.uids[i])
            exposure[uid] = exposure.get(uid, 0) + 1

    # ── 性状轨迹（每 trait_every 抽一帧 + 终帧）──
    picks = [f for f in frames
             if f.tick % trait_every == 0 or f is frames[-1]]
    traj = []
    for f in picks:
        d = np.asarray(f.defense, dtype=np.float64)
        a = np.asarray(f.attack_drive, dtype=np.float64)
        traj.append({
            "tick": int(f.tick),
            "n": int(d.size),
            "defense_mean": float(d.mean()) if d.size else 0.0,
            "defense_var": float(d.var()) if d.size else 0.0,
            "attack_mean": float(a.mean()) if a.size else 0.0,
            "attack_var": float(a.var()) if a.size else 0.0,
            "bc_defense": bimodality_coefficient(d),
            "bc_attack": bimodality_coefficient(a),
        })

    # ── 终局存活 vs 死亡个体的防御（判据③）──
    last = frames[-1]
    d_last = np.asarray(last.defense, dtype=np.float64)
    alive_uids = {int(u) for u in last.uids}
    dead_cause = {uid: cause for uid, cause, _t in log.deaths}
    alive_def = [float(d_last[i]) for i in range(d_last.size)
                 if int(last.uids[i]) in alive_uids]
    dead_def = [float(dead_def_v) for dead_def_v in
                _death_defense(log, frames)]
    killed_def = [v for v in dead_def]

    # ── 谱系口径：谱系 mean defense vs 谱系存活率（判据③谱系版）──
    from biosphere.criteria import lineage_root_map
    roots = lineage_root_map([(p, c, cell) for p, c, cell in log.births])
    by_root: Dict[int, List[float]] = {}
    by_root_alive: Dict[int, List[int]] = {}
    for f in frames:
        for i in range(f.uids.shape[0]):
            uid = int(f.uids[i])
            r = roots.get(uid, uid)
            by_root.setdefault(r, []).append(float(f.defense[i]))
            if f is frames[-1]:
                by_root_alive.setdefault(r, []).append(
                    1 if uid in alive_uids else 0)
    lin_def, lin_surv = [], []
    for r, vals in sorted(by_root.items()):
        if len(vals) < 5:            # 谱系太小 ⇒ 不估
            continue
        lin_def.append(math.fsum(vals) / len(vals))
        surv = by_root_alive.get(r, [])
        lin_surv.append((math.fsum(surv) / len(surv)) if surv else 0.0)

    def _spearman(xs: List[float], ys: List[float]) -> float:
        if len(xs) < 4:
            return 0.0
        rx = _rank(xs)
        ry = _rank(ys)
        mx = math.fsum(rx) / len(rx)
        my = math.fsum(ry) / len(ry)
        num = math.fsum((a_ - mx) * (b_ - my) for a_, b_ in zip(rx, ry))
        dx = math.sqrt(math.fsum((a_ - mx) ** 2 for a_ in rx))
        dy = math.sqrt(math.fsum((b_ - my) ** 2 for b_ in ry))
        return (num / (dx * dy)) if (dx > 1e-12 and dy > 1e-12) else 0.0

    return {
        "enabled": True,
        "attacks": {
            "n_attacks": n_attack, "n_kills": n_kill, "n_wounded": n_wound,
            "n_failed": n_fail,
            "lethality": (n_kill / (n_kill + n_wound))
                         if (n_kill + n_wound) else 0.0,
            "loot_total": float(loot_total),
            "loot_per_kill": (loot_total / n_kill) if n_kill else 0.0,
            "counter_total": float(counter_total),
        },
        "per_capita": {
            "attacks_per_capita": _safe_div(n_attack, sum(exposure.values())),
            "kills_per_capita": _safe_div(n_kill, sum(exposure.values())),
            "loot_per_capita": _safe_div(loot_total, sum(exposure.values())),
        },
        "trajectory": traj,
        "final": {
            "defense_mean": float(d_last.mean()) if d_last.size else 0.0,
            "defense_var": float(d_last.var()) if d_last.size else 0.0,
            "attack_mean": float(np.asarray(last.attack_drive,
                                            dtype=np.float64).mean())
                           if last.attack_drive is not None
                           and len(last.attack_drive) else 0.0,
            "attack_var": float(np.asarray(last.attack_drive,
                                           dtype=np.float64).var())
                          if last.attack_drive is not None
                          and len(last.attack_drive) else 0.0,
            "bc_defense": bimodality_coefficient(d_last),
            "bc_attack": bimodality_coefficient(last.attack_drive),
            "n_final": int(d_last.size),
        },
        "survival": {
            "alive_mean_defense": (math.fsum(alive_def) / len(alive_def))
                                  if alive_def else 0.0,
            "killed_mean_defense": (math.fsum(killed_def) / len(killed_def))
                                   if killed_def else 0.0,
            "n_alive_end": len(alive_def),
            "n_killed": len(killed_def),
            "attacked_per_exposure": {str(u): _safe_div(attacked_of.get(u, 0),
                                                        exposure.get(u, 0))
                                      for u in sorted(exposure)},
        },
        "lineage": {
            "n_lineages": len(lin_def),
            "spearman_defense_survival": _spearman(lin_def, lin_surv),
        },
        "_per_uid": {
            "loot": {str(u): float(v) for u, v in sorted(loot_of.items())},
            "net_attack": {str(u): float(v)
                           for u, v in sorted(net_of.items())},
            "attacked": {str(u): int(v)
                         for u, v in sorted(attacked_of.items())},
            "exposure": {str(u): int(v) for u, v in sorted(exposure.items())},
            "killed": {str(u): int(v) for u, v in sorted(killed_of.items())},
            "births": {str(u): int(v) for u, v in sorted(born_of.items())},
            "dead_cause": {str(u): c for u, c in sorted(dead_cause.items())},
        },
        "note": ("loot=致死掠食取走死前能量×attack_loot_frac；"
                 "被攻击率=命中事件/个体·tick；BC=SAS 二型化系数(>0.555 双峰)"),
    }


def _rank(values: Sequence[float]) -> List[float]:
    """平均秩（并列取均值；Spearman 用）。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and \
                values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def _safe_div(num: float, den: float) -> float:
    return (float(num) / float(den)) if den else 0.0


def _death_defense(log: SimLog, frames: List["Frame"]) -> List[float]:
    """被打死个体的 defense（最后一次出现在帧里的值）。"""
    last_seen: Dict[int, float] = {}
    for f in frames:
        for i in range(f.uids.shape[0]):
            last_seen[int(f.uids[i])] = float(f.defense[i])
    out: List[float] = []
    for uid, cause, _tick in log.deaths:
        if cause == "attacked" and uid in last_seen:
            out.append(last_seen[uid])
    return out


def _crossval_section(log: SimLog, world, thresholds) -> Dict:
    """无标签交叉验证段（延迟 import 防环：crossval 依赖本模块常量）。"""
    from biosphere.crossval import label_free_crossval
    return label_free_crossval(log, world, thresholds)


def metrics_json(log: SimLog, world, seed: int, ticks: int,
                 thresholds: Optional[BehaviorThresholds] = None) -> Dict:
    """汇总全部度量为一个定键 dict（json.dump(sort_keys=True) 后逐位可复现）。"""
    alive_series: List[int] = []
    energy_series: List[float] = []
    for f in log.frames:
        alive_series.append(int(f.uids.shape[0]))
        energy_series.append(float(np.sum(f.energies)))
    death_causes: Dict[str, int] = {}
    for _uid, cause, _tick in log.deaths:
        death_causes[cause] = death_causes.get(cause, 0) + 1
    action_counts: Dict[str, int] = {}
    for e in log.events:
        key = f"{e.kind}:{e.status}"
        action_counts[key] = action_counts.get(key, 0) + 1

    labeler = BehaviorLabeler(log, thresholds)
    behavior_counts = {lab: 0 for lab in BEHAVIOR_LABELS}
    for lab_map in labeler.label_all(world).values():
        for lab, on in lab_map.items():
            if on:
                behavior_counts[lab] += 1
    series = labeler.block_series(world)

    return {
        "schema": SCHEMA,
        "seed": int(seed),
        "ticks": int(ticks),
        "world": {"rows": int(world.rows), "cols": int(world.cols),
                  "n_cells": int(world.n_cells),
                  "tilt_rad": float(world.tilt_rad),
                  "season_period": int(world.season_period)},
        "population": {
            "final_alive": alive_series[-1] if alive_series else 0,
            "peak_alive": max(alive_series) if alive_series else 0,
            "total_born": len(log.births),
            "total_died": len(log.deaths),
            "death_causes": death_causes,
            "alive_series": alive_series,
            "energy_series": energy_series,
        },
        "energy": {
            "intent_cost_paid": float(log.energy_paid),
            "world_inflow": float(log.energy_in),
            "endowment_paid": float(log.endowment_paid),
            "final_total": energy_series[-1] if energy_series else 0.0,
        },
        "behaviors": {
            "counts": behavior_counts,
            "series": series,
            "block_ticks": BLOCK_TICKS,
        },
        "flee_gain_distribution": flee_gain_distribution(log, world),
        "chase_gain_distribution": chase_gain_distribution(log, world),
        "raw_distributions": raw_distributions_json(log, world),
        "crossval": _crossval_section(log, world, thresholds),
        "groups": groups_json(log, world),
        "c3": c3_json(log, world),
        "lineage": lineage_json(log),
        "interactions": interaction_json(log),
        "action_counts": action_counts,
    }


__all__ = [
    "BehaviorThresholds", "raw_distributions_json", "c3_json",
    "FORAGE_GAIN_MIN", "FORAGE_SPEED_MIN", "FLEE_DIST_GAIN_MIN",
    "CHASE_DIST_DROP_MIN", "GROUP_NN_MAX", "GROUP_SAMPLES_MIN",
    "SIGNAL_COUNT_MIN", "BLOCK_TICKS", "BEHAVIOR_LABELS", "SCHEMA",
    "Frame", "SimLog", "BehaviorLabeler", "lineage_json",
    "interaction_json", "metrics_json",
    "flee_gain_distribution", "chase_gain_distribution",
]
