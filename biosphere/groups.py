"""biosphere.groups：W4/W4b 团机制（两套团定义 / 团共享 / 组繁殖）。

通俗理解
--------
W4b 起**团分两套定义**（B12 核验件要求）：

- **邻接团**（`build_group_index`）：占用图连通片（相邻即成团）。
  🔴 只作**对照**——拥挤本身就成团，不能用它支持"粘连导致成团"。
- **粘附链团**（`build_adhesion_chain_index`）：仅按**粘附边**（相邻
  **且两端都粘**）的连通片。**所有主张只看这套**（团大小/组繁殖/作弊者）。

机制（共享能量、按团繁殖、粘附硬过滤）一律跑在**粘附链团**上；邻接团
只进度量对照段。机制细节：

- **粘附（adhesion）**：可演化基因；表达受双稳态分化调制
  （state=0 只表达一半）；粘附细胞 MOVE 有贴团奖励与"不离团"硬过滤；
- **团共享能量**：每 tick 每团建公共池——成员按
  `SHARE_BASE × share_drive` 比例注入，池子平分给全体。
  注入低于平分线 1/n 者**净蹭团**（cheater，作弊者）；
- **按团繁殖（组选择）**：团**总能量**过人均门槛才可繁殖，成本团员平摊、
  基因组取团内最高能量者（"生殖系"）——高共享团积累快、繁殖多。

纪律：全部机制默认关（`W4Config.enabled=False` ⇒ 与 W0-B 逐位等价）；
W4/W4b 实验由 `biosphere.experiments w4` 显式打开。所有浮点账按 uid 序、
公共池用 fsum（精确、序无关）。>1000 tick 跑批走 checkpoint 协议
（`biosphere/checkpoint.py`；续跑 = 直跑逐位）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from biosphere.genome import Genome, mutate

# ── W4 常量（冻结；调参须改规格页并重走预注册） ──────────────────────────

#: 共享注入基础比例（注入 = 此值 × share_drive；低于 1/n 即净蹭团）
SHARE_BASE: float = 0.30
#: 作弊者判定边际（净蹭能量超过此值才记作弊，防噪声）
CHEATER_MARGIN: float = 1.0
#: 组繁殖：人均能量门槛（团总能量 ≥ 此值 × 团大小 才可繁殖；
#: 80 = 个体繁殖门槛的均值量级 ⇒ 团繁殖不获得退化性优势）
GROUP_REPRO_THRESH_PER_CAPITA: float = 80.0
#: 组繁殖：每次子代数 / 每个子代 endowment（父团平摊成本）
GROUP_CHILD_CELLS: int = 2
GROUP_ENDOWMENT: float = 20.0
#: 粘附硬过滤下限（adhesion ≥ 此值 ⇒ 团在时只贴团移动）
ADHESION_STICKY: float = 0.50
#: 贴团 MOVE 奖励（× adhesion × 候选格有团邻居）
ADHESION_COHESION: float = 2.0
#: W4c-G2 亲缘结构：计数加权贴团奖励权重 + 粘附个体抖动缩放（降扩散）
KIN_COHESION_WEIGHT: float = 1.0
STICKY_JITTER_SCALE: float = 0.5
#: W4c-G3 群体觅食分红：每 tick 每多体链团均分
#: `strength × GROUP_FORAGE_BASE × 团大小` 能量（世界侧流入）
GROUP_FORAGE_BASE: float = 0.1
#: 分化度量监控的表达参数（团内方差）
DIFF_PARAMS: Tuple[str, ...] = ("attack_drive", "share_drive", "adhesion",
                                "metabolism")


@dataclass
class W4Config:
    """W4 机制开关（enabled=False ⇒ 全关 ⇒ 与 W0-B 逐位等价）。

    零模型（W4 四臂）由本配置的组合表达：
    - Z1 关粘附：`adhesion=False`；
    - Z2 关组选择：`group_reproduction=False`；
    - Z3 全混合：`full_mixing=True`（每 tick 全局重排，空间结构归零）；
    - Z4 关信号：`signal=False`（SIGNAL 意图抑制 + 信号层不写）。

    W4c 三旋钮（预注册 §13.12，默认值 = W4b baseline）：
    - G1 `group_repro_thresh_per_capita`：组繁殖人均门槛 {40, 80, 160}；
    - G2 `kin_cohesion`：亲缘结构（计数加权贴团 + 粘附个体降扩散）；
    - G3 `adhesion_benefit`：群体觅食分红强度 {0, 0.5, 1.0}。
    """

    enabled: bool = False
    adhesion: bool = True
    group_sharing: bool = True
    group_reproduction: bool = True
    full_mixing: bool = False
    signal: bool = True
    # W4c 旋钮（默认 = W4b baseline ⇒ 不动即等价）
    group_repro_thresh_per_capita: float = GROUP_REPRO_THRESH_PER_CAPITA
    kin_cohesion: bool = False
    adhesion_benefit: float = 0.0


@dataclass
class GroupIndex:
    """一个 tick 的团结构（占用图连通片）。"""

    group_of: Dict[int, int] = field(default_factory=dict)   # uid → gid
    members: Dict[int, List[int]] = field(default_factory=dict)  # gid → uids
    n_groups: int = 0

    def size(self, uid: int) -> int:
        gid = self.group_of.get(uid)
        return len(self.members[gid]) if gid is not None else 1

    def group_id(self, uid: int) -> Optional[int]:
        return self.group_of.get(uid)


class _UnionFind:
    """按最小根合并的并查集（根 = 团內最小 uid ⇒ 编号确定性）。"""

    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}

    def find(self, x: int) -> int:
        p = self.parent
        p.setdefault(x, x)
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:      # 路径压缩
            p[x], x = root, p[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def _components(occ: Dict[int, int], world, edge_ok) -> GroupIndex:
    """按边谓词求占用图连通片（格序升序处理 ⇒ 结果与边序无关）。"""
    uf = _UnionFind()
    for cell in sorted(occ):
        uid = occ[cell]
        uf.find(uid)
        for nb in world.neighbors[cell]:
            nb = int(nb)
            if nb in occ and nb != cell and edge_ok(uid, occ[nb]):
                uf.union(uid, occ[nb])
    comp: Dict[int, List[int]] = {}
    for uid in sorted(occ.values()):
        comp.setdefault(uf.find(uid), []).append(uid)
    index = GroupIndex()
    for gid, (root, uids) in enumerate(sorted(comp.items())):
        index.members[gid] = uids
        for uid in uids:
            index.group_of[uid] = gid
    index.n_groups = len(index.members)
    return index


def build_group_index(population, world) -> GroupIndex:
    """**邻接团**（W4b 起仅作对照）：占用图连通片（相邻即成团）。

    🔴 密度混淆根源：拥挤本身就成团 ⇒ 不能用它支持"粘连导致成团"。
    """
    occ: Dict[int, int] = {ind.cell: ind.uid
                           for ind in population.individuals}
    return _components(occ, world, lambda a, b: True)


def is_sticky(controller, diff_state: int) -> bool:
    """细胞是否"粘"（表达后粘附 ≥ 门槛；受双稳态分化调制）。"""
    from biosphere.cell import adhesion_expression
    return (adhesion_expression(controller.adhesion, diff_state)
            >= ADHESION_STICKY)


def build_adhesion_chain_index(population, world,
                               controllers) -> GroupIndex:
    """**粘附链团**（W4b 主张用）：仅按**粘附边**（两端都粘）的连通片。

    非粘附细胞在本镜头下恒为单体 ⇒ "成团"必须由粘连解释，不受拥挤
    混淆。所有 W4b 主张（团大小/组繁殖/作弊者）只看这个镜头。
    """
    from biosphere.cell import adhesion_expression
    occ: Dict[int, int] = {ind.cell: ind.uid
                           for ind in population.individuals}
    sticky: Dict[int, bool] = {}
    for ind in population.individuals:
        sticky[ind.uid] = (adhesion_expression(
            controllers[ind.uid].adhesion, ind.diff_state)
            >= ADHESION_STICKY)

    def edge_ok(u: int, v: int) -> bool:
        return bool(sticky.get(u)) and bool(sticky.get(v))

    return _components(occ, world, edge_ok)


def share_injection_ratio(uids: List[int], controllers) -> Dict[str, float]:
    """团的共享注入比（判读"高共享团 vs 低共享团"的分组量）。

    - ``mean_share_drive``：团内 share_drive 均值；
    - ``ratio_literal`` = mean × n（派工口径：mean share_drive 对 1/n 的比）；
    - ``ratio_net`` = SHARE_BASE × mean × n：**平均注入是否超过平分线 1/n**
      ⇒ >1 = 该团平均净奉献（高共享），≤1 = 净蹭（低共享）。
    """
    n = max(1, len(uids))
    vals = [controllers[u].share_drive for u in uids]
    mean = math.fsum(vals) / n
    return {
        "n": n,
        "mean_share_drive": mean,
        "ratio_literal": mean * n,
        "ratio_net": SHARE_BASE * mean * n,
        "high_share": SHARE_BASE * mean * n > 1.0,
    }


def share_group_energy(population, index: GroupIndex, controllers,
                       report) -> None:
    """团共享能量：公共池注入（按 share_drive）→ 平分；记账到个体。

    只对 size ≥ 2 的团生效（单人"团"无共享）。注入/收取累计到
    `Individual.pool_out / pool_in`（作弊者判据的数据源）。
    """
    by_uid = {ind.uid: ind for ind in population.individuals}
    for gid in sorted(index.members):
        uids = index.members[gid]
        if len(uids) < 2:
            continue
        inds = [by_uid[u] for u in uids]           # uid 序
        contribs = [SHARE_BASE * controllers[u].share_drive * ind.energy
                    for u, ind in zip(uids, inds)]
        pool = math.fsum(contribs)                 # 精确、序无关
        recv_each = pool / len(inds)
        for ind, c in zip(inds, contribs):
            ind.energy += recv_each - c
            ind.pool_out += c
            ind.pool_in += recv_each
        report.group_share_events += 1


def reproduce_groups(population, index: GroupIndex, world, controllers,
                     rng, tick: int, report, max_pop: int,
                     events_out: Optional[List[Dict]] = None,
                     census_out: Optional[List[Dict]] = None,
                     thresh_per_capita: float
                     = GROUP_REPRO_THRESH_PER_CAPITA) -> None:
    """按团繁殖（组选择）：团总能量过门槛 ⇒ 成本平摊、基因组取最高能量者。

    ``events_out`` 传入列表时，逐事件追加组繁殖构成记录（W4b 判据数据）：
    团大小、共享注入比（mean share_drive 与 1/n 之比）、生殖系（团内最高
    能量者）uid/能量、子代 uid、成本。
    """
    by_uid = {ind.uid: ind for ind in population.individuals}
    occ: Dict[int, int] = {ind.cell: ind.uid
                           for ind in population.individuals}
    for gid in sorted(index.members):
        uids = index.members[gid]
        if len(uids) < 2:
            continue
        inds = [by_uid[u] for u in uids]
        total = math.fsum(ind.energy for ind in inds)
        comp = share_injection_ratio(uids, controllers)
        eligible = total >= thresh_per_capita * len(inds)
        # 🔴 普查记在**决策点**（扣费前）：高共享团是净奉献者，扣完费就不达
        # 门槛 ⇒ 事后普查会系统性漏记它们（曾致"高共享繁殖率恒 0"的假读数）
        row = {"tick": tick, "gid": gid, "size": len(inds),
               "mean_share_drive": comp["mean_share_drive"],
               "ratio_literal": comp["ratio_literal"],
               "ratio_net": comp["ratio_net"],
               "high_share": comp["high_share"],
               "eligible": bool(eligible), "reproduced": False}
        if census_out is not None:
            census_out.append(row)
        if not eligible:
            continue
        if len(population.individuals) >= max_pop:
            continue
        # 生殖系 = 团内最高能量者（平手取最低 uid）
        germ = min(inds, key=lambda i: (-i.energy, i.uid))
        # 子代格：团成员邻域中的空格（flat 序升序，确定性）
        free: List[int] = []
        for u in uids:
            for nb in world.neighbors[by_uid[u].cell]:
                nb = int(nb)
                if nb not in occ and nb not in free:
                    free.append(nb)
        free.sort()
        n_children = min(GROUP_CHILD_CELLS, len(free))
        if n_children <= 0:
            continue
        cost_each = GROUP_ENDOWMENT * n_children / len(inds)
        for ind in inds:
            ind.energy -= cost_each
        row["reproduced"] = True
        for k in range(n_children):
            cell = free[k]
            child_genome, _rec = mutate(germ.genome, rng,
                                        uid=population.next_id)
            child = population.spawn(child_genome, cell, GROUP_ENDOWMENT,
                                     tick, germ.uid)
            occ[cell] = child.uid
            by_uid[child.uid] = child
            report.group_births.append((gid, germ.uid, child.uid, cell))
            if events_out is not None:
                events_out.append({
                    "tick": tick, "gid": gid, "size": len(inds),
                    "mean_share_drive": comp["mean_share_drive"],
                    "ratio_literal": comp["ratio_literal"],
                    "ratio_net": comp["ratio_net"],
                    "high_share": comp["high_share"],
                    "germ_uid": germ.uid, "germ_energy": germ.energy,
                    "child_uid": child.uid, "cost_each": cost_each,
                })


def group_forage_dividend(population, index: GroupIndex,
                            strength: float) -> float:
    """W4c-G3 群体觅食分红：多体链团每 tick 均分
    ``strength × GROUP_FORAGE_BASE × 团大小`` 能量。

    返回总分红（调用方计入 report.energy_in —— 世界侧流入，不是无中生有）。
    """
    if strength <= 0.0:
        return 0.0
    by_uid = {ind.uid: ind for ind in population.individuals}
    total_out = 0.0
    for gid in sorted(index.members):
        uids = index.members[gid]
        if len(uids) < 2:
            continue
        pot = strength * GROUP_FORAGE_BASE * len(uids)
        each = pot / len(uids)
        for u in uids:
            by_uid[u].energy += each
        total_out += pot
    return total_out


def group_expression_variance(index: GroupIndex, controllers) -> Dict[str, float]:
    """团内表达方差（分化读数）：对 DIFF_PARAMS 各算团内方差，再取均值。

    缺控制器项的个体（已死/新生未入快照）跳过 ⇒ 不炸。
    """
    per_param: Dict[str, List[float]] = {p: [] for p in DIFF_PARAMS}
    for gid, uids in index.members.items():
        if len(uids) < 2:
            continue
        for p in DIFF_PARAMS:
            vals = [getattr(controllers[u], p) for u in uids
                    if u in controllers]
            if len(vals) < 2:
                continue
            mu = math.fsum(vals) / len(vals)
            var = math.fsum((v - mu) ** 2 for v in vals) / len(vals)
            per_param[p].append(var)
    out: Dict[str, float] = {}
    for p, vs in per_param.items():
        out[p] = math.fsum(vs) / len(vs) if vs else 0.0
    out["mean"] = math.fsum(out[p] for p in DIFF_PARAMS) / len(DIFF_PARAMS)
    return out


__all__ = [
    "SHARE_BASE", "CHEATER_MARGIN", "GROUP_REPRO_THRESH_PER_CAPITA",
    "GROUP_CHILD_CELLS", "GROUP_ENDOWMENT", "ADHESION_STICKY",
    "ADHESION_COHESION", "KIN_COHESION_WEIGHT", "STICKY_JITTER_SCALE",
    "GROUP_FORAGE_BASE", "DIFF_PARAMS",
    "W4Config", "GroupIndex", "build_group_index",
    "build_adhesion_chain_index", "is_sticky", "share_injection_ratio",
    "share_group_energy", "reproduce_groups", "group_forage_dividend",
    "group_expression_variance",
]
