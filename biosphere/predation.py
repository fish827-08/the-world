"""biosphere.predation：捕食-猎物军备竞赛（B16；只写物理与测量，不写角色）。

通俗理解
--------
这一轮要检验的是：**"打"和"防"能不能在世界里互相推着往上走**（军备竞赛）。
世界里已经存在的物理（不是本轮新造的）：

| 环节 | 现有物理 | 出处 |
|---|---|---|
| 攻击威力 | ``power = 4 + 8×attack_drive``（+ 强度项） | ``resolve.ATTACK_POWER_*`` |
| 防御减伤 | ``damage × (1 − 0.6×defense)`` | ``resolve.DEFENSE_MIT`` |
| 反击风险 | 目标 ``8×defense − power`` 超额 ⇒ 攻击者倒付能量 | B14.2 ``COUNTER_*`` |
| 能量转移 | 致死 ⇒ 攻击者取走死前能量 × ``attack_loot_frac``（**唯一收益通道**） | ``resolve.ATTACK_LOOT_FRAC`` |

**不手派角色**：没有人被贴上"捕食者/猎物"标签，两个性状（attack_drive /
defense）的初始方差来自随机基因组（可显式加宽），谁攻谁防由行为涌现。

Oracle 正对照（B16.1，诊断用，**默认关**）
----------------------------------------
演化跑不出来时，分不清是"选择压不够"还是"因果链根本不存在"。
oracle 的做法是**直接把性状钉死成极值**再测收益：

- ``oracle_defense``：按 uid 奇偶把个体钉成 defense=1 / defense=0 两半；
- ``oracle_attack``：按 uid 奇偶把个体钉成 attack_drive=1 / 0 两半。

若连"钉死的极端防御"都换不来存活差、连"钉死的极端攻击"都拿不到正 loot
⇒ 因果链断（与 B15 威慑线同一条诊断纪律）。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict

from biosphere.develop import ControllerParams
from biosphere.genome import PARAM_SLOTS, Genome

#: oracle 高组/低组的默认极值（defense / attack_drive 都 ∈ [0,1]）
ORACLE_HI: float = 1.0
ORACLE_LO: float = 0.0


def oracle_group(uid: int) -> int:
    """个体属于哪一半：**0 = 低组 / 1 = 高组**（uid 奇偶，确定性）。

    为什么用 uid 奇偶：① 与基因无关 ⇒ 不会被"高性状个体恰好也在高组"
    污染；② 跨 seed/跨臂同一规则 ⇒ 组间可比；③ 无需额外随机源。
    """
    return int(uid) & 1


@dataclass
class PredationConfig:
    """B16 开关（``enabled=False`` ⇒ 与既有口径逐位等价）。

    Attributes:
        enabled: 总开关（关 ⇒ 不改任何行为、不记性状帧）。
        oracle_defense: 强制半数 defense=hi、半数=lo（诊断正对照）。
        oracle_attack: 强制半数 attack_drive=hi、半数=lo（诊断正对照）。
        init_trait_spread: 初代两个性状基因的注入标准差（0 = 不改初始分布，
            用默认随机基因组；>0 时按 N(0, spread) 重画 defense/attack 槽位）。
        record_traits: 是否每 tick 记 defense/attack_drive 帧（判据数据源）。
    """

    enabled: bool = False
    # ── B16.1 oracle 正对照（默认关 ⇒ 逐位等价） ──
    oracle_defense: bool = False
    oracle_attack: bool = False
    defense_hi: float = ORACLE_HI
    defense_lo: float = ORACLE_LO
    attack_hi: float = ORACLE_HI
    attack_lo: float = ORACLE_LO
    # ── B16.2 初始方差注入（不手派角色） ──
    init_trait_spread: float = 0.0
    record_traits: bool = True


def apply_oracle_traits(controllers: Dict[int, ControllerParams],
                        cfg: PredationConfig,
                        ) -> Dict[int, ControllerParams]:
    """把 oracle 极值钉到控制器副本上（不动缓存里的发育产物）。

    ``ControllerParams`` 是 frozen dataclass ⇒ 用 ``replace`` 造副本；
    未开 oracle 时原样返回（**同一批对象**，非拷贝）。
    """
    if not (cfg.oracle_defense or cfg.oracle_attack):
        return controllers
    out: Dict[int, ControllerParams] = {}
    for uid, c in controllers.items():
        hi = oracle_group(uid) == 1
        if cfg.oracle_defense:
            c = replace(c,
                        defense=cfg.defense_hi if hi else cfg.defense_lo)
        if cfg.oracle_attack:
            c = replace(c, attack_drive=(cfg.attack_hi if hi
                                         else cfg.attack_lo))
        out[uid] = c
    return out


def spread_founder_traits(genome: Genome, rng, cfg: PredationConfig,
                          ) -> Genome:
    """初代性状方差注入：重画 defense / attack_drive 两个槽位。

    ``init_trait_spread <= 0`` ⇒ 原样返回（默认 ⇒ 逐位等价）。
    RNG 消费顺序写死：先 defense 后 attack_drive。
    """
    if cfg.init_trait_spread <= 0.0:
        return genome
    s = float(cfg.init_trait_spread)
    g = genome
    for name in ("defense", "attack_drive"):
        g = g.with_gene(PARAM_SLOTS[name],
                        float(rng.normal(0.0, s)))
    return g


__all__ = [
    "ORACLE_HI", "ORACLE_LO", "PredationConfig", "oracle_group",
    "apply_oracle_traits", "spread_founder_traits",
]
