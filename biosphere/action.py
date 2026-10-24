"""biosphere.action：六类动作意图 + 显式能量成本（B 线 MVP，W0-B）。

通俗理解
--------
个体每一步先"想"（并行写意图），再由 ``resolve`` 统一裁定。想法本身是一个
``Intent``：谁、想干什么、对谁/哪格、带多少能量。**每个意图都带显式成本**
——没有免费的动作；成本在裁定阶段（resolve）支付，无论成败
（"尝试即耗能"，与规格页 §四 记账口径一致）。

六类动作（总纲 §四 表型行）：

| kind | 目标 | 效果 | 基准成本 |
|---|---|---|---|
| MOVE | 目标格 | 位移（一格一写者） | 0.30 |
| EAT | 自身格资源 | 从资源池进食 | 0.05 |
| ATTACK | 目标个体 | 削减对方能量，致死则掠食 | 0.40 |
| SHARE | 目标个体 | 向对方转移能量 | 0.20 |
| SIGNAL | 自身格 | 写入信号层（邻居下 tick 可读） | 0.15 |
| REPRODUCE | 无（自身） | 产子（变异基因组 + 能量投入） | 0.50 |
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict

from biosphere.develop import ControllerParams


class ActionKind(IntEnum):
    """六类动作意图（值 = 排序键，勿改）。"""

    MOVE = 0
    EAT = 1
    ATTACK = 2
    SHARE = 3
    SIGNAL = 4
    REPRODUCE = 5


#: 动作名（日志/JSON 用；索引对齐 ActionKind）
ACTION_NAMES: Dict[int, str] = {
    int(ActionKind.MOVE): "move",
    int(ActionKind.EAT): "eat",
    int(ActionKind.ATTACK): "attack",
    int(ActionKind.SHARE): "share",
    int(ActionKind.SIGNAL): "signal",
    int(ActionKind.REPRODUCE): "reproduce",
}

#: 基准能量成本（乘子前的部分；MOVE 另乘 controller.move_cost_mult）
ACTION_BASE_COST: Dict[int, float] = {
    int(ActionKind.MOVE): 0.30,
    int(ActionKind.EAT): 0.05,
    int(ActionKind.ATTACK): 0.40,
    int(ActionKind.SHARE): 0.20,
    int(ActionKind.SIGNAL): 0.15,
    int(ActionKind.REPRODUCE): 0.50,
}


class ResolveStatus:
    """意图裁定结果（字符串常量；进度量日志）。"""

    OK = "ok"                        # 成功生效
    STAY_OCCUPIED = "stay_occupied"  # 移动：目标格有未离开的占用者
    STAY_CONTESTED = "stay_contested"  # 移动：一格一写者竞争失败
    STAY_HABITAT = "stay_habitat"    # 移动：栖息不适配被拦
    EAT_EMPTY = "eat_empty"          # 进食：资源池已空
    ATTACK_NO_TARGET = "attack_no_target"  # 攻击：目标已死/不在邻域
    ATTACK_SURVIVED = "attack_survived"    # 攻击：命中但目标未死
    ATTACK_FAILED = "attack_failed"        # B14.2：目标反击过强，攻击失败
    SHARE_NO_TARGET = "share_no_target"    # 分享：目标无效
    REPRO_NO_ENERGY = "repro_no_energy"    # 繁殖：能量不足
    REPRO_NO_ROOM = "repro_no_room"        # 繁殖：无空邻格/种群上限
    SIGNAL_OK = "signal_ok"          # 信号：恒成功（成本照付）


@dataclass(frozen=True)
class Intent:
    """一个个体在一个 tick 的动作意图（不可变值对象）。

    Attributes:
        actor_id: 意图作者（个体 uid）。
        kind: 动作类型。
        target_cell: MOVE 的目标格；EAT/SIGNAL 为自身格；其余 -1。
        target_id: ATTACK/SHARE 的目标个体 uid；无目标为 -1。
        payload: SHARE 的转移能量 / SIGNAL 的信号强度；其余 0。
        energy_cost: **显式**能量成本（决策时按控制器算定，裁定期支付）。
    """

    actor_id: int
    kind: ActionKind
    target_cell: int = -1
    target_id: int = -1
    payload: float = 0.0
    energy_cost: float = 0.0


def intent_cost(kind: ActionKind, controller: ControllerParams) -> float:
    """按控制器算显式成本（决策时调用；写进 Intent 后不再变）。"""
    base = ACTION_BASE_COST[int(kind)]
    if kind == ActionKind.MOVE:
        return base * controller.move_cost_mult
    return base


def validate_intent(intent: Intent) -> None:
    """意图结构自检（构造期 fail-loud）。"""
    if intent.energy_cost < 0.0:
        raise ValueError(f"energy_cost 不能为负: {intent.energy_cost}")
    if intent.kind == ActionKind.MOVE and intent.target_cell < 0:
        raise ValueError("MOVE 意图须带 target_cell")
    if intent.kind in (ActionKind.ATTACK, ActionKind.SHARE) and intent.target_id < 0:
        raise ValueError(f"{ACTION_NAMES[int(intent.kind)]} 意图须带 target_id")
    if intent.kind == ActionKind.SHARE and intent.payload <= 0.0:
        raise ValueError("SHARE 意图须带正 payload")


__all__ = [
    "ActionKind", "ACTION_NAMES", "ACTION_BASE_COST", "ResolveStatus",
    "Intent", "intent_cost", "validate_intent",
]
