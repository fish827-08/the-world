"""biosphere.c3：C3 v2（发送者获益通道；owner 2026-10-19 已批）。

**写死的只有物理，不写死语义**（H2 教训 + owner 批条）：
- 徽章（badge）K 类可遗传，**含义为空**——没有任何"信号=威慑"规则；
- 信号发射**耗能**（cost = base × emit_rate），写入 K 通道信号层；
- 接收反应 ``response[K]`` 可遗传（signed：正 = 倾向攻击该徽章载体，
  负 = 倾向远离）；
- 打斗结果由**实力**决定（strength/defense + B14.2 反击风险）；
- 记忆（可选）：learning_rate 基因，打斗结果更新 response（首轮默认关）。

"威慑/友善"等含义**只允许**从「打斗代价 → 选择/学习 → response 符号」
中涌现；本模块不包含任何把信号直接接到攻击欲的规则。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

#: 徽章种类数（K；冻结，改 = 新预注册轮）
C3_BADGE_K: int = 4

#: 发射耗能基数（cost = SIGNAL_EMIT_COST_BASE × emit_rate）
SIGNAL_EMIT_COST_BASE: float = 0.20
#: 徽章可见下限（信号层该通道强度 ≥ 此值才算"带着可见徽章"）
EMIT_VISIBLE_MIN: float = 0.10
#: response 正项对 ATTACK 得分的缩放
RESPONSE_ATTACK_SCALE: float = 1.0
#: response 负项对 MOVE 的回避缩放（每多一个邻接的负徽章载体扣这么多）
RESPONSE_AVOID_SCALE: float = 1.0
#: strength 对攻击威力的加成系数（0 = 关强度不对称的负对照）
STRENGTH_POWER_COEF: float = 6.0
#: 记忆更新幅度（× learning_rate）
MEMORY_STEP: float = 0.25


@dataclass
class C3Config:
    """C3 v2 开关与系数（enabled=False ⇒ 全关 ⇒ 与既有口径逐位等价）。"""

    enabled: bool = False
    badge_k: int = C3_BADGE_K
    memory_enabled: bool = False        # 首轮默认关（负对照 = 默认）
    strength_effect: bool = True        # False = 关强度不对称（负对照）
    badge_reshuffle: bool = False       # True = 子代徽章每代重随机（负对照）
    # ── B15.1 oracle 正对照（诊断用；默认关 ⇒ 逐位等价） ──
    #: oracle-response：每 tick 把所有个体 response_state 覆写为
    #: “按徽章载体真实 strength 设符号”（强徽章⇒回避、弱徽章⇒倾向攻）
    oracle_response: bool = False
    #: oracle-emitter：只有 strength ≥ emitter_strength_min 的个体可发射
    #: （强制 SIGNAL 胜出）；低于门槛 SIGNAL 被 -inf 硬压
    oracle_emitter: bool = False
    emitter_strength_min: float = 0.50
    #: oracle-response 臂的 SIGNAL 保底得分（无保底则无人发射、判据③无分母）
    oracle_signal_floor: float = 0.50
    signal_emit_cost_base: float = SIGNAL_EMIT_COST_BASE
    emit_visible_min: float = EMIT_VISIBLE_MIN
    response_attack_scale: float = RESPONSE_ATTACK_SCALE
    response_avoid_scale: float = RESPONSE_AVOID_SCALE
    strength_power_coef: float = STRENGTH_POWER_COEF
    memory_step: float = MEMORY_STEP


def badge_from_response(cfg: C3Config, response: Tuple[float, ...]) -> int:
    """（工具）response 里最负/最正的徽章——度量用，不参与机制。"""
    if not response:
        return -1
    return int(max(range(len(response)), key=lambda k: response[k]))


def visible_badge(badge_row: np.ndarray,  # noqa: F821 - (K,) 该格各通道强度
                  visible_min: float) -> int:
    """该格当前可见徽章（强度最大的通道且过可见下限；-1 = 不可见）。"""
    k = int(np.argmax(badge_row))
    if float(badge_row[k]) >= visible_min:
        return k
    return -1


def learning_update(response: np.ndarray,  # noqa: F821 - (K,) 可变
                    badge: int, outcome: float, learning_rate: float,
                    step: float = MEMORY_STEP) -> float:
    """记忆更新：outcome>0（得手）⇒ response[badge] += lr×step；
    outcome<0（吃亏）⇒ 减。返回实际变化量（lr=0 ⇒ 0）。"""
    if learning_rate <= 0.0 or badge < 0 or badge >= response.shape[0]:
        return 0.0
    delta = learning_rate * step * (1.0 if outcome > 0 else -1.0)
    response[badge] = float(np.clip(response[badge] + delta, -1.0, 1.0))
    return delta


__all__ = [
    "C3_BADGE_K", "SIGNAL_EMIT_COST_BASE", "EMIT_VISIBLE_MIN",
    "RESPONSE_ATTACK_SCALE", "RESPONSE_AVOID_SCALE", "STRENGTH_POWER_COEF",
    "MEMORY_STEP", "C3Config", "badge_from_response", "visible_badge",
    "learning_update",
]
