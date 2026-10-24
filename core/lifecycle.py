"""Lifecycle 基础契约：死亡原因枚举（模块二 · 依赖文件）。

为什么单独放一个文件
--------------------
引擎在统计"每 tick 死了多少、为什么死"时需要一个统一的死因标记
（饿死 / 老死），死因会写进 TickStats 的统计账本。这个枚举属于
"生命规则层"的契约，比引擎更底层，所以放在 core 包里单独成文件，
引擎只负责填充它 —— 单一职责。

和旧项目 digital_life 的关系
----------------------------
旧项目的 Lifecycle 类还包含年龄推进、存活状态、寿命判定等逻辑；
新项目（球面版）这些逻辑已经向量化进 SphereEngine 的数组操作里，
当前只需要"死因枚举"这一份契约。等 Rust 热核描绘边界（模块三）时，
再决定各类契约是否以本文件为基础扩展。
"""
from __future__ import annotations

from enum import Enum


class DeathCause(Enum):
    """个体死亡原因（引擎统计用）。"""

    STARVATION = "starvation"  # 饿死：能量 ≤ 0
    OLD_AGE = "old_age"        # 老死：年龄 ≥ 寿命（基因 g3 决定）
    PREDATION = "predation"    # 被捕食：被其他个体攻击致死（L4）