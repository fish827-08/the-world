"""TickStats：单个 tick 结束时的世界快照（模块二 · 依赖文件）。

通俗理解
--------
引擎每推进一个 tick，就"拍一张照"，记录这一瞬间世界的状态：
有多少生物活着、这 1 tick 里出生了几个、死了几个、各自为什么死、
所有生物手里总共有多少可用能量、全世界还剩多少食物。

这些快照串起来就是"全程录像"：
- 观察层（observatory / 前端）需要按 tick 回放 → 读这份数据；
- 复现验证（跑两次应该得到一样的序列）→ 比较这份数据；
- 存档恢复（中途停下来接着跑）→ 从这份数据接回去。

为什么是 frozen dataclass
-------------------------
- frozen：快照一旦生成就不可变，防止观察者或后续代码不小心改坏历史；
- dataclass：字段一目了然，构造/比较/打印都方便，零额外依赖。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from core.lifecycle import DeathCause


@dataclass(frozen=True)
class TickStats:
    """一个 tick 结束时可观测的世界状态。"""

    tick: int            # 第几个 tick（从 1 开始）
    population: int      # 当前存活生物数量
    born: int            # 本 tick 出生数量
    died: int            # 本 tick 死亡数量
    deaths_by_cause: Counter  # 按死因（DeathCause）统计的死亡人数
    total_energy: float  # 全体存活生物"可用能量"之和
    total_resource: float  # 全世界剩余食物总量

    def to_dict(self) -> dict:
        """转成 JSON 安全的普通字典（存档/对比用）。

        死因是枚举，不能直接落盘，这里转成字符串（"starvation" 等）。
        """
        deaths = {
            (
                cause.value
                if isinstance(cause, DeathCause)
                else str(cause)
            ): count
            for cause, count in self.deaths_by_cause.items()
        }
        return {
            "tick": self.tick,
            "population": self.population,
            "born": self.born,
            "died": self.died,
            "deaths": deaths,
            "total_energy": self.total_energy,
            "total_resource": self.total_resource,
        }