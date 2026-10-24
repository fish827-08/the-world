"""EvolutionObserver：以"世代"为轴的系统采样器（模块四 · 球面引擎适配版）。

采样事件（两个触发器，保证时间轴与世代轴都连续）：
1) 每当种群 max_generation 前进（新世代出现）时采样一次；
2) 按 tick_interval 兜底采样（世代长期停滞 / 灭绝中的空种群，
   也保证时间序列单调有覆盖）。

样本内容：当前全种群统计（GenerationStats）+ 自上次采样以来的
出生/死亡窗口聚合，出生率/死亡率以 per-tick 记（个体/tick），
窗口跨度 span_ticks 一并给出，供下游归一化比较。

与旧版（digital_life/observatory/observer.py）的差异
-----------------------------------------------------
旧版挂 SimulationEngine（Organism 对象列表 + engine.population 门面）；
本版挂 SphereEngine（SoA 数组存储，无 population 对象），直接读
引擎内部世代号（engine._max_generation），其余编排逻辑不变。
"""
from __future__ import annotations

from dataclasses import dataclass

from observatory.statistics import GenerationStats, generation_statistics
from simulation.sphere_engine import SphereEngine
from simulation.tick import TickStats


@dataclass(frozen=True)
class GenerationSample:
    """一个观测点（写 CSV 的一行）。"""

    tick: int
    generation: int
    span_ticks: int  # 距上一观测点的 tick 数
    born_since_prev: int
    died_since_prev: int
    birth_rate: float  # 上一观测点以来的出生数 / span_ticks
    death_rate: float
    stats: GenerationStats

    def flatten(self) -> dict:
        """扁平化为一行（键即 CSV 列名）。"""
        d = self.stats.to_dict()
        trait_means: dict[str, float] = d.pop("trait_means")
        trait_stds: dict[str, float] = d.pop("trait_stds")
        d.pop("generation_histogram")  # 直方图只进 JSON，不进 CSV
        row: dict = {
            "tick": self.tick,
            "generation": self.generation,
            "span_ticks": self.span_ticks,
            "born_since_prev": self.born_since_prev,
            "died_since_prev": self.died_since_prev,
            "birth_rate": self.birth_rate,
            "death_rate": self.death_rate,
        }
        # 每性状两列：trait_<name>_mean / trait_<name>_std，按字母序固定列序
        for name in sorted(trait_means):
            row[f"trait_{name}_mean"] = trait_means[name]
            row[f"trait_{name}_std"] = trait_stds[name]
        row.update(d)
        return row

    def to_dict(self) -> dict:
        d = self.flatten()
        d["generation_histogram"] = self.stats.generation_histogram
        return d


class EvolutionObserver:
    """挂在一台球面引擎上，按世代/节拍采集 GenerationSample。

    窗口聚合不读引擎历史序列：出生/死亡计数由调用方逐 tick 喂入
    （observe(stats)），观察者内部只做整数累积 → 引擎可用有界历史
    模式跑百万 tick 级实验，观察者不依赖被裁剪的历史（解耦）。
    """

    def __init__(self, engine: SphereEngine, tick_interval: int = 100) -> None:
        if tick_interval < 0:
            raise ValueError(f"tick_interval 不能为负：{tick_interval}")
        self.engine = engine
        self.tick_interval = tick_interval
        self._samples: list[GenerationSample] = []
        self._last_max_gen = 0
        self._prev_tick = 0
        self._window_born = 0
        self._window_died = 0

    # ---- 只读 ----------------------------------------------------------

    @property
    def samples(self) -> list[GenerationSample]:
        return self._samples

    # ---- 采样 ----------------------------------------------------------

    def observe(self, tick_stats: TickStats | None = None) -> None:
        """推进一次观察；调用方在引擎每 tick 之后调用一次。

        tick_stats 为可选的该 tick 统计（供窗口聚合）。
        """
        eng = self.engine
        if tick_stats is not None:
            self._window_born += tick_stats.born
            self._window_died += tick_stats.died
        gen = eng._max_generation  # 球面引擎无 population 门面，直接读世代号
        generation_advanced = gen > self._last_max_gen
        tick_due = (
            self.tick_interval > 0
            and eng.tick - self._prev_tick >= self.tick_interval
        )
        if not (generation_advanced or tick_due):
            return

        # 窗口聚合：自上一采样点以来的出生/死亡
        born, died = self._window_born, self._window_died
        span = max(1, eng.tick - self._prev_tick)

        stats = generation_statistics(eng)
        sample = GenerationSample(
            tick=eng.tick,
            generation=gen,
            span_ticks=span,
            born_since_prev=born,
            died_since_prev=died,
            birth_rate=born / span,
            death_rate=died / span,
            stats=stats,
        )
        self._samples.append(sample)
        self._last_max_gen = gen
        self._prev_tick = eng.tick
        self._window_born = 0
        self._window_died = 0