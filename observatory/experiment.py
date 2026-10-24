"""ExperimentRunner：确定性实验调度与结果聚合（模块四 · 球面引擎适配版）。

实验固定契约：
- 每个实验 = (ExperimentSpec, 派生 seed)。种子由
  SeedSequence([base_seed, group_id, seed_index]) 确定性派生 →
  同一命令行参数必然复现同一批结果（deterministic experiment support）。
- 运行语义：engine 每 tick 推进，observer 每 tick 观察；停止条件为
  三者任一：达到 max_generations（世代达标）/ 引擎自身结束（跑满
  tick 或灭绝防护）/ 达到 max_ticks 硬上限（防失控）。
- runner 不引入任何"适应度"概念：演化方向完全由既有生命规则涌现。

与旧版（digital_life/observatory/experiment.py）的差异
-----------------------------------------------------
- 配置构造改为球面 SimConfig（嵌套 dataclass）：overrides 按子配置
  分组（如 {"resources": {"regrowth_rate": ...}}）合并进默认值；
- 引擎只保留 SphereEngine 一个实现（数组化 SoA），use_vec 取消、
  由 use_sim_core（Rust 数值管线下沉）替代；
- 资源分布组只保留球面资源场支持的均匀形态（sparse/rich）；
  斑块形态（patchiness）在球面引擎未实现前不预设实验（诚实标注）。
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from observatory.observer import EvolutionObserver, GenerationSample
from observatory.statistics import TRAIT_ORDER
from simulation.config import (
    GenomeConfig,
    LightConfig,
    OrganismConfig,
    PopulationConfig,
    ResourceConfig,
    SimConfig,
    SimulationConfig,
    WorldConfig,
)
from simulation.sphere_engine import SphereEngine

# 硬上限：单个实验最长 tick 数（60×120 下约数小时；防参数失误失控）
DEFAULT_MAX_TICKS = 1_200_000

# 顶层配置的子配置类型表（overrides 分组 → 对应 dataclass）
_SUB_CONFIGS = {
    "world": WorldConfig,
    "light": LightConfig,
    "resources": ResourceConfig,
    "organisms": OrganismConfig,
    "genome": GenomeConfig,
    "population": PopulationConfig,
    "simulation": SimulationConfig,
}


@dataclass(frozen=True)
class ExperimentSpec:
    """一份实验规格（可复现的最小单位）。"""

    name: str
    group: str  # baseline / resource_pressure / resource_distribution / mutation_rate / repeated_seeds
    description: str
    overrides: dict = field(default_factory=dict)  # SimConfig 分组覆盖（如 {"resources": {...}}）
    max_generations: int = 10_000
    max_ticks: int = DEFAULT_MAX_TICKS
    seed_index: int = 0  # 组内序号（repeated seeds 用）

    @property
    def group_id(self) -> int:
        return _GROUP_IDS[self.group]


# 组 → 稳定整数 id（种子派生用；新增组需登记）
_GROUP_IDS = {
    "baseline": 0,
    "resource_pressure": 1,
    "resource_distribution": 2,
    "mutation_rate": 3,
    "repeated_seeds": 4,
}


def derive_seed(base_seed: int, group_id: int, seed_index: int) -> int:
    """确定性派生一个实验种子（与运行次数无关）。"""
    return int(np.random.SeedSequence([base_seed, group_id, seed_index]).generate_state(1)[0])


@dataclass(frozen=True)
class ExperimentRun:
    """一个具体运行实例 = spec + 派生种子。"""

    spec: ExperimentSpec
    seed: int


@dataclass
class ExperimentResult:
    """一次运行的完整结果（可持久化）。"""

    run: ExperimentRun
    config: SimConfig
    finished_normally: bool  # True = 世代达标
    ended_reason: str  # "generations_reached" / "engine_finished" / "max_ticks"
    extinct: bool
    total_ticks: int
    final_population: int
    duration_s: float
    samples: list[GenerationSample] = field(default_factory=list)
    totals: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)


# ---- 预置实验矩阵（5 类，与 Stage 2 对齐） ---------------------------------


def build_plan(
    base_seed: int = 42,
    max_generations: int = 1_000,
    short_generations: int = 250,
    repeated_seed_count: int = 3,
    repeated_generations: int = 250,
    baseline_max_ticks: int | None = None,
) -> list[ExperimentRun]:
    """构建完整实验计划。

    规模语义（世代深度由生态物理决定）：
    - baseline 是唯一的长程实验。默认精简约 1,000 代（60×120 下约
      40-60 分钟），用于验证"跨千代的选择性变化"；其 tick 硬上限
      由 max_generations 自动推导（~1,800 tick/代 × 安全系数），
      也可用 baseline_max_ticks 显式覆盖；
    - 完整能力档（10,000 代）通过 max_generations=10_000 显式启用；
    - 其余对照实验跑 short_generations 代（默认 250 代）；
    - repeated_seeds 组同配置多种子跑 repeated_generations 代，
      用于跨 seed 方差分析。

    注：球面资源场（模块一）只支持均匀填充，暂不支持斑块分布，
    故 resource_distribution 组只保留 sparse/rich 两个均匀对照。
    """
    if baseline_max_ticks is None:
        baseline_max_ticks = max(DEFAULT_MAX_TICKS, int(max_generations * 1_800))
    specs: list[ExperimentSpec] = [
        # Baseline：默认生态（对照组，完整长程）
        ExperimentSpec(
            "baseline", "baseline",
            "默认生态参数（对照组）：均匀资源、默认变异率", {},
            max_generations=max_generations,
            max_ticks=baseline_max_ticks,
        ),
        # Resource Pressure：资源再生压力（短程对照）
        ExperimentSpec(
            "pressure_low", "resource_pressure",
            "资源再生率降至默认 100 分之一（食物供给收紧）",
            {"resources": {"regrowth_rate": 0.005}},
            max_generations=short_generations,
        ),
        ExperimentSpec(
            "pressure_critical", "resource_pressure",
            "资源再生率近零（慢性饥饿，检验灭绝动力学）",
            {"resources": {"regrowth_rate": 0.001}},
            max_generations=short_generations,
        ),
        # Resource Distribution：资源丰度（球面仅均匀形态，短程对照）
        ExperimentSpec(
            "dist_uniform_sparse", "resource_distribution",
            "初始资源稀疏但均匀（总供给偏紧）",
            {"resources": {"initial_fill": 0.08}},
            max_generations=short_generations,
        ),
        ExperimentSpec(
            "dist_uniform_rich", "resource_distribution",
            "初始资源富饶且均匀",
            {"resources": {"initial_fill": 0.8}},
            max_generations=short_generations,
        ),
        # Mutation Rate：变异强度（短程对照）
        ExperimentSpec(
            "mutation_low", "mutation_rate",
            "变异率降为默认 1/10",
            {"genome": {"mutation_rate": 0.005}},
            max_generations=short_generations,
        ),
        ExperimentSpec(
            "mutation_high", "mutation_rate",
            "变异率升为默认 6 倍",
            {"genome": {"mutation_rate": 0.3}},
            max_generations=short_generations,
        ),
    ]
    runs = [ExperimentRun(spec=spec, seed=derive_seed(base_seed, spec.group_id, spec.seed_index)) for spec in specs]
    # Repeated Seeds：同 baseline 配置多种子（跨 seed 方差分析）
    for i in range(repeated_seed_count):
        spec = ExperimentSpec(
            f"repeated_seed_{i + 1}", "repeated_seeds",
            f"baseline 配置 × 种子变体 {i + 1}（跨 seed 可重复性）", {},
            max_generations=repeated_generations,
            seed_index=i,
        )
        runs.append(ExperimentRun(spec=spec, seed=derive_seed(base_seed, spec.group_id, i)))
    return runs


# ---- 配置装配 --------------------------------------------------------------


def build_config(run: ExperimentRun, world_size: tuple[int, int]) -> SimConfig:
    """组装一份 SimConfig：种子 + 世界尺寸 + 有界历史 + overrides 合并。

    overrides 是按子配置分组的浅覆盖（如 {"genome": {"mutation_rate": v}}），
    未覆盖的键保留默认值；"simulation"/"world" 仅显式提供时覆盖。
    """
    cfg = SimConfig(
        seed=run.seed,
        world=WorldConfig(rows=world_size[0], cols=world_size[1]),
        simulation=SimulationConfig(
            ticks=max(run.spec.max_ticks, 1),
            stop_on_extinction=True,
            # 有界历史：观察者从 per-tick 统计累积窗口，不依赖全量历史
            # → 百万 tick 级长程实验内存有上界（避免 OOM）。
            history_limit=4_096,
        ),
    )
    for key, patch in run.spec.overrides.items():
        sub = _SUB_CONFIGS[key]
        merged = dict(asdict(getattr(cfg, key)))
        merged.update(patch)
        setattr(cfg, key, sub(**merged))
    return cfg


# ---- Runner ---------------------------------------------------------------


class ExperimentRunner:
    """按计划跑实验，产出可持久化的 ExperimentResult。"""

    def __init__(
        self,
        base_seed: int = 42,
        world_size: tuple[int, int] = (60, 120),
        tick_interval: int = 100,
        use_sim_core: bool = False,
    ) -> None:
        self.base_seed = base_seed
        self.world_size = world_size
        self.tick_interval = tick_interval
        self.use_sim_core = use_sim_core  # True = 种群数值管线走 Rust（sim_core）

    # ---- 计划 --------------------------------------------------------

    def plan(self, **kwargs) -> list[ExperimentRun]:
        return build_plan(base_seed=self.base_seed, **kwargs)

    # ---- 单跑 ---------------------------------------------------------

    def run_single(self, run: ExperimentRun) -> ExperimentResult:
        cfg = build_config(run, self.world_size)
        if self.use_sim_core:
            cfg.simulation.use_sim_core = True  # 引擎侧开关（等效于 override）
        engine = SphereEngine(cfg)
        observer = EvolutionObserver(engine, tick_interval=self.tick_interval)

        target = run.spec.max_generations
        cap = run.spec.max_ticks
        t0 = time.perf_counter()
        while (
            not engine.finished
            and engine.tick < cap
            and engine._max_generation < target
        ):
            stats = engine.step()
            observer.observe(stats)
            # 与 engine.run() 对齐：每 tick 后按终止条件置位（灭绝/跑满）
            engine._finished = engine._end_condition_met()
        duration = time.perf_counter() - t0

        if engine._max_generation >= target:
            ended_reason = "generations_reached"
            finished_normally = True
        elif engine.finished:
            ended_reason = "engine_finished"
            finished_normally = False
        else:
            ended_reason = "max_ticks"
            finished_normally = False

        return ExperimentResult(
            run=run,
            config=cfg,
            finished_normally=finished_normally,
            ended_reason=ended_reason,
            extinct=engine.extinct,
            total_ticks=engine.tick,
            final_population=engine.alive_count(),
            duration_s=duration,
            samples=observer.samples,
            totals=_totals(engine),
            summary=_summarize(run, engine, observer),
        )

    # ---- 批量 -----------------------------------------------------------

    def run_plan(
        self,
        runs: list[ExperimentRun],
        only: Optional[list[str]] = None,
        quiet: bool = False,
        out_dir: str | Path | None = None,
    ) -> dict[str, ExperimentResult]:
        """批量运行；out_dir 给定时，每个实验完成后立即落盘（可中途停止不丢结果）。"""
        results: dict[str, ExperimentResult] = {}
        selected = [r for r in runs if only is None or r.spec.name in only]
        for i, run in enumerate(selected, 1):
            if not quiet:
                print(
                    f"[{i}/{len(selected)}] {run.spec.name} "
                    f"(seed={run.seed}, {run.spec.description})",
                    flush=True,
                )
            res = self.run_single(run)
            results[run.spec.name] = res
            if out_dir is not None:
                # 即时持久化：跑到一半就停也不丢已完成实验
                from persistence.io import save_experiment

                save_experiment(res, Path(out_dir) / run.spec.name)
            if not quiet:
                status = "达标" if res.finished_normally else res.ended_reason
                print(
                    f"     -> 世代={res.summary.get('generations_reached', 0)}, "
                    f"tick={res.total_ticks}, 种群={res.final_population}, "
                    f"状态={status}, 耗时={res.duration_s:.0f}s",
                    flush=True,
                )
        return results


# ---- 结果加工 --------------------------------------------------------------


def _totals(engine: SphereEngine) -> dict:
    view = engine.population_view()
    return {
        "total_born": engine.total_born,
        "total_died": engine.total_died,
        "deaths_by_cause": {
            (k.value if hasattr(k, "value") else str(k)): v
            for k, v in engine.death_cause_totals().items()
        },
        "total_energy": view.total_energy,
        "total_resource": engine.resources.total(),
    }


def _summarize(run: ExperimentRun, engine: SphereEngine, observer: EvolutionObserver) -> dict:
    """观测点首/末/中 + 关键趋势（只聚合，不做价值判断）。"""
    samples = observer.samples
    if not samples:
        return {"generations_reached": engine._max_generation}
    first, last = samples[0], samples[-1]
    mid = samples[len(samples) // 2]
    s_f, s_m, s_l = first.stats, mid.stats, last.stats
    trait_drift = {
        t: {
            "start": s_f.trait_means[t],
            "mid": s_m.trait_means[t],
            "end": s_l.trait_means[t],
            "drift": s_l.trait_means[t] - s_f.trait_means[t],
        }
        for t in TRAIT_ORDER
    }
    return {
        "generations_reached": last.generation,
        "first_tick": first.tick,
        "last_tick": last.tick,
        "final_population": last.stats.population,
        "population_series": [s_f.population, s_m.population, s_l.population],
        "diversity": {
            "start": s_f.genome_diversity,
            "mid": s_m.genome_diversity,
            "end": s_l.genome_diversity,
            "drift": s_l.genome_diversity - s_f.genome_diversity,
        },
        "mean_life_span": {
            "start": s_f.trait_means["life_span"],
            "end": s_l.trait_means["life_span"],
        },
        "trait_drift": trait_drift,
        "final_birth_rate": last.birth_rate,
        "final_death_rate": last.death_rate,
    }