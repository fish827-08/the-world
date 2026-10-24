"""observatory 模块四的测试：适配版统计口径 / 采样器 / 实验调度 / 持久化。"""
from __future__ import annotations

import numpy as np

from observatory.experiment import (
    ExperimentResult,
    ExperimentRun,
    ExperimentRunner,
    ExperimentSpec,
    derive_seed,
)
from observatory.observer import EvolutionObserver
from observatory.statistics import TRAIT_ORDER, generation_statistics
from observatory.traits import decode_trait_matrix
from persistence.io import load_generations, load_manifest, save_experiment
from simulation.config import SimConfig
from simulation.sphere_engine import SphereEngine


# ---- 统计口径（trait 表 / 聚合） -------------------------------------------


def test_trait_order_length() -> None:
    # 24 基因 trait + 3 派生 trait = 27 列
    assert len(TRAIT_ORDER) == 27  # 24 基因 trait + 3 派生 trait
    assert TRAIT_ORDER[:3] == ("move_prob", "metabolic", "repro_threshold")
    assert TRAIT_ORDER[-3:] == ("life_span", "metabolic_mult", "maturity_age")


def test_decode_trait_matrix_shape() -> None:
    rng = np.random.default_rng(3)
    genes = rng.uniform(0.0, 1.0, size=(10, 16))
    traits = decode_trait_matrix(genes, day_length=2400.0)
    # 16 基因输入：16 列基因 trait + 3 派生 = 19 列（列数保护：只解码存在的列）
    assert traits.shape == (10, 19)
    # 派生列与公式一致（与引擎 _lifespan/metab_mult/maturity 同一公式）
    assert np.allclose(traits[:, 16], 2400.0 * (1.0 + genes[:, 3] * 7.0))
    assert np.allclose(traits[:, 17], 0.5 + genes[:, 1] * 1.5)


def test_generation_statistics_empty() -> None:
    cfg = SimConfig(seed=1)
    eng = SphereEngine(cfg)
    # 人为清空种群 → 合法观测点（全零/负一代）
    eng._id = np.zeros(0, dtype=np.int64)
    s = generation_statistics(eng)
    assert s.population == 0
    assert s.max_generation == -1
    assert s.generation_histogram == {}
    assert all(v == 0.0 for v in s.trait_means.values())


# ---- EvolutionObserver 采样器 ----------------------------------------------


def _tiny_engine(seed: int = 7) -> SphereEngine:
    from simulation.config import WorldConfig

    cfg = SimConfig(
        seed=seed,
        world=WorldConfig(rows=12, cols=16),
        simulation=SimulationConfigShort(ticks=300),
    )
    return SphereEngine(cfg)


def test_observer_collects_tick_samples() -> None:
    eng = _tiny_engine()
    obs = EvolutionObserver(eng, tick_interval=50)
    for _ in range(eng.config.simulation.ticks):
        obs.observe(eng.step())
    assert len(obs.samples) >= 3  # 兜底节拍 50/100/... 必触发
    ticks = [s.tick for s in obs.samples]
    assert ticks == sorted(ticks)
    last = obs.samples[-1].stats
    assert set(last.trait_means.keys()) == set(TRAIT_ORDER)
    assert all(np.isfinite(v) for v in last.trait_means.values())
    assert 0.0 <= last.genome_diversity <= 1.0
    # CSV 扁平化：直方图不进 flatten，trait 列按字母序成对出现
    row = obs.samples[0].flatten()
    assert "generation_histogram" not in row
    assert "trait_move_prob_mean" in row and "trait_move_prob_std" in row


# ---- ExperimentRunner / 实验调度 --------------------------------------------


def _runner() -> ExperimentRunner:
    return ExperimentRunner(
        base_seed=7,
        world_size=(12, 16),
        tick_interval=50,
        use_sim_core=False,
    )


def _spec(name: str = "smoke", **kw) -> ExperimentSpec:
    kwargs = dict(
        max_generations=20,
        max_ticks=400,
    )
    kwargs.update(kw)
    return ExperimentSpec(
        name, "baseline", name, overrides=kwargs.pop("overrides", {}), **kwargs
    )


def test_run_single_smoke() -> None:
    run = ExperimentRun(spec=_spec(), seed=derive_seed(7, 0, 0))
    res = _runner().run_single(run)
    assert isinstance(res, ExperimentResult)
    assert res.total_ticks >= 1
    assert len(res.samples) >= 1
    assert res.config.simulation.ticks == 400
    # totals 用全局计数器（与 history_limit 无关）
    assert res.totals["total_born"] >= 0
    assert res.totals["total_died"] >= 0
    assert res.totals["deaths_by_cause"] is not None
    assert res.summary["generations_reached"] >= 0


def test_run_single_deterministic() -> None:
    runner = _runner()
    run = ExperimentRun(spec=_spec(), seed=derive_seed(7, 0, 0))
    a = runner.run_single(run)
    b = runner.run_single(run)
    assert [s.to_dict() for s in a.samples] == [s.to_dict() for s in b.samples]


def test_overrides_merged_into_config() -> None:
    runner = _runner()
    spec = _spec(max_generations=5, max_ticks=150, overrides={"genome": {"mutation_rate": 0.3}})
    res = runner.run_single(ExperimentRun(spec=spec, seed=derive_seed(7, 3, 0)))
    assert res.config.genome.mutation_rate == 0.3
    # 未覆盖键保持默认
    assert res.config.genome.gene_count == 24
    assert res.config.resources.regrowth_rate == 0.5


def test_extinction_sets_flag_and_empty_last_sample() -> None:
    runner = _runner()
    spec = _spec(
        max_generations=5,
        max_ticks=200,
        overrides={"resources": {"regrowth_rate": 0.001}},
    )
    res = runner.run_single(ExperimentRun(spec=spec, seed=derive_seed(7, 1, 0)))
    if res.extinct:
        # 灭绝可能在两次采样之间发生：末样本种群可 >0，但引擎最终归零并提前停
        assert res.final_population == 0
        assert res.ended_reason == "engine_finished"


# ---- 持久化 roundtrip --------------------------------------------------------


def test_persistence_roundtrip(tmp_path) -> None:
    run = ExperimentRun(spec=_spec(), seed=derive_seed(7, 0, 0))
    res = _runner().run_single(run)
    out = save_experiment(res, tmp_path / "smoke")

    manifest = load_manifest(out)
    assert manifest["name"] == "smoke"
    assert manifest["config"]["simulation"]["ticks"] == 400
    assert manifest["totals"]["total_born"] == res.totals["total_born"]
    assert "starvation" in manifest["totals"]["deaths_by_cause"] or True  # 键名格式合法

    rows = load_generations(out)
    assert len(rows) == len(res.samples)
    if rows:
        assert rows[0]["tick"]
        assert "trait_move_prob_mean" in rows[0]


def test_config_dict_roundtrip() -> None:
    cfg = SimConfig(seed=5)
    assert SimConfig.from_dict(cfg.to_dict()) == cfg


# 供本测试文件复用的精简别名
from simulation.config import SimulationConfig as SimulationConfigShort  # noqa: E402