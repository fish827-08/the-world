"""tests.test_bio_b16：B16 捕食-猎物军备竞赛（关档等价 / oracle / 并行 / 度量）。

TIER = "mainline"

钉住四件事（B16 硬约束）：

1. **默认关 ⇒ 逐位等价**：关档 300 tick 的流水指纹等于 B16 接线前的
   黄金指纹（1000 tick 等价亦已实测：见汇报 B16.0）；
2. **oracle 是"钉死"不是"偏置"**：分组规则确定性、极值真的落到控制器、
   且**进决策**（高攻组攻击次数显著高于低攻组）；
3. **收益通道 loot 只有致死才有**，负对照（loot=0）必须为零；
4. **并行不改变结果**：进程池结果与串行逐位一致（slow：spawn 开销）。
"""
from __future__ import annotations

import numpy as np
import pytest

from biosphere.develop import develop
from biosphere.experiments import B16_RACE_ARMS, build_b16_config
from biosphere.metrics import bimodality_coefficient, predation_json, run_digest
from biosphere.parallel import JobSpec, canonical_sha, run_parallel, run_serial
from biosphere.predation import (
    PredationConfig, apply_oracle_traits, oracle_group,
    spread_founder_traits,
)
from biosphere.simulator import BioMVPConfig, BiosphereSimulator

TIER = "mainline"

#: 关档黄金指纹（300 tick；B16 接线前的口径，接线后重跑仍须逐位相同）
GOLDEN_DIGEST_300 = {
    42: "0503de6c4700ef294efa83f5d48bf76d3fd395359282c11cce43cf7f4c1ffc0c",
    43: "02e8e55345de8bd02fcf62e1bfd0bce36ac70b1f1d0d841cb86ffbbf813ad025",
}


def _run(cfg: BioMVPConfig):
    sim = BiosphereSimulator(cfg)
    return sim, sim.run()


def test_default_off_digest_matches_golden():
    """默认关 ⇒ 与 B16 接线前逐位一致（黄金指纹）。"""
    for seed, golden in GOLDEN_DIGEST_300.items():
        _sim, log = _run(BioMVPConfig(ticks=300, seed=seed))
        assert run_digest(log) == golden, (
            f"seed {seed} 关档指纹变了 ⇒ 默认路径被 B16 污染")


def test_explicit_disabled_equals_default():
    """显式 PredationConfig(enabled=False) 与不传 ⇒ 同一指纹。"""
    _s1, log1 = _run(BioMVPConfig(ticks=300, seed=42))
    _s2, log2 = _run(BioMVPConfig(ticks=300, seed=42,
                                  predation=PredationConfig()))
    assert run_digest(log1) == run_digest(log2)


def test_predation_on_records_trait_frames():
    """开档 ⇒ 每帧带 defense/attack_drive，度量段 enabled。"""
    _sim, log = _run(build_b16_config("b16_full", seed=42, ticks=300))
    assert all(f.defense is not None for f in log.frames)
    pred = predation_json(log)
    assert pred["enabled"] is True
    assert pred["trajectory"][0]["tick"] == 0
    assert 0.0 <= pred["final"]["defense_mean"] <= 1.0


def test_oracle_group_is_deterministic():
    assert [oracle_group(u) for u in range(6)] == [0, 1, 0, 1, 0, 1]
    assert oracle_group(12345) == oracle_group(12345)


def test_oracle_pins_extremes_and_reaches_decisions():
    """oracle_attack：高组攻击次数必须显著高于低组（证明覆写进了决策）。"""
    _sim, log = _run(build_b16_config("b16_oracle_attack", seed=42,
                                      ticks=600))
    hi = lo = 0
    for e in log.events:
        if e.kind != "attack":
            continue
        if oracle_group(e.actor_id) == 1:
            hi += 1
        else:
            lo += 1
    assert hi > 2 * max(lo, 1), f"高攻组没真的更爱打：hi={hi} lo={lo}"


def test_oracle_defense_pins_controller_values():
    """oracle_defense：控制器副本里 defense ∈ {0,1} 且按 uid 奇偶。"""
    cfg = build_b16_config("b16_oracle_defense", seed=42, ticks=200)
    sim = BiosphereSimulator(cfg)
    ctrl = sim._controllers()
    assert ctrl, "种群为空"
    for uid, c in ctrl.items():
        assert c.defense == (1.0 if oracle_group(uid) == 1 else 0.0)
    # 钉死值不能污染发育缓存（缓存里仍是基因版）
    cached = sim.cache.get(sim.population.individuals[0].genome)
    assert 0.0 <= cached.defense <= 1.0


def test_apply_oracle_traits_noop_when_off():
    """两个 oracle 都关 ⇒ 原样返回（同一对象，不拷贝）。"""
    from biosphere.develop import ControllerParams
    fields = [k for k in ControllerParams.__dataclass_fields__]
    c0 = ControllerParams(**{k: (0.0,) * 4 if k == "response" else 0.5
                             for k in fields})
    out = apply_oracle_traits({0: c0, 1: c0}, PredationConfig())
    assert out[0] is c0 and out[1] is c0


def test_loot_only_on_kill_and_negative_control_zero():
    """loot 只在致死事件上为正；no_loot 臂必须全零。"""
    _sim, log = _run(build_b16_config("b16_full", seed=42, ticks=400))
    killed = [e for e in log.events if e.kind == "attack" and e.status == "ok"]
    other = [e for e in log.events if e.kind == "attack" and e.status != "ok"]
    assert killed, "400 tick 内应至少有一次致死掠食"
    assert all(e.loot > 0.0 for e in killed)
    assert all(e.loot == 0.0 for e in other)

    _sim2, log2 = _run(build_b16_config("b16_no_loot", seed=42, ticks=400))
    assert predation_json(log2)["attacks"]["loot_total"] == 0.0


def test_init_trait_spread_changes_founders_only_when_on():
    """方差注入默认关（⇒ 与旧口径同分布）；开时初代性状方差变大。"""
    import numpy as np
    from biosphere.genome import random_genome
    rng = np.random.default_rng(7)
    base = [develop(random_genome(rng, uid=i)).defense for i in range(40)]
    rng2 = np.random.default_rng(7)
    spread = [develop(spread_founder_traits(
        random_genome(rng2, uid=i), rng2,
        PredationConfig(init_trait_spread=2.0))).defense for i in range(40)]
    assert np.var(spread) > np.var(base)


def test_bimodality_coefficient_separates_single_and_twin_peaks():
    rng = np.random.default_rng(3)
    single = rng.normal(0.5, 0.05, size=400)
    twin = np.concatenate([rng.normal(0.15, 0.03, size=200),
                           rng.normal(0.85, 0.03, size=200)])
    assert bimodality_coefficient(single) < 0.555
    assert bimodality_coefficient(twin) > 0.555


def test_race_arms_defined_with_negative_controls():
    assert "b16_no_loot" in B16_RACE_ARMS and "b16_no_defense" in B16_RACE_ARMS
    assert B16_RACE_ARMS["b16_no_loot"]["resolve"]["attack_loot_frac"] == 0.0
    assert B16_RACE_ARMS["b16_no_defense"]["resolve"]["defense_mit"] == 0.0


@pytest.mark.slow
def test_parallel_matches_serial_bitwise():
    """进程池并行 ⇒ 与串行结果逐位一致（spawn 起进程，故标 slow）。"""
    jobs = [JobSpec(arm="b16_natural", seed=41 + i, ticks=200)
            for i in range(3)]
    ser = run_serial(jobs)
    par = run_parallel(jobs, workers=3)
    assert [r["ok"] for r in par] == [True] * 3
    assert canonical_sha([r["row"] for r in ser]) == \
        canonical_sha([r["row"] for r in par])
