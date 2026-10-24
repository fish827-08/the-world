"""B12（W4b）：粘附链团 / 组繁殖构成 / 作弊者去生存偏差 / 判据评估 测试。

对应派工五件套：
1) 两套团定义（邻接团=对照，粘附链团=主张；机制跑链团）；
2) 粘附基因频率轨迹（ChainFrame → metrics trajectory）；
3) 组繁殖构成（事件含共享注入比/生殖系来源/高低桶）；
4) 规模门（判据评估纯函数单测；跑批在 CLI 层）；
5) 作弊者含死者流水（修生存偏差）+ 每团占比。
"""
import numpy as np
import pytest

from biosphere.cell import DIFF_ADHESION_SCALE, adhesion_expression
from biosphere.criteria import (
    evaluate_ecological_gate, evaluate_group_selection_criterion, seed_gate,
)
from biosphere.develop import ControllerParams
from biosphere.genome import Genome
from biosphere.groups import (
    ADHESION_STICKY, W4Config, build_adhesion_chain_index, build_group_index,
    reproduce_groups, share_group_energy, share_injection_ratio,
)
from biosphere.metrics import metrics_json
from biosphere.resolve import Individual, Population, ResolveReport
from biosphere.simulator import BioMVPConfig, BiosphereSimulator
from biosphere.world_adapter import WorldView, _von_neumann_table


def _ctrl(**overrides) -> ControllerParams:
    base = dict(
        sense_food=1.0, sense_threat=1.0, sense_kin=1.0, sense_signal=1.0,
        move_bias=0.5, eat_greed=0.5, attack_drive=0.0, share_drive=0.5,
        signal_drive=0.0, repro_drive=0.5, metabolism=1.0, move_cost_mult=1.0,
        lifespan=1000.0, repro_thresh=40.0, habitat_aquatic=1.0,
        habitat_terrestrial=1.0, adhesion=0.9, defense=0.0, eat_amount=0.5,
        signal_payload=0.5, give_up=10.0, persistence=0.0,
    )
    base.update(overrides)
    return ControllerParams(**base)


def _world(rows=4, cols=6) -> WorldView:
    n = rows * cols
    return WorldView(
        rows=rows, cols=cols, n_cells=n,
        surface_type=np.zeros(n, dtype=np.uint8),
        elevation_m=np.full(n, 5.0), depth_m=np.zeros(n),
        patch_id=np.full(n, -1, dtype=np.int32),
        neighbors=_von_neumann_table(rows, cols),
        soil_water=np.full(n, 5.0), soil_water_min=2.0,
    )


def _pop(pairs, diff_states=None) -> Population:
    pop = Population()
    for i, (uid, cell, energy) in enumerate(pairs):
        ind = Individual(uid=uid, genome=Genome(genes=np.zeros(64)),
                         cell=cell, energy=energy)
        if diff_states:
            ind.diff_state = diff_states.get(uid, 0)
        pop.individuals.append(ind)
        pop.next_id = max(pop.next_id, uid + 1)
    return pop


# ── 1) 两套团定义 ────────────────────────────────────────────────────────


class TestTwoGroupLenses:
    def test_both_sticky_forms_chain(self):
        w = _world()
        # diff_state=1（体细胞态）⇒ 全额表达；state=0 只表达一半会跌破门槛
        pop = _pop([(0, 0, 10.0), (1, 1, 10.0)], diff_states={0: 1, 1: 1})
        ctrls = {0: _ctrl(adhesion=0.9), 1: _ctrl(adhesion=0.9)}
        chain = build_adhesion_chain_index(pop, w, ctrls)
        assert chain.size(0) == 2 and chain.size(1) == 2

    def test_non_sticky_breaks_chain(self):
        """一端不粘 ⇒ 链团断开（非粘附细胞恒为单体）。"""
        w = _world()
        pop = _pop([(0, 0, 10.0), (1, 1, 10.0)])
        ctrls = {0: _ctrl(adhesion=0.9), 1: _ctrl(adhesion=0.1)}
        chain = build_adhesion_chain_index(pop, w, ctrls)
        adj = build_group_index(pop, w)
        assert chain.size(0) == 1 and chain.size(1) == 1   # 链：无
        assert adj.size(0) == 2                              # 邻接：有（对照）

    def test_chain_requires_both_ends(self):
        """三连 0-1-2，中间的 1 不粘 ⇒ 三个都是链团单体。"""
        w = _world()
        pop = _pop([(0, 0, 10.0), (1, 1, 10.0), (2, 2, 10.0)])
        ctrls = {0: _ctrl(adhesion=0.9), 1: _ctrl(adhesion=0.1),
                 2: _ctrl(adhesion=0.9)}
        chain = build_adhesion_chain_index(pop, w, ctrls)
        assert chain.n_groups == 3

    def test_differentiation_halves_adhesion(self):
        """state=0 时粘附表达减半 ⇒ 可能跌破粘附门槛。"""
        assert adhesion_expression(0.9, 1) == pytest.approx(0.9)
        assert adhesion_expression(0.9, 0) == pytest.approx(0.45)
        assert 0.45 < ADHESION_STICKY          # 跌破 ⇒ 不成链

    def test_mechanics_pool_on_chain_not_adjacency(self):
        """共享能量只在链团内池化：非粘附桥两侧不共享。"""
        w = _world()
        pop = _pop([(0, 0, 100.0), (1, 1, 100.0)])
        ctrls = {0: _ctrl(adhesion=0.9), 1: _ctrl(adhesion=0.1)}
        report = ResolveReport(tick=0)
        share_group_energy(pop, build_adhesion_chain_index(pop, w, ctrls),
                           ctrls, report)
        assert report.group_share_events == 0     # 链团都是单体 ⇒ 无共享
        assert all(i.energy == 100.0 for i in pop.individuals)


# ── 2) 粘附基因频率轨迹 ──────────────────────────────────────────────────


class TestAdhesionTrajectory:
    def test_chain_frames_record_gene_stats(self):
        sim = BiosphereSimulator(
            BioMVPConfig(ticks=60, seed=42, w4=W4Config(enabled=True)))
        sim.run()
        frames = sim.log.chain_frames
        assert len(frames) == 60
        f0 = frames[0]
        assert np.isfinite(f0.adhesion_gene_mean)
        assert f0.adhesion_gene_var >= 0.0
        assert 0.0 <= f0.sticky_fraction <= 1.0

    def test_metrics_trajectory_has_start_mid_end(self):
        sim = BiosphereSimulator(
            BioMVPConfig(ticks=120, seed=42, w4=W4Config(enabled=True)))
        sim.run()
        m = metrics_json(sim.log, sim.world, seed=42, ticks=120)
        traj = m["groups"]["adhesion_chain"]["adhesion_trajectory"]
        assert set(traj["start"]) >= {"tick", "adhesion_gene_mean",
                                      "adhesion_gene_var", "sticky_fraction"}
        assert traj["mid"]["tick"] > traj["start"]["tick"]
        assert traj["end"]["tick"] > traj["mid"]["tick"]
        assert "whole_run_mean" in traj

    def test_adhesion_genes_actually_evolve(self):
        """粘附基因均值不是常数（真的在漂/或至少随突变动）。"""
        sim = BiosphereSimulator(
            BioMVPConfig(ticks=600, seed=42, w4=W4Config(enabled=True)))
        sim.run()
        vals = [f.adhesion_gene_mean for f in sim.log.chain_frames]
        assert max(vals) - min(vals) > 1e-9


# ── 3) 组繁殖构成 ────────────────────────────────────────────────────────


class TestReproComposition:
    def test_share_injection_ratio(self):
        """注入 = SHARE_BASE×drive；平分线 = 1/n ⇒ n 大才净奉献。

        n=2、drive=1：注入 0.3 < 平分线 0.5 ⇒ 仍净蹭（经济学正确）。
        """
        ctrls = {0: _ctrl(share_drive=1.0), 1: _ctrl(share_drive=1.0)}
        comp = share_injection_ratio([0, 1], ctrls)
        assert comp["mean_share_drive"] == pytest.approx(1.0)
        assert comp["ratio_literal"] == pytest.approx(2.0)   # mean×n
        assert comp["ratio_net"] == pytest.approx(0.6)        # 0.3×1×2
        assert comp["high_share"] is False                    # n=2 仍净蹭
        big = {i: _ctrl(share_drive=1.0) for i in range(4)}
        comp_big = share_injection_ratio([0, 1, 2, 3], big)
        assert comp_big["ratio_net"] == pytest.approx(1.2)    # 0.3×1×4
        assert comp_big["high_share"] is True                 # 净奉献团
        ctrls_lo = {0: _ctrl(share_drive=0.1), 1: _ctrl(share_drive=0.1)}
        assert share_injection_ratio([0, 1], ctrls_lo)["high_share"] is False

    def test_events_record_germ_and_share(self):
        w = _world()
        pop = _pop([(0, 0, 90.0), (1, 1, 95.0)], diff_states={0: 1, 1: 1})
        ctrls = {0: _ctrl(share_drive=1.0), 1: _ctrl(share_drive=1.0)}
        events = []
        reproduce_groups(pop, build_adhesion_chain_index(pop, w, ctrls), w,
                         ctrls, np.random.default_rng(0), 3,
                         ResolveReport(tick=3), max_pop=512,
                         events_out=events)
        assert len(events) == 2                      # 每次 2 个子代
        ev = events[0]
        assert ev["size"] == 2
        assert ev["germ_uid"] == 1                    # 最高能量者（95 > 90）
        assert ev["high_share"] is False      # n=2、drive=1 ⇒ 净蹭桶
        assert ev["ratio_net"] == pytest.approx(0.6)
        assert ev["child_uid"] in (2, 3)

    def test_high_vs_low_bucket_split(self):
        sim = BiosphereSimulator(
            BioMVPConfig(ticks=300, seed=42, w4=W4Config(enabled=True)))
        sim.run()
        repro = metrics_json(sim.log, sim.world, seed=42,
                             ticks=300)["groups"]["adhesion_chain"][
            "repro_composition"]
        assert repro["n_events"] == (repro["n_high_share_events"]
                                     + repro["n_low_share_events"])
        assert repro["germ_source"].startswith("团内最高能量者")
        if repro["n_events"]:
            assert repro["eligible_group_ticks_high"] >= repro[
                "n_high_share_events"]


# ── 4) 判据评估（预注册判据的纯函数） ────────────────────────────────────


class TestCriteriaEvaluation:
    def test_seed_gate_pass_and_fail(self):
        ok = seed_gate([5, 5, 4, 3, 2, 1])
        assert ok["passed"] is True
        dead = seed_gate([5, 5, 0, 0, 0, 0])
        assert dead["final_alive"] == 0 and dead["passed"] is False

    def test_ecological_gate_majority(self):
        gate = evaluate_ecological_gate({
            41: [5, 4, 3, 2, 1], 42: [5, 4, 3, 2, 1], 43: [5, 4, 3, 2, 1],
            44: [5, 0, 0, 0], 45: [5, 0, 0, 0]})
        assert gate["n_passed"] == 3 and gate["majority_passed"] is True
        gate_fail = evaluate_ecological_gate({
            41: [5, 0, 0], 42: [5, 0, 0], 43: [5, 0, 0]})
        assert gate_fail["majority_passed"] is False
        assert "生态门未过" in gate_fail["verdict"]

    def _rows(self, chain_full, chain_z2, adh_full, adh_z2):
        full = [{"seed": s, "chain_ratio_mean": chain_full[i],
                 "adhesion_gene_mean": adh_full[i]} for i, s in
                enumerate([41, 42, 43, 44, 45])]
        z2 = [{"seed": s, "chain_ratio_mean": chain_z2[i],
               "adhesion_gene_mean": adh_z2[i]} for i, s in
              enumerate([41, 42, 43, 44, 45])]
        return full, z2

    def test_criterion_established_4of5(self):
        # full 在 seed 41–44 上两项都更高（41-44 方向一致），seed 45 反向
        full, z2 = self._rows([0.50, 0.45, 0.40, 0.35, 0.10],
                              [0.20, 0.25, 0.30, 0.30, 0.40],
                              [1.0, 0.9, 0.8, 0.7, 0.1],
                              [0.5, 0.6, 0.7, 0.6, 0.8])
        out = evaluate_group_selection_criterion(full, z2)
        assert out["n_chain_direction_ok"] == 4
        assert out["n_adhesion_direction_ok"] == 4
        assert out["n_both_direction_ok"] == 4
        assert out["established"] is True and out["verdict"] == "已建立"

    def test_criterion_not_established_3of5(self):
        full, z2 = self._rows([0.5, 0.4, 0.3, 0.2, 0.1],
                              [0.4, 0.4, 0.4, 0.4, 0.4],
                              [0.5, 0.5, 0.5, 0.5, 0.5],
                              [0.8, 0.8, 0.8, 0.8, 0.8])
        out = evaluate_group_selection_criterion(full, z2)
        assert out["n_both_direction_ok"] == 0
        assert out["established"] is False
        assert out["verdict"] == "未建立"


# ── 5) 作弊者：含死者 + 每团占比 ─────────────────────────────────────────


class TestCheaterAccounting:
    def test_dead_cheater_counted(self):
        """死者流水入账 ⇒ "蹭完就死"的作弊者不被洗掉（B12 修复点）。"""
        from biosphere.metrics import adhesion_chain_json
        from biosphere.groups import CHEATER_MARGIN
        from biosphere.metrics import ChainFrame, SimLog

        log = SimLog()
        # 造一帧 + 流水：uid1 净蹭 5.0（作弊）且已死；uid2 净奉献 -5.0
        log.chain_frames.append(ChainFrame(
            tick=0, n_chain_groups=1, n_multi_chain=1, in_multi_chain=1,
            max_chain_size=2, mean_chain_size=2.0, size_histogram={2: 1},
            adhesion_gene_mean=0.0, adhesion_gene_var=0.0,
            adhesion_expr_mean=0.9, sticky_fraction=1.0,
            focal_size=2, focal_centroid_row=0.0, focal_centroid_col=0.0,
            focal_radius=1.0, n_multi_high_share=1, n_multi_low_share=0,
            n_eligible_high=1, n_eligible_low=0, n_repro_high=0,
            n_repro_low=0))
        log.final_flows = [(2, 5.0, 10.0)]         # uid2 活且净奉献
        log.dead_flows = [(1, 5.0, 0.0)]           # uid1 死后被记录：净蹭作弊
        log.final_chain_groups = [[1, 2]]          # 最终帧的链团成员
        out = adhesion_chain_json(log, _world())
        ch = out["cheaters"]
        assert ch["n_dead_included"] == 1
        assert ch["n_cheaters"] == 1              # 死者 uid1 也算
        assert ch["frequency"] == pytest.approx(0.5)
        assert ch["per_final_group"][0]["cheater_fraction"] == \
            pytest.approx(0.5)

    def test_alive_cheater_unchanged(self):
        sim = BiosphereSimulator(
            BioMVPConfig(ticks=300, seed=42, w4=W4Config(enabled=True)))
        sim.run()
        ch = metrics_json(sim.log, sim.world, seed=42,
                          ticks=300)["groups"]["adhesion_chain"]["cheaters"]
        assert ch["n_members"] >= ch["n_cheaters"]
        assert 0.0 <= ch["frequency"] <= 1.0


# ── 确定性 / checkpoint ─────────────────────────────────────────────────


class TestW4bDeterminism:
    def test_same_seed_bit_identical(self):
        def run():
            sim = BiosphereSimulator(
                BioMVPConfig(ticks=200, seed=42, w4=W4Config(enabled=True)))
            sim.run()
            return ([(i.uid, i.cell, i.energy, i.diff_state)
                     for i in sim.population.individuals],
                    [(f.n_multi_chain, f.in_multi_chain,
                      round(f.adhesion_gene_mean, 12))
                     for f in sim.log.chain_frames])
        assert run() == run()

    def test_checkpoint_resume_bit_identical_with_w4b(self, tmp_path):
        straight = BiosphereSimulator(
            BioMVPConfig(ticks=120, seed=43, w4=W4Config(enabled=True)))
        straight.run()
        part = BiosphereSimulator(
            BioMVPConfig(ticks=60, seed=43, w4=W4Config(enabled=True)))
        part.run(ckpt_dir=tmp_path, ckpt_every=60, tag="w4b")
        from biosphere.checkpoint import load_checkpoint
        resumed = load_checkpoint(next(tmp_path.glob("ckpt_w4b_t60.pkl")))
        resumed.run(ticks=120)
        key = lambda sim: [(i.uid, i.cell, i.energy, i.diff_state, i.pool_in,
                            i.pool_out)
                           for i in sim.population.individuals]
        assert key(resumed) == key(straight)
        assert len(resumed.log.chain_frames) == len(straight.log.chain_frames)

    def test_w4_off_still_bit_identical(self):
        """硬约束：W4Config 默认关 ⇒ 与 W0-B 逐位等价（W4b 改动不漏）。"""
        a = BiosphereSimulator(BioMVPConfig(ticks=80, seed=42))
        a.run()
        b = BiosphereSimulator(
            BioMVPConfig(ticks=80, seed=42, w4=W4Config()))
        b.run()
        assert [(i.uid, i.cell, i.energy) for i in a.population.individuals] \
            == [(i.uid, i.cell, i.energy) for i in b.population.individuals]