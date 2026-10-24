"""B11：W4 多细胞/合作 + 细胞尺度模块 + checkpoint 协议测试。

覆盖补充指令五项增量：
1. 细胞定义（agent：粘/双稳态分化/动 ≤1 格）—— cell.py 类型/约束/分化；
2. 水陆 = 字段约束（depth>0 / soil_water ≥ 世界阈值，两栖有代价）；
3. 团运动学度量（质心位移 + 形态，焦点团序列）；
4. 判据含作弊者（团大小/分化/作弊者频率/谱系存活）；
5. checkpoint 可恢复（>1000 tick 跑批；续跑与直跑逐位一致）。
"""
import numpy as np
import pytest

from biosphere.cell import (
    DIFF_HI, DIFF_LO, TYPE_AMPHIBIOUS, TYPE_AQUATIC, TYPE_TERRESTRIAL,
    TYPE_UNADAPTED, adhesion_expression, can_occupy, cell_type,
    occupancy_penalty, update_differentiation,
)
from biosphere.develop import ControllerParams, develop
from biosphere.genome import Genome, random_genome
from biosphere.groups import (
    GROUP_ENDOWMENT, GROUP_REPRO_THRESH_PER_CAPITA, GroupIndex,
    W4Config, build_group_index, reproduce_groups, share_group_energy,
)
from biosphere.resolve import Individual, Population, ResolveReport
from biosphere.simulator import BioMVPConfig, BiosphereSimulator
from biosphere.world_adapter import (
    WorldView, _von_neumann_table, synthetic_world,
)


def _ctrl(**overrides) -> ControllerParams:
    base = dict(
        sense_food=1.0, sense_threat=1.0, sense_kin=1.0, sense_signal=1.0,
        move_bias=0.5, eat_greed=0.5, attack_drive=0.0, share_drive=0.5,
        signal_drive=0.0, repro_drive=0.5, metabolism=1.0, move_cost_mult=1.0,
        lifespan=1000.0, repro_thresh=40.0, habitat_aquatic=0.0,
        habitat_terrestrial=0.0, adhesion=0.5, defense=0.0, eat_amount=0.5,
        signal_payload=0.5, give_up=10.0, persistence=0.0,
    )
    base.update(overrides)
    return ControllerParams(**base)


def _world(rows=4, cols=6, soil=None, soil_min=2.0) -> WorldView:
    n = rows * cols
    st = np.zeros(n, dtype=np.uint8)          # 全陆
    if soil is None:
        soil = np.full(n, 5.0)                 # 默认湿润（陆生可居）
    return WorldView(
        rows=rows, cols=cols, n_cells=n,
        surface_type=st,
        elevation_m=np.zeros(n), depth_m=np.zeros(n),
        patch_id=np.full(n, -1, dtype=np.int32),
        neighbors=_von_neumann_table(rows, cols),
        soil_water=np.asarray(soil, dtype=np.float64),
        soil_water_min=soil_min,
    )


# ── 1. 细胞类型与栖息约束（补充指令 2） ──────────────────────────────────


class TestCellType:
    def test_type_mapping(self):
        assert cell_type(_ctrl(habitat_aquatic=1.0, habitat_terrestrial=0.0)) \
            == TYPE_AQUATIC
        assert cell_type(_ctrl(habitat_aquatic=0.0, habitat_terrestrial=1.0)) \
            == TYPE_TERRESTRIAL
        assert cell_type(_ctrl(habitat_aquatic=1.0, habitat_terrestrial=1.0)) \
            == TYPE_AMPHIBIOUS
        assert cell_type(_ctrl(habitat_aquatic=0.0, habitat_terrestrial=0.0)) \
            == TYPE_UNADAPTED

    def test_aquatic_needs_depth(self):
        w = _world()
        w.depth_m[3] = 1.0                      # cell 3 是水
        assert can_occupy(TYPE_AQUATIC, w, 3) is True
        assert can_occupy(TYPE_AQUATIC, w, 0) is False

    def test_terrestrial_needs_soil_water(self):
        soil = np.full(24, 5.0)
        soil[3] = 1.0                           # cell 3 干
        w = _world(soil=soil, soil_min=2.0)
        assert can_occupy(TYPE_TERRESTRIAL, w, 0) is True
        assert can_occupy(TYPE_TERRESTRIAL, w, 3) is False

    def test_threshold_from_world_adapter(self):
        """阈值是世界侧参数：换 soil_water_min 结论随之翻。"""
        w = _world(soil=np.full(24, 2.5), soil_min=3.0)
        assert can_occupy(TYPE_TERRESTRIAL, w, 0) is False
        w2 = _world(soil=np.full(24, 2.5), soil_min=2.0)
        assert can_occupy(TYPE_TERRESTRIAL, w2, 0) is True

    def test_amphibious_and_unadapted_unconstrained(self):
        w = _world(soil=np.zeros(24), soil_min=2.0)   # 全干
        for t in (TYPE_AMPHIBIOUS, TYPE_UNADAPTED):
            assert can_occupy(t, w, 0) is True

    def test_occupancy_penalty(self):
        w = _world()
        w.depth_m[3] = 1.0
        assert occupancy_penalty(TYPE_AQUATIC, w, 3) == 0.0
        assert occupancy_penalty(TYPE_AQUATIC, w, 0) > 0.0     # 水生上岸
        assert occupancy_penalty(TYPE_TERRESTRIAL, w, 0) == 0.0  # 陆生湿地
        soil = np.zeros(24)
        w2 = _world(soil=soil)
        assert occupancy_penalty(TYPE_TERRESTRIAL, w2, 0) > 0.0  # 脱水
        assert occupancy_penalty(TYPE_AMPHIBIOUS, w, 0) > 0.0   # 有代价
        assert occupancy_penalty(TYPE_AMPHIBIOUS, w, 3) > 0.0

    def test_differentiation_hysteresis(self):
        """迟滞：≥HI 翻 1、≤LO 翻 0、中间保持（双稳态）。"""
        assert update_differentiation(0, DIFF_HI) == 1
        assert update_differentiation(1, DIFF_LO) == 0
        assert update_differentiation(0, DIFF_LO) == 0      # 中间保持
        assert update_differentiation(1, DIFF_HI - 1) == 1  # 中间保持

    def test_adhesion_expression_by_state(self):
        assert adhesion_expression(0.8, 1) == pytest.approx(0.8)
        assert adhesion_expression(0.8, 0) == pytest.approx(0.4)  # 半表达


# ── 2. 团判定与共享账 ────────────────────────────────────────────────────


def _pop(pairs) -> Population:
    """pairs: [(uid, cell, energy)] ⇒ Population。"""
    pop = Population()
    for uid, cell, energy in pairs:
        ind = Individual(uid=uid, genome=Genome(genes=np.zeros(64)),
                         cell=cell, energy=energy)
        pop.individuals.append(ind)
        pop.next_id = max(pop.next_id, uid + 1)
    return pop


class TestGroupIndex:
    def test_connected_components(self):
        w = _world(4, 6)
        # 0-1 相邻；12（row2col0）与两者都不邻（注意 0 与 5 列环绕相邻！）
        pop = _pop([(0, 0, 10.0), (1, 1, 10.0), (2, 12, 10.0)])
        idx = build_group_index(pop, w)
        assert idx.group_of[0] == idx.group_of[1]
        assert idx.group_of[2] != idx.group_of[0]
        assert idx.size(0) == 2 and idx.size(2) == 1
        assert idx.n_groups == 2

    def test_chain_merges(self):
        w = _world(4, 6)
        pop = _pop([(0, 0, 10.0), (1, 1, 10.0), (2, 2, 10.0)])  # 链 0-1-2
        idx = build_group_index(pop, w)
        assert idx.size(0) == 3 and idx.n_groups == 1


class TestGroupSharing:
    def test_pool_math_exact(self):
        """注入 = 0.3×share_drive×energy；平分；账目逐分对得上。"""
        w = _world()
        pop = _pop([(0, 0, 100.0), (1, 1, 100.0)])
        controllers = {0: _ctrl(share_drive=1.0), 1: _ctrl(share_drive=0.0)}
        report = ResolveReport(tick=0)
        share_group_energy(pop, build_group_index(pop, w), controllers,
                           report)
        inds = {i.uid: i for i in pop.individuals}
        # 注入池 = 0.3×1.0×100 = 30；每人 +15；uid0 净 −15，uid1 净 +15
        assert inds[0].energy == pytest.approx(85.0)
        assert inds[1].energy == pytest.approx(115.0)
        assert inds[0].pool_out == pytest.approx(30.0)
        assert inds[0].pool_in == pytest.approx(15.0)
        assert inds[1].pool_out == pytest.approx(0.0)
        assert inds[1].pool_in == pytest.approx(15.0)
        # 总量守恒
        assert inds[0].energy + inds[1].energy == pytest.approx(200.0)

    def test_singleton_group_no_share(self):
        w = _world()
        pop = _pop([(0, 0, 100.0), (1, 10, 100.0)])   # 不相邻
        controllers = {0: _ctrl(share_drive=1.0), 1: _ctrl(share_drive=1.0)}
        report = ResolveReport(tick=0)
        share_group_energy(pop, build_group_index(pop, w), controllers,
                           report)
        assert report.group_share_events == 0
        assert all(i.energy == 100.0 for i in pop.individuals)

    def test_cheater_detection_data(self):
        """低 share_drive 成员净蹭 ⇒ pool_in − pool_out > 0（判据数据源）。"""
        w = _world()
        pop = _pop([(0, 0, 100.0), (1, 1, 100.0), (2, 2, 100.0)])
        controllers = {0: _ctrl(share_drive=1.0), 1: _ctrl(share_drive=0.0),
                       2: _ctrl(share_drive=0.0)}
        share_group_energy(pop, build_group_index(pop, w), controllers,
                           ResolveReport(tick=0))
        inds = {i.uid: i for i in pop.individuals}
        assert inds[1].pool_in - inds[1].pool_out > 0.0   # 蹭团
        assert inds[0].pool_in - inds[0].pool_out < 0.0   # 奉献


class TestGroupReproduction:
    def test_gate_and_cost_split(self):
        w = _world()
        pop = _pop([(0, 0, 90.0), (1, 1, 90.0)])   # 总 180 ≥ 80×2
        controllers = {0: _ctrl(), 1: _ctrl()}
        report = ResolveReport(tick=5)
        rng = np.random.default_rng(0)
        reproduce_groups(pop, build_group_index(pop, w), w, controllers,
                         rng, 5, report, max_pop=512)
        assert len(pop.individuals) == 4            # 2 子代
        cost_each = GROUP_ENDOWMENT * 2 / 2
        assert pop.individuals[0].energy == pytest.approx(90.0 - cost_each)
        assert pop.individuals[1].energy == pytest.approx(90.0 - cost_each)
        for child in pop.individuals[2:]:
            assert child.energy == pytest.approx(GROUP_ENDOWMENT)
            assert child.parent in (0, 1)
        assert len(report.group_births) == 2

    def test_below_gate_no_repro(self):
        w = _world()
        pop = _pop([(0, 0, 30.0), (1, 1, 30.0)])   # 总 60 < 160
        controllers = {0: _ctrl(), 1: _ctrl()}
        report = ResolveReport(tick=0)
        reproduce_groups(pop, build_group_index(pop, w), w, controllers,
                         np.random.default_rng(0), 0, report, max_pop=512)
        assert len(pop.individuals) == 2
        assert report.group_births == []

    def test_determinism(self):
        def run():
            w = _world()
            pop = _pop([(0, 0, 90.0), (1, 1, 90.0)])
            controllers = {0: _ctrl(), 1: _ctrl()}
            reproduce_groups(pop, build_group_index(pop, w), w, controllers,
                             np.random.default_rng(7), 0,
                             ResolveReport(tick=0), max_pop=512)
            return [(i.uid, i.cell, i.energy, i.parent)
                    for i in pop.individuals]
        assert run() == run()


# ── 3. W4 开关与模拟器集成 ───────────────────────────────────────────────


class TestW4Integration:
    def test_w4_off_matches_baseline(self):
        """W4 关 ⇒ 与 W0-B 逐位同构（默认关纪律）。"""
        a = BiosphereSimulator(BioMVPConfig(ticks=60, seed=42))
        a.run()
        b = BiosphereSimulator(
            BioMVPConfig(ticks=60, seed=42, w4=W4Config(enabled=False)))
        b.run()
        assert [(i.uid, i.cell, i.energy) for i in a.population.individuals] \
            == [(i.uid, i.cell, i.energy) for i in b.population.individuals]

    def test_w4_on_runs_and_records_groups(self):
        sim = BiosphereSimulator(
            BioMVPConfig(ticks=200, seed=42, w4=W4Config(enabled=True)))
        sim.run()
        assert len(sim.log.group_frames) == 200
        assert sim.log.group_frames[-1].n_groups >= 1

    def test_differentiation_updates(self):
        sim = BiosphereSimulator(
            BioMVPConfig(ticks=100, seed=42, w4=W4Config(enabled=True)))
        sim.run()
        states = {i.diff_state for i in sim.population.individuals}
        assert states <= {0, 1}

    def test_full_mixing_zeroes_structure(self):
        """全混合零模型：位置每 tick 重排（仍有占用唯一性）。"""
        sim = BiosphereSimulator(
            BioMVPConfig(ticks=50, seed=42,
                         w4=W4Config(enabled=True, full_mixing=True)))
        sim.run()
        cells = [i.cell for i in sim.population.individuals]
        assert len(cells) == len(set(cells))

    def test_no_signal_zero_model(self):
        """关信号零模型：无 signal 意图、信号层恒 0。"""
        sim = BiosphereSimulator(
            BioMVPConfig(ticks=100, seed=42,
                         w4=W4Config(enabled=True, signal=False)))
        sim.run()
        assert not any(e.kind == "signal" for e in sim.log.events)
        assert float(np.max(np.abs(sim.signal_layer))) == 0.0

    def test_habitat_constraint_blocks_move(self):
        """陆生细胞进不去干格（字段约束，非基因阈值）。"""
        soil = np.full(36, 5.0)
        soil[1] = 0.0                              # cell 1 干
        w = _world(6, 6, soil=soil, soil_min=2.0)
        from biosphere.action import ActionKind, Intent
        from biosphere.resolve import commit_intents, resolve_tick
        pop = _pop([(0, 0, 50.0)])
        # 陆生控制器：habitat_terrestrial=1.0
        c = develop(random_genome(np.random.default_rng(3)))
        controllers = {0: _ctrl(habitat_terrestrial=1.0, habitat_aquatic=0.0)}
        intents = [Intent(actor_id=0, kind=ActionKind.MOVE, target_cell=1,
                          energy_cost=0.3)]
        report = resolve_tick(pop, commit_intents(pop, intents), w,
                              _resource(w), np.zeros(36), 0,
                              np.random.default_rng(0), controllers)
        assert report.events[0].status == "stay_habitat"
        assert pop.individuals[0].cell == 0


def _resource(w: WorldView):
    from biosphere.world_adapter import SyntheticResource
    return SyntheticResource(amount=np.ones(w.n_cells),
                             capacity=np.ones(w.n_cells), regrow_rate=0.0)


# ── 4. checkpoint 协议（补充指令 5） ──────────────────────────────────────


class TestCheckpoint:
    def test_resume_matches_straight_run(self, tmp_path):
        """直跑 120 tick vs 60+checkpoint+续跑 60 ⇒ 逐位一致。"""
        straight = BiosphereSimulator(BioMVPConfig(ticks=120, seed=42))
        straight.run()
        key = lambda sim: [(i.uid, i.cell, i.energy, i.age, i.threat_age,
                            i.diff_state)
                           for i in sim.population.individuals]

        part = BiosphereSimulator(BioMVPConfig(ticks=60, seed=42))
        part.run(ckpt_dir=tmp_path, ckpt_every=60, tag="t")
        ckpts = list(tmp_path.glob("ckpt_t_t60.pkl"))
        assert len(ckpts) == 1
        from biosphere.checkpoint import load_checkpoint
        resumed = load_checkpoint(ckpts[0])
        assert resumed.tick_count == 60
        resumed.run(ticks=120)
        assert key(resumed) == key(straight)
        assert len(resumed.log.frames) == len(straight.log.frames)

    def test_genome_readonly_after_unpickle(self, tmp_path):
        from biosphere.checkpoint import load_checkpoint, save_checkpoint
        sim = BiosphereSimulator(BioMVPConfig(ticks=5, seed=42))
        sim.run()
        path = save_checkpoint(sim, tmp_path / "g.pkl")
        back = load_checkpoint(path)
        with pytest.raises(ValueError):
            back.population.individuals[0].genome.genes[0] = 1.0

    def test_run_log_appends(self, tmp_path):
        from biosphere.checkpoint import append_run_log
        append_run_log(tmp_path, "tag", "line-1")
        append_run_log(tmp_path, "tag", "line-2")
        text = (tmp_path / "tag.run.log").read_text(encoding="utf-8")
        assert "line-1" in text and "line-2" in text


# ── 5. W4 度量形状 ───────────────────────────────────────────────────────


class TestW4Metrics:
    def test_groups_json_two_lenses(self):
        """两镜头：粘附链团（主张）+ 邻接团（对照）。"""
        sim = BiosphereSimulator(
            BioMVPConfig(ticks=100, seed=42, w4=W4Config(enabled=True)))
        sim.run()
        from biosphere.metrics import metrics_json
        m = metrics_json(sim.log, sim.world, seed=42, ticks=100)
        g = m["groups"]
        assert set(g) == {"adhesion_chain", "adjacency_control"}
        chain = g["adhesion_chain"]
        assert chain["enabled"] is True
        for key in ("n_multi_chain_final", "size_histogram_final",
                    "adhesion_trajectory", "repro_composition", "cheaters",
                    "kinematics", "chain_ratio_mean"):
            assert key in chain
        traj = chain["adhesion_trajectory"]
        assert set(traj["start"]) >= {"adhesion_gene_mean", "adhesion_gene_var",
                                       "sticky_fraction"}
        # 对照镜头保留旧读数（密度混淆但仍是读数）
        ctrl = g["adjacency_control"]
        assert ctrl["enabled"] is True
        assert "n_multi_final" in ctrl
        k = chain["kinematics"]
        assert "不用单细胞速度" in k["note"]
        # 判据四件套在链团段
        assert chain["cheaters"]["n_members"] >= 0
        assert "per_final_group" in chain["cheaters"]

    def test_groups_json_empty_when_off(self):
        sim = BiosphereSimulator(BioMVPConfig(ticks=10, seed=42))
        sim.run()
        from biosphere.metrics import metrics_json
        g = metrics_json(sim.log, sim.world, seed=42, ticks=10)["groups"]
        assert g["adhesion_chain"]["enabled"] is False
        assert g["adjacency_control"]["enabled"] is False
