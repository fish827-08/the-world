"""B4：基因组模块测试（突变不变量 + 注册表 + 不可变性）。

覆盖 B1 genome.py 的冻结口径：
- 布局常量自洽（22 参数 + 42 GRN = 64 名义槽）；
- 突变 RNG 顺序写死 ⇒ 同 seed 逐位一致；
- 长度不变量（插删永不越界，贴边退化不消费异常）；
- 基因组不可变（只读视图）；
- validate_genome 的不变量清单。
"""
import numpy as np
import pytest

from biosphere.genome import (
    GENOME_MAX_SLOTS, GENOME_MIN_SLOTS, GENOME_NOMINAL_SLOTS,
    GRN_GENE_COUNT, GRN_GENE_OFFSET, PARAM_SLOTS, PARAM_SLOT_COUNT,
    Genome, MutationKind, MUTATION_PROBS, POINT_MUTATION_SCALE,
    mutate, random_genome, validate_genome,
)


class TestLayoutInvariants:
    """布局常量自洽（改布局 = 存档格式升级，须先报变更单）。"""

    def test_nominal_layout_adds_up(self):
        assert PARAM_SLOT_COUNT + GRN_GENE_COUNT == GENOME_NOMINAL_SLOTS
        assert GRN_GENE_OFFSET == PARAM_SLOT_COUNT

    def test_param_slots_contiguous(self):
        assert len(PARAM_SLOTS) == PARAM_SLOT_COUNT
        assert set(PARAM_SLOTS.values()) == set(range(PARAM_SLOT_COUNT))

    def test_bounds_sane(self):
        assert 0 < GENOME_MIN_SLOTS <= GENOME_NOMINAL_SLOTS <= GENOME_MAX_SLOTS


class TestRandomGenome:
    def test_shape_and_finite(self):
        rng = np.random.default_rng(7)
        g = random_genome(rng, uid=3)
        assert len(g) == GENOME_NOMINAL_SLOTS
        assert g.uid == 3
        assert np.all(np.isfinite(g.genes))

    def test_deterministic(self):
        a = random_genome(np.random.default_rng(11), uid=0)
        b = random_genome(np.random.default_rng(11), uid=0)
        assert np.array_equal(a.genes, b.genes)

    def test_genes_readonly(self):
        g = random_genome(np.random.default_rng(1))
        with pytest.raises(ValueError):
            g.genes[0] = 1.0

    def test_grn_genes_always_42(self):
        short = Genome(genes=np.zeros(GENOME_MIN_SLOTS))
        assert short.grn_genes().shape == (GRN_GENE_COUNT,)
        assert np.all(short.grn_genes() == 0.0)


class TestMutationInvariants:
    def test_same_seed_bit_identical(self):
        g0 = random_genome(np.random.default_rng(5), uid=0)
        g1, r1 = mutate(g0, np.random.default_rng(99), uid=1)
        g2, r2 = mutate(g0, np.random.default_rng(99), uid=1)
        assert np.array_equal(g1.genes, g2.genes)
        assert r1.kind == r2.kind and r1.position == r2.position
        assert (np.isnan(r1.old_value) and np.isnan(r2.old_value)) or \
            r1.old_value == r2.old_value

    def test_point_mutation_changes_one_gene(self):
        rng = np.random.default_rng(3)
        g0 = random_genome(rng, uid=0)
        # 强制点突变：直接构造足够多次直到拿到 point
        for _ in range(200):
            g1, rec = mutate(g0, np.random.default_rng(1000 + _), uid=1)
            if rec.kind == MutationKind.POINT:
                diff = np.flatnonzero(g1.genes != g0.genes)
                assert diff.shape == (1,)
                assert diff[0] == rec.position
                return
        pytest.fail("200 次突变未出现点突变（概率配置坏了吗）")

    def test_insert_delete_change_length(self):
        rng = np.random.default_rng(21)
        g0 = random_genome(rng, uid=0)
        saw_insert = saw_delete = False
        for i in range(300):
            g1, rec = mutate(g0, np.random.default_rng(5000 + i), uid=1)
            if rec.kind == MutationKind.INSERT:
                assert len(g1) == len(g0) + 1
                saw_insert = True
            elif rec.kind == MutationKind.DELETE:
                assert len(g1) == len(g0) - 1
                saw_delete = True
        assert saw_insert and saw_delete, "插删突变都应出现"

    def test_length_bounds_hold_over_many_steps(self):
        """从最短/最长出发各走 400 步：长度永不越界。"""
        for start in (GENOME_MIN_SLOTS, GENOME_MAX_SLOTS):
            rng = np.random.default_rng(42)
            g = Genome(genes=np.clip(rng.standard_normal(start), -3, 3), uid=0)
            for i in range(400):
                g, _rec = mutate(g, np.random.default_rng(90000 + i), uid=i + 1)
                assert GENOME_MIN_SLOTS <= len(g) <= GENOME_MAX_SLOTS

    def test_all_three_kinds_appear(self):
        rng = np.random.default_rng(8)
        g = random_genome(rng, uid=0)
        kinds = set()
        for i in range(400):
            g, rec = mutate(g, np.random.default_rng(700 + i), uid=i + 1)
            kinds.add(rec.kind)
        assert kinds == {MutationKind.POINT, MutationKind.INSERT,
                         MutationKind.DELETE}

    def test_mutation_probs_normalized(self):
        assert abs(sum(MUTATION_PROBS) - 1.0) < 1e-12
        assert POINT_MUTATION_SCALE > 0


class TestValidate:
    def test_random_genome_passes(self):
        rng = np.random.default_rng(2)
        for _ in range(20):
            assert validate_genome(random_genome(rng)) == []

    def test_too_short(self):
        g = Genome(genes=np.zeros(GENOME_MIN_SLOTS - 1))
        errs = validate_genome(g)
        assert any("length" in e for e in errs)

    def test_too_long(self):
        g = Genome(genes=np.zeros(GENOME_MAX_SLOTS + 1))
        errs = validate_genome(g)
        assert any("length" in e for e in errs)

    def test_nan_rejected(self):
        g = Genome(genes=np.full(GENOME_NOMINAL_SLOTS, np.nan))
        errs = validate_genome(g)
        assert any("非有限" in e for e in errs)

    def test_huge_value_rejected(self):
        genes = np.zeros(GENOME_NOMINAL_SLOTS)
        genes[3] = 50.0
        errs = validate_genome(Genome(genes=genes))
        assert any("> 10" in e for e in errs)
