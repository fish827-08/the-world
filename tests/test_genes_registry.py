"""C3 G2：基因语义注册表双写校验测试（Python ↔ Rust 漂移检测）。

核心：validate_gene_wiring() 返回不一致列表。正常情况为空；
故意改一侧索引后必须非空（测试必须失败的反面验证）。
"""
import pytest

from simulation.genes import (
    Gene, GENE_COUNT, GENE_SEMANTICS, GENE_TRAIT_NAMES,
    GENE_WIRED, GENE_META, gene_meta, gene_index, gene_name,
    validate_gene_wiring,
)


class TestGeneRegistryIntegrity:
    """注册表自身完整性：数量、命名、元数据。"""

    def test_gene_count_matches_enum(self):
        assert GENE_COUNT == len(Gene) == 24

    def test_semantics_count_matches(self):
        assert len(GENE_SEMANTICS) == GENE_COUNT

    def test_trait_names_count_matches(self):
        assert len(GENE_TRAIT_NAMES) == GENE_COUNT

    def test_meta_count_matches(self):
        assert len(GENE_META) == GENE_COUNT

    def test_gene_values_are_contiguous(self):
        """基因枚举值必须是 0~23 连续。"""
        values = sorted(int(g) for g in Gene)
        assert values == list(range(GENE_COUNT))

    def test_gene_names_are_unique(self):
        names = [g.name for g in Gene]
        assert len(names) == len(set(names))

    def test_meta_mutation_scale_positive(self):
        """所有基因的 mutation_scale 必须 > 0。"""
        for i in range(GENE_COUNT):
            assert GENE_META[i].mutation_scale > 0, f"g{i} mutation_scale <= 0"

    def test_meta_selection_direction_valid(self):
        """selection_direction 必须是 -1/0/+1。"""
        for i in range(GENE_COUNT):
            assert GENE_META[i].selection_direction in (-1, 0, 1), f"g{i} invalid direction"

    def test_gene_index_roundtrip(self):
        """gene_index ↔ gene_name 往返一致。"""
        for g in Gene:
            assert gene_index(g.name) == int(g)
            assert gene_name(int(g)) == g.name.lower()

    def test_wired_genes_are_subset(self):
        """GENE_WIRED 中的索引必须在有效范围内。"""
        for idx in GENE_WIRED:
            assert 0 <= idx < GENE_COUNT


class TestGeneWiringValidation:
    """Python ↔ Rust 双写校验（validate_gene_wiring）。"""

    def test_no_drift_in_clean_state(self):
        """正常状态：无漂移。"""
        drift = validate_gene_wiring()
        assert drift == [], f"存在基因索引漂移: {drift}"

    def test_drift_detected_when_py_value_changed(self):
        """故意改 Python 侧值→校验必须返回非空（漂移检测生效）。

        模拟：把 G_PERCEPTION 的值从 14 改成 15，调用 Rust 校验。
        """
        import sim_core
        py_names = [g.name for g in Gene]
        py_values = [int(g) for g in Gene]
        # 故意篡改：PERCEPTION 从 14 改成 15
        percep_idx = py_names.index("PERCEPTION")
        py_values[percep_idx] = 15  # 原本是 14
        drift = sim_core.validate_gene_wiring(py_names, py_values)
        assert len(drift) > 0, "篡改后应检测到漂移，但返回空"
        # R204 起条目名带 `|KIND` 归因后缀（如 `PERCEPTION|VALUE_MISMATCH`），比名字段
        assert any(d[0].split("|")[0] == "PERCEPTION" for d in drift), f"漂移应包含 PERCEPTION: {drift}"

    def test_drift_detected_when_py_name_changed(self):
        """故意改 Python 侧名字→校验必须返回非空。"""
        import sim_core
        py_names = [g.name for g in Gene]
        py_values = [int(g) for g in Gene]
        # 故意篡改：把 MOVE_PROB 改成 MOVEMENT_PROB
        py_names[0] = "MOVEMENT_PROB"
        drift = sim_core.validate_gene_wiring(py_names, py_values)
        assert len(drift) > 0, "篡改名字后应检测到漂移"

    def test_drift_detected_when_count_mismatch(self):
        """基因数量不一致→校验必须返回非空。"""
        import sim_core
        py_names = [g.name for g in Gene][:-1]  # 少一个
        py_values = [int(g) for g in Gene][:-1]
        drift = sim_core.validate_gene_wiring(py_names, py_values)
        assert len(drift) > 0, "数量不一致应检测到漂移"

    def test_all_24_genes_are_validated(self):
        """validate_gene_wiring 必须覆盖全部 24 个基因（不是只查 3 个）。"""
        import sim_core
        # 逐个篡改每个基因的值，确认每个都能被检测到
        for target_gene in Gene:
            py_names = [g.name for g in Gene]
            py_values = [int(g) for g in Gene]
            idx = py_names.index(target_gene.name)
            py_values[idx] = (py_values[idx] + 1) % 24  # 改成不同的值
            drift = sim_core.validate_gene_wiring(py_names, py_values)
            assert any(d[0].split("|")[0] == target_gene.name for d in drift), \
                f"{target_gene.name} 篡改后未被检测到（可能未覆盖此基因）"


class TestGeneMetaAccess:
    """GeneMeta 访问函数。"""

    def test_gene_meta_returns_correct_type(self):
        m = gene_meta(0)
        assert hasattr(m, "mutation_scale")
        assert hasattr(m, "selection_direction")

    def test_gene_meta_out_of_range_raises(self):
        with pytest.raises(IndexError):
            gene_meta(99)

    def test_wired_genes_have_conservative_mutation_scale(self):
        """核心基因（g1 代谢, g3 寿命）的 mutation_scale 应小于探索性基因（g15 信号）。"""
        metabolic_scale = gene_meta(int(Gene.METABOLIC)).mutation_scale
        life_scale = gene_meta(int(Gene.LIFE_GENE)).mutation_scale
        signal_scale = gene_meta(int(Gene.SIGNAL_STRENGTH)).mutation_scale
        assert metabolic_scale < signal_scale
        assert life_scale < signal_scale
