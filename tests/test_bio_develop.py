"""B4：GRN 发育测试（确定性 + 参数域 + 映射可演化）。

覆盖 B1 develop.py：
- develop 是纯函数：同基因组 ⇒ 控制器逐位一致；
- 全部参数落在文档化钳制域内；
- 突变改变基因组 ⇒（几乎必然）改变控制器（"映射本身可演化"）；
- ControllerCache 按基因组 uid 缓存。
"""
import numpy as np

from biosphere.develop import ControllerCache, develop
from biosphere.genome import Genome, mutate, random_genome

#: (字段名, 下界, 上界) —— 与 develop.py 的 _clamp 一致；改域须同步规格页
PARAM_DOMAINS = [
    ("sense_food", 0.0, 2.0), ("sense_threat", 0.0, 2.0),
    ("sense_kin", 0.0, 2.0), ("sense_signal", 0.0, 2.0),
    ("move_bias", 0.0, 1.0), ("eat_greed", 0.0, 1.0),
    ("attack_drive", 0.0, 1.0), ("share_drive", 0.0, 1.0),
    ("signal_drive", 0.0, 1.0), ("repro_drive", 0.0, 1.0),
    ("metabolism", 0.5, 2.0), ("move_cost_mult", 0.5, 2.0),
    ("lifespan", 200.0, 2000.0), ("repro_thresh", 40.0, 120.0),
    ("habitat_aquatic", 0.0, 1.0), ("habitat_terrestrial", 0.0, 1.0),
    ("adhesion", 0.0, 1.0), ("defense", 0.0, 1.0),
    ("eat_amount", 0.2, 1.0), ("signal_payload", 0.0, 1.0),
    ("give_up", 3.0, 20.0), ("persistence", 0.0, 1.0),
    # C3 v2（B14.3）
    ("badge", 0, 3), ("emit_rate", 0.0, 1.0), ("strength", 0.0, 1.0),
    ("response_0", -1.0, 1.0), ("response_1", -1.0, 1.0),
    ("response_2", -1.0, 1.0), ("response_3", -1.0, 1.0),
    ("learning_rate", 0.0, 1.0),
]


class TestDevelopDeterminism:
    def test_same_genome_bit_identical(self):
        g = random_genome(np.random.default_rng(4), uid=0)
        p1 = develop(g)
        p2 = develop(g)
        assert p1 == p2  # dataclass 逐字段 == ⇒ 浮点逐位一致

    def test_pure_function_no_side_effect(self):
        g = random_genome(np.random.default_rng(4), uid=0)
        before = g.fingerprint()
        develop(g)
        assert g.fingerprint() == before

    def test_all_domains_hold(self):
        rng = np.random.default_rng(6)
        for _ in range(50):
            p = develop(random_genome(rng))
            for name, lo, hi in PARAM_DOMAINS:
                if name.startswith("response_"):
                    v = p.response[int(name.split("_")[-1])]
                else:
                    v = getattr(p, name)
                assert lo <= v <= hi, f"{name}={v} 越界 [{lo}, {hi}]"

    def test_as_dict_roundtrip(self):
        p = develop(random_genome(np.random.default_rng(9), uid=0))
        d = p.as_dict()
        assert len(d) == len(PARAM_DOMAINS)
        assert all(np.isfinite(v) for v in d.values())


class TestMappingEvolvable:
    def test_mutation_changes_controller_within_k_steps(self):
        """突变的控制器几乎必然变；参数钳制饱和（渠道化）会使**单次**点突变
        偶发静默（≈45%，已实测）⇒ 断言有限步内必变（种子固定，无 flake）。"""
        for s in range(20):
            rng = np.random.default_rng(1000 + s)
            g = random_genome(rng, uid=0)
            p0 = develop(g)
            changed = False
            for step in range(10):
                g, _rec = mutate(g, np.random.default_rng(2000 + s * 100 + step),
                                 uid=step + 1)
                if develop(g) != p0:
                    changed = True
                    break
            assert changed, f"seed={s}: 10 次突变后控制器仍与初始一致"

    def test_single_mutation_often_silent_by_saturation(self):
        """登记事实：单次点突变可能因钳制饱和而表型静默（不为凑数而删）。"""
        silent = 0
        trials = 0
        for s in range(40):
            rng = np.random.default_rng(3000 + s)
            g0 = random_genome(rng, uid=0)
            g1, rec = mutate(g0, np.random.default_rng(4000 + s), uid=1)
            if rec.kind == "point":
                trials += 1
                if develop(g0) == develop(g1):
                    silent += 1
        assert 0 < silent < trials, "要么全变要么全不变，与渠道化预期不符"

    def test_extreme_genomes_differ(self):
        lo = develop(Genome(genes=np.full(64, -3.0)))
        hi = develop(Genome(genes=np.full(64, 3.0)))
        assert lo != hi


class TestControllerCache:
    def test_cache_hit_same_object(self):
        cache = ControllerCache()
        g = random_genome(np.random.default_rng(12), uid=5)
        assert cache.get(g) is cache.get(g)
        assert len(cache) == 1

    def test_cache_distinguishes_uid(self):
        cache = ControllerCache()
        g1 = random_genome(np.random.default_rng(13), uid=1)
        g2 = random_genome(np.random.default_rng(13), uid=2)
        cache.get(g1)
        cache.get(g2)
        assert len(cache) == 2
