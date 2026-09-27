"""F-D2 回归（R230 T-C，B′ 方案）：同进程多引擎不得互相污染随机流。

缺陷（F-D2）：`simulation/sphere_engine.py` 的 D2 路径曾用**进程级全局 `np.random`**
——感知噪声（`perception_noise>0`）与 Steels 随机配对（`steels_alignment`），共 3 处 4 行。
⇒ 同进程两个引擎**交错** `step()` 时互相错位消费同一条全局流 ⇒ **同配置也会分化**
（R226 以"假性分化"侧记：4 个个体移动到不同格）；跨进程单引擎运行则逐位一致
（⇒ 对生产无影响，缺陷只暴露在"同进程多引擎"的一切用法：测试、探针、对照实验）。
修复（B′）：4 行全部改绑**每引擎独立**的 legacy `RandomState`（`self._d2_rng`，
种子/算法与旧全局播种完全相同）⇒ 单引擎轨迹逐位不变、多引擎互不干扰。
（为何不按字面改绑 `self.rng`：那是 PCG64 另一条流，会作废 17 处钉死的 D2 digest
基线 —— 实测 3/3 抽样文件挂，违反本任务自设判据①。详见板上 T-C 报告。）

⚠️ 防复发的**唯一有效形态 = 交错跑**：
  · "顺序跑两遍一致"在缺陷下**也**成立（构造时 `np.random.seed` 重置起点）⇒ 无诊断力；
  · 同进程、同配置、**逐 tick 交错**步进仍逐位一致，才是修复生效的证据。
  实测：把 4 行改回 `np.random.*`（B′ 的 `_d2_rng` 留空转）本测试**第 1 tick 即分化**。

覆盖口径：Python 路径（`use_sim_core=False`）——D2 的生产路径（AGENTS.md 硬约束；
a4_* 全族与本测试同口径）。Rust 路径侧的同格对齐洗牌在纪律下不可达，一并改绑但不在此覆盖。
"""
from __future__ import annotations

import hashlib

import numpy as np

from simulation.config import InfoStructureConfig, SimConfig
from simulation.sphere_engine import SphereEngine

TICKS = 40


def _sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _digest(e: SphereEngine) -> tuple:
    """全状态摘要（含 D2 特有状态）——任一项不等即判分化。"""
    return (
        e._tick, len(e._id), _sha(e._flat), _sha(e._genes),
        _sha(e.resources._grid), _sha(e.signals._marks), _sha(e.signals._age),
        _sha(e._interpret), _sha(e._codebook),
        round(float(e._energy.sum()), 6), round(float(e._stomach.sum()), 6),
        round(float(e._trust.sum()), 6),
    )


class _MethodRecorder:
    """记录经 D2 独立流的抽取方法调用次数（纯直传，不改随机流）。"""

    def __init__(self, gen) -> None:
        self._gen = gen
        self.counts: dict[str, int] = {}

    def __getattr__(self, name):
        attr = getattr(self._gen, name)
        if not callable(attr):
            return attr

        def wrapper(*a, **k):
            self.counts[name] = self.counts.get(name, 0) + 1
            return attr(*a, **k)

        return wrapper


def _engine() -> SphereEngine:
    """D2 全机制开（噪声>0 + Steels 对齐 + 码本）⇒ 修复的 4 个调用行全部处于可触发态。"""
    cfg = SimConfig(seed=11)
    cfg.info_structure = InfoStructureConfig(
        enabled=True, learning_bottleneck=True, arbitrary_codebook=True,
        steels_alignment=True,
        perception_radius=4, perception_noise=0.05, softmax_tau=0.15,
        alignment_rate=0.3, alignment_step=0.2, alignment_noise=0.05,
    )
    cfg.simulation.use_sim_core = False        # D2 只在 Python 路径（AGENTS.md 硬约束）
    cfg.population.initial_count = 120
    cfg.population.max_count = 800
    return SphereEngine(cfg)


def test_two_same_config_engines_interleaved_bitwise_identical():
    """同进程、同配置、交错步进 ⇒ 轨迹与主 rng 消费逐位一致（F-D2 回归）。"""
    a, b = _engine(), _engine()
    rec = _MethodRecorder(a._d2_rng)
    a._d2_rng = rec                             # 仅观测 a 的抽取序列，不影响随机流
    assert _digest(a) == _digest(b), "构造态即不一致 ⇒ 测试前提不成立"

    for t in range(1, TICKS + 1):
        a.step()
        b.step()
        assert _digest(a) == _digest(b), (
            f"tick {t}：同配置两引擎交错步进后分化 ⇒ 存在进程级共享随机源（F-D2 复发）"
        )
    assert a.rng_draws == b.rng_draws, "两引擎主 rng 消费次数不同 ⇒ 随机流被污染"

    # 非空转：D2 噪声/对齐路径确实消费了独立流（否则本测试只是"空转等价"的平凡真）
    assert rec.counts.get("normal", 0) > 0, (
        "D2 噪声路径未消费 `_d2_rng` ⇒ 修复行未被触发，本测试失去防复发力"
    )
    assert rec.counts.get("shuffle", 0) > 0, (
        "Steels 配对路径 40 tick 内未触发（无同格相遇）⇒ 洗牌行未被覆盖"
    )


def test_d2_rng_stream_matches_legacy_global_seed():
    """B′ 核心不变量：`_d2_rng` 与旧全局播种（`np.random.seed(s)`）是**同一条流**。

    这是"单引擎逐位不变"（17 处钉死 digest 全绿）的机制解释：同种子 + 同算法
    （MT19937 legacy）⇒ 同序列；B′ 只是把它从进程级搬成了引擎级。
    """
    seed = 11
    legacy = np.random.RandomState(seed)
    ref_normal = legacy.normal(0, 0.05, size=7)
    ref_shuffle_in = np.arange(9)
    legacy.shuffle(ref_shuffle_in)

    e = _engine()
    got_normal = e._d2_rng.normal(0, 0.05, size=7)
    arr = np.arange(9)
    e._d2_rng.shuffle(arr)
    assert np.array_equal(got_normal, ref_normal), "normal 流与 legacy 全局不同源"
    assert np.array_equal(arr, ref_shuffle_in), "shuffle 流与 legacy 全局不同源"


def test_snapshot_roundtrip_restores_d2_rng_stream():
    """快照续跑一致性：`_d2_rng` 状态必须随快照存取（否则续跑 ≠ 连续跑，静默退化）。"""
    import os
    import tempfile

    e = _engine()
    for _ in range(6):
        e.step()
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "s.npz")
        e.save_snapshot(p)
        r = SphereEngine.load_snapshot(p)
    assert r._d2_rng.get_state()[1].tobytes() == e._d2_rng.get_state()[1].tobytes(), (
        "`_d2_rng` 状态未随快照恢复 ⇒ 续跑轨迹将与连续跑静默分叉"
    )
    # 连续性：两引擎继续各跑 5 tick，逐位一致（含 D2 分支）
    for _ in range(5):
        e.step()
        r.step()
    assert _digest(e) == _digest(r), "快照续跑与连续跑分叉 ⇒ d2_rng_state 存取有误"
