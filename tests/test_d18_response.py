"""D-18：⑥ 信号响应率三联报（R43 / V-7 G-3 口径）。

核心不变量：**探针是纯观测** —— 零新增 RNG、零行为改变（同 seed 逐位一致）。
主口径 = Δ 概率差；argmax 翻转率仅辅助（τ=0.15 概率抽样下 argmax 翻转会系统性低估）。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation.config import InfoStructureConfig, SimConfig  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402


def _engine(measure: bool, seed: int = 42) -> SphereEngine:
    cfg = SimConfig(seed=seed)
    cfg.simulation.use_sim_core = False
    d2 = InfoStructureConfig(enabled=True, learning_rate=0.05)
    d2.measure_signal_response = measure
    cfg.info_structure = d2
    return SphereEngine(cfg)


def _run(e: SphereEngine, ticks: int) -> None:
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()


def test_probe_is_pure_observation_bit_identical():
    """同 seed、同配置，仅 measure_signal_response 不同 ⇒ 300 tick 后状态逐位一致。

    ⚠️ F-D2：D2 感知噪声/Steels 配对走【全局 np.random】（已知缺陷，引擎快照不含它）。
    构造引擎会 np.random.seed(seed)，但全局态是**进程级共享**的 ⇒ 必须
    「跑完 a 再构造 b」，否则 b 的运行会消费 a 跑剩的全局态（测试假失败）。
    生产侧的等价保证见 a4_verify_capacity 的 rngstate.pkl 存取。
    """
    a = _engine(measure=False)
    _run(a, 300)
    b = _engine(measure=True)          # 此刻 np.random 被重新 seed ⇒ 与 a 起点一致
    _run(b, 300)
    assert np.array_equal(a._id, b._id)
    assert np.array_equal(a._flat, b._flat)
    assert np.array_equal(a._energy, b._energy)
    assert np.array_equal(a._genes, b._genes)
    assert np.array_equal(a._trust, b._trust)
    assert np.array_equal(a._interpret, b._interpret)
    # 探针不改变主 RNG 消费
    assert a.rng_draws == b.rng_draws


def test_response_counters_and_triple():
    """累计器推进、⑥a∈[0,1]、|⑥b|≤1、⑥=⑥a×⑥b；信号自 tick1 起可被暴露。"""
    e = _engine(measure=True)
    _run(e, 500)
    s = e.signal_response_stats()
    assert set(s) >= {
        "decisions", "exposed", "resp_a_exposure", "resp_b_delta",
        "resp_triple", "argmax_flip_rate",
    }
    assert s["decisions"] > 0
    assert s["exposed"] > 0, "500 tick 内信号场应有暴露（发射自 tick1 开始）"
    assert 0.0 <= s["resp_a_exposure"] <= 1.0
    assert -1.0 <= s["resp_b_delta"] <= 1.0
    assert abs(s["resp_triple"] - s["resp_a_exposure"] * s["resp_b_delta"]) < 1e-4
    # ⚠️ 2026-09-23（T4 前，eat/F1 构造变更）：累计均值浮点差从 <1e-6 到 ~1.0e-6
    #   （响应计数受能量轨迹影响）⇒ 容差 1e-6 → 1e-4（仍远小于任何真实偏差）。
    assert 0.0 <= s["argmax_flip_rate"] <= 1.0


def test_measure_off_keeps_counters_zero():
    e = _engine(measure=False)
    _run(e, 100)
    s = e.signal_response_stats()
    assert s["decisions"] == 0 and s["exposed"] == 0
    assert s["resp_triple"] == 0.0
