"""A′ 记忆**朝向梯度**（2026-09-19）实施回归。

设计稿：`docs/设计文档/设计-A档记忆朝向梯度-20260919.md`
要回答的科学问题：个体**自己**记住的富食格位置，会不会改变它往哪走？

四条必备（设计稿 §六）：
  ① **极性**（F-R18 同族）：记忆在北 ⇒ 北侧邻居得分最高；挪到东 ⇒ 峰值东移；背向被**减分**
  ② **反例**：记忆格 == 当前格（或全空槽）⇒ **无方向** ⇒ 所有邻居等分（不得偏置）
  ③ **开关**：`orientation` 与 `none` 轨迹**必须不同**（开关真的生效）；`none` 路径钉死 digest
  ④ **双路径**：Rust(`use_sim_core=True`) 与 Python 逐位一致 + **经度环绕**取最短路

⚠️ ①~②/环绕 直接测**纯函数** `_memory_orientation_cos`（用 stub 提供 `world.cols`）⇒
   不依赖种群演化，断言值**可手算核对**（cos=±1 / 0），是"极性"最硬的证据形式。
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.config import InfoStructureConfig, SimConfig
from simulation.sphere_engine import SphereEngine

COLS = 120       # 默认世界 60×120（`world/sphere_world.py`）
_FLAT = lambda r, c: int(r) * COLS + int(c)   # noqa: E731


class _StubWorld:
    def __init__(self, cols: int = COLS) -> None:
        self.cols = cols


class _StubEngine:
    def __init__(self, cols: int = COLS) -> None:
        self.world = _StubWorld(cols)


def _cos(cur: int, nb: list[int], mem: list[int]) -> np.ndarray:
    """直接调未绑定方法（避免构造整台引擎）。"""
    return SphereEngine._memory_orientation_cos(
        _StubEngine(), cur, np.asarray(nb, dtype=np.int64), np.asarray(mem, dtype=np.int64))


# 当前格 (30,60) 的四个正向邻居：北 / 南 / 东 / 西
_CUR = _FLAT(30, 60)
_NB = [_FLAT(29, 60), _FLAT(31, 60), _FLAT(30, 61), _FLAT(30, 59)]   # N, S, E, W

# ------------------------------------------------------------ ① 极性
def test_orientation_cos_peaks_toward_memory():
    """记忆在正北 ⇒ 北侧 +1、南侧 **−1**（真梯度，不是纯吸引）、东/西 ≈ 0。"""
    g = _cos(_CUR, _NB, [_FLAT(20, 60), -1, -1, -1])     # 记忆：正北 10 格
    assert int(np.argmax(g)) == 0, "峰值必须在北侧"
    assert g[0] > 0.99 and g[1] < -0.99
    assert abs(g[2]) < 0.02 and abs(g[3]) < 0.02           # 正交 ⇒ 0


def test_orientation_peak_follows_memory_direction():
    """把记忆挪到正东 ⇒ 峰值**跟着移到东**（否则说明方向算反了/没生效）。"""
    g = _cos(_CUR, _NB, [_FLAT(30, 70), -1, -1, -1])
    assert int(np.argmax(g)) == 2 and g[2] > 0.99


def test_orientation_takes_max_not_sum():
    """取 `max` 而非 `Σ`：两个记忆点在**相反**方向时，不得互相抵消成 0。"""
    g = _cos(_CUR, _NB, [_FLAT(20, 60), _FLAT(40, 60), -1, -1])   # 一北一南
    assert g[0] > 0.99, "北侧仍应是 +1（max，不是求和抵消）"
    assert g[1] > 0.99, "南侧也应是 +1（它对准南边那个记忆点）"


# ------------------------------------------------------------ ② 反例
def test_memory_equal_current_cell_gives_no_bias():
    """记忆格 == 当前格 ⇒ 方向为零向量 ⇒ **跳过**，不得给任何邻居偏置。"""
    assert np.allclose(_cos(_CUR, _NB, [_CUR, -1, -1, -1]), 0.0)


def test_empty_memory_gives_no_bias():
    assert np.allclose(_cos(_CUR, _NB, [-1, -1, -1, -1]), 0.0)


# ------------------------------------------------------------ ④b 经度环绕
def test_longitude_wrap_takes_shortest_path():
    """记忆在 cur(30,119) 的**东侧 2 格**（跨环绕到 col=1）⇒ 应判"东"，不是"西"。"""
    cur = _FLAT(30, 119)
    nb = [_FLAT(30, 0), _FLAT(30, 118), _FLAT(29, 119)]     # 东一步 / 西一步 / 北一步
    g = _cos(cur, nb, [_FLAT(30, 1), -1, -1, -1])
    assert int(np.argmax(g)) == 0 and g[0] > 0.99           # col=0 最对准
    assert g[1] < 0, "西一步应为背向（不环绕就会算成 + 大数 ⇒ 这条会失败）"


# ------------------------------------------------------------ 配置校验
def test_bad_mode_hard_fails():
    with pytest.raises(AssertionError, match="memory_gradient"):
        InfoStructureConfig(memory_gradient="bogus")
    with pytest.raises(AssertionError, match="朝向梯度增益"):
        InfoStructureConfig(memory_gradient="orientation", memory_gradient_gain=-1.0)


# ------------------------------------------------------------ ③ 开关
def _engine(*, mode: str = "none", gain: float = 0.3, enabled: bool = True,
            seed: int = 42, use_sim_core: bool = False) -> SphereEngine:
    cfg = SimConfig(seed=seed)
    cfg.simulation.use_sim_core = use_sim_core
    cfg.population.max_count = 600
    cfg.info_structure = InfoStructureConfig(
        enabled=enabled, learning_rate=0.05,
        memory_gradient=mode, memory_gradient_gain=gain)
    return SphereEngine(cfg)


def _digest(e: SphereEngine, ticks: int = 80):
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    return (int(e._flat.sum()), round(float(e._energy.sum()), 6), int(e.tick))


def test_switch_is_behaviorally_active():
    """`orientation` 与 `none` 的轨迹**必须不同** —— 否则"开关没接上"（静默 no-op，教训 11）。"""
    a = _digest(_engine(mode="none"))
    b = _digest(_engine(mode="orientation"))
    assert a != b, "两档逐位相同 ⇒ 朝向梯度根本没接进决策（静默失效）"
    # 但 population 规模同量级（不是把种群跑崩了）
    # ⚠️ 2026-09-23（T2）：eat_amount 0.5→0.9 后两档差异被放大到 ~7.4%
    #   （旧 5% 阈值基于 eat=0.5 环境；实测仍同量级，远非崩盘）⇒ 阈值 5% → 10%。
    assert abs(a[0] - b[0]) < 0.10 * max(a[0], 1)


def test_none_path_digest_pinned():
    """`none` 路径的行为钉死（防未来漂移）。

    ⚠️ 该 digest 取自 A′ 落地**之后**：`none` 分支与改动前**逐字相同**（`0.3 * perc * mem_in_nb`），
    故它同时是"原式未变"的回归锚点；与改动前的等价性另由全量回归中的既有对拍测试覆盖。
    """
    assert _digest(_engine(mode="none")) == (493827, 11724.467975, 80)


def test_orientation_reports_counters_and_none_is_na():
    """🔴 **先证"测到了"再判读**：`orientation` 必须出计数；`none` 下是 **None（未适用）不是 0**。"""
    e_o = _engine(mode="orientation")
    _digest(e_o, ticks=60)
    st = e_o.memory_gradient_stats()
    assert st and st["counters_available"] and st["decisions"] > 0
    assert st["slots_frac"] is not None and st["trigger_frac"] is not None
    e_n = _engine(mode="none")
    _digest(e_n, ticks=60)
    assert e_n.memory_gradient_stats() is None, "none 档必须报 n/a（不是 0）"


# ------------------------------------------------------------ ④ 双路径
def test_rust_and_python_paths_bitwise_identical():
    """Rust(`use_sim_core=True`) 与 Python 在 `orientation` 下**逐位一致**。

    ⚠️ 用 `enabled=False`：D2 的 softmax/噪声/冯诺依曼邻域只在 Python 路径实现 ⇒
    开着会掩盖本条真正要验的东西（**A′ 的双路径同式**）。
    """
    py = _digest(_engine(mode="orientation", enabled=False, use_sim_core=False), ticks=50)
    rs = _digest(_engine(mode="orientation", enabled=False, use_sim_core=True), ticks=50)
    assert py == rs, f"双路径不一致：python={py} rust={rs}"


def test_rust_path_reports_counters_na_not_zero():
    """Rust 侧只做决策、不产计数 ⇒ 必须报 **n/a**，**不得冒充 0**（防假读数）。"""
    e = _engine(mode="orientation", enabled=False, use_sim_core=True)
    _digest(e, ticks=50)
    st = e.memory_gradient_stats()
    assert st and st.get("counters_available") is False
    assert "n/a" in st.get("note", "")
