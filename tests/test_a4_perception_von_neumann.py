"""A4 回归测试：D2 感知半径 4 = Von Neumann（各向同性），不得为"北+西"单向偏置。

背景：旧实现用 `nb[:4]` 取 Moore 8 邻的"前 4 个"，而列序以 [上左,上,上右,左] 打头，
个体只能向北/向西移动 → 种群被驱赶至极区、招募崩塌（见
_archive/2026-10-10-退役团队-归档/docs-旧档/决策与评审/A4-崩溃溯源报告-20260912.md）。本测试锁死修复不被回退。
"""
import numpy as np

from simulation.config import InfoStructureConfig, SimConfig
from simulation.sphere_engine import SphereEngine
from world.sphere_world import SphereWorld


def test_von_neumann_is_orthogonal_4():
    """普通格的 Von Neumann 邻居恒为 4 个正交格（曼哈顿距离 1）。"""
    w = SphereWorld(60, 120)
    c = 30 * 120 + 60
    nb = w.neighbors_von_neumann(c)
    assert len(nb) == 4
    r0, c0 = c // 120, c % 120
    ds = {((int(n) // 120 - r0), (int(n) % 120 - c0)) for n in nb}
    assert ds == {(-1, 0), (1, 0), (0, -1), (0, 1)}, f"应为正交4邻，实得 {ds}"


def test_von_neumann_includes_south_and_east():
    """修复核心：4 邻必须同时含南向与北向（旧 bug 只有北向/同排）。"""
    w = SphereWorld(60, 120)
    for c in (10 * 120 + 5, 30 * 120 + 60, 49 * 120 + 117):
        r0 = c // 120
        nb = w.neighbors_von_neumann(c)
        drows = {int(n) // 120 - r0 for n in nb}
        assert -1 in drows and 1 in drows, f"cell {c} 缺北或南向: {drows}"


def test_old_bug_would_have_failed():
    """显式对照：旧 `nb[:4]` 在同一格上确实是"北+西"（证明测试有区分力）。"""
    w = SphereWorld(60, 120)
    c = 30 * 120 + 60
    r0 = c // 120
    old = w.neighbors(c)[:4]
    old_drows = {int(n) // 120 - r0 for n in old}
    assert 1 not in old_drows, "旧 nb[:4] 不应含南向（本断言记录 bug 形态）"
    new_drows = {int(n) // 120 - r0 for n in w.neighbors_von_neumann(c)}
    assert 1 in new_drows, "新实现必须含南向"


def test_von_neumann_pole_returns_four():
    """极点格（邻居为相邻整行）也取 4 个，且各向同性（按经度均匀）。"""
    w = SphereWorld(60, 120)
    for c in (0, 59 * 120 + 33):
        nb = w.neighbors_von_neumann(c)
        assert len(nb) == 4


def test_radius4_engine_can_move_south():
    """引擎级：radius=4 下，个体应能双向移动（旧 bug 只能向北）。"""
    cfg = SimConfig(seed=1)
    cfg.population.initial_count = 1
    cfg.population.max_count = 5000
    cfg.simulation.ticks = 400
    cfg.info_structure = InfoStructureConfig(
        enabled=True, perception_radius=4, perception_noise=0.0, softmax_tau=0.0
    )
    e = SphereEngine(cfg)
    e._genes[0, 0] = 1.0     # g0 移动概率
    e._genes[0, 14] = 1.0    # g14 高感知
    e._genes[0, 13] = 0.5    # g13 中性群居
    e._genes[0, 10] = 0.0    # g10 关邻格觅食
    e._genes[0, 2] = 1.0     # g2 极高繁殖门槛，避免繁殖扰乱单个体轨迹
    e._genes[0, 15] = 0.0    # g15 关信号（否则信号标记会吸引个体、干扰方向判定）
    e._genes[0, 19] = 0.0    # g19 关扎根（否则 move_prob 被压低甚至为 0）
    e.resources._grid[:] = e.resources._capacity[:]   # 各邻格食物相等
    e._flat[0] = 30 * 120 + 60
    start_row = int(e._flat[0]) // 120

    n_north = n_south = 0
    for _ in range(400):
        if len(e._id) == 0:
            break
        e._energy[0] = 100.0     # 低于繁殖门槛 270，且不饿死
        prev = int(e._flat[0]) // 120
        e.step()
        if len(e._id) == 0:
            break
        cur = int(e._flat[0]) // 120
        n_north += cur < prev
        n_south += cur > prev

    assert n_south > 0, "radius=4 应能向南移动（旧 nb[:4] 永不出现）"
    assert n_north > 0, "radius=4 应能向北移动"
