"""`world/subpos.py` 单元测试 —— 亚格连续坐标（13.4 波 1）。

覆盖的不变式
------------
1. **往返一致**：``flat → (sub_r, sub_c) → flat`` 恒等（含极点行、经度首尾）。
2. **格中心对齐**：``init_from_flat`` 的连续行坐标 == ``row + 0.5``，
   且纬度与 ``world._lat[row]`` 严格相等（几何基准不得漂移）。
3. **量化表**：``speed_steps`` 的档位边界与钳制。
4. **位移拓扑**：经度环绕、纬度不环绕、极点归位。
5. **球面距离**：南北邻格 == 1.0、赤道东西邻格 ≈ 1.0、同点 == 0、
   极区东西邻格 ≪ 1（"跳数 ≠ 球面距离"的量化登记）、对称性。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world.sphere_world import SphereWorld  # noqa: E402
from world.subpos import (  # noqa: E402
    DEFAULT_SUBDIV,
    advance_sub,
    flat_of_sub,
    init_from_flat,
    speed_steps,
    sphere_dist_rows,
    subpos_latlon,
    subpos_rf_cf,
)


@pytest.fixture
def world() -> SphereWorld:
    return SphereWorld(60, 120)


# ------------------------------------------------------------ 1. 往返一致


def _canonical(flats: np.ndarray, world: SphereWorld) -> np.ndarray:
    """`flat` 的**规范形式**：极点行的冗余槽位归并到 col 0。

    为什么需要：世界模块的极点行是**多对一**的（row 0 / row 59 的任意 col
    都坍缩为同一物理格），因此 `flat_to_rc` 与 `rc_to_flat` 在该行**不互逆**
    —— `flat=1`（极点行 col 1）是冗余槽位，其规范形式是 `flat=0`。
    这是世界模块的既有语义，不是本模块的缺陷。
    """
    row, col = world.flat_to_rc(flats)
    return np.asarray(world.rc_to_flat(row, col), dtype=np.int64)


@pytest.mark.parametrize("subdiv", [1, 2, 4])
def test_flat_roundtrip_is_idempotent_and_canonical(
    world: SphereWorld, subdiv: int
) -> None:
    """`flat -> subpos -> flat` 必须给出**规范形式**，且**幂等**（再走一遍不变）。

    ⚠️ 断言不得写成"逐位还原" —— 极点行有多对一冗余槽位（见 `_canonical`）。
    """
    rng = np.random.default_rng(20260923)
    flats = rng.integers(0, world.n_cells, size=500, dtype=np.int64)
    # 手工塞入极端位置：两极点、首尾行、经度首尾列
    flats = np.concatenate([
        flats,
        np.array([0, world.cols, world.cols - 1,
                  (world.rows - 1) * world.cols,
                  (world.rows - 2) * world.cols + world.cols - 1], dtype=np.int64),
    ])
    sub_r, sub_c = init_from_flat(flats, subdiv, world)
    back = flat_of_sub(sub_r, sub_c, subdiv, world)
    want = _canonical(flats, world)
    assert np.array_equal(back, want), (
        f"往返未给出规范形式：{int((back != want).sum())} / {len(flats)} 格偏离"
        f"（subdiv={subdiv}）"
    )
    # 幂等：对规范形式再走一遍 → 不变
    sub_r2, sub_c2 = init_from_flat(back, subdiv, world)
    back2 = flat_of_sub(sub_r2, sub_c2, subdiv, world)
    assert np.array_equal(back2, back), "往返不幂等（第二次结果又变了）"


@pytest.mark.parametrize("subdiv", [1, 2, 4])
def test_flat_roundtrip_strict_off_pole_rows(world: SphereWorld, subdiv: int) -> None:
    """**非极点行**必须严格逐位还原（那里 `flat_to_rc` / `rc_to_flat` 互逆）。"""
    rng = np.random.default_rng(99)
    rows = rng.integers(1, world.rows - 1, size=400, dtype=np.int64)  # 排除 0 / rows-1
    cols = rng.integers(0, world.cols, size=400, dtype=np.int64)
    flats = rows * world.cols + cols
    sub_r, sub_c = init_from_flat(flats, subdiv, world)
    back = flat_of_sub(sub_r, sub_c, subdiv, world)
    assert np.array_equal(back, flats), (
        f"非极点行往返不严格：{int((back != flats).sum())} 格偏离（subdiv={subdiv}）"
    )


def test_subdiv_one_degenerates_to_integer_grid(world: SphereWorld) -> None:
    """`subdiv=1` 时亚格坐标 == (row, col)，且 flat 派生回到规范形式。

    这条是**自检**：说明本模块在退化情形不与整数格世界冲突
    （⚠️ 关档仍走引擎原路径，不走这里 —— 见模块 docstring）。
    """
    flats = np.array([0, 121, 119, 120, 3600, 7199], dtype=np.int64)
    sub_r, sub_c = init_from_flat(flats, 1, world)
    row, col = world.flat_to_rc(flats)
    # subdiv=1 时 half=0 ⇒ 亚格坐标就是 (row, col)
    assert np.array_equal(sub_r, row)
    assert np.array_equal(sub_c, col)
    assert np.array_equal(flat_of_sub(sub_r, sub_c, 1, world), _canonical(flats, world))


# ------------------------------------------------------------ 2. 格中心对齐


def test_cell_centre_matches_world_latitude(world: SphereWorld) -> None:
    """格中心的连续行坐标必须 == row + 0.5，纬度与 `world._lat` 严格一致。"""
    s = DEFAULT_SUBDIV
    rows = np.arange(world.rows, dtype=np.int64)
    flats = rows * world.cols  # 每行 col 0
    sub_r, sub_c = init_from_flat(flats, s, world)
    rf, _ = subpos_rf_cf(sub_r, sub_c, s)
    assert np.allclose(rf, rows + 0.5, atol=1e-12), (
        f"格中心行坐标偏移：max |Δ| = {np.abs(rf - (rows + 0.5)).max():.3e}"
    )
    lat, _ = subpos_latlon(sub_r, sub_c, s, world)
    assert np.allclose(lat, world._lat, atol=1e-12), (
        f"纬度与 world._lat 不一致：max |Δ| = {np.abs(lat - world._lat).max():.3e}"
    )


# ------------------------------------------------------------ 3. 量化表


def test_speed_steps_quantisation_table() -> None:
    """`subdiv=4` 的档位边界（四舍五入）与钳制。"""
    s = 4
    cases = {
        0.0: 0, 0.10: 0, 0.124: 0,          # 0 档
        0.125: 1, 0.25: 1, 0.374: 1,        # 0.25 格
        0.375: 2, 0.5: 2, 0.624: 2,         # 0.5 格
        0.625: 3, 0.75: 3, 0.874: 3,        # 0.75 格
        0.875: 4, 1.0: 4,                   # 1.0 格
    }
    for speed, want in cases.items():
        got = int(speed_steps(speed, s))
        assert got == want, f"speed={speed} 期望 {want} 亚格，实得 {got}"


def test_speed_steps_clamps_out_of_range() -> None:
    """负速度 ⇒ 0（不许倒退）；速度 > 1 格 ⇒ 钳到 subdiv（每 tick 最多 1 格）。"""
    assert int(speed_steps(-0.5, 4)) == 0
    assert int(speed_steps(1.9, 4)) == 4
    assert int(speed_steps(3.0, 4)) == 4


def test_speed_steps_is_vectorised() -> None:
    """数组入参逐元素量化（移动段按 N 个体一次性调用）。"""
    got = speed_steps(np.array([0.0, 0.25, 0.5, 0.75, 1.0]), 4)
    assert np.array_equal(got, np.array([0, 1, 2, 3, 4], dtype=np.int64))


# ------------------------------------------------------------ 4. 位移拓扑


def test_advance_zero_steps_stays_put(world: SphereWorld) -> None:
    """`steps=0` ⇒ 位置不变（这是老逻辑**没有**的"原地不动"）。"""
    s = DEFAULT_SUBDIV
    flat = np.array([3600], dtype=np.int64)
    sub_r, sub_c = init_from_flat(flat, s, world)
    r2, c2 = advance_sub(sub_r, sub_c, np.array([1]), np.array([1]),
                         np.array([0]), s, world)
    assert int(r2[0]) == int(sub_r[0]) and int(c2[0]) == int(sub_c[0])


def test_advance_longitude_wraps(world: SphereWorld) -> None:
    """经度环绕：从 col 0 向西走一步 ⇒ 落到 col cols-1 的同一亚位置。"""
    s = DEFAULT_SUBDIV
    flat = np.array([30 * world.cols + 0], dtype=np.int64)  # row 30, col 0
    sub_r, sub_c = init_from_flat(flat, s, world)
    r2, c2 = advance_sub(sub_r, sub_c, np.array([0]), np.array([-1]),
                         np.array([s]), s, world)
    back = flat_of_sub(r2, c2, s, world)
    assert int(back[0]) == 30 * world.cols + (world.cols - 1), (
        f"经度未环绕：flat={int(back[0])}"
    )
    assert int(r2[0]) == int(sub_r[0]), "经度环绕不应改变纬度"


def test_advance_latitude_does_not_wrap(world: SphereWorld) -> None:
    """纬度不环绕：极点行继续向北 ⇒ 钳在极点行（与 world 的规则一致）。"""
    s = DEFAULT_SUBDIV
    flat = np.array([0], dtype=np.int64)  # 上极（坍缩为 col 0）
    sub_r, sub_c = init_from_flat(flat, s, world)
    r2, c2 = advance_sub(sub_r, sub_c, np.array([-1]), np.array([0]),
                         np.array([s]), s, world)
    assert int(r2[0]) == 0, f"纬度越界：sub_r={int(r2[0])}"
    assert int(flat_of_sub(r2, c2, s, world)[0]) == 0


def test_advance_pole_clamps_subcol(world: SphereWorld) -> None:
    """🔴 极点归位：从 row 1 走进极点行时，sub_c 必须钳到 col 0 的亚位置。

    否则会出现 "flat 相同（都是 0）但亚格不同" 的幽灵位移：
    个体在极点上继续"沿纬度带滑动"，而物理上极点是一个点。
    """
    s = DEFAULT_SUBDIV
    # row 1 的 col 60（经度中部），向北一步 ⇒ 进极点行 col? ⇒ 坍缩为 0
    flat = np.array([1 * world.cols + 60], dtype=np.int64)
    sub_r, sub_c = init_from_flat(flat, s, world)
    r2, c2 = advance_sub(sub_r, sub_c, np.array([-1]), np.array([0]),
                         np.array([s]), s, world)
    assert int(r2[0] // s) == 0, "应落入上极行"
    assert 0 <= int(c2[0]) < s, f"极点行 sub_c 未钳制：{int(c2[0])}"
    assert int(flat_of_sub(r2, c2, s, world)[0]) == 0


# ------------------------------------------------------------ 5. 球面距离


def test_sphere_dist_north_south_neighbour_is_one(world: SphereWorld) -> None:
    """南北相邻两格（格中心）距离恒为 1.0 —— 这是距离单位的定义。"""
    rf = np.array([10.5, 29.5, 45.5])
    cf = np.array([60.5, 60.5, 60.5])
    d = sphere_dist_rows(rf, cf, rf + 1.0, cf, world)
    assert np.allclose(d, 1.0, atol=1e-9), f"南北邻格距离 = {d}"


def test_sphere_dist_equatorial_east_west_is_one(world: SphereWorld) -> None:
    """赤道附近东西相邻两格距离 ≈ 1.0（60×120 网格下 π/rows == 2π/cols）。"""
    rf = np.array([29.5, 30.5])
    cf = np.array([60.5, 60.5])
    d = sphere_dist_rows(rf, cf, rf, cf + 1.0, world)
    assert np.allclose(d, 1.0, atol=1e-3), f"赤道东西邻格距离 = {d}"


def test_sphere_dist_self_is_zero(world: SphereWorld) -> None:
    d = sphere_dist_rows(np.array([30.5]), np.array([60.5]),
                         np.array([30.5]), np.array([60.5]), world)
    assert float(d[0]) == pytest.approx(0.0, abs=1e-12)


def test_sphere_dist_is_symmetric(world: SphereWorld) -> None:
    rng = np.random.default_rng(7)
    rf1, cf1 = rng.uniform(0, 60, 200), rng.uniform(0, 120, 200)
    rf2, cf2 = rng.uniform(0, 60, 200), rng.uniform(0, 120, 200)
    d1 = sphere_dist_rows(rf1, cf1, rf2, cf2, world)
    d2 = sphere_dist_rows(rf2, cf2, rf1, cf1, world)
    assert np.allclose(d1, d2, atol=1e-12)


def test_sphere_dist_polar_east_west_is_tiny(world: SphereWorld) -> None:
    """🔴 定量登记"跳数 ≠ 球面距离"：极区东西相邻两格距离 ≈ cos(87°) ≈ 0.05 格。

    含义：在极点带，"相邻 2 格"覆盖的**物理距离**只有赤道的约 1/20；
    因此任何按跳数定义的视野半径，在极区都会系统性放大作用范围
    （R148 已登记为构造不公平，波 2 的 `hunt_radius` 必须用本函数）。
    """
    pole_rf = np.array([59.5])          # row 59 = 下极行中心
    d = sphere_dist_rows(pole_rf, np.array([60.5]), pole_rf, np.array([61.5]), world)
    assert float(d[0]) < 0.1, f"极区东西邻格距离应 ≪ 1，实得 {float(d[0]):.4f}"
    # 赤道处同操作应 ≈ 1.0（对照）
    d_eq = sphere_dist_rows(np.array([29.5]), np.array([60.5]),
                            np.array([29.5]), np.array([61.5]), world)
    assert float(d_eq[0]) == pytest.approx(1.0, abs=1e-2)


def test_sphere_dist_radius_two_covers_more_at_pole(world: SphereWorld) -> None:
    """"d <= 2 格"在极区覆盖的**格数**远多于赤道（量化 R148 登记的偏差）。"""
    # 采样整条纬度带，数落在 d<=2 内的格占比
    def covered(row: int) -> int:
        rf = float(row) + 0.5
        cols = np.arange(world.cols, dtype=np.float64) + 0.5
        d = sphere_dist_rows(np.full_like(cols, rf), np.full_like(cols, 60.5),
                             np.full_like(cols, rf), cols, world)
        return int((d <= 2.0).sum())

    eq = covered(30)
    pole = covered(59)
    assert pole > eq, f"极区覆盖格数应多于赤道（赤道 {eq} / 极点 {pole}）"
