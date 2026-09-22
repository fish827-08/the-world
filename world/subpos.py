"""亚格连续坐标（subpos）—— 把"每 tick 必走满 1 格"细化为 0.25 格粒度。

为什么需要
----------
现状（13.3 及以前）：移动决策从 8 邻格里选 1，移动**必然走满 1 格**，且候选
不含自身 ⇒ **没有"原地不动"**。后果是世界体感小：个体每 tick 都在跳整格，
既不能慢行、也不能停下来进食。

本模块提供**亚格坐标**：每格细分为 ``subdiv × subdiv`` 个亚位置
（默认 ``subdiv = 4`` ⇒ 最小步长 = **0.25 格**）。

核心不变式（关档逐位等价的依据）
--------------------------------
* 个体位置 = ``(sub_r, sub_c)``，整数，范围 ``[0, rows*subdiv) × [0, cols*subdiv)``。
* 派生格索引 ``flat = (sub_r // subdiv) * cols + (sub_c // subdiv)`` ——
  **取食 / 信号 / 尸体 / 资源 / 光照全部仍按 7200 整数格**，一行不改。
* 因此本机制**不增加任何格数组内存、不增加全格操作的每 tick 开销**；
  只增加 O(N) 的坐标派生（N ≈ 2000）。

🔴 极点处理
-----------
极点行（``row 0`` / ``row rows-1``）在物理上是**一个点**（见
``world.sphere_world.SphereWorld`` 的极点坍缩）。⇒ 落在极点行时把 ``sub_c``
钳到 ``[0, subdiv)``（即极点格 col 0 的亚位置），避免"flat 相同但亚格不同"
的幽灵位移。

🔴 与关档的关系
---------------
开关关闭时**完全不进本模块**（引擎走原整数格移动路径）⇒ 逐位等价。
本模块的 ``subdiv = 1`` 退化情形（``sub`` 即整数格）仅用于测试自检，
**不**作为关档的实现方式（关档必须走原路径，否则 RNG 消费顺序会变）。
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DEFAULT_SUBDIV",
    "init_from_flat",
    "flat_of_sub",
    "advance_sub",
    "speed_steps",
    "subpos_rf_cf",
    "subpos_latlon",
    "sphere_dist_rows",
]

#: 默认亚格细分数（每格 4×4 = 16 个亚位置 ⇒ 最小步长 0.25 格）。
DEFAULT_SUBDIV: int = 4


def _i64(x) -> NDArray[np.int64]:
    """把入参转成 int64 数组（标量也可）。"""
    return np.asarray(x, dtype=np.int64)


def _f64(x) -> NDArray[np.float64]:
    """把入参转成 float64 数组（标量也可）。"""
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------- 位置转换


def init_from_flat(flat, subdiv: int, world) -> tuple:
    """把整数格索引转成**处于格中心**的亚格坐标。

    参数
    ----
    flat : int | NDArray[int64]
        整数格平铺索引（0 .. rows*cols-1）。
    subdiv : int
        亚格细分数（>= 1）。``subdiv=4`` ⇒ 每格 16 个亚位置。
    world : SphereWorld
        世界拓扑（提供 ``flat_to_rc`` / ``rows`` / ``cols``）。

    返回
    ----
    (sub_r, sub_c) : (NDArray[int64], NDArray[int64])
        ``sub_r = row*subdiv + subdiv//2``、``sub_c = col*subdiv + subdiv//2``；
        取 ``subdiv//2`` 使代表点 ``sub/subdiv`` 恰为**格中心**（row + 0.5），
        因此本函数与 ``world._lat[row]`` 给出的纬度**严格一致**。

    说明
    ----
    极点行经 ``world.flat_to_rc`` 后 col 恒为 0（坍缩规则由世界模块负责），
    故这里无需特判。
    """
    if subdiv < 1:
        raise ValueError(f"subdiv 必须 >= 1，收到 {subdiv!r}")
    flat = _i64(flat)
    row, col = world.flat_to_rc(flat)
    half = np.int64(subdiv // 2)
    sub_r = (row.astype(np.int64) * np.int64(subdiv) + half)
    sub_c = (col.astype(np.int64) * np.int64(subdiv) + half)
    return sub_r.astype(np.int64), sub_c.astype(np.int64)


def flat_of_sub(sub_r, sub_c, subdiv: int, world) -> NDArray[np.int64]:
    """把亚格坐标**派生**为整数格索引（唯一定义处）。

    参数
    ----
    sub_r, sub_c : int | NDArray[int64]
        亚格坐标。``sub_c`` 允许越界（内部按经度环绕），``sub_r`` 越界未定义
        （调用方保证在 ``[0, rows*subdiv)`` 内，见 ``advance_sub``）。
    subdiv : int
        亚格细分数。
    world : SphereWorld
        世界拓扑。

    返回
    ----
    NDArray[int64] : 格索引。

    🔴 实现要点：极点坍缩与经度环绕**一律走 ``world.rc_to_flat``**，
    本模块不重复实现该规则 —— 否则两处规则会漂移（C9 类缺陷）。
    """
    if subdiv < 1:
        raise ValueError(f"subdiv 必须 >= 1，收到 {subdiv!r}")
    sub_r = _i64(sub_r)
    sub_c = _i64(sub_c)
    row = sub_r // np.int64(subdiv)
    col = sub_c // np.int64(subdiv)
    return np.asarray(world.rc_to_flat(row, col), dtype=np.int64)


# ---------------------------------------------------------------- 移动


def speed_steps(speed, subdiv: int) -> NDArray[np.int64]:
    """把连续速度（单位：格/tick）量化为**亚格步数**（四舍五入到最近档）。

    参数
    ----
    speed : float | NDArray[float64]
        期望位移（格）。常用范围 ``[0, 1]``；``>1`` 会被钳到 ``subdiv``
        （即每 tick 最多 1 格 —— 这是本批的硬约束，冲刺由 ``l2_dash`` 另管）。
    subdiv : int
        亚格细分数。

    返回
    ----
    NDArray[int64] : 每轴亚格步数，范围 ``[0, subdiv]``。

    映射（``subdiv = 4``）
    ---------------------
    ``0 → 0`` ｜ ``0.125–0.375 → 1 (0.25 格)`` ｜ ``0.375–0.625 → 2`` ｜
    ``0.625–0.875 → 3`` ｜ ``0.875–1.0 → 4 (1.0 格)``。
    边界取"四舍五入"（``+0.5`` 后下取整），故 ``speed=0`` **恰好**给 0 步
    ⇒ 个体可以**停在原地**（老逻辑没有这个选项）。
    """
    if subdiv < 1:
        raise ValueError(f"subdiv 必须 >= 1，收到 {subdiv!r}")
    s = _f64(speed)
    steps = np.floor(s * float(subdiv) + 0.5)
    return np.clip(steps, 0.0, float(subdiv)).astype(np.int64)


def advance_sub(sub_r, sub_c, drow, dcol, steps, subdiv: int, world) -> tuple:
    """按 (方向, 步数) 推进亚格坐标，处理**经度环绕**与**极点钳制**。

    参数
    ----
    sub_r, sub_c : int | NDArray[int64]
        当前亚格坐标。
    drow, dcol : int | NDArray[int64]
        方向分量，各取 ``-1 / 0 / +1``（与 ``world`` 的 Moore 8 邻一致；
        ``drow=0`` 且 ``dcol=0`` 表示不转向 = 原地停留）。
    steps : int | NDArray[int64]
        每轴亚格步数（``0 .. subdiv``），通常由 :func:`speed_steps` 给出。
    subdiv : int
        亚格细分数。
    world : SphereWorld
        世界拓扑（提供 ``rows`` / ``cols``）。

    返回
    ----
    (sub_r, sub_c) : 新亚格坐标（已钳制 / 环绕 / 极点归位）。

    行为
    ----
    * **纬度不环绕**：``sub_r`` 钳到 ``[0, rows*subdiv-1]``（与 ``rc_to_flat``
      的 "纬度方向钳回界内" 一致）。
    * **经度环绕**：``sub_c %= cols*subdiv``（球面东西方向是闭环）。
    * **极点钳制**：结果落在极点行时，``sub_c`` 钳到 ``[0, subdiv-1]``
      —— 极点物理上是一个点，不允许沿纬度带"滑动"。
    """
    if subdiv < 1:
        raise ValueError(f"subdiv 必须 >= 1，收到 {subdiv!r}")
    sub_r = _i64(sub_r)
    sub_c = _i64(sub_c)
    drow = _i64(drow)
    dcol = _i64(dcol)
    steps = _i64(steps)

    r2 = sub_r + drow * steps
    c2 = sub_c + dcol * steps

    # 纬度：不环绕，钳回界内
    r2 = np.clip(r2, 0, world.rows * subdiv - 1)
    # 经度：环绕
    c2 = np.mod(c2, world.cols * subdiv)

    # 极点：物理坍缩为一点 ⇒ sub_c 归位到 col 0 的亚位置
    row2 = r2 // np.int64(subdiv)
    pole = (row2 == 0) | (row2 == world.rows - 1)
    c2 = np.where(pole, np.clip(c2, 0, subdiv - 1), c2)

    return r2.astype(np.int64), c2.astype(np.int64)


# ---------------------------------------------------------------- 几何


def subpos_rf_cf(sub_r, sub_c, subdiv: int) -> tuple:
    """亚格坐标 → **连续行/列坐标**（单位：格）。

    取代表点 ``sub / subdiv``（亚格左下角），因此
    ``init_from_flat`` 给出的格中心（``row*subdiv + subdiv//2``）映到
    ``row + 0.5`` —— 与 ``world._lat[row]`` 严格一致。
    """
    if subdiv < 1:
        raise ValueError(f"subdiv 必须 >= 1，收到 {subdiv!r}")
    return (_f64(sub_r) / float(subdiv), _f64(sub_c) / float(subdiv))


def subpos_latlon(sub_r, sub_c, subdiv: int, world) -> tuple:
    """亚格坐标 → (纬度, 经度)，单位弧度。

    纬度 ``= -π/2 + rf * (π/rows)``（``rf`` 为连续行坐标，见 :func:`subpos_rf_cf`）；
    经度 ``= cf * (2π/cols)``。
    """
    rf, cf = subpos_rf_cf(sub_r, sub_c, subdiv)
    lat = -np.pi / 2.0 + rf * (np.pi / float(world.rows))
    lon = cf * (2.0 * np.pi / float(world.cols))
    return lat, lon


def sphere_dist_rows(rf1, cf1, rf2, cf2, world) -> NDArray[np.float64]:
    """**球面大圆距离**，单位 = "一个纬度格宽"。

    为什么不用跳数
    --------------
    "跳几格"在球面上**不是距离**：同一个 d=2，在赤道是 8 个格，在极区是
    **整条纬度带**（``cols`` 个格）。用跳数当半径 ⇒ 同一个基因在不同纬度
    买到完全不同的物理距离（R148 已登记为构造不公平）。

    本函数给出物理一致的距离：南北相邻两格 = **1.0**；赤道东西相邻两格
    ≈ 1.0（因 60×120 网格下 ``π/rows == 2π/cols``）；纬度 φ 处的东邻格
    ≈ ``cos φ``；极区 ≈ 0.05。

    参数
    ----
    rf1, cf1, rf2, cf2 : float | NDArray[float64]
        **连续**行 / 列坐标（单位：格）。
    world : SphereWorld
        世界拓扑。

    返回
    ----
    NDArray[float64] : 大圆距离（纬度格宽为单位），范围 ``[0, rows]``。
    """
    dlat = np.pi / float(world.rows)
    dlon = 2.0 * np.pi / float(world.cols)

    lat1 = -np.pi / 2.0 + _f64(rf1) * dlat
    lon1 = _f64(cf1) * dlon
    lat2 = -np.pi / 2.0 + _f64(rf2) * dlat
    lon2 = _f64(cf2) * dlon

    cos1, sin1 = np.cos(lat1), np.sin(lat1)
    cos2, sin2 = np.cos(lat2), np.sin(lat2)
    dlon_diff = lon1 - lon2

    # 🔴 必须用 atan2 形式，**不能用 arccos(dot)**：
    # arccos 在 dot→1 处导数发散 ⇒ 同点距离会算出 ~3e-7 格的伪残差，
    # 而 "自己到自己 = 0" 会被波 2 的视野筛选（d <= r）逐个体调用，
    # 伪残差会让判定对同一位置上/下取整抖动。atan2(|cross|, dot) 在 θ→0 稳定。
    x = sin1 * sin2 + cos1 * cos2 * np.cos(dlon_diff)
    y = np.sqrt(
        (cos2 * np.sin(dlon_diff)) ** 2
        + (cos1 * sin2 - sin1 * cos2 * np.cos(dlon_diff)) ** 2
    )
    theta = np.arctan2(y, x)
    return (theta / dlat).astype(np.float64)
