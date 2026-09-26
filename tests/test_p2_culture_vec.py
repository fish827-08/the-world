"""P2 向量化对拍：文化学习段（``SphereEngine._culture_learn_python``）。

参考实现 = 旧逐个体循环版（S1d 稀疏字典 + ``sorted(set(...))`` +
``(K,16).mean(axis=0)``，见下方 `_ref_culture_learn`），与被测的批式向量化实现
在**同一随机状态**上逐位比较（``view(np.int64).tobytes()``，含 ±0.0 区分）。

覆盖面：极点带未成年（邻居 = 整带 cols 格）、极点邻带（邻居表含重复格 ⇒
必须去重）、同格密集（同组多未成年）、孤立未成年（邻域无成年 ⇒ 不动）、
K=1 单成年邻居、以及混合量级数据（1e-6~1e6，逼出求和顺序差异）。
"""

import numpy as np
import pytest

from simulation.config import SimConfig
from simulation.sphere_engine import SphereEngine


def _make(seed=1, n=8, rows=6, cols=8):
    cfg = SimConfig(seed=seed)
    cfg.world.rows = rows
    cfg.world.cols = cols
    cfg.population.initial_count = max(n, 1)
    return SphereEngine(cfg)


def _ref_culture_learn(e, j_idx, adult_qual):
    """旧实现（逐个体 Python 循环）—— 语义参考，勿改。"""
    interp = e._interpret.copy()
    cell_adults: dict[int, list[int]] = {}
    for a in np.flatnonzero(adult_qual):
        cell_adults.setdefault(int(e._flat[int(a)]), []).append(int(a))
    for idx in j_idx:
        c = int(e._flat[int(idx)])
        nb = e.world.neighbors(c)
        lst: list[int] = []
        for nc in nb:
            v = cell_adults.get(int(nc))
            if v:
                lst.extend(v)
        if lst:
            lst = sorted(set(lst))
            mean_interpret = interp[lst].mean(axis=0)
            interp[int(idx)] += 0.1 * (mean_interpret - interp[int(idx)])
    return interp


def _bits(a):
    return a.view(np.int64).tobytes()


def _scatter(e, n, rng):
    """随机铺位：年龄混合 + 混合量级解读表 + 强制覆盖特殊格。"""
    n_cells, cols = e.world.n_cells, e.world.cols
    flat = rng.integers(0, n_cells, size=n)
    if n >= 4 * cols:
        flat[0:cols] = np.arange(cols)                             # 上极带
        flat[cols:2 * cols] = np.arange(n_cells - cols, n_cells)   # 下极带
        flat[2 * cols:3 * cols] = np.arange(cols, 2 * cols)        # 上极邻带（邻居表有重复格）
        flat[3 * cols:4 * cols] = np.arange(cols)                  # 上极带内聚堆（同格多人）
    e._flat[:] = flat
    adult_qual = rng.random(n) < 0.5
    adult_qual[0] = True
    e._interpret[:] = (rng.standard_normal((n, 16))
                       * 10.0 ** rng.integers(-6, 7, size=(n, 1)))
    return adult_qual


@pytest.mark.parametrize("seed,n", [(1, 40), (2, 41), (3, 60), (7, 61)])
def test_bit_exact_vs_reference(seed, n):
    e = _make(seed=seed, n=n)
    rng = np.random.default_rng(seed)
    adult_qual = _scatter(e, n, rng)
    j_idx = np.flatnonzero(~adult_qual)
    assert j_idx.size > 0
    ref = _ref_culture_learn(e, j_idx, adult_qual)
    e._culture_learn_python(j_idx, adult_qual)
    assert _bits(e._interpret) == _bits(ref), "批式实现与旧循环参考不逐位一致"


def test_pole_and_dense_are_covered():
    """覆盖度自检：随机用例确实踩到极点格与同格多未成年（否则对拍价值打折）。"""
    e = _make(seed=5, n=40)
    rng = np.random.default_rng(5)
    _scatter(e, 40, rng)
    j_idx = np.flatnonzero(np.arange(40) % 2 == 1)
    assert np.asarray(e.world.is_pole(e._flat[j_idx])).any(), "未覆盖极点未成年"
    _, counts = np.unique(e._flat[j_idx], return_counts=True)
    assert counts.max() >= 2, "未覆盖同格多未成年"


def test_no_adults_is_noop():
    e = _make(seed=9, n=12)
    rng = np.random.default_rng(9)
    e._flat[:] = rng.integers(0, e.world.n_cells, size=12)
    e._interpret[:] = rng.standard_normal((12, 16))
    before = e._interpret.copy()
    e._culture_learn_python(np.arange(12), np.zeros(12, dtype=bool))
    assert _bits(e._interpret) == _bits(before)


def test_isolated_juvenile_untouched():
    """邻域内无成年 ⇒ 该未成年解读表不动（旧实现 `if _lst:` 分支）。"""
    e = _make(seed=11, n=3)
    rng = np.random.default_rng(11)
    e._flat[:] = [20, 21, 22]
    e._interpret[:] = rng.standard_normal((3, 16))
    before = e._interpret.copy()
    e._culture_learn_python(np.array([0, 1, 2]), np.zeros(3, dtype=bool))
    assert _bits(e._interpret) == _bits(before)


def test_same_cell_adult_not_counted():
    """同格成年**不属于**邻域（邻居表不含自身格）⇒ 均值集为空、不动。"""
    e = _make(seed=13, n=2)
    e._flat[:] = [15, 15]
    e._interpret[0] = 0.0
    e._interpret[1] = 1.0
    before = e._interpret.copy()
    e._culture_learn_python(np.array([0]), np.array([False, True]))
    assert _bits(e._interpret) == _bits(before)


def test_single_adult_neighbor_k1():
    """K=1：均值 = 该成年行自身；且与参考实现逐位一致。"""
    e = _make(seed=17, n=2)
    rng = np.random.default_rng(17)
    e._flat[:] = [20, 21]  # 相邻两格（cols=8：col 4/5 同行）
    e._interpret[:] = rng.standard_normal((2, 16)) * 1e5
    row0 = e._interpret[0].copy()
    row1 = e._interpret[1].copy()
    ref = _ref_culture_learn(e, np.array([0]), np.array([False, True]))
    e._culture_learn_python(np.array([0]), np.array([False, True]))
    assert _bits(e._interpret) == _bits(ref)
    assert _bits(e._interpret[0]) == _bits(row0 + 0.1 * (row1 - row0))
