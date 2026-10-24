"""SphereWorld 世界模块回归测试（模块一 · 邻居表预计算优化）。

背景：neighbors() 从"每次现算坐标"改为"构造时预计算全网格邻居表、查询时
直接查行"。本文件保证这一改动是【行为零变化】的：
- 参照实现 = 优化前的逐格算法（flat_to_rc/clip/模/rc_to_flat），逐 bit 对拍；
- 覆盖默认 60×120、小号 8×12、非方形 5×7，含极点坍缩与经度环绕两条特殊路径。
"""
import numpy as np
import pytest

from world.sphere_world import SphereWorld


def reference_neighbors(w: SphereWorld, flat: int) -> np.ndarray:
    """优化前 neighbors() 的逐格算法（作为参照实现，独立于缓存表）。"""
    row, col = w.flat_to_rc(np.asarray(flat, dtype=np.int64))
    row, col = int(row), int(col)
    if row in (w._pole_top, w._pole_bottom):
        adj = w._pole_top + 1 if row == w._pole_top else w._pole_bottom - 1
        return (np.arange(w.cols) + adj * w.cols).astype(np.int64)
    drow = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int64)
    dcol = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int64)
    r = np.clip(row + drow, 0, w.rows - 1)
    c = (col + dcol) % w.cols
    return w.rc_to_flat(r, c)


@pytest.mark.parametrize("rows,cols", [(60, 120), (8, 12), (5, 7)])
def test_neighbors_cache_matches_reference_every_cell(rows, cols):
    """预计算邻居表与逐格参照算法在【全部格子】上逐位一致（含极点/环绕）。"""
    w = SphereWorld(rows=rows, cols=cols)
    for f in range(w.n_cells):
        new = w.neighbors(f)
        old = reference_neighbors(w, f)
        assert new.shape == old.shape and (new == old).all(), (
            f"({rows},{cols}) flat={f} 不一致: new={new} old={old}"
        )


def test_neighbors_core_properties():
    """形状与语义抽检：普通格 8 邻、极点格整行、极点邻居不越界。"""
    w = SphereWorld(rows=8, cols=12)
    # 普通格（行 1~6）：恒 8 个邻居，右邻含经度环绕
    nb = w.neighbors(1 * 12 + 0)  # (row=1, col=0)，左邻应环绕到 col 11
    assert nb.shape == (8,)
    assert 1 * 12 + 11 in nb          # 经度环绕：col0 的左邻 = 同行的 col11
    # 极点格：恒 cols 个邻居 = 相邻纬度带整行
    top_nb = w.neighbors(0 * 12 + 0)
    assert top_nb.shape == (12,)
    assert list(top_nb) == list(range(12, 24))          # 上极 → row1 整行
    bot_nb = w.neighbors(7 * 12 + 0)
    assert list(bot_nb) == list(range(72, 84))          # 下极 → row6 整行
    # 极点坍缩：极点行任意经度槽位查到的邻居相同（物理上只有一格）
    assert (w.neighbors(0 * 12 + 5) == top_nb).all()