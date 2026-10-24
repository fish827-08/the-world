"""A19.3 回归：风平流算子 + 气味场接线（默认关）。

写死账本之外的一条纪律：**平流只作用于被动诊断量**（气味），不参与任何守恒账本。
本测试钉三件事：
1. 无风 ⇒ **逐位恒等**（连浮点误差都不许有）；
2. 均匀东风 ⇒ 质心按解析位移移动（`u·Δt/(R·cosφ·Δλ)` 格），质量近似守恒；
3. `SmellConfig.advect_with_wind=False`（默认）⇒ `set_wind` 完全无效（逐位不变）；
   `True` ⇒ 浓度质心沿风向移动。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import SmellConfig  # noqa: E402
from world.advect import advect_scalar, centroid  # noqa: E402
from world.smell_field import SmellField  # noqa: E402
from world.sphere_world import SphereWorld  # noqa: E402

TIER = "fast"
ROWS, COLS = 30, 60


def _blob(rows=ROWS, cols=COLS, r0=None, c0=None, sigma=2.0) -> np.ndarray:
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    r0 = rows // 2 if r0 is None else r0
    c0 = cols // 4 if c0 is None else c0
    d2 = (rr - r0) ** 2 + np.minimum(np.abs(cc - c0), cols - np.abs(cc - c0)) ** 2
    return np.exp(-d2 / (2.0 * sigma ** 2))


def test_zero_wind_is_bitwise_identity():
    f = _blob()
    out = advect_scalar(f, 0.0, 0.0, 3600.0)
    assert np.array_equal(out, f), "无风时必须逐位恒等（否则关档会漂）"


def test_uniform_east_wind_shifts_centroid_by_analytic_amount():
    f = _blob()
    u = 10.0
    dt = 7200.0
    out = advect_scalar(f, u, 0.0, dt)
    rows, cols = f.shape
    dlam = 2.0 * np.pi / cols
    phi = np.radians(-90.0 + (rows // 2 + 0.5) * 180.0 / rows)
    expect = u * dt / (6.371e6 * np.cos(phi) * dlam)      # 格
    _, c_before = centroid(f)
    _, c_after = centroid(out)
    d = (c_after - c_before) % cols
    assert d == pytest.approx(expect, abs=0.08), f"位移 {d:.3f} 格 ≠ 解析 {expect:.3f} 格"
    assert out.sum() == pytest.approx(f.sum(), rel=2e-3), "双线性平流不该明显增减总量"


def test_wrap_across_dateline():
    f = _blob(c0=COLS - 1)                 # 贴着接缝
    out = advect_scalar(f, 15.0, 0.0, 3600.0 * 4)
    assert np.isfinite(out).all()
    assert out.sum() == pytest.approx(f.sum(), rel=5e-3), "跨接缝质量应基本保住"


def _smell(advect: bool) -> SmellField:
    w = SphereWorld(ROWS, COLS)
    cfg = SmellConfig(channels=("food",), update_every=1, decay=1.0,
                      inject_food=1.0, advect_with_wind=advect)
    return SmellField(w, cfg)


def test_smell_default_off_ignores_wind():
    sm = _smell(False)
    cell = (ROWS // 2) * COLS + COLS // 4
    sm.update({"food": (np.array([cell]), np.array([1.0]))}, tick=0)
    before = sm._S[0].copy()
    sm.set_wind(20.0, 0.0, dt_s=3600.0)
    sm.update({}, tick=1)
    assert np.array_equal(sm._S[0], before), "默认关时 set_wind 不许有任何效果"


def test_smell_with_wind_moves_centroid():
    sm = _smell(True)
    cell = (ROWS // 2) * COLS + COLS // 4
    sm.update({"food": (np.array([cell]), np.array([1.0]))}, tick=0)
    before = sm._S[0].reshape(ROWS, COLS).copy()
    sm.set_wind(25.0, 0.0, dt_s=12 * 3600.0)   # 25 m/s × 12 h ≈ 1.6 格（60 列、赤道）
    sm.update({}, tick=1)
    after = sm._S[0].reshape(ROWS, COLS)
    _, c0 = centroid(before)
    _, c1 = centroid(after)
    assert ((c1 - c0) % COLS) > 1.0, f"浓度质心没沿风向移动（Δ={((c1 - c0) % COLS):.2f} 格）"
