"""A18.1 回归：NPP 三种形式（multiplicative / liebig / light_temp_only）。

写死审计 H3 的验收点
--------------------
1. **默认形式 = 现状**：`form="multiplicative"` 与不传 form **逐位相等**（不是"近似"，是 bitwise）。
2. **三种形式的定义差异**：干格（P=0）下 mul/liebig 必为 0，light_temp_only **不为 0**（它不看水）。
3. **极温**：三种形式都趋于 0（`ft` 指数衰减），防止"形式一换就把极端处变成绿洲"。
4. **liebig 是下界**：`liebig ≤ min(光, 温, 水, 干旱)` 这条数学性质不能被改坏。
5. 未知形式 / 非法配置 ⇒ 报错（不许静默退回默认）。

规模：纯数组运算，< 0.1 s；tier = fast。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import NPPConfig  # noqa: E402
from world.npp import compute_npp  # noqa: E402

TIER = "fast"


def _grid():
    lat = np.array([0.0, -1.2, 1.2, -1.4, 0.0])      # 赤道 / 中纬 / 高纬
    temp = np.array([25.0, 25.0, 25.0, 25.0, -60.0])  # 前三常温，最后一个极冷
    precip = np.array([50.0, 0.0, 5.0, 50.0, 50.0])   # 第二个完全干
    return lat, temp, precip


def test_default_form_is_bitwise_identical_to_multiplicative():
    lat, t, p = _grid()
    a = compute_npp(t, p, lat)
    b = compute_npp(t, p, lat, form="multiplicative")
    assert np.array_equal(a, b), "默认形式必须与显式 multiplicative 逐位相同"


def test_dry_cell_behavior_differs_by_form():
    lat, t, p = _grid()
    mul = compute_npp(t, p, lat, form="multiplicative")
    lie = compute_npp(t, p, lat, form="liebig")
    lto = compute_npp(t, p, lat, form="light_temp_only")
    i = 1                                   # P=0 的干格
    assert mul[i] == 0.0 and lie[i] == 0.0, "有水的形式在零水下必须为 0"
    assert lto[i] > 0.0, "light_temp_only 不看水 ⇒ 干格仍有产量（这正是它的定义）"


def test_extreme_temperature_kills_all_forms():
    """极冷格相对常温格必须趋近 0（注意：compute_npp 会按均值归一，**绝对值没有意义**，
    所以只能比"冷格 / 暖格"的比值，不能断言绝对值为 0）。"""
    lat = np.array([0.0, 0.0, 0.0, 0.0])
    t = np.array([25.0, 25.0, 25.0, -60.0])
    p = np.full(4, 50.0)
    for form in ("multiplicative", "liebig", "light_temp_only"):
        v = compute_npp(t, p, lat, form=form)
        assert v[3] < 1e-3 * v[0], f"{form}: 极冷/常温 = {v[3] / v[0]:.3g}（应趋近 0）"


def test_liebig_is_the_minimum_of_factors():
    """数学性质：liebig ≤ 每个单项因子（归一化前后比值不变 ⇒ 直接比同输入的两形式）。"""
    lat, t, p = _grid()
    lie = compute_npp(t, p, lat, form="liebig", light_exponent=1.0)
    # 用同样输入算单项：光=cos^1、温=exp(...)、水=p/(p+15)、干旱=p/(p+20+2T)
    light = np.clip(np.cos(lat), 0.02, None)
    ft = np.exp(-((t - 25.0) / 12.0) ** 2)
    fp = p / (p + 15.0)
    arid = p / (p + 20.0 + 2.0 * np.clip(t, 0.0, None))
    raw_min = np.minimum(np.minimum(light, ft), np.minimum(fp, arid))
    # 归一化是同一个常数 ⇒ 比值应处处相等（除零处单独比）
    ok = raw_min > 0
    assert np.allclose(lie[ok] / raw_min[ok], lie[ok][0] / raw_min[ok][0], rtol=1e-12)


def test_unknown_form_raises():
    lat, t, p = _grid()
    with pytest.raises(ValueError):
        compute_npp(t, p, lat, form="magic")
    with pytest.raises(AssertionError):
        NPPConfig(form="magic")


def test_light_temp_only_is_water_blind():
    """同一个光温、不同水 ⇒ 产量完全一样（水盲）。"""
    lat = np.array([0.0, 0.0])
    t = np.array([25.0, 25.0])
    dry = compute_npp(t, np.array([0.0, 0.0]), lat, form="light_temp_only")
    wet = compute_npp(t, np.array([500.0, 500.0]), lat, form="light_temp_only")
    assert np.array_equal(dry, wet)
