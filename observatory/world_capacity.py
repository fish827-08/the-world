#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C8 前提对账仪器：容量图的**空间可预测性**（R127，2026-09-19 立）。

背景（为什么需要这个量）
------------------------
`AGENT.md` §12.2 的 C4 问「开关**生效**了吗（声明 = 实际）」；**C8 问的是上一层**：
「**脚本意图 = 科学问题所需的前提吗**」。首例事故 = `a4_verify_capacity.py` **硬编码**
`distribution="uniform"` ⇒ C1a/C1b/C2/α/gate/α8 全部跑在 uniform 世界，
而 C4 对账时「声明」与「实际」两边都是 uniform ⇒ **判通过**（C4 没失职）。

均匀世界里，容量 `= cell_area × capacity_per_area`，而 `cell_area` **只随纬度（行）变**
⇒ 「食物在哪里」**可由位置直接推出** ⇒ **信息价值 = 0**
⇒ 在"信息本不值钱"的世界里测"信号是否传递信息" ⇒ **结构上必然 null**。

核心指标
--------
**「纬度」对容量的可解释度 `R²`**（= 1 − SS_行内 / SS_总）：
  · `R² = 1.000` ⇒ 纬度**完全**解释容量 ⇒ 位置可完全预测 ⇒ **信息价值 = 0**（前提不成立）
  · `R² < 阈值` ⇒ 存在"纬度解释不了"的部分 ⇒ 那部分**必须"知道"才知道** ⇒ 信息有价值
⇒ 该量是**一切"信息价值"类实验的前提**；本模块把它做成仪器（可复算、可入 Pre-Flight）。

口径说明
--------
- 用**同一套配置与 patch 种子**构造容量图（`patch_seed=config.seed`，与引擎一致，
  见 `sphere_engine.py` 的 `ResourceField(...)` 调用）⇒ 复现引擎真实世界，不另造。
- `R²` 用**行均值**作纬度预测器（行 = 纬度带；这是最自然的"只看位置就能猜到"的代理）。
- 纯函数（`latitude_r2` / `capacity_stats`）可单测；`build_capacity_map` 负责构造。
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "latitude_r2", "capacity_stats", "build_capacity_map", "DEFAULT_MAX_LAT_R2",
]

# 判据阈值（R127 §五.2 请内评定；此处给默认值，可被 CLI 覆盖）
# 依据：uniform ⇒ 1.000（不成立）；patchy 实测 ≈0.317（成立）⇒ 0.9 留在两者之间且远离两端。
DEFAULT_MAX_LAT_R2 = 0.9


def latitude_r2(capacity_2d) -> float:
    """「纬度（行）」对容量的可解释度 R² = 1 − SS_行内 / SS_总。

    · 容量只随行变（uniform）⇒ SS_行内 = 0 ⇒ **R² = 1.0**（完全可预测）
    · 常量图（SS_总 = 0）⇒ 约定返回 **1.0**（无信息可变 ⇒ 也不可"靠信息获益"）
    """
    cap = np.asarray(capacity_2d, dtype=np.float64)
    if cap.ndim != 2:
        raise ValueError(f"capacity 必须是二维 (rows, cols)，收到 shape={cap.shape}")
    ss_tot = float(((cap - cap.mean()) ** 2).sum())
    if ss_tot == 0.0:
        return 1.0
    ss_within = float(((cap - cap.mean(axis=1, keepdims=True)) ** 2).sum())
    return float(1.0 - ss_within / ss_tot)


def capacity_stats(capacity_2d) -> dict:
    """容量图的 C8 统计（全部为**纯观测量**，判据由调用方给阈值）。"""
    cap = np.asarray(capacity_2d, dtype=np.float64)
    r2 = latitude_r2(cap)
    row_mean = cap.mean(axis=1, keepdims=True)
    mean = float(cap.mean())
    # 「行内偏离」比例：均匀世界里恒 0（每格 = 该行均值），patchy 世界里 >0
    off_row = float((np.abs(cap - row_mean) > 1e-9).mean())
    # 行内相邻列差非零比例（斑块边界的代理；相邻列含经度环绕视为不相邻，上界口径）
    adj = np.abs(np.diff(cap, axis=1))
    adj_nonzero = float((adj > 1e-9).mean())
    cv = float(cap.std() / mean) if mean else 0.0
    # 行均值取整（相对容差 1e-9）后的**不同取值个数**：uniform=rows、patchy 接近 rows
    rm = row_mean.ravel()
    rounded = np.round(rm / (abs(mean) * 1e-9 + 1e-12)).astype(np.int64)
    unique_row_means = int(len(np.unique(rounded)))
    return {
        "lat_r2": round(r2, 6),
        "lat_unexplained": round(1.0 - r2, 6),
        "off_row_frac": round(off_row, 6),
        "adjacent_col_diff_nonzero_frac": round(adj_nonzero, 6),
        "cv": round(cv, 6),
        "unique_row_means": unique_row_means,
        "shape": list(cap.shape),
    }


def build_capacity_map(distribution: str = "uniform", *, rows: int = 60, cols: int = 120,
                       seed: int = 42, resource=None):
    """按**引擎同一套配置与 patch 种子**构造容量图（rows, cols）。

    `patch_seed=seed` 与 `sphere_engine.py` 的 `ResourceField(..., patch_seed=config.seed)`
    保持一致 ⇒ 复算的图 = 引擎真实世界（不是另造一个近似）。
    """
    from world.light_and_temperature import LightAndTemperature
    from world.resource_field import ResourceField
    from world.sphere_world import SphereWorld
    from simulation.config import ResourceConfig

    rc = resource if resource is not None else ResourceConfig()
    if distribution not in ("uniform", "patchy"):
        raise ValueError(f"未知 distribution={distribution!r}（只支持 uniform / patchy）")
    w = SphereWorld(rows=rows, cols=cols)
    lt = LightAndTemperature(w)
    field = ResourceField(
        w, lt,
        capacity_per_area=rc.capacity_per_area,
        regrowth_rate=rc.regrowth_rate,
        temp_sensitivity=rc.temp_sensitivity,
        distribution=distribution,
        patch_count=rc.patch_count,
        patch_radius=rc.patch_radius,
        patch_capacity_mult=rc.patch_capacity_mult,
        patch_regrowth_mult=rc.patch_regrowth_mult,
        background_fill=rc.background_fill,
        initial_fill=rc.initial_fill,
        patch_seed=seed,
    )
    cap = field.capacity_at(np.arange(w.n_cells))
    return np.asarray(cap, dtype=np.float64).reshape(rows, cols)
