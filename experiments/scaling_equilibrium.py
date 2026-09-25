"""尺度均衡求解器（14.10）—— 在锚点约束下枚举 (世界大小 × 时间压缩) 的代价面。

锚点（fish 定义，2026-09-26）
--------------------------
> 基因拉满的生物，在**不吃食物、无限能量**的前提下，一生直线走**刚好绕赤道一圈**。

    cols = v_max × L_max          其中 L_max = day × lifespan_mult × (1 + 7·g3_max) = day × 8

⇒ 三个量被一条方程锁在一起：**世界宽 cols、最大速度 v_max、一昼夜 tick 数 day**。
⇒ 只有两个自由旋钮；第三个由锚点定。

派生量
------
* `k = 2400 / day`  —— **时间压缩倍率**（相对现状一昼夜 2400 tick）
=> `v_max = cols × k / 19200`（现状 L_max = 8×2400 = 19200 tick）
* 世代门 ≥30 代 ⇒ `T_total = 30 × maturity_fraction × L_max = 4.5 × L_max`
* 性能 `ms/tick ≈ a·cells + b·N + c`（实测拟合 a=0.0551 μs/格、b=5.8 μs/个）
* **速度档位** `= v_max × subdiv`（`steps = floor(speed×subdiv+0.5)`）
  ⇒ 要 ≥5 档必须 `subdiv ≥ 5 / v_max`

用法
----
    .venv\\Scripts\\python.exe experiments/scaling_equilibrium.py
    .venv\\Scripts\\python.exe experiments/scaling_equilibrium.py --pop 2000 --a 0.0551 --b 5.8
"""
from __future__ import annotations

import argparse
import math

DAY0 = 2400.0            # 现状一昼夜 tick
LIFE0 = 8.0 * DAY0       # 现状最大寿命 tick（lifespan_mult=1、g3=1 ⇒ ×8）
MATURITY = 0.15          # 成熟比例（世代时长 ≈ 0.15 × 寿命）
GENS = 30                # 世代门


def one(cols: float, k: float, pop: int, a: float, b: float, c: float,
        subdiv: float, grid: float = 0.5) -> dict:
    rows = cols * grid
    day = DAY0 / k
    L_max = 8.0 * day
    v_max = cols / L_max
    cells = rows * cols
    t_total = GENS * MATURITY * L_max
    ms = a * 1e-6 * cells * 1e3 + b * 1e-6 * pop * 1e3 + c
    return {
        "cols": cols, "k": k, "day": day, "v_max": v_max, "L_max": L_max,
        "cells": cells, "t_total": t_total, "ms": ms,
        "wall_min": t_total * ms / 1000 / 60,
        "levels": v_max * subdiv,
        "subdiv_need": math.ceil(5.0 / v_max) if v_max > 0 else 0,
        "q_cells": v_max / subdiv,           # 一个量子 = 多少格（赤道）
    }


def fmt(v: float) -> str:
    if v >= 1e6:
        return f"{v/1e6:.2f}M"
    if v >= 1e3:
        return f"{v/1e3:.0f}k"
    return f"{v:.3g}"


def main() -> None:
    ap = argparse.ArgumentParser(description="锚点约束下的尺度均衡表")
    ap.add_argument("--pop", type=int, default=2000)
    ap.add_argument("--a", type=float, default=0.0551, help="μs/格（实测拟合）")
    ap.add_argument("--b", type=float, default=5.8, help="μs/个体（实测拟合）")
    ap.add_argument("--c", type=float, default=0.4, help="固定开销 ms")
    ap.add_argument("--subdiv", type=float, default=20.0)
    a = ap.parse_args()

    print(f"锚点：cols = v_max × L_max，L_max = 8 × day。"
          f"个体 {a.pop}，subdiv = {a.subdiv:.0f}")
    print(f"性能模型：ms/tick ≈ {a.a}·cells + {a.b}·N + {a.c}\n")

    hdr = (f"{'世界':>11}{'k':>5}{'一昼夜':>7}{'v_max':>7}{'寿命':>7}"
           f"{'格数':>8}{'30代tick':>9}{'ms/tick':>8}{'墙钟/run':>10}"
           f"{'速度档':>7}{'需subdiv':>9}")
    print(hdr)
    print("-" * 78)
    for k in (1.0, 2.5, 5.0, 10.0):
        for cols in (120, 240, 480, 960, 1920):
            r = one(cols, k, a.pop, a.a, a.b, a.c, a.subdiv)
            flag = ""
            if r["wall_min"] > 60:
                flag = " ❌"
            elif r["wall_min"] <= 30:
                flag = " ✅"
            if r["v_max"] < 1.0 / a.subdiv:
                flag += "⚠脉冲"
            print(f"{int(r['cols']//2):>5}x{int(r['cols']):<5}{k:>5.1f}{r['day']:>7.0f}"
                  f"{r['v_max']:>7.3f}{r['L_max']:>7.0f}{fmt(r['cells']):>8}"
                  f"{fmt(r['t_total']):>9}{r['ms']:>8.1f}{r['wall_min']:>9.1f}分"
                  f"{r['levels']:>7.0f}{r['subdiv_need']:>9}{flag}")
        print()

    print("读法：")
    print("  · 同一 k 下世界放大 2× ⇒ 格数 4× ⇒ 墙钟约 4×（格项主导）")
    print("  · 同一世界下 k 放大 2× ⇒ 墙钟约 1/2（tick 数减半），但时间分辨率同步变粗 2×")
    print("  · 「速度档」= v_max × subdiv；<5 档 ⇒ 速度基因退化成开关（实测见 speed_quant_probe）")
    print("  · v_max < 1/subdiv ⇒ 平均个体被量化成「静止」（v_typ = v_max/2 更早踩线）")


if __name__ == "__main__":
    main()
