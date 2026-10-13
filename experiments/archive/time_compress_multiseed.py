"""A4 增强：多臂 + 多种子 —— 判断「尾段相对差」是**噪声**还是**系统性漂移**。

为什么要扩
----------
单臂单种子（`time_compress_check.py`）只给一个数。判据 0.20 是**中位**线，
但"中位数超线"在 n=3 时可能只是随机漂。本脚本：
1. k ∈ {1,2,3,5}（D = 2400,1200,800,480）⇒ 看是否**随 k 单调恶化**（真漂移的特征）
2. 多 seed ⇒ 每次 replicate 给一条「尾段中位相对差」，报**均值 ± 标准差**
3. 报「配对 t 检验」式的朴素判读：若 k=5 的均值 - k=1 的均值 远大于 seed 间 SD，
   才是系统漂移。

判据（沿用任务书）：尾 2/3 相对差中位 < 0.20 ⇒ 通过。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.time_compress_check import run_day   # noqa: E402
# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print 会让脚本 rc=1 假失败；F-R15 族）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # 非 TTY / 旧解释器：不因诊断能力缺失而阻断运行


def rel_series(d0, i0, i1) -> list[float]:
    n = min(len(i0["pops"]), len(i1["pops"]))
    return [abs(i1["pops"][j] - i0["pops"][j]) / max(i0["pops"][j], 1.0)
            for j in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(description="A4 多臂多种子")
    ap.add_argument("--rows", type=int, default=60)
    ap.add_argument("--cols", type=int, default=120)
    ap.add_argument("--patches", type=int, default=30)
    ap.add_argument("--pop", type=int, default=500)
    ap.add_argument("--days", type=float, default=6.0)
    ap.add_argument("--travel-per-day", type=float, default=15.0)
    ap.add_argument("--seeds", default="42,7,13,101,2024")
    ap.add_argument("--ds", default="2400,1200,800,480")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    Ds = [int(x) for x in a.ds.split(",") if x.strip()]
    print(f"== A4 多臂：{a.rows}x{a.cols} 斑块{a.patches} 初始{a.pop} "
          f"{a.days}日 行程{a.travel_per_day}格 ==")
    print(f"   臂 D={Ds}  种子={seeds}\n")

    # 基准臂 D=2400
    base = {}
    for s in seeds:
        _, info = run_day(a.rows, a.cols, a.patches, a.pop, Ds[0], a.days,
                          True, seed=s, travel_per_day=a.travel_per_day)
        base[s] = info

    out = {"seeds": seeds, "ds": Ds, "days": a.days, "arms": {}}
    print(f"{'D':>6}{'k':>6}{'利用率':>10}"
          f"{'尾段中位(mean)':>17}{'SD':>9}{'全程中位(mean)':>17}{'n过线':>8}")
    print("-" * 76)
    for D in Ds:
        tails, alls, utils = [], [], []
        for s in seeds:
            _, info = run_day(a.rows, a.cols, a.patches, a.pop, D, a.days,
                              True, seed=s, travel_per_day=a.travel_per_day)
            rel = rel_series(None, base[s], info)
            n = len(rel)
            tails.append(float(np.median(rel[max(0, n // 3):])))
            alls.append(float(np.median(rel)))
            utils.append(float(info["util_final"]))
        k = 2400.0 / D
        npass = sum(1 for t in tails if t < 0.20)
        print(f"{D:>6}{k:>6.1f}{np.mean(utils):>10.3f}"
              f"{np.mean(tails):>17.4f}{np.std(tails):>9.4f}"
              f"{np.mean(alls):>17.4f}{npass:>5}/{len(seeds)}")
        out["arms"][str(D)] = {"k": k, "tails": tails, "all_med": alls,
                               "util": utils,
                               "tail_mean": float(np.mean(tails)),
                               "tail_sd": float(np.std(tails)),
                               "n_pass": int(npass)}
    print("-" * 76)
    print("判读：若 tail_mean 随 k **单调上升**且远超 SD ⇒ 系统性漂移（漏标）；")
    print("      若无趋势、量级 ≈ SD ⇒ 只是生态随机性（非漏标）。")
    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        print(f"\n已落 JSON：{a.json}")


if __name__ == "__main__":
    main()
