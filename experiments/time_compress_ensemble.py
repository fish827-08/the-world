"""A4 判据适用性检验：把「单次实现」换成「多种子均值曲线」后，跨 k 差还剩多少。

动机（2026-09-26 实测）
----------------------
`time_compress_check.py` 的判据是「同 seed、跨 k 的尾段相对差中位 < 0.20」。
但实测**噪声地板**（同 k、异 seed）就已经 0.33–0.66 ⇒ 判据线 0.20
**低于本系统可分辨的下限**，任何实现都可能"碰巧"过或不过。

本脚本给出判据的**正确形式**：
* 用 S 个 seed 的**均值轨迹** `mean_s N_k(t)` 代表"k 臂的期望行为"
* 跨 k 比较**均值轨迹**的相对差 ⇒ 消掉单次实现噪声，剩下的才是系统性偏差
* 同时报「均值轨迹」的**种子间 SD / sqrt(S)** = 均值的标准误 ⇒ 给判读置信度

判读：
  · 均值轨迹跨 k 差 ≪ 单次实现的种子间 SD  ⇒ 无系统性漂移，判据应以"均值轨迹"为准
  · 均值轨迹跨 k 差 ≫ 标准误 且随 k 单调 ⇒ 真漂移
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=60)
    ap.add_argument("--cols", type=int, default=120)
    ap.add_argument("--patches", type=int, default=30)
    ap.add_argument("--pop", type=int, default=500)
    ap.add_argument("--days", type=float, default=6.0)
    ap.add_argument("--travel-per-day", type=float, default=15.0)
    ap.add_argument("--seeds", default="42,7,13,101,2024,99,5,77")
    ap.add_argument("--ds", default="2400,1200,800,480")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    Ds = [int(x) for x in a.ds.split(",") if x.strip()]
    S = len(seeds)

    # 收集各臂各 seed 的按日曲线
    curves: dict[int, np.ndarray] = {}
    lens = []
    for D in Ds:
        rows = []
        for s in seeds:
            _, info = run_day(a.rows, a.cols, a.patches, a.pop, D, a.days,
                              True, seed=s, travel_per_day=a.travel_per_day)
            rows.append(info["pops"])
            lens.append(len(info["pops"]))
        n = min(lens[-1], min(len(r) for r in rows))
        curves[D] = np.array([r[:n] for r in rows], dtype=float)
        print(f"  D={D} 完成（{S} seed，{n} 日采样）", flush=True)

    n = min(c.shape[1] for c in curves.values())
    tail = slice(max(0, n // 3), n)

    out = {"seeds": seeds, "ds": Ds, "days": a.days, "arms": {}}
    print(f"\n{'D':>6}{'k':>6}{'mean_N':>10}"
          f"{'均值轨迹跨k差':>16}{'±SEM':>10}{'单次种子间SD':>16}{'判定':>12}")
    print("-" * 78)
    base = curves[Ds[0]].mean(axis=0)
    # 单次实现的种子间 SD（在尾段按日算，再取中位）
    for D in Ds:
        C = curves[D]
        mean_tr = C.mean(axis=0)
        sd_seed = float(np.median(C[:, tail].std(axis=0, ddof=1)))
        sem = sd_seed / np.sqrt(S)
        if D == Ds[0]:
            cross = 0.0
            verdict = "基准"
        else:
            rel = np.abs(mean_tr[tail] - base[tail]) / np.maximum(base[tail], 1.0)
            cross = float(np.median(rel))
            ratio = cross / max(sem, 1e-12)
            verdict = (f"✅ 噪声内" if cross <= 2 * sem else
                       (f"⚠️ {ratio:.1f}×SEM" if ratio < 4 else f"🔴 {ratio:.1f}×SEM"))
        k = 2400.0 / D
        print(f"{D:>6}{k:>6.1f}{mean_tr[tail].mean():>10.1f}"
              f"{cross:>16.4f}{sem:>10.4f}{sd_seed:>16.4f}{verdict:>12}")
        out["arms"][str(D)] = {"k": k, "mean_N": float(mean_tr[tail].mean()),
                               "cross_vs_k1": cross, "sem": float(sem),
                               "sd_seed": sd_seed, "n_days": n}
    print("-" * 78)
    print("说明：`均值轨迹跨k差` = |mean_s N_k(t) − mean_s N_1(t)| / N_1(t) 的尾段中位。")
    print("      `±SEM` = 单次种子的尾段 SD / sqrt(S)。")
    print("      判据：跨 k 差 ≤ 2×SEM ⇒ 与噪声不可分辨 ⇒ **无系统性漂移**。")
    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        print(f"\n已落 JSON：{a.json}")


if __name__ == "__main__":
    main()
