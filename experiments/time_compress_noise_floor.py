"""A4 噪声地板：同 k、不同 seed 之间的相对差 —— 用来判定 k 臂间的差是不是"超噪声"。

为什么必须要这个
----------------
`time_compress_check.py` 的「相对差」是**同 seed、跨 k** 比的。
但生态随机性让**同 k 不同 seed**之间本身就能差很多 ⇒ 若不做噪声地板，
任何跨 k 的差都会被误读成"漂移"。

本脚本：对每个 k，跑 S 个 seed，两两配对算「尾段中位相对差」⇒
得到每个 k 的**种子间离散度**（噪声地板）。然后比较：
  跨 k 的差（k vs k=1，同 seed）      —— 是否
  远大于 噪声地板（同 k，异 seed）
"""
from __future__ import annotations

import argparse
import itertools
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


def tail_med(i_a: dict, i_b: dict) -> float:
    n = min(len(i_a["pops"]), len(i_b["pops"]))
    rel = [abs(i_b["pops"][j] - i_a["pops"][j]) / max(i_a["pops"][j], 1.0)
           for j in range(n)]
    return float(np.median(rel[max(0, n // 3):]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=60)
    ap.add_argument("--cols", type=int, default=120)
    ap.add_argument("--patches", type=int, default=30)
    ap.add_argument("--pop", type=int, default=500)
    ap.add_argument("--days", type=float, default=6.0)
    ap.add_argument("--travel-per-day", type=float, default=15.0)
    ap.add_argument("--seeds", default="42,7,13,101,2024,99")
    ap.add_argument("--ds", default="2400,1200,800,480")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    Ds = [int(x) for x in a.ds.split(",") if x.strip()]

    runs: dict[int, dict[int, dict]] = {}
    for D in Ds:
        runs[D] = {}
        for s in seeds:
            _, info = run_day(a.rows, a.cols, a.patches, a.pop, D, a.days,
                              True, seed=s, travel_per_day=a.travel_per_day)
            runs[D][s] = info
        print(f"  D={D} 完成（{len(seeds)} seed）", flush=True)

    out = {"seeds": seeds, "ds": Ds, "floor": {}, "cross": {}}
    print(f"\n{'D':>6}{'k':>6}{'噪声地板(同k异seed) mean±SD':>32}{'vs k=1(同seed) mean±SD':>30}")
    print("-" * 78)
    for D in Ds:
        # 噪声地板：同 k 内所有 seed 对
        pairs = [tail_med(runs[D][x], runs[D][y])
                 for x, y in itertools.combinations(seeds, 2)]
        # 跨 k：同 seed，D vs Ds[0]
        cross = [tail_med(runs[Ds[0]][s], runs[D][s]) for s in seeds]
        k = 2400.0 / D
        fm, fs = float(np.mean(pairs)), float(np.std(pairs))
        cm, cs = float(np.mean(cross)), float(np.std(cross))
        flag = ""
        if D != Ds[0]:
            # 跨 k 差 vs 噪声地板：用"超地板倍数"判读
            ratio = cm / max(fm, 1e-9)
            flag = f"  ← 跨k/地板 = {ratio:.2f}×"
        print(f"{D:>6}{k:>6.1f}{f'{fm:.4f} ± {fs:.4f}':>32}"
              f"{f'{cm:.4f} ± {cs:.4f}':>30}{flag}")
        out["floor"][str(D)] = {"k": k, "mean": fm, "sd": fs, "pairs": pairs}
        out["cross"][str(D)] = {"k": k, "mean": cm, "sd": cs, "per_seed": cross}

    print("-" * 78)
    print("判读规则：")
    print("  · 跨k均值 ≲ 2× 噪声地板  ⇒ 差异**淹没在生态噪声里** ⇒ 不能判'漂移'")
    print("  · 跨k均值 ≫ 噪声地板（≥3×）且随 k 单调 ⇒ 系统性漂移（漏标）")
    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        print(f"\n已落 JSON：{a.json}")


if __name__ == "__main__":
    main()
