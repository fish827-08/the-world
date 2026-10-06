# -*- coding: utf-8 -*-
"""E-057 K2 基线计算器（砚）：E-056 逐 seed (F, M, M-F, peak_gain)——预注册 §三附复现用。

口径：index 式（n=20）：F=窗1-5，M=窗8-13（floor(0.35n)..ceil(0.65n) 一基），peak_gain=max(F..L窗)-F。
用法：py tools/e057_baseline_calc.py --data the-world-data/e056_m1
"""
from __future__ import annotations
import argparse, csv, glob, math, os, re, statistics

def segs(vals):
    n = len(vals)
    k = int(0.25 * n)
    a, b = math.floor(0.35 * n) + 1, math.ceil(0.65 * n)  # 1-based inclusive-ish
    F = statistics.mean(vals[:k])
    M = statistics.mean(vals[a - 1:b])  # zero-based slice [a-1, b)
    P = max(vals)
    return F, M, P - F

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="the-world-data/e056_m1")
    a = ap.parse_args()
    print("seed | F | M | M-F | peak_gain | K1hit(>=0.010 & M>=F)")
    for fp in sorted(glob.glob(os.path.join(a.data, "e056_m1_d1700_s*_t2000.csv"))):
        seed = re.search(r"_s(\d+)_t", os.path.basename(fp)).group(1)
        rows = list(csv.DictReader(open(fp, encoding="utf-8")))
        vals = [float(r["g15_mean"]) for r in rows]
        assert not any(v != v for v in vals), f"NaN g15_mean in {fp} (fail-loud)"
        F, M, pg = segs(vals)
        print(f"{seed} | {F:.4f} | {M:.4f} | {M - F:+.4f} | {pg:.4f} | {'Y' if (pg >= 0.010 and M >= F) else 'N'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
