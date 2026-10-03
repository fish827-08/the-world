# -*- coding: utf-8 -*-
"""定位"单调下降"不成立的具体位置（诚实性核查，不是挑刺）。"""
import csv
from pathlib import Path
rows = list(csv.DictReader(open(Path("_trash_local/p1c/p1c_s207_t3000_v2.csv"), encoding="utf-8")))
print("tick | Moore%(全槽) | Δ     | Moore%(d>0) | Δ")
prev_a = prev_b = None
ups = []
for r in rows:
    t = int(r["tick"]); a = float(r["frac_read_moore"])*100; b = float(r["pos_frac_read_moore"])*100
    da = "" if prev_a is None else f"{a-prev_a:+.2f}"
    db = "" if prev_b is None else f"{b-prev_b:+.2f}"
    if prev_b is not None and b > prev_b + 1e-9:
        ups.append((t, b - prev_b))
    print(f"{t:4d} | {a:11.2f} {da:>6} | {b:11.2f} {db:>6}")
    prev_a, prev_b = a, b
print()
print(f"🔴 d>0 序列**非单调**的 tick（回升）：{[(t, f'{d:+.2f}pp') for t, d in ups]}")
print(f"   回升次数 = {len(ups)} / {len(rows)-1} 段")
