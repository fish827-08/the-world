# -*- coding: utf-8 -*-
"""汇总 rd-A 静默复测 JSON：逐轮比值（抗漂移）+ digest 交叉核验。

base = `_wt_rd_base` @ 0a38d5b（T1b，无 A2）；wt = `_wt_lq_rd` @ 8d36bdf（T1b + A2）。
"""
import glob
import json
import os
import statistics as st

BASE = r"C:\Users\圣羽\Desktop\temp\tempCode"
TREES = {"base": os.path.join(BASE, "_wt_rd_base"), "wt": os.path.join(BASE, "_wt_lq_rd")}
VARIANTS = ["base_s0", "base_s1", "wt_s0", "wt_s1"]

data = {}   # variant -> {round: ms}
for v in VARIANTS:
    tree, tag = v.split("_")
    pat = os.path.join(TREES[tree], "_trash_local", "_rd_a2_timing_q", f"{v}_*.json")
    for p in sorted(glob.glob(pat)):
        d = json.load(open(p, encoding="utf-8"))
        rnd = os.path.basename(p).split("_")[-1].split(".")[0]
        data.setdefault(v, {})[rnd] = d

rounds = sorted({r for v in data for r in data[v]})
print("=== 逐轮中位 ms/t（静默窗口） ===")
print(f"{'round':>5} | " + " | ".join(f"{v:>8}" for v in VARIANTS)
      + " | b0/w1 b0/w0 b0/b1 w0/w1")
def g(v, r):
    d = data.get(v, {}).get(r)
    return d["ms_per_tick"] if d else None
for r in rounds:
    row = []
    for v in VARIANTS:
        x = g(v, r)
        row.append(f"{x:>8.1f}" if x is not None else f"{'-':>8}")
    a, b, c, e = g("base_s0", r), g("base_s1", r), g("wt_s0", r), g("wt_s1", r)
    rat = f" {a/e:.2f}  {a/c:.2f}  {a/b:.2f}  {c/e:.2f}" if None not in (a, b, c, e) else ""
    print(f"{r:>5} | " + " | ".join(row) + " |" + rat)

print("\n=== digest 交叉核验 ===")
ref = refname = None
for v in VARIANTS:
    for r, d in sorted(data.get(v, {}).items()):
        dg = d["digest"]
        nm = f"{v}_{r}"
        if ref is None:
            ref, refname = dg, nm
            print(f"  [ref] {nm}: t={dg['t']} n={dg['n']} ms={d['ms_per_tick']}")
        else:
            diffs = [k for k in ref if ref[k] != dg.get(k)]
            print(f"  {'✓' if not diffs else '🔴'} {nm}: t={dg['t']} n={dg['n']} "
                  f"ms={d['ms_per_tick']}" + ("" if not diffs else f"  DIFF={diffs}"))

print("\n=== 汇总（4 轮中位） ===")
for v in VARIANTS:
    vals = [d["ms_per_tick"] for d in data.get(v, {}).values()]
    if not vals:
        print(f"  {v:>8}: 无数据")
        continue
    print(f"  {v:>8}: 中位 {st.median(vals):.1f}｜各轮 {['%.1f' % x for x in vals]}")

print("\n=== 逐轮比值汇总（中位） ===")
for nm, (x, y) in {"b0/w1（A1+A2）": ("base_s0", "wt_s1"),
                   "b0/w0（A2 单独）": ("base_s0", "wt_s0"),
                   "b0/b1（A1 单独）": ("base_s0", "base_s1"),
                   "w0/w1（A1 on A2）": ("wt_s0", "wt_s1")}.items():
    ratios = []
    for r in rounds:
        a, b = g(x, r), g(y, r)
        if a is not None and b is not None:
            ratios.append(a / b)
    if ratios:
        print(f"  {nm}: {['%.2f' % z for z in ratios]} ⇒ 中位 ×{st.median(ratios):.2f}")
