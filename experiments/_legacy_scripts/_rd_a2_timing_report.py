# -*- coding: utf-8 -*-
"""汇总配对计时 JSON：逐轮比值（抗窗口漂移）+ digest 交叉核验。"""
import glob
import json
import os
import statistics as st

BASE = r"C:\Users\圣羽\Desktop\temp\tempCode"
TREES = {"main": os.path.join(BASE, "the-world"), "wt": os.path.join(BASE, "_wt_lq_rd")}
VARIANTS = ["main_s0", "main_s1", "wt_s0", "wt_s1"]

data = {}   # variant -> {round: ms}
for v in VARIANTS:
    tree, tag = v.split("_")
    pat = os.path.join(TREES[tree], "_trash_local", "_rd_a2_timing", f"{v}_*.json")
    for p in sorted(glob.glob(pat)):
        d = json.load(open(p, encoding="utf-8"))
        if "prerep" in p:
            continue
        rnd = os.path.basename(p).split("_")[-1].split(".")[0]
        data.setdefault(v, {})[rnd] = d

rounds = sorted({r for v in data for r in data[v]})
print("=== 逐轮中位 ms/t ===")
print(f"{'round':>5} | " + " | ".join(f"{v:>8}" for v in VARIANTS) + " | s0/s1(main) s0/s1(wt) m0/w1")
for r in rounds:
    row = []
    for v in VARIANTS:
        d = data.get(v, {}).get(r)
        row.append(f"{d['ms_per_tick']:>8.1f}" if d else f"{'-':>8}")
    def g(v):
        d = data.get(v, {}).get(r)
        return d["ms_per_tick"] if d else None
    a, b, c, e = g("main_s0"), g("main_s1"), g("wt_s0"), g("wt_s1")
    rat = f" {a/b:.2f}  {c/e:.2f}  {a/e:.2f}" if None not in (a, b, c, e) else ""
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

print("\n=== 汇总 ===")
for v in VARIANTS:
    vals = [d["ms_per_tick"] for d in data.get(v, {}).values()]
    mins = [min(d["reps_ms"]) for d in data.get(v, {}).values()]
    print(f"  {v:>8}: 中位 {st.median(vals):.1f}｜各轮 {['%.1f' % x for x in vals]}"
          f"｜各轮 fastest-rep {['%.1f' % x for x in mins]}")

ratios = []
for r in rounds:
    try:
        ratios.append(data["main_s0"][r]["ms_per_tick"] / data["wt_s1"][r]["ms_per_tick"])
    except KeyError:
        pass
print(f"\n  逐轮比值 main_s0/wt_s1 = {['%.2f' % x for x in ratios]}"
      f" ⇒ 中位 ×{st.median(ratios):.2f}")
