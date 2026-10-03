# -*- coding: utf-8 -*-
"""S3 判读（R257 预注册口径）：主门 A = never_visited 逐 seed 配对差 <0 ≥14/24。"""
import csv
from statistics import median

rows = []
for f in ["the-world-data/s3_r257_v2/s3_memory_a.csv", "the-world-data/s3_r257_v2/s3_memory_b.csv"]:
    rows.extend(csv.DictReader(open(f, encoding="utf-8")))

# 完整性校验
ticks = sorted({r["tick"] for r in rows})
runs = {(r["seed"], r["arm"]) for r in rows}
seeds = sorted({int(r["seed"]) for r in rows})
print(f"完整性: {len(seeds)} seed × 2 臂 = {len(runs)} run | 采样 tick = {ticks}")
print(f"预期 48 run: {'✅' if len(runs)==48 else '🔴 缺 ' + str(48-len(runs))}")

# 取终态 tick=2000
term = {}
for r in rows:
    if r["tick"].strip() == "2000":
        term[(int(r["seed"]), r["arm"])] = r

# 判据 A：never_visited 配对差（on - off）
diffs_a, pairs = [], []
for s in seeds:
    on, off = term.get((s,"mem_on")), term.get((s,"mem_off"))
    if on and off:
        d = float(on["never_visited_patch_cell_frac"]) - float(off["never_visited_patch_cell_frac"])
        diffs_a.append((s, d))
        pairs.append((s, d,
            int(float(on["pop"])), int(float(off["pop"])),
            float(on["bg_resid_frac"]), float(off["bg_resid_frac"])))

neg = sum(1 for _, d in diffs_a if d < 0)
print(f"\n=== 判据 A（主门）：never_visited 配对差 on−off ===")
print(f"负向（记忆降低未访问格）: {neg}/{len(diffs_a)}  |  通过线 ≥14/24 ⇒ {'✅ 通过' if neg>=14 else '❌ 不通过'}")
print(f"中位差: {median([d for _,d in diffs_a])*100:.2f} pp | 均值: {sum(d for _,d in diffs_a)/len(diffs_a)*100:.2f} pp")

# 判据 B：pop 配对（只报）
print(f"\n=== 判据 B（只报不设门）：pop 配对 on−off ===")
dp = [(s, po-pf) for s,d,po,pf,_,_ in pairs]
neg_pop = sum(1 for _,d in dp if d<0)
print(f"on<off（探索成本方向）: {neg_pop}/24 | 中位差: {median([d for _,d in dp]):.0f}")

# 判据 C：bg_resid 终态配对（只报）
print(f"\n=== 判据 C（只报不设门）：bg_resid_frac 终态配对 ===")
db = [(s, bo-bf) for s,_,_,_,bo,bf in pairs]
print(f"中位差: {median([d for _,d in db]):+.4f}")

# 型别划分（bg_resid off 臂 < 0.226996 = 留守）与明细表
print(f"\n=== 24 seed 明细（按 off 臂 bg_resid 排序；型别按 S2 cut=0.226996）===")
print(f"{'seed':>5} {'型别':>4} {'nv差(pp)':>9} {'pop差':>7} {'K_on':>6} {'K_off':>6}")
for s, d, po, pf, bo, bf in sorted(pairs, key=lambda x: x[4]):
    typ = "出走" if bf > 0.226996 else "留守"
    mark = "⬇" if d < 0 else "⬆"
    print(f"{s:>5} {typ:>4} {d*100:>+8.2f}{mark} {po-pf:>+7} {po:>6} {pf:>6}")

# 分型别聚合（出走型种子上判据 A 是否不同？）
for typ, lo_hi in [("留守", lambda bf: bf <= 0.226996), ("出走", lambda bf: bf > 0.226996)]:
    sub = [(s,d) for s,d,_,_,bo,_ in pairs if lo_hi(bo)]
    if sub:
        n = sum(1 for _,d in sub if d<0)
        print(f"\n[{typ}型 {len(sub)} seed] 判据A负向: {n}/{len(sub)} | 中位差: {median([d for _,d in sub])*100:+.2f} pp")
