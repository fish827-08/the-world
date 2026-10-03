# -*- coding: utf-8 -*-
"""D1 独立复核：重算 A4 探针 CSV，与 summary 声称值对账。"""
import csv, json, statistics as st
from pathlib import Path

D = Path("_trash_local/p1c")
rows = list(csv.DictReader(open(D / "p1c_s207_t3000_v2.csv", encoding="utf-8")))
summ = json.load(open(D / "p1c_s207_t3000_v2.summary.json", encoding="utf-8"))

print("=" * 70)
print(f"【1】CSV 完整性（教训㉔：先验完整性）")
print("=" * 70)
print(f"  行数 = {len(rows)}（期望 3000/100 = 30）")
ticks = [int(r["tick"]) for r in rows]
print(f"  tick 范围 = {min(ticks)}..{max(ticks)}  步长唯一值 = {sorted(set(b-a for a,b in zip(ticks, ticks[1:])))}")
print(f"  末行 n_agents = {rows[-1]['n_agents']}  末行 extinct? {summ.get('extinct_at')}")
print(f"  首行 n_agents = {rows[0]['n_agents']}（探针 docstring 报 834）")

print()
print("=" * 70)
print("【2】重算池化统计（用 n_slots / n_agents 无法还原分布 ⇒ 只校核可还原量）")
print("=" * 70)
# 逐采样点：frac_* 是比例，n_slots 是分母 ⇒ 可还原计数
tot_slots = sum(int(r["n_slots"]) for r in rows)
tot_pos = sum(int(r["pos_n_slots"]) for r in rows)
readable_full = sum(float(r["frac_read_moore"]) * int(r["n_slots"]) for r in rows)
readable_pos = sum(float(r["pos_frac_read_moore"]) * int(r["pos_n_slots"]) for r in rows)
print(f"  池化 n_slots           = {tot_slots:,}   （帖报 876,290）")
print(f"  池化 pos_n_slots       = {tot_pos:,}   （帖报 505,281）")
print(f"  池化 Moore 可直读(全槽) = {readable_full/tot_slots*100:.2f}%  （帖报 92.04%）")
print(f"  池化 Moore 可直读(d>0) = {readable_pos/tot_pos*100:.2f}%  （帖报 86.20%）")

print()
print("=" * 70)
print("【3】趋势单调性（帖报'单调下降'）")
print("=" * 70)
pts = [(int(r["tick"]), float(r["frac_read_moore"])*100,
        float(r["pos_frac_read_moore"])*100) for r in rows]
mono = all(pts[i][2] >= pts[i+1][2] - 1e-9 for i in range(len(pts)-1))
print("  tick | Moore%(全槽) | Moore%(d>0)")
for t, a, b in pts[::5]:
    print(f"  {t:5d} | {a:11.2f} | {b:11.2f}")
print(f"  d>0 序列单调下降 = {mono}")
print(f"  末点 d>0 = {pts[-1][2]:.2f}%  （帖报 81.9%）")

print()
print("=" * 70)
print("【4】summary 内部一致性：per_sample 条数 vs CSV 行数")
print("=" * 70)
ps = summ.get("per_sample", [])
print(f"  summary.per_sample = {len(ps)} 条；CSV = {len(rows)} 行 ⇒ {'✅ 一致' if len(ps)==len(rows) else '🔴 不一致'}")
if ps:
    print(f"  per_sample[0].n_agents = {ps[0]['n_agents']}  vs CSV 首行 = {rows[0]['n_agents']}")
    print(f"  per_sample[-1].n_agents = {ps[-1]['n_agents']} vs CSV 末行 = {rows[-1]['n_agents']}")
