# -*- coding: utf-8 -*-
"""S2 终判：按云归预注册 4 条口径判读（R253）。cut=0.226996 固定，不在新数据上重算。"""
import csv, json
from statistics import median

def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return rows

def judge(path, name, s1_cut=None):
    rows = load(path)
    on = {int(r["seed"]): r for r in rows if r["arm"] == "on"}
    seeds = sorted(on)
    CUT = s1_cut if s1_cut else 0.226996
    out = {"name": name, "n": len(seeds), "cut": CUT}
    # 终态 bg_resid_frac
    term = {s: float(on[s]["bg_resid_frac"]) for s in seeds}
    vals = sorted(term.values())
    # 最大间隙
    gaps = [(vals[i+1]-vals[i], vals[i], vals[i+1]) for i in range(len(vals)-1)]
    gmax, lo, hi = max(gaps)
    out["gap_max"] = round(gmax, 4); out["split_at"] = (round(lo,4), round(hi,4))
    out["c1_K2_gap_ge_0.3"] = bool(len(seeds) >= 2 and gmax >= 0.3)
    # 型别（固定切点）：bg_resid > cut = 出走
    label = {s: (1 if term[s] > CUT else 0) for s in seeds}  # 1=出走 0=留守
    n_chu = sum(label.values())
    out["n_chuzou"] = n_chu; out["frac_chuzou"] = round(n_chu/len(seeds), 4)
    lo_w, hi_w = 0.3087, 0.6355
    out["c2_basin_in_wilson"] = bool(lo_w <= n_chu/len(seeds) <= hi_w)
    # H1 外推：t=1000 切 vs t=2000 切
    agree = 0; have = 0
    for s in seeds:
        r1k = [r for r in rows if r["arm"]=="on" and int(r["seed"])==s and int(r["tick"])==1000]
        if r1k:
            have += 1
            early = 1 if float(r1k[0]["bg_resid_frac"]) > CUT else 0
            agree += (early == label[s])
    out["h1_agree"] = f"{agree}/{have}"; out["c3_h1_ge_90"] = bool(have and agree/have >= 0.9)
    # L1 null：两型 l1_visited_patch_frac（t=2000）中位差
    l1_0 = [float(on[s]["l1_visited_patch_frac"]) for s in seeds if label[s]==0]
    l1_1 = [float(on[s]["l1_visited_patch_frac"]) for s in seeds if label[s]==1]
    if l1_0 and l1_1:
        d = abs(median(l1_0) - median(l1_1))
        out["l1_med_diff_pp"] = round(d*100, 2)
        out["c4_l1_le_5pp"] = bool(d*100 <= 5.0)
    # 细节
    out["types"] = {s: {"bg_term": round(term[s],4), "label": "出走" if label[s] else "留守",
                        "pop": int(float(on[s]["pop"])),
                        "l1": round(float(on[s]["l1_visited_patch_frac"]),3)} for s in seeds}
    return out

new = judge("the-world-data/s2_r243_repl_newmachine.csv", "新机77-100")
s1  = judge("the-world-data/results/s2_r243_s0/s2_r243_s1_census.csv", "S1(53-76+45-52)")
print(json.dumps({"新机77-100": {k:v for k,v in new.items() if k!="types"},
                  "S1交叉验证": {k:v for k,v in s1.items() if k!="types"}}, ensure_ascii=False, indent=1))
print("\n=== 新机型别明细（seed: bg终态/型别/pop/l1）===")
for s, d in sorted(new["types"].items(), key=lambda x: x[1]["bg_term"]):
    print(f"  seed{s}: bg={d['bg_term']:.4f} {d['label']} pop={d['pop']} l1={d['l1']}")
