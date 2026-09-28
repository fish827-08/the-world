#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R243-① S1 普查判读：合并 S0(8 seed)+S1(24 seed) = 32 seed 池，
按同一套无监督归属口径给出吸引子个数 / 盆地大小(Wilson CI) / 指纹稳定性。
判据预注册于板帖 8b2110a9 + R246 §四：只用 bg_resid(F1) 判别，不用 L1/几何。"""
import argparse, csv, json, math, os, statistics as st
from collections import defaultdict
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def fnum(v):
    try:
        f=float(v); return f if math.isfinite(f) else float("nan")
    except (TypeError,ValueError): return float("nan")


def wilson(k,n,z=1.96):
    if n==0: return (0.0,0.0,0.0)
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (p,max(0.0,c-h),min(1.0,c+h))


def cut_1d(vals):
    xs=sorted(vals)
    if len(xs)<2: return None,None
    gaps=[(xs[i+1]-xs[i],i) for i in range(len(xs)-1)]
    g,i=max(gaps)
    return (xs[i]+xs[i+1])/2.0, g


def load(paths):
    g=defaultdict(list)
    for p in paths:
        if not os.path.exists(p): continue
        for r in csv.DictReader(open(p)): g[(r["seed"],r["arm"])].append(r)
    return g


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", default=["results/s2_r243_s0_attractor.csv","results/s2_r243_s1_census.csv"])
    ap.add_argument("--out", default="results/s2_r243_s1_readout.json")
    a=ap.parse_args()
    g=load(a.csv)
    seeds=sorted({s for (s,ar) in g}, key=lambda x:int(x))
    print("="*76); print(f"R243-① S1 普查判读  |  {len(seeds)} seed 池  |  文件 {a.csv}"); print("="*76)

    off_final={}
    for s in seeds:
        rs=sorted(g.get((s,"off"),[]),key=lambda r:int(r["tick"]))
        if rs: off_final[s]=fnum(rs[-1]["pop"])

    samples=[]
    for s in seeds:
        rs=sorted(g.get((s,"on"),[]),key=lambda r:int(r["tick"]))
        if not rs: continue
        rt=rs[-1]
        samples.append({"seed":s,"bg_term":fnum(rt["bg_resid_frac"]),
                        "K_on":fnum(rt["pop"]),"K_off":off_final.get(s,float("nan")),
                        "sat_on":fnum(rt["global_sat"]),
                        "l1":fnum(rt["l1_visited_patch_frac"]),
                        "l3net":fnum(rt["l3_net_per_capita"]),
                        "rs":rs})

    vals=[x["bg_term"] for x in samples if math.isfinite(x["bg_term"])]
    cut,gap=cut_1d(vals)
    for x in samples: x["label"]=0 if x["bg_term"]<=cut else 1

    print(f"\n【归属判别】一维最大间隙：切分点={cut:.4f}  间隙={gap:.4f}")
    for lb in sorted({x['label'] for x in samples}):
        grp=[x for x in samples if x['label']==lb]
        nm='留守' if lb==0 else '出走'
        km=st.median([x['K_on'] for x in grp])
        sm=st.median([x['sat_on'] for x in grp])
        print(f"  [{lb}]{nm}: n={len(grp):>2}  bg_resid中位={st.median([x['bg_term'] for x in grp]):.3f}  "
              f"K_on中位={km:.0f}  sat中位={sm:.3f}")

    print("\n【盆地大小 + Wilson 95% CI】")
    n=len(samples)
    for lb in sorted({x['label'] for x in samples}):
        k=sum(1 for x in samples if x['label']==lb)
        p,lo,hi=wilson(k,n)
        nm='留守型' if lb==0 else '出走型'
        print(f"  {nm}: {k}/{n} = {p:.1%}  (95% CI: {lo:.1%} ~ {hi:.1%})")

    print("\n【H1 早期锁定外推（t=2000）】")
    ok=miss=0
    for x in samples:
        r=[q for q in x['rs'] if abs(int(q['tick'])-2000)<=250]
        if not r: continue
        b=fnum(r[0]['bg_resid_frac']); pred=0 if b<=cut else 1
        m=(pred==x['label']); ok+=m; miss+=not m
        if not m: print(f"  ✗ seed{x['seed']}: t=2000={b:.3f} pred={pred} term={x['bg_term']:.3f} lab={x['label']}")
    print(f"  一致率 = {ok}/{ok+miss} = {(ok/(ok+miss) if ok+miss else 0):.1%}")

    print("\n【指纹表（32 seed 池）】")
    print(f"  {'类别':<6}{'n':>4} | {'F1归属':>7} {'F2 K_on':>8} {'K_on/K_off':>10} {'F4饱和':>7} {'L3净/人':>8}")
    for lb in sorted({x['label'] for x in samples}):
        grp=[x for x in samples if x['label']==lb]
        nm='留守' if lb==0 else '出走'
        f1=st.median([x['bg_term'] for x in grp]); f2=st.median([x['K_on'] for x in grp])
        rr=[x['K_on']/x['K_off'] for x in grp if math.isfinite(x['K_off']) and x['K_off']]
        f2r=st.median(rr) if rr else float('nan')
        f4=st.median([x['sat_on'] for x in grp]); l3=st.median([x['l3net'] for x in grp])
        print(f"  {nm:<6}{len(grp):>4} | {f1:>7.3f} {f2:>8.0f} {f2r:>10.4f} {f4:>7.3f} {l3:>8.3f}")

    # L1 全程 null 复核
    print("\n【L1 是否仍不可分（复核）】")
    g0=[x['l1'] for x in samples if x['label']==0]; g1=[x['l1'] for x in samples if x['label']==1]
    print(f"  留守 L1中位={st.median(g0):.3f}  出走 L1中位={st.median(g1):.3f}  "
          f"差={(st.median(g1)-st.median(g0))/st.median(g0)*100:+.1f}%")

    summ={"n_seeds":len(samples),"cut":cut,"gap":gap,
          "basins":{('留守' if lb==0 else '出走'):{"n":sum(1 for x in samples if x['label']==lb),
                    "wilson":wilson(sum(1 for x in samples if x['label']==lb),n)}
                    for lb in sorted({x['label'] for x in samples})},
          "early_lock":f"{ok}/{ok+miss}",
          "samples":[{k:v for k,v in x.items() if k!='rs'} for x in samples]}
    if a.out:
        json.dump(summ,open(a.out,'w'),ensure_ascii=False,indent=2); print(f"\n[写出] {a.out}")


if __name__=="__main__":
    main()
