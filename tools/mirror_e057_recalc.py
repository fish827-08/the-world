# -*- coding: utf-8 -*-
"""E-057 rep_w 档扫批 独立复算（镜 · R394④ 纯 stdlib 独立脚本）。

独立性（R394④）：不 import s3/e056/e057_readout/引擎/任何我参与链路模块；只读原始 CSV + summary.json。
口径来源（引用不发明）：预注册 E-057 @7605da1 + PI 会签 @2306949（+8 同树 solo∧0.0 锚臂）+ §五复算计划。

- index 式段切（PI 裁②）：k=int(0.25n)；F=首段窗 1..k，L=末段窗 (n-k+1)..n，M=中段窗 8..13（1 基闭区间）。
- K1 形态辅判：peak_gain=P−F≥0.010 且 M≥F ⇒ 上升形态计数（档内仅形态描述，不单独判正）。
- K2 主判据：Δ=(M−F)_档 − (M−F)_同树锚臂(solo∧0.0)，逐 seed 207–214 配对；同向(Δ>0)计数报两线（≥4/5 与 ratio≥0.8 即 7/8），
  以预注册锁定线为准（判读稿用 7/8）。E-056 降为跨树核验（不作 K2 基线）。
- K3 生态门（前置）：每档 ≥5/8 存活至 2000t + pop_end≥6945；不过门档 K1/K2 不可执行。
- §四-4 全 20 窗双口径敏感性：主判据=锁定中段 8..13；对照=中央对称窗 7..14，双 m_minus_f 并报。

用法：py tools/mirror_e057_recalc.py --data the-world-data/e057_repw [--e056 the-world-data/e056_m1]
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import re
import statistics

ANCHOR_TAG = 0.0


def seg_stats(vals, mid_lo=8, mid_hi=13):
    n = len(vals)
    k = int(0.25 * n)
    F = statistics.mean(vals[:k]) if k > 0 else float("nan")
    L = statistics.mean(vals[-k:]) if k > 0 else float("nan")
    lo = max(0, mid_lo - 1)
    hi = min(n, mid_hi)
    M = statistics.mean(vals[lo:hi])
    P = max(vals)
    return {"n": n, "k": k, "F": F, "M": M, "L": L, "P": P,
            "peak_gain": P - F, "m_minus_f": M - F}


def load_arm(fp):
    """读一份 CSV → {g15, pop_end, alive(stop==ticks & pop>0), seed, rep_w_effective}。"""
    rows = list(csv.DictReader(open(fp, encoding="utf-8")))
    rows.sort(key=lambda r: int(float(r.get("tick", 0))))
    g15 = [float(r["g15_mean"]) for r in rows]
    seed = re.search(r"_s(\d+)_", os.path.basename(fp)).group(1)
    summ = json.load(open(fp.replace(".csv", ".summary.json"), encoding="utf-8"))
    r0 = summ["runs"][0]
    pops = [float(r["pop"]) for r in rows]
    return {"seed": seed, "rep_w": float(r0.get("reputation_weight_effective", r0.get("rep_w_declared", -1))),
            "solo": bool(r0.get("rep_w_solo", False)), "enabled": bool(r0.get("info_structure_enabled_effective", True)),
            "g15": g15, "pop_end": pops[-1], "alive": (r0.get("stop") == "ticks" and pops[-1] > 0)}


def k2_pair(arm_segs, anchor_segs, line):
    deltas = {}
    pos = 0
    for seed in sorted(arm_segs):
        if seed in anchor_segs:
            d = arm_segs[seed]["m_minus_f"] - anchor_segs[seed]["m_minus_f"]
            deltas[seed] = d
            if d > 0:
                pos += 1
    return pos >= line, pos, len(deltas), deltas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="the-world-data/e057_repw")
    ap.add_argument("--e056", default="the-world-data/e056_m1")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.data, "e057_*.csv")))
    if not files:
        print("FAIL-LOUD: 未找到 E-057 CSV", a.data)
        return 1

    # 按 (rep_w) 分组：seed → rec
    arms = {}
    for fp in files:
        rec = load_arm(fp)
        arms.setdefault(rec["rep_w"], {})[rec["seed"]] = rec
    anchors = arms.get(ANCHOR_TAG)
    if not anchors:
        print("FAIL-LOUD: 缺锚臂 rep_w=0.0")
        return 1

    print("== K3 生态门（前置，逐档）==")
    gate = {}
    for w, recs in sorted(arms.items()):
        alive_n = sum(1 for r in recs.values() if r["alive"])
        p3 = sum(1 for r in recs.values() if r["pop_end"] >= 6945)
        nn = len(recs)
        ok = alive_n >= 5 and p3 >= 5
        gate[w] = ok
        print(f"  rep_w={w}: 存活 {alive_n}/{nn}  P3@2k≥6945 {p3}/{nn}  "
              f"pop_end范围[{min(r['pop_end'] for r in recs.values()):.0f},"
              f"{max(r['pop_end'] for r in recs.values()):.0f}]  ⇒ {'过门' if ok else '不过门'}")

    print("\n== K1 形态辅判 + K2 主判据（主口径：中段 8..13）==")
    print("  rep_w | K1上升形态 | K2同向(Δ>0) | ≥4/5线 | 7/8线(ratio0.8) | Δ均值 | 逐seed Δ(207→214)")
    main_segs = {}
    for w, recs in sorted(arms.items()):
        segs = {s: seg_stats(r["g15"]) for s, r in recs.items()}
        main_segs[w] = segs
        k1 = sum(1 for s in segs.values() if (s["peak_gain"] >= 0.010 and s["M"] >= s["F"]))
        if w == ANCHOR_TAG:
            print(f"  {w:>5} | {k1}/8（锚臂，K2 不适用）")
            continue
        if not gate.get(w, False):
            print(f"  {w:>5} | {k1}/8 ⇒ K3 不过门，K2 不可执行")
            continue
        ok4, pos, nn, dl = k2_pair(segs, main_segs[ANCHOR_TAG], 5)
        ok8, _, _, _ = k2_pair(segs, main_segs[ANCHOR_TAG], 7)
        dvals = [dl[s] for s in sorted(dl)]
        dmean = statistics.mean(dvals)
        dtxt = " ".join(f"{dl[s]:+.4f}" for s in sorted(dl))
        print(f"  {w:>5} | {k1}/8 | {pos}/{nn} | {'成' if ok4 else '否'} | {'成' if ok8 else '否'} | {dmean:+.4f} | {dtxt}")

    print("\n== §四-4 全 20 窗双口径敏感性：中央对称窗 7..14（对照主口径 8..13）==")
    print("  rep_w | K2同向(窗7-14) | 与主口径同向差")
    sens_segs = {w: {s: seg_stats(r["g15"], 7, 14) for s, r in recs.items()} for w, recs in arms.items()}
    for w, segs in sorted(sens_segs.items()):
        if w == ANCHOR_TAG or not gate.get(w, False):
            continue
        _, pos14, _, _ = k2_pair(segs, sens_segs[ANCHOR_TAG], 1)
        _, pos_main, _, _ = k2_pair(main_segs[w], main_segs[ANCHOR_TAG], 1)
        print(f"  {w:>5} | {pos14}/8 | 主口径{pos_main}/8 → 差{pos14 - pos_main:+d}")

    # 跨树核验：E-057 锚臂(solo) M−F vs E-056 M−F 逐 seed
    e056 = sorted(glob.glob(os.path.join(a.e056, "*.csv")))
    if e056:
        print("\n== 锚臂跨树核验：E-057 rpw0(solo) M−F vs E-056 M−F 逐 seed（sanity，不作 K2 基线）==")
        e56 = {}
        for fp in e056:
            rec = load_arm(fp)
            e56[rec["seed"]] = seg_stats(rec["g15"])
        common = sorted(set(main_segs[ANCHOR_TAG]) & set(e56))
        diffs = [(s, main_segs[ANCHOR_TAG][s]["m_minus_f"] - e56[s]["m_minus_f"]) for s in common]
        if diffs:
            mx = max(abs(d) for _, d in diffs)
            print(f"  共有 seed {len(common)}个: " + " ".join(f"s{s}:{d:+.5f}" for s, d in diffs))
            print(f"  max|M−F差| = {mx:.5f} ⇒ {'逐位复现(≤0.00005)' if mx <= 5e-5 else '存在漂移，须查'}")
        else:
            print("  无共有 seed（seed 区间不同？）")

    # fail-loud：NaN / 缺列
    bad = []
    for w, recs in arms.items():
        for s, r in recs.items():
            if any(v != v for v in r["g15"]) or len(r["g15"]) != 20:
                bad.append(f"rep_w={w}/s{s}: g15 len={len(r['g15'])}")
    if bad:
        print("\nFAIL-LOUD g15:", "; ".join(bad))
        return 1
    print("\n复算完成（纯 stdlib，未复用参与链路脚本，R394④）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
