#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R243-① 吸引子刻画扫描 —— S0/S1 判读脚本。

设计要点（对应板帖 8b2110a9）：
  判据口径已由 R243-① 改为「吸引子归属」，不再用 R229 §二 多数 seed 投票。
  本脚本做四件事：
    1. 归属判别：用 bg_resid_frac（背景停留占比，即 F1 归属量）把 on 臂样本
       无监督聚成 K 类（不预设 K=2），K 由轮廓/间隔启发式给出候选。
    2. 早期锁定核验：t=2000 处的 bg_resid 是否与终态归属一致（H1 外推检验）。
    3. 盆地大小：各类占比 + Wilson 95% 置信区间。
    4. 指纹表：每类的 F1 归属量 / F2 规模量(K_on/K_off) / F3 锁定时刻 / F4 饱和量。
  输出：stdout 报告 + 同名 .json 摘要。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

# R98 纪律：非 ASCII 输出（如 ⇒、【】）在 GBK 控制台会 rc=1 假失败，入口统一兜底。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson 95% 置信区间。"""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def fnum(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def at_tick(rs, target, tol=250):
    """取最接近 target 的采样行。"""
    cand = [r for r in rs if abs(int(r["tick"]) - target) <= tol
            and math.isfinite(fnum(r.get("bg_resid_frac")))]
    if not cand:
        return None
    return min(cand, key=lambda r: abs(int(r["tick"]) - target))


def cluster_1d(vals):
    """一维无监督聚类：用最大间隙法给候选切分点（不预设 K）。

    返回 (labels, cut, n_clusters)；cut 为最大间隙中点，label 0=低值组(留守)，1=高值组(出走)。
    """
    xs = sorted(vals)
    if len(xs) < 2:
        return [0] * len(vals), None, 1
    gaps = [(xs[i + 1] - xs[i], i) for i in range(len(xs) - 1)]
    g, i = max(gaps)
    cut = (xs[i] + xs[i + 1]) / 2.0
    labels = [0 if v <= cut else 1 for v in vals]
    k = len(set(labels))
    return labels, (cut, g), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/s2_r243_s0_attractor.csv")
    ap.add_argument("--lock-tick", type=int, default=2000,
                    help="早期锁定判定所用 tick（默认 2000，H1 已验证）")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.inp)))
    g = defaultdict(list)
    for r in rows:
        g[(r["seed"], r["arm"])].append(r)

    seeds = sorted({r["seed"] for r in rows}, key=lambda s: int(s))
    print("=" * 78)
    print(f"R243-① 吸引子刻画判读  |  {a.inp}  |  {len(rows)} 行 / {len(seeds)} seed")
    print("=" * 78)

    # ---- 1) off 臂终态（作为 F2 规模量的对照 K_off）----
    off_final = {}
    for s in seeds:
        rs = sorted(g.get((s, "off"), []), key=lambda r: int(r["tick"]))
        if rs:
            off_final[s] = fnum(rs[-1]["pop"])

    # ---- 2) on 臂终态归属判别 ----
    samples = []
    for s in seeds:
        rs = sorted(g.get((s, "on"), []), key=lambda r: int(r["tick"]))
        if not rs:
            continue
        rt = rs[-1]
        b_term = fnum(rt["bg_resid_frac"])
        samples.append({
            "seed": s,
            "bg_term": b_term,
            "K_on": fnum(rt["pop"]),
            "K_off": off_final.get(s, float("nan")),
            "sat_on": fnum(rt["global_sat"]),
            "ms": fnum(rt["ms_per_tick"]),
            "rs": rs,
        })

    vals = [s["bg_term"] for s in samples if math.isfinite(s["bg_term"])]
    labels, cutinfo, nk = cluster_1d(vals)
    for s, lb in zip(samples, labels):
        s["label"] = lb

    print(f"\n【归属判别】一维最大间隙聚类（不预设 K）⇒ 候选 K={nk}")
    if cutinfo:
        cut, gap = cutinfo
        print(f"  切分点 = {cut:.4f}（间隙 {gap:.4f}）")
    for lb in sorted({s["label"] for s in samples}):
        grp = [s for s in samples if s["label"] == lb]
        name = "低位组(疑似留守)" if lb == 0 else "高位组(疑似出走)"
        km = st.median([s["K_on"] for s in grp if math.isfinite(s["K_on"])] or [float("nan")])
        print(f"  [{lb}] {name}: n={len(grp)}  seeds={[s['seed'] for s in grp]}  "
              f"bg_resid 中位={st.median([s['bg_term'] for s in grp]):.3f}  K_on 中位={km:.0f}")

    # ---- 3) 盆地大小 + Wilson CI ----
    print("\n【盆地大小】")
    n = len(samples)
    for lb in sorted({s["label"] for s in samples}):
        k = sum(1 for s in samples if s["label"] == lb)
        p, lo, hi = wilson(k, n)
        nm = "留守型" if lb == 0 else "出走型"
        print(f"  {nm}: {k}/{n} = {p:.1%}  (95% Wilson: {lo:.1%} ~ {hi:.1%})")

    # ---- 4) H1 早期锁定外推检验 ----
    L = a.lock_tick
    print(f"\n【H1 早期锁定外推检验】t={L} 处 bg_resid 是否与终态归属一致")
    ok = miss = nan = 0
    # 用终态聚类得到的切分点作为阈值（真正的外推：不看 t=L 的值去定阈值）
    thr = cutinfo[0] if cutinfo else None
    for s in samples:
        r = at_tick(s["rs"], L)
        if r is None:
            nan += 1
            continue
        b_l = fnum(r["bg_resid_frac"])
        pred = 0 if b_l <= thr else 1
        match = (pred == s["label"])
        ok += int(match)
        miss += int(not match)
        flag = "✓" if match else "✗"
        print(f"  seed{s['seed']:>4}: t={L} bg_resid={b_l:.3f} ⇒ 预测{('留守' if pred==0 else '出走')}"
              f" | 终态={s['bg_term']:.3f}({('留守' if s['label']==0 else '出走')}) {flag}")
    tot = ok + miss
    print(f"  一致率 = {ok}/{tot} = {(ok/tot if tot else float('nan')):.1%}"
          + (f"（{nan} 个无采样点）" if nan else ""))

    # ---- 5) 锁定时刻估计（bg_resid 首次进入终态类区间的 tick）----
    print("\n【F3 锁定时刻】bg_resid 首次持续落入终态类别区间的 tick（连续 2 个采样点）")
    if thr is not None:
        for s in samples:
            rs = s["rs"]
            side = 0 if s["label"] == 0 else 1
            lock_t = None
            run = 0
            for r in rs:
                b = fnum(r["bg_resid_frac"])
                if not math.isfinite(b):
                    continue
                cur = 0 if b <= thr else 1
                if cur == side:
                    run += 1
                    if run >= 2 and lock_t is None:
                        lock_t = int(r["tick"])
                else:
                    run = 0
                    lock_t = None
            s["lock_t"] = lock_t
            lt = f"{lock_t}" if lock_t is not None else "未锁定/数据不足"
            print(f"  seed{s['seed']:>4}({'留守' if side==0 else '出走'}): 锁定 tick = {lt}")

    # ---- 6) 指纹汇总 ----
    print("\n【吸引子指纹表】")
    print(f"  {'类别':<6} {'n':>3} | {'F1归属量':>9} {'F2规模K_on':>10} {'K_on/K_off':>10} "
          f"{'F3锁定t':>8} {'F4饱和':>8}")
    for lb in sorted({s["label"] for s in samples}):
        grp = [s for s in samples if s["label"] == lb]
        nm = "留守" if lb == 0 else "出走"
        f1 = st.median([s["bg_term"] for s in grp])
        f2 = st.median([s["K_on"] for s in grp if math.isfinite(s["K_on"])])
        rr = [s["K_on"] / s["K_off"] for s in grp
              if math.isfinite(s["K_on"]) and math.isfinite(s["K_off"]) and s["K_off"]]
        f2r = st.median(rr) if rr else float("nan")
        f3s = [s["lock_t"] for s in grp if s.get("lock_t")]
        f3 = st.median(f3s) if f3s else float("nan")
        f4 = st.median([s["sat_on"] for s in grp if math.isfinite(s["sat_on"])])
        print(f"  {nm:<6} {len(grp):>3} | {f1:>9.3f} {f2:>10.0f} {f2r:>10.4f} "
              f"{f3:>8.0f} {f4:>8.3f}")

    # ---- 7) off 臂对照 ----
    print("\n【off 臂对照（终态）】")
    for s in samples:
        print(f"  seed{s['seed']:>4}: K_off={s['K_off']:.0f}  sat_off="
              + f"{fnum([r for r in sorted(g[(s['seed'],'off')], key=lambda r:int(r['tick']))][-1]['global_sat']):.3f}")

    # ---- JSON 摘要 ----
    if a.out:
        summ = {
            "input": a.inp,
            "n_seeds": len(samples),
            "lock_tick": L,
            "cluster_cut": (cutinfo[0] if cutinfo else None),
            "candidate_k": nk,
            "early_lock_agree": f"{ok}/{ok+miss}",
            "basins": {("留守" if lb == 0 else "出走"): {
                "n": sum(1 for s in samples if s["label"] == lb),
                "seeds": [s["seed"] for s in samples if s["label"] == lb],
                "wilson": wilson(sum(1 for s in samples if s["label"] == lb), len(samples)),
            } for lb in sorted({s["label"] for s in samples})},
            "samples": [{k: v for k, v in s.items() if k != "rs"} for s in samples],
        }
        json.dump(summ, open(a.out, "w"), ensure_ascii=False, indent=2)
        print(f"\n[写出] {a.out}")


if __name__ == "__main__":
    main()
