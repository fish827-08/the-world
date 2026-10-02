#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交叉验证：R336(新档) vs 旗舰60k(旧档) —— rd on/off 的效应量是否一致。

用途：fish 质疑"这次实验属于白费，之前有类似实验"⇒ 必须逐项核对，
      确认 S2-6 是否与旗舰 60k / 历史批重复。
数据源：
  R336      : _rerun_logs/s2g2_newdev/results/s2g2_20{1..6}_{off,on}.csv   （新档0.0/0.0）
  旗舰60k   : the-world-data/flagship60k/{local,cloud}_arm/f60k_1{65..72}_{off,on}.csv （旧档）
口径统一：末段 20 行均值（与 R336 一致）。
"""
from __future__ import annotations

import csv
import os
import sys

TAIL_N = 20
COLS = [
    ("pop", "pop"),
    ("l1_visited_patch_frac", "l1_visited_patch_frac"),
    ("patch_sat_init_var", "patch_sat_init_var"),
    ("patch_sat_mean", "patch_sat_mean"),
    ("global_sat", "global_sat"),
    ("bg_resid_frac", "bg_resid_frac"),
]


def tail_mean(rows, key, n=TAIL_N):
    vals = []
    for r in rows[-n:]:
        v = r.get(key, "")
        if v in ("", "nan", "None", None):
            continue
        try:
            vals.append(float(v))
        except (ValueError, TypeError):
            continue
    return sum(vals) / len(vals) if vals else float("nan")


def load(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_col(rows, cand):
    for c in cand:
        if rows and c in rows[0]:
            return c
    return None


def report(name, pairs):
    print(f"\n{'='*72}\n## {name}   （末段 {TAIL_N} 行均值）\n{'='*72}")
    hdr = f"{'seed':<7}{'臂':<5}" + "".join(f"{c[1][:16]:>18}" for c in COLS)
    print(hdr)
    stats = {c[1]: {"ratios": [], "deltas": [], "wins": 0, "n": 0} for c in COLS}
    for seed, (ro, rn) in sorted(pairs.items()):
        for arm, rows in (("off", ro), ("on", rn)):
            cells = []
            for cands, label in COLS:
                key = find_col(rows, [cands])
                cells.append(f"{tail_mean(rows, key):>18.5f}" if key else f"{'—':>18}")
            print(f"{seed:<7}{arm:<5}" + "".join(cells))
        # 配对差
        deltas = []
        for cands, label in COLS:
            ko, kn = find_col(ro, [cands]), find_col(rn, [cands])
            if ko and kn:
                vo, vn = tail_mean(ro, ko), tail_mean(rn, kn)
                d = vn - vo
                r = (vn / vo) if vo not in (0, float("nan")) else float("nan")
                deltas.append(f"{label}: Δ={d:+.5f} 比={r:.3f}")
                stats[label]["ratios"].append(r)
                stats[label]["deltas"].append(d)
                stats[label]["n"] += 1
                if d > 0:
                    stats[label]["wins"] += 1
        print(f"{'':12}配对 → " + " | ".join(deltas))
    print(f"\n### {name} 汇总")
    print(f"{'指标':<28}{'on>off 票':>12}{'比值中位':>12}{'Δ 中位':>14}")
    for cands, label in COLS:
        s = stats[label]
        if not s["n"]:
            continue
        rr = sorted(x for x in s["ratios"] if x == x)
        dd = sorted(s["deltas"])
        print(f"{label:<28}{str(s['wins'])+'/'+str(s['n']):>12}"
              f"{(rr[len(rr)//2] if rr else float('nan')):>12.3f}{dd[len(dd)//2]:>+14.5f}")


def collect_new():
    base = "_rerun_logs/s2g2_newdev/results"
    pairs = {}
    for s in range(201, 207):
        try:
            ro = load(os.path.join(base, f"s2g2_{s}_off.csv"))
            rn = load(os.path.join(base, f"s2g2_{s}_on.csv"))
            pairs[s] = (ro, rn)
        except FileNotFoundError:
            pass
    return pairs


def collect_flagship():
    pairs = {}
    root = "the-world-data/flagship60k"
    for arm_dir in ("local_slice_off", "local_slice_on", "cloud_arm", "off_arm"):
        d = os.path.join(root, arm_dir)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith(".csv"):
                continue
            stem = fn[:-4]              # f60k_169_on
            parts = stem.split("_")
            if len(parts) < 4:
                continue
            seed, arm = parts[2], parts[3]
            try:
                rows = load(os.path.join(d, fn))
            except Exception:
                continue
            if rows:                      # 跳过空文件
                pairs.setdefault(seed, {})[arm] = rows
    return {s: (v["off"], v["on"]) for s, v in pairs.items() if "off" in v and "on" in v}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("new", "both"):
        report("R336 S2-6（新装置档 bg_low 0.0/0.0，全 A 支，seed 201–206 × 40k）",
               collect_new())
    if which in ("flag", "both"):
        report("旗舰 60k（旧装置档，rd on/off，seed 165–172）", collect_flagship())