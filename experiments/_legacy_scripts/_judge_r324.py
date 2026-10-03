# -*- coding: utf-8 -*-
"""R324 四条线正式判读：分层 + 效应量 + bootstrap 置信区间 + 符号检验。

数据（全部 [实测]）：
  设计 1  _rerun_logs/disp_20k_v2/results/*.pairs.csv   （seed 193-196 × off/on）
  设计 3  _rerun_logs/d3_local2/*.csv (196/197) + cloud 拉回 (194/195)
  设计 4  _rerun_logs/d4_local/*.csv                    （194/195/198/199 × off/on × go/rand）
  g22     _rerun_logs/g22_local/*.csv                   （165/167 × off/on × init 0.1/0.9）
"""
import csv
import glob
import os
import re

import numpy as np

RNG = np.random.default_rng(20261002)
CLOUD = "_rerun_logs/flagship60k_cloud/../d3_cloud_holder"  # 占位，实际用下面统一路径


def rows(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fl(x, k, d=float("nan")):
    try:
        v = x.get(k, "")
        return float(v) if v not in ("", None) else d
    except Exception:
        return d


def boot_ci(v, n=20000, lo=2.5, hi=97.5):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) < 2:
        return (float("nan"), float("nan"), float("nan"))
    idx = RNG.integers(0, len(v), size=(n, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, lo)), float(np.percentile(means, hi))


def sign_p(v):
    v = [x for x in v if np.isfinite(x) and x != 0]
    n = len(v)
    if n == 0:
        return float("nan"), 0, 0
    k = min(sum(1 for x in v if x > 0), sum(1 for x in v if x < 0))
    from math import comb
    p = min(2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n, 1.0)
    return p, sum(1 for x in v if x > 0), sum(1 for x in v if x < 0)


def branch(abs_cap):
    """⚠️ 旧口径（按 abs_cap 阈值）**已作废**：abs_cap 随 patch 数放大（400→1e6、1700→3.4e6、4000→7.1e6）
    ⇒ 会把整档错标。真支别由 `_branch_table.py` 直接读 `bg_production_zero` 得到，见 BRANCH_MAP。"""
    return "A" if abs_cap < 3.3e6 else "B"


# 真支别（pc=1700；实测 pc=400/4000 对下列 seed 结果相同）—— 来源 `_trash_local/_branch_table.py`
BRANCH_MAP = {165: "A", 167: "B", 192: "A", 194: "A", 195: "A",
              196: "A", 197: "B", 198: "B", 199: "A"}


def br_of(seed):
    return BRANCH_MAP.get(seed, "?")


SEP = "=" * 84

# ============================================================ 设计 1
print(SEP)
print("【设计 1】r_kin（亲子）vs r_placebo（同出生格非亲缘）")
print(SEP)


def act(x):
    try:
        s = int(x["seen"])
        return float(x["move"]) / (s - 1) if s >= 3 else None
    except Exception:
        return None


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) >= 10 else float("nan")


d1 = []
for p in sorted(glob.glob("_rerun_logs/disp_20k_v2/results/*.pairs.csv")):
    tag = os.path.basename(p).replace("disp_", "").replace(".pairs.csv", "")
    seed, arm = tag.split("_")
    rs = rows(p)
    by = {x["id"]: x for x in rs}
    ka, kb = [], []
    for x in rs:
        y = by.get(x.get("parent", "-1"))
        if y is None:
            continue
        va, vb = act(x), act(y)
        if va is not None and vb is not None:
            ka.append(va)
            kb.append(vb)
    r_kin = pearson(ka, kb)
    groups = {}
    for x in rs:
        v = act(x)
        if v is None:
            continue
        try:
            groups.setdefault((int(x["birth_r"]), int(x["birth_c"])), []).append((x["id"], v))
        except Exception:
            pass
    pa, pb = [], []
    for lst in groups.values():
        if len(lst) < 2:
            continue
        idx = RNG.permutation(len(lst))
        for i in range(0, len(idx) - 1, 2):
            (i1, v1), (i2, v2) = lst[idx[i]], lst[idx[i + 1]]
            if str(i1) == str(by.get(str(i2), {}).get("parent")) or \
               str(i2) == str(by.get(str(i1), {}).get("parent")):
                continue
            pa.append(v1)
            pb.append(v2)
    r_pl = pearson(pa, pb)
    d1.append((seed, arm, r_kin, len(ka), r_pl, len(pa)))

print("%-6s %-4s %9s %7s %11s %7s %10s" % ("seed", "臂", "r_kin", "n", "r_placebo", "n", "差(kin-pl)"))
for s, a, rk, nk, rp, np_ in d1:
    print("%-6s %-4s %9.3f %7d %11.3f %7d %+10.3f" % (s, a, rk, nk, rp, np_, rk - rp))
d = [rk - rp for _, _, rk, _, rp, _ in d1]
m, l, h = boot_ci(d)
p_, npos, nneg = sign_p(d)
print("-" * 84)
print("r_kin  均值 = %.3f ｜ r_placebo 均值 = %.3f"
      % (np.mean([x[2] for x in d1]), np.mean([x[4] for x in d1])))
print("配对差 (r_kin − r_placebo)：均值 %+.4f  95%%CI [%+.4f, %+.4f]  符号检验 p=%.3f（+%d/−%d）"
      % (m, l, h, p_, npos, nneg))
for arm in ("on", "off"):
    sub = [rk - rp for s, a, rk, _, rp, _ in d1 if a == arm]
    mm, ll, hh = boot_ci(sub)
    print("   仅 %-3s 臂（n=%d）：均值 %+.4f  95%%CI [%+.4f, %+.4f]" % (arm, len(sub), mm, ll, hh))

# ============================================================ 设计 3
print()
print(SEP)
print("【设计 3】斑块几何（400 / 1700 / 4000）× 20k tick")
print(SEP)


def load_d3():
    out = []
    for p in sorted(glob.glob("_rerun_logs/d3_local2/*.csv")) + \
             sorted(glob.glob("the-world-data/r323_20261002/d3_cloud/*.csv")):
        if p.endswith(".log"):
            continue
        r = rows(p)
        if not r:
            continue
        m = re.match(r"d3_p(\d+)_(\d+)_(off|on)$", os.path.basename(p)[:-4])
        if not m:
            continue
        pc, sd, arm = int(m.group(1)), int(m.group(2)), m.group(3)
        x = r[-1]
        out.append(dict(pc=pc, sd=sd, arm=arm, host="cloud" if "cloud" in p else "local",
                        K=fl(x, "pop"), ac=fl(x, "abs_cap"),
                        out_frac=1 - fl(x, "hr_hist_0_5"),
                        nv=fl(x, "never_visited_patch_cell_frac"),
                        gini=fl(x, "l2_visited_gini"), sat=fl(x, "patch_sat_mean")))
    return out


D3 = load_d3()
print("%-6s %-6s %-4s %-5s %9s %11s %11s %13s" % ("patches", "seed", "臂", "支", "K_end", "出走占比", "未访斑块格", "斑块饱和"))
for x in sorted(D3, key=lambda z: (z["pc"], z["sd"], z["arm"])):
    print("%-6d %-6d %-4s %-5s %9d %10.4f %11.4f %13.4f"
          % (x["pc"], x["sd"], x["arm"], br_of(x["sd"]), x["K"], x["out_frac"], x["nv"], x["sat"]))
print("-" * 84)
for pc in (400, 1700, 4000):
    sub = [x for x in D3 if x["pc"] == pc]
    K = [x["K"] for x in sub]
    of = [x["out_frac"] for x in sub if np.isfinite(x["out_frac"])]
    nv = [x["nv"] for x in sub if np.isfinite(x["nv"])]
    print("p%-5d n=%2d  K 中位 %6.0f [%5.0f–%6.0f]  出走占比均值 %.4f  未访斑块格均值 %.4f"
          % (pc, len(sub), np.median(K), min(K), max(K), np.mean(of), np.mean(nv)))
print("-" * 84)
print("按支拆分（on 臂，避开 off 臂的近零值）：出走占比 vs 斑块数")
print("%-6s %10s %10s %10s %10s" % ("斑块数", "A支 n", "A支 出走占比", "B支 n", "B支 出走占比"))
for pc in (400, 1700, 4000):
    for tag in ("A", "B"):
        pass
    a_ = [x["out_frac"] for x in D3 if x["pc"] == pc and x["arm"] == "on" and br_of(x["sd"]) == "A"]
    b_ = [x["out_frac"] for x in D3 if x["pc"] == pc and x["arm"] == "on" and br_of(x["sd"]) == "B"]
    print("%-6d %10s %10s %10s %10s"
          % (pc, len(a_), ("%.4f" % np.mean(a_)) if a_ else "-",
             len(b_), ("%.4f" % np.mean(b_)) if b_ else "-"))

# ============================================================ 设计 4
print()
print(SEP)
print("【设计 4】go（移除出走型）vs rand（等量随机）—— 12k tick")
print(SEP)
D4 = {}
for p in sorted(glob.glob("_rerun_logs/d4_local/*.csv")):
    m = re.match(r"d3_p(\d+)_(\d+)_(off|on)_(go|rand)$", os.path.basename(p)[:-4])
    if not m:
        continue
    sd, arm, mode = int(m.group(2)), m.group(3), m.group(4)
    r = rows(p)
    x = r[-1]
    D4[(sd, arm, mode)] = dict(K=fl(x, "pop"), ac=fl(x, "abs_cap"),
                               drs=fl(x, "hr_dr_stay"), drsm=fl(x, "hr_dr_stay_m"),
                               drg=fl(x, "hr_dr_go"), drgm=fl(x, "hr_dr_go_m"),
                               of=1 - fl(x, "hr_hist_0_5"), g22=fl(x, "g22_mean"))

print("%-6s %-4s %-5s %9s %9s %9s %9s %10s %10s"
      % ("seed", "臂", "支", "K_go", "K_rand", "ΔK%", "dr_stay_go", "dr_stay_rand", "Δdr_stay"))
for sd in (194, 195, 198, 199):
    for arm in ("on", "off"):
        g, r_ = D4.get((sd, arm, "go")), D4.get((sd, arm, "rand"))
        if not g or not r_:
            continue
        dk = (g["K"] - r_["K"]) / max(r_["K"], 1) * 100
        dd = g["drsm"] - r_["drsm"]
        print("%-6d %-4s %-5s %9d %9d %+8.1f%% %10.4f %11.4f %+10.4f"
              % (sd, arm, br_of(sd), g["K"], r_["K"], dk, g["drsm"], r_["drsm"], dd))
print("-" * 84)
for arm in ("on", "off"):
    dk = []
    dd = []
    for sd in (194, 195, 198, 199):
        g, r_ = D4.get((sd, arm, "go")), D4.get((sd, arm, "rand"))
        if not g or not r_:
            continue
        dk.append((g["K"] - r_["K"]) / max(r_["K"], 1) * 100)
        dd.append(g["drsm"] - r_["drsm"])
    m1, l1, h1 = boot_ci(dk)
    p1, np1, nn1 = sign_p(dk)
    m2, l2, h2 = boot_ci(dd)
    p2, np2, nn2 = sign_p(dd)
    print("%-4s 臂 (n=%d)：" % (arm, len(dk)))
    print("     ΔK%%        均值 %+7.2f%%  95%%CI [%+.2f%%, %+.2f%%]  符号 p=%.3f (+%d/−%d)"
          % (m1, l1, h1, p1, np1, nn1))
    print("     Δdr_stay_m 均值 %+.5f  95%%CI [%+.5f, %+.5f]  符号 p=%.3f (+%d/−%d)"
          % (m2, l2, h2, p2, np2, nn2))

# ============================================================ g22
print()
print(SEP)
print("【g22 双端扰动】初始 0.1 / 0.9，之后自由演化 20k")
print(SEP)
print("%-6s %-4s %7s %10s %10s %10s %11s %9s"
      % ("seed", "臂", "init", "g22@2k", "g22@10k", "g22@20k", "Δ(20k-2k)", "K_end"))
gd = {}
for p in sorted(glob.glob("_rerun_logs/g22_local/*.csv")):
    m = re.match(r"d3_p\d+_(\d+)_(off|on)_g([\d.]+)$", os.path.basename(p)[:-4])
    if not m:
        continue
    sd, arm, ini = int(m.group(1)), m.group(2), float(m.group(3))
    r = rows(p)
    dd = {x["tick"]: x for x in r}
    def g(t):
        return fl(dd[t], "g22_mean") if t in dd else float("nan")
    gd[(sd, arm, ini)] = dict(g2=g("2000"), g10=g("10000"), g20=g("20000"), K=fl(r[-1], "pop"))
    print("%-6d %-4s %7.1f %10.4f %10.4f %10.4f %+11.4f %9d"
          % (sd, arm, ini, g("2000"), g("10000"), g("20000"), g("20000") - g("2000"), fl(r[-1], "pop")))
print("-" * 84)
for ini in (0.1, 0.9):
    dv = [v["g20"] - v["g2"] for k, v in gd.items() if k[2] == ini]
    m1, l1, h1 = boot_ci(dv)
    print("init=%.1f：Δ(g22@20k − @2k) 均值 %+.4f  95%%CI [%+.4f, %+.4f]  ⇒ %s"
          % (ini, m1, l1, h1,
             "向高漂" if m1 > 0.02 else ("向低漂" if m1 < -0.02 else "基本不动")))
print()
print("on 臂 K（适应度代理）：低起始 vs 高起始")
for sd in (165, 167):
    lo = gd.get((sd, "on", 0.1), {}).get("K", float("nan"))
    hi = gd.get((sd, "on", 0.9), {}).get("K", float("nan"))
    if np.isfinite(lo) and np.isfinite(hi):
        print("   seed %d：g22=0.1 → K=%d ｜ g22=0.9 → K=%d ｜ 差 %+.1f%%"
              % (sd, lo, hi, (lo - hi) / hi * 100))
