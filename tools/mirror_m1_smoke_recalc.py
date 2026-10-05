# -*- coding: utf-8 -*-
"""M1-SMOKE 独立复算（镜，2026-10-05）——PI 核数门件（板 697e171 二·③）。

独立性口径（R394③ / @64f060f 家规）：
  · 输入 = 数据仓原始 `m1_smoke_d1700_s*_t2000.csv` + `*.summary.json`，**逐字段自读**；
  · **不 import** 砚 `tools/m1_smoke_readout.py` / 探针内部 / 任何判读模块（实现独立 ⇒ 口径可对照）；
  · 只做复算与对账，**不改判读原稿**（AGENT.md §二）、**不代判**（判读权归砚）、**不裁路线**（PI/fish）。

窗切点：本脚本**同时**按两种口径算——
  A = M1 v1.1 §四 底（@c5d34a1）锁定式：`k=int(0.25*n)` ⇒ `samples[:k]` / `samples[-k:]`
  B = tick 阈值式：`tick <= T/4` / `tick > 3T/4`
两者是否等价**逐 seed 报出**（本批恰合 ⇒ 读数不受影响；正式批漂移 ⇒ 落纸）。

用法：
  py -3 tools/mirror_m1_smoke_recalc.py --data the-world-data/m1_smoke --ticks 2000
  py -3 tools/mirror_m1_smoke_recalc.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
import sys

NAN = float("nan")
# 单侧 95% t 临界值（df -> t_{0.95}），手工表避免外部依赖
T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895,
       8: 1.860, 9: 1.833, 10: 1.812, 11: 1.796, 12: 1.782, 13: 1.771,
       14: 1.761, 15: 1.753, 20: 1.725, 24: 1.711, 29: 1.699, 30: 1.697}

# 砚判读稿 §四（c354678）报告值 —— 仅用于**对账**，不参与任何计算
YAN_TABLE = {
    207: (0.5300, 0.5218, 0.985, -0.008, 5739, 776, 0.364),
    208: (0.4828, 0.4930, 1.021, +0.010, 5914, 788, 0.333),
    209: (0.4931, 0.5275, 1.070, +0.034, 6012, 829, 0.357),
    210: (0.4968, 0.4995, 1.006, +0.003, 6037, 812, 0.324),
    211: (0.4881, 0.4988, 1.022, +0.011, 5782, 818, 0.355),
    212: (0.4951, 0.4876, 0.985, -0.008, 5803, 805, 0.358),
    213: (0.4909, 0.4978, 1.014, +0.007, 5465, 819, 0.356),
    214: (0.4928, 0.4828, 0.980, -0.010, 5757, 833, 0.289),
}
YAN_AGG = {"ratio_mean": 1.0189, "ge_first": 5, "lt_first": 3}


def fnum(val):
    """CSV 值 → float；空/NaN 一律显式返回 NAN（不静默填 0，S4′ 精神）。"""
    if val is None:
        return NAN
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return NAN
    try:
        return float(s)
    except ValueError:
        return NAN


def mean_ns(vals):
    clean = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return (statistics.fmean(clean) if clean else NAN), len(vals) - len(clean)


def windows_index(rows, frac=0.25):
    """口径 A = 锁定 index 切片（M1 v1.1 §四 底 @c5d34a1）。"""
    n = len(rows)
    k = int(frac * n)
    return rows[:k], (rows[n - k:] if k else []), k


def windows_tick(rows, ticks):
    """口径 B = tick 阈值（q1=T/4 含，q3=3T/4 不含）。"""
    q1, q3 = ticks / 4.0, 3.0 * ticks / 4.0
    return ([r for r in rows if r["tick"] <= q1],
            [r for r in rows if r["tick"] > q3], None)


def read_run(path):
    with open(path, encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))
    if not raw:
        raise SystemExit(f"fail-loud: {path} 无数据行")
    cols = set(raw[0].keys())
    rows = []
    for r in raw:
        rows.append({
            "seed": int(r["seed"]), "arm": r["arm"], "tick": int(r["tick"]),
            "pop": fnum(r.get("pop")), "g15": fnum(r.get("g15_mean")),
            "n_patches": fnum(r.get("n_patches")),
            "g5_dir": fnum(r.get("g5_dir_diff")), "g5_move": fnum(r.get("g5_move_diff")),
            "g5_n_sig": fnum(r.get("g5_n_sig")), "g4_bit": fnum(r.get("g4_mem_bit_frac")),
        })
    sig_cols = sorted(c for c in cols if c.startswith("sig_run_len_hist_"))
    for r, raw_r in zip(rows, raw):
        r["sig_hist_sum"] = sum(fnum(raw_r[c]) for c in sig_cols)
    return rows, cols, sig_cols


def check_series(rows, sample, ticks):
    """落盘结构校验：等差/递增/无重复/行数——复算前置门，不做静默容忍。"""
    t = [r["tick"] for r in rows]
    probs = []
    if len(set(t)) != len(t):
        probs.append("tick 重复")
    if any(b <= a for a, b in zip(t, t[1:])):
        probs.append("tick 非严格递增")
    if any(b - a != sample for a, b in zip(t, t[1:])):
        probs.append("tick 步长 != sample")
    exp = ticks // sample
    if len(rows) != exp:
        probs.append(f"行数 {len(rows)} != ticks/sample {exp}")
    if t[0] != sample:
        probs.append(f"首采样点 tick={t[0]}（!= sample ⇒ tick 0 未采样）")
    return probs


def summarize_run(rows, sig_cols):
    pops = [r["pop"] for r in rows]
    nps = [r["n_patches"] for r in rows]
    g4 = [r["g4_bit"] for r in rows]
    return {
        "pop_first_sampled": pops[0],
        "pop_min_sampled": min(pops),
        "pop_last": pops[-1],
        "n_patch_first": nps[0], "n_patch_last": nps[-1], "n_patch_min": min(nps),
        "g4_bit_nonzero": any((not math.isnan(v)) and v > 0 for v in g4),
        "sig_hist_total": sum(r["sig_hist_sum"] for r in rows),
    }


def stats_block(vals):
    vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"n": 0, "mean": NAN, "sd": NAN, "lb": NAN, "t": NAN}
    m = statistics.fmean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    df = len(vals) - 1
    tc = T95.get(df, 1.96)
    if len(vals) > 1 and sd > 0:
        lb = m - tc * sd / math.sqrt(len(vals))
        tt = m / (sd / math.sqrt(len(vals)))
    else:
        lb, tt = (m if sd == 0 else NAN), (NAN if sd == 0 else NAN)
    return {"n": len(vals), "mean": m, "sd": sd, "lb": lb, "t": tt}


def recalc(data_dir, ticks, sample):
    files = sorted(glob.glob(os.path.join(data_dir, "m1_smoke_d1700_s*_t2000.csv")))
    if not files:
        raise SystemExit(f"fail-loud: {data_dir} 无 m1_smoke csv")
    per = {}
    for fp in files:
        seed = int(re.search(r"_s(\d+)_t", os.path.basename(fp)).group(1))
        rows, cols, sig_cols = read_run(fp)
        probs = check_series(rows, sample, ticks)
        a1, a9, ka = windows_index(rows)
        b1, b9, _ = windows_tick(rows, ticks)
        same_cut = ([r["tick"] for r in a1] == [r["tick"] for r in b1]
                    and [r["tick"] for r in a9] == [r["tick"] for r in b9])

        def win(rs):
            m, nnan = mean_ns([r["g15"] for r in rs])
            return m, nnan, [r["tick"] for r in rs]

        a_f, a_f_nan, a_f_t = win(a1)
        a_l, a_l_nan, a_l_t = win(a9)
        b_f, _, _ = win(b1)
        b_l, _, _ = win(b9)
        g15_nan_total = sum(1 for r in rows if math.isnan(r["g15"]))
        ratio_a = a_l / a_f if a_f and not math.isnan(a_f) else NAN
        delta_a = a_l - a_f
        mor_pairs = [r2["g15"] / r1["g15"] for r1, r2 in zip(a1, a9)
                     if r1["g15"] and not math.isnan(r1["g15"])
                     and not math.isnan(r2["g15"])]
        mor = statistics.fmean(mor_pairs) if len(mor_pairs) == len(a1) and mor_pairs else NAN
        g5d, g5d_nan = mean_ns([r["g5_dir"] for r in rows])
        g5m, _ = mean_ns([r["g5_move"] for r in rows])
        nsig, _ = mean_ns([r["g5_n_sig"] for r in rows])

        summ_p = fp[:-4] + ".summary.json"
        stop, k_final, meta = "no-summary", NAN, {}
        if os.path.exists(summ_p):
            with open(summ_p, encoding="utf-8") as fh:
                sj = json.load(fh)
            hit = [r for r in sj.get("runs", [])
                   if str(r.get("seed")) == str(seed) and r.get("arm") == rows[0]["arm"]]
            if len(hit) != 1:
                probs.append(f"summary (seed={seed},arm={rows[0]['arm']}) 命中 {len(hit)} 条（应为 1）")
            if hit:
                stop, k_final = hit[0].get("stop"), fnum(hit[0].get("K_final"))
            meta = sj.get("meta", {})
        per[seed] = {
            "n_rows": len(rows), "k": ka,
            "a_first": a_f, "a_last": a_l, "ratio_a": ratio_a, "delta_a": delta_a,
            "mor": mor, "b_first": b_f, "b_last": b_l,
            "ratio_b": (b_l / b_f) if b_f else NAN,
            "cut_same": same_cut, "a_first_ticks": a_f_t, "a_last_ticks": a_l_t,
            "g15_nan": g15_nan_total, "win_nan": a_f_nan + a_l_nan,
            "g5_dir": g5d, "g5_dir_nan": g5d_nan, "g5_move": g5m, "g5_n_sig": nsig,
            "stop": stop, "k_final": k_final,
            "emit_col": next((c for c in cols if "emit" in c.lower()), None),
            "probs": probs, "summ_meta": meta,
            **summarize_run(rows, sig_cols),
        }
    return per


def report(per, ticks):
    seeds = sorted(per)
    print(f"# 独立复算 M1-SMOKE（镜）｜T={ticks}")
    print("\n## 一、逐 seed 复算（口径 A = 锁定 index 切片）\n")
    print("seed | n | k=int(.25n) | g15首 | g15末 | 比值 | Δ | 比值(口径B) | 两切法一致 | 存活stop | pop末 | K_final | pop最小(采样) | g5_dir | sig_histΣ")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in seeds:
        d = per[s]
        print(f"{s} | {d['n_rows']} | {d['k']} | {d['a_first']:.4f} | {d['a_last']:.4f} | "
              f"{d['ratio_a']:.4f} | {d['delta_a']:+.4f} | {d['ratio_b']:.4f} | "
              f"{'✅' if d['cut_same'] else '🔴否'} | {d['stop']} | {d['pop_last']:.0f} | "
              f"{d['k_final']:.0f} | {d['pop_min_sampled']:.0f} | {d['g5_dir']:.4f} | {d['sig_hist_total']:.0f}")

    print("\n### 窗内 tick 集（首/末，口径 A）")
    for s in seeds:
        d = per[s]
        print(f"  seed {s}: first={d['a_first_ticks']} last={d['a_last_ticks']}")

    print("\n## 二、聚合统计（seed 级，n=存活 seed 数；SMOKE 不出 L0 判定）\n")
    ag_a = stats_block([per[s]["ratio_a"] for s in seeds])
    ag_b = stats_block([per[s]["ratio_b"] for s in seeds])
    ag_mor = stats_block([per[s]["mor"] for s in seeds])
    ag_d = stats_block([per[s]["delta_a"] for s in seeds])
    ge = sum(1 for s in seeds if per[s]["delta_a"] >= 0)
    lt = len(seeds) - ge
    abs_le_001 = sum(1 for s in seeds if abs(per[s]["delta_a"]) <= 0.01)
    print(f"比值(末/首，均值口径)  口径A: mean={ag_a['mean']:.4f} SD={ag_a['sd']:.4f} "
          f"单侧95%下界={ag_a['lb']:.4f}")
    print(f"                       口径B: mean={ag_b['mean']:.4f} SD={ag_b['sd']:.4f}")
    print(f"逐窗比值的均值(mean-of-ratios): mean={ag_mor['mean']:.4f} SD={ag_mor['sd']:.4f} "
          f"（与比值-of-均值 差 {ag_mor['mean']-ag_a['mean']:+.4f}）")
    print(f"Δ(末−首): mean={ag_d['mean']:+.4f} SD={ag_d['sd']:.4f} t={ag_d['t']:+.3f} "
          f"单侧95%下界={ag_d['lb']:+.4f}")
    print(f"方向一致数: 末≥首 {ge}/{len(seeds)}，末<首 {lt}/{len(seeds)}；|Δ|≤0.01 中性 {abs_le_001}/{len(seeds)}")
    print(f"两切法一致 seed 数: {sum(1 for s in seeds if per[s]['cut_same'])}/{len(seeds)}")

    print("\n## 三、与砚判读稿 §四 对账（c354678，容差 5e-4 / pop 整数 / 比值 1e-3）\n")
    bad = 0
    print("seed | 项 | 砚值 | 我值 | 差 | 判")
    print("|---|---|---|---|---|---|")
    for s in seeds:
        d = per[s]
        y = YAN_TABLE.get(s)
        if not y:
            print(f"{s} | — | 砚稿无此行 | | | 🔴")
            bad += 1
            continue
        mine = (d["a_first"], d["a_last"], d["ratio_a"], d["delta_a"],
                d["pop_last"], d["pop_min_sampled"], d["g5_dir"])
        names = ("g15首", "g15末", "比值", "Δ", "pop_end", "pop_min", "g5_dir")
        for nm, yv, mv in zip(names, y, mine):
            tol = 0.5 if nm in ("pop_end", "pop_min") else (1e-3 if nm in ("比值", "Δ", "g5_dir") else 5e-4)
            ok = (not math.isnan(mv)) and abs(yv - mv) <= tol
            if not ok:
                bad += 1
            print(f"{s} | {nm} | {yv:.4f} | {mv:.4f} | {mv-yv:+.4f} | {'✅' if ok else '🔴差'}")
    agg_ok = abs(ag_a["mean"] - YAN_AGG["ratio_mean"]) <= 1e-3
    if not agg_ok:
        bad += 1
    sign_ok = ge == YAN_AGG["ge_first"]
    if not sign_ok:
        bad += 1
    print(f"聚合 | 比值均值 | {YAN_AGG['ratio_mean']:.4f} | {ag_a['mean']:.4f} | "
          f"{ag_a['mean']-YAN_AGG['ratio_mean']:+.4f} | {'✅' if agg_ok else '🔴差'}")
    print(f"聚合 | 末≥首 | {YAN_AGG['ge_first']}/8 | {ge}/8 | — | "
          f"{'✅' if sign_ok else '🔴差'}")
    print(f"\n对账不符项合计 = {bad}（逐格容差：pop 0.5 / 比值·Δ·g5 1e-3 / g15 5e-4）")

    print("\n## 四、独立旁证与结构校验（fail-loud 位）\n")
    print(f"g15_mean NaN 行数合计 = {sum(per[s]['g15_nan'] for s in seeds)}"
          f"；窗内 NaN = {sum(per[s]['win_nan'] for s in seeds)}（>0 即须上板，不得静默）")
    print(f"g5_dir_diff NaN 行数合计 = {sum(per[s]['g5_dir_nan'] for s in seeds)}")
    print(f"emit_rate 列存在性 = {sorted({str(per[s]['emit_col']) for s in seeds})}"
          f"（None ⇒ I-7「发射率不降」**无列可算**，不只是口径不等价）")
    print(f"summary K_final vs CSV 末行 pop 最大差 = "
          f"{max(abs(per[s]['k_final'] - per[s]['pop_last']) for s in seeds):.0f}")
    print(f"stop 取值集合 = {sorted({per[s]['stop'] for s in seeds})}"
          f" ⇒ 存活判据 = s3 `stop`（N==0 ⇒ '灭绝'），**无 extinct_at 字段**（中途复元不可见）")
    print("斑块存续（配置 patches=1700 对照）：")
    for s in seeds:
        d = per[s]
        print(f"  seed {s}: 首采样 n_patches={d['n_patch_first']:.0f} "
              f"末采样={d['n_patch_last']:.0f} 最小={d['n_patch_min']:.0f} "
              f"⇒ 相对 1700 存续率 首={d['n_patch_first']/1700:.3f} 末={d['n_patch_last']/1700:.3f}")
    print(f"G4 mem_bit 非零 = {sorted({per[s]['g4_bit_nonzero'] for s in seeds})}"
          f"（全 False ⇒ 与 alphabet '16' 偏差自洽）")
    print("\n结构校验：")
    any_prob = False
    for s in seeds:
        for p in per[s]["probs"]:
            any_prob = True
            print(f"  🔴 seed {s}: {p}")
    if not any_prob:
        print("  ✅ 8 run 全部：tick 等差=100/严格递增/无重复/行数=20")
    print("  ⚠️ 首采样点 = tick 100 ⇒ tick 0–100 轨迹**未采样**：pop_min 是**采样最小值**不是轨迹最小值；"
          "「10000→~0.8k 早期瓶颈」中的谷底时刻与深浅不可从 CSV 断定")
    argv_dev = sorted({tuple(sorted((k, v) for k, v in per[s]["summ_meta"].items()
                                    if k in ("device", "rows", "cols", "patches", "pop",
                                             "rgm", "bg_low_prod_frac", "bg_low_cap_mult",
                                             "sample", "ticks", "arms", "m0_instruments",
                                             "signal_alphabet")))
                       for s in seeds})
    print(f"装置对账（summary.meta）: {argv_dev}")
    return bad


def selftest():
    """口径 A/B 等价性自证：本批参数重合，正式批（T=3000）末窗漂移 8 vs 7。"""
    print("# selftest 窗切点口径")
    ok = True
    for ticks, exp_a, exp_b_first, exp_b_last in ((2000, 5, 5, 5), (3000, 7, 7, 8)):
        rows = [{"tick": t, "g15": float(t)} for t in range(100, ticks + 1, 100)]
        a1, a9, k = windows_index(rows)
        b1, b9, _ = windows_tick(rows, ticks)
        line = (f"T={ticks} n={len(rows)} 口径A k=int(.25n)={k} ⇒ 首{len(a1)}窗/末{len(a9)}窗；"
                f"口径B ⇒ 首{len(b1)}窗/末{len(b9)}窗")
        good = (k == exp_a and len(a1) == exp_a and len(a9) == exp_a
                and len(b1) == exp_b_first and len(b9) == exp_b_last)
        ok = ok and good
        print(("  ✅ " if good else "  🔴 ") + line)
    rows = [{"tick": t, "g15": 0.5 + 0.001 * i} for i, t in enumerate(range(100, 2001, 100))]
    a1, a9, _ = windows_index(rows)
    m1, _ = mean_ns([r["g15"] for r in a1])
    m9, _ = mean_ns([r["g15"] for r in a9])
    up = m9 > m1
    print(f"  {'✅' if up else '🔴'} 合成上升序列 ⇒ 比值 {m9/m1:.4f} > 1（T6「上升必 >1」）")
    flat = [{"tick": t, "g15": 0.5} for t in range(100, 2001, 100)]
    f1, f9, _ = windows_index(flat)
    r_flat = f9[0]["g15"] / f1[0]["g15"]
    print(f"  {'✅' if abs(r_flat-1) < 1e-12 else '🔴'} 合成平稳序列 ⇒ 比值 {r_flat:.4f} = 1")
    print(f"  {'✅' if not math.isnan(mean_ns([1.0, NAN, 2.0])[0]) and mean_ns([1.0, NAN, 2.0])[1] == 1 else '🔴'} "
          "NaN 过滤计数（不静默填 0）")
    return 0 if ok and up else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="the-world-data/m1_smoke")
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    per = recalc(a.data, a.ticks, a.sample)
    bad = report(per, a.ticks)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
