# -*- coding: utf-8 -*-
"""ZEROPAT-AUDIT · 砚数据面独立清算脚本（2026-10-08，纯 stdlib，只读，不重跑不改数据）。

独立读路纪律（R394④）：不 import `tools/mirror_zeropat_audit.py`（镜链路）、
不 import `tools/e056_readout.py`（原读数链路）——本脚本自带解析与统计，
仅共享同一**已锁口径文本**（index 式 k=int(0.25n)、floor；A1 裁定）。

三选一证据面：
- (c) G6 仪表 bug：由 CSV 的 `g6_emit_events_cum / g6_emit_person_ticks_cum`
  用**合并比**独立重算 emit_rate_first/last/ratio，与 summary 原生值对表；
  另核逐窗列自洽（`g6_emit_rate_window` ≡ Δevents/Δpt）。
- (a) 口径重复：逐 seed `emit_rate_ratio` vs `g15_ratio`（同切点）符号一致率＋
  |差|；逐窗 `g6_emit_rate_window` 对 `g15_mean` 的 Pearson r（逐 seed＋合并）。
- zero_pat 语义：summary `emit_zero_pattern_n` 值 + `emit_events_total>0`
  （码路事实："8" 档 code_offset=1 ⇒ pattern 0 结构不可达，见清算稿）。
- G4 附项：`g4_mem_bit_frac` 逐行 ≡ on/n 自洽核 + 逐 seed 均值。

用法：PYTHONIOENCODING=utf-8 py tools/yan_zeropat_data_audit.py --data the-world-data/e056_m1
"""
from __future__ import annotations
import argparse, csv, glob, json, math, os, re, statistics


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def f(v):
    v = (v or "").strip()
    if v in ("", "nan", "NaN"):
        return None
    return float(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="the-world-data/e056_m1")
    a = ap.parse_args()

    print("seed | n | k | g15_ratio | emit_ratio(独立) | emit_ratio(summary) | Δ独立-sum | sign(emit,g15) | |差| | r(逐窗) | zp | ev_tot>0 | g4均值 | g4自洽maxΔ | 窗列自洽maxΔ")
    per_seed, pooled_x, pooled_y = [], [], []
    max_d_sum = 0.0
    max_d_window = 0.0
    max_d_g4 = 0.0
    n_sign_match = 0
    n_seed = 0
    diffs = []
    for fp in sorted(glob.glob(os.path.join(a.data, "e056_m1_d1700_s*_t2000.csv"))):
        seed = re.search(r"_s(\d+)_t", os.path.basename(fp)).group(1)
        rows = list(csv.DictReader(open(fp, encoding="utf-8")))
        n = len(rows)
        k = int(0.25 * n)                      # 已锁 index 式（floor，A1）
        g15 = [f(r["g15_mean"]) for r in rows]
        erw = [f(r["g6_emit_rate_window"]) for r in rows]
        ev = [int(float(r["g6_emit_events_cum"])) for r in rows]
        pt = [int(float(r["g6_emit_person_ticks_cum"])) for r in rows]

        # (c) 合并比独立重算（首/末 k 窗）
        first = (ev[k - 1] - 0) / (pt[k - 1] - 0) if k >= 1 and pt[k - 1] else float("nan")
        last = (ev[n - 1] - ev[n - k - 1]) / (pt[n - 1] - pt[n - k - 1]) if k >= 1 else float("nan")
        er_ind = last / first
        summ = json.load(open(fp.replace(".csv", ".summary.json"), encoding="utf-8"))
        r0 = [r for r in summ["runs"] if str(r.get("seed")) == seed][-1]
        er_sum = r0["emit_rate_ratio"]
        d_sum = abs(er_ind - er_sum)
        max_d_sum = max(max_d_sum, d_sum)

        # 逐窗列自洽（CSV 内部）
        d_win = 0.0
        for i in range(n):
            base_ev = ev[i - 1] if i else 0
            base_pt = pt[i - 1] if i else 0
            dev, dpt = ev[i] - base_ev, pt[i] - base_pt
            exp = dev / dpt if dpt else None
            got = erw[i]
            if exp is None or got is None:
                if exp is not None or got is not None:
                    d_win = max(d_win, 1.0)     # NaN 形态不一致
            else:
                d_win = max(d_win, abs(exp - got))
        max_d_window = max(max_d_window, d_win)

        # (a) g15 同切点比值＋逐窗 Pearson r
        g15f, g15l = statistics.fmean(g15[:k]), statistics.fmean(g15[-k:])
        gr = g15l / g15f
        idx = [i for i in range(n) if erw[i] is not None and g15[i] is not None]
        r_seed = pearson([erw[i] for i in idx], [g15[i] for i in idx])
        for i in idx:
            pooled_x.append(erw[i])
            pooled_y.append(g15[i])
        s_em, s_g = (er_ind >= 1.0) == (gr >= 1.0), abs(er_ind - gr)
        n_sign_match += int(s_em)
        diffs.append(s_g)
        n_seed += 1
        per_seed.append((seed, gr, er_ind, er_sum, s_g, r_seed))

        # zero_pat 语义件
        zp = r0.get("emit_zero_pattern_n")
        ev_tot = r0.get("emit_events_total")
        # G4 自洽：frac ≡ on/n（逐行）
        d_g4 = 0.0
        g4vals = []
        for row in rows:
            fr, on, nn = f(row.get("g4_mem_bit_frac")), f(row.get("g4_mem_bit_on")), f(row.get("g4_mem_bit_n"))
            if fr is None:
                continue
            g4vals.append(fr)
            if on is not None and nn:
                d_g4 = max(d_g4, abs(fr - on / nn))
        max_d_g4 = max(max_d_g4, d_g4)

        print(f"{seed} | {n} | {k} | {gr:.4f} | {er_ind:.4f} | {er_sum:.4f} | {d_sum:.2e} | "
              f"{'同' if s_em else '异'} | {s_g:.4f} | {r_seed:.4f} | {zp} | {ev_tot and ev_tot > 0} | "
              f"{statistics.fmean(g4vals):.4f} | {d_g4:.2e} | {d_win:.2e}")

    r_pool = pearson(pooled_x, pooled_y)
    print(f"\nAGG | seeds={n_seed} 符号一致={n_sign_match}/{n_seed} "
          f"|emit_ratio-g15_ratio| max={max(diffs):.4f} mean={statistics.fmean(diffs):.4f} "
          f"| emit_ratio 独立重算 vs summary maxΔ={max_d_sum:.2e} "
          f"| 逐窗列自洽 maxΔ={max_d_window:.2e} | G4 frac≡on/n maxΔ={max_d_g4:.2e} "
          f"| Pearson r 合并({len(pooled_x)}窗)={r_pool:.4f}")
    rs = [p[5] for p in per_seed if not math.isnan(p[5])]
    print(f"AGG2 | Pearson r 逐seed min={min(rs):.4f} max={max(rs):.4f}（各 {len(pooled_x)//n_seed} 窗）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
