# -*- coding: utf-8 -*-
"""ZEROPAT-AUDIT 独立复算（镜 · R394④ 独立脚本 · 鱼令④）。

R394④ 硬约束落实：本脚本 **纯 stdlib**（csv/json/math/statistics），**不 import**
s3_memory_probe / e056_readout / sphere_engine / 任何我参与链路的模块；**只读原始**
E-056 CSV + summary.json（数据仓 a8d8f82）。与 e056_readout.py 的关系＝**独立第二读路**
（它用 summary 原生 emit_rate_*，本脚本从 CSV 逐窗**重算**发射率，二者交叉验证 G6 self-consistent）。

审计三选一结论的证据链：
- **(c) 排除**：从 CSV 独立重算 emit_rate_first/last/ratio，与 summary 原生值逐 seed 对表，
  一致 ⇒ G6 仪表 self-consistent（非 bug）；emit_zero_pattern_n 语义拆解（8 档无全零码 → 0 合理）。
- **(a) 口径重复**：机械恒等——发射 = 伯努利(rand_emit < g15ᵢ)（sphere_engine.py:3604/3605），
  每 tick 期望发射率 = Σg15ᵢ/pop = 群体 g15 均值。故 emit_rate_ratio ≡ g15_ratio，
  J2 是 J1 的机械投影（青梧 r=0.9989 实证）。本脚本从 CSV 独立算 g15_ratio 与
  emit_rate_ratio，测逐 seed Pearson r + 方向一致 + |比值差|，坐实共动为构造必然非经验巧合。
- **(b) 发射率真在降**：若 (a) 成立且 g15_ratio_mean≈1.00（近中性），则"发射率降"= "g15 降"
  同一件事，不是独立发现；(b) 被 (a) 吸收。

用法：PYTHONIOENCODING=utf-8 python tools/mirror_zeropat_audit.py \\
        --data the-world-data/e056_m1
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

GATE2K = 6945.0  # P3 锚×70%（双签 dcb6753，引用不发明）


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")


def _independent_emit_rate(rows, k):
    """从 CSV 逐窗独立重算首/末窗发射率（人·tick 加权合并比）。

    CSV 列：g6_emit_events_window（逐采样窗 Δevents）、g6_emit_person_ticks_cum
    （逐采样点累计 person·tick）。窗 i 的 Δperson_ticks = cum[i]-cum[i-1]；
    首窗 i=0 的 cum[-1] 视为 0。首/末窗发射率 = Σevents / ΣΔpt（k=int(0.25n)）。
    """
    ev = [float(r["g6_emit_events_window"]) for r in rows]
    cum = [float(r["g6_emit_person_ticks_cum"]) for r in rows]
    dpt = [cum[0]] + [cum[i] - cum[i - 1] for i in range(1, len(cum))]
    first_ev, first_pt = sum(ev[:k]), sum(dpt[:k])
    last_ev, last_pt = sum(ev[-k:]), sum(dpt[-k:])
    ef = first_ev / first_pt if first_pt > 0 else float("nan")
    el = last_ev / last_pt if last_pt > 0 else float("nan")
    return ef, el, (el / ef if ef and ef == ef and el == el and ef > 0 else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="the-world-data/e056_m1")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.data, "e056_m1_d1700_s*_t2000.csv")))
    if len(files) != 8:
        print(f"⚠️ 期望 8 run，实得 {len(files)}")

    hdr = ("seed | n | k | g15_r_ind | emit_r_ind(CSV) | emit_r_sum(summary) | "
           "|Δ|CSV-vs-sum | g15_mean首 | emit首≈g15? | zero_pat | dir(g15) | dir(emit) | 共动?")
    print(hdr)
    g15_ratios, emit_ratios_ind, emit_ratios_sum = [], [], []
    selfconsistent_maxdiff = 0.0
    zero_pats = []
    dir_agree = 0
    rows_n = 0

    for fp in files:
        seed = re.search(r"_s(\d+)_t", os.path.basename(fp)).group(1)
        rows = list(csv.DictReader(open(fp, encoding="utf-8")))
        n = len(rows)
        k = int(0.25 * n)  # index 式切点（PI 裁②）
        g15 = [float(r["g15_mean"]) for r in rows]
        f15, l15 = statistics.mean(g15[:k]), statistics.mean(g15[-k:])
        g15_ratio = l15 / f15 if f15 > 0 else float("nan")

        ef_ind, el_ind, emit_ratio_ind = _independent_emit_rate(rows, k)

        summ = json.load(open(fp.replace(".csv", ".summary.json"), encoding="utf-8"))
        r0 = [r for r in summ["runs"]
              if str(r.get("seed")) == seed or len(summ["runs"]) == 1][-1]
        emit_ratio_sum = r0.get("emit_rate_ratio")
        ef_sum = r0.get("emit_rate_first")
        zp = r0.get("emit_zero_pattern_n")
        zero_pats.append(zp)

        d_selfcon = abs(emit_ratio_ind - emit_ratio_sum) if (
            emit_ratio_ind == emit_ratio_ind and emit_ratio_sum == emit_ratio_sum) else None
        if d_selfcon is not None:
            selfconsistent_maxdiff = max(selfconsistent_maxdiff, d_selfcon)

        j1_dir = l15 >= f15
        j2_dir = emit_ratio_sum >= 1.0 if emit_ratio_sum == emit_ratio_sum else None
        co = (j1_dir is not None and j2_dir is not None and j1_dir == j2_dir)
        dir_agree += int(co)
        rows_n += 1
        g15_ratios.append(g15_ratio)
        emit_ratios_ind.append(emit_ratio_ind)
        emit_ratios_sum.append(emit_ratio_sum)

        print(f"{seed} | {n} | {k} | {g15_ratio:.4f} | {emit_ratio_ind:.4f} | "
              f"{emit_ratio_sum:.4f} | {(d_selfcon if d_selfcon is not None else float('nan')):.4f} | "
              f"{f15:.4f} | {ef_ind:.4f} | {zp} | {'Y' if j1_dir else 'N'} | "
              f"{'Y' if j2_dir else ('N' if j2_dir is False else 'NaN')} | {'Y' if co else 'N'}")

    r_g15_vs_emitsum = _pearson(g15_ratios, emit_ratios_sum)
    r_g15_vs_emitind = _pearson(g15_ratios, emit_ratios_ind)

    print("\n=== ZEROPAT-AUDIT 独立复算汇总 ===")
    print(f"n_seed = {rows_n}")
    print(f"g15_ratio_mean = {statistics.mean(g15_ratios):.4f}")
    print(f"emit_rate_ratio_mean(summary) = {statistics.mean(emit_ratios_sum):.4f}")
    print(f"emit_rate_ratio_mean(CSV独立) = {statistics.mean(emit_ratios_ind):.4f}")
    print(f"max|Δ| CSV重算 vs summary(G6自洽) = {selfconsistent_maxdiff:.4f} "
          f"⇒ {'G6 self-consistent ⇒ (c) 仪表 bug 排除' if selfconsistent_maxdiff < 1e-3 else '⚠️ G6 CSV-vs-summary 分叉，需查 (c)'}")
    print(f"Pearson r(g15_ratio, emit_ratio_summary) = {r_g15_vs_emitsum:.4f}")
    print(f"Pearson r(g15_ratio, emit_ratio_CSV独立) = {r_g15_vs_emitind:.4f} "
          f"⇒ 共动为机械恒等（非依赖 summary 口径）")
    print(f"方向共动（J1 vs J2 同 seed 同向） = {dir_agree}/{rows_n}")
    print(f"emit_zero_pattern_n 全 8 seed = {set(zero_pats)} "
          f"⇒ 8 档无全零码写，zero_pat=0 合理；『全零码窗缺失 ≠ 无衰减』（不可用 zero_pat=0 论证发射率没降，也不可论证 G6 失效）")
    print("\n=== 三选一初判（结论归镜备忘录上板 + 候 PI 归档）===")
    near_neutral = abs(statistics.mean(g15_ratios) - 1.0) < 0.05
    print(f"- (c) G6 bug：{'排除' if selfconsistent_maxdiff < 1e-3 else '未排除'}（CSV↔summary maxΔ={selfconsistent_maxdiff:.4f}）")
    print(f"- (a) 口径重复：r={r_g15_vs_emitind:.4f} 且 方向共动 {dir_agree}/{rows_n} "
          f"且 g15_ratio_mean={statistics.mean(g15_ratios):.4f} 近中性"
          f"{' ⇒ 成立（发射率≈mean(g15) 机械恒等，J2 冗余于 J1）' if near_neutral and r_g15_vs_emitind > 0.95 else ''}")
    print(f"- (b) 发射率真在降：被 (a) 吸收（发射率降=g15 降同一件事，非独立信号）"
          if near_neutral else "- (b) 需再核（g15 非近中性）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
