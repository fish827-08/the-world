#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""判读S2 判据②（新装置档 12 run）—— 严格按预注册稿 v4 口径，不改判据。

预注册：docs/设计文档/预注册-S2判据②-新装置档-20261002.md
口径要点：
  §4.0 先做吸引子归类（繁荣/压制/崩溃），同 seed 两臂不同吸引子 => 该配对单列
  §4.1 主判据 = patch_sat_init_var 末段20行 均值，全部 6 seed 均 on>off 为全票
  §4.2 支持判据 S1 l1_visited_patch_frac / S2 patch_sat_mean / S3 pop(止损<=50%) / S4 仅报
  §4.3 三态 + 第四态(纪元不足: <30代 或 有效tick<40000)
  §4.4 反证条件
用法： python _judge_s2g2.py <results_dir>
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import OrderedDict

RES = sys.argv[1] if len(sys.argv) > 1 else "_rerun_logs/s2g2_newdev/results"
TAIL_N = 20  # 预注册 §4.1 v3：末段 20 行（最后 5000 tick）


def tail_mean(rows, key, n=TAIL_N):
    """末段 n 行均值，缺值剔除。"""
    vals = []
    for r in rows[-n:]:
        v = r.get(key, "")
        if v in ("", "nan", "None"):
            continue
        try:
            vals.append(float(v))
        except ValueError:
            continue
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def classify_attractor(k_tail, k_final, k_peak):
    """§4.0 吸引子归类：按K 轨迹相对起点/峰值。
    判据（保守、可解释）：以峰值 K 为参照。
      崩溃：final/peak < 0.30
      压制：final/peak < 0.70
      繁荣：final/peak >= 0.90
      过渡：其余
    """
    if k_peak != k_peak or k_peak <= 0:
        return "未知"
    r = k_final / k_peak
    if k_final != k_final:
        return "未知"
    if r < 0.30:
        return "崩溃"
    if r < 0.70:
        return "压制"
    if r >= 0.90:
        return "繁荣"
    return "过渡"


def main():
    files = sorted(f for f in os.listdir(RES) if f.endswith(".csv") and "summary" not in f)
    data = OrderedDict()
    for fn in files:
        seed, arm = fn.replace("s2g2_", "").replace(".csv", "").split("_")
        rows = load(os.path.join(RES, fn))
        data[(int(seed), arm)] = rows

    seeds = sorted({s for s, _ in data})
    arms = sorted({a for _, a in data})
    print(f"# 判读 S2 判据②（新装置档）  results={RES}")
    print(f"# 装置回显核对 + 完整度自检")
    print()

    # ---------- 完整度自检（§七-2）----------
    print("## 零、完整度自检（预注册 §七-2）")
    bad = []
    for (s, a), rows in sorted(data.items()):
        ticks = rows[-1]["tick"]
        nrows = len(rows)
        bg = rows[-1].get("bg_low_frac_actual", "")
        flags = []
        if int(ticks) != 40000:
            flags.append(f"末tick={ticks}")
        if nrows != 161:
            flags.append(f"行数={nrows}")
        if bg not in ("0.0", "0", "0.0 ", ""):
            flags.append(f"bg_low={bg}")
        status = "OK" if not flags else "⚠️ " + ";".join(flags)
        if flags:
            bad.append(f"{s}_{a}:{'|'.join(flags)}")
        print(f"  seed{s}_{a:<4} 行={nrows:<4} 末tick={ticks:<6} bg_low={bg:<5} {status}")
    print(f"\n  ⇒ 完整度：{'✅ 全部达标' if not bad else '❌ 异常 ' + str(bad)}")
    print()

    # ---------- §4.0 吸引子归类 ----------
    print("## 一、§4.0 前置步：吸引子归类（判读第一步，非比数值）")
    attr = {}
    print(f"{'seed':<6}{'arm':<6}{'K_init':>10}{'K_peak':>10}{'K_final':>10}{'final/peak':>12}  归类")
    for s in seeds:
        for a in arms:
            rows = data.get((s, a))
            if not rows:
                continue
            ks = [float(r["pop"]) for r in rows if r.get("pop") not in ("", "nan", "None")]
            if not ks:
                continue
            k_init, k_peak, k_final = ks[0], max(ks), ks[-1]
            at = classify_attractor(tail_mean(rows, "pop"), k_final, k_peak)
            attr[(s, a)] = at
            print(f"{s:<6}{a:<6}{k_init:>10.0f}{k_peak:>10.0f}{k_final:>10.0f}{k_final/k_peak:>12.3f}  {at}")

    print("\n  **同 seed 双臂吸引子一致性**（§4.0：不同吸引子 ⇒ 该配对不可直接比，单列）")
    pair_attr = {}
    for s in seeds:
        o, n = attr.get((s, "off")), attr.get((s, "on"))
        if o is None or n is None:
            continue
        same = (o == n)
        pair_attr[s] = (o, n, same)
        mark = "✅ 同" if same else "🔴 不同 ⇒ 单列"
        print(f"  seed{s}: off={o:<4} on={n:<4} {mark}")
    print()

    # ---------- §4.1 主判据 ----------
    print("## 二、§4.1 主判据：`patch_sat_init_var` 末段 20 行均值，on > off逐 seed")
    print(f"{'seed':<6}{'off_tail':>12}{'on_tail':>12}{'Δ(on-off)':>14}{'on>off?':>10}   备注")
    deltas = []
    votes = []
    detail = OrderedDict()
    for s in seeds:
        ro, rn = data.get((s, "off")), data.get((s, "on"))
        if not ro or not rn:
            continue
        vo = tail_mean(ro, "patch_sat_init_var")
        vn = tail_mean(rn, "patch_sat_init_var")
        d = vn - vo
        win = vn > vo
        deltas.append(d)
        votes.append(win)
        note = ""
        if not pair_attr.get(s, (None, None, True))[2]:
            note = "⚠️ 双臂不同吸引子（§4.0 单列）"
        detail[s] = dict(off=vo, on=vn, d=d, win=win, note=note)
        print(f"{s:<6}{vo:>12.6f}{vn:>12.6f}{d:>14.6f}{'YES' if win else 'NO ':>10}   {note}")

    n_win = sum(votes)
    n_valid = len(votes)
    print(f"\n  ⇒ 票数：**{n_win}/{n_valid}**")
    if deltas:
        mean_d = sum(deltas) / len(deltas)
        print(f"  ⇒ Δ 均值 = {mean_d:+.6f}   中位 = {sorted(deltas)[len(deltas)//2]:+.6f}")
        pos = sum(1 for d in deltas if d > 0)
        print(f"  ⇒ Δ 为正：{pos}/{len(deltas)}")

    # ---------- on臂异质性必要条件 ----------
    on_vals = [detail[s]["on"] for s in detail]
    print(f"  ⇒ 必要条件「on 臂 patch_sat_init_var_tail > 0」："
          f"{'✅ 满足' if all(v > 0 for v in on_vals) else '❌ 不满足'}"
          f"（min={min(on_vals):.6f}）")
    print()

    # ---------- §4.2 支持判据 ----------
    print("## 三、§4.2 支持判据（各自独立判阴）")
    sup = [("S1", "l1_visited_patch_frac", "up"), ("S2", "patch_sat_mean", "down"),
           ("S4", "l2_visited_gini", "report")]
    sup_res = {}
    for name, key, want in sup:
        print(f"\n  **{name}** `{key}`（期望 {'on > off' if want=='up' else 'on < off' if want=='down' else '仅报数值'}）")
        ws = []
        for s in seeds:
            ro, rn = data.get((s, "off")), data.get((s, "on"))
            if not ro or not rn:
                continue
            vo, vn = tail_mean(ro, key), tail_mean(rn, key)
            if want == "up":
                w = vn > vo
            elif want == "down":
                w = vn < vo
            else:
                w = None
            if w is not None:
                ws.append(w)
            d = vn - vo
            arrow = "" if w is None else ("  ✓" if w else "  ✗")
            print(f"    seed{s}: off={vo:>12.5f}  on={vn:>12.5f}  Δ={d:>+12.5f}{arrow}")
        if ws:
            sup_res[name] = (sum(ws), len(ws))
            print(f"    ⇒ {sum(ws)}/{len(ws)}")

    # S3 承载力 / 止损红线
    print(f"\n  **S3** `pop`（止损红线：on 降幅 >50% 才判负）")
    ratios = []
    for s in seeds:
        ro, rn = data.get((s, "off")), data.get((s, "on"))
        if not ro or not rn:
            continue
        vo, vn = tail_mean(ro, "pop"), tail_mean(rn, "pop")
        r = vn / vo if vo else float("nan")
        ratios.append((s, vo, vn, r))
        flag = "🔴 击穿止损" if r < 0.5 else ""
        print(f"    seed{s}: off_tail={vo:>10.0f}  on_tail={vn:>10.0f}  K比={r:>6.3f}  {flag}")
    if ratios:
        rr = [r for _, _, _, r in ratios]
        print(f"    ⇒ K比 中位 = {sorted(rr)[len(rr)//2]:.3f}   均值 = {sum(rr)/len(rr):.3f}")
        brk = [s for s, _, _, r in ratios if r < 0.5]
        print(f"    ⇒ 击穿 50% 止损线：{brk if brk else '无✅'}")
    print()

    # ---------- §4.3 判定 ----------
    print("## 四、§4.3 判定")
    # 第四态：纪元不足
    gens_ok, tick_ok = True, True
    for (s, a), rows in data.items():
        if rows[-1].get("tick"):
            if int(rows[-1]["tick"]) < 40000:
                tick_ok = False
    print(f"  第四态检查：末 tick 全=40000 {'✅' if tick_ok else '❌'}｜代数为30 需另核（本批探针无代数列 ⇒ 按 §4.3 以tick 判）")

    diff_attr = [s for s, (_, _, same) in pair_attr.items() if not same]
    if diff_attr:
        print(f"  ⚠️ 有 {len(diff_attr)} 个 seed 双臂落在不同吸引子：{diff_attr}")
        print(f"     ⇒ 按 §4.0 这些配对**单列**，不计入主判据全票口径")

    eff_votes = n_win - sum(1 for s in diff_attr if s in detail and detail[s]["win"])
    if n_win == n_valid and n_valid > 0:
        verdict = "✅ 成立（主判据全票通过）" if not diff_attr else "⚠️ 形式全票，但有配对不同吸引子 ⇒须单列复核"
    elif n_win >= n_valid - 1 and n_win > 0:
        verdict = "🟡 弱（待复测）—— 不得计入成立"
    else:
        verdict = "❌ 不成立"

    # §4.4 反证
    if deltas:
        med_o = sorted([detail[s]["off"] for s in detail])[len(detail)//2]
        med_n = sorted([detail[s]["on"] for s in detail])[len(detail)//2]
        print(f"  §4.4 反证检查：on 中位={med_n:.6f} vs off 中位={med_o:.6f} ⇒ "
              f"{'🔴 触发「on中位 ≤ off中位」⇒ 直接判不成立' if med_n <= med_o else '未触发'}")
    print(f"\n  ## 判定结论：{verdict}")
    print()

    # ---------- 与历史对照 ----------
    print("## 五、与历史批对照（避免重复实验/ 交叉验证）")
    print("  历史 6 批 × 12 配对（末段20行口径，R334）：合计4/12 = 33%（随机水平）")
    print(f"  本批（新装置档、全 A 支、patches1700/pop10000、末段20行）：**{n_win}/{n_valid}**")
    print("  ⇒ 装置差异：新档 bg_low 0.0/0.0（无抽奖）｜规模 patches 1700 / pop 10000（旧批 442/2000 或 1329/10000）")
    print()

    out = {
        "n_win": n_win, "n_valid": n_valid,
        "verdict": verdict,
        "deltas": {str(s): detail[s]["d"] for s in detail},
        "attractor": {f"{s}_{a}": at for (s, a), at in attr.items()},
        "diff_attr": diff_attr,
        "support": sup_res,
        "K_ratio_median": sorted([r for _, _, _, r in ratios])[len(ratios)//2] if ratios else None,
    }
    with open(os.path.join(RES, "_judge_s2g2_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"机器可读结果已写：{os.path.join(RES, '_judge_s2g2_result.json')}")


if __name__ == "__main__":
    main()