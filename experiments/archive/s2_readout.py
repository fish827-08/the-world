"""S2 三判据读判（R225 §三 预注册）。从 `s2_depletion.csv` 计算：

① 代价   ：K 相对 S1 基线（K≈915 @ rgm=1.195）的下降幅度（≤30% 为通过线）
② 信息价值（核心）：on 臂「斑块级饱和度空间方差」显著 > off 臂
③ 消费端 ：饿死占比是否上升（S1 基线 0.22–0.30 往上）

用法：python3 -m experiments.s2_readout --csv results/s2_depletion.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys

# R98 纪律：本脚本 print 含 GBK 不可编码字符（⇒），中文 Windows GBK 控制台会抛
#   UnicodeEncodeError ⇒ rc=1 假失败（外围污染批次状态）。入口做 UTF-8 兜底。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# R225 §三 预注册线
# 判据① 参照：**同批 off 臂**（配对，最干净）。S1 绝对基线随 patches/ticks 而变
#   （锚定 K=673 @ patches=120/20k t；对齐检查 K=763 @ patches=480/6k t），
#   ⇒ 不用硬编码常数，改由本批 off 臂实测给出；此处仅留作 cross-check 参考。
K_BASELINE_S1_REF = None       # 见 s2_align_check / s2_anchor 记录（非固定值）
CRITERION1_DROP_LINE = 0.30    # K 下降 ≤ 30% 为「代价可接受」
STARV_BASE_LO, STARV_BASE_HI = 0.22, 0.30   # S1 饿死占比基线带（判据③ 参照）


def _load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def _f(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return None


def _tail(rows, key, frac=0.1):
    n = max(1, int(len(rows) * frac))
    vals = [v for v in (_f(r, key) for r in rows[-n:]) if v is not None]
    return st.mean(vals) if vals else None


def _readout_layers(rows, seeds):
    """R241 四层判读（L0 轮休 / L1 探索 / L2 分化 / L3 人均摄入）。

    预注册规则（R241 §一）：
      L0：on 臂必须出现耗竭/休耕事件（`l0_any_dead`/`l0_any_rest` > 0）。
          **若恒 0 ⇒ rd 没在轮休 ⇒ 判"rd 未轮休"，不判效应**（先调参）。
      L1：on 臂首访中位数**更早** / 覆盖斑块比例**更高**（≥2/3 seed）。
      L2：on 臂访问掩码内方差/基尼 **>** off（≥2/3 seed）。
      L3：**无方向预设**，成本/收益双读（on<off ⇒ 探索有成本；on≥off ⇒ 探索没挨饿）。
      主门 = 门A（L0 + L2）× 门B（L1）联合成立。
    """
    by = {}
    for r in rows:
        by.setdefault((r["seed"], r["arm"]), []).append(r)

    # ---------- L0：rd 轮休是否发生（只看 on 臂） ----------
    l0 = []
    for sd in seeds:
        on = by.get((sd, "on"), [])
        if not on:
            continue
        last = on[-1]
        l0.append({
            "seed": sd,
            "peak_dead": _f(last, "l0_peak_dead"),
            "peak_rest": _f(last, "l0_peak_rest"),
            "peak_demoted": _f(last, "l0_peak_demoted"),
            "any_dead_ticks": _f(last, "l0_any_dead"),
            "any_rest_ticks": _f(last, "l0_any_rest"),
            "any_demoted_ticks": _f(last, "l0_any_demoted"),
        })
    l0_active = all((x["any_dead_ticks"] or 0) > 0 or (x["any_rest_ticks"] or 0) > 0
                    for x in l0) if l0 else None
    res_l0 = {
        "line": "on 臂必须出现耗竭/休耕事件（>0）；恒 0 ⇒ 判『rd 未轮休』",
        "per_seed": l0,
        "verdict_rd_cycling": l0_active,
    }

    # ---------- L1：探索度（首访中位数更早 / 覆盖率更高） ----------
    l1 = []
    for sd in seeds:
        on = by.get((sd, "on"), [])
        off = by.get((sd, "off"), [])
        if not on or not off:
            continue
        # 用末点累计覆盖率；首访中位数取末点
        of = _f(off[-1], "l1_visited_patch_frac")
        nf = _f(on[-1], "l1_visited_patch_frac")
        om = _f(off[-1], "l1_first_visit_median")
        nm = _f(on[-1], "l1_first_visit_median")
        l1.append({
            "seed": sd,
            "off_visited_frac": of, "on_visited_frac": nf,
            "cover_on_gt_off": (nf > of) if (of is not None and nf is not None) else None,
            "off_first_visit_median": om, "on_first_visit_median": nm,
            "first_visit_on_earlier": (nm < om) if (om is not None and nm is not None) else None,
        })
    cov_pos = sum(1 for x in l1 if x["cover_on_gt_off"])
    ear_pos = sum(1 for x in l1 if x["first_visit_on_earlier"])
    res_l1 = {
        "line": "on 臂覆盖率更高 / 首访更早（≥2/3 seed）",
        "per_seed": l1,
        "n_seed": len(l1),
        "n_cover_on_gt_off": cov_pos,
        "n_first_visit_earlier": ear_pos,
        "verdict_majority_pass": (max(cov_pos, ear_pos) * 2 > len(l1)) if l1 else None,
    }

    # ---------- L2：分化度（访问掩码内方差/基尼 on > off） ----------
    l2 = []
    for sd in seeds:
        on = by.get((sd, "on"), [])
        off = by.get((sd, "off"), [])
        if not on or not off:
            continue
        og = _tail(off, "l2_visited_gini")
        ng = _tail(on, "l2_visited_gini")
        ov = _tail(off, "l2_visited_var")
        nv = _tail(on, "l2_visited_var")
        l2.append({
            "seed": sd,
            "off_visited_gini": og, "on_visited_gini": ng,
            "gini_on_gt_off": (ng > og) if (og is not None and ng is not None) else None,
            "gini_ratio": (ng / og) if (og and ng is not None) else None,
            "off_visited_var": ov, "on_visited_var": nv,
            "var_on_gt_off": (nv > ov) if (ov is not None and nv is not None) else None,
            "var_ratio": (nv / ov) if (ov and nv is not None) else None,
        })
    gini_pos = sum(1 for x in l2 if x["gini_on_gt_off"])
    var_pos = sum(1 for x in l2 if x["var_on_gt_off"])
    res_l2 = {
        "line": "on 臂访问掩码内方差/基尼 > off（≥2/3 seed）",
        "per_seed": l2,
        "n_seed": len(l2),
        "n_gini_on_gt_off": gini_pos,
        "n_var_on_gt_off": var_pos,
        "verdict_majority_pass": (max(gini_pos, var_pos) * 2 > len(l2)) if l2 else None,
    }

    # ---------- L3：人均摄入（无方向预设，双读） ----------
    l3 = []
    for sd in seeds:
        on = by.get((sd, "on"), [])
        off = by.get((sd, "off"), [])
        if not on or not off:
            continue
        oi = _f(off[-1], "l3_intake_per_capita")
        ni = _f(on[-1], "l3_intake_per_capita")
        onet = _f(off[-1], "l3_net_per_capita")
        nnet = _f(on[-1], "l3_net_per_capita")
        l3.append({
            "seed": sd,
            "off_intake_per_capita": oi, "on_intake_per_capita": ni,
            "intake_ratio_on_off": (ni / oi) if (oi and ni is not None) else None,
            "off_net_per_capita": onet, "on_net_per_capita": nnet,
            "intake_on_lt_off": (ni < oi) if (oi is not None and ni is not None) else None,
        })
    lo = sum(1 for x in l3 if x["intake_on_lt_off"])
    res_l3 = {
        "line": "无方向预设：on<off ⇒ 探索有成本；on≥off ⇒ 探索没挨饿",
        "per_seed": l3,
        "n_seed": len(l3),
        "n_intake_on_lt_off": lo,
        "mean_intake_ratio_on_off": (
            st.mean([x["intake_ratio_on_off"] for x in l3
                     if x["intake_ratio_on_off"] is not None])
            if any(x["intake_ratio_on_off"] is not None for x in l3) else None),
    }

    # ---------- 主门：门A（L0 × L2）× 门B（L1） ----------
    res_main = {
        "gate_A_rd_cycling": l0_active,
        "gate_A_L2_differentiation": res_l2["verdict_majority_pass"],
        "gate_B_L1_exploration": res_l1["verdict_majority_pass"],
        "verdict_main_gate_pass": None,
    }
    if l0_active is False:
        res_main["verdict_main_gate_pass"] = False
        res_main["note"] = "L0 恒 0 ⇒ rd 未轮休，按 R241 §一 不判效应（先调参）"
    else:
        res_main["verdict_main_gate_pass"] = bool(
            l0_active and res_l2["verdict_majority_pass"] and res_l1["verdict_majority_pass"])
        res_main["note"] = "门A(L0 轮休 ∧ L2 分化) × 门B(L1 探索) 联合成立才通过"

    return {"L0_rd_cycling": res_l0, "L1_exploration": res_l1,
            "L2_differentiation": res_l2, "L3_intake": res_l3,
            "main_gate": res_main}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/s2_depletion.csv")
    ap.add_argument("--out", default=None, help="JSON 输出路径（默认同年月 .readout.json）")
    a = ap.parse_args()

    rows = _load(a.csv)
    by = {}
    for r in rows:
        by.setdefault((r["seed"], r["arm"]), []).append(r)

    seeds = sorted({k[0] for k in by})
    res = {"per_seed": {}, "criteria": {}}

    # ---- ① K 下降幅度（on vs off；并以 S1 基线为绝对参照）----
    c1 = []
    for sd in seeds:
        on = by.get((sd, "on"), [])
        off = by.get((sd, "off"), [])
        if not on or not off:
            continue
        k_on = _f(on[-1], "pop")
        k_off = _f(off[-1], "pop")
        drop_vs_off = (k_off - k_on) / k_off if k_off else None
        c1.append({"seed": sd, "K_on": k_on, "K_off": k_off,
                   "drop_vs_off": drop_vs_off,
                   "pass_le_30pct": (drop_vs_off is not None and drop_vs_off <= CRITERION1_DROP_LINE)})
    res["criteria"]["criterion_1_K_drop"] = {
        "line": f"K 下降 ≤ {int(CRITERION1_DROP_LINE*100)}%",
        "per_seed": c1,
        "verdict_all_pass": (all(x["pass_le_30pct"] for x in c1) if c1 else None),
        "mean_drop_vs_off": (st.mean([x["drop_vs_off"] for x in c1]) if c1 else None),
    }

    # ---- ② 斑块方差 on > off ----
    c2 = []
    for sd in seeds:
        on = by.get((sd, "on"), [])
        off = by.get((sd, "off"), [])
        if not on or not off:
            continue
        vo = _tail(off, "patch_sat_init_var")
        vn = _tail(on, "patch_sat_init_var")
        c2.append({"seed": sd, "off_var_tail": vo, "on_var_tail": vn,
                   "on_gt_off": (vn is not None and vo is not None and vn > vo),
                   "ratio": (round(vn / vo, 3) if vo else None)})
    n_pos = sum(1 for x in c2 if x["on_gt_off"])
    res["criteria"]["criterion_2_patch_var"] = {
        "line": "on 臂斑块方差 显著 > off 臂",
        "per_seed": c2,
        # R229 §二 / R232 口径：**按 seed 分层**，多数 seed 一致为正即通过
        #   （取代 R225 原「全 seed 通过」；原口径在 3 seed 下任一翻转即否，过严）
        "n_seed": len(c2),
        "n_seed_on_gt_off": n_pos,
        "verdict_majority_pass": (n_pos * 2 > len(c2)) if c2 else None,
        "verdict_all_pass": (all(x["on_gt_off"] for x in c2) if c2 else None),
        "mean_off_var": (st.mean([x["off_var_tail"] for x in c2]) if c2 else None),
        "mean_on_var": (st.mean([x["on_var_tail"] for x in c2]) if c2 else None),
    }

    # ---- ③ 饿死占比上升 ----
    c3 = []
    for sd in seeds:
        on = by.get((sd, "on"), [])
        off = by.get((sd, "off"), [])
        if not on or not off:
            continue
        # d_starv = 累计饿死计数（绝对）；用末值占「末值总死亡」的比例
        def starv_frac(rows_l):
            last = rows_l[-1]
            ds = _f(last, "d_starv") or 0.0
            do = _f(last, "d_old") or 0.0
            dp = _f(last, "d_pred") or 0.0
            tot = ds + do + dp
            return (ds / tot) if tot > 0 else None
        fo = starv_frac(off)
        fn = starv_frac(on)
        c3.append({"seed": sd, "off_starv_frac": fo, "on_starv_frac": fn,
                   "rise": (fn is not None and fo is not None and fn > fo),
                   "in_s1_band_off": (fo is not None and STARV_BASE_LO <= fo <= STARV_BASE_HI),
                   "in_s1_band_on": (fn is not None and STARV_BASE_LO <= fn <= STARV_BASE_HI)})
    res["criteria"]["criterion_3_starvation"] = {
        "line": f"饿死占比上升（S1 基线 {STARV_BASE_LO}–{STARV_BASE_HI}）",
        "per_seed": c3,
        "verdict_all_rise": (all(x["rise"] for x in c3) if c3 else None),
    }

    # ---- cap_lost_frac 健全性（应≈0：修复后生产格零损失）----
    cap_lost = [(_f(r, "cap_lost_frac"), r["seed"], r["arm"]) for r in rows
                if _f(r, "cap_lost_frac") is not None]
    res["sanity_cap_lost"] = {
        "max_cap_lost_frac": max((x[0] for x in cap_lost), default=None),
        "note": "修复后生产格零损失 ⇒ 应 ≈ 0；非 0 说明 rd 又在丢格",
    }

    # ================= R241 四层判读（L0–L3） =================
    # 预注册（R241 §一，板面 2026-09-28 02:1x）：
    #   主门 = 门A（L0 耗竭 + L2 分化）× 门B（L1 探索）联合成立；L3 是效应量读数
    #   判读按 R229 §二「多数 seed」口径
    #   若 L0 恒 0 ⇒ 判"rd 未轮休"，**不判效应**（先调参）
    has_l = any("l1_visited_patch_n" in r for r in rows) if rows else False
    if has_l:
        res["layers_r241"] = _readout_layers(rows, seeds)

    out = a.out or a.csv.replace(".csv", ".readout.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    # 人类可读
    print("=== S2 三判据读判 ===")
    for k, v in res["criteria"].items():
        print(f"\n-- {k} --  线：{v['line']}")
        for x in v["per_seed"]:
            print("   ", x)
        verdict = next((vv for kk, vv in v.items() if kk.startswith("verdict")), None)
        print("   ⇒ verdict:", verdict)
    print("\n-- cap_lost 健全性 --", res["sanity_cap_lost"])

    if "layers_r241" in res:
        L = res["layers_r241"]
        print("\n=== R241 四层判读（L0–L3）===")
        for key in ("L0_rd_cycling", "L1_exploration", "L2_differentiation",
                    "L3_intake"):
            v = L[key]
            print(f"\n-- {key} --  线：{v['line']}")
            for x in v["per_seed"]:
                print("   ", x)
            verdict = next((vv for kk, vv in v.items() if kk.startswith("verdict")), "（无方向预设）")
            print("   ⇒ verdict:", verdict)
        print("\n-- main_gate --")
        for k, v in L["main_gate"].items():
            print(f"    {k}: {v}")

    print(f"\n写出：{out}")


if __name__ == "__main__":
    main()
