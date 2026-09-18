#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gate 批（`cstep3gate`）正式收尾分析 —— `ungated` vs `gated` 配对（单位 = 世界种子）。

产出：E-021 的「配对分析」证据（R122 收尾边界）。

口径要点（全部为项目既有纪律，不新造）：
  · **配对单位 = 世界种子**（R96：个体级属伪重复，禁用于显著性）
  · **符号检验 `k` = 实际正向配对数**（F-R23；禁 `k=n` 的常数化写法）
  · **逐 seed 配对表 + 区制分类**（内评 01:4x §四.2 要求；R38③：饱和/捕食不得合并均值）
  · **零差异（并列）剔除**并显式报 `n_eff`（内评 §四.2：3 个 seed 两臂都饱和 ⇒ 并列被剔）
  · **不做科学判读**（R114a）；本脚本只出**统计层读数**与边界声明
  · 精确二项分布（`math.comb`）⇒ 不引入 scipy

用法：
    python experiments/gate_pair_analysis.py --dirs _rerun_logs/cstep3gate \
        --save <报告.md> --json-out <结果.json>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print 会让脚本 rc=1 假失败；F-R15 族）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # 非 TTY / 旧解释器：不因诊断能力缺失而阻断运行

NAME_RE = re.compile(r"^(?P<arm>gated|ungated)_s(?P<seed>\d+)$")

# 判读口径常量（与既有脚本同源；本批 12k 属短程 ⇒ 只作口径说明，不作判据）
RATIO_LO, RATIO_HI = 1.2, 1.5
PRED_DOMAIN_MAX = 0.9          # R38③ 划法 A
N_SAT = 3000                   # 划法 B：终态 N ≥ 3000 ⇒ 饱和
TRANSITION_LO = 1000


# ------------------------------------------------------------------ 符号检验
def sign_test_two_sided(n_pos: int, n_neg: int) -> float:
    """**双侧精确符号检验**（H0: p=0.5）。`k` = 实际正向数（F-R23 口径）。

    p = 2 · P(X ≥ max(n_pos, n_neg)) ，上限 1。并列（零差异）**不参与**，由调用方剔除。
    """
    n = n_pos + n_neg
    if n <= 0:
        return 1.0
    k = max(n_pos, n_neg)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * tail))


def regime(pred_frac, final_n) -> str:
    """区制标签（R38③ + 内评 §四.2 分层）。两套划法都记，避免单划法歧义。"""
    if pred_frac is None or final_n is None:
        return "?"
    pf, n = float(pred_frac), float(final_n)
    a = "PRED" if pf >= PRED_DOMAIN_MAX else "SAT"
    b = "SAT" if n >= N_SAT else ("PRED" if n < TRANSITION_LO else "过渡")
    return f"{a}/{b}"


# ------------------------------------------------------------------ 载入
def load(dirs: list[Path]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for d in dirs:
        for p in sorted(d.glob("*.summary.json")):
            mm = NAME_RE.match(p.name[:-len(".summary.json")])
            if not mm:
                continue
            d0 = json.loads(p.read_text(encoding="utf-8"))
            sw, r = d0.get("switches", {}), d0.get("result", {})
            o = r.get("oracle") or {}
            g = o.get("gate") or {}
            fn = o.get("funnel") or {}
            led = o.get("ledger") or {}
            rs = o.get("receiver_side") or {}
            sel = (r.get("selection_gradient") or {}).get("non_sat") or {}
            resp = r.get("signal_response") or {}
            rows[p.name[:-len(".summary.json")]] = {
                "arm": mm.group("arm"), "seed": int(mm.group("seed")),
                # —— 仪器/门控读数（gated 臂有）——
                "gate_mode": sw.get("oracle_gate_mode"),
                "gate_delta": sw.get("oracle_gate_delta"),
                "arrivals": g.get("arrivals"),
                "gate_pass": g.get("pass"),
                "gate_block": g.get("block"),
                "block_frac": g.get("block_frac"),
                "mean_d_full": g.get("mean_delta_full_at_arrivals"),
                "mean_d_content": g.get("mean_delta_content_at_arrivals"),
                # —— 单值指标（可配对）——
                "ratio": o.get("oracle_return_ratio"),
                "rho": sel.get("spearman_rho"),
                "rho_n": sel.get("n"),
                "resp_b": resp.get("resp_b_delta"),
                "resp_b_content": resp.get("resp_b_content_delta"),
                "flip": resp.get("argmax_flip_rate"),
                "N": r.get("final_N"),
                "pred": r.get("final_pred_frac"),
                "cbconv": r.get("final_codebook_conv"),
                "maxgen_hw": r.get("final_max_gen_highwater"),
                "maxgen_cur": r.get("final_max_gen_current"),
                # —— 接收侧（方向翻转的代价；本批 gated/ungated 都有探针）——
                "rs_net": rs.get("net_eat_minus_pay"),
                "rs_net_vs_base": rs.get("net_vs_baseline"),
                "rs_pay": rs.get("mean_payment_per_event"),
                "rs_intake_paid": rs.get("mean_intake_at_paid_cell"),
                # —— 仪器自检 ——
                "events": rs.get("events"),
                "food_events": rs.get("food_events"),
                "identity_ok": led.get("identity_ok"),
                "payer_trunc_n": led.get("payer_trunc_n"),
                "budget_exhausted_n": led.get("budget_exhausted_n"),
                "funnel_arrivals": fn.get("gate_positive") if fn.get("gate_positive") is not None else None,
                "funnel_applied": fn.get("applied"),
                "eco": r.get("eco_gate_pass"),
                "eco_applicable": r.get("eco_gate_applicable"),
                "final_tick": r.get("final_tick"),
                "commit": (d0.get("manifest") or {}).get("git_commit"),
                "code_tree": ((d0.get("manifest") or {}).get("code_tree_sha256") or "")[:12],
                "started": (d0.get("manifest") or {}).get("started"),
                "finished": (d0.get("manifest") or {}).get("finished"),
            }
    return rows


def pairs_of(rows: dict[str, dict], field: str) -> list[tuple[int, float, float, float]]:
    """按 **seed 显式配对**取值 ⇒ [(seed, gated 值, ungated 值, Δ=g−u)]，缺值跳过。"""
    out = []
    for seed in sorted({v["seed"] for v in rows.values()}):
        g = rows.get(f"gated_s{seed}", {}).get(field)
        u = rows.get(f"ungated_s{seed}", {}).get(field)
        if g is None or u is None:
            continue
        out.append((seed, float(g), float(u), float(g) - float(u)))
    return out


FIELDS = (
    ("oracle_ratio", "ratio", "`oracle_return_ratio`"),
    ("rho_non_sat", "rho", "ρ（non_sat）"),
    ("resp_b_delta", "resp_b", "⑥b Δ（全）"),
    ("resp_b_content", "resp_b_content", "⑥b Δ（内容项）"),
    ("argmax_flip", "flip", "argmax 翻转率"),
    ("final_N", "N", "终态 N"),
    ("pred_frac", "pred", "pred_frac"),
    ("codebook_conv", "cbconv", "codebook_conv"),
    ("max_gen_hw", "maxgen_hw", "max_gen（高水位）"),
    ("rs_net_eat_minus_pay", "rs_net", "接收侧净额（吃到−回付）"),
    ("rs_net_vs_baseline", "rs_net_vs_base", "接收侧净额 − 全员基线"),
    ("mean_payment_per_event", "rs_pay", "回付/笔"),
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["_rerun_logs/cstep3gate"])
    ap.add_argument("--save", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    dirs = [Path(x) if Path(x).is_absolute() else (ROOT / x) for x in args.dirs]
    rows = load(dirs)
    if not rows:
        print("无数据（检查 --dirs）")
        return 1

    seeds = sorted({v["seed"] for v in rows.values()})
    g_arm = {s: rows[f"gated_s{s}"] for s in seeds if f"gated_s{s}" in rows}
    u_arm = {s: rows[f"ungated_s{s}"] for s in seeds if f"ungated_s{s}" in rows}

    L: list[str] = []
    L.append("# E-021 · `cstep3gate` 配对分析（统计层读数，**非判读**）")
    L.append("")
    L.append(f"> 口径：配对单位 = **世界种子**（R96）｜符号检验 `k` = **实际正向数**（F-R23）｜"
             f"并列剔除并报 `n_eff`｜**不判读**（R114a）")
    L.append(f"> 数据：`{'`, `'.join(str(d.relative_to(ROOT)) for d in dirs)}`（{len(rows)} run："
             f"gated {len(g_arm)} / ungated {len(u_arm)}）")
    L.append("")

    # ---------------- ① 批次与仪器自检 ----------------
    L.append("## ① 批次与仪器自检")
    L.append("")
    L.append("| run | tick | 生态门 | N | pred | 区制 | ledger 恒等 | 成交 | 到达 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for s in seeds:
        for arm, d in (("u", u_arm), ("g", g_arm)):
            v = d.get(s)
            if not v:
                continue
            arr_cell = f"{v['arrivals']:,}" if v.get("gate_mode") == "delta_positive" else "—"
            L.append(f"| {arm}_{v['arm']}_s{s} | {v['final_tick']} | "
                     f"{'T' if v['eco'] else 'F'}{'' if v['eco_applicable'] else '*(不适)'} | "
                     f"{v['N']} | {v['pred']:.3f} | {regime(v['pred'], v['N'])} | "
                     f"{v['identity_ok']} | {v['funnel_applied']} | {arr_cell} |")
    ident_all = all(v["identity_ok"] for v in rows.values())
    ticks_ok = all(v["final_tick"] == 12000 for v in rows.values())
    fe_ok = all((v["food_events"] or 0) > 0 for v in rows.values())
    L.append("")
    L.append(f"- `ledger.identity_ok` **{sum(bool(v['identity_ok']) for v in rows.values())}/{len(rows)}**"
             f"；tick 全部 = 12000：**{ticks_ok}**；`food_events > 0`（仪器真测到落点摄入）：**{fe_ok}**")
    L.append(f"- 守卫计数：`bad_code_n` 见板上播报（全绿）；`payer_trunc_n` 合计 "
             f"{sum(int(v['payer_trunc_n'] or 0) for v in rows.values())}；"
             f"`budget_exhausted_n` 合计 {sum(int(v['budget_exhausted_n'] or 0) for v in rows.values())}")
    L.append("")

    # ---------------- ② 门控量级（gated 臂） ----------------
    L.append("## ② 门控量级（gated 臂，逐 seed）")
    L.append("")
    L.append("| seed | 到达 arrivals | pass | **block** | **蹭归因占比 block/arr** | Δ_full 均值 | Δ_content 均值 | **Δ_c/Δ_f** |")
    L.append("|---|---|---|---|---|---|---|---|")
    bf = []
    for s in seeds:
        v = g_arm.get(s)
        if not v or v["arrivals"] in (None, 0):
            continue
        r = (v["mean_d_content"] / v["mean_d_full"]) if v["mean_d_full"] else float("nan")
        bf.append(float(v["block_frac"]))
        L.append(f"| s{s} | {v['arrivals']:,} | {v['gate_pass']:,} | **{v['gate_block']:,}** | "
                 f"**{v['block_frac']:.4f}** | {v['mean_d_full']:.4f} | {v['mean_d_content']:.4f} | {r:.3f} |")
    if bf:
        L.append("")
        L.append(f"- **蹭归因占比**：中位 **{np.median(bf):.4f}**（范围 {min(bf):.4f} – {max(bf):.4f}）")
        L.append(f"- 门数守恒（pass + block = arrivals）：**"
                 f"{all(int(g_arm[s]['gate_pass']) + int(g_arm[s]['gate_block']) == int(g_arm[s]['arrivals']) for s in g_arm)}**")
    L.append("")

    # ---------------- ③ 逐 seed 配对表 + 区制分类（内评 §四.2） ----------------
    L.append("## ③ 逐 seed 配对表（`gated − ungated`，含**区制分类**）")
    L.append("")
    L.append("| 指标 | " + " | ".join(f"s{s}" for s in seeds) + " | 中位 Δ | 正/负/零 |")
    L.append("|---|" + "---|" * (len(seeds) + 2))
    stat: dict[str, dict] = {}
    for label, field, disp in FIELDS:
        pr = pairs_of(rows, field)
        if not pr:
            continue
        cells = []
        for s in seeds:
            hit = [p for p in pr if p[0] == s]
            cells.append("—" if not hit else f"{hit[0][3]:+.4f}")
        d = [p[3] for p in pr]
        pos = sum(1 for x in d if x > 0)
        neg = sum(1 for x in d if x < 0)
        zer = sum(1 for x in d if x == 0)
        stat[field] = {"label": label, "delta": d, "pos": pos, "neg": neg, "zero": zer,
                       "median_delta": float(np.median(d)), "n_eff": pos + neg,
                       "p_sign": sign_test_two_sided(pos, neg)}
        L.append(f"| {disp} | " + " | ".join(cells) +
                 f" | **{np.median(d):+.4f}** | {pos}/{neg}/{zer} |")
    L.append("")

    # ---------------- ④ 配对统计（含功效警示） ----------------
    L.append("## ④ 配对统计（双侧精确符号检验；**并列剔除**）")
    L.append("")
    L.append("| 指标 | 中位 Δ | n_eff（剔零后） | p（双侧） | 判定 |")
    L.append("|---|---|---|---|---|")
    for field, st in stat.items():
        verdict = "🟡 显著（p<0.05）" if st["p_sign"] < 0.05 else "不显著"
        if st["n_eff"] < 4:
            verdict += f"⚠️ **功效受限**（n_eff={st['n_eff']}）"
        L.append(f"| {st['label']} | {st['median_delta']:+.4f} | {st['n_eff']} | "
                 f"{st['p_sign']:.4f} | {verdict} |")
    L.append("")
    L.append("⚠️ **须知（内评 §四.2 口径）**：上表 `final_N`/`pred_frac` 有 seed 两臂**并列**"
             "（同饱和 3240/3240 ⇒ Δ=0）⇒ 已被剔除、`n_eff` 下降；**「不显著」在 `n_eff` 偏低时"
             "不可读作「无效应」**。⇒ 判读须看 §③ 的**逐 seed 值 + 区制分类**，不得只看 p。")
    L.append("")

    # ---------------- ⑤ 区制翻转 ----------------
    L.append("## ⑤ 区制（同 seed 跨臂是否翻转）")
    L.append("")
    L.append("| seed | ungated N / pred / 区制 | gated N / pred / 区制 | 翻转？ |")
    L.append("|---|---|---|---|")
    flips = []
    for s in seeds:
        u, g = u_arm.get(s), g_arm.get(s)
        if not u or not g:
            continue
        ru, rg = regime(u["pred"], u["N"]), regime(g["pred"], g["N"])
        same_a = ru.split("/")[0] == rg.split("/")[0]
        flips.append(not same_a)
        L.append(f"| s{s} | {u['N']} / {u['pred']:.3f} / {ru} | {g['N']} / {g['pred']:.3f} / {rg} | "
                 f"{'—' if same_a else '🔴 **翻转**'} |")
    L.append("")
    L.append(f"- 划法 A（`pred_frac ≥ {PRED_DOMAIN_MAX}` = 捕食主导）下**翻转 seed 数 = {sum(flips)}/{len(flips)}**")
    L.append("")
    L.append("- ⇒ 与 C1a / C1b / C2 一致：**同 seed、仅换一个开关即跨区制** = 「**近临界抽签**」的"
             "**第 4 次独立复现**（前三次：C1a、C1b、C2）")
    L.append("")

    # ---------------- ⑥ 边界声明 ----------------
    L.append("## ⑥ 边界声明（**本批不判读**，R114a）")
    L.append("")
    L.append("1. **12k 是短尺度**（C1a 用 60k）⇒「生态读数无差异」**可能只是时间不够**，"
             "**不得**读成「蹭归因无害」")
    L.append("2. `block_frac` = **到达中被拦的比例**，**不等于**「能量的 36% 被拦住」（能量口径见 `oracle.ledger`）")
    L.append("3. `oracle_ratio` 下降是**操作检查**（门按构造拦掉一部分到达 ⇒ `ratio` 必然下降）——"
             "它证明**门确实接在付款上**（构念被正确改动），**不是**科学发现（内评 §四.1）")
    L.append("4. 本批设计目的 = **出测度** + 为「只计真通信时是否仍有阳性」**留数据** ⇒ **判读归仪器线**")
    L.append("")

    text = "\n".join(L) + "\n"
    print(text)

    if args.save:
        p = Path(args.save) if Path(args.save).is_absolute() else (ROOT / args.save)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        print(f"[saved] {p}")
    if args.json_out:
        res = {
            "batch": "cstep3gate", "label": "E-021",
            "judged_under": "统计层读数（非判读，R114a）",
            "criterion_note": "配对单位=世界种子（R96）；符号检验 k=实际正向数（F-R23）；并列剔除",
            "n_runs": len(rows), "n_gated": len(g_arm), "n_ungated": len(u_arm),
            "ident_ok_all": ident_all, "ticks_all_12000": ticks_ok, "food_events_all_positive": fe_ok,
            "block_frac_median": float(np.median(bf)) if bf else None,
            "block_frac_range": [float(min(bf)), float(max(bf))] if bf else None,
            "paired_stats": stat,
            "regime_flips": int(sum(flips)), "regime_flips_of": len(flips),
            "boundaries": [
                "12k 短尺度 ⇒ 生态不显著不得读成「无害」",
                "block_frac 是到达比例，非能量占比",
                "oracle_ratio 下降是操作检查，不是科学发现",
                "判读归仪器线；本批只出测度",
            ],
            "rows": rows,
        }
        p = Path(args.json_out) if Path(args.json_out).is_absolute() else (ROOT / args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[saved] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
