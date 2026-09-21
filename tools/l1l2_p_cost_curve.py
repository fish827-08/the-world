"""L2 冲刺代价指数 p 的**解析**成本占比曲线（R149-7；零机时，不跑批）。

**背景**：外鉴判定"代价对速度凸"有强支持，但**指数 2 是插值约定** ⇒ 所有者裁定
「默认 p=2（物理先验）；p 作敏感性档并预注册；**零机时先算三档下的成本占比曲线**」。

**模型**（与设计稿 §4.2 一致）：
    每次移动的成本倍率 = 1 + κ·(d^p − 1)，κ=1.0，d ∈ {1, 2}
    ⇒ d=1 恒 1；d=2 ⇒ **p=1: ×2 ｜ p=2: ×4 ｜ p=3: ×8**
    令 dash 比例 f（= 走 2 格的移动占比）⇒ 平均倍率 `mult(p, f) = 1 + f·(2^p − 1)`

**输入**（全部来自既有产物，`[实测]`）：`_rerun_logs/calib1/*.summary.json` 的
`result.energy_ledger`（逐个体分通道记账，`_ec_flush` 每 tick 结算）。
⚠️ calib1 属**旧纪元**（无能量封顶）⇒ 只用于**成本占比的量级**，不作新纪元断言。

用法：
    python.exe -X utf8 tools/l1l2_p_cost_curve.py
"""
from __future__ import annotations

import glob
import json
import statistics as st
import sys


def load_runs() -> list[dict]:
    out = []
    for p in sorted(glob.glob("_rerun_logs/calib1/*.summary.json")):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        el = (d.get("result") or {}).get("energy_ledger")
        if not el or not el.get("global"):
            continue
        g = el["global"]
        n_sum = sum(v.get("n_sum", 0) for v in (el.get("groups") or {}).values())
        out.append({
            "file": p.split("/")[-1].split("\\")[-1],
            "n_sum": n_sum,
            "move": g.get("cost_move_sum", 0.0),
            "meta": g.get("cost_meta_sum", 0.0),
            "attack": g.get("cost_attack_sum", 0.0),
            "forage": g.get("intake_forage_sum", 0.0),
            "pred": g.get("intake_pred_sum", 0.0),
            "photo": g.get("intake_photo_sum", 0.0),
        })
    return out


def main() -> int:
    runs = load_runs()
    if not runs:
        print("❌ 未找到 calib1 的 energy_ledger 产物（路径变了？）")
        return 1

    print("=== 实测基线：逐个体人均（`[实测]` calib1，旧纪元；n_sum = ΣN·tick）===")
    print(f"  {'run':<18}{'cost_move':>11}{'cost_meta':>11}{'net(不含光合)':>15}{'move/(move+meta)':>18}")
    per = []
    for r in runs:
        n = max(1, r["n_sum"])
        cm, mt = r["move"] / n, r["meta"] / n
        net = (r["forage"] + r["pred"] - cm - mt - r["attack"]) / n
        share = cm / max(1e-12, cm + mt)
        per.append((cm, mt, net, share))
        print(f"  {r['file'][:18]:<18}{cm:>11.5f}{mt:>11.5f}{net:>15.5f}{share:>17.1%}")

    cm0 = st.median([x[0] for x in per])
    mt0 = st.median([x[1] for x in per])
    net0 = st.median([x[2] for x in per])
    sh0 = st.median([x[3] for x in per])
    print(f"  ⇒ 中位：cost_move={cm0:.5f}｜cost_meta={mt0:.5f}｜net={net0:.5f}｜"
          f"移动占(移动+维持) = {sh0:.1%}")

    print()
    print("=== 解析曲线：平均移动成本倍率与净收入变化（f = dash 比例）===")
    print(f"  {'f':>6}" + "".join(f"{'p=' + str(p) + ' mult':>13}{'Δnet%':>9}" for p in (1, 2, 3)))
    for f in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        row = f"  {f:>6.2f}"
        for p in (1, 2, 3):
            mult = 1.0 + f * (2 ** p - 1)
            cm = cm0 * mult
            net = net0 - (cm - cm0)
            dn = (net / net0 - 1.0) if net0 else float("nan")
            row += f"{mult:>13.3f}{dn * 100:>8.1f}%"
        print(row)

    print()
    print("=== 判读锚点 ===")
    print(f"  · 走 2 格的单次代价：p=1 ×2｜p=2 ×4｜p=3 ×8（κ=1.0）")
    print(f"  · 净收入归零的 dash 比例 f*（解析）：", end="")
    for p in (1, 2, 3):
        denom = cm0 * (2 ** p - 1)
        fstar = net0 / denom if denom else float("inf")
        print(f"p={p}: {fstar:.3f}" + ("  " if p < 3 else ""), end="")
    print("（f* >1 ⇒ 该档下「逃/追」的成本吃不满净收入）")
    print("  · ⚠️ 旧纪元数据 ⇒ 只作**量级锚**；新纪元（封顶开）须在段一冒烟里重测此表")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
