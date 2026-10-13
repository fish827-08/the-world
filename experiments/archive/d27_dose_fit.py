"""D-27④-B：oracle 剂量-响应拟合与**可达性检查**（R86 修订版 / R88 / R91）。

用途
----
R86 修订版要求：A 步四点剂量测量（0.674/1.0/1.5/2.0）→ B 步**拟合求交**，目标
`ratio ≥ 1.2`，并**必报拟合残差**与**可达性检查**（R88 补充）。

⚠️ 本脚本的核心结论很可能是"**无解**"：`ratio ≤ 1` 由 C-9 保本封顶**构造性**保证
（见 `simulation/oracle.py` 与 `sphere_engine.py` 的 `budget` 式），故任何
`donation` 都到不了 1.2 —— 本脚本把这件事**从推断变成实测**，并给出"最大可达
ratio"的实测值与生态门代价，供裁定者选构造级路径。

用法
----
    python experiments/d27_dose_fit.py [--dir _rerun_logs/d27] [--target 1.2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import SIGNAL_COST  # 单一真源（C5：禁止字面量副本）


# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print 会让脚本 rc=1 假失败；F-R15 族）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # 非 TTY / 旧解释器：不因诊断能力缺失而阻断运行


def load(dirp: Path) -> list[dict]:
    out = []
    for sp in sorted(dirp.glob("dose*.summary.json")):
        d = json.loads(sp.read_text(encoding="utf-8"))
        sw, res = d.get("switches", {}), d.get("result", {})
        orc = res.get("oracle") or {}
        fn = orc.get("funnel") or {}
        don = sw.get("oracle_donation")
        em = fn.get("emissions")
        ap = fn.get("applied")
        resp = res.get("signal_response") or {}
        out.append({
            "run": sp.name.replace(".summary.json", ""),
            "seed": sw.get("seed"),
            "donation": don,
            "ratio": orc.get("oracle_return_ratio"),
            "emissions": em,
            "applied": ap,
            "conversion": (ap / em) if (ap and em) else None,
            "N": res.get("final_N"),
            "eco_gate": res.get("eco_gate_pass"),
            "extinct": res.get("extinct"),
            "pred_frac": res.get("final_pred_frac"),
            "final_tick": res.get("final_tick"),
            "g15_final": res.get("final_g15"),
            # 内评 §四：对账 + 偿付约束
            "ledger": orc.get("ledger") or {},
            # 内评 §三 观察项 1：接收侧净能量效应
            "rs": orc.get("receiver_side") or {},
            # 内评 §三 观察项 2：行为侧（⑥ 响应/暴露；下降 ≠ 信号无用，是激励变了）
            "resp_a": resp.get("resp_a_exposure"),
            "resp_b": resp.get("resp_b_delta"),
            "resp_tri": resp.get("resp_triple"),
        })
    return out


def cap_of(donation: float) -> float:
    """结构性上限：C-9 使 ratio ≤ 1（与 donation 无关）；同时 ratio ≤ donation/SIGNAL_COST。"""
    return min(1.0, donation / SIGNAL_COST)


def fit_linear(xs, ys):
    A = np.vstack([np.asarray(xs), np.ones_like(np.asarray(xs, dtype=float))]).T
    coef, *_ = np.linalg.lstsq(A, np.asarray(ys, dtype=float), rcond=None)
    pred = A @ coef
    resid = np.asarray(ys, dtype=float) - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((np.asarray(ys, dtype=float) - np.mean(ys)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(coef[0]), float(coef[1]), r2, float(np.sqrt(ss_res / max(len(ys) - 2, 1)))


def fit_ratio_model(xs, ys):
    """饱和模型 ratio = min(1, k·d)（单参数 k），以最小二乘拟合（网格搜索，稳健）。"""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    best = None
    for k in np.linspace(0.01, 5.0, 5000):
        pred = np.minimum(1.0, k * xs)
        ss = float(np.sum((ys - pred) ** 2))
        if best is None or ss < best[1]:
            best = (float(k), ss, pred)
    k, ss, pred = best
    resid = ys - pred
    ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
    r2 = 1 - ss / ss_tot if ss_tot > 0 else float("nan")
    return k, r2, resid


def receiver_side_section(rows: list[dict]) -> None:
    """内评《复核-R102方向修复》§三 的**三条接收侧观察项**（方向翻转新引入的代价）。

    1. 接收侧净能量效应：`吃到（t+1 落点实测摄入） − 回付` vs `不通信者平均摄入`；
    2. 接收者是否学会规避标记格：看 ⑥ 暴露/响应（**下降 ≠ 信号无用**，是激励变了）；
    3. `ratio` 的"谁来付"分解：`Σreceipts ÷ Σpayer 扣减`（应恒 1）+ **偿付截断率**。
    """
    ds = sorted({r["donation"] for r in rows})
    print("\n" + "=" * 96)
    print("接收侧效应（内评 §三；方向翻转 F-R18 的代价——必须量化，不是「方向修错了」）")
    print("=" * 96)

    # ---- 观察项 3：对账 + 偿付约束（纯机械，最硬）----
    print("\n【观察项 3】对账（Σtransfers ≡ Σpayer ≡ Σ收款，应恒等）+ 偿付约束量化：")
    hdr = (f"{'donation':>9}{'笔数':>12}{'Σpayer':>14}{'恒等':>7}"
           f"{'偿付截断n':>10}{'截断额':>10}{'付款方破产n':>12}{'额度用尽n':>12}"
           f"{'额度截断n':>12}{'截断率':>9}")
    print(hdr)
    print("-" * len(hdr))
    for d in ds:
        sel = [r for r in rows if r["donation"] == d and r["ledger"]]
        if not sel:
            continue
        cnt = sum(int(r["ledger"].get("payer_trunc_n", 0)) for r in sel)
        amt = sum(float(r["ledger"].get("payer_trunc_amt", 0.0)) for r in sel)
        broke = sum(int(r["ledger"].get("payer_broke_n", 0)) for r in sel)
        exh = sum(int(r["ledger"].get("budget_exhausted_n", 0)) for r in sel)
        btr = sum(int(r["ledger"].get("budget_trunc_n", 0)) for r in sel)
        ev = sum(int(r["rs"].get("events", 0)) for r in sel)
        paid = sum(float(r["ledger"].get("payer_paid", 0.0)) for r in sel)
        recv = sum(float(r["ledger"].get("sender_received", 0.0)) for r in sel)
        ok = all(bool(r["ledger"].get("identity_ok")) for r in sel)
        rate = (cnt / ev) if ev else 0.0
        print(f"{d:>9.3f}{ev:>12,}{paid:>14.1f}{str(ok):>7}"
              f"{cnt:>10,}{amt:>10.3f}{broke:>12,}{exh:>12,}{btr:>12,}{rate:>9.5f}")
    print("  ⚠️ 定位说明（内评 §四 命名建议已采纳）：三账恒等**由构造保证** ⇒ 它是**对账字段**，"
          "不是守恒检验；\n     真正有信息量的是右侧约束列：`偿付截断n`（接收者付不起满额）"
          "与 `额度截断n`/`额度用尽n`（收款方 C-9 额度用尽）**谁主导**。")

    # ---- 观察项 1：接收侧净能量效应 ----
    print("\n【观察项 1】接收侧净能量效应（吃到 − 回付 vs 不通信者基线）：")
    hdr1 = (f"{'donation':>9}{'落点摄入':>10}{'回付/笔':>10}{'净(吃到−付)':>12}"
            f"{'全员平均摄入':>13}{'净 vs 基线':>12}{'判定':>10}")
    print(hdr1)
    print("-" * len(hdr1))
    for d in ds:
        sel = [r for r in rows if r["donation"] == d and r["rs"]]
        if not sel:
            continue
        food = sum(float(r["rs"].get("mean_intake_at_paid_cell", 0.0)) for r in sel) / len(sel)
        pay = sum(float(r["rs"].get("mean_payment_per_event", 0.0)) for r in sel) / len(sel)
        base = sum(float(r["rs"].get("mean_intake_all_eaters", 0.0)) for r in sel) / len(sel)
        net = food - pay
        verdict = "净正 ✅" if net > 0 and net >= base else ("净负 🔴" if net <= 0 else "正但低于基线 ⚠️")
        print(f"{d:>9.3f}{food:>10.4f}{pay:>10.4f}{net:>12.4f}{base:>13.4f}{net - base:>12.4f}{verdict:>10}")
    print("  ⚠️ 口径：落点按 **cell** 键控 ⇒ 同格多人会并入（属近似，量级判断用）；"
          "接收者可能进食前死亡。")
    print("  ⇒ **净负**意味着「移动到被标记的格子」变成**净亏** ⇒ 接收侧出现规避激励（观察项 2 的成因）。")

    # ---- 观察项 2：行为侧（⑥）----
    print("\n【观察项 2】行为侧 ⑥ 响应（若随剂量下降——**不得**读成「信号无用」）：")
    hdr2 = f"{'donation':>9}{'⑥a 暴露率':>12}{'⑥b Δ':>10}{'⑥ 三联':>10}"
    print(hdr2)
    print("-" * len(hdr2))
    for d in ds:
        sel = [r for r in rows if r["donation"] == d and r["resp_a"] is not None]
        if not sel:
            continue
        ea = float(np.mean([r["resp_a"] for r in sel]))
        eb = float(np.mean([r["resp_b"] for r in sel]))
        et = float(np.mean([r["resp_tri"] for r in sel]))
        print(f"{d:>9.3f}{ea:>12.4f}{eb:>10.4f}{et:>10.4f}")


def purity_check(dir_a: Path, dir_b: Path) -> None:
    """仪器纯度对拍：同配置两批，唯一差别 = 新增仪器 ⇒ `final_N`/`ratio` 应**逐位一致**。

    这是 F-R18 纪律（仪器类通道必须验极性/纯度）的同族检查：证明新计数**不改变轨迹与随机流**。
    """
    a = {r["run"]: r for r in load(dir_a)}
    b = {r["run"]: r for r in load(dir_b)}
    common = sorted(set(a) & set(b))
    print("\n" + "=" * 96)
    print(f"仪器纯度对拍（{dir_a.name} vs {dir_b.name}；同配置、唯一差别=新仪器）")
    print("=" * 96)
    bad = []
    for k in common:
        same_n = a[k]["N"] == b[k]["N"]
        same_r = (a[k]["ratio"] is not None and b[k]["ratio"] is not None
                  and abs(a[k]["ratio"] - b[k]["ratio"]) < 1e-12)
        mark = "✅" if (same_n and same_r) else "❌"
        if not (same_n and same_r):
            bad.append(k)
        print(f"  {mark} {k:16s} N {a[k]['N']} vs {b[k]['N']}   ratio "
              f"{a[k]['ratio']:.6f} vs {b[k]['ratio']:.6f}")
    print(f"\n  结论：{len(common) - len(bad)}/{len(common)} 逐位一致 ⇒ "
          f"{'✅ 新仪器为**纯观测**（不改轨迹/随机流）' if not bad else f'🔴 有差异：{bad}'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="_rerun_logs/d27")
    ap.add_argument("--target", type=float, default=1.2, help="R88 判定目标")
    ap.add_argument("--compare", default=None,
                    help="仪器纯度对拍：与之逐位比 final_N/ratio 的对照批目录")
    ap.add_argument("--receiver-side", action="store_true",
                    help="追加内评 §三 的三条接收侧观察项报告")
    args = ap.parse_args()
    dirp = Path(args.dir)
    if not dirp.is_absolute():
        dirp = ROOT / dirp
    rows = load(dirp)
    if not rows:
        print(f"无数据：{dirp}")
        return 1

    print("=" * 96)
    print(f"D-27④-B 剂量-响应（目标 ratio ≥ {args.target}；SIGNAL_COST={SIGNAL_COST}）")
    print("=" * 96)
    hdr = f"{'run':14s}{'d':>7}{'ratio':>9}{'cap':>7}{'ratio/cap':>10}{'转化率':>9}{'N':>7}{'灭绝':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cap = cap_of(r["donation"])
        print(f"{r['run']:14s}{r['donation']:>7.3f}{r['ratio']:>9.4f}{cap:>7.3f}"
              f"{(r['ratio'] / cap if cap else 0):>10.4f}"
              f"{(r['conversion'] or 0):>9.4f}{str(r['N']):>7}{str(r['extinct']):>6}")

    # ---- 按剂量聚合（seed 均值）----
    ds = sorted({r["donation"] for r in rows})
    xs, ys, convs = [], [], []
    print("\n按剂量聚合（3 seed 均值）：")
    for d in ds:
        sel = [r for r in rows if r["donation"] == d]
        mr = float(np.mean([r["ratio"] for r in sel]))
        mc = float(np.mean([r["conversion"] for r in sel])) if all(
            r["conversion"] is not None for r in sel) else float("nan")
        gates = [bool(r["eco_gate"]) for r in sel]
        xs.append(d); ys.append(mr); convs.append(mc)
        print(f"  donation={d:.3f}  ratio={mr:.4f}  转化率={mc:.4f}  生态门 {sum(gates)}/{len(gates)}")

    # ---- 拟合（R91：四点拟合 + 必报残差）----
    print("\n拟合（R91：必须报残差）：")
    if len(xs) >= 2:
        slope, icept, r2, rmse = fit_linear(xs, ys)
        print(f"  线性 ratio = {slope:.4f}·d + {icept:.4f}   R²={r2:.4f}  RMSE={rmse:.4f}")
        if slope > 1e-12:
            d_solve = (args.target - icept) / slope
            print(f"    ⇒ 线性外推求交 ratio={args.target}: donation ≈ {d_solve:.4f}")
    if len(xs) >= 2:
        k, r2m, resid = fit_ratio_model(xs, ys)
        print(f"  饱和 ratio = min(1, {k:.4f}·d)          R²={r2m:.4f}  残差="
              f"{[round(float(v), 4) for v in resid]}")
        print(f"    ⇒ 该模型下 ratio 的**渐近上界 = 1.0000**（< {args.target}）")

    # ---- 可达性检查（R88 补充：本步是 C 步的前置闸）----
    print("\n🔎 可达性检查（R88 补充：无解 ⇒ 暂停 C、上板请裁）：")
    observed_max = float(np.max([r["ratio"] for r in rows]))
    struct_cap = cap_of(max(xs))
    print(f"  实测最大 ratio        = {observed_max:.4f}")
    print(f"  C-9 结构性上限        = 1.0000（与 donation 无关）")
    print(f"  当前最大剂量的 cap    = {struct_cap:.4f}")
    if args.target <= 1.0:
        print(f"  ⇒ 目标 {args.target} 在可达范围内（需继续标定）")
    else:
        print(f"  ⇒ 🔴 **目标 {args.target} > 构造上限 1.0 ⇒ 在 C-2 守恒 + C-9 保本封顶下无解**")
        print(f"     ⇒ 按 R88 补充：**暂停 C、上板请裁**；构造级变更（协调红利型／增益档）"
              f"不自行实施")
    # 生态门代价（O-5 预检）
    # ⚠️ `eco_gate_pass` 的判据含 `last >= 10000` 硬编码（为 60k 批定义）⇒ **对短程批恒 False**，
    #    直接引用会把"短程"误读成"生态崩溃"。故此处**不引用该门**，改用更朴素、
    #    对短程成立的判据：灭绝数 + 终局 N 分布（并保持分层）。
    n_ext = sum(1 for r in rows if r.get("extinct"))
    ns = sorted(r["N"] for r in rows if r["N"] is not None)
    dom = [r["N"] for r in rows if (r.get("pred_frac") or 0) >= 0.9]
    non = [r["N"] for r in rows if (r.get("pred_frac") or 0) < 0.9]
    print(f"\n  生态（O-5 替代判据，短程适用）：灭绝 **{n_ext}/{len(rows)}**；"
          f"终局 N 中位 **{int(np.median(ns))}**（min {ns[0]}, max {ns[-1]}）")
    print(f"    分层：捕食主导层 N 中位 {int(np.median(dom)) if dom else '—'}"
          f"（n={len(dom)}）／ 非捕食层 N 中位 {int(np.median(non)) if non else '—'}（n={len(non)}）")
    print("    ⚠️ 本批 `eco_gate_pass` 全 False 是**短程伪象**（判据含 `last>=10000`），"
          "**非生态失败**——勿引该列下结论。")

    # ---- 内评 §三：接收侧效应（方向翻转的代价，必须量化）----
    if args.receiver_side:
        receiver_side_section(rows)

    # ---- 仪器纯度对拍（F-R18 纪律同族：新仪器必须证明不改轨迹）----
    if args.compare:
        cp = Path(args.compare)
        if not cp.is_absolute():
            cp = ROOT / cp
        purity_check(dirp, cp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
