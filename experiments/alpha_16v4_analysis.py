"""α 批（`cstep3alpha`）配对分析：信号字母表 **16 vs 4**（同批、同参数、仅字母表不同）。

用途：R121 §五 的"**方向门槛**"筛选 —— 先看切档有没有**方向性**差异，再决定是否值得上 60k 确认批。
口径纪律：
- **配对单位 = 世界种子**（R96）；`k` = 实际正向配对数（F-R23 修后口径）
- 每臂的 `codebook_conv` 用**各自档的 `n_states`** 计（16 / 4），不跨档直接比大小
- ⚠️ **8k 不可判区制**（09-18 实测：SAT/PRED 在 8k 重叠、12k 才分离）⇒ 本脚本**不做区制分类**，
  只报 `N`/`pred_frac` 原值，避免"用重叠分布硬分类"。
- 纯读产物，**零机时**；不修改任何数据。
"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print 会让脚本 rc=1 假失败）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ARMS = ("a16", "a4")
SEEDS = (42, 43, 44, 45, 46, 47)


def sign_test_pvalue(k: int, n: int) -> float:
    """单侧精确符号检验 `P(X ≥ k | n, p=0.5)`（F-R23：k = **实际正向数**）。"""
    if n <= 0 or k <= 0:
        return 1.0
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def last_row(csv_path: Path) -> dict:
    """取 CSV 最后一行（dict）。"""
    txt = csv_path.read_text(encoding="utf-8").strip().splitlines()
    if len(txt) < 2:
        return {}
    head = txt[0].split(",")
    vals = txt[-1].split(",")
    return dict(zip(head, vals))


def load_one(d: Path, arm: str, seed: int) -> dict | None:
    """载入单 run：summary + CSV 末行。缺任一 ⇒ None。"""
    sp = d / f"{arm}_s{seed}.summary.json"
    cp = d / f"{arm}_s{seed}.csv"
    if not sp.exists() or not cp.exists():
        return None
    s = json.loads(sp.read_text(encoding="utf-8"))
    r, sw = s.get("result", {}), s.get("switches", {})
    row = last_row(cp)

    def f(x, default=None):
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    sel = (r.get("selection_gradient") or {}).get("non_sat", {}) or {}
    resp = r.get("signal_response") or {}
    return {
        "arm": arm, "seed": seed, "alphabet": sw.get("signal_alphabet"),
        "N": r.get("final_N"), "pred": r.get("final_pred_frac"),
        "cb_conv": r.get("final_codebook_conv"),
        "g15": f(row.get("g15")), "rho": sel.get("spearman_rho"),
        "resp_a": resp.get("resp_a_exposure"), "resp_b": resp.get("resp_b_delta"),
        "resp_tri": resp.get("resp_triple"),
        "alphabet_stats": r.get("alphabet") or {},
        "inheritance": r.get("inheritance") or {},
        "eco_gate": r.get("eco_gate_pass"), "eco_applicable": r.get("eco_gate_applicable"),
        "dir": str(sp.parent.name),
    }


def fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["_rerun_logs/cstep3alpha"])
    ap.add_argument("--save", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    dirs = [Path(x) for x in args.dirs]
    runs: dict[tuple[str, int], dict] = {}
    for d in dirs:
        for arm in ARMS:
            for sd in SEEDS:
                one = load_one(d, arm, sd)
                if one is not None:
                    runs[(arm, sd)] = one

    L: list[str] = []
    L.append("=" * 92)
    L.append("α 批配对分析：信号字母表 16 vs 4（同批、仅字母表不同）")
    L.append("口径：配对单位 = 世界种子（R96）；k = 实际正向数（F-R23）；各臂 cb_conv 用各自 n_states")
    L.append("⚠️ 8k 不做区制分类（09-18 实测：8k 重叠、12k 才分离）—— 只报 N/pred 原值")
    L.append("=" * 92)
    L.append(f"载入 {len(runs)} run（期望 {len(ARMS) * len(SEEDS)}）")

    L.append("\n① 逐 seed 原值")
    hd = (f"{'seed':>5}{'品目':>12}{'a16':>12}{'a4':>12}{'Δ(a4−a16)':>13}")
    L.append(hd)
    L.append("-" * len(hd))
    metrics = [
        ("N", lambda r: r["N"], 0),
        ("g15 终值", lambda r: r["g15"], 4),
        ("pred_frac", lambda r: r["pred"], 4),
        ("cb_conv", lambda r: r["cb_conv"], 4),
        ("rho(non_sat)", lambda r: r["rho"], 4),
        ("⑥a 暴露", lambda r: r["resp_a"], 4),
        ("⑥b Δ", lambda r: r["resp_b"], 4),
        ("⑥ 三联", lambda r: r["resp_tri"], 4),
    ]
    deltas: dict[str, list[float]] = {m[0]: [] for m in metrics}
    for sd in SEEDS:
        a, b = runs.get(("a16", sd)), runs.get(("a4", sd))
        for name, pick, nd in metrics:
            va = pick(a) if a else None
            vb = pick(b) if b else None
            dv = (vb - va) if (va is not None and vb is not None) else None
            if dv is not None:
                deltas[name].append(float(dv))
            if name == "N":
                L.append(f"{sd:>5}{name:>12}{fmt(va, nd):>12}{fmt(vb, nd):>12}{fmt(dv, nd):>13}")
            else:
                L.append(f"{'':>5}{name:>12}{fmt(va, nd):>12}{fmt(vb, nd):>12}{fmt(dv, nd):>13}")
        L.append("")

    L.append("② 配对汇总（n = 配对数；正/负/零；符号检验 k = 正向数）")
    hd2 = f"{'品目':>12}{'n':>4}{'均值Δ':>11}{'正':>4}{'负':>4}{'零':>4}{'符号检验p':>12}"
    L.append(hd2)
    L.append("-" * len(hd2))
    summary: dict[str, dict] = {}
    for name, _pick, _nd in metrics:
        ds = deltas[name]
        if not ds:
            continue
        pos = sum(1 for x in ds if x > 0)
        neg = sum(1 for x in ds if x < 0)
        zer = sum(1 for x in ds if x == 0)
        pv = sign_test_pvalue(pos, len(ds))
        summary[name] = {"n": len(ds), "mean_delta": float(np.mean(ds)),
                         "pos": pos, "neg": neg, "zero": zer, "sign_p": pv}
        L.append(f"{name:>12}{len(ds):>4}{np.mean(ds):>11.4f}{pos:>4}{neg:>4}{zer:>4}{pv:>12.4f}")

    L.append("\n③ 守卫与 counter（切档正确性 + §4 前置小改是否出数）")
    hd3 = f"{'arm':>5}{'seed':>5}{'alphabet':>10}{'n_states':>9}{'code_max':>9}{'bad_code_n':>11}{'mem_far_frac':>13}{'eco_appl':>9}"
    L.append(hd3)
    L.append("-" * len(hd3))
    guard_bad = 0
    for arm in ARMS:
        for sd in SEEDS:
            r = runs.get((arm, sd))
            if not r:
                continue
            a = r["alphabet_stats"]
            inh = r["inheritance"]
            bad = a.get("bad_code_n")
            if isinstance(bad, int) and bad > 0:
                guard_bad += 1
            L.append(f"{arm:>5}{sd:>5}{str(r['alphabet']):>10}{str(a.get('n_states')):>9}"
                     f"{str(a.get('code_max')):>9}{str(bad):>11}"
                     f"{fmt(inh.get('mem_inherit_far_frac'), 4):>13}"
                     f"{str(r['eco_applicable']):>9}")
    L.append(f"\n  守卫：`bad_code_n > 0` 的 run 数 = **{guard_bad}**（期望 0）")

    L.append("\n④ 边界声明（措辞纪律）")
    L.append("  · 本批为 **8k 短程方向筛选**，**不作任何科学判读**（R114a：数据收集≠判读）")
    L.append("  · 字母表是**纪元变更**（R121）⇒ 与 16 档旧批的 ratio/cb_conv **不可跨批比较**，只在**批内配对**")
    L.append("  · 8k 下 `N`/`pred_frac` 的区制**不可分** ⇒ 不得据此分 SAT/PRED 层")
    L.append("  · 未定：方向门槛阈值（R121 §六.3 登记为「未核实」）⇒ 本报告只出数值，不设通过线")

    text = "\n".join(L)
    print(text)
    if args.save:
        Path(args.save).write_text(text + "\n", encoding="utf-8")
        print(f"\n[已保存] {args.save}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"runs": {f"{k[0]}_s{k[1]}": v for k, v in runs.items()},
             "paired_summary": summary, "bad_code_runs": guard_bad},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[已保存] {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
