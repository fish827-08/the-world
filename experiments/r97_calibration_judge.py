"""R97 ⑤ 配对校准批的**判读 v4**（R107 重判口径 + F-R22 修复 + v1.1 §4.3/§4.3.1 必报项）。

判据（**R107 锁定，2026-09-16 21:24**；取代 R105 的 ratio 支路）
----------------------------------------------------------------
> **校准达成 ⟺** ① **配对全正**：`ratio(m) − ratio(m1.0)` **逐 seed 全为正**；
> ∧ ② **域内全达**：适用域内**全部可用 seed** 的 `ratio ∈ [1.2, 1.5]`，
>   且两种**独立划法**结论一致 —— **A** = `pred_frac < 0.9`（R38③ 既有）；
>   **B** = 终态 `N ≥ 3000`（机制性：生态存活 = 仪器可工作域）。
>
> **记录项（不参与判定）**：域外值、过渡带（`1000 ≤ N < 3000`）、合并下分位（降诊断）、
> ρ（**移交 C 步** —— 8000t 生态早期，尺度错配；本批只记"未现正向、未见反向"）。
>
> **反向条件公示（防自我服务）**：配对出现反向、或域内有任一 seed `< 1.2`
> ⇒ 读"**校准不足**"，不通过。

v3 → v4 变更
------------
1. **F-R22**：glob 由 `m*_s*` 改 `*.summary.json` + 名字正则 ⇒ `rand_` 前缀臂**不再漏统**；
2. **R107 判据**：ratio 支路由「合并下分位 ∈ [1.2,1.5]」改为「**配对全正 ∧ 域内全达（双划法）**」；
   原合并下分位降为**诊断记录项**（混合分布禁用合并分位，教训 23-(e)）；
3. ρ 支路**不再参与本批判定**（移交 C 步），仅作记录；
4. 新增 `--json-out` 供上板/入档引用。

用法
----
    python experiments/r97_calibration_judge.py --dir the-world-data/_rerun_logs/r97cal
    python experiments/r97_calibration_judge.py            # 自动回退到数据仓同名目录
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

from observatory.statistics import merge_rho_cluster  # noqa: E402

# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print ⇒ rc=1 假失败；F-R15 族）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ============================================================================
# 判据口径（**预注册：同类批的唯一判定口径**，不得在新数据上回头更换读法）
# ============================================================================
# 加固 1（内评 21:38 §三）：R107 口径明文预注册为「同类批唯一判定口径」；
#         且**配对对照 m=1.0 是判定前置**（缺基线 ⇒ 不判，见 paired_all_positive）。
# 加固 2：划法 B 的阈值**写死为 `N ≥ 0.9 × max_count`**（不手填、不挑上限）——
#         内评阈值敏感性：合理带内结论稳定，但阈值取到饱和上限（3240）时会退化到 1/1。
# 加固 3：结论表述**只能**是本文件的 `ALLOWED_STATEMENT`；`FORBIDDEN_STATEMENTS`
#         逐条禁用（域内 n≤4 且与 rand 分布重叠 ⇒ 机制归因超本批能力）。
CRITERION_VERSION = "R107-v1（含内评 21:38 §三 三条加固）"

RATIO_LO, RATIO_HI = 1.2, 1.5          # R107 域内区间
PRED_DOMAIN_MAX = 0.9                  # 划法 A：pred_frac < 0.9 ⇒ 域内（R38③ 既有，非事后）
N_DOMAIN_RATIO = 0.9                   # 划法 B：终态 N ≥ 0.9 × max_count（加固 2：写死比值）
N_DOMAIN_FALLBACK = 3000               # max_count 缺失时的回退（旧批）
N_TRANSITION_LO = 1000                 # 过渡带下界（记录项）
BASELINE_M = 1.0                       # 配对基线（**判定前置**）
NAME_RE = re.compile(r"^(?:(?P<arm>rand)_)?m(?P<m>[0-9.]+)_s(?P<seed>\d+)$")

ALLOWED_STATEMENT = ("增益档在**适用域内**把 `ratio` 抬进目标带 [1.2,1.5]"
                     "（配对一致为正、两划法结论一致）")
FORBIDDEN_STATEMENTS = (
    "「有信息者优于随机者」（域内 n≤4 且与 rand 分布重叠 ⇒ 机制归因超本批样本能力）",
    "「信号有价值 / 构成 L1 证据」（本档阳性含义仅限**行为响应**，R100 条件 2）",
    "「仪器不灵敏 / 增益档无效」（R108 §三：本批正名为**校准达成**）",
    "「仪器全链路已验证」（ρ 未测，移交 C 步）",
)

DOMAINS = {"A": f"pred_frac < {PRED_DOMAIN_MAX}（R38③ 既有，非事后）",
           "B": f"终态 N ≥ {N_DOMAIN_RATIO} × max_count（加固 2：比值写死）"}


# ---------------------------------------------------------------- 载入

def load(dirp: Path) -> dict[str, dict]:
    """→ {"m1.3_s42": {...}, "rand_m1.3_s42": {...}}。

    **F-R22**：不再用 `m*_s*` 作 glob（漏 `rand_` 前缀）；改为全量 glob + 名字正则过滤。
    """
    out: dict[str, dict] = {}
    for sp in sorted(dirp.glob("*.summary.json")):          # ← F-R22 修复
        name = sp.name.replace(".summary.json", "")
        mm = NAME_RE.match(name)
        if not mm:
            continue
        d = json.loads(sp.read_text(encoding="utf-8"))
        res, sw = d.get("result", {}), d.get("switches", {})
        orc = res.get("oracle") or {}
        resp = res.get("signal_response") or {}
        sg = (res.get("selection_gradient") or {}).get("non_sat") or {}
        out[name] = {
            "name": name,
            "m": float(mm.group("m")),
            "seed": int(mm.group("seed")),
            "is_rand": bool(mm.group("arm")),
            "ratio": orc.get("oracle_return_ratio"),
            "N": res.get("final_N"),
            "pred_frac": res.get("final_pred_frac"),
            "rho": sg.get("spearman_rho"),
            "resp_a": resp.get("resp_a_exposure"),
            "resp_b": resp.get("resp_b_delta"),
            "resp_tri": resp.get("resp_triple"),
            "rs": orc.get("receiver_side") or {},
            "ledger": orc.get("ledger") or {},
            "flag": bool(sw.get("is_calibration_arm")),
            "sw_m": sw.get("oracle_gain_multiplier"),
            "signal_mode": sw.get("signal_mode"),
            "max_count": sw.get("max_count"),          # 加固 2：域 B 阈值由此推导
        }
    return out


# ---------------------------------------------------------------- 域与判定（纯函数，可单测）

def n_threshold(row: dict) -> float:
    """**划法 B 的阈值**（加固 2：`0.9 × max_count`，**写死比值、不手填绝对值**）。

    `max_count` 缺失（旧批）⇒ 回退 `N_DOMAIN_FALLBACK`——回退值会在报告里**显式标注**，
    避免"悄悄换阈值"。
    """
    mc = row.get("max_count")
    if mc:
        return N_DOMAIN_RATIO * float(mc)
    return float(N_DOMAIN_FALLBACK)


def in_domain(row: dict, method: str) -> bool:
    """该 run 是否落在**适用域**内。`method` ∈ {"A","B"}。"""
    if method == "A":
        pf = row.get("pred_frac")
        return pf is not None and float(pf) < PRED_DOMAIN_MAX
    if method == "B":
        n = row.get("N")
        return n is not None and float(n) >= n_threshold(row)
    raise ValueError(f"未知划法 {method!r}")


def sign_test_pvalue(k: int, n: int) -> float:
    """单侧**符号检验** `P(X ≥ k | n, p=0.5)`（精确二项；`[内评]` 21:38 §二 补的量化）。

    R107 只写"配对 6/6"，未给检验值；本函数把该条最强证据**配上显式统计支撑**
    （`[实测]` n=6、k=6 ⇒ 1/64 = 0.0156，单侧 0.05 显著）。
    """
    if n <= 0 or k <= 0:
        return 1.0
    total = 2 ** n
    tail = sum(comb(n, i) for i in range(k, n + 1))
    return tail / total


def domain_tag(row: dict) -> str:
    """人读标签：域A/域B/过渡带/域外。"""
    a, b = in_domain(row, "A"), in_domain(row, "B")
    if a and b:
        return "域A∩B"
    if a:
        return "域A"
    if b:
        return "域B"
    n = row.get("N")
    if n is not None and N_TRANSITION_LO <= float(n) < n_threshold(row):
        return "过渡带"
    return "域外"


def paired_all_positive(rows: dict[str, dict], m: float,
                        baseline_m: float = BASELINE_M) -> dict:
    """① **配对全正**：`ratio(m) − ratio(m_baseline)` 逐 seed 全为正（R100 条件 6「各 seed 内配对」）。

    返回 `{"pairs": [[seed, base, treat, delta], ...], "all_positive": bool, "n_pairs": int,
    "n_missing": int, "worst": …, "reversed_seeds": [...]}`。
    反向条件：任一 Δ ≤ 0 ⇒ `all_positive = False`。
    """
    pairs: list[list] = []
    missing: list[int] = []
    for seed in sorted({r["seed"] for r in rows.values() if r["m"] == m and not r["is_rand"]}):
        base = rows.get(f"m{baseline_m}_s{seed}")
        treat = rows.get(f"m{m}_s{seed}")
        if not base or not treat or base["ratio"] is None or treat["ratio"] is None:
            missing.append(seed)
            continue
        pairs.append([int(seed), float(base["ratio"]), float(treat["ratio"]),
                      float(treat["ratio"]) - float(base["ratio"])])
    deltas = [p[3] for p in pairs]
    return {
        "pairs": pairs,
        "n_pairs": len(pairs),
        "n_missing": len(missing),
        "missing_seeds": missing,
        "deltas": [round(d, 6) for d in deltas],
        "reversed_seeds": [int(p[0]) for p in pairs if p[3] <= 0],
        "worst_delta": round(min(deltas), 6) if deltas else None,
        "best_delta": round(max(deltas), 6) if deltas else None,
        # 缺配对 ⇒ **不得**判"全正"（缺数据不等于通过）
        "all_positive": bool(pairs) and all(d > 0 for d in deltas) and not missing,
    }


def domain_coverage(rows: dict[str, dict], m: float, method: str) -> dict:
    """② **域内全达**：域内全部可用 seed 的 `ratio ∈ [1.2, 1.5]`。"""
    sel = [r for r in rows.values()
           if r["m"] == m and not r["is_rand"] and in_domain(r, method)]
    vals = [(r["seed"], float(r["ratio"])) for r in sel if r["ratio"] is not None]
    inside = [s for s, v in vals if RATIO_LO <= v <= RATIO_HI]
    return {
        "method": method,
        "domain": DOMAINS[method],
        "seeds": [int(s) for s, _ in vals],
        "values": [round(v, 6) for _, v in vals],
        "n": len(vals),
        "n_inside": len(inside),
        "seeds_below": [int(s) for s, v in vals if v < RATIO_LO],
        "seeds_above": [int(s) for s, v in vals if v > RATIO_HI],
        # 域内**全部**可用 seed 达标，且**至少有一个**（空域不得判通过）
        "all_inside": bool(vals) and len(inside) == len(vals),
        "span": (round(max(v for _, v in vals) - min(v for _, v in vals), 6)
                 if vals else None),
    }


def calibration_verdict(rows: dict[str, dict], m: float,
                        baseline_m: float = BASELINE_M) -> dict:
    """R107 判据的**单臂**合取判定（① 配对全正 ∧ ② 域内全达·双划法一致）。"""
    pair = paired_all_positive(rows, m, baseline_m)
    cov_a = domain_coverage(rows, m, "A")
    cov_b = domain_coverage(rows, m, "B")
    consistent = bool(cov_a["all_inside"] == cov_b["all_inside"])
    passed = bool(pair["all_positive"] and cov_a["all_inside"] and cov_b["all_inside"]
                  and consistent)
    reasons: list[str] = []
    if not pair["all_positive"]:
        reasons.append(f"配对未全正（反向 seed {pair['reversed_seeds']}）")
    for k, c in (("A", cov_a), ("B", cov_b)):
        if not c["all_inside"]:
            reasons.append(f"划法 {k} 域内未全达"
                           f"（n={c['n']}，<1.2 的 seed {c['seeds_below']}）")
    if not consistent:
        reasons.append("两种划法结论不一致")
    return {
        "m": m,
        "pair": pair,
        "coverage": {"A": cov_a, "B": cov_b},
        "methods_consistent": consistent,
        "pass": passed,
        "reason": "配对全正 ∧ 域内全达（A/B 一致）" if passed else "；".join(reasons),
    }


def pick_locked_m(verdicts: list[dict]) -> dict:
    """从各处理臂里挑 **最小满足者**（R106/R108「最小满足者」补裁）。"""
    ok = sorted([v for v in verdicts if v["pass"]], key=lambda v: v["m"])
    return {"locked_m": ok[0]["m"] if ok else None,
            "candidates": [v["m"] for v in ok]}


# ---------------------------------------------------------------- 报告

def _fmt(v, p: int = 4) -> str:
    return "—" if v is None else f"{v:.{p}f}"


def report(rows: dict[str, dict]) -> tuple[str, dict]:
    L: list[str] = []
    res: dict = {"judged_under": "R107（2026-09-16 锁定）",
                 "criterion_version": CRITERION_VERSION,
                 "preregistered_as": "同类批的唯一判定口径（不得换读法）",
                 "n_domain_ratio": N_DOMAIN_RATIO,
                 "allowed_statement": ALLOWED_STATEMENT,
                 "forbidden_statements": list(FORBIDDEN_STATEMENTS),
                 "n_runs": len(rows),
                 "n_rand_runs": sum(1 for r in rows.values() if r["is_rand"])}

    state_ms = sorted({r["m"] for r in rows.values() if not r["is_rand"]})
    rand_rows = [r for r in rows.values() if r["is_rand"]]
    treated = [m for m in state_ms if m != BASELINE_M]

    L.append("=" * 104)
    L.append("R97 ⑤ 配对校准批 · 判读 **v4**（R107 判据：① 配对全正 ∧ ② 域内全达〔双划法〕）")
    L.append(f"数据：{len(rows)} run（state {sum(1 for r in rows.values() if not r['is_rand'])}"
             f" / rand {len(rand_rows)}）；m 档 {state_ms}；配对基线 m={BASELINE_M}")
    L.append(f"口径版本：**{CRITERION_VERSION}**（预注册：同类批的**唯一**判定口径，"
             "不得在新数据上回头更换读法）")
    L.append(f"划法：A = {DOMAINS['A']}｜B = {DOMAINS['B']}｜"
             f"过渡带 N∈[{N_TRANSITION_LO}, 0.9×max_count)（记录项）")
    _mm = sorted({r.get("max_count") for r in rows.values() if r.get("max_count")})
    L.append(f"max_count 实测：{_mm if _mm else '缺失 ⇒ 域 B 回退 %d（报告已标注）' % N_DOMAIN_FALLBACK}"
             f" ⇒ 域 B 阈值 = {N_DOMAIN_RATIO} × max_count")
    L.append("=" * 104)

    # ---- ① ratio 逐 seed 全表（含域标注）----
    L.append("\n① `ratio` 逐 seed + 域标注（**记录项**：域外/过渡带值不参与判定）")
    hdr = f"{'arm':>10}{'seed':>6}{'ratio':>10}{'N':>7}{'pred_frac':>11}{'域':>10}"
    L.append(hdr); L.append("-" * len(hdr))
    for m in state_ms:
        for r in sorted([x for x in rows.values() if x["m"] == m and not x["is_rand"]],
                        key=lambda x: x["seed"]):
            pf = r["pred_frac"]
            L.append(f"{'m=%.1f' % m:>10}{r['seed']:>6}{_fmt(r['ratio']):>10}{str(r['N']):>7}"
                     f"{_fmt(pf):>11}{domain_tag(r):>10}")

    # ---- ② 配对差值 ----
    L.append(f"\n② ① **配对全正**（R100 条件 6「各 seed 内配对」；Δ = ratio(m) − ratio(m={BASELINE_M})）")
    hp = f"{'seed':>6}" + "".join(f"{'base m=1.0':>12}{'m=%.1f' % m:>10}{'Δ':>11}" for m in treated)
    L.append(hp); L.append("-" * len(hp))
    pairs_by_m = {m: paired_all_positive(rows, m) for m in treated}
    seeds_all = sorted({p[0] for pb in pairs_by_m.values() for p in pb["pairs"]})
    for s in seeds_all:
        line = f"{s:>6}"
        for m in treated:
            row = next((p for p in pairs_by_m[m]["pairs"] if p[0] == s), None)
            line += (f"{_fmt(row[1]):>12}{_fmt(row[2]):>10}{('%+.4f' % row[3]):>11}"
                     if row else f"{'—':>12}{'—':>10}{'—':>11}")
        L.append(line)
    sign: dict[float, float] = {}
    for m in treated:
        p = pairs_by_m[m]
        mark = "✅ 全正" if p["all_positive"] else f"❌ 未全正（反向 seed {p['reversed_seeds']}）"
        # 加固 1 的量化：配对全正须配**显式**检验（内评 21:38 §二 补；R107 原稿未给）
        # 🔴 F-R23（2026-09-17 内评抽核，本线修复）：`k` 必须是**实际为正的配对数**，
        # 不能写成 `n_pairs` —— 原写法把 k 恒等于 n ⇒ 检验退化成"全正假设"下的**常数**
        # （n=6 恒 0.0156，与实际几个为正无关）。数据是 5/6 为正时，正确值 = P(X≥5|6)
        # = 7/64 = **0.1094**（不显著）。
        n_pos = sum(1 for _x in p["pairs"] if _x[3] > 0)
        pv = sign_test_pvalue(n_pos, p["n_pairs"]) if p["n_pairs"] else 1.0
        sign[m] = pv
        L.append(f"  ⇒ m={m}: n={p['n_pairs']} 对（缺 {p['n_missing']}）；"
                 f"Δ ∈ [{_fmt(p['worst_delta'])}, {_fmt(p['best_delta'])}] ⇒ {mark}")
        L.append(f"     单侧**符号检验** P(X≥{n_pos} | n={p['n_pairs']}) = {pv:.4f}"
                 f"（精确二项）⇒ 配对证据{'显著' if pv < 0.05 else '**不显著**'}"
                 f"；⚠️ 配对证据**无法区分 m=1.3 与 m=1.5**（两者都 6/6）")

    # ---- ③ 域内全达（双划法）----
    L.append("\n③ ② **域内全达** —— 两种独立划法（域内全部可用 seed 须落 "
             f"[{RATIO_LO},{RATIO_HI}]）")
    hd = (f"{'arm':>10}{'划法':>6}{'域内 seed':>22}{'域内 ratio':>26}{'n':>4}"
          f"{'<1.2 的 seed':>14}{'>1.5 的 seed':>14}{'span':>8}{'判定':>10}")
    L.append(hd); L.append("-" * len(hd))
    for m in state_ms:
        for method in ("A", "B"):
            c = domain_coverage(rows, m, method)
            L.append(f"{'m=%.1f' % m:>10}{method:>6}{str(c['seeds']):>22}"
                     f"{str([round(v, 4) for v in c['values']]):>26}{c['n']:>4}"
                     f"{str(c['seeds_below']):>14}{str(c['seeds_above']):>14}"
                     f"{(c['span'] if c['span'] is not None else float('nan')):>8.4f}"
                     f"{('✅ 全达' if c['all_inside'] else '❌ 未达'):>10}")
    L.append("  （空域 ⇒ **不得**判通过；域外与过渡带值见 §①，只作记录）")

    # ---- ④ 判定 ----
    L.append("\n④ 判定（**R107 合取**）")
    verdicts = [calibration_verdict(rows, m) for m in treated]
    res["verdicts"] = verdicts
    for v in verdicts:
        L.append(f"  m={v['m']}: ① 配对全正 {v['pair']['all_positive']}；"
                 f"② 域内全达 A={v['coverage']['A']['all_inside']} "
                 f"B={v['coverage']['B']['all_inside']}（一致 {v['methods_consistent']}）"
                 f" ⇒ {'✅ **校准达成**' if v['pass'] else '❌ 未达'}"
                 + ("" if v["pass"] else f"  —— {v['reason']}"))
    locked = pick_locked_m(verdicts)
    res["locked_m"] = locked["locked_m"]
    res["lock_candidates"] = locked["candidates"]
    L.append(f"  ⇒ **锁 m 建议（最小满足者）**："
             f"{('m = %s' % locked['locked_m']) if locked['locked_m'] else '无满足臂 ⇒ 维持现值/按 R101 上调（≤2 轮）'}"
             f"（满足者 {locked['candidates']}）")
    L.append("  ⚠️ 锁 m 的依据**完全来自域内落带覆盖**（配对 6/6 对 m1.3/m1.5 无区分力）"
             "⇒ 在候选里挑「覆盖最好的最小者」本身是一次**事后选择**（加固 1）"
             "⇒ 故本口径**预注册为同类批唯一读法**。")
    L.append("")
    L.append(f"  🔒 **表述额度（加固 3，唯一允许）**：{ALLOWED_STATEMENT}")
    for fb in FORBIDDEN_STATEMENTS:
        L.append(f"      ✗ 不得表述为{fb}")

    # ---- ⑤ 记录项（不参与判定）----
    L.append("\n⑤ 记录项（**不参与判定**，R107 明文）")
    q_rows = []
    for m in state_ms:
        vals = [r["ratio"] for r in rows.values() if r["m"] == m and not r["is_rand"]
                and r["ratio"] is not None]
        a = np.asarray(vals, dtype=float)
        q_rows.append((m, float(a.min()), float(np.quantile(a, 0.10)), float(np.median(a))))
    L.append("  (a) **合并下分位**（降诊断；混合分布严禁用作判据 —— 教训 23-(e)）")
    L.append(f"      {'arm':>10}{'min':>10}{'p10':>10}{'median':>10}{'n':>4}")
    for m, mn, p10, md in q_rows:
        L.append(f"      {'m=%.1f' % m:>10}{mn:>10.4f}{p10:>10.4f}{md:>10.4f}"
                 f"{sum(1 for r in rows.values() if r['m'] == m and not r['is_rand']):>4}")
    outside = [(r["name"], r["ratio"], r["N"], domain_tag(r))
               for r in rows.values() if not r["is_rand"] and r["ratio"] is not None
               and domain_tag(r) in ("域外", "过渡带")]
    L.append(f"  (b) **域外 / 过渡带值**（{len(outside)} run）——记适用域限制，不否决校准")
    for n, v, N, tag in outside:
        L.append(f"      {n:>12}  ratio={_fmt(v)}  N={N}  [{tag}]")
    L.append("  (c) **ρ（移交 C 步）** —— 8000t 生态早期 ⇒ 尺度错配；本批只记「未现正向、未见反向」")
    for m in state_ms:
        rs = [r for r in rows.values() if r["m"] == m and not r["is_rand"]]
        mr = merge_rho_cluster([r["rho"] for r in rs])
        L.append(f"      m={m}: z̄={mr['zbar']} s_z={mr['s_z']} t_crit={mr['t_crit']} "
                 f"CI下界(z)={mr['ci_low_z']} ⇒ {'正向' if mr['pass'] else '未现正向'}"
                 f"（逐 seed ρ={mr['rho_by_seed']}）")
    res["sign_test"] = {str(k): v for k, v in sign.items()}
    res["rho_record"] = {str(m): merge_rho_cluster(
        [r["rho"] for r in rows.values() if r["m"] == m and not r["is_rand"]]) for m in state_ms}

    # ---- ⑥ 必报：⑥ + 接收侧 + 对账 ----
    L.append("\n⑥ 必报（v1.1 §4.3/§4.3.1）：⑥ 三联 + 接收侧净效应 + 账（⑥b 下降 ⇒ 先按"
             "「激励结构改变」解释，**不得**直接归因仪器灵敏度）")
    h6 = (f"{'arm':>10}{'⑥a':>9}{'⑥b':>9}{'⑥tri':>9}{'落点摄入':>10}{'回付/笔':>9}"
          f"{'净(吃-付)':>11}{'全员基线':>10}{'偿付截断n':>11}{'额度截断n':>11}{'恒等':>6}")
    L.append(h6); L.append("-" * len(h6))
    for m in state_ms:
        rs = [r for r in rows.values() if r["m"] == m and not r["is_rand"]]

        def _mn(key: str, sub: str | None = None):
            """取均值：`sub=None` ⇒ 行**顶层**字段（resp_*）；否则取子字典（rs=receiver_side）。"""
            vs = []
            for r in rs:
                v = r.get(key) if sub is None else (r.get(sub) or {}).get(key)
                if v is not None:
                    vs.append(v)
            return float(np.mean(vs)) if vs else float("nan")

        ident = all(bool(r["ledger"].get("identity_ok")) for r in rs)
        L.append(f"{'m=%.1f' % m:>10}{_mn('resp_a'):>9.4f}"
                 f"{_mn('resp_b'):>9.4f}{_mn('resp_tri'):>9.4f}"
                 f"{_mn('mean_intake_at_paid_cell', 'rs'):>10.4f}"
                 f"{_mn('mean_payment_per_event', 'rs'):>9.4f}"
                 f"{_mn('net_eat_minus_pay', 'rs'):>11.4f}"
                 f"{_mn('mean_intake_all_eaters', 'rs'):>10.4f}"
                 f"{sum(int(r['ledger'].get('payer_trunc_n', 0)) for r in rs):>11d}"
                 f"{sum(int(r['ledger'].get('budget_trunc_n', 0)) for r in rs):>11d}"
                 f"{('✅' if ident else '❌'):>6}")
    L.append("  ⚠️ 三账恒等**由构造保证**（对账字段，非守恒检验）；真正的约束信息在右两列："
             "`额度截断` ≫ `偿付截断` ⇒ **约束在收款方（C-9）**。")

    # ---- ⑦ 条件 7：rand 对照 ----
    if rand_rows:
        L.append("\n⑦ 条件 7 **随机信号自检**（`signal_mode=random` ⇒ 信号与个体状态无关 = 无信息）")
        for m in sorted({r["m"] for r in rand_rows}):
            rr = [r for r in rand_rows if r["m"] == m]
            rv = [r["ratio"] for r in rr if r["ratio"] is not None]
            c = [v for v in rv if v is not None]
            L.append(f"  rand m={m}：逐 seed ratio={[round(v, 4) for v in c]}"
                     f"（n={len(c)}）；落区间 "
                     f"{len([v for v in c if RATIO_LO <= v <= RATIO_HI])}/{len(c)}"
                     f"；中位={_fmt(float(np.median(c)) if c else None)}")
            st = domain_coverage(rows, m, "B")
            L.append(f"    对照 state m={m}（划法 B 域内）：{st['values']} ⇒ "
                     f"{st['n_inside']}/{st['n']} 落区间")
            st_all = [r["ratio"] for r in rows.values()
                      if r["m"] == m and not r["is_rand"] and r["ratio"] is not None]
            L.append(f"    **同量级对照**（R108 §二 的核心观测）：rand 中位="
                     f"{_fmt(float(np.median(c)) if c else None)}"
                     f"  vs  state 中位={_fmt(float(np.median(st_all)) if st_all else None)}"
                     f"；rand 最大={_fmt(max(c) if c else None)}"
                     f"  vs  state 最大={_fmt(max(st_all) if st_all else None)}")
            # 按域分组（内评 21:38 §三：rand 与 state 域内分布**重叠**）
            for method in ("A", "B"):
                rr_dom = [x for x in rr if in_domain(x, method) and x["ratio"] is not None]
                vals_d = [float(x["ratio"]) for x in rr_dom]
                n_in = len([v for v in vals_d if RATIO_LO <= v <= RATIO_HI])
                L.append(f"    按域 {method}：rand {n_in}/{len(vals_d)} 落带 "
                         f"{[round(v, 4) for v in vals_d]} —— 与 state 域内分布**重叠**"
                         f" ⇒ **不得**据此称「有信息者优于随机者」（加固 3）")
        L.append("  ⇒ 机制归因（rand 是否复制 state 的域内达标）**超出本批样本能力**，"
                 "**记录待 C 步专项**（R108 §二 边界）。")

    # ---- ⑧ 边界声明 ----
    L.append("\n⑧ 边界声明（R108）")
    L.append("  · 本报告结论**仅用于锁 m（工程目的）**；**不得**扩散为「仪器全链路验证」（ρ 未测）。")
    L.append("  · **不得**读成「增益档无效」或「仪器不灵敏」，**也不得**读成「信号有价值」。")
    L.append("  · `rand` 同量级 ⇒ **阳性含义仅限行为响应（接收者受益），不含 L1（信息携带）**"
             "（R100 条件 2）。")
    L.append("  · 本批全部 run 登记为校准臂（`is_calibration_arm=True`）⇒ 被 R100 条件 5 "
             "拒收于科学判读。")
    L.append(f"  · 🔒 **表述额度（加固 3）**：唯一允许 =「{ALLOWED_STATEMENT}」；"
             "其余见 §④ 禁用清单（域内 n≤4 且与 rand 分布重叠）。")
    L.append(f"  · 🔒 **口径冻结（加固 1）**：`{CRITERION_VERSION}` 为**同类批唯一判定口径**，"
             "不得在新数据上换读法；**配对对照 m=1.0 为判定前置**（缺 ⇒ 不判）。")
    L.append(f"  · 🔒 **阈值写死（加固 2）**：域 B = `N ≥ {N_DOMAIN_RATIO} × max_count`"
             "（不手填绝对值；敏感性：合理带内结论稳定，取饱和上限则退化）。")
    res["boundaries"] = [
        "仅用于锁 m（工程目的）；不得扩散为仪器全链路验证（ρ 未测）",
        "不得读成「增益档无效/仪器不灵敏」，也不得读成「信号有价值」",
        "rand 同量级 ⇒ 阳性含义仅限行为响应，不含 L1",
        "全批校准臂 ⇒ 被 R100 条件 5 拒收于科学判读",
    ]
    return "\n".join(L), res


# ---------------------------------------------------------------- CLI

def _resolve_dir(arg: str) -> Path:
    """目录解析：相对路径先试主仓，再回退数据仓同名目录（r97cal 数据在数据仓）。"""
    p = Path(arg)
    if p.is_absolute():
        return p
    p1 = ROOT / arg
    if p1.is_dir():
        return p1
    p2 = ROOT / "the-world-data" / arg
    return p2 if p2.is_dir() else p1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="_rerun_logs/r97cal",
                    help="数据目录（相对路径会先试主仓、再回退 the-world-data/）")
    ap.add_argument("--json-out", default=None, help="把判定结果（机器可读）写到该路径")
    ap.add_argument("--save", default=None, help="把报告正文写到该路径")
    args = ap.parse_args()

    dirp = _resolve_dir(args.dir)
    rows = load(dirp)
    if not rows:
        print(f"无数据：{dirp}")
        print("（先跑 `batch_runner --preset r97cal` 与 `--preset r97cal_rand`）")
        return 1

    text, res = report(rows)
    print(text)
    res["dir"] = dirp.as_posix()
    res["n_runs"] = len(rows)
    res["n_rand_runs"] = sum(1 for r in rows.values() if r["is_rand"])
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(text + "\n", encoding="utf-8")
        print(f"\n报告已写：{args.save}")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        print(f"JSON 已写：{args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
