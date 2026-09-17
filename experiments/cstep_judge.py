"""C 步判读（R109 §五 判读准备）：**Q1 迁移性复核 → Q2 G-A 重判（仪器门）→ Q3/Q4 科学侧**。

设计依据
--------
- `_share/设计-R97-C步-20260916.md`（R109 §二 照准）§二 判读设计；
- **R107-v1 口径**（含内评 21:38 §三 三条加固）—— 常量与纯函数**直接 import**
  自 `experiments/r97_calibration_judge.py`（**单一真源**，禁在此重写判据）；
- R109 §4.1 落法：**仪器集与科学集分别计算**；G-A 报告段**强制打标**；科学集**绝不含校准臂**。

判读链
------
| 段 | 问 | 判据 |
|---|---|---|
| **Q1** | 8000t 锁定的 `m=1.3` 在 **60k** 是否仍落 `[1.2,1.5]`？ | R107 合取：① 配对全正（`ratio(m1.3)−ratio(m1.0)` 逐 seed）∧ ② **适用域内**全达（双划法 A/B 结论一致） |
| **Q2** | 校准好的 oracle **能否检出阳性**？（G-A 重判） | ⑤ ρ **簇级配对**（oracle_m1.3 vs zero，Fisher z + 单侧 t，n=6）CI 下界 > 0 ∧ ⑥ 同向；**前置 = Q1 通过** |
| **Q3/Q4** | 科学臂在 60k 上 ⑤/⑥ 是否可判读、ρ 是否正向？ | main vs zero 的 ⑤ 簇级（单臂 CI + 配对）；⑥ 三联；生态门率 |

⚠️ **适用域（R109 §二）**：Q1 的域 = **生态门通过**（60k ⇒ `eco_gate_applicable=True`）；
域外值只作记录（记录项），**不参与判定**。

用法
----
    python experiments/cstep_judge.py --dirs _rerun_logs/cstep1a _rerun_logs/cstep1b
    python experiments/cstep_judge.py --save <报告路径> --json-out <JSON 路径>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---- R107-v1 判据：常量与纯函数**单一真源**（禁在本文件重写）----
from experiments.r97_calibration_judge import (  # noqa: E402
    ALLOWED_STATEMENT, CRITERION_VERSION, FORBIDDEN_STATEMENTS, RATIO_HI, RATIO_LO,
    domain_coverage, n_threshold, sign_test_pvalue,
)
from observatory.statistics import (  # noqa: E402
    merge_rho_cluster, merge_rho_cluster_paired,
)

# --- R98：Windows GBK 控制台兜底 ---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ============================================================================
# **60k 判据版本**（R110 §一，2026-09-16 22:37 裁定；**C1a 完成前锁定 ⇒ 属预注册**）
# ============================================================================
# 变更（相对 `R107-v1`，**仅适用于 60k C 步；8k r97cal 的已判结果不回改**）：
#   ① Q1 验收由「域内**全部**可用 seed 落带」改为 **k-of-n**：
#      通过 ⟺ 域内可用 seed ≥ 5/6 **∧** 至多 1 个例外（域内落带数 ≥ 域内数 − 1）；
#      域内 < 5 ⇒ **「样本不足、不判」**（既非通过也非失败 ⇒ 停下上板；
#      与「值不达」**分开记录**，避免事后无法区分"迁移失败"与"域内种子太少"）。
#      理由：零容忍在 n=6 下对 1 个噪声 seed 过度敏感 ⇒ 假失败 ⇒ 错误停链、白烧机时。
#   ② 划法 A/B 在 60k + 生态门域下**高度冗余**（生态门要求 N 稳定>0 持续 50k ⇒ 达标者
#      终态 N 基本 ≥ 0.9×max_count）⇒ **「两法一致」不再作为独立护栏，降为诊断列**。
#      理由：避免"我们做了双重交叉验证"的错觉（内评 22:24 §三）。
CRITERION_VERSION_60K = ("R110-v1（60k：Q1 k-of-n〔域内 ≥5/6 ∧ 至多 1 例外〕；"
                         "两划法 60k 已退化为一 ⇒ 降诊断列）")
Q1_DOMAIN_MIN_SEEDS = 5        # R110 §一①：域内可用 seed ≥ 5/6
Q1_MAX_EXCEPTIONS = 1          # R110 §一①：至多 1 个例外

NAME_RE = re.compile(
    r"^(?P<rand>rand_)?(?P<body>oracle_m[0-9.]+|zero|main|m[0-9.]+)_s(?P<seed>\d+)$")
INSTRUMENT_MARK = "【仪器门 · 使用已登记校准臂 · 不构成科学阳性】"


# ---------------------------------------------------------------- 载入与命名

def parse_name(name: str) -> dict | None:
    """解析 C 步 run 名 → `{kind, m, is_rand, seed}`。

    支持四种写法：
    `oracle_m1.3_s42`（处理臂）/ `oracle_m1.0_s42`（配对基线）/ `zero_s42` / `main_s42`，
    以及 C2 的 `rand_m1.3_s42`（`rand_` 前缀 + m ⇒ 仍是 **oracle 类**，只是信号改随机）。
    """
    mm = NAME_RE.match(name)
    if not mm:
        return None
    body, rand = mm.group("body"), bool(mm.group("rand"))
    if body.startswith("oracle_m") or re.fullmatch(r"m[0-9.]+", body):
        kind, m = "oracle", float(body.split("m", 1)[1])
    else:
        kind, m = body, None
    return {"name": name, "kind": kind, "m": m, "is_rand": rand,
            "seed": int(mm.group("seed"))}


def load(dirs: list[Path]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for d in dirs:
        for sp in sorted(d.glob("*.summary.json")):
            name = sp.name.replace(".summary.json", "")
            meta = parse_name(name)
            if not meta:
                continue
            dd = json.loads(sp.read_text(encoding="utf-8"))
            res, sw = dd.get("result", {}), dd.get("switches", {})
            orc = res.get("oracle") or {}
            resp = res.get("signal_response") or {}
            sg = (res.get("selection_gradient") or {}).get("non_sat") or {}
            rows[name] = dict(meta,
                              dir=d.name,
                              ratio=orc.get("oracle_return_ratio"),
                              N=res.get("final_N"),
                              pred_frac=res.get("final_pred_frac"),
                              eco_gate=res.get("eco_gate_pass"),
                              eco_applicable=res.get("eco_gate_applicable"),
                              rho=sg.get("spearman_rho"),
                              resp_a=resp.get("resp_a_exposure"),
                              resp_b=resp.get("resp_b_delta"),
                              resp_tri=resp.get("resp_triple"),
                              rs=orc.get("receiver_side") or {},
                              ledger=orc.get("ledger") or {},
                              is_cal=bool(sw.get("is_calibration_arm")),
                              sw_m=sw.get("oracle_gain_multiplier"),
                              signal_mode=sw.get("signal_mode"),
                              max_count=sw.get("max_count"))
    return rows


def split_sets(rows: dict[str, dict]) -> tuple[dict, dict]:
    """**仪器集 / 科学集**（R109 §4.1）：以 `is_calibration_arm` 划分，**互斥**。

    🔴 科学集**绝不允许**含校准臂（否则校准数据会静默混进科学结论）。
    """
    inst = {k: v for k, v in rows.items() if v["is_cal"]}
    sci = {k: v for k, v in rows.items() if not v["is_cal"]}
    leak = sorted(k for k, v in sci.items() if v["is_cal"])
    assert not leak, f"R100 条件 5 违规：科学集含校准臂 {leak}"
    return inst, sci


def usable_domain(rows: dict[str, dict]) -> dict[str, bool]:
    """**适用域 = 生态门通过**（R109 §二）；`eco_applicable` 为假时不得引用该门。"""
    return {k: bool(v.get("eco_gate")) for k, v in rows.items()}


def _arm(rows: dict[str, dict], kind: str, m: float | None = None,
         rand: bool = False) -> list[dict]:
    out = [v for v in rows.values()
           if v["kind"] == kind and (m is None or v["m"] == m) and v["is_rand"] == rand]
    return sorted(out, key=lambda v: v["seed"])


def _pair(rows: dict[str, dict], a_kind: str, a_m: float | None,
          b_kind: str, b_m: float | None, field: str) -> tuple[list, list, list[int]]:
    """按 **seed 显式配对**取值（R100 条件 6）；返回 (a, b, 缺配对的 seed)。"""
    A = {v["seed"]: v for v in _arm(rows, a_kind, a_m)}
    B = {v["seed"]: v for v in _arm(rows, b_kind, b_m)}
    va, vb, missing = [], [], []
    for s in sorted(set(A) | set(B)):
        if s not in A or s not in B:
            missing.append(s)
            continue
        x, y = A[s].get(field), B[s].get(field)
        if x is None or y is None:
            missing.append(s)
            continue
        va.append(float(x)); vb.append(float(y))
    return va, vb, missing


# ---------------------------------------------------------------- Q1

def q1_verdict(rows: dict[str, dict], m: float, baseline_m: float) -> dict:
    """Q1（**60k 口径 = `R110-v1`**）：① 配对全正 ∧ ② 域内 **k-of-n**（≥5/6 ∧ 至多 1 例外）。

    **域 = seed 级生态门**（该 run 独立满足 R3 门；实现对应 `eco_gate_pass`）。
    三态输出：`pass` / `fail` / `insufficient`（域内 < 5 ⇒ **不判**，既非通过也非失败）。
    `coverage_in_domain` 的 A/B 两划法 **仅为诊断列**（60k 下二者冗余，不作独立护栏）。
    """
    # ---- ① 配对全正（R100 条件 6：各 seed 内配对；域口径不参与）----
    A = {v["seed"]: v for v in _arm(rows, "oracle", m)}
    B = {v["seed"]: v for v in _arm(rows, "oracle", baseline_m)}
    pairs, missing = [], []
    for sd in sorted(set(A) | set(B)):
        if sd not in A or sd not in B or A[sd]["ratio"] is None or B[sd]["ratio"] is None:
            missing.append(sd)
            continue
        pairs.append([sd, float(B[sd]["ratio"]), float(A[sd]["ratio"]),
                      float(A[sd]["ratio"]) - float(B[sd]["ratio"])])
    deltas = [p[3] for p in pairs]
    pair_ok = bool(pairs) and all(d > 0 for d in deltas) and not missing
    # 🔴 F-R23（2026-09-17 内评抽核，本线修复）：`k` 必须是**实际为正的配对数**，
    # 不是 `n_pairs`。原写法 ⇒ 检验退化为"全正假设"的常数（n=6 恒 0.0156），
    # 与实际几个 seed 为正无关；60k 实测是 5/6 为正 ⇒ 正确值 P(X≥5|6)=0.1094（不显著）。
    n_pos = sum(1 for d in deltas if d > 0)
    pv = sign_test_pvalue(n_pos, len(pairs)) if pairs else 1.0

    # ---- ② 域内 k-of-n（域 = seed 级生态门）----
    in_dom = [(v["seed"], float(v["ratio"])) for v in _arm(rows, "oracle", m)
              if v.get("eco_gate") and v["ratio"] is not None]
    found = [int(sd) for sd, x in in_dom if RATIO_LO <= x <= RATIO_HI]
    exceptions = [int(sd) for sd, x in in_dom if not (RATIO_LO <= x <= RATIO_HI)]
    below = [int(sd) for sd, x in in_dom if x < RATIO_LO]
    above = [int(sd) for sd, x in in_dom if x > RATIO_HI]
    n_dom = len(in_dom)
    insufficient = n_dom < Q1_DOMAIN_MIN_SEEDS
    domain_ok = (not insufficient) and len(exceptions) <= Q1_MAX_EXCEPTIONS
    passed = bool(pair_ok and domain_ok)

    # ---- 诊断列：两划法（60k 下冗余，**不作判定**）----
    diag: dict[str, dict] = {}
    for method in ("A", "B"):
        vals = [(v["seed"], float(v["ratio"])) for v in _arm(rows, "oracle", m)
                if v["ratio"] is not None and v.get("eco_gate")
                and domain_coverage({v["name"]: v}, m, method)["n"] == 1]
        inside = [sd for sd, x in vals if RATIO_LO <= x <= RATIO_HI]
        diag[method] = {"seeds": [int(sd) for sd, _ in vals],
                        "values": [round(x, 6) for _, x in vals],
                        "n": len(vals), "n_inside": len(inside),
                        "all_inside": bool(vals) and len(inside) == len(vals)}

    state = "pass" if passed else ("insufficient" if insufficient else "fail")
    reasons: list[str] = []
    if not pair_ok:
        # 失败原因分三类，**不得混写**（原写法对"无缺失但未全正"也打印「缺 []」⇒ 会被误读）
        if missing:
            reasons.append("**缺配对**的 seed %s ⇒ 不判（配对是判定前置）" % missing)
        elif not pairs:
            reasons.append("**配对样本为空** ⇒ 不判")
        else:
            _rev = [p[0] for p in pairs if p[3] <= 0]
            reasons.append("配对**未全正**（反向 seed %s；正向 %d/%d）"
                           % (_rev, sum(1 for d in deltas if d > 0), len(deltas)))
    if insufficient:
        reasons.append("域内可用 seed %d < %d ⇒ **样本不足、不判**（停下上板）"
                       % (n_dom, Q1_DOMAIN_MIN_SEEDS))
    elif len(exceptions) > Q1_MAX_EXCEPTIONS:
        reasons.append("域内例外 %d 个（> %d）：seed=%s"
                       % (len(exceptions), Q1_MAX_EXCEPTIONS, exceptions))
    return {"m": m, "baseline_m": baseline_m,
            "pairs": pairs, "deltas": [round(d, 6) for d in deltas],
            "missing_seeds": missing,
            "pair_all_positive": pair_ok, "sign_test_p": pv,
            "n_positive": n_pos, "sign_test_k_definition": "k = 正向配对数（F-R23 修复）",
            "domain": {"n_seeds": n_dom, "seeds": [int(sd) for sd, _ in in_dom],
                       "values": [round(x, 6) for _, x in in_dom],
                       "seeds_found": found, "exceptions": exceptions,
                       "seeds_below": below, "seeds_above": above,
                       "n_inside": len(found),
                       "span": (round(max(x for _, x in in_dom) - min(x for _, x in in_dom), 6)
                                if in_dom else None)},
            "min_seeds": Q1_DOMAIN_MIN_SEEDS, "max_exceptions": Q1_MAX_EXCEPTIONS,
            "coverage_in_domain": diag,
            "methods_role": "诊断列（60k 下两划法已退化为一，不作独立护栏）",
            "state": state, "pass": passed,
            "reason": reasons[0] if reasons else "配对全正 ∧ 域内 k-of-n 达标"}


# ---------------------------------------------------------------- Q2（G-A，仪器门）

def q2_ga(rows: dict[str, dict], m: float) -> dict:
    """Q2：G-A 重判 —— oracle(m) vs `zero` 的 ⑤ **簇级配对** + ⑥ 同向。**仪器门。**"""
    va, vb, missing = _pair(rows, "oracle", m, "zero", None, "rho")
    cp = merge_rho_cluster_paired(va, vb)
    ra, rb, _ = _pair(rows, "oracle", m, "zero", None, "resp_tri")
    resp_a = [v["resp_tri"] for v in _arm(rows, "oracle", m) if v["resp_tri"] is not None]
    resp_z = [v["resp_tri"] for v in _arm(rows, "zero") if v["resp_tri"] is not None]
    return {
        "marked": INSTRUMENT_MARK,
        "n_pairs": cp["n_seeds"], "missing_seeds": missing,
        "rho_paired": cp,
        "rho_oracle": [round(x, 6) for x in va],
        "rho_zero": [round(x, 6) for x in vb],
        "resp_triple_oracle_mean": round(float(np.mean(resp_a)), 6) if resp_a else None,
        "resp_triple_zero_mean": round(float(np.mean(resp_z)), 6) if resp_z else None,
        "pass": bool(cp["pass"]),
    }


# ---------------------------------------------------------------- Q3/Q4（科学侧）

def q3_science(rows: dict[str, dict]) -> dict:
    """Q3/Q4：科学臂 `main` vs `zero` 的 ⑤ 簇级（单臂 + 配对）与 ⑥；**不含校准臂**。"""
    inst, sci = split_sets(rows)
    assert all(not v["is_cal"] for v in sci.values()), "科学集混入校准臂"
    main_rho = [v["rho"] for v in _arm(sci, "main") if v["rho"] is not None]
    zero_rho = [v["rho"] for v in _arm(sci, "zero") if v["rho"] is not None]
    merged = merge_rho_cluster(main_rho)
    va, vb, missing = _pair(sci, "main", None, "zero", None, "rho")
    paired = merge_rho_cluster_paired(va, vb)
    dom = usable_domain(sci)
    return {
        "main_rho_by_seed": merged["rho_by_seed"],
        "main_cluster": merged,
        "paired_main_vs_zero": paired,
        "missing_seeds": missing,
        "n_eco_pass_main": sum(1 for v in _arm(sci, "main") if dom.get(v["name"])),
        "n_eco_pass_zero": sum(1 for v in _arm(sci, "zero") if dom.get(v["name"])),
        "resp_triple_main_mean": (round(float(np.mean(
            [v["resp_tri"] for v in _arm(sci, "main") if v["resp_tri"] is not None])), 6)
            if any(v["resp_tri"] is not None for v in _arm(sci, "main")) else None),
        "pass_rho_positive": bool(merged["pass"]),
    }


# ---------------------------------------------------------------- 报告

def _fmt(x, p: int = 4) -> str:
    return "—" if x is None else f"{x:.{p}f}"


def report(rows: dict[str, dict]) -> tuple[str, dict]:
    L: list[str] = []
    inst, sci = split_sets(rows)
    ms = sorted({v["m"] for v in inst.values() if v["kind"] == "oracle" and v["m"] != 1.0})
    res: dict = {"judged_under": "R109 §五 / R110-v1（60k 判据）",
                 "criterion_version": CRITERION_VERSION_60K,
                 "criterion_version_8k": CRITERION_VERSION,
                 "n_runs": len(rows), "n_instrument": len(inst), "n_science": len(sci),
                 "allowed_statement": ALLOWED_STATEMENT,
                 "forbidden_statements": list(FORBIDDEN_STATEMENTS)}

    L.append("=" * 104)
    L.append("C 步判读（R109 §五）：Q1 迁移性复核 → Q2 G-A 重判（仪器门）→ Q3/Q4 科学侧")
    L.append(f"口径版本：**{CRITERION_VERSION_60K}**"
             f"（预注册：C1a 完成前锁定；8k 批沿用 `{CRITERION_VERSION}`，已判结果不回改）")
    L.append(f"数据：{len(rows)} run＝仪器集 {len(inst)}（{sorted(inst)[:3]}…）+ 科学集 {len(sci)}")
    L.append(f"分组依据：`is_calibration_arm`（R109 §4.1）—— 两组**互斥**，分别判定")
    L.append("=" * 104)

    # ---- ① 数据总览 ----
    L.append("\n① 数据总览（生态门 = 适用域；域外值仅记录）")
    h = f"{'run':>18}{'ratio':>10}{'N':>7}{'pred':>8}{'eco':>6}{'rho':>10}{'⑥tri':>9}"
    L.append(h); L.append("-" * len(h))
    for k in sorted(rows):
        v = rows[k]
        L.append(f"{k:>18}{_fmt(v['ratio']):>10}{str(v['N']):>7}{_fmt(v['pred_frac']):>8}"
                 f"{str(bool(v['eco_gate'])):>6}{_fmt(v['rho'], 4):>10}"
                 f"{_fmt(v['resp_tri']):>9}")

    # ---- ② Q1 ----
    L.append(f"\n② **Q1 迁移性复核**（{CRITERION_VERSION_60K}；域 = seed 级生态门）")
    q1s = []
    for m in ms:
        v = q1_verdict(rows, m, 1.0)
        q1s.append(v)
        d = v["domain"]
        L.append(f"  m={m} vs m=1.0：配对 {len(v['pairs'])} 对"
                 f"（缺 {v['missing_seeds']}）Δ = {v['deltas']}"
                 f" ⇒ 全正 {v['pair_all_positive']}；正向 {v['n_positive']}/{len(v['pairs'])}"
                 f" ⇒ 符号检验 **p={v['sign_test_p']:.4f}**（F-R23 修后：k=正向数）")
        L.append(f"    域（seed 级生态门）：可用 {d['n_seeds']} 个 seed={d['seeds']}"
                 f"；ratio={d['values']}")
        L.append(f"    落带 {d['n_inside']}/{d['n_seeds']}；例外 seed={d['exceptions']}"
                 f"（<1.2 的 {d['seeds_below']}；>1.5 的 {d['seeds_above']}）；span={d['span']}")
        L.append(f"    **门槛（R110-v1）**：域内 ≥ {v['min_seeds']}/6 ∧ 至多 {v['max_exceptions']} 例外")
        for meth in ("A", "B"):
            c = v["coverage_in_domain"][meth]
            L.append(f"    [诊断列] 划法 {meth} 域内 seed={c['seeds']}"
                     f" ⇒ {c['n_inside']}/{c['n']} 落带（60k 下两法冗余，**不作判定**）")
        tag3 = {"pass": "✅ **Q1 通过**", "fail": "❌ **Q1 不过**",
                "insufficient": "⏸ **Q1 样本不足、不判**（停下上板）"}[v["state"]]
        L.append(f"    ⇒ {tag3}" + ("" if v["state"] == "pass" else f"（{v['reason']}）"))
    res["q1"] = q1s
    res["q1_states"] = [v["state"] for v in q1s]
    q1_pass = bool(q1s) and all(v["state"] == "pass" for v in q1s)

    # ---- ③ Q2 ----
    L.append(f"\n③ **Q2 G-A 重判** {INSTRUMENT_MARK}")
    if not q1_pass:
        L.append("  ⏸ **Q1 未过 ⇒ 按 R109 停 + 上板，不判 Q2**"
                 "（判定点不得落在中性边界，教训 18）")
        res["q2"] = {"skipped": True, "reason": "Q1 未过（R109：不判）"}
    else:
        q2 = q2_ga(rows, ms[0])
        res["q2"] = q2
        cp = q2["rho_paired"]
        L.append(f"  前置 Q1 ✅ ⇒ 判 Q2；配对 n={q2['n_pairs']}（缺 {q2['missing_seeds']}）")
        L.append(f"  ⑤ ρ：oracle {q2['rho_oracle']} vs zero {q2['rho_zero']}")
        L.append(f"    簇级配对 z 尺度：d̄={cp['dbar']} s_d={cp['s_d']} t_crit={cp['t_crit']}"
                 f" CI下界(z)={cp['ci_low_z']} ⇒ {'✅ 检出阳性' if cp['pass'] else '❌ 未检出'}")
        L.append(f"  ⑥ 三联均值：oracle={q2['resp_triple_oracle_mean']}"
                 f" vs zero={q2['resp_triple_zero_mean']}"
                 f" ⇒ {'同向' if (q2['resp_triple_oracle_mean'] or 0) > (q2['resp_triple_zero_mean'] or 0) else '未同向'}")
        L.append("  ⚠️ 本段结论**仅限仪器**：不得表述为「信号有价值」（R109 §六）。")

    # ---- ④ Q3/Q4 ----
    L.append("\n④ **Q3/Q4 科学侧**（`main` vs `zero`；**不含校准臂**）")
    q3 = q3_science(rows)
    res["q3"] = q3
    mc = q3["main_cluster"]
    L.append(f"  main 逐 seed ρ = {q3['main_rho_by_seed']}；z̄={mc['zbar']} s_z={mc['s_z']}"
             f" t_crit={mc['t_crit']} CI下界(z)={mc['ci_low_z']}"
             f" ⇒ {'✅ 正向' if mc['pass'] else '未现正向'}")
    pp = q3["paired_main_vs_zero"]
    L.append(f"  配对（main−zero）d̄={pp['dbar']} CI下界(z)={pp['ci_low_z']}"
             f" ⇒ {'✅ main 显著高于 zero' if pp['pass'] else '未显著'}")
    L.append(f"  生态门：main {q3['n_eco_pass_main']}/6、zero {q3['n_eco_pass_zero']}/6")
    if q3["resp_triple_main_mean"] is not None:
        L.append(f"  ⑥ 三联（main 均值）={q3['resp_triple_main_mean']}")

    # ---- ⑤ 记录项 ----
    L.append("\n⑤ 记录项（**不参与判定**）")
    L.append("  (a) `rand` 对照（C2，若已跑）：")
    rand_rows = [v for v in rows.values() if v["is_rand"]]
    if rand_rows:
        for v in sorted(rand_rows, key=lambda x: x["seed"]):
            L.append(f"      {v['name']:>16} ratio={_fmt(v['ratio'])} N={v['N']} ρ={_fmt(v['rho'])}")
    else:
        L.append("      （未跑：`--preset cstep2rand`）")
    L.append("  (b) 域外/过渡带值：见 §① 中 eco=False 的行 ⇒ 只记适用域限制")
    L.append("  (c) 合并下分位：C 步**不引用**（混合分布禁用合并分位，教训 23-(e)）；"
             "口径见 `r97_calibration_judge.py`")

    # ---- ⑥ 边界 ----
    L.append("\n⑥ 边界声明（R109 §六 / R100 条件 2 / 加固 3）")
    L.append("  · **Q1/Q2 属仪器**，不得读成「信号有价值」；Q3/Q4 才有科学含义（仍受 L0–L4 约束）。")
    L.append("  · `rand` 同量级 ⇒ 阳性含义**仅限行为响应**，**不含 L1（信息携带）**。")
    L.append(f"  · 唯一允许表述：{ALLOWED_STATEMENT}")
    for fb in FORBIDDEN_STATEMENTS:
        L.append(f"      ✗ 不得表述为{fb}")
    L.append("  · 校准臂（`is_calibration_arm=True`）只进**仪器集**；科学集不含校准臂（机器强制）。")
    res["boundaries"] = [
        "Q1/Q2 属仪器，不得读成「信号有价值」",
        "rand 同量级 ⇒ 阳性含义仅限行为响应，不含 L1",
        f"唯一允许表述：{ALLOWED_STATEMENT}",
        "科学集不含校准臂（机器强制）",
    ]
    res["q1_pass"] = q1_pass
    return "\n".join(L), res


# ---------------------------------------------------------------- CLI

def _resolve(arg: str) -> Path:
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
    ap.add_argument("--dirs", nargs="*",
                    default=["_rerun_logs/cstep1a", "_rerun_logs/cstep1b"],
                    help="数据目录（相对路径先试主仓、再回退 the-world-data/）")
    ap.add_argument("--save", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    dirs = [_resolve(d) for d in args.dirs]
    rows = load(dirs)
    if not rows:
        print("无数据：" + "、".join(str(d) for d in dirs))
        return 1
    text, res = report(rows)
    print(text)
    res["dirs"] = [d.as_posix() for d in dirs]
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
