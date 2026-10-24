#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**通用两臂配对分析器**（单位 = 世界种子）—— 收尾用，**不做科学判读**（R114a）。

首次用途 = `cstep3gate`（`gated` vs `ungated`，E-021）；现已**通用化**以复用（`cstep3alpha8` = E-022 等）。
文件名保留 `gate_pair_analysis`（板上/E-021 已引用该名），但**臂名、标题、标签全部走参数**。

口径要点（项目既有纪律，不新造）：
  · **配对单位 = 世界种子**（R96：个体级属伪重复，禁用于显著性）
  · **符号检验 `k` = 实际正向配对数**（F-R23；禁 `k=n` 的常数化写法）
  · **逐 seed 配对表 + 区制分类**（R38③ 划法 A ＋ 内评 §四.2 划法 B，并列输出）
  · **零差异（并列）剔除**并显式报 `n_eff`
  · **精确二项分布**（`math.comb`）⇒ 不引入 scipy
  · 指标缺值（如非 oracle 臂无 `oracle.*`）⇒ **自动跳过**，不报错、不当 0
  · **不判读**；出**统计层读数**与**边界声明**

用法：
    python experiments/gate_pair_analysis.py --dirs <目录> [--arm-a gated --arm-b ungated]
        [--label E-021] [--save <报告.md>] [--json-out <结果.json>]
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

# 判读口径常量（显式声明，便于审阅对照）
PRED_DOMAIN_MAX = 0.9          # R38③ 划法 A
N_SAT = 3000                   # 划法 B：终态 N ≥ 3000 ⇒ 饱和
TRANSITION_LO = 1000


def make_name_re(arm_a: str, arm_b: str) -> re.Pattern:
    """按臂名构造 run 名正则（`<arm>_s<seed>`）。"""
    alts = "|".join(re.escape(x) for x in (arm_a, arm_b))
    return re.compile(rf"^(?P<arm>{alts})_s(?P<seed>\d+)$")


# ------------------------------------------------------------------ 符号检验
def sign_test_two_sided(n_pos: int, n_neg: int) -> float:
    """**双侧精确符号检验**（H0: p=0.5）。`k` = 实际正向数（F-R23 口径）。

    p = 2 · P(X ≥ max(n_pos, n_neg))，上限 1。并列（零差异）**不参与**，由调用方剔除。
    """
    n = n_pos + n_neg
    if n <= 0:
        return 1.0
    k = max(n_pos, n_neg)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * tail))


# 双侧 0.05 / 80% 功效的合并系数 K(df) = t_{0.975,df} + t_{0.80,df}（df = n−1）。
# 显式表 ⇒ **不引入 scipy 依赖**（与 sign_test 用 math.comb 同一取舍）；>120 用正态近似 2.802。
_POWER_K = {1: 14.082, 2: 5.364, 3: 4.160, 4: 3.717, 5: 3.491, 6: 3.353, 7: 3.261,
            8: 3.195, 9: 3.145, 10: 3.107, 12: 3.052, 15: 2.997, 20: 2.946,
            24: 2.921, 30: 2.896, 40: 2.872, 60: 2.848, 120: 2.825}


def power_k(n: int) -> float:
    """n 个配对对应的功效系数 `K = t₀.₉₇₅,df + t₀.₈₀,df`（df=n−1；>120 ⇒ 2.802）。"""
    df = max(1, n - 1)
    if df in _POWER_K:
        return _POWER_K[df]
    if df > 120:
        return 2.802
    return _POWER_K[min(_POWER_K, key=lambda k: abs(k - df))]


def detectable_delta(deltas) -> float:
    """**可检出 |Δ|**（双侧 0.05 / 80%）= `K(df) · σ_d / √n`（σ_d = 实测配对差样本标准差）。

    ⚠️ 口径（内评 09-19 §三 请求）：这是「**能检出的最小效应**」，**不是**「观测到的效应」
    ⇒ 只能用来把「不显著」翻译成「**排除了 |Δ| ≳ 该值的效应**」；阈值**由功效推**，
    禁"先看效应量再倒推门槛"（教训 2/9）。n<2 ⇒ `nan`（无法估计）。
    """
    n = len(deltas)
    if n < 2:
        return float("nan")
    return float(power_k(n) * np.std(deltas, ddof=1) / (n ** 0.5))


def regime(pred_frac, final_n) -> str:
    """区制标签（R38③ + 内评 §四.2 分层）。两套划法都记，避免单划法歧义。"""
    if pred_frac is None or final_n is None:
        return "?"
    pf, n = float(pred_frac), float(final_n)
    a = "PRED" if pf >= PRED_DOMAIN_MAX else "SAT"
    b = "SAT" if n >= N_SAT else ("PRED" if n < TRANSITION_LO else "过渡")
    return f"{a}/{b}"


def matched_seeds(map_a: dict, map_b: dict, seeds: list[int]) -> list[int]:
    """**区制匹配**筛选（内评 2026-09-19 §二 建议）：只保留**两臂同态**的 seed（划法 A）。

    **为什么需要**：12k 批里"同 seed 仅换一个开关"就可能跨区制（E-021 起已独立复现 6 次）
    ⇒ 未匹配的配对是「**两态相比**」，把区制差混进任何指标差 —— 这是 E-023 `p8` 那个
    "CI 下界 +0.0023 勉强过线"最可能的成因（该臂 4/6 seed 翻转）。

    ⚠️ **口径边界（必须与 §③/§④ 并列报，不得单独用）**：
      ① 本筛选是**后处理**（conditional-on-post-treatment）⇒ 只能作**稳健性检查**，
         **不替代**预注册的全配对口径；
      ② 子集 `n` 会大幅下降（翻转越多、剩得越少）⇒ **功效受限**，
         "子集里不显著"**不可**读作"无效应"（教训 2/9）；
      ③ 子集与全配对结论冲突 ⇒ 判 **不可判**，**不得**挑对自己有利的那一组。
    """
    out: list[int] = []
    for s in seeds:
        x, y = map_a.get(s), map_b.get(s)
        if not x or not y:
            continue
        if regime(x["pred"], x["N"]).split("/")[0] == regime(y["pred"], y["N"]).split("/")[0]:
            out.append(s)
    return out


# ------------------------------------------------------------------ 载入
def load(dirs: list[Path], arm_a: str, arm_b: str) -> dict[str, dict]:
    name_re = make_name_re(arm_a, arm_b)
    rows: dict[str, dict] = {}
    for d in dirs:
        for p in sorted(d.glob("*.summary.json")):
            mm = name_re.match(p.name[: -len(".summary.json")])
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
            inh = r.get("inheritance") or {}
            alp = r.get("alphabet") or {}
            rows[p.name[: -len(".summary.json")]] = {
                "arm": mm.group("arm"), "seed": int(mm.group("seed")),
                # —— 档位/守卫（新档位批次必看）——
                "alphabet": sw.get("signal_alphabet"),
                "n_states": alp.get("n_states"),
                "code_max": alp.get("code_max"),
                "bad_code_n": alp.get("bad_code_n"),
                # R128：mem_bit 取值分布（非 "8" 档为 None ⇒ **未适用**，须按 n/a 报）
                "mem_bit_frac": alp.get("mem_bit_frac"),
                "mem_bit_n": alp.get("mem_bit_n"),
                # —— 门控读数（仅 gated 类臂有）——
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
                "resp_a": resp.get("resp_a_exposure"),
                "resp_b": resp.get("resp_b_delta"),
                "resp_b_content": resp.get("resp_b_content_delta"),
                "flip": resp.get("argmax_flip_rate"),
                "N": r.get("final_N"),
                "pred": r.get("final_pred_frac"),
                "cbconv": r.get("final_codebook_conv"),
                "maxgen_hw": r.get("final_max_gen_highwater"),
                "maxgen_cur": r.get("final_max_gen_current"),
                "memfar": inh.get("mem_inherit_far_frac"),
                "meminh": inh.get("mem_inherit_n"),
                # —— 接收侧（方向翻转的代价）——
                "rs_net": rs.get("net_eat_minus_pay"),
                "rs_net_vs_base": rs.get("net_vs_baseline"),
                "rs_pay": rs.get("mean_payment_per_event"),
                "rs_intake_paid": rs.get("mean_intake_at_paid_cell"),
                # —— 仪器自检 ——
                "events": rs.get("events"),
                "food_events": rs.get("food_events"),
                "identity_ok": led.get("identity_ok"),
                "oracle_on": bool(sw.get("oracle_enabled")),
                "payer_trunc_n": led.get("payer_trunc_n"),
                "budget_exhausted_n": led.get("budget_exhausted_n"),
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


def pairs_of(rows: dict[str, dict], arm_a: str, arm_b: str,
             field: str) -> list[tuple[int, float, float, float]]:
    """按 **seed 显式配对**取值 ⇒ [(seed, a 值, b 值, Δ=a−b)]，缺值跳过。"""
    out = []
    for seed in sorted({v["seed"] for v in rows.values()}):
        x = rows.get(f"{arm_a}_s{seed}", {}).get(field)
        y = rows.get(f"{arm_b}_s{seed}", {}).get(field)
        if x is None or y is None:
            continue
        out.append((seed, float(x), float(y), float(x) - float(y)))
    return out


FIELDS = (
    ("oracle_ratio", "ratio", "`oracle_return_ratio`"),
    ("rho_non_sat", "rho", "ρ（non_sat）"),
    ("resp_a_exposure", "resp_a", "⑥a 暴露率"),
    ("resp_b_delta", "resp_b", "⑥b Δ（去两项）"),
    ("resp_b_content", "resp_b_content", "⑥b Δ（只去内容项）"),
    ("argmax_flip", "flip", "argmax 翻转率"),
    ("final_N", "N", "终态 N"),
    ("pred_frac", "pred", "pred_frac"),
    ("codebook_conv", "cbconv", "codebook_conv"),
    ("mem_bit_frac", "mem_bit_frac", "**`mem_bit_frac`**（「8」档；其余档 n/a）"),
    ("max_gen_hw", "maxgen_hw", "max_gen（高水位）"),
    ("mem_inherit_far_frac", "memfar", "记忆继承远格占比"),
    ("mem_inherit_n", "meminh", "记忆继承事件数"),
    ("rs_net_eat_minus_pay", "rs_net", "接收侧净额（吃到−回付）"),
    ("rs_net_vs_baseline", "rs_net_vs_base", "接收侧净额 − 全员基线"),
    ("mean_payment_per_event", "rs_pay", "回付/笔"),
)

# **仅 oracle 臂存在**的指标：两臂都非 oracle 时它们是**不适用**（值恒 0 ⇒ Δ 恒 0），
# 若照常报会显示成「不显著 + n_eff=0」，**容易被误读为「无效应」**（2026-09-19 修）。
# ⚠️ 集合里放的是 `FIELDS` 的**标签**（第 1 元），**不是**内部字段键（第 2 元）——
#    2026-09-19 我曾把两者搞混 ⇒ 跳过逻辑**静默失效**（报告照出误导行）；现已加测试钉死约定。
ORACLE_ONLY_LABELS = frozenset({
    "oracle_ratio", "rs_net_eat_minus_pay", "rs_net_vs_baseline", "mean_payment_per_event",
})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["_rerun_logs/cstep3gate"])
    ap.add_argument("--arm-a", default="gated", help="处理臂（Δ = a − b）")
    ap.add_argument("--arm-b", default="ungated", help="对照臂")
    ap.add_argument("--label", default="配对分析", help="报告标题前缀（如 E-021/E-022）")
    ap.add_argument("--boundaries-file", default=None,
                    help="**推荐**：从文件读批次专属边界（每行一条；空行与 `#` 开头跳过）"
                         "—— 避免 shell 引号/反引号陷阱（本项目已付学费：反引号触发命令替换）")
    ap.add_argument("--save", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    A, B = args.arm_a, args.arm_b
    dirs = [Path(x) if Path(x).is_absolute() else (ROOT / x) for x in args.dirs]
    rows = load(dirs, A, B)
    if not rows:
        print("无数据（检查 --dirs / --arm-a / --arm-b）")
        return 1

    seeds = sorted({v["seed"] for v in rows.values()})
    a_arm = {s: rows[f"{A}_s{s}"] for s in seeds if f"{A}_s{s}" in rows}
    b_arm = {s: rows[f"{B}_s{s}"] for s in seeds if f"{B}_s{s}" in rows}
    dirs_disp = "`, `".join(str(d.relative_to(ROOT)) for d in dirs)

    L: list[str] = []
    L.append(f"# {args.label} · `{A}` vs `{B}` 配对分析（统计层读数，**非判读**）")
    L.append("")
    L.append("> 口径：配对单位 = **世界种子**（R96）｜符号检验 `k` = **实际正向数**（F-R23）｜"
             "并列剔除并报 `n_eff`｜区制两套划法并列（R38③ / 内评 §四.2）｜**不判读**（R114a）")
    L.append(f"> 数据：`{dirs_disp}`（{len(rows)} run：{A} {len(a_arm)} / {B} {len(b_arm)}）")
    L.append("")

    gate_arm = any(v.get("gate_mode") == "delta_positive" for v in rows.values())
    alp_set = sorted({str(v.get("alphabet")) for v in rows.values() if v.get("alphabet")})

    # ---------------- ① 批次与仪器自检 ----------------
    L.append("## ① 批次与仪器自检")
    L.append("")
    L.append("| run | tick | 档位 | 生态门 | N | pred | 区制 | 成交 | 到达 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for s in seeds:
        for tag, d in ((B, b_arm), (A, a_arm)):
            v = d.get(s)
            if not v:
                continue
            arr_cell = f"{v['arrivals']:,}" if v.get("gate_mode") == "delta_positive" else "—"
            L.append(f"| {tag}_s{s} | {v['final_tick']} | {v.get('alphabet')} | "
                     f"{'T' if v['eco'] else 'F'}{'' if v['eco_applicable'] else '*(不适)'} | "
                     f"{v['N']} | {v['pred']:.3f} | {regime(v['pred'], v['N'])} | "
                     f"{v['funnel_applied']} | {arr_cell} |")
    L.append("")
    ticks_ok = len({v["final_tick"] for v in rows.values()}) == 1
    badcode = sorted({str(v.get("bad_code_n")) for v in rows.values()})
    ident = [v["identity_ok"] for v in rows.values() if v["identity_ok"] is not None]
    L.append(f"- 档位集合：**{alp_set}**｜`code_max` = "
             f"{sorted({v.get('code_max') for v in rows.values() if v.get('code_max')})}｜"
             f"`n_states` = {sorted({v.get('n_states') for v in rows.values() if v.get('n_states')})}")
    L.append(f"- **`bad_code_n` 取值集合 = {badcode}**（全 0 ⇒ 码域守卫全绿）｜tick 全部相同：**{ticks_ok}**")
    if ident:
        L.append(f"- `ledger.identity_ok` **{sum(bool(x) for x in ident)}/{len(ident)}**"
                 f"（非 oracle 臂无 ledger 属正常）")
    if gate_arm:
        L.append(f"- 守卫计数：`payer_trunc_n` 合计 {sum(int(v['payer_trunc_n'] or 0) for v in rows.values())}；"
                 f"`budget_exhausted_n` 合计 {sum(int(v['budget_exhausted_n'] or 0) for v in rows.values())}")
    L.append("")

    # ---------------- ② 门控量级（仅当有门控臂） ----------------
    bf: list[float] = []
    if gate_arm:
        L.append(f"## ② 门控量级（`{A}` 臂中 `gate_mode=delta_positive` 者，逐 seed）")
        L.append("")
        L.append("| seed | 到达 arrivals | pass | **block** | **蹭归因占比 block/arr** | Δ_full 均值 | Δ_content 均值 | **Δ_c/Δ_f** |")
        L.append("|---|---|---|---|---|---|---|---|")
        for s in seeds:
            v = a_arm.get(s)
            if not v or v.get("gate_mode") != "delta_positive" or not v.get("arrivals"):
                continue
            r = (v["mean_d_content"] / v["mean_d_full"]) if v["mean_d_full"] else float("nan")
            bf.append(float(v["block_frac"]))
            L.append(f"| s{s} | {v['arrivals']:,} | {v['gate_pass']:,} | **{v['gate_block']:,}** | "
                     f"**{v['block_frac']:.4f}** | {v['mean_d_full']:.4f} | {v['mean_d_content']:.4f} | {r:.3f} |")
        L.append("")
        if bf:
            ok_cons = all(int(a_arm[s]["gate_pass"]) + int(a_arm[s]["gate_block"]) == int(a_arm[s]["arrivals"])
                          for s in a_arm if a_arm[s].get("arrivals"))
            L.append(f"- **蹭归因占比**：中位 **{np.median(bf):.4f}**（范围 {min(bf):.4f} – {max(bf):.4f}）")
            L.append(f"- 门数守恒（pass + block = arrivals）：**{ok_cons}**")
        L.append("")

    # ---------------- ③ 逐 seed 配对表 ----------------
    L.append(f"## ③ 逐 seed 配对表（`{A} − {B}`，含**区制分类**）")
    L.append("")
    L.append("| 指标 | " + " | ".join(f"s{s}" for s in seeds) + " | 中位 Δ | 正/负/零 |")
    L.append("|---|" + "---|" * (len(seeds) + 2))
    stat: dict[str, dict] = {}
    skipped_na: list[str] = []
    any_oracle = any(v.get("oracle_on") for v in rows.values())
    for label, field, disp in FIELDS:
        if label in ORACLE_ONLY_LABELS and not any_oracle:
            skipped_na.append(disp)
            continue
        pr = pairs_of(rows, A, B, field)
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
    if skipped_na:
        L.append("")
        L.append(f"- ℹ️ **不适用（已略）**：{'、'.join(skipped_na)} —— 本批两臂**均非 oracle 臂** ⇒ "
                 "这些指标不存在（**不是**「无效应」）")
    L.append("")

    # ---------------- ④ 配对统计 ----------------
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
    L.append("⚠️ **须知**：并列（两臂同值）已被剔除 ⇒ `n_eff` 可能下降；**「不显著」在 `n_eff` 偏低时"
             "不可读作「无效应」** ⇒ 判读须看 §③ 的**逐 seed 值 + 区制分类**，不得只看 p。")
    L.append("")

    # ---------------- ④b 区制匹配子集（内评 09-19 §二 建议） ----------------
    matched = matched_seeds(b_arm, a_arm, seeds)
    stat_m: dict[str, dict] = {}
    L.append("## ④b 区制匹配子集（**同 seed 两臂同态**才配；稳健性检查，**非主口径**）")
    L.append("")
    L.append(f"- 匹配 seed = **{len(matched)}/{len(seeds)}**")
    if matched:
        L.append(f"- 匹配 seed 列表：{'、'.join(f's{s}' for s in matched)}")
    L.append("")
    if not matched:
        L.append("⚠️ **无任何同态 seed** ⇒ 本批**无法做区制匹配**；§③/§④ 的全配对只能按"
                 "「**两态相比**」解释（区制差已混入）。")
    else:
        L.append("| 指标 | 匹配 seed 数 | 中位 Δ | n_eff | p（双侧） | 判定 |")
        L.append("|---|---|---|---|---|---|")
        for label, field, disp in FIELDS:
            if label in ORACLE_ONLY_LABELS and not any_oracle:
                continue
            pr = [p for p in pairs_of(rows, A, B, field) if p[0] in set(matched)]
            if not pr:
                continue
            d = [p[3] for p in pr]
            pos = sum(1 for x in d if x > 0)
            neg = sum(1 for x in d if x < 0)
            zer = sum(1 for x in d if x == 0)
            neff = pos + neg
            pm = sign_test_two_sided(pos, neg)
            stat_m[field] = {"label": label, "delta": d, "pos": pos, "neg": neg, "zero": zer,
                             "median_delta": float(np.median(d)), "n_eff": neff, "p_sign": pm}
            verdict = "🟡 显著（p<0.05）" if pm < 0.05 else "不显著"
            if neff < 4:
                verdict += f"⚠️ **功效受限**（n_eff={neff}）"
            L.append(f"| {disp} | {len(pr)} | {np.median(d):+.4f} | {neff} | {pm:.4f} | {verdict} |")
        L.append("")
    L.append("⚠️ **口径**：本表是**后处理筛选**（同 seed 两臂同态）⇒ 只作**稳健性检查**，"
             "**不替代** §③/§④ 的预注册全配对；子集 `n` 小 ⇒ 「不显著」**不可**读作「无效应」；"
             "与全配对**冲突时判「不可判」**，不得挑对自己有利的那组（内评 09-19 §二）。")
    L.append("")

    # ---------------- ④c 功效口径（可检出阈，自算，不硬编码） ----------------
    L.append("## ④c 功效口径（**可检出阈**，双侧 0.05 / 80%，由实测 σ_d 推出）")
    L.append("")
    L.append("> 口径：可检出 |Δ| ≈ `(t₀.₉₇₅,df + t₀.₈₀,df) · σ_d / √n`（σ_d = 实测配对差的样本标准差，df = n−1）。")
    L.append("> 用途（内评 09-19 §三 请求）：把「**不显著**」写成**可证伪的句子** —— 只能是")
    L.append("> 「**排除了 |Δ| ≳ 该值的效应**」，**不是**「无效应」。阈值**由功效推**，禁事后倒推（教训 2/9）。")
    L.append("")
    L.append("| 指标 | 配对数 n | σ_d | **可检出 \\|Δ\\|** | 中位 Δ |")
    L.append("|---|---|---|---|---|")
    det: dict[str, float] = {}
    for field, st in stat.items():
        d = st["delta"]
        if len(d) < 2:
            continue
        thr = detectable_delta(d)
        det[field] = thr
        L.append(f"| {st['label']} | {len(d)} | {np.std(d, ddof=1):.4f} | "
                 f"**{thr:.4f}** | {st['median_delta']:+.4f} |")
    L.append("")
    if stat:
        _worst = max(det.values()) if det else None
        if _worst is not None:
            L.append(f"- ⇒ 本批**最宽松**的可检出阈 = **{_worst:.4f}**（最大者）；"
                     "任何「不显著」都只能写作「排除了 ≳ 该量级的效应」。")
            L.append("")

    # ---------------- ⑤ 区制 ----------------
    L.append("## ⑤ 区制（同 seed 跨臂是否翻转）")
    L.append("")
    L.append(f"| seed | {B} N / pred / 区制 | {A} N / pred / 区制 | 翻转？ |")
    L.append("|---|---|---|---|")
    flips = []
    for s in seeds:
        u, g = b_arm.get(s), a_arm.get(s)
        if not u or not g:
            continue
        ru, rg = regime(u["pred"], u["N"]), regime(g["pred"], g["N"])
        same_a = ru.split("/")[0] == rg.split("/")[0]
        flips.append(not same_a)
        L.append(f"| s{s} | {u['N']} / {u['pred']:.3f} / {ru} | {g['N']} / {g['pred']:.3f} / {rg} | "
                 f"{'—' if same_a else '🔴 **翻转**'} |")
    L.append("")
    if flips:
        L.append(f"- 划法 A（`pred_frac ≥ {PRED_DOMAIN_MAX}` = 捕食主导）下**翻转 seed 数 = "
                 f"{sum(flips)}/{len(flips)}**")
        L.append("- ⇒ 若翻转 ≥1 ⇒ 与 C1a/C1b/C2/gate 同型：**同 seed、仅换一个开关即跨区制** = "
                 "「**近临界抽签**」的又一次独立复现")
    L.append("")

    # ---------------- ⑥ 边界声明 ----------------
    L.append("## ⑥ 边界声明（**本批不判读**，R114a）")
    L.append("")
    # 批次专属边界（推荐走文件：避免 shell 引号/反引号陷阱）
    extra_b: list[str] = []
    if args.boundaries_file:
        bp = Path(args.boundaries_file)
        if not bp.is_absolute():
            bp = ROOT / bp
        for line in bp.read_text(encoding="utf-8").splitlines():
            t = line.strip()
            if t and not t.startswith("#"):
                extra_b.append(t)

    L.append("1. 本报告只出**统计层读数**；**判读归仪器线**（R114a）")
    L.append("2. **区制二分 + 近临界抽签** ⇒ 生态类指标必须**分层报**，不得合并均值（R38③）")
    L.append("3. **`n_eff` 偏低**时的「不显著」**不可**读作「无效应」（功效受限）")
    L.append("4. 若两臂属**纪元变更**（如字母表档位不同）⇒ **不是单变量对照**，结构差异可混入任何差异")
    L.append("5. **§④b 区制匹配子集是后处理筛选**（conditional-on-post-treatment）⇒ 只作**稳健性检查**，"
             "不替代预注册全配对；与全配对**冲突 ⇒ 判「不可判」**，不得挑组（内评 09-19 §二）")
    L.append("6. **「不显著」必须按 §④c 的可检出阈写成「排除了 |Δ| ≳ X」**（内评 09-19 §三）；"
             "不得写成「无效应」——**12k 短批只能排除大效应**")
    _COMMON_B = ["（通用 1）只出统计层读数", "（通用 2）区制须分层", "（通用 3）n_eff 偏低不可读无效应",
                 "（通用 4）纪元变更非单变量", "（通用 5）区制匹配子集仅作稳健性检查、冲突判不可判",
                 "（通用 6）不显著须写成「排除了 ≳ 可检出阈」"]
    if extra_b:
        L.append("")
        L.append("**本批专属边界**（调用方传入；缺省则只有上六条通用项）：")
        L.append("")
        for b in extra_b:
            L.append(f"- {b}")
        res_boundaries = _COMMON_B + list(extra_b)
    else:
        res_boundaries = list(_COMMON_B)
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
            "batch_dirs": [str(d.relative_to(ROOT)) for d in dirs], "label": args.label,
            "arm_a": A, "arm_b": B,
            "judged_under": "统计层读数（非判读，R114a）",
            "criterion_note": "配对单位=世界种子（R96）；符号检验 k=实际正向数（F-R23）；并列剔除",
            "n_runs": len(rows), "n_a": len(a_arm), "n_b": len(b_arm),
            "alphabet_set": alp_set, "bad_code_n_set": badcode, "ticks_same": ticks_ok,
            "block_frac_median": float(np.median(bf)) if bf else None,
            "block_frac_range": [float(min(bf)), float(max(bf))] if bf else None,
            "paired_stats": stat,
            "paired_stats_regime_matched": stat_m,
            "regime_matched_seeds": matched, "regime_matched_of": len(seeds),
            "detectable_delta": det,
            "regime_flips": int(sum(flips)), "regime_flips_of": len(flips),
            "boundaries": res_boundaries,
            "rows": rows,
        }
        p = Path(args.json_out) if Path(args.json_out).is_absolute() else (ROOT / args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[saved] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
