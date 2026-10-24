"""R44 功效预跑：为 G-A 判定定 **seed 数**，并检验候选判据的功效形态。

R44 规定（所有者 2026-09-13 16:55 采纳 `[内评]` G-5）
--------------------------------------------------
1. 功效预跑须覆盖 **≥3 条生态结局不同的臂（含一条预期饱和臂）**，按**最大方差臂**定 seed 数；
2. ⑤ 的样本量须按「**个体层 RS 方差**」定，**不得**用群体均值方差外推；
3. D-20 须先实测该方差。

本工具（**零机时**：用既有产物估方差，不再跑实验）
------------------------------------------------
- 输入 = 既有批次的 summary（默认 `_rerun_logs/d24`，可多目录合并）
- 逐臂/逐 seed 抽取 ⑤ `spearman_rho`（非饱和窗）、⑥ `resp_triple`、个体样本量 `n`
- 估 **臂内跨 seed 方差** ⇒ 取**最大方差臂**（R44 §①）
- 对候选判据做功效分析（解析式 + 蒙特卡洛）⇒ 给出**所需 seed 数**
- 🔴 特别检验 R96 建议判据「**各 seed 方向一致为正 ∧ 合并 ρ ≥ 0.10**」的功效形态
  （怀疑它是**反功效**规则：n 越大越难通过）

用法
----
    python experiments/r44_power_prerun.py [--dirs _rerun_logs/d24,_rerun_logs/d27]
                                           [--alt 0.10] [--power 0.8]
"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb, erf
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from observatory.statistics import t_quantile  # noqa: E402  R105：t 临界值单一真源

# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print 会让脚本 rc=1 假失败）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


RNG = np.random.default_rng(20260916)


def load_runs(dirs: list[Path]) -> dict[str, list[dict]]:
    """→ {arm: [{seed, rho, triple, n_ind, pred_frac, ratio}, ...]}（去重按 (arm, seed)）"""
    out: dict[str, list[dict]] = {}
    seen: set[tuple] = set()
    for dirp in dirs:
        for sp in sorted(dirp.glob("*.summary.json")):
            try:
                d = json.loads(sp.read_text(encoding="utf-8"))
            except Exception:
                continue
            sw, res = d.get("switches", {}), d.get("result", {})
            arm = sw.get("arm")
            don = sw.get("oracle_donation")
            if arm and don is not None:
                # 剂量不同 ⇒ 不得并入同一臂（否则跨剂量方差被误当跨 seed 方差）
                arm = f"{arm}@{don}"
            seed = sw.get("seed")
            if arm is None or seed is None:
                continue
            key = (arm, seed, dirp.name)
            if key in seen:
                continue
            seen.add(key)
            sg = res.get("selection_gradient") or {}
            ns = sg.get("non_sat") or {}
            resp = res.get("signal_response") or {}
            out.setdefault(arm, []).append({
                "seed": seed,
                "rho": ns.get("spearman_rho"),
                "triple": resp.get("resp_triple"),
                "n_ind": ns.get("n"),
                "pred_frac": res.get("final_pred_frac"),
                "ratio": (res.get("oracle") or {}).get("oracle_return_ratio"),
            })
    return out


def between_seed_sd(vals: list[float]) -> float:
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    return float(np.std(a, ddof=1)) if a.size >= 2 else float("nan")


def power_all_positive(n: int, mu: float, sd: float, n_mc: int = 200_000) -> float:
    """判据 R1a：**n 个 seed 的统计量全为正**。P(单 seed 为正) = Φ(μ/σ) ⇒ 功效 = p^n。"""
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    p_pos = float(0.5 * (1 + erf((mu / sd) / np.sqrt(2))))
    return p_pos ** n


def power_sign_test(n: int, mu: float, sd: float, alpha: float = 0.05) -> float:
    """判据 R3：**符号检验**（单尾）：至少 k 个为正，k = 二项(0.5) 的 1−α 临界值。"""
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    p_pos = float(0.5 * (1 + erf((mu / sd) / np.sqrt(2))))
    # 临界值：在 H0(p=0.5) 下最小 k 使 P(X>=k) <= alpha
    k = n
    for kk in range(0, n + 1):
        tail = sum(comb(n, i) for i in range(kk, n + 1)) / (2 ** n)
        if tail <= alpha:
            k = kk
            break
    # 功效：二项(n, p_pos) 达到 k 的概率
    return float(sum(comb(n, i) * p_pos ** i * (1 - p_pos) ** (n - i)
                     for i in range(k, n + 1)))


def power_ci_zero(n: int, mu: float, sd: float, alpha: float = 0.05,
                  n_mc: int = 200_000) -> float:
    """判据 R2（**σ 已知**版）：合并统计量的种子级 CI 下界 > 0（单尾 α）。"""
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    z = 1.6448536269514722  # 单尾 95%
    se = sd / np.sqrt(n)
    samples = RNG.normal(mu, se, n_mc)
    return float(np.mean(samples - z * se > 0))


def _t_crit(df: int, alpha: float = 0.05) -> float:
    """单尾 t 临界值 —— **委托** `observatory.statistics.t_quantile`（单一真源，2026-09-16 归一）。

    归一理由：R105/ρ v3 的簇级 CI 用同一个临界值；两处各写一份 ⇒ 迟早漂移（本项目已付学费）。
    """
    return t_quantile(df, alpha)


def power_ci_t(n: int, mu: float, sd: float, alpha: float = 0.05,
               n_mc: int = 200_000) -> float:
    """判据 R2 的**小样本版**（`[内评]` 00:28 §未核实② 要求）：σ **未知** ⇒ 单尾 **t** 临界值。

    n=4 时 `t_{3,.95} = 2.353` ≫ z = 1.645 ⇒ 比 z 版保守 ⇒ **n=4 可能不够**。
    """
    df = n - 1
    if df < 1:
        return float("nan")
    t_crit = _t_crit(df, alpha)
    xs = RNG.normal(mu, sd, size=(n_mc, n))
    m = xs.mean(axis=1)
    s = xs.std(axis=1, ddof=1)
    return float(np.mean(m - t_crit * s / np.sqrt(n) > 0))


def power_rule1_bound(n: int, mu: float, sd: float, floor: float) -> float:
    """判据 R1 的**上界（保守）**：P(A ∧ B) ≤ min(P(A), P(B))。

    ⚠️ 这不是精确功效（两合取项相关）。仅当 μ = floor 时，`P(B) ≡ 0.5` ⇒ 上界 ≤ 0.5
    ⇒ "R1 不可达 0.8" **由该上界即可判定**（a fortiori）。
    """
    return min(power_all_positive(n, mu, sd), power_floor(n, mu, sd, floor))


def power_rule1_joint(n: int, mu: float, sd: float, floor: float,
                      n_mc: int = 200_000) -> float:
    """判据 R1 的**精确联合**功效（蒙特卡洛）：n 个 seed **全为正** ∧ **合并(均值) ≥ floor**。

    `[内评]` 00:28 §二 问"R1 列是双重条件联合功效还是别的 p+" ⇒ 本条即**双重条件的联合功效**；
    表中 0.500 是**上界**（`power_rule1_bound`）在 μ=floor 下的必然值（第二合取项恒 0.5）。
    """
    xs = RNG.normal(mu, sd, size=(n_mc, n))
    ok = (xs > 0).all(axis=1) & (xs.mean(axis=1) >= floor)
    return float(ok.mean())


def power_floor(n: int, mu: float, sd: float, floor: float = 0.10) -> float:
    """判据 R1b：**合并统计量 ≥ floor**（点估计门槛，非检验）。

    ⚠️ 若真实效应恰好等于 floor ⇒ 功效恒 ≈ **0.5**（与 n 无关）——这是该口径的固有性质。
    """
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    se = sd / np.sqrt(n)
    return float(0.5 * (1 + erf(((mu - floor) / se) / np.sqrt(2))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="_rerun_logs/d24,_rerun_logs/d27")
    ap.add_argument("--alt", type=float, default=0.10, help="设计备择：真实合并 ρ（R94 floor）")
    ap.add_argument("--power", type=float, default=0.8, help="目标功效")
    ap.add_argument("--nmax", type=int, default=24)
    args = ap.parse_args()

    dirs = []
    for d in args.dirs.split(","):
        p = Path(d)
        dirs.append(p if p.is_absolute() else ROOT / p)
    runs = load_runs(dirs)
    if not runs:
        print("无数据")
        return 1

    print("=" * 100)
    print("R44 功效预跑（零机时：以既有产物估方差）  设计备择 合并ρ = "
          f"{args.alt}  目标功效 = {args.power}")
    print(f"数据源：{', '.join(str(p.relative_to(ROOT)) for p in dirs)}")
    print("=" * 100)

    # ---- ① 逐臂跨 seed 方差（R44 §①：须 ≥3 条异质臂 + 含预期饱和臂）----
    print("\n① 逐臂跨 seed 离散度（R44 §①：按**最大方差臂**定 seed 数）")
    hdr = f"{'arm':>12}{'k(seed)':>8}{'⑤ρ 值':>34}{'⑤ρ sd':>9}{'⑥tri sd':>9}{'n_个体':>22}{'区制':>14}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for arm, rs in sorted(runs.items()):
        rhos = [r["rho"] for r in rs if r["rho"] is not None]
        trips = [r["triple"] for r in rs if r["triple"] is not None]
        ns = [r["n_ind"] for r in rs if r["n_ind"] is not None]
        pf = [r["pred_frac"] for r in rs if r["pred_frac"] is not None]
        sd_r = between_seed_sd(rhos)
        sd_t = between_seed_sd(trips)
        regime = ("捕食主导" if pf and float(np.median(pf)) >= 0.9 else "非捕食")
        rows.append({"arm": arm, "k": len(rs), "sd_rho": sd_r, "sd_tri": sd_t})
        print(f"{arm:>12}{len(rs):>8}{str([round(v, 4) for v in rhos]):>34}"
              f"{sd_r:>9.4f}{sd_t:>9.4f}{str([int(v) for v in ns]):>22}{regime:>14}")

    print("\n  R44 §① 要求：**≥3 条生态结局不同的臂**（含一条预期饱和臂）")
    saturated = [r["arm"] for r in rows if r["arm"] in ("main", "control", "cb1")]
    print(f"  ⇒ 本批覆盖：{sorted(runs)}；预期饱和臂存在：{'main/control' if saturated else '待核'}")

    # ---- ② 最大方差臂（逐统计量）----
    print("\n② 最大方差臂（保守取用其 σ）")
    arm_rho = max(rows, key=lambda r: (r["sd_rho"] if np.isfinite(r["sd_rho"]) else -1))
    arm_tri = max(rows, key=lambda r: (r["sd_tri"] if np.isfinite(r["sd_tri"]) else -1))
    print(f"  ⑤ρ   最大方差臂 = **{arm_rho['arm']}**（σ = {arm_rho['sd_rho']:.4f}）")
    print(f"  ⑥tri 最大方差臂 = **{arm_tri['arm']}**（σ = {arm_tri['sd_tri']:.4f}）")

    # ---- ③ 个体层样本量（R44 §②）----
    print("\n③ 个体层样本量（R44 §②：⑤ 的样本量须按**个体层 RS 方差**定，非群体均值方差）")
    alln = [r["n_ind"] for rs in runs.values() for r in rs if r["n_ind"]]
    if alln:
        fishers = [1.0 / np.sqrt(max(1, v - 3)) for v in alln]
        print(f"  个体层 n 逐 run ∈ [{min(alln):,}, {max(alln):,}]（差 {max(alln)//max(1,min(alln))}×）")
        print(f"  ⇒ 个体层 Fisher-z SE ∈ [{min(fishers):.4f}, {max(fishers):.4f}]"
              f"（**饱和臂样本最少 ⇒ SE 最大**）")
        print("  ⚠️ 与 R95（提案 d-2 公共 n 子采样）联动：跨 seed 比较须用 `n_c = min` 子采样并**并报两列**。")

    # ---- ④ 候选判据的功效曲线 ----
    sd = arm_rho["sd_rho"]
    print(f"\n④ 候选判据功效曲线（σ = {sd:.4f}，最大方差臂 {arm_rho['arm']}；设计备择 μ = {args.alt}）")
    cands = [
        ("R1 联合(上界)", lambda n: power_rule1_bound(n, args.alt, sd, args.alt)),
        ("R1 联合(精确)", lambda n: power_rule1_joint(n, args.alt, sd, args.alt)),
        ("R2 CI>0 (z)", lambda n: power_ci_zero(n, args.alt, sd)),
        ("R2 CI>0 (t)", lambda n: power_ci_t(n, args.alt, sd)),
        ("R3 符号检验", lambda n: power_sign_test(n, args.alt, sd)),
    ]
    hdr2 = f"{'n(seed)':>8}" + "".join(f"{c[0]:>17}" for c in cands)
    print(hdr2)
    print("-" * len(hdr2))
    req: dict[str, int | None] = {}
    for n in range(3, args.nmax + 1):
        cells = []
        for name, fn in cands:
            pw = fn(n)
            cells.append(f"{pw:>17.3f}" if np.isfinite(pw) else f"{'—':>17}")
            if np.isfinite(pw) and pw >= args.power and req.get(name) is None:
                req[name] = n
        print(f"{n:>8}" + "".join(cells))

    print("\n⑤ 达到目标功效所需 seed 数")
    for name, _ in cands:
        r = req.get(name)
        print(f"  {name:32s} ⇒ **n = {r if r else f'>{args.nmax}（不可达）'}**")
    print("  ⚠️ R2「σ 未知 ⇒ 用 t 临界值」是**保守口径**（`[内评]` 00:28 §未核实② 要求）"
          "⇒ 建议**以 t 版为准**定 seed 数。")

    print("\n🔴 判据形态警示（R44 的核心产出）")
    p_pos = 0.5 * (1 + erf((args.alt / sd) / np.sqrt(2)))
    print(f"  单 seed 方向为正的概率 p+ = Φ(μ/σ) = **{p_pos:.3f}**")
    print(f"  ⇒ 「**各 seed 方向一致为正**」的功效 = p+^n："
          f"n=3 → {p_pos**3:.3f}，n=6 → {p_pos**6:.3f}，n=12 → {p_pos**12:.3f}")
    print("  ⇒ 🛑 **该合取项是「反功效」规则**（n 越大越难全为正，且 p+ 永远 <1）")
    print(f"  ⇒ 「**合并 ρ ≥ floor**」若真实效应恰等于 floor ⇒ 功效恒 ≈ 0.5（与 n 无关）")
    print("  ⇒ 建议（供裁定）：判定形态改为「**合并 ρ 的种子级 CI 下界 > 0**」，"
          "floor 用作**设计备择**（μ=ρ_target）而非点估计门槛；")
    print("     并把「方向一致」从**全一致**降为**符号检验**（或要求 ≥⌈0.8n⌉ 为正）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
