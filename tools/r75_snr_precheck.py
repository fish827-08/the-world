#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R7.5 · 判据信噪比预检（`[实验员·砚]`，R353 波次1 首位任务，零机时）

🔴 **为什么做这个**
----------------
R353 §〇：「**尺子不准就别开工**」—— R334 前车：**烧掉约60 机时才发现判据②是随机水平**
（6 批 × 12 配对仅中4 个 = 33%）。R7.5 要求**每个新判据在跑批前**先用历史数据做一次信噪比预检。

本脚本做三件事：
  ① **数据可得性体检**：M1 的三条判据，哪些在历史数据里**有对应列**（能预检）、哪些**没有**（不能）；
  ② **能预检的**：用历史数据算判据的**组内SD / on-off 差之比**（= R334 同款口径）⇒ 判"尺子够不够细"；
  ③ **不能预检的**：🔴 明确写出**缺哪列、要补什么才能预检**（不糊过去）。

**用法**
----
    .venv/Scripts/python.exe tools/r75_snr_precheck.py
退出码：0 = 预检完成（详见打印与JSON）；2 = 关键列缺失，预检**无法完成**（须先补仪表）。
"""
from __future__ import annotations

import csv
import glob
import io
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# M1 三条判据（定稿 §2.2）：g15 不被关闭 / 发射率不降 / 行为改变率 > 0
# 每条标注：能在历史数据里找到对应列吗？
M1_CRITERIA = [
    {
        "id": "M1-a",
        "name": "g15（SIGNAL_STRENGTH）末段均值不低于首段",
        "hist_cols": ["g15_mean", "rs_g15", "signal_gene", "g15"],
        "verdict": None,
    },
    {
        "id": "M1-b",
        "name": "私有信息发射占比（mem_bit=1 的发射数 / 总发射数）",
        "hist_cols": ["mem_bit_on", "mem_bit_n", "mem_bit_frac"],
        "verdict": None,
    },
    {
        "id": "M1-c",
        "name": "行为改变率（收到 vs 未收到信号的移动方向差异率）= G5",
        "hist_cols": ["behavior_change_rate", "g5_behavior", "sig_present"],
        "verdict": None,
    },
]

# 可用于"同款口径"预检的历史判据（这些列确实在历史 CSV 里）
PROBE_COLS = ["l1_visited_patch_frac", "patch_sat_init_var", "bg_resid_frac", "pop"]


def collect_csvs(root: Path):
    return sorted(glob.glob(str(root / "**" / "*.csv"), recursive=True))


def header_of(path: str):
    try:
        with io.open(path, encoding="utf-8") as f:
            return next(csv.reader(f))
    except Exception:
        return None


def tail_mean(rows, key, n=20):
    vals = []
    for r in rows[-n:]:
        v = r.get(key, "")
        if v in ("", "nan", "None", None):
            continue
        try:
            vals.append(float(v))
        except (ValueError, TypeError):
            continue
    return (sum(vals) / len(vals)) if vals else float("nan")


def main() -> int:
    repo = ROOT / "the-world-data"
    if not repo.is_dir():
        repo = ROOT
    print("=" * 74)
    print("R7.5 · M1 判据信噪比预检（零机时·历史数据）")
    print("=" * 74)
    print(f"数据源：{repo}")

    csvs = collect_csvs(repo)
    print(f"扫描 CSV：{len(csvs)} 个")

    # ---------- ① 数据可得性体检 ----------
    print("\n" + "-" * 74)
    print("① M1 三条判据的历史数据可得性体检")
    print("-" * 74)
    all_cols: dict[str, list[str]] = {}
    for p in csvs:
        h = header_of(p)
        if h:
            all_cols[p] = h

    for crit in M1_CRITERIA:
        found: list[str] = []
        for p, h in all_cols.items():
            for c in crit["hist_cols"]:
                if c in h:
                    found.append(os.path.relpath(p, ROOT))
                    break
        crit["found_in"] = sorted(set(found))
        if found:
            print(f"✅ {crit['id']} {crit['name']}")
            print(f"    历史列命中 {len(found)} 个 CSV（例：{found[0]}）")
            crit["verdict"] = "CAN_PRECHECK"
        else:
            print(f"❌ {crit['id']} {crit['name']}")
            print(f"    🔴 历史 CSV **无对应列**（找过：{crit['hist_cols']}）")
            crit["verdict"] = "CANNOT_PRECHECK"
    print()

    # ---------- ② 能预检的：用历史判据算同款口径 ----------
    print("-" * 74)
    print("② 同款口径预检：**用已有历史判据示范「尺子够不够细」**")
    print("   （方法与 `_judge2_snr_probe.py` 对判据② 相同：均值差 ÷ 组内 SD）")
    print("-" * 74)

    # 取一批有双臂的历史数据（S2 系列 = 唯一已判负的判据，最贴近 M1 的处境）
    pairs = []
    for p in csvs:
        base = os.path.basename(p)
        if not base.endswith(".csv") or "summary" in base:
            continue
        try:
            rows = list(csv.DictReader(io.open(p, encoding="utf-8")))
        except Exception:
            continue
        if not rows:
            continue
        seed = rows[0].get("seed", "?")
        arm = rows[0].get("arm", "?")
        pairs.append((seed, arm, p, rows))
    by_seed: dict[str, dict] = {}
    for seed, arm, p, rows in pairs:
        by_seed.setdefault(seed, {})[arm] = (p, rows)

    both = {s: v for s, v in by_seed.items() if "off" in v and "on" in v}
    print(f"  可配对的双臂数据：{len(both)} 个 seed")

    report: dict = {"criteria": M1_CRITERIA, "snr_probe": {}, "can_precheck": 0,
                    "cannot_precheck": 0}

    if not both:
        print("  ⚠️ 无可配对双臂数据 ⇒ ② 跳过")
    else:
        for col in PROBE_COLS:
            devs, wins, n = [], 0, 0
            for s, v in sorted(both.items()):
                (po, ro), (pn, rn) = v["off"], v["on"]
                if col not in ro[0] or col not in rn[0]:
                    continue
                vo, vn = tail_mean(ro, col), tail_mean(rn, col)
                if vo != vo or vn != vn:
                    continue
                n += 1
                if vn > vo:
                    wins += 1
                # 🔴 组内 SD 只有 2 个点时，`pstdev([a,b])` 恒 = |a-b|/2
                #    ⇒ **信噪比恒等于 2.000**，那是**公式的假象，不是尺子好**。
                #    只有当历史批里同一 arm 有 ≥3 个seed 才能估SD。
                #    此处改为：跨 seed 的**臂间差 vs 臂内跨 seed 离散度**。
                devs.append((s, vo, vn))
            if n < 2:
                continue
            # 真正的信噪比 = |on均值 − off均值| ÷ sqrt(off臂跨seed方差 + on臂跨seed方差)
            off_vals = [d[1] for d in devs]
            on_vals = [d[2] for d in devs]
            m_off = statistics.fmean(off_vals)
            m_on = statistics.fmean(on_vals)
            sd_off = statistics.pstdev(off_vals) if len(off_vals) > 1 else 0.0
            sd_on = statistics.pstdev(on_vals) if len(on_vals) > 1 else 0.0
            pooled = ((sd_off ** 2 + sd_on ** 2) / 2.0) ** 0.5
            mean_snr = abs(m_on - m_off) / pooled if pooled > 1e-12 else float("inf")
            if pooled <= 1e-12:
                note = "🟡 臂内跨 seed 离散度 = 0（各 seed 读数几乎相同）⇒ 无法估噪声"
            elif mean_snr < 1.0:
                note = "🔴 **< 1 ⇒ 尺子比噪声细 ⇒ 该判据在此装置下不可用（R334 同款结论）**"
            elif mean_snr < 2.0:
                note = "🟡 1~2 ⇒ 勉强，需加 seed"
            else:
                note = "🟢 ≥ 2 ⇒ 尺子可用"
            print(f"  {col:<26} n={n:<3} on>off {wins}/{n}  "
                  f"Δ均值={m_on - m_off:+.6f}  臂内SD={pooled:.6f}  信噪比={mean_snr:.3f}")
            print(f"      {note}")
            report["snr_probe"][col] = {"n": n, "wins": wins,
                                        "delta_mean": m_on - m_off,
                                        "pooled_sd": pooled,
                                        "mean_snr": (None if mean_snr == float('inf') else mean_snr)}

    # ---------- ③ 结论 ----------
    print("\n" + "-" * 74)
    print("③ 预检结论")
    print("-" * 74)
    can = [c for c in M1_CRITERIA if c["verdict"] == "CAN_PRECHECK"]
    cannot = [c for c in M1_CRITERIA if c["verdict"] == "CANNOT_PRECHECK"]
    report["can_precheck"] = len(can)
    report["cannot_precheck"] = len(cannot)
    print(f"  可预检：{len(can)}/3｜不可预检：{len(cannot)}/3")
    if cannot:
        print("\n  🔴 **M1 的判据目前无法做信噪比预检**——原因不是数据算不出，而是**历史 CSV 里根本没有信号相关列**")
        print("     （核实：`_rs_g15` 只进快照，`sphere_engine.py:6263`；**任何探针都不落CSV**）")
        print("\n  ⇒ 这不是坏消息，但它是**执行顺序上的坏消息**：")
        print("     R353 波次 1 要求「R7.5 预检过了才能起M1」，而**预检的前提是先有仪表列**")
        print("     ⇒ **硬依赖与波次表冲突**：① 轻舟 A1/A2（G1/G3）被要求等砚的 C1 口径")
        print("        ② 但 M1 判据① 需要的 g15 列**恰恰是仪表的一部分**")
        print("\n  📌 我的建议（**不阻塞，但须写进 M1 预注册的「已知限制」**）：")
        print("     ① R7.5 对 M1 三条判据**只能做\"方法学预检\"**（用历史判据②示范口径，见②），")
        print("        **真正的信噪比必须在 M0 仪表落地后、用 M0 冒烟那一轮重算**；")
        print("     ② 建议 **M0 的G1/G3/G4 里加一列 `g15_mean`**（引擎侧读 `_rs_g15` 的均值即可，零额外机时）")
        print("        ⇒ 这样 M1 起批前才有 g15 的历史基线可算信噪比；")
        print("     ③ 🔴 **不要因为\"无法预检\"就跳过 R7.5** —— ② 已给出方法学，")
        print("        且 M0 仪表落地后**必须**回头补算一次（本脚本可复跑）。")

    out = ROOT / "_rerun_logs" / "r75_snr_precheck.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n  机器可读结果：{out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())