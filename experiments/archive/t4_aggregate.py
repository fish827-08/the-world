"""T4 交付聚合（480×960 稳态 K @ k=2.5）—— 把 6 个 seed 的全量轨迹汇成分层表。

判据（任务书，落在完整轨迹尾部**事后**判）：
* **平台** = 末 3 个采样点相对变化 < 3%（`--stop-stable 0` 跑满预算后才判，避免中途截断）
* **K** = 平台期中位数（末 20% 样本）；**按 seed 分层**，不跨 seed 合并
* 未达平台 ⇒ **只报下界**（R208 §一 补报项 ②）

R208 §一 补报项 ①（撞顶）：逐 seed 报 N 曲线 vs **软顶 0.9×max_count=27 000** / **硬顶 30 000** 的贴近度。

用法
----
    .venv\\Scripts\\python.exe experiments/t4_aggregate.py
    .venv\\Scripts\\python.exe experiments/t4_aggregate.py --dir results/t4_steady_k_k25 --suffix _full
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print 会让脚本 rc=1 假失败；F-R15 族）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # 非 TTY / 旧解释器：不因诊断能力缺失而阻断运行


def load_run(csv_path: Path, summ_path: Path) -> dict:
    with io.open(csv_path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    ticks = [int(r["tick"]) for r in rows]
    pops = [int(r["pop"]) for r in rows]
    # R211 前旧产物列名为 `util`（留档不改）⇒ 两名列都认
    sats = [float(r["saturation"] if "saturation" in r else r["util"]) for r in rows]
    mss = [float(r["ms_per_tick"]) for r in rows]
    gens = [float(r["mean_gen"]) for r in rows]
    with io.open(summ_path, encoding="utf-8") as fh:
        doc = json.load(fh)
    s = doc["runs"][-1] if isinstance(doc.get("runs"), list) and doc["runs"] else doc
    n = len(pops)
    tail3 = pops[-3:]
    near = [abs(tail3[i + 1] - tail3[i]) / max(tail3[i], 1) for i in range(2)]
    platform = all(x <= 0.03 for x in near)
    window = max(int(n * 0.8), 1)
    k_meas = int(sorted(pops[window:])[len(pops[window:]) // 2]) if pops[window:] else 0
    span = min(8, n - 1)                       # 末 8 个采样点 = 2000 tick
    rise_span = (pops[-1] - pops[-1 - span]) / max(pops[-1 - span], 1) if span > 0 else float("nan")
    span40 = min(40, n - 1)                    # 末 40 个采样点 = 10000 tick
    rise_10k = (pops[-1] - pops[-1 - span40]) / max(pops[-1 - span40], 1) if span40 > 0 else float("nan")
    curve = {}
    for tt in (1000, 2500, 5000, 10000, 20000, 30000, 40000):
        hit = [(int(r["tick"]), int(r["pop"])) for r in rows if int(r["tick"]) == tt]
        if hit:
            curve[tt] = hit[0][1]
    soft, hard = 27000, 30000
    return {
        "seed": s["seed"], "cells": s["cells"], "k": s["k"],
        "prod_cells": s["productive_cells"], "K_extrap": s["K_extrapolated"],
        "K": k_meas, "pop_max": max(pops), "pop_final": pops[-1],
        "tail3": tail3, "platform": platform, "near3": [round(x, 4) for x in near],
        "rise_2000tick": round(rise_span, 4),
        "rise_10000tick": round(rise_10k, 4),
        "K_per_cell": round(k_meas / s["productive_cells"], 4),
        "curve": curve,
        "saturation_tail": round(sum(sats[window:]) / len(sats[window:]), 4),
        "ms_tick_tail": round(sorted(mss[window:])[len(mss[window:]) // 2], 2),
        "max_gen": max(gens), "ticks": ticks[-1], "n_samples": n,
        "soft_hit": max(pops) >= soft, "hard_hit": max(pops) >= hard,
        "soft_frac": round(max(pops) / soft, 3), "hard_frac": round(max(pops) / hard, 3),
        "stop": s["stop"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="T4 分层 K 汇总（6 seed × 全量轨迹）")
    ap.add_argument("--dir", default="results/t4_steady_k_k25")
    ap.add_argument("--suffix", default="_full")
    ap.add_argument("--out", default="results/t4_steady_k_k25/summary_table.json")
    a = ap.parse_args()

    d = Path(a.dir)
    runs = []
    for csv_path in sorted(d.glob(f"s*{a.suffix}.csv")):
        stem = csv_path.stem
        summ = csv_path.with_suffix(".summary.json")
        if summ.exists():
            runs.append(load_run(csv_path, summ))

    if not runs:
        print(f"❗ 没找到 `{d}/s*{a.suffix}.csv` + 同名 .summary.json")
        sys.exit(2)

    print(f"== T4 分层汇总：{len(runs)} seed（{a.dir}，后缀 {a.suffix}）==")
    print(f"{'seed':>5}{'产能格':>8}{'外推K':>8}{'K':>7}{'峰值':>7}{'tail3':>26}"
          f"{'平台':>5}{'末2000':>8}{'末10000':>9}{'K/格':>7}{'饱和度':>8}{'ms/tick':>9}"
          f"{'世代':>5}{'软顶比':>8}{'tick':>7}  终止")
    print("-" * 140)
    for r in runs:
        t3 = "/".join(str(x) for x in r["tail3"])
        print(f"{r['seed']:>5}{r['prod_cells']:>8}{r['K_extrap']:>8.0f}{r['K']:>7}"
              f"{r['pop_max']:>7}{t3:>26}{'是' if r['platform'] else '否':>4}"
              f"{r['rise_2000tick'] * 100:>7.1f}%{r['rise_10000tick'] * 100:>8.1f}%"
              f"{r['K_per_cell']:>7.3f}{r['saturation_tail']:>8}{r['ms_tick_tail']:>9.2f}"
              f"{r['max_gen']:>5.0f}{r['soft_frac']:>8.2f}{r['ticks']:>7}  {r['stop']}")

    ticks_axis = sorted({t for r in runs for t in r["curve"]})
    if ticks_axis:
        print(f"\n跨 seed 曲线（中位 / 最小–最大）")
        print(f"{'tick':>8}{'N 中位':>9}{'N 范围':>14}")
        for tt in ticks_axis:
            vals = [r["curve"][tt] for r in runs if tt in r["curve"]]
            vals.sort()
            med = vals[len(vals) // 2] if len(vals) % 2 else (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2
            print(f"{tt:>8}{med:>9.0f}{f'{vals[0]}–{vals[-1]}':>14}")

    plat = [r for r in runs if r["platform"]]
    ks = sorted(r["K"] for r in plat)
    print(f"\n平台 seed 数：{len(plat)}/{len(runs)}")
    if ks:
        med = ks[len(ks) // 2] if len(ks) % 2 else (ks[len(ks) // 2 - 1] + ks[len(ks) // 2]) / 2
        print(f"平台 seed 的 K：{ks} ⇒ 中位 {med}")
    else:
        print("❗ 无 seed 达平台 ⇒ **只报下界**（R208 §一 ②）："
              f"全体 pop_final 最小/最大 = {min(r['pop_final'] for r in runs)}/"
              f"{max(r['pop_final'] for r in runs)}")
    if any(r["soft_hit"] for r in runs):
        print("🔴 有 seed 撞软顶（27 000）⇒ 该 seed 的 K 是配置读数，不可用")
    extrap = runs[0]["K_extrap"]
    print(f"外推参照：{extrap}（3.6667 × 产能格；仅 seed 各自的 '外推K' 列有效）")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump({"runs": runs, "n_platform": len(plat), "K_platform": ks}, fh,
                  ensure_ascii=False, indent=2)
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
