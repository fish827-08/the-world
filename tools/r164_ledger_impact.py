"""R164 记账污染量化（R163 复核附带；老工线，2026-09-22）。

把三处记账缺口换算成 **能量/人·tick**，用于修正 p1corpse 批里"净收入"类读数：

  1. 尸体来源的**质量**被记进 `intake_forage`，且消化时经 `x eat_efficiency(3.0)` ⇒
     若尸体池按设计稿是**能量单位**，则多记 `2 x corpse_eaten`（正确应为 1x）
  2. `contest_cost_energy` 不入账 ⇒ 漏计 `0.5 x contest_n`（只影响开争夺的臂）
  3. `wound_heal_energy_cost` 亦未入账（本脚本不估，量级更小）

`∫N dt` 用 `final_N x ticks` 近似（饱和态下误差小）。纯 ASCII，只读。

用法：在仓库根运行
    .venv/Scripts/python.exe -X utf8 tools/r164_ledger_impact.py
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys

BATCH = sys.argv[1] if len(sys.argv) > 1 else "_rerun_logs/p1corpse"
TICKS = 8000.0

rows = []
for p in sorted(glob.glob(os.path.join(BATCH, "*.summary.json"))):
    d = json.load(open(p, encoding="utf-8"))
    r = d["result"]
    name = os.path.basename(p)
    arm = name.split("_")[0]
    seed = name.split("_s")[-1][:-13]
    n = int(r["final_N"])
    pt = n * TICKS
    ce = float(r["corpse"]["corpse_eaten"])
    cn = float(r["wound"].get("contest_n") or 0.0)
    infl = 2.0 * ce / pt
    miss = float(r["wound"]["contest_cost_energy"]) * cn / pt
    rows.append((arm, seed, n, ce, cn, infl, miss))

print(f"=== ledger contamination audit: {BATCH} ===")
print(f"  {'arm':<4}{'seed':>5}{'N':>6}{'corpse_eaten':>14}{'contest_n':>12}"
      f"{'+3x infl':>11}{'miss cost':>11}{'sum':>9}")
for a, s, n, ce, cn, i, m in rows:
    print(f"  {a:<4}{s:>5}{n:>6}{ce:>14.1f}{cn:>12.0f}{i:>11.4f}{m:>11.4f}{i + m:>9.4f}")

print()
print("=== per-arm median (energy per person-tick) ===")
agg = {}
for a, s, n, ce, cn, i, m in rows:
    agg.setdefault(a, []).append((i, m))
for a in sorted(agg):
    v = agg[a]
    print(f"  {a}: +3x={st.median([x[0] for x in v]):>7.4f}  "
          f"miss={st.median([x[1] for x in v]):>7.4f}  "
          f"total={st.median([x[0] + x[1] for x in v]):>7.4f}")
print()
print("note: '+3x infl' 是**均摊上界**；实际 `scav_mult` 强 g16 门控 ⇒ 收入集中在")
print("      mid/hi 箱 ⇒ hi 箱的真实修正更大（分层需 `intake_scav` 通道）。")
