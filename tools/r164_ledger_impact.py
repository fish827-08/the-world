"""记账污染量化（R163 复核附带；**2026-09-23 更正版**）。

## 为什么要更正（老工自曝）

初版把 `corpse_probe()["corpse_eaten"]` 当作**质量**算"×3 放大"。实际上它是
`_corpse_eaten_n` = **个体计数**（每 tick"取到食的个体数"累加，见
`_step_scavenging` 的 `+= int(np.count_nonzero(take > 1e-12))`）——
**既不是质量、也不是能量**（所有者 R164 帖里写的就是"万**次**"）。
⇒ 初版那条 "+3x infl ≈ 0.06 能量/人·tick" 量纲错误，**已作废**。

## 现在能算的

| 项 | 可否量化 | 口径 |
|---|---|---|
| `contest_cost` 漏计 | ✅ 可算 | `contest_cost_energy × contest_n ÷ ∫N dt`（两者同为"能量"） |
| `heal_cost` 漏计 | ⚠️ 未算（量级更小） | `wound_heal_energy_cost × 愈合个体·tick` |
| 尸体 ×3 放大 | ❌ 13.3 批**不可事后量化** | 缺质量/能量计数 ⇒ 13.4 起用 `intake_scav` 前向测 |

用法：`.venv/Scripts/python.exe -X utf8 tools/r164_ledger_impact.py [批目录]`
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
    miss = float(r["wound"]["contest_cost_energy"]) * float(
        r["wound"].get("contest_n") or 0.0) / pt
    led = (r.get("energy_ledger") or {}).get("global") or {}
    scav = led.get("intake_scav_sum")
    rows.append((arm, seed, n, miss, scav))

print(f"=== ledger contamination audit: {BATCH} ===")
print("  NOTE: `corpse_eaten` 是**个体计数**，不可当质量/能量用（初版错误，已更正）")
print(f"  {'arm':<4}{'seed':>5}{'N':>6}{'missed contest cost':>21}"
      f"{'intake_scav_sum':>18}")
for a, s, n, m, sc in rows:
    txt = f"{sc:.4f}" if sc is not None else "n/a (13.3 批无此列)"
    print(f"  {a:<4}{s:>5}{n:>6}{m:>21.4f}{txt:>18}")

print()
print("=== per-arm median: missed contest cost (energy per person-tick) ===")
agg = {}
for a, s, n, m, sc in rows:
    agg.setdefault(a, []).append(m)
for a in sorted(agg):
    print(f"  {a}: {st.median(agg[a]):>7.4f}")
print()
print("=> This term biases arms with contests ON (D) systematically POSITIVE:")
print("   R164's 'net income turned positive' must subtract it.")
print("=> The corpse x3 inflation CANNOT be quantified post-hoc for 13.3 runs;")
print("   from 13.4 on it is read directly from energy_ledger.intake_scav_sum.")
