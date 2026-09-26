"""尺度重标定勘察探针（14.10 / 评审 P0.0 & P0.2）—— 回答外部评审的 4 个提问。

回答什么
--------
1. **`ā`（年龄因子均值）**：`speed = clip(g18 × age_factor × gain, 0, cap)` 里
   `age_factor` 的实测均值 ⇒ 决定 `gain = v_max / ā` 的标定
2. **`τ_row = cap_row / ḡ_row` 的分布（尤其 max）** ⇒ 决定"食物惰性求值"是否需要建 P 表
   （若 max τ_row ≤ ~500 ⇒ 直接求和即可，**不需要建表**）
3. **"20 tick 窗口"到底是哪个机制** ⇒ 我们这边是 `ars.giveup`（连走 20 tick 无获 ⇒ 确定性右转）
4. **tick 面额常数清点**（哪些是"每 tick 速率"、哪些是"tick 时长"、哪些是"每 tick 衰减"）

用法
----
    .venv\\Scripts\\python.exe experiments/tick_scaling_audit.py
    .venv\\Scripts\\python.exe experiments/tick_scaling_audit.py --ticks 4000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation.config import SimConfig                      # noqa: E402
from simulation.genes import Gene                            # noqa: E402
from simulation.sphere_engine import SphereEngine            # noqa: E402

# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print 会让脚本 rc=1 假失败；F-R15 族）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # 非 TTY / 旧解释器：不因诊断能力缺失而阻断运行



def age_factor_of(age: np.ndarray, lifespan: np.ndarray, ocfg) -> np.ndarray:
    af = np.ones_like(age, dtype=np.float64)
    af[age < ocfg.maturity_fraction * lifespan] = ocfg.young_mob_mult
    af[age >= ocfg.senile_fraction * lifespan] = ocfg.old_mob_mult
    return af


def main() -> None:
    ap = argparse.ArgumentParser(description="尺度重标定勘察：ā / τ_row / tick 面额常数")
    ap.add_argument("--rows", type=int, default=60)
    ap.add_argument("--cols", type=int, default=120)
    ap.add_argument("--patches", type=int, default=30)
    ap.add_argument("--pop", type=int, default=400)
    ap.add_argument("--ticks", type=int, default=4000)
    a = ap.parse_args()

    c = SimConfig(seed=42)
    c.world.rows, c.world.cols = a.rows, a.cols
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = True
    c.resources.patch_regrowth_mult = 1.195
    c.resources.patch_count = a.patches
    c.population.initial_count = a.pop
    eng = SphereEngine(c)

    # ---------- 1. 年龄因子均值 ā ----------
    print("=" * 78)
    print("① 年龄因子 ā（gain 标定用：gain = v_max / ā）")
    print("=" * 78)
    ages = []
    for _ in range(a.ticks):
        eng.step()
        if len(eng._flat) == 0:
            break
        if eng._tick % 200 == 0:
            ls = eng._lifespan(eng._genes[:, Gene.LIFE_GENE])
            af = age_factor_of(eng._age, ls, c.organisms)
            ages.append((int(eng._tick), int(len(af)), float(af.mean()),
                         float(eng._age.mean()), float(np.median(ls)),
                         float((eng._age < c.organisms.maturity_fraction * ls).mean()),
                         float((eng._age >= c.organisms.senile_fraction * ls).mean())))
    if ages:
        print(f"{'tick':>7}{'N':>7}{'ā':>9}{'均龄':>8}{'寿命中位':>10}{'幼体占比':>9}{'老体占比':>9}")
        for t, n, af, ag, ls, yng, old in ages[::max(1, len(ages) // 8)]:
            print(f"{t:>7}{n:>7}{af:>9.4f}{ag:>8.1f}{ls:>10.0f}{yng:>9.3f}{old:>9.3f}")
        af_all = [x[2] for x in ages]
        print(f"\n⇒ **ā 实测 = {np.mean(af_all):.4f}**（范围 {min(af_all):.3f}–{max(af_all):.3f}）")
        print(f"   幼体占比中位 {np.median([x[5] for x in ages]):.3f}｜"
              f"老体占比中位 {np.median([x[6] for x in ages]):.3f}")
        print(f"   ⇒ gain 标定（v_max=0.25）：gain = 0.25 / {np.mean(af_all):.3f} "
              f"= **{0.25 / np.mean(af_all):.3f}**")

    # ---------- 2. τ_row 分布 ----------
    print()
    print("=" * 78)
    print("② τ_row = cap_row / ḡ_row（食物惰性求值是否需要建 P 表）")
    print("=" * 78)
    R = eng.resources
    cap = R._capacity
    rows, cols = a.rows, a.cols
    cap_2d = cap.reshape(rows, cols)
    # 取一天内的平均再生速率（含昼夜温度调制）⇒ 用"最冷季/最冷时刻"作为保守上界
    g_sum = np.zeros_like(cap)
    D = int(c.light.rotation_period)
    sample = max(1, D // 60)
    n_s = 0
    for t in range(0, D, sample):
        g_sum += R._regrowth_amount(t)
        n_s += 1
    g_avg = g_sum / n_s
    g_2d = g_avg.reshape(rows, cols)
    prod = cap > 0
    print(f"{'行':>5}{'纬度':>9}{'产能格':>8}{'容量中位':>10}{'日均速率':>11}"
          f"{'τ_row中位':>11}{'τ_row最大':>11}")
    taus, tau_maxes = [], []
    for r in range(rows):
        pm = prod.reshape(rows, cols)[r]
        if not pm.any():
            continue
        c_r = cap_2d[r][pm]
        g_r = g_2d[r][pm]
        tau = np.where(g_r > 1e-12, c_r / np.maximum(g_r, 1e-12), np.inf)
        taus.append(float(np.median(tau)))
        tau_maxes.append(float(np.max(tau)))
        if r % max(1, rows // 10) == 0 or r in (0, rows - 1):
            lat = 90.0 - (r + 0.5) * 180.0 / rows
            print(f"{r:>5}{lat:>9.1f}{int(pm.sum()):>8}{np.median(c_r):>10.1f}"
                  f"{np.median(g_r):>11.4f}{np.median(tau):>11.1f}{np.max(tau):>11.1f}")
    print(f"\n⇒ **τ_row 中位 = {np.median(taus):.0f} tick｜τ_row 最大 = {max(tau_maxes):.0f} tick**")
    verdict = ("**≤ 500 ⇒ 不需要建 P 表，直接求和即可（精确）**" if max(tau_maxes) <= 500
               else "> 500 ⇒ 需要 P 表（评审 §4.2 的两张表，约 3.2 MB）")
    print(f"   评审判据（max τ_row ≤ ~500）：{verdict}")
    print("   实测「空→满」（整场，含温度调制）= 201 tick（前一轮已测，与本表同量级）")

    # ---------- 4. tick 面额常数清点 ----------
    print()
    print("=" * 78)
    print("④ tick 面额常数清点（D 从 2400 → 480 时必须怎么改）")
    print("=" * 78)
    print(f"{'类别':<14}{'参数':<38}{'现值':>10}{'改法':>10}")
    print("-" * 78)
    rows_tbl = [
        ("每 tick 速率", "resources.regrowth_rate", 0.5, "×k"),
        ("每 tick 速率", "organisms.base_metabolism", 0.6, "×k"),
        ("每 tick 速率", "organisms.move_cost", 0.4, "×k"),
        ("每 tick 速率", "organisms.photo_max", 0.1, "×k"),
        ("每 tick 速率", "organisms.homeo_upkeep", 0.15, "×k"),
        ("每 tick 速率", "fruit.charge_rate", 0.1, "×k"),
        ("每 tick 速率", "fruit.eat_rate", 0.05, "×k"),
        ("每 tick 速率", "corpse_wound.wound_heal_rate", 0.001, "×k"),
        ("每 tick 速率", "pleasure.alpha", 0.05, "×k"),
        ("每 tick 速率", "info_structure.learning_rate", 0.15, "×k"),
        ("每 tick 速率", "info_structure.alignment_rate", 0.1, "×k"),
        ("每事件量", "organisms.eat_amount", 0.9, "×k ?"),
        ("tick 时长", "light.rotation_period", 2400, "= D"),
        ("tick 时长", "info_structure.learning_maturity_ticks", 1200, "÷k"),
        ("tick 时长", "corpse_wound.corpse_decay_ticks", 600, "÷k"),
        ("tick 时长", "resource_dynamics.rest_ticks", 60, "÷k"),
        ("tick 时长", "resource_dynamics.dead_regen_ticks", 2000, "÷k"),
        ("tick 时长", "ars.fast_tau", 50.0, "÷k"),
        ("tick 时长", "ars.slow_tau", 500.0, "÷k"),
        ("tick 时长", "ars.giveup  ★评审问的 20 tick 窗口", 20, "÷k"),
        ("tick 时长", "pleasure.expectation_size", 120, "÷k"),
        ("tick 时长", "oracle.persistence", 10, "÷k"),
        ("tick 时长", "SignalField.duration（信号寿命）", 50, "÷k"),
        ("每 tick 衰减", "pleasure.valence_decay", 0.95, "^k"),
        ("每 tick 衰减", "pleasure.arousal_decay", 0.97, "^k"),
    ]
    for cat, name, val, how in rows_tbl:
        print(f"{cat:<14}{name:<38}{val:>10}{how:>10}")
    print("\n不改 ⇒ 相对昼夜的含义自动放大 k 倍（静默改科学）。")


if __name__ == "__main__":
    main()
