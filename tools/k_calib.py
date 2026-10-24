"""13.5 K 校准计算器（零机时，只读）。

用途：13.5 的**唯一硬判据**是「素食者实测 K ∈ [1200, 1900]」。K 由两个量决定：

    K = Σ_斑块名义再生(质量/tick) ÷ 人均需求(质量/tick)
    人均需求 = 人均支出(能量/tick) ÷ (eat_efficiency × 吸收率)

本工具把这条公式做成**可执行形式**：给定 `eat_efficiency` / `assim_herb` / 人均支出 /
目标 K，直接反推需要的 `patch_regrowth_mult`（以及当前参数下的 K）。跑批前就能拦住拍脑袋的值。

用法（**给 Windows Python 传路径一律用 Windows 形式**）：

    .venv/Scripts/python.exe -X utf8 tools/k_calib.py
    .venv/Scripts/python.exe -X utf8 tools/k_calib.py --eat-eff 7.5 --assim-herb 0.4
    .venv/Scripts/python.exe -X utf8 tools/k_calib.py --target-k 1600 --eat-eff 7.5

🔴 为什么需要它（2026-09-24 实况）：
    派工单 §二的算术隐含「`assim_herb=0.4` 会把人均需求从 0.254 提到 0.635」⇒ 但 `[所有者]`
    R187 为修"吸收率致灭绝"把 `eat_efficiency` 3.0→7.5（语义 = 完全燃烧值）⇒ **净吸收 =
    7.5×0.4 = 3.0 = 现状** ⇒ 人均需求仍是 **0.254** ⇒ 吸收率**不再是 K 的杠杆**。
    ⇒ 此时若仍按派工单把斑块再生 ×1.5（mult 3.0），K 会变成 **≈4 016**（≫ 软顶）⇒ 不达标。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: 人均支出（能量/tick）——R185 实测（食物丰裕条件下；食物紧时移动耗能上升 ⇒ 实际需求更高）
PER_CAPITA_SPEND_E = 0.762
#: 软顶（`soft_cap_target=0.6 × max_count 3240`）——K 必须低于它，食物才算绑定
SOFT_CAP = 1944.0


def sigma_coefficient(patch_regrowth_mult: float = 1.0, seed: int = 42,
                      tick: int = 0, bg_production_zero: bool = True) -> float:
    """返回「每单位 `patch_regrowth_mult` 对应的 Σ_斑块名义再生」（质量/tick）。

    Σ 与倍率**严格线性** ⇒ 量一次即可外推到任意倍率（本函数就是那条直线的斜率）。
    """
    from simulation.config import SimConfig
    from simulation.sphere_engine import SphereEngine

    c = SimConfig(seed=seed)
    c.simulation.use_sim_core = False
    c.resources.distribution = "patchy"
    c.resources.patch_regrowth_mult = float(patch_regrowth_mult)
    c.resources.bg_production_zero = bool(bg_production_zero)
    e = SphereEngine(c)
    rf = e.resources
    m = rf._patch_mask
    g = rf._regrowth_amount(tick)
    if bg_production_zero:
        return float(g[m].sum()) / max(1e-12, patch_regrowth_mult)
    return float(g.sum()) / max(1e-12, patch_regrowth_mult)


def k_of(sigma: float, eat_eff: float, assim: float,
         spend_e: float = PER_CAPITA_SPEND_E) -> float:
    """K = Σ ÷ 人均需求；人均需求 = 支出 ÷ (eat_efficiency × 吸收率)。"""
    need = spend_e / (eat_eff * assim)
    return sigma / max(1e-12, need)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eat-eff", type=float, default=None,
                    help="eat_efficiency（默认取 `SimConfig()` 当前值）")
    ap.add_argument("--assim-herb", type=float, default=None,
                    help="食草吸收率（默认取 `SimConfig()` 当前值）")
    ap.add_argument("--target-k", type=float, default=1600.0, help="目标 K（默认 1600）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tick", type=int, default=0)
    args = ap.parse_args()

    from simulation.config import SimConfig
    c = SimConfig()
    eat_eff = float(c.organisms.eat_efficiency if args.eat_eff is None else args.eat_eff)
    assim = float(c.organisms.assim_herb if args.assim_herb is None else args.assim_herb)
    mult_now = float(c.resources.patch_regrowth_mult)

    coef = sigma_coefficient(1.0, seed=args.seed, tick=args.tick)
    sigma = coef * mult_now
    need = PER_CAPITA_SPEND_E / (eat_eff * assim)
    k_now = sigma / need
    mult_need = args.target_k * need / coef

    print("=== 13.5 K 校准计算器（背景归零 + patchy；零机时）===")
    print(f"  Σ 斜率（每单位 patch_regrowth_mult）= {coef:.2f} 质量/tick"
          f"   ⇒ mult={mult_now:g} 时 Σ = **{sigma:.1f}**")
    print(f"  eat_efficiency = {eat_eff:g}｜assim_herb = {assim:g}"
          f"｜净吸收 = {eat_eff * assim:g}（现状基准 3.0）")
    print(f"  人均支出 = {PER_CAPITA_SPEND_E} 能量/tick ⇒ 人均需求 = **{need:.4f}** 质量/tick")
    print(f"  ⇒ **K(当前参数) = {k_now:,.0f}**"
          f"  {'✅ 绑定' if k_now < SOFT_CAP else '❌ 不绑定（≥ 软顶）'}"
          f"（软顶 {SOFT_CAP:.0f}）")
    print()
    print(f"  目标 K = {args.target_k:,.0f} ⇒ 需要 Σ = {args.target_k * need:.1f}"
          f" ⇒ **patch_regrowth_mult = {mult_need:.3f}**")
    print()
    print("=== 三种口径对照（说明 ② 的校准方向**取决于 ③ 的净吸收**）===")
    print(f"  {'口径':<34}{'净吸收':>8}{'人均需求':>10}{'K(mult=当前)':>14}{'目标K所需mult':>15}")
    for lab, eff, asm in (("现状：eff 3.0 / assim 1.0", 3.0, 1.0),
                          ("派工单 §二：eff 3.0 / assim 0.4", 3.0, 0.4),
                          ("R187 修法：eff 7.5 / assim 0.4", 7.5, 0.4),
                          ("R187 修法+食肉：eff 7.5 / assim 0.8", 7.5, 0.8)):
        nd = PER_CAPITA_SPEND_E / (eff * asm)
        print(f"  {lab:<34}{eff * asm:>8.1f}{nd:>10.4f}"
              f"{coef * mult_now / nd:>14,.0f}{args.target_k * nd / coef:>15.3f}")
    print()
    print("  ⇒ 🔴 读法：**净吸收相同 ⇒ K 相同**。R187 把 `eat_efficiency` 提到 7.5 就是为了让"
          "\n     `7.5×0.4 = 3.0` 回到现状净吸收（否则会灭绝）⇒ 于是**吸收率不再是 K 的杠杆**，"
          "\n     校准必须回到**生产侧**（`patch_regrowth_mult`）—— 且方向与派工单 §二相反（**要下调**）。")


if __name__ == "__main__":
    main()
