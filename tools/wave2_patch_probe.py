"""13.4 波 2 前置探针：斑块"休耕—死亡—轮作"（R176 §12.5）的**阈值可达性**与**守恒性**。

零 RNG、只读、不跑 tick。一条命令复现设计稿引用的全部数字：

    .venv/Scripts/python.exe -X utf8 tools/wave2_patch_probe.py

三个问题（都是"动手前必须先回答"的）：

  A. `kill_frac = 被吃量 / capacity` 这个口径**在真实世界几何下可达吗？**
     —— `capacity = capacity_per_area × 格面积`，而格面积跨纬度差 **~40 倍**
        （赤道 ≈1，极点 ≈0.026）⇒ 同一个 `kill_frac=0.7` 在赤道与极点是**两个世界**。
  B. 换成"被吃量 / 当期再生量"（= 格内**吃>长**的倍数）是否**纬度无关**、阈值是否好定？
  C. 「死格重入候选池」若真的把斑块加成**搬走**，Σcapacity / Σ再生倍率还守恒吗？
     —— 若要求**精确守恒**，就必须**同行（同面积）**交换；本探针证明这一点。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build(patch_count: int = 30, patch_radius: int = 2, occ_cap: int = 3,
          eat_amount: float = 0.5, max_count: int = 3240):
    """构造与 a4 正式批同构的引擎（patchy / 16 码 / 软顶 0.6）。"""
    from experiments.a4_verify_capacity import build as a4build
    return a4build("on", True, 42, 1, max_count=max_count, distribution="patchy",
                   signal_alphabet="16", soft_cap_target=0.6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--occ-cap", type=int, default=3, help="单格个体上限（波 2 §十一.2 拟 3）")
    ap.add_argument("--eat-mult-max", type=float, default=1.5,
                    help="`eat_mult = 0.5 + g4` 的上限（g4=1 ⇒ 1.5）")
    args = ap.parse_args()

    e = build()
    rf = e.resources
    w = e.world
    cfg = e.config
    n = w.n_cells
    cap = np.asarray(rf._capacity, dtype=np.float64)
    mask = rf._patch_mask
    assert mask is not None, "需要 distribution='patchy'"

    # 面积（按行；同纬度带面积相同 —— 这是 C 段「同行交换」守恒的根据）
    area = cap / float(cfg.resources.capacity_per_area)
    row = np.arange(n) // w.cols
    row_area = np.array([area[row == r][0] for r in range(w.rows)])

    print("=== A. 容量几何（patchy / 30 中心 / radius 2）===")
    print(f"  cells = {n}｜patch 格 = {int(mask.sum())} ({mask.mean() * 100:.2f}%)"
          f"｜背景格 = {int((~mask).sum())}")
    print(f"  Σcapacity = {cap.sum():,.1f}｜Σcell_area = {area.sum():.1f}"
          f"｜capacity_per_area = {cfg.resources.capacity_per_area}")
    print(f"  {'格类':<8}{'min':>9}{'p50':>9}{'mean':>9}{'max':>10}")
    for lab, m in (("patch", mask), ("bg", ~mask)):
        c = cap[m]
        print(f"  {lab:<8}{c.min():>9.4f}{np.median(c):>9.4f}{c.mean():>9.4f}{c.max():>10.2f}")
    print(f"  ⇒ 容量跨纬度比 = max/min = **{cap.max() / cap.min():.1f} 倍**"
          "（赤道格 vs 极点格）")

    # 每 tick 每格**最大可达**取食量（受单格上限 & 基因上限约束，且受存量约束）
    intake_max = float(cfg.organisms.eat_amount) * args.eat_mult_max * args.occ_cap
    print()
    print("=== B. `kill_frac` 两种分母的可达性（阈 0.7）===")
    print(f"  每格每 tick 最大取食量 = eat_amount({cfg.organisms.eat_amount}) × eat_mult_max"
          f"({args.eat_mult_max}) × occ_cap({args.occ_cap}) = **{intake_max:.3f}**")

    # 当期再生（温度因子 =1 的赤道正午上界）：patch ×2.0，背景 ×bg_mult
    rg = float(cfg.resources.regrowth_rate)
    growth_patch = rg * rf._patch_regrowth_mult
    growth_bg = rg * rf._bg_regrowth_mult
    print(f"  当期再生（温度因子=1）：patch {growth_patch:.3f}/tick｜背景 {growth_bg:.3f}/tick")

    print()
    print("  ① 分母 = `capacity`（设计稿 §12.5 字面）:")
    r1 = intake_max / cap
    for lab, m in (("patch", mask), ("bg", ~mask)):
        v = r1[m]
        print(f"     {lab:<6} 比值 max={v.max():.3f}｜过 0.7 的格 = "
              f"{int((v > 0.7).sum())}/{int(v.size)} ({float((v > 0.7).mean()) * 100:.2f}%)")
    # 按行看（谁会被杀）
    hit = np.flatnonzero(r1 > 0.7)
    print(f"     过阈格共 {hit.size} 个 ⇒ 所在行 = "
          f"{sorted(set((hit // w.cols).tolist()))[:12]}{' …' if hit.size > 12 else ''}")
    print(f"     赤道行(29/30) 比值 = {r1[29 * w.cols:30 * w.cols].max():.4f}"
          f"｜极点行(0) 比值 = {r1[:w.cols].max():.3f}")

    print()
    print("  ② 分母 = **当期再生量**（'格内吃>长'的倍数）:")
    for lab, g, m in (("patch", growth_patch, mask), ("bg", growth_bg, ~mask)):
        v = intake_max / g
        print(f"     {lab:<6} 单人={1.0 * float(cfg.organisms.eat_amount) * args.eat_mult_max / g:.3f}"
              f"｜满格({args.occ_cap}人)={intake_max / g:.3f}")
    print("     ⇒ 与纬度**无关**（再生只随温度变，不随面积变）⇒ 阈值有单一物理含义："
          "\n        「**同格几个人的取食量超过这格的再生**」")

    # 静止独占一格的存量寿命（供 §2.5「吃>长」交叉核对）
    print()
    print("=== C. 静止者的存量寿命（'站着不动到底能不能活'）===")
    print(f"  {'参数档':<16}{'格类':<7}{'1人净耗':>9}{'1人吃空':>10}{'3人净耗':>9}{'3人吃空':>10}")
    for tag, ea, rr in (("现状 0.5/0.5", float(cfg.organisms.eat_amount),
                         float(cfg.resources.regrowth_rate)),
                        ("波2 0.6/0.4", 0.6, 0.4)):
        gp = rr * rf._patch_regrowth_mult
        gb = rr * rf._bg_regrowth_mult
        for lab, g, c in (("patch", gp, float(cap[mask].mean())),
                          ("bg", gb, float(cap[~mask].mean()))):
            row_s = f"  {tag:<16}{lab:<7}"
            for occ in (1, 3):
                net = ea * occ - g
                life = c / net if net > 0 else float("inf")
                ls = "  ∞（永不空）" if life == float("inf") else f"{life:>9.0f}"
                row_s += f"{net:>+9.3f}{ls:>10}"
            print(row_s)
    print("  🔴 关键：**斑块格再生(1.0) > 单人取食(0.5)** ⇒ 单人在斑块格上静止**永不枯竭**"
          "（净耗 −0.5）")
    print("     ⇒ 「吃>长」在**斑块格 × 单人**这个组合上**依旧不成立**，"
          "只有『多人同格』或『背景格』才成立")
    print("  📐 设计稿 §2.5 的「200 tick 吃空一格」= 40/(0.6−0.4) ⇒ 那是**背景格基准容量**口径；")
    print(f"     斑块格容量均值 {cap[mask].mean():.1f}（×3 倍率）⇒ 若要 3 人同格吃空需 "
          f"{cap[mask].mean() / max(1e-9, 0.6 * 3 - 0.4 * rf._patch_regrowth_mult):.0f} tick")

    # D 段：轮作守恒
    print()
    print("=== D. 「斑块加成搬走」的守恒性（C 段问题）===")
    print("  面积只依赖纬度（同一行内所有格面积相同）")
    area = np.asarray(w.cell_area(np.arange(n)), dtype=np.float64)
    base = float(cfg.resources.capacity_per_area) * area
    bg_cap_mult = float(cap[~mask][0] / base[~mask][0])
    p_mult = float(cap[mask][0] / base[mask][0])
    print(f"  基准容量/格 = capacity_per_area({cfg.resources.capacity_per_area}) × 面积"
          f"｜patch 倍率 = {p_mult:.2f}｜背景容量倍率 = {bg_cap_mult:.4f}")
    tot_area = float(area.sum())

    def sigma_cap(m):
        return float(np.where(m, base * p_mult, base * bg_cap_mult).sum())

    def regrow_bg_mult(m):
        pa = float(area[m].sum())
        return (tot_area - rf._patch_regrowth_mult * pa) / (tot_area - pa)

    def sigma_rg_mult(m):
        """面积加权再生倍率之和 ÷ 总面积（构造期守恒式 = 1 ⇒ 周期总再生量不变）。"""
        m2 = rf._patch_regrowth_mult
        gb = regrow_bg_mult(m)
        return float((np.where(m, m2, gb) * area).sum() / tot_area)

    c0, g0 = sigma_cap(mask), sigma_rg_mult(mask)
    print(f"  构造期：Σcapacity = {c0:,.1f}｜面积加权 Σ再生倍率/总面积 = {g0:.6f}（应 = 1）")
    # 实测：同行交换 vs 跨行交换（**按 mask 重算容量**，而非搬数值）
    rng = np.random.default_rng(7)
    for lab, same_row in (("同行（同面积）", True), ("跨行", False)):
        m = mask.copy()
        sw = 0
        for _ in range(1000):
            p = np.flatnonzero(m)
            if p.size == 0:
                break
            i = int(rng.choice(p))
            cand = np.flatnonzero(~m & ((row == row[i]) if same_row else (row != row[i])))
            if cand.size == 0:
                continue
            j = int(rng.choice(cand))
            m[i], m[j] = False, True
            sw += 1
        print(f"  {lab:<14} 交换 {sw:>4} 次：ΔΣcapacity = {sigma_cap(m) - c0:+.6e}"
              f"｜Δ(Σ再生倍率/总面积) = {sigma_rg_mult(m) - g0:+.6e}")
    print("  ⇒ 结论：轮作若真要搬走斑块加成，**必须同行（同面积）交换**，"
          "否则破坏『Σ容量守恒』与『周期总再生量守恒』两条既有不变量")


if __name__ == "__main__":
    main()
