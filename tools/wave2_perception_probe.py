"""波 2 前置几何/量纲探针（只读；为零 RNG 的**构造期**测量）。

用途：把"把感知从 1 格扩到 2 格"的**连带后果**变成硬数据，供设计稿引用：
  A. `_nb_table` 的 stride 与**每格实际邻居数**（现在被当成同一个量用 → 隐患）
  B. span=2 的候选集规模分布（含极区退化）
  C. `densities = occ / nb_max * smw` 的**实际量级**（社交项到底有多弱）
  D. 候选数变化对决策段成本的一阶估计

纯 ASCII / 零 RNG（只做构造期与静态量计算）。
用法：在仓库根运行
    .venv/Scripts/python.exe -X utf8 tools/wave2_perception_probe.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402

from simulation.config import SimConfig  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402


def ring2_of(world, cell: int) -> np.ndarray:
    """距离恰为 2 的格（跳数语义；与 L2 `_far_cells` 同构）。"""
    r1 = set(int(x) for x in world.neighbors(cell))
    r2 = set()
    for c in r1:
        r2.update(int(x) for x in world.neighbors(c))
    r2.discard(int(cell))
    r2 -= r1
    return np.array(sorted(r2), dtype=np.int64)


def main() -> None:
    cfg = SimConfig(seed=42)
    cfg.simulation.use_sim_core = False
    cfg.population.max_count = 3240
    e = SphereEngine(cfg)
    w = e.world
    print("=== A. _nb_table 的 stride vs 每格实际邻居数 ===")
    tab = e._nb_table
    n_actual = (tab >= 0).sum(axis=1)
    print(f"  stride = _nb_table.shape[1] = {tab.shape[1]}   (nb_stride = max(8, cols={w.cols}))")
    print(f"  实际邻居数：min={n_actual.min()} max={n_actual.max()} "
          f"众数={np.bincount(n_actual).argmax()} 唯一值={sorted(set(n_actual.tolist()))[:6]}")
    print(f"  ⇒ 引擎用 shape[1]={tab.shape[1]} 作 `nb_max` 归一化 `densities`；"
          f"而普通格只有 8 邻 ⇒ **社交项被除以 {tab.shape[1]} 而非 8（弱化 {tab.shape[1] / 8:.1f} 倍）**")
    print(f"  ⇒ 这一点是**既有**行为（基线含它）：C9「文档声明≠代码实际」—— 注释写"
          f"「归一化 bincount(0~8)」但实际除以 {tab.shape[1]}")

    print()
    print("=== B. span=2 候选集规模（ring1 ∪ ring2，跳数语义）===")
    sizes1, sizes2 = [], []
    rows = [0, 1, 2, 15, 29, 30, 31, 45, 58, 59]
    for r in rows:
        for c in (0, 30, 60, 90, 119):
            if c >= w.cols:
                continue
            cell = int(w.rc_to_flat(np.int64(r), np.int64(c)))
            n1 = len(w.neighbors(cell))
            n2 = n1 + len(ring2_of(w, cell))
            sizes1.append((r, c, n1))
            sizes2.append((r, c, n2))
    print(f"  {'row':>4}{'col':>5}{'ring1':>7}{'ring1+2':>9}")
    for (r, c, n1), (_, _, n2) in zip(sizes1, sizes2):
        flag = "  ← 极区/邻带" if n1 > 8 else ""
        print(f"  {r:>4}{c:>5}{n1:>7}{n2:>9}{flag}")
    a1 = np.array([x[2] for x in sizes1]); a2 = np.array([x[2] for x in sizes2])
    print(f"  ⇒ ring1: 中位 {int(np.median(a1))}｜ring1+2: 中位 {int(np.median(a2))} "
          f"max {a2.max()}")
    print(f"  ⇒ 🔴 极区会把 stride 从 {tab.shape[1]} 推高到 {a2.max()}（若按 span=2 重建表）"
          f" ⇒ `nb_max` 会从 {tab.shape[1]} 变成 {a2.max()} ⇒ **densities 再被压小 "
          f"{a2.max() / tab.shape[1]:.2f} 倍**（静默！）")

    print()
    print("=== C. 社交项的实际量级（densities = occ/nb_max*smw）===")
    P = len(e._id)
    occ = np.bincount(e._flat[:P], minlength=w.n_cells)
    smw = float(cfg.simulation.social_move_weight)
    nb_max = float(tab.shape[1])
    dens = occ / nb_max * smw
    nz = dens[occ > 0]
    print(f"  social_move_weight = {smw}｜nb_max = {nb_max:.0f}")
    print(f"  densities：非零格 n={nz.size}｜mean={nz.mean():.6f}｜max={nz.max():.6f}")
    print(f"  感知项典型量级（perc=0.5 档、fr≈0.5）≈ {0.5 * 0.5 * 0.5:.4f}")
    print(f"  ⇒ 社交项/感知项 ≈ {nz.mean() / (0.5 * 0.5 * 0.5):.4f}"
          f"（按 nb_max=8 归一化则 ≈ {nz.mean() * nb_max / 8 / (0.5 * 0.5 * 0.5):.4f}）")
    print(f"  ⇒ 即：社交项当前比「按 8 邻归一化」弱 {nb_max / 8:.0f} 倍；"
          "g13「仅弱群居」的观测与之一致")


    print()
    print("=== D. 全球质量账（供 2.5「吃>长」复核）===")
    _rc = getattr(cfg, "resources", None)
    cpa = float(getattr(_rc, "capacity_per_area", 40.0))
    cap_arr = np.asarray(e.resources._capacity, dtype=np.float64)
    area = cap_arr / cpa
    print(f"  Σcell_area = {area.sum():.1f}｜capacity_per_area = {cpa}")
    print(f"  {'方案':<26}{'总再生':>10}{'总需求':>10}{'裕度':>9}")
    n_now = 1944          # 饱和构造值（软顶 0.6 x 3240）
    for tag, ea, rg in (("现状 0.5/0.5", 0.5, 0.5), ("建议 0.6/0.4", 0.6, 0.4),
                        ("0.7/0.35", 0.7, 0.35), ("0.7/0.3", 0.7, 0.3)):
        tot_r = area.sum() * rg
        tot_d = n_now * ea
        print(f"  {tag:<26}{tot_r:>10.1f}{tot_d:>10.1f}"
              f"{(tot_r / tot_d - 1.0) * 100:>8.0f}%")
    print(f"  赤道满格吃空时间（capacity={cpa}/净耗）= "
          f"{cpa / (0.6 - 0.4):.0f} tick（建议档）")


if __name__ == "__main__":
    main()
