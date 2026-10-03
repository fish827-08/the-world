#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tier-0 斑块几何（零机时二次分析）—— T1 根因判别的前置。

用途（设计稿 `设计-R358-T1-根因判别-H1H2-20261003.md` §2.1 原文）：
    锁死"世界里到底有没有远格可踩"的**几何先验**，排除**平凡 H1**
    —— 若世界里根本没有远富食格 ⇒ H2 自动出局 ⇒ Tier-1 可省。

判据（T1 稿 §2.1，**天平已于 R363 锁定**）：
    跨斑块距离中位 **≥5 格** ⇒ 远格必然存在（判别得下去）
    **≤2 格** ⇒ 平凡 H1 ⇒ 停下回报

距离口径：复用 **`world.subpos.sphere_dist_rows`**（atan2 形式，极区自动正确）——
          与 P1-c 探针**同一函数** ⇒ 读数与 Tier-1 可比（不直译 9×9，符合 R148）。

用法：
    .venv/Scripts/python.exe experiments/t0_patch_geometry_probe.py
    .venv/Scripts/python.exe experiments/t0_patch_geometry_probe.py --snap <path.npz> [--snap ...]

作者：`[所有者·天平]`（零机时轻量分析；若轻舟已有实现以他的为准）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from world.subpos import sphere_dist_rows  # noqa: E402
from world.sphere_world import SphereWorld  # noqa: E402

DEFAULT_SNAPS = [
    "_trash_local/rd_a2_base_snap.npz",
    "_trash_local/rd_a2_new_snap.npz",
    "_trash_local/_p2_snap.npz",
    "_trash_local/_snap_invariant.npz",
    "the-world-data/_rerun_logs/cstep_snap_ext/oracle_m1.3_s46_ext.snapshot.npz",
]


def label_components(mask2d: np.ndarray, cols: int) -> list[tuple[float, float, int]]:
    """4-邻域连通域（**列方向环绕** = 球面拓扑；行方向不环绕）。

    返回 [(质心行, 质心列, 格数), ...]。
    无 scipy ⇒ 手写 BFS（斑块格仅约 1.3e4，足够快）。
    """
    rows, _ = mask2d.shape
    seen = np.zeros_like(mask2d, dtype=bool)
    out: list[tuple[float, float, int]] = []
    idx = np.argwhere(mask2d)
    for r0, c0 in idx:
        if seen[r0, c0]:
            continue
        stack = [(int(r0), int(c0))]
        seen[r0, c0] = True
        cells: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                if rr < 0 or rr >= rows:
                    continue
                cc = (c + dc) % cols  # 🔴 列环绕（球面）
                if mask2d[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        arr = np.asarray(cells, dtype=np.float64)
        out.append((float(arr[:, 0].mean()), float(arr[:, 1].mean()), len(cells)))
    return out


def analyse(path: Path) -> dict | None:
    if not path.exists():
        print(f"  [跳过] 不存在：{path}")
        return None
    d = np.load(path, allow_pickle=True)
    keys = set(d.keys())
    if "resource_patch_mask" not in keys:
        print(f"  [跳过] 无 resource_patch_mask：{path.name}")
        return None

    cfg_raw = d["config_dict"]
    cfg = json.loads(cfg_raw.item() if hasattr(cfg_raw, "item") else str(cfg_raw))
    rows = int(cfg.get("world", {}).get("rows", 480))
    cols = int(cfg.get("world", {}).get("cols", 960))
    seed = cfg.get("seed")

    mask = np.asarray(d["resource_patch_mask"], dtype=bool)
    if mask.size != rows * cols:
        # 容错：the-world 世界比例恒为 rows:cols = 1:2 ⇒ 按此推断
        cols = int(round(np.sqrt(mask.size * 2)))
        rows = mask.size // cols if cols > 0 else 0
        if rows <= 0 or rows * cols != mask.size:
            print(f"  [跳过] 尺寸无法推断：mask.size={mask.size}, cfg 声明 {rows}x{cols}")
            return None
        print(f"  [注意] 尺寸与 cfg 不符 ⇒ 按 1:2 推断 rows={rows} cols={cols}")

    mask2d = mask.reshape(rows, cols)
    comps = label_components(mask2d, cols)
    n_patch = len(comps)
    sizes = np.array([c[2] for c in comps], dtype=float)

    if n_patch < 2:
        return {"path": str(path), "seed": seed, "rows": rows, "cols": cols,
                "n_patch": n_patch, "note": "斑块 < 2 ⇒ 无跨斑块距离可算", "tick": int(d["tick"])}

    cen = np.array([[c[0], c[1]] for c in comps], dtype=np.float64)
    world = SphereWorld(rows, cols)

    # 两两球面距离（上三角）
    ri, ci = np.triu_indices(n_patch, k=1)
    dmat = sphere_dist_rows(cen[ri, 0], cen[ri, 1], cen[ci, 0], cen[ci, 1], world)
    dmat = np.asarray(dmat, dtype=np.float64)

    return {
        "path": str(path), "file": path.name, "seed": seed, "rows": rows, "cols": cols,
        "tick": int(d["tick"]), "count": int(d.get("count", -1)),
        "n_patch": n_patch,
        "patch_cells_total": int(mask.sum()),
        "patch_cells_median": float(np.median(sizes)),
        "n_pairs": int(dmat.size),
        "d_median": float(np.median(dmat)),
        "d_p10": float(np.percentile(dmat, 10)),
        "d_p25": float(np.percentile(dmat, 25)),
        "d_p90": float(np.percentile(dmat, 90)),
        "d_max": float(dmat.max()),
        "d_mean": float(dmat.mean()),
    }


def verdict(d_median: float | None) -> str:
    if d_median is None:
        return "不可判（无跨斑块距离）"
    if d_median >= 5.0:
        return "🟢 远格必然存在（≥5 格）⇒ 判别得下去"
    if d_median <= 2.0:
        return "🔴 平凡 H1（≤2 格）⇒ 停下回报"
    return f"🟡 灰区（2 < {d_median:.2f} < 5）⇒ 需结合 Tier-1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", action="append", default=None,
                    help="快照路径（可多次；默认跑 5 份既有素材）")
    args = ap.parse_args()
    snaps = [Path(s) for s in (args.snap or DEFAULT_SNAPS)]

    print("=" * 96)
    print("Tier-0 斑块几何（零机时）—— T1 根因判别前置：世界里有没有「远格」？")
    print("距离口径 = world.subpos.sphere_dist_rows（与 P1-c 探针同函数）")
    print("=" * 96)

    recs = []
    for s in snaps:
        p = s if s.is_absolute() else ROOT / s
        print(f"\n▸ {p.name}")
        r = analyse(p)
        if r:
            recs.append(r)
            if "d_median" in r:
                print(f"   装置 rows×cols={r['rows']}×{r['cols']} seed={r['seed']} tick={r['tick']} count={r['count']}")
                print(f"   斑块数 {r['n_patch']}｜斑块格 {r['patch_cells_total']}（中位 {r['patch_cells_median']:.1f} 格/斑块）")
                print(f"   🔴 跨斑块球面距离（{r['n_pairs']} 对）："
                      f"中位 **{r['d_median']:.3f}**｜p10 {r['d_p10']:.3f}｜p25 {r['d_p25']:.3f}｜"
                      f"p90 {r['d_p90']:.3f}｜max {r['d_max']:.3f}")
                print(f"   判读：{verdict(r['d_median'])}")
            else:
                print(f"   {r.get('note')}")

    meds = [r["d_median"] for r in recs if "d_median" in r]
    print("\n" + "=" * 96)
    if meds:
        pooled = float(np.median(meds))
        print(f"跨快照汇总：可判 {len(meds)} 份｜**跨斑块距离中位（再取中位）= {pooled:.3f} 格**")
        print(f"⇒ {verdict(pooled)}")
    else:
        print("无可判快照")
    print("=" * 96)

    out = ROOT / "_trash_local" / "t0_patch_geometry.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"读数已落：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
