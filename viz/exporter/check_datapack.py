#!/usr/bin/env python3
"""viz v0 数据包校验器（契约 viz/CONTRACT.md §8）。

用法：python viz/exporter/check_datapack.py <datapack_dir>
退出码：0=过；2=失败。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        print("用法: python viz/exporter/check_datapack.py <datapack_dir>", file=sys.stderr)
        return 2
    root = Path(argv[0])
    fails = []
    warns = []

    meta_p = root / "meta.json"
    if not meta_p.exists():
        print(f"[FAIL] 缺 meta.json: {root}")
        return 2
    meta = json.loads(meta_p.read_text(encoding="utf-8"))

    cv = str(meta.get("contract_version", ""))
    if not cv.startswith("v0"):
        fails.append(f"contract_version 非 v0: {cv!r}")
    rows = int(meta["world"]["rows"])
    cols = int(meta["world"]["cols"])
    n_frames = int(meta["tick"]["n_frames"])
    n_cells = int(meta["world"]["n_cells"])
    if rows * cols != n_cells:
        fails.append(f"rows*cols({rows * cols}) != n_cells({n_cells})")
    tk = meta.get("tick", {})
    if int(tk.get("end", -1)) != int(tk["start"]) + (n_frames - 1) * int(tk["stride"]):
        fails.append(
            f"tick 轴不自洽：end={tk.get('end')} != start + (n_frames-1)*stride"
            f" = {int(tk['start']) + (n_frames - 1) * int(tk['stride'])}"
        )

    for ch in meta.get("channels", []):
        key = ch["key"]
        d = root / "frames" / f"ch_{key}"
        if not d.is_dir():
            fails.append(f"缺帧目录 {d}")
            continue
        bins = sorted(d.glob("*.bin"))
        if len(bins) != n_frames:
            fails.append(f"ch_{key}: 帧文件数 {len(bins)} != n_frames {n_frames}")
        for i, b in enumerate(bins):
            if b.name != f"{i:05d}.bin":
                fails.append(f"ch_{key}: 帧命名不连续/不合法 -> {b.name}")
                break
        for b in bins:
            sz = b.stat().st_size
            if sz != n_cells:
                fails.append(f"ch_{key}/{b.name}: 帧长 {sz} != rows*cols {n_cells}")
                break
        if not ch.get("scale", {}).get("p_hi", 0) > ch.get("scale", {}).get("p_lo", 0):
            fails.append(f"ch_{key}: scale 无效 {ch.get('scale')}")

    ed = root / "entities"
    ent_cols = meta.get("entities", {}).get("columns", [])
    rec = 4 * (len(ent_cols) if ent_cols else 7)  # v0 契约 = 7 列 × f32
    sd = meta.get("entities", {}).get("subdiv")
    if sd is None:
        warns.append("meta.entities.subdiv 缺失（旧包 ⇒ 前端回退格中心；新包应带，契约 v0 修订1）")
    elif not (isinstance(sd, (int, float)) and int(sd) == sd and int(sd) >= 1):
        fails.append(f"meta.entities.subdiv 非法: {sd!r}")
        sd = None
    if not ed.is_dir():
        fails.append("缺 entities/ 目录")
    else:
        fs = sorted(ed.glob("*.f32"))
        if len(fs) != n_frames:
            fails.append(f"entities: 文件数 {len(fs)} != n_frames {n_frames}")
        for i, p in enumerate(fs):
            if p.name != f"{i:05d}.f32":
                fails.append(f"entities: 命名不连续 -> {p.name}")
                break
            if p.stat().st_size % rec != 0:
                fails.append(f"entities/{p.name}: 大小 {p.stat().st_size} 非 {rec} 倍数（{rec // 4}×f32）")
        for p in fs[: min(3, len(fs))]:
            if p.stat().st_size:
                a = np.fromfile(p, dtype="<f4")
                if not np.isfinite(a).all():
                    fails.append(f"entities/{p.name}: 含 NaN/Inf")
                if a.size >= 7:
                    flat = a[0::7]
                    if flat.size and (flat.min() < 0 or flat.max() >= n_cells):
                        fails.append(f"entities/{p.name}: flat 越界 [{flat.min()}, {flat.max()}]")
                    if sd is not None:
                        sr, sc = a[1::7], a[2::7]
                        if sr.size and (sr.min() < 0 or sr.max() >= rows * sd):
                            fails.append(
                                f"entities/{p.name}: sub_r 越界 [{sr.min()}, {sr.max()}]"
                                f"（应 ⊆ [0, {rows * int(sd)})，subdiv={int(sd)}）"
                            )
                        if sc.size and (sc.min() < 0 or sc.max() >= cols * sd):
                            fails.append(
                                f"entities/{p.name}: sub_c 越界 [{sc.min()}, {sc.max()}]"
                                f"（应 ⊆ [0, {cols * int(sd)})，subdiv={int(sd)}）"
                            )

    if not (root / "series.csv").exists():
        warns.append("无 series.csv（可选件）")

    for w in warns:
        print(f"[WARN] {w}")
    if fails:
        for f in fails:
            print(f"[FAIL] {f}")
        print(f"[viz-check] 失败 {len(fails)} 项：{root}")
        return 2
    print(f"[viz-check] OK：{root}｜帧 {n_frames}×通道 {len(meta.get('channels', []))}｜世界 {rows}x{cols}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
