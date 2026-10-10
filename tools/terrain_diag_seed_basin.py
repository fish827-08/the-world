#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3 判据风险 —— 决策诊断 ③（2026-09-24 入库 `tools/`）：`提高初始投放` 在**非增长 seed** 上是否有效。

（历史：原为"只跑不写"的一次性诊断；按 R194 入库，逻辑未改，仅路径/注释更新。）

已有事实：
  forest@200 四 seed 全低（13/11/48/5）；ctrl@200 只有 seed42 涨（258），其余 11–18
  ⇒ 低 N 与地形无关，是 13.5 regime 的**多重稳态**（低盆地 ≈5–50 / 高盆地 ≈250+）。
⇒ 于是"提高初始投放"必须单独验证：在**低盆地 seed**（13/7）上，600 投放能否把系统推进高盆地？
  · 能 ⇒ 投放是"换盆地"的有效手段（但注意：那是在**改变被观测系统**）
  · 不能 ⇒ 投放只放大瞬态，低盆地锁定与投放无关

用法：
  python tools/terrain_diag_seed_basin.py [ticks]       # 默认 3000

输出样例（3k）：`_rerun_logs/terrain_s2_smoke/diag_seed_basin.txt`
（sha256 见证据包 `_archive/2026-10-10-退役团队-归档/_audit/S3-pre/SHA256.txt`）
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.terrain_diag_options import run  # noqa: E402

FOREST600 = dict(patch_count=12, patch_radius=3, patch_mult=1.62, initial=600)
CTRL600 = dict(patch_count=30, patch_radius=2, patch_mult=1.195, initial=600)

if __name__ == "__main__":
    TICKS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3000
    print(f"=== 非增长 seed（13/7）× 初始投放 600（{TICKS} tick）===")
    print(f"{'arm':11s} {'seed':>4s} {'t1k':>5s} {'t2k':>5s} {'t3k':>5s} {'末N':>5s} {'灭绝':>5s} 死因(饿/捕/老)")
    for tag, kw in (("forest600", FOREST600), ("ctrl600", CTRL600)):
        for seed in (13, 7):
            r = run(f"{tag}@{seed}", seed, TICKS, **kw)
            ns = [x[1] for x in r["traj"]]
            pad = ns + [""] * (3 - len(ns))
            dc = r["dc"]
            print(f"{tag:11s} {seed:>4d} {str(pad[0]):>5s} {str(pad[1]):>5s} {str(pad[2]):>5s} "
                  f"{r['final_N']:>5d} {str(r['extinct']):>5s} "
                  f"{dc.get('STARVATION',0)}/{dc.get('PREDATION',0)}/{dc.get('OLD_AGE',0)}")