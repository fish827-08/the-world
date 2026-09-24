#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3 判据风险 —— 决策诊断 ②（2026-09-24 入库 `tools/`）：多 seed 复核。

（历史：原为"只跑不写"的一次性诊断；按 R194 入库，逻辑未改，仅路径/注释更新。
  ⚠️ 另修一处**传了没生效**（B3 家族）：原脚本在 import 前把 `sys.argv` 截断成
  `[argv[0]]`，导致 `[ticks]` 参数被吞、恒跑 3000（实测传 60 仍 3000）
  ⇒ 已改为"**先取后截断**"。）

问题：`forest@200` 在 seed42 平台 ≈11，而 `forest@600` 涨到 263、`ctrl@200` 涨到 370。
两种解释必须分开：
  (i) **建群限**（初始投放少 ⇒ 长期低位） ⇒ "提高初始投放"有效
  (ii) **多重稳态 / seed 抽签**（R190 已记录：同配置 C 臂 4 seed = 14/103/295/406）
       ⇒ 提高投放只是**换抽签**，不分可性（判据风险仍在）

判法：把 `forest@200` 换 seed 跑（7/11/13）——若有的 seed 能到几百 ⇒ (ii) 主导。

用法：
  python tools/terrain_diag_multiseed.py [ticks]        # 默认 3000

输出样例（3k）：`_rerun_logs/terrain_s2_smoke/diag_multiseed.txt`
（sha256 见证据包 `_audit/S3-pre/SHA256.txt`）
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_USAGE_TICKS = sys.argv[1] if len(sys.argv) > 1 else None   # 先取（下方截断：防误读）
sys.argv = [sys.argv[0]]          # 防误读
from tools.terrain_diag_options import run  # noqa: E402

FOREST = dict(patch_count=12, patch_radius=3, patch_mult=1.62, initial=200)
CTRL = dict(patch_count=30, patch_radius=2, patch_mult=1.195, initial=200)

if __name__ == "__main__":
    TICKS = int(_USAGE_TICKS) if _USAGE_TICKS and _USAGE_TICKS.isdigit() else 3000
    print(f"=== forest@200 / ctrl@200 多 seed（{TICKS} tick）===")
    print(f"{'arm':10s} {'seed':>4s} {'t1k':>5s} {'t2k':>5s} {'t3k':>5s} {'末N':>5s} {'灭绝':>5s}")
    for tag, kw in (("forest", FOREST), ("ctrl", CTRL)):
        for seed in (42, 7, 11, 13):
            r = run(f"{tag}@{seed}", seed, TICKS, **kw)
            ns = [x[1] for x in r["traj"]]
            pad = ns + [""] * (3 - len(ns))
            print(f"{tag:10s} {seed:>4d} {str(pad[0]):>5s} {str(pad[1]):>5s} {str(pad[2]):>5s} "
                  f"{r['final_N']:>5d} {str(r['extinct']):>5s}")