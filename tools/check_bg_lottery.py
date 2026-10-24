# -*- coding: utf-8 -*-
"""🔴 R320 回归守卫：**背景产能抽奖**自检（R300 缺陷的快速判据）。

**缺陷**（`world/resource_dynamics.py:161` + `:222`，**已在 R328 甲案修复**）：
`ResourceDynamics` 判断"背景产能是否为零"时曾**不读配置里的显式字段**，而是从
`cap[~patch_mask][0]`（背景的**第 0 个格**）反推 `bg_capacity_mult`。若装置带低产能带
（`bg_low_prod_frac > 0`），那 40% 被抽中的背景格会让这个"第 0 格"偶发变成低产能格
⇒ `bg_production_zero` 被误判成 False ⇒ **整个世界从"纯斑块"退化成"全图可产"**
（= A/B 支随机分化）。

**R328 甲修复**：`from_field` 直读显式真值 `rfield.bg_production_zero`（True ⇒
`b_mult = 0.0` 确定性"纯斑块"）⇒ 带低产能带的档**也不再抽奖**。

**本脚本**：只**构造世界**（不跑模拟，秒级），对一组 seed 逐个报
`ResourceDynamics.bg_production_zero`，从而直接回答"抽奖还在不在"。

用法：
    python.exe tools\\check_bg_lottery.py                 # 默认装置（0.0 档）应 0 例
    python.exe tools\\check_bg_lottery.py --seeds 165-172,189-192
    python.exe tools\\check_bg_lottery.py --bg-low-prod-frac 0.4
        # 旧档参数：**修复前**约 40% 判成 B 支（R300 复现口径）；
        # **R328 甲修复后恒 0 例**（确定性 A 支）⇒ 本模式 = 甲修复的回归门。

判据：**任何装置档下都必须 0 例被判成"全图可产"**（含非零档——甲修复后带档
也已是确定性"纯斑块"）；出现任何一例 ⇒ 装置档被改回、甲修复被回退、或出现新抽奖路径。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from experiments.s3_memory_probe import S1_BASE  # noqa: E402
from experiments.steady_k_probe import apply_post_build, make_cfg  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402
from tools.device_presets import DEVICE_PRESETS  # noqa: E402
from world.resource_dynamics import ResourceDynamics  # noqa: E402


def parse_seeds(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> int:
    d = DEVICE_PRESETS["s2"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="165-172,189-192")
    ap.add_argument("--bg-low-prod-frac", type=float, default=None,
                    help="覆盖装置档（用于复现旧档抽奖）；默认用装置档的值")
    ap.add_argument("--bg-low-cap-mult", type=float, default=None)
    a = ap.parse_args()

    frac = d["bg_low_prod_frac"] if a.bg_low_prod_frac is None else a.bg_low_prod_frac
    mult = d["bg_low_cap_mult"] if a.bg_low_cap_mult is None else a.bg_low_cap_mult
    seeds = parse_seeds(a.seeds)

    print(f"# 装置档 bg_low_prod_frac={frac} bg_low_cap_mult={mult} ｜ {len(seeds)} 个 world")
    bad = []
    for s in seeds:
        c, n = make_cfg(s, d["rows"], d["cols"], d["pop"], d["patches"], True,
                        S1_BASE["speed_max"], S1_BASE["gain"], S1_BASE["subdiv"],
                        S1_BASE["k"], S1_BASE["max_count"], d["rgm"],
                        bg_low_prod_frac=frac, bg_low_cap_mult=mult)
        c.simulation.use_sim_core = False
        c.resource_dynamics.enabled = True
        e = SphereEngine(c)
        apply_post_build(e, n)
        rd = ResourceDynamics.from_field(e.config.resource_dynamics,
                                         e.resources, e.world)
        pm = np.asarray(e.resources._patch_mask)
        cap = np.asarray(e.resources._capacity)
        bg_nz = int(((~pm) & (cap > 1e-12)).sum())
        tag = "零背景" if rd.bg_production_zero else "🔴全图可产(B支)"
        if not rd.bg_production_zero:
            bad.append(s)
        print(f"  seed={s:<5d} bg_production_zero={rd.bg_production_zero!s:<5s} "
              f"背景非零格={bg_nz:<7d} {tag}")

    print(f"\n⇒ {len(bad)}/{len(seeds)} 个 world 被判成『全图可产』"
          f"{'（seed ' + str(bad) + '）' if bad else ''}")
    if bad:
        print("❌ **抽奖回来了** —— 装置档或代码被改动，请立即排查。"
              "（R328 甲修复后：无论是否带低产能带，bg_production_zero 均须直读显式真值。）")
        return 1
    if frac == 0.0:
        print("✅ 抽奖已消除（背景格全零 ⇒ 判据恒为 True）。")
    else:
        print("✅ 甲修复（R328）生效：带低产能带的档也为确定性『零背景』（A 支）——"
              "单格抽奖已消失。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
