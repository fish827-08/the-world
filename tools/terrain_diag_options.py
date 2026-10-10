#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3 判据风险 —— 决策诊断 ①（2026-09-24 入库 `tools/`，供 [所有者] 复核 / 重跑）。

历史：本件原为"只跑不写"的一次性决策诊断（当时不入仓）；S1/S2 回板把
「bgzero 下 N 天然 5–95 ⇒ S3 判据可能无区分度（B5）」列为待裁风险 ⇒
按 R194（证据可复核）入库。**逻辑未改**，仅：ROOT 由硬编码改为仓库相对 + 注释/用法更新。

问题：`bgzero` regime 下三地形 N 天然低位（森林/荒漠 6k 不恢复）。
三种候选修法的可行性，取决于"低 N 到底是 **容量限** 还是 **建群限**"：

  arm A `forest@200`   森林几何 + 现有初始投放（200）
  arm B `forest@600`   森林几何 + **3× 初始投放**（= "提高初始投放"选项的直接检验）
  arm C `ctrl@200`     现状几何（30/2）+ 现有初始投放（= 既有 C 臂参照）

若 B 的**平台**显著高于 A ⇒ 建群限（提高投放有效）；
若 B 与 A 平台相同 ⇒ 容量限（投放只买来一段瞬态，改 K / 改判据才有效）。

用法：
  python tools/terrain_diag_options.py [ticks]          # 默认 4000

输出样例（seed42 / 4k）：`_rerun_logs/terrain_s2_smoke/diag_option_analysis.txt`
（sha256 见证据包 `_archive/2026-10-10-退役团队-归档/_audit/S3-pre/SHA256.txt`）
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from experiments.a4_verify_capacity import SpatialReadings  # noqa: E402
from simulation.config import InfoStructureConfig, SimConfig  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402


def make(seed: int, *, patch_count: int, patch_radius: int, patch_mult: float,
         initial: int) -> SphereEngine:
    """13.5 全开（bgzero + ③）+ 指定地形几何（= 三地形 preset 的配置）。"""
    c = SimConfig(seed=seed)
    c.simulation.use_sim_core = False
    c.population.max_count = 3240
    c.population.initial_count = int(initial)
    c.population.soft_cap_target = 0.6
    c.predation.forage_tradeoff_k = 0.0
    c.info_structure = InfoStructureConfig(enabled=True, learning_rate=0.05,
                                           memory_gradient="none")
    c.organisms.energy_cap_enabled = True
    c.organisms.photo_max = 0.0
    c.organisms.eat_efficiency = 7.5
    c.organisms.assim_herb = 0.4
    c.organisms.assim_carn = 0.8
    c.organisms.stomach_cap_mass = 25.0
    c.organisms.eat_threshold_frac = 0.6
    c.organisms.starve_frac = 0.30
    c.organisms.exhaust_frac = 0.17
    c.organisms.max_energy = 600.0
    c.organisms.initial_energy = 300.0
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = True
    c.resources.patch_regrowth_mult = float(patch_mult)
    c.resources.patch_count = int(patch_count)
    c.resources.patch_radius = int(patch_radius)
    return SphereEngine(c)


def run(tag: str, seed: int, ticks: int, **kw) -> dict:
    e = make(seed, **kw)
    sr = SpatialReadings(e)
    out = {"tag": tag, "traj": []}
    for t in range(1, ticks + 1):
        if e.extinct:
            break
        e.step()
        sr.observe(e)
        if t % 1000 == 0:
            s = sr.sample(e, t)
            out["traj"].append((t, len(e._id),
                                None if s["food_util_frac"] is None else round(s["food_util_frac"], 5),
                                None if s["patch_stock_frac"] is None else round(s["patch_stock_frac"], 4),
                                None if s["on_patch_frac"] is None else round(s["on_patch_frac"], 4),
                                None if s["patch_visit_frac"] is None else round(s["patch_visit_frac"], 4),
                                None if s["gud_var"] is None else round(s["gud_var"], 5)))
    out["final_N"] = len(e._id)
    out["extinct"] = bool(e.extinct)
    out["dc"] = {str(k).split(".")[-1]: int(v) for k, v in e.death_cause_totals().items()}
    out["sigma_regen"] = round(sr.sigma_regen, 1)
    return out


ARMS = [
    ("forest@200", dict(patch_count=12, patch_radius=3, patch_mult=1.62, initial=200)),
    ("forest@600", dict(patch_count=12, patch_radius=3, patch_mult=1.62, initial=600)),
    ("ctrl@200", dict(patch_count=30, patch_radius=2, patch_mult=1.195, initial=200)),
]

if __name__ == "__main__":
    TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    print(f"{'arm':11s} {'Σ再生':>7s} {'t=1k':>5s} {'2k':>4s} {'3k':>4s} {'4k':>4s} "
          f"{'末N':>5s} {'灭绝':>4s}  死因(饿/捕/老)")
    for tag, kw in ARMS:
        r = run(tag, 42, TICKS, **kw)
        ns = [x[1] for x in r["traj"]] + [r["final_N"]]
        dc = r["dc"]
        print(f"{tag:11s} {r['sigma_regen']:7.1f} " + " ".join(f"{v:>4d}" for v in ns[:4+1])
              + f"  {r['final_N']:>4d} {str(r['extinct']):>4s}  "
              f"{dc.get('STARVATION',0)}/{dc.get('PREDATION',0)}/{dc.get('OLD_AGE',0)}")
        print("            读数: " + " | ".join(
            f"t{t}: N={n}, util={u}, stock={st}, on_patch={op}, visit={v}, gud_var={gv}"
            for (t, n, u, st, op, v, gv) in r["traj"]))