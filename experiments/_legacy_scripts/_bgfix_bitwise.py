# -*- coding: utf-8 -*-
"""R328 甲交付证据：**现档逐位零变化** —— 同一脚本在两棵树各跑一次，比终态 digest。

装置 = R326 新档（bg_low 0.0/0.0，纯零背景产能）+ rd 开 + sparse 由 CLI。
main 树 = 修复前代码；_wt_lq_bgfix = 修复后代码 ⇒ 两树 digest 必须**逐位相同**
（bgzero=True 时两法同值：旧推断 cap[~m][0]=0，新显式 b_mult=0.0）。

用法（两棵树各自运行，输出到同一目录便于对比）：
    <py> _trash_local/_bgfix_bitwise.py --seed 201 --sparse 0 \
        --tag main --out <abs>/_bgfix_bitwise/main_s201_sp0.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from experiments.steady_k_probe import make_cfg, apply_post_build  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _rng_sha(rng) -> str:
    st = rng.bit_generator.state
    return hashlib.sha256(
        json.dumps(st, sort_keys=True, default=str).encode()).hexdigest()


def digest(e) -> dict:
    n = e.world.n_cells
    rd = e._rd
    return {
        "t": int(e.tick),
        "n": int(len(e._flat)),
        "flat": int(e._flat.sum()),
        "ids": int(e._id.sum()),
        "e_energy": float(e._energy.sum()),
        "grid": sha(e.resources._grid),
        "cap": sha(e.resources._capacity),
        "pmask": sha(e.resources._patch_mask
                     if e.resources._patch_mask is not None
                     else np.zeros(n, dtype=bool)),
        "marks": sha(e.signals._marks),
        "age": sha(e.signals._age),
        "rd_mask": sha(rd._mask),
        "rd_dead": sha(rd._dead),
        "rd_rest": sha(rd._rest_until),
        "rd_dsince": sha(rd._dead_since),
        "rd_damage": sha(rd._damage),
        "rd_cnt": [int(rd.patch_kill_n), int(rd.patch_reborn_n),
                   int(rd.forced_reborn_n), int(rd.promote_n),
                   int(rd.rest_set_n)],
        "rd_rng": _rng_sha(rd._rng),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=201)
    ap.add_argument("--sparse", type=int, default=0)
    ap.add_argument("--warm", type=int, default=800)
    ap.add_argument("--ticks", type=int, default=1000)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    c, notes = make_cfg(a.seed, 480, 960, 10000, 1700, True, 0.125, 0.125, 80, 2.5,
                        30000, 1.195, bg_low_prod_frac=0.0, bg_low_cap_mult=0.0)
    c.simulation.use_sim_core = False
    c.resource_dynamics.enabled = True
    c.simulation.sparse_fields = bool(a.sparse)
    e = SphereEngine(c)
    apply_post_build(e, notes)

    for _ in range(a.warm):
        if e.extinct:
            break
        e.step()
    for _ in range(a.ticks):
        if e.extinct:
            break
        e.step()

    diag = {
        "tag": a.tag, "tree": os.path.basename(ROOT), "seed": a.seed,
        "sparse_cli": a.sparse, "ticks_total": a.warm + a.ticks,
        "N": int(len(e._flat)), "extinct": bool(e.extinct),
        "rd_on": bool(e._rd.enabled),
        "rd_bgzero": bool(e._rd.bg_production_zero),
        "rd_bg_mult": float(e._rd.bg_capacity_mult),
        "rd_moves": bool(e._rd.rotate_moves_patch),
        "lazy": bool(getattr(e.resources, "_lazy", False)),
        "sparse_engine": bool(getattr(e, "_sparse_fields", False)),
        "dead": int(e._rd._dead.sum()),
        "resting": int((e._rd._rest_until >= 0).sum()),
        "digest": digest(e),
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in diag.items() if k != "digest"},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
