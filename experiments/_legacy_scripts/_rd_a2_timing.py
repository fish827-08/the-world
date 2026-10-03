# -*- coding: utf-8 -*-
"""rd-A 交付：旗舰装置配对计时 + 终态 digest（窗口内口径，两树同脚本）。

装置 = 旗舰 off 臂档（make_cfg + apply_post_build；无 memory_v2），rd on/off 与
sparse on/off 由 CLI 指定。seed=101（★ A 支：b_mult=0 ⇒ rd.bgzero=True，sparse∧rd 放行）。

⚠️ 本机有 4 路 60k 批跑叠加 ⇒ 绝对 ms/t 不可比"静默"；**同窗口 ABBA 配对比值**为准。
⚠️ 4 个变体（main/wt × s0/s1）在**逐位等价**设计下，t=900 终态 digest 应**全部相同**
   ⇒ 每次运行自带一次旗舰级逐位交叉验证。

用法：<py> _trash_local/_rd_a2_timing.py --sparse 0|1 [--rd 1|0] --tag <name> --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

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
    d = {
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
    pc = getattr(e, "_patch_centroid", None)
    if pc is not None:
        d["centroid"] = sha(pc)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse", type=int, default=0)
    ap.add_argument("--rd", type=int, default=1)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--warm", type=int, default=800)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tk", type=int, default=20)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    c, notes = make_cfg(a.seed, 480, 960, 10000, 1700, True, 0.125, 0.125, 80, 2.5,
                        30000, 1.195, bg_low_prod_frac=0.4, bg_low_cap_mult=0.05)
    c.simulation.use_sim_core = False
    c.resource_dynamics.enabled = bool(a.rd)
    c.simulation.sparse_fields = bool(a.sparse)
    e = SphereEngine(c)
    apply_post_build(e, notes)

    diag = {
        "tag": a.tag, "tree": os.path.basename(ROOT), "seed": a.seed,
        "sparse_cli": a.sparse, "rd_cli": a.rd,
        "rd_on": bool(e._rd.enabled), "rd_bgzero": bool(e._rd.bg_production_zero),
        "lazy": bool(getattr(e.resources, "_lazy", False)),
        "sparse_engine": bool(getattr(e, "_sparse_fields", False)),
    }

    t0 = time.perf_counter()
    for _ in range(a.warm):
        if e.extinct:
            break
        e.step()
    diag["warm_wall_s"] = round(time.perf_counter() - t0, 1)

    reps_ms = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        for _ in range(a.tk):
            if e.extinct:
                break
            e.step()
        reps_ms.append((time.perf_counter() - t0) / a.tk * 1e3)

    diag["N"] = int(len(e._flat))
    diag["extinct"] = bool(e.extinct)
    diag["reps_ms"] = [round(x, 3) for x in reps_ms]
    diag["ms_per_tick"] = round(float(np.median(reps_ms)), 3)
    diag["ms_min"] = round(float(np.min(reps_ms)), 3)
    if a.rd:
        diag["dead"] = int(e._rd._dead.sum())
        diag["resting"] = int((e._rd._rest_until >= 0).sum())
    diag["digest"] = digest(e)

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in diag.items() if k != "digest"},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
