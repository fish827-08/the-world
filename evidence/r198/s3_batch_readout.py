#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""R198 / 13.8 S3 批产 CSV 的**种群级**读数（§5.2 必报项）。

🔴 边界（R196）：
  · 本脚本**只出种群级描述**（世代列 / 种群纬度质心 / polar_frac / N 轨迹）；
  · **判据**（个体级 ρ(Δ|φ|,g23)）**不在这里** —— 在 `s3_judge.py`（需逐个体 _id+基因，
    批产 CSV 不含 ⇒ 见 s3_judge 模块 docstring §5.1）。
  · 本脚本的 `mean_row` 轨迹**仅作描述与自证（防"质心伪造"）**，**不得**当判据。
"""
from __future__ import annotations

import glob
import os
import sys
import csv
import json
import math
import numpy as np

ARMS = {
    "mig_base":     "_rerun_logs/mig_base/base_s*.csv",
    "mig_g":        "_rerun_logs/mig_g/g20_s*.csv",
    "mig_2g":       "_rerun_logs/mig_2g/g40_s*.csv",
    "mig_noseason": "_rerun_logs/mig_noseason/noseason_s*.csv",
}
ROOT = "/tmp/tw"


def _f(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows


def main():
    print(f"{'臂':14s} {'seed':>5s} {'末tick':>7s} {'末N':>6s} {'末世代':>7s} "
          f"{'mean_row初':>10s} {'mean_row末':>10s} {'Δmean_row':>10s} {'polar终':>8s}")
    out = {}
    for arm, pat in ARMS.items():
        files = sorted(glob.glob(os.path.join(ROOT, pat)))
        if not files:
            print(f"{arm:14s}  （无产物）")
            continue
        per = []
        for p in files:
            seed = os.path.basename(p).split("_s")[-1].replace(".csv", "")
            r = load(p)
            if not r:
                continue
            t = [int(x["tick"]) for x in r]
            N = [_f(x.get("N")) for x in r]
            gen = [_f(x.get("max_gen_cur")) for x in r]
            mr = [_f(x.get("mean_row")) for x in r]
            pf = [_f(x.get("polar_frac")) for x in r]
            mr0, mr1 = mr[0], mr[-1]
            dmr = (mr1 - mr0) if (mr0 is not None and mr1 is not None) else None
            per.append(dict(seed=seed, t_end=t[-1], N=N[-1], gen=gen[-1],
                            mr0=mr0, mr1=mr1, dmr=dmr, pf=pf[-1]))
            print(f"{arm:14s} {seed:>5s} {t[-1]:>7d} {N[-1]:>6.0f} {gen[-1]:>7.0f} "
                  f"{mr0:>10.4f} {mr1:>10.4f} {dmr:>+10.4f} {pf[-1]:>8.4f}")
        if per:
            d = [x["dmr"] for x in per if x["dmr"] is not None]
            out[arm] = dict(
                n_seed=len(per),
                dmr_mean=float(np.mean(d)) if d else None,
                dmr_std=float(np.std(d)) if d else None,
                N_end_mean=float(np.mean([x["N"] for x in per if x["N"] is not None])),
                gen_end_mean=float(np.mean([x["gen"] for x in per if x["gen"] is not None])),
            )
    print("\n=== 臂级汇总（种群级描述，非判据）===")
    for a, v in out.items():
        print(f"{a:14s} seeds={v['n_seed']}  Δmean_row={v['dmr_mean']:+.4f}"
              f"±{v['dmr_std']:.4f}  末N均值={v['N_end_mean']:.0f}"
              f"  末世代均值={v['gen_end_mean']:.1f}")

    # ---- migration_probe 读数（§5.2 必报：mig_flat_frac / P1–P7 复测 / g23 相关）----
    print("\n=== migration_probe 读数（来自 *.summary.json，§5.2 必报）===")
    probe_arms = {}
    for arm in ARMS:
        sj = sorted(glob.glob(os.path.join(
            ROOT, ARMS[arm].replace("*.csv", "*.summary.json"))))
        if not sj:
            continue
        probes = []
        for p in sj:
            try:
                d = json.load(open(p, encoding="utf-8"))
            except Exception as exc:      # 🔴 不静默：报出文件与原因（否则会掩盖真实 bug）
                print(f"  ⚠️ 读取失败 {os.path.basename(p)}: {type(exc).__name__}: {exc}")
                continue
            mg = (d.get("result") or {}).get("migration")
            if mg:
                probes.append(mg)
        if probes:
            def _m(key):
                xs = [x.get(key) for x in probes if isinstance(x.get(key), (int, float))]
                return float(np.mean(xs)) if xs else None
            probe_arms[arm] = {
                "n_run": len(probes),
                "mig_term_abs_mean": _m("mig_term_abs_mean"),
                "mig_flat_frac": _m("mig_flat_frac"),
                "mig_zero_frac": _m("mig_zero_frac"),
                "mig_skip_frac": _m("mig_skip_frac"),
                "pp_anom_min": min((x.get("pp_anom_min") for x in probes
                                    if x.get("pp_anom_min") is not None), default=None),
                "pp_anom_max": max((x.get("pp_anom_max") for x in probes
                                    if x.get("pp_anom_max") is not None), default=None),
                # 🔴 P2 的**有效**证据 = 运行期累积极值（末 tick 快照常因 δ=0 落 0）
                "pp_anom_lo_run": min((x.get("pp_anom_lo_run") for x in probes
                                       if x.get("pp_anom_lo_run") is not None), default=None),
                "pp_anom_hi_run": max((x.get("pp_anom_hi_run") for x in probes
                                       if x.get("pp_anom_hi_run") is not None), default=None),
                "lat_abs_max": probes[0].get("lat_abs_max"),
                "migrate_gene_slot": probes[0].get("migrate_gene_slot"),
                "migration_enabled": probes[0].get("migration_enabled"),
                "migration_gain": probes[0].get("migration_gain"),
            }
            v = probe_arms[arm]
            print(f"{arm:14s} n={v['n_run']} slot={v['migrate_gene_slot']} "
                  f"enabled={v['migration_enabled']} gain={v['migration_gain']} "
                  f"flat={v['mig_flat_frac']} zero={v['mig_zero_frac']} "
                  f"|term|={v['mig_term_abs_mean']} "
                  f"A_run∈[{v['pp_anom_lo_run']},{v['pp_anom_hi_run']}] "
                  f"lat_max={v['lat_abs_max']}")
        else:
            print(f"{arm:14s} （无 migration 读数：空臂/无季节臂 ⇒ probe=None，符合预期）")

    out["_migration_probe"] = probe_arms
    with open("/workspace/r198/s3_batch_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已写 /workspace/r198/s3_batch_summary.json")


if __name__ == "__main__":
    main()
