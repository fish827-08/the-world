"""J1 映射件的可复现自证：甲尺度合成原生场 → 映射 → loader → 15 条逐条可算性（不用 A1 数据）。

两臂（唯一差别 = σ 阶梯里换一层）：
  A = 甲默认 σ（0.0625…0.9375 ⇒ p=σ·ps0 无 200–250 hPa 层）
  B = 把 σ=0.1875 换成 0.22（其余全同）
用途：冻结文档 §5 表的每个"可算/不可算"都必须能被第三方一键复现。场是解析构造的
合成值 ⇒ 判据读数（HS-JET-U=29.97 之类）**无物理意义**，本件只回答"管线能不能算"。

用法::

    python -m benchmarks.atmo.jia_smoke --out _rerun_logs/20261021-AV-j1-mapping
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from benchmarks.atmo import jia_remap, loader, suites

NRING, NLAT_HALF, REACH, NPTS = 48, 24, 87.1591, 3168
SIGMA_A = [0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375]
SIGMA_B = [0.0625, 0.2200, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375]


def build_native(nd: Path, sigma, seed=11, n_snap=6):
    """甲 T=32 / L8 的环结构复刻（48 环、20…112…20、总 3168 点、纬度触达 ±87.1591°、无极点）。"""
    ramp = 20 + 4 * np.arange(NLAT_HALF)
    counts = np.concatenate((ramp, ramp[::-1]))
    if counts.sum() != NPTS:
        raise AssertionError("合成网格复刻与甲接口事实（3168 点）不符")
    mu = np.linspace(-math.sin(math.radians(REACH)), math.sin(math.radians(REACH)), NRING)
    lat = np.repeat(np.rad2deg(np.arcsin(mu)), counts).astype(np.float32)
    lon = np.concatenate([np.arange(n) * (360.0 / n) for n in counts]).astype(np.float32)
    nd.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n_snap):
        la = lat.astype(np.float64)
        prof = np.exp(-((np.abs(la) - 40.0) / 18.0) ** 2)
        ln_ps0 = math.log(jia_remap.PROTOCOL["ps0_Pa"])  # 甲磁盘约定：ps 存 ln(Pa)
        np.savez(nd / f"snap_{i + 1:05d}.npz",
                 u=(30.0 * prof[:, None] + 3.0 * rng.standard_normal((NPTS, 8))).astype(np.float32),
                 v=(2.0 * rng.standard_normal((NPTS, 8))).astype(np.float32),
                 T=((300.0 - 0.6 * np.abs(la))[:, None] + 2.0 * rng.standard_normal((NPTS, 8))).astype(np.float32),
                 ps=(ln_ps0 + 1e-4 * rng.standard_normal(NPTS)).astype(np.float32),
                 lat=lat, lon=lon)
    mf = {
        "benchmark": "held_suarez", "days": 30.0, "dt_atmo_s": 900.0,
        "snap_days": [5.0 * (i + 1) for i in range(n_snap)], "obliquity_deg": 0.0,
        "seed": 11, "config_hash": "synthetic-jia-smoke", "cfl_reported": [0.4] * n_snap,
        "grid": {"npoints": NPTS, "nlayers": 8, "sigma": sigma, "truncation": 32},
        "export": {"dtype": "float32", "ps_units": "log(Pa)",
                   "field_shape": "npoints,nlayers", "byte_order": "little"},
        "ic": {"name": "iso_w1", "sha256": "0" * 64},
        "env": {"julia": "1.13.2", "speedyweather": "0.23.0", "nthreads": 4},
        "forcing": {**suites.HS94_PARAMS, **jia_remap.PROTOCOL},
    }
    (nd / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")


def suite_probe(outdir, outdir_b=None):
    try:
        res = suites.evaluate(suites.HELD_SUAREZ, outdir, outdir_b)
    except Exception as e:  # noqa: BLE001 - "谁把套件打断"正是要测的事实
        return f"{type(e).__name__}: {e}", {}, None
    return None, {r["id"]: (r["status"], r["measured"]) for r in res["rows"]}, res["verdict"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="av-jia-smoke")
    ap.add_argument("--out", default="_rerun_logs/20261021-AV-j1-mapping")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {}
    for arm, sigma in (("A_default_sigma", SIGMA_A), ("B_sigma_0p22", SIGMA_B)):
        nd = out / f"native_{arm}"
        build_native(nd, sigma)
        t0 = time.perf_counter()
        c = jia_remap.convert_native_run(nd, out / f"avrun_{arm}")
        secs = time.perf_counter() - t0
        od = out / f"avrun_{arm}"
        if arm == "B_sigma_0p22":
            jia_remap.convert_native_run(nd, out / f"avrun_{arm}_b")
            err, rows, verdict = suite_probe(od, out / f"avrun_{arm}_b")
        else:
            err, rows, verdict = suite_probe(od)
        mb = sum(p.stat().st_size for p in od.glob("snap_*.npz")) / 1e6
        print(f"[{arm}] convert {secs:.2f} s/6 快照  self_check_bad={c['self_check_bad']}  "
              f"mu_defect={c['mu_coverage_defect']:.2e}  npz={mb:.1f} MB")
        print(f"  target {c['target']['nlat']}x{c['target']['nlon']}  "
              f"纬度触达 {c['target']['mu_cover_deg']:.4f}°  原生 CFL {c['cfl_native_max']:.4f}")
        if err:
            print(f"  SUITE ABORT: {err}  ⇒ 已记录判据 0/{len(suites.HELD_SUAREZ.criteria_ids)}")
        else:
            print(f"  verdict={verdict}  rows={len(rows)}/15")
            for cid in suites.HELD_SUAREZ.criteria_ids:
                st, ms = rows.get(cid, ("MISSING", None))
                print(f"    {cid:16s} {st:6s} measured={ms}")
        rep[arm] = {"convert_seconds": secs, "target": c["target"], "rings": c["rings"],
                    "cfl_native_max": c["cfl_native_max"], "self_check_bad": c["self_check_bad"],
                    "abort": err, "verdict": verdict, "snap_mb": mb,
                    "rows": {k: [v[0], str(v[1])] for k, v in rows.items()}}
    (out / "smoke.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
