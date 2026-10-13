#!/usr/bin/env python3.12
"""S4 判据（零成本验证）：季节开 ⇒ 富集纬度带是否**确实随时间周期移动**？

只有环境（不跑生物演化）—— 直接构造引擎，每 tick 读 band_lat(t)，
检验它是否随 sin(2πt/P) 周期摆动、摆幅多大、相位对不对。

⚠️ 关键点：band_lat 的移动**依赖** `resources.temp_sensitivity`（资源再生对温度敏感）
   ⇒ 若 sensitivity=0，季节只改温度场但**不改资源** ⇒ band_lat 不动。
   本探针同时输出「光照/温度场」与「资源带」两个量，把二者区分开。

用法：
  python3.12 probe_band_movement.py --tilt 23.44 --season-period 6000 --ticks 18000 \
      --temp-sensitivity 1.0 --out /workspace/band_move.csv
"""
import argparse
import importlib.util
import math

import numpy as np

A4 = "/tmp/tw/experiments/a4_verify_capacity.py"


def load_build():
    spec = importlib.util.spec_from_file_location("a4mod", A4)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def band_lat_light(eng):
    """光照加权纬度带中心（纯环境，不依赖生物/资源）。"""
    illum = eng.light._ensure_cache(eng.tick) if hasattr(eng.light, "tick") else None
    # 直接取缓存
    eng.light._ensure_cache(int(eng.tick))
    ill = np.asarray(eng.light._cache_illum_all, dtype=float)
    rows, cols = eng.world.rows, eng.world.cols
    ill2 = ill.reshape(rows, cols)
    lat_rows = eng.world.latitude_of(np.arange(rows))
    row_ill = ill2.sum(axis=1)
    if row_ill.sum() <= 1e-12:
        return None
    w = row_ill * np.cos(lat_rows)
    return float((lat_rows * w).sum() / w.sum())


def band_lat_resource(eng):
    stock = np.asarray(eng.resources.snapshot(), dtype=float)
    if stock.ndim == 1:
        stock = stock.reshape(eng.world.rows, eng.world.cols)
    lat_rows = eng.world.latitude_of(np.arange(stock.shape[0]))
    row_stock = stock.sum(axis=1)
    if row_stock.sum() <= 1e-12:
        return None
    w = np.maximum(row_stock, 0.0) * np.cos(lat_rows)
    if w.sum() <= 1e-12:
        return None
    return float((lat_rows * w).sum() / w.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tilt", type=float, default=23.44)
    ap.add_argument("--season-period", type=int, default=6000)
    ap.add_argument("--ticks", type=int, default=18000)
    ap.add_argument("--log-interval", type=int, default=250)
    ap.add_argument("--temp-sensitivity", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mod = load_build()
    kw = dict(mode="on", codebook=True, seed=args.seed, ticks=args.ticks,
              max_count=3240, soft_cap_target=0.6, distribution="patchy",
              energy_cap=True, signal_alphabet="16", forage_tradeoff_k=0.0,
              measure=False,
              tilt_deg=float(args.tilt), season_period=int(args.season_period))
    eng = mod.build(**kw)
    cfg = eng.config
    if args.temp_sensitivity is not None:
        cfg.resources.temp_sensitivity = float(args.temp_sensitivity)
        from simulation.sphere_engine import SphereEngine
        eng = SphereEngine(cfg)
    print(f"[init] tilt={args.tilt} P={args.season_period} ticks={args.ticks} "
          f"season_on={eng.light._season_on} "
          f"temp_sens={cfg.resources.temp_sensitivity} rust={eng._use_sim_core}",
          flush=True)

    rows = []
    for t in range(0, args.ticks + 1, args.log_interval):
        # 快进到 t（从 0 开始，逐步 step）
        if t > 0:
            for _ in range(args.log_interval):
                eng.step()
        eng.light._ensure_cache(eng.tick)
        bl_env = band_lat_light(eng)
        bl_res = band_lat_resource(eng)
        rows.append((t, "" if bl_env is None else f"{math.degrees(bl_env):.4f}",
                     "" if bl_res is None else f"{math.degrees(bl_res):.4f}"))
        if t % (args.log_interval * 8) == 0:
            print(f"  t={t} band_env={rows[-1][1]} band_res={rows[-1][2]}", flush=True)

    with open(args.out, "w") as f:
        f.write("tick,band_env_deg,band_res_deg\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    env = np.array([float(r[1]) for r in rows if r[1] != ""])
    res = np.array([float(r[2]) for r in rows if r[2] != ""])
    print("\n=== 结果 ===")
    print(f"band_env : min={env.min():.3f} max={env.max():.3f} "
          f"sweep={env.max()-env.min():.3f}° std={env.std():.4f}")
    if len(res) > 0:
        print(f"band_res : min={res.min():.3f} max={res.max():.3f} "
              f"sweep={res.max()-res.min():.3f}° std={res.std():.4f}")
    # 理论摆幅：δ_max = tilt；纬度带中心在 -tilt..+tilt 间摆（近似）
    print(f"理论最大摆幅（±tilt）= {2*args.tilt:.2f}°")


if __name__ == "__main__":
    main()
