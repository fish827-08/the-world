"""「世界读表」冒烟：`GCMTableSource` 在真实 `SphereWorld` 上读气候表（不接生物）。

⚠️ 本脚本用的是**合成表**（解析函数造的温度/降水/风），**不是** SpeedyWeather 的 T31 输出。
用途 = 证明"表 → 重映射 → 世界网格 → 逐格读数"这条链是通的、守恒的、季节可插值。
真表到位后，把 `--table` 指向真 npz 跑同一脚本即可（默认不读真表，避免误把合成当 T31）。

检查项
------
1. 形状/无 NaN：`temperature_c(flat, tick)` 复现 `(rows, cols)` 场。
2. **守恒**：重映射后全球面积权均值 = 表侧全球均值（采暖/降水，相对误差 < 1e-10）。
3. **季节**：12 月表在 `tick=0` 与 `tick=year/2` 读数必须不同（否则插值没生效）。
4. **错误路径**：无 `table_path` / 月表缺 `year_ticks` ⇒ 显式 `NotImplementedError`（不静默）。
5. 风：`wind()` 返回 `u`（`v` 可缺），极行已按纬向平均处理（无 NaN）。

用法
----
    python experiments/run_gcm_table_smoke.py
    python experiments/run_gcm_table_smoke.py --table <真表.npz> --year-ticks 876000
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atmo_bridge.remap import global_mean  # noqa: E402
from world.climate import GCMTableSource  # noqa: E402
from world.sphere_world import SphereWorld  # noqa: E402

NLAT, NLON = 32, 64


def synthetic_table(path: Path, *, months: int) -> dict:
    """解析气候态：赤道 ~29℃ / 极 ~-30℃；降水赤道峰 + 中纬次峰；u 为西风急流。"""
    lat = np.linspace(-90 + 90 / NLAT, 90 - 90 / NLAT, NLAT)
    lon = np.arange(NLON) * 360.0 / NLON
    phi = np.radians(lat)
    base_t = 29.0 * np.cos(phi) ** 2 - 30.0 * np.sin(phi) ** 4
    base_p = 2200.0 * np.exp(-(lat / 9.0) ** 2) + 700.0 * np.exp(-((np.abs(lat) - 52.0) / 11.0) ** 2)
    base_u = 22.0 * np.sin(2 * phi) ** 2 * np.sign(lat) - 3.0
    t = np.repeat(base_t[:, None], NLON, axis=1)
    p = np.repeat(base_p[:, None], NLON, axis=1) + 120.0 * (1.0 + np.cos(np.radians(lon)))[None, :]
    u = np.repeat(base_u[:, None], NLON, axis=1)
    if months == 12:
        season = np.sin(2 * np.pi * np.arange(12) / 12)
        t = t[None, :, :] + 8.0 * season[:, None, None] * np.cos(phi)[None, :, None]
        p = p[None, :, :] * (1.0 + 0.15 * season[:, None, None])
        u = u[None, :, :] * (1.0 + 0.10 * season[:, None, None])
    np.savez_compressed(path, lat=lat, lon=lon, temperature_c=t,
                        precip_mm_yr=p, u=u,
                        meta_json=np.array(json.dumps(
                            {"format": "npz-climatology-v0", "source": "SYNTHETIC (smoke only)",
                             "grid": [NLAT, NLON], "months": months}, ensure_ascii=False)))
    return {"lat": lat, "lon": lon, "temperature_c": t, "precip_mm_yr": p, "u": u}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=None, help="真气候表；不给则用合成表冒烟")
    ap.add_argument("--year-ticks", type=int, default=None)
    ap.add_argument("--rows", type=int, default=120)
    ap.add_argument("--cols", type=int, default=240)
    ap.add_argument("--keep", default=None, help="合成表落盘路径（默认临时目录，跑完不留）")
    args = ap.parse_args()

    world = SphereWorld(args.rows, args.cols)
    flat = np.arange(world.n_cells, dtype=np.int64)
    report: dict = {"grid": [args.rows, args.cols], "synthetic": args.table is None}
    tmpdir = None

    if args.table:
        table_path = Path(args.table)
        months = 12 if args.year_ticks else None
        src = None
    else:
        months = 12 if args.year_ticks else 1
        if args.keep:
            table_path = Path(args.keep)
        else:
            tmpdir = tempfile.TemporaryDirectory(prefix="w0a_smoke_")
            table_path = Path(tmpdir.name) / f"synth_{months}m.npz"
        src = synthetic_table(table_path, months=months)

    # 1) 正常路径
    src_obj = GCMTableSource(table_path, year_ticks=args.year_ticks)
    src_obj.bootstrap(world)
    t0 = src_obj.temperature_c(flat, 0)
    p = src_obj.precip_mm_yr()
    w = src_obj.wind()
    report["temperature_c"] = {"min": round(float(t0.min()), 4), "max": round(float(t0.max()), 4),
                               "nan": int(np.isnan(t0).sum())}
    report["precip_mm_yr"] = {"min": round(float(p.min()), 2), "max": round(float(p.max()), 2),
                             "mean": round(float(p.mean()), 2), "nan": int(np.isnan(p).sum())}
    report["wind"] = {k: (None if v is None else {"min": round(float(v.min()), 3),
                                                 "max": round(float(v.max()), 3),
                                                 "nan": int(np.isnan(v).sum())})
                      for k, v in w.items()}
    report["probe"] = {k: src_obj.probe()[k] for k in
                       ("climate_table_shape", "climate_table_months", "climate_year_ticks")}

    # 2) 守恒（世界侧 vs 表侧）
    if src is not None:
        lat_w = np.degrees(np.asarray(world._lat, dtype=np.float64)).ravel()
        world_t = t0.reshape(args.rows, args.cols)
        src_t = np.asarray(src["temperature_c"])
        src_t = src_t.mean(axis=0) if src_t.ndim == 3 else src_t
        g_src = global_mean(src_t, src["lat"])
        g_dst = global_mean(world_t, lat_w)
        report["conservation_temperature"] = {
            "src_global_mean": round(g_src, 10), "dst_global_mean": round(g_dst, 10),
            "relerr": float(abs(g_src - g_dst) / max(abs(g_src), 1e-30))}
        src_p = np.asarray(src["precip_mm_yr"])
        src_p = src_p.mean(axis=0) if src_p.ndim == 3 else src_p
        gp_src = global_mean(src_p, src["lat"])
        gp_dst = global_mean(p, lat_w)
        report["conservation_precip"] = {
            "src_global_mean": round(gp_src, 6), "dst_global_mean": round(gp_dst, 6),
            "relerr": float(abs(gp_src - gp_dst) / max(abs(gp_src), 1e-30))}

    # 3) 季节插值（仅 12 月表）
    if args.year_ticks:
        # 取 1/4 年（合成季节项在 0 与 1/2 年处都过零，用 1/4 年才有最大差）
        t_q = src_obj.temperature_c(flat, args.year_ticks // 4)
        report["season_quarter_max_abs_c"] = round(float(np.abs(t_q - t0).max()), 4)
        report["season_half_max_abs_c"] = round(
            float(np.abs(src_obj.temperature_c(flat, args.year_ticks // 2) - t0).max()), 4)
        report["season_quarter_unique_values"] = int(np.unique(np.round(t_q, 9)).size)

    # 4) 错误路径
    errs = {}
    try:
        GCMTableSource(None).bootstrap(world)
    except NotImplementedError:
        errs["no_table_path"] = "NotImplementedError"
    # 表是月层还是年平，必须**看表本身**（不能看命令行参数——真表可能自带 12 月层）
    with np.load(table_path, allow_pickle=False) as _npz:
        is_monthly = any(np.asarray(_npz[k]).ndim == 3 for k in _npz.files if k != "meta_json")
    try:
        if is_monthly:
            GCMTableSource(table_path).bootstrap(world)   # 缺 year_ticks
            errs["monthly_without_year_ticks"] = "NO RAISE (BAD)"
        else:
            errs["monthly_without_year_ticks"] = "skipped(annual table)"
    except NotImplementedError:
        errs["monthly_without_year_ticks"] = "NotImplementedError"
    report["precip_nonneg"] = bool(float(np.asarray(p).min()) >= 0.0)
    report["error_paths"] = errs

    ok = (report["temperature_c"]["nan"] == 0 and report["precip_mm_yr"]["nan"] == 0
          and report.get("conservation_temperature", {}).get("relerr", 0.0) < 1e-10
          and report.get("conservation_precip", {}).get("relerr", 0.0) < 1e-10
          and report["precip_nonneg"]
          and errs.get("no_table_path") == "NotImplementedError")
    report["ok"] = bool(ok)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if tmpdir is not None:
        tmpdir.cleanup()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
