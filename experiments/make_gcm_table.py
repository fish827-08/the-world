"""把 GCM 导出（CSV 长表）整理成 `npz-climatology-v0` 气候表 + manifest。

为什么用 CSV 中转
----------------
云端机只有 2 核 2 GB，装 `netCDF4/xarray` 有风险；Julia 侧用 `println` 写 CSV 零依赖。
重量级解析都留给本脚本（本机 Python 环境齐全）。

CSV 格式（长表，首行表头）
-------------------------
    lat,lon[,month],temperature_c,precip_mm_yr,u,v
* `lat` 单位度（南负北正），`lon` 单位度（0..360）
* `month` 可选：1..12（有 ⇒ 生成 (12, nlat, nlon) 月层表；无 ⇒ 年平 (nlat, nlon)）
* 缺列允许（只导入有的字段）

用法
----
    python experiments/make_gcm_table.py --csv t31.csv --out t31_earth_v0.npz \
        --source "SpeedyWeather.jl 0.22.1 T31 10L 3yr spin-up + 1yr mean" --months 12

产物
----
* `<out>.npz`：`lat/lon/temperature_c/precip_mm_yr/u/v` + `meta_json`
* `<out>.manifest.json`：行数、网格、缺测统计、各字段全球均值（面积权）、sha256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atmo_bridge.remap import global_mean  # noqa: E402

FIELDS = ("temperature_c", "precip_mm_yr", "u", "v")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="CSV → npz-climatology-v0")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True, help="输出 .npz 路径")
    ap.add_argument("--source", required=True, help="来源说明（进 meta_json）")
    ap.add_argument("--months", type=int, default=0, choices=(0, 12),
                    help="0=年平表；12=月层表（CSV 必须带 month 列）")
    args = ap.parse_args()

    csv_path, out_path = Path(args.csv), Path(args.out)
    with csv_path.open(encoding="utf-8") as fh:
        header = [h.strip() for h in fh.readline().split(",")]
    need = {"lat", "lon"}
    if not need.issubset(header):
        raise SystemExit(f"CSV 缺列 {sorted(need - set(header))}；表头={header}")
    cols = {name: i for i, name in enumerate(header)}
    present = [f for f in FIELDS if f in cols]
    if not present:
        raise SystemExit("CSV 里没有任何字段列（temperature_c/precip_mm_yr/u/v）")

    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=np.float64,
                         encoding="utf-8")
    lat = np.unique(data["lat"])
    lon = np.unique(data["lon"])
    nlat, nlon = lat.size, lon.size
    months = args.months if args.months else 1
    shape = (months, nlat, nlon) if months > 1 else (nlat, nlon)

    li = np.searchsorted(lat, data["lat"])
    oi = np.searchsorted(lon, data["lon"])
    if months > 1:
        if "month" not in cols:
            raise SystemExit("--months 12 需要 CSV 有 month 列")
        mi = np.clip(data["month"].astype(int), 1, 12) - 1
    else:
        mi = np.zeros(data.shape[0], dtype=int)

    out = {"lat": lat, "lon": lon}
    stats = {}
    for f in present:
        arr = np.full(shape, np.nan, dtype=np.float64)
        arr[mi, li, oi] = data[f]
        miss = int(np.isnan(arr).sum())
        if miss:
            raise SystemExit(f"{f} 缺测 {miss}/{arr.size} 格 —— 不给 NaN 表（会污染世界）")
        out[f] = arr
        annual = arr.mean(axis=0) if arr.ndim == 3 else arr
        stats[f] = {"global_mean": round(global_mean(annual, lat), 6),
                    "min": round(float(annual.min()), 6),
                    "max": round(float(annual.max()), 6)}

    meta = {"format": "npz-climatology-v0", "source": args.source,
            "grid": [int(nlat), int(nlon)], "months": int(months),
            "units": {"temperature_c": "degC", "precip_mm_yr": "mm/yr",
                      "u": "m/s (east+)", "v": "m/s (north+)"},
            "generated_by": "experiments/make_gcm_table.py",
            "converter_argv": sys.argv[1:]}
    out["meta_json"] = np.array(json.dumps(meta, ensure_ascii=False))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    manifest = {"npz": str(out_path), "sha256": _sha256(out_path),
                "rows": int(data.shape[0]), "grid": [int(nlat), int(nlon)],
                "months": int(months), "fields": stats, "meta": meta}
    man_path = out_path.with_suffix(".manifest.json")
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"npz": str(out_path), "manifest": str(man_path),
                      "sha256": manifest["sha256"], "fields": stats},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
