"""GCM 气候表 ↔ 世界气候态：三条纬向剖面对照（W0-A / A4 交付件）。

为什么有这个脚本
----------------
W1 要把气候源换成外部 GCM（SpeedyWeather / ExoPlaSim 的离线表）。换源前必须先有
**可复算的对照口径**：同一条纬线上，世界给出的温度/降水/风 与 GCM 表差多少。
本脚本只做"读两侧 → 算纬向平均 → 出 CSV + SVG"，不含任何判据、不碰引擎状态。

两侧口径（写死，改口径 = 作废旧图）
--------------------------------
* **世界侧**：`LightAndTemperature.temperature()` 在**一个自转周期内均匀采样 12 次**取均值
  （默认无季节 ⇒ 这就是年周期平均；有季节时须把 `--season-period` 给上与 SimConfig 相同的值，
  采样窗口仍是一个自转周期 ⇒ **只代表"当前季节"**，跨季节对比要另跑 4 次）。
  降水 = `EnvField.pr`（**proxy**，hPa 量纲，不是 mm/yr；只能比**空间型**，不能比绝对值）。
  风 = **N/A**：`EnvField` 没有风；`Atmosphere.u_*` 是诊断量且需长跑自旋 ⇒ 本脚本不产出。
* **GCM 侧**：`npz` 气候表（格式见 `world.climate.GCM_TABLE_FORMAT`），纬向平均；若有 12 月层则先对月平均。

用法
----
    python experiments/run_gcm_compare.py --device s2
    python experiments/run_gcm_compare.py --device s2 --table <gcm_table.npz>

产物
----
    results/w0a_profiles_world.csv / .svg      （只跑世界侧）
    results/w0a_profiles_compare.csv / .svg    （两侧都给）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import EnvFieldConfig  # noqa: E402
from tools.device_presets import DEVICE_PRESETS  # noqa: E402
from world.env_field import EnvField  # noqa: E402
from world.light_and_temperature import LightAndTemperature  # noqa: E402
from world.sphere_world import SphereWorld  # noqa: E402


def zonal_mean(field: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """(rows*cols,) 或 (rows, cols) → 每行均值（纬度带平均）。"""
    a = np.asarray(field, dtype=np.float64)
    if a.size == rows * cols:
        a = a.reshape(rows, cols)
    return a.mean(axis=1)


def world_profiles(rows: int, cols: int, *, rotation_period: int, tilt_rad: float,
                   season_period: int, samples: int) -> dict:
    t0 = time.perf_counter()
    world = SphereWorld(rows, cols)
    lt = LightAndTemperature(world, rotation_period=rotation_period,
                             tilt_rad=tilt_rad, season_period=season_period)
    env = EnvField(world, lt, EnvFieldConfig())
    flat = np.arange(world.n_cells, dtype=np.int64)

    acc = np.zeros(world.n_cells, dtype=np.float64)
    ticks = [int(round(i * rotation_period / samples)) for i in range(samples)]
    for tick in ticks:
        acc += lt.temperature(flat, tick)
    t_mean = acc / samples

    lat_deg = np.degrees(np.asarray(world._lat, dtype=np.float64)).ravel()
    assert lat_deg.size == rows, f"world._lat 应为每行一个（{rows}），实得 {lat_deg.size}"
    out = {
        "lat_deg": lat_deg,
        "t_world_c": zonal_mean(t_mean, rows, cols),
        "pr_world_proxy": zonal_mean(env.pr, rows, cols),
        "wall_s": time.perf_counter() - t0,
        "grid": f"{rows}x{cols}",
        "samples": samples,
        "ticks": ticks,
    }
    return out


def gcm_profiles(table_path: Path) -> dict:
    with np.load(table_path, allow_pickle=False) as npz:
        lat = np.asarray(npz["lat"], dtype=np.float64).ravel()
        out = {"lat_deg": lat}
        for key, name in (("temperature_c", "t_gcm_c"),
                          ("precip_mm_yr", "p_gcm_mm_yr"),
                          ("u", "u_gcm_ms"), ("v", "v_gcm_ms")):
            if key not in npz:
                continue
            arr = np.asarray(npz[key], dtype=np.float64)
            if arr.ndim == 3:                      # (12, nlat, nlon) ⇒ 月平均
                arr = arr.mean(axis=0)
            out[name] = arr.mean(axis=1)
        if "meta_json" in npz:
            out["meta"] = json.loads(str(npz["meta_json"]))
    order = np.argsort(out["lat_deg"])
    for k in list(out):
        if k == "meta":
            continue
        out[k] = np.asarray(out[k])[order]
    return out


def interp_to(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    return np.interp(x_dst, x_src, y_src)


#: 判据分带（A17.1，2026-10-19 审核线裁定：**判据更正，不是模型失败**）
#: 南极单独一条：地球南极高原年均 ≈ −50 ℃；本模型 T21（≈5.6°）+ 无深海热输送 +
#: 高原反照率 ⇒ −53.6 属合理范围，原「极地 −20…−40」对南极不成立。
CRITERIA_BANDS: dict[str, dict] = {
    "equator_t_c": {"band": (-5.0, 5.0), "lo": 25.0, "hi": 30.0,
                    "desc": "赤道年均气温 25–30 ℃"},
    # 北极取 ≥70°N（北极圈内、以海洋为主）——地球该带年均 ≈ −20…−30 ℃
    "north_polar_t_c": {"band": (70.0, 90.0), "lo": -40.0, "hi": -20.0,
                        "desc": "北极（≥70°N 带均） −20…−40 ℃"},
    # 南极取 ≤75°S：判据说的是**高原**。60–90°S 整带里南大洋占大半，地球该带也才 ≈ −25 ℃
    # ⇒ 用整带配「−40…−60」对任何模型都不成立（那是把判据写错了，不是模型错了）。
    "south_plateau_t_c": {"band": (-90.0, -75.0), "lo": -60.0, "hi": -40.0,
                          "desc": "南极高原（≤75°S 带均） −40…−60 ℃"},
}

#: 诊断用（**不进判据**）：整带 60–90 的带均，用来说明"为什么南极不能用整带"
DIAG_BANDS: dict[str, tuple[float, float]] = {"north_60_90": (60.0, 90.0),
                                              "south_60_90": (-90.0, -60.0)}


def _band_value(lat: np.ndarray, v: np.ndarray, lo: float, hi: float) -> float:
    m = (lat >= lo) & (lat <= hi)
    return float(np.mean(np.asarray(v, dtype=np.float64)[m]))


def criteria_report(lat: np.ndarray, t_c: np.ndarray | None,
                    p_mm: np.ndarray | None = None) -> dict:
    """把判据算成可进汇报的结构（**只判不算**：不过就如实写差值，不调参）。"""
    lat = np.asarray(lat, dtype=np.float64)
    out: dict = {"bands": {k: dict(v) for k, v in CRITERIA_BANDS.items()}}

    if t_c is not None:
        t = np.asarray(t_c, dtype=np.float64)
        out["diag_bands"] = {k: round(_band_value(lat, t, *b), 2)
                             for k, b in DIAG_BANDS.items()}
        for key, spec in CRITERIA_BANDS.items():
            val = _band_value(lat, t, *spec["band"])
            out[key] = {
                "value": round(val, 2), "target": [spec["lo"], spec["hi"]],
                "pass": bool(spec["lo"] <= val <= spec["hi"]),
                "delta_out_of_range": round(
                    0.0 if spec["lo"] <= val <= spec["hi"]
                    else (val - spec["hi"] if val > spec["hi"] else val - spec["lo"]), 2),
            }

    if p_mm is not None:
        p = np.asarray(p_mm, dtype=np.float64)
        # ③ 赤道降水峰：赤道带（|lat|≤10）最大值，且高于 ±30° 与极区
        eq = (np.abs(lat) <= 10.0)
        i_eq = int(np.argmax(np.where(eq, p, -np.inf)))
        p30 = _band_value(lat, p, 25.0, 35.0)
        p_eq = float(p[i_eq])
        out["equator_precip_peak"] = {
            "value_mm_yr": round(p_eq, 1), "lat_deg": round(float(lat[i_eq]), 2),
            "p_30deg_mean": round(p30, 1),
            "pass": bool(p_eq > p30),
        }
        # ④ 中纬次峰：30–60 带内的最大值必须高于 30° 带与 60° 带（局地极大）
        subs = {}
        for name, lo, hi in (("north", 30.0, 60.0), ("south", -60.0, -30.0)):
            m = (lat >= lo) & (lat <= hi)
            i = int(np.argmax(np.where(m, p, -np.inf)))
            p_60 = _band_value(lat, p, 55.0, 65.0)
            subs[name] = {
                "value_mm_yr": round(float(p[i]), 1), "lat_deg": round(float(lat[i]), 2),
                "p_30deg_mean": round(p30, 1), "p_60deg_mean": round(p_60, 1),
                "pass": bool(float(p[i]) > p30 and float(p[i]) > p_60),
            }
        out["midlat_secondary_peak"] = subs

    # 附：纬向风统计（描述性，**无判据**——风尚未接进世界）
    return out


def wind_stats(lat: np.ndarray, u: np.ndarray) -> dict:
    u = np.asarray(u, dtype=np.float64)
    return {
        "note": "描述性统计，无判据；风尚未接进世界",
        "mean": round(float(np.mean(u)), 2), "min": round(float(np.min(u)), 2),
        "max": round(float(np.max(u)), 2),
        "u_45s": round(_band_value(lat, u, -50.0, -40.0), 2),
        "u_15n": round(_band_value(lat, u, 10.0, 20.0), 2),
    }


def write_csv(path: Path, lat: np.ndarray, cols: dict[str, np.ndarray]) -> None:
    names = list(cols)
    lines = ["lat_deg," + ",".join(f"{n}" for n in names if n != "lat_deg")]
    for i, la in enumerate(lat):
        vals = [f"{float(cols[n][i]):.6g}" for n in names if n != "lat_deg"]
        lines.append(f"{la:.4f}," + ",".join(vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _panel(x, series, *, x0, y0, w, h, title, ylab, colors):
    """一个面板的 SVG 片段（手写，零依赖）。"""
    xs = np.asarray(x, dtype=np.float64)
    allv = np.concatenate([np.asarray(v, dtype=np.float64) for _, v in series])
    lo, hi = float(np.min(allv)), float(np.max(allv))
    pad = (hi - lo) * 0.12 or 1.0
    lo, hi = lo - pad, hi + pad

    def px(lat_):
        return x0 + (lat_ + 90.0) / 180.0 * w

    def py(v):
        return y0 + h - (v - lo) / (hi - lo) * h

    out = [f'<text x="{x0}" y="{y0 - 8}" font-size="13" fill="#1f2937">{title}</text>',
           f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#fbfbfd" stroke="#d1d5db"/>']
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = y0 + h - frac * h
        out.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0 + w}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        out.append(f'<text x="{x0 - 6}" y="{yy + 4:.1f}" font-size="10" fill="#6b7280" '
                   f'text-anchor="end">{lo + frac * (hi - lo):.3g}</text>')
    for lat_ in (-90, -60, -30, 0, 30, 60, 90):
        xx = px(lat_)
        out.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0 + h}" stroke="#eef0f3"/>')
        out.append(f'<text x="{xx:.1f}" y="{y0 + h + 14}" font-size="10" fill="#6b7280" '
                   f'text-anchor="middle">{lat_}</text>')
    for (label, vals), color in zip(series, colors):
        pts = " ".join(f"{px(a):.1f},{py(b):.1f}" for a, b in zip(xs, np.asarray(vals)))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
    out.append(f'<text x="{x0}" y="{y0 + h + 30}" font-size="10" fill="#374151">{ylab}</text>')
    lx = x0 + w - 4
    for i, ((label, _), color) in enumerate(zip(series, colors)):
        out.append(f'<text x="{lx}" y="{y0 + 14 + i * 14}" font-size="10" fill="{color}" '
                   f'text-anchor="end">{label}</text>')
    return "\n".join(out)


def write_svg(path: Path, lat: np.ndarray, panels: list[tuple[str, list, str]]) -> None:
    w, h = 640, 190
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="720" '
             f'height="{60 + len(panels) * (h + 60)}" viewBox="0 0 720 {60 + len(panels) * (h + 60)}">',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             '<text x="40" y="28" font-size="15" fill="#111827">'
             'Latitude profiles: world climatology vs GCM table (W0-A / A4)</text>']
    for i, (title, series, ylab) in enumerate(panels):
        y0 = 60 + i * (h + 60)
        parts.append(_panel(lat, series, x0=90, y0=y0, w=w, h=h, title=title,
                            ylab=ylab, colors=["#b91c1c", "#1d4ed8", "#047857"][:len(series)]))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="世界气候态 vs GCM 气候表：三条纬向剖面")
    ap.add_argument("--device", default="s2", help="装置预设档（默认 s2=480x960）")
    ap.add_argument("--table", default=None, help="GCM 气候表 .npz（不给则只出世界侧）")
    ap.add_argument("--out", default=str(ROOT / "results"), help="产物目录")
    ap.add_argument("--rotation-period", type=int, default=2400)
    ap.add_argument("--tilt-rad", type=float, default=0.0, help="黄赤交角（弧度），默认 0=无季节")
    ap.add_argument("--season-period", type=int, default=0)
    ap.add_argument("--samples", type=int, default=12, help="一个自转周期内采样次数")
    args = ap.parse_args()

    preset = DEVICE_PRESETS.get(args.device)
    if preset is None:
        ap.error(f"未知 --device {args.device!r}；可选 {sorted(DEVICE_PRESETS)}")
    rows, cols = int(preset["rows"]), int(preset["cols"])

    t_all = time.perf_counter()
    wp = world_profiles(rows, cols, rotation_period=args.rotation_period,
                        tilt_rad=args.tilt_rad, season_period=args.season_period,
                        samples=args.samples)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = "compare" if args.table else "world"
    lat = np.asarray(wp["lat_deg"], dtype=np.float64)
    cols_out = {"lat_deg": lat, "t_world_c": wp["t_world_c"],
                "pr_world_proxy": wp["pr_world_proxy"]}
    panels = [("Temperature (C): world annual mean",
               [("world", wp["t_world_c"])], "deg C")]

    if args.table:
        gp = gcm_profiles(Path(args.table))
        for name in ("t_gcm_c", "p_gcm_mm_yr", "u_gcm_ms", "v_gcm_ms"):
            if name in gp:
                cols_out[name] = interp_to(gp["lat_deg"], gp[name], lat)
        if "t_gcm_c" in cols_out:
            panels = [("Temperature (C): world vs GCM",
                       [("world", wp["t_world_c"]), ("GCM", cols_out["t_gcm_c"])], "deg C")]
        if "p_gcm_mm_yr" in cols_out:
            panels.append(("Precipitation (mm/yr, GCM) vs world proxy",
                           [("GCM", cols_out["p_gcm_mm_yr"])], "mm/yr"))
        if "u_gcm_ms" in cols_out:
            panels.append(("Zonal wind u (m/s, GCM)", [("GCM", cols_out["u_gcm_ms"])], "m/s"))
    else:
        panels.append(("Precipitation proxy (hPa, world EnvField.pr)",
                       [("world", wp["pr_world_proxy"])], "hPa"))

    csv_path = outdir / f"w0a_profiles_{tag}.csv"
    svg_path = outdir / f"w0a_profiles_{tag}.svg"
    write_csv(csv_path, lat, cols_out)
    write_svg(svg_path, lat, panels)
    wall = time.perf_counter() - t_all

    print(json.dumps({
        "grid": wp["grid"], "device": args.device, "csv": str(csv_path), "svg": str(svg_path),
        "world_wall_s": round(wp["wall_s"], 2), "total_wall_s": round(wall, 2),
        "samples": wp["samples"], "season": {"tilt_rad": args.tilt_rad,
                                             "season_period": args.season_period},
        "wind": "N/A (EnvField 无风；Atmosphere.u_* 需长跑自旋，本脚本不产出)",
        "criteria": (criteria_report(lat, cols_out.get("t_gcm_c"),
                                     cols_out.get("p_gcm_mm_yr"))
                     if args.table else {"note": "只跑世界侧时不出判据（无 GCM 场）"}),
        "wind_stats": (wind_stats(lat, cols_out["u_gcm_ms"])
                       if "u_gcm_ms" in cols_out else None),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
