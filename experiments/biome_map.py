"""A20.2 群系分类图 + 阈值敏感性（±20%）。

输入 = 30 年气候表（年平温度 + 年降水），先按世界网格（默认 s2）重映射，再逐格分类。
⚠️ 分类阈值是**人类给的**（Köppen 量级）⇒ 必须同时报 ±20% 敏感性，不许只报一张图。

用法
----
    python experiments/biome_map.py --device s2 --table <npz> --out results/biome_s2.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import BiomeConfig  # noqa: E402
from tools.device_presets import DEVICE_PRESETS  # noqa: E402
from world.biome import BIOME_NAMES, biome_stats, classify  # noqa: E402
from world.climate import GCMTableSource  # noqa: E402
from world.env_field import EnvField  # noqa: E402
from world.light_and_temperature import LightAndTemperature  # noqa: E402
from world.sphere_world import SphereWorld  # noqa: E402
from simulation.config import EnvFieldConfig  # noqa: E402

TIER = "fast"
COLORS = {
    "ice": "#e8eef7", "tundra": "#b8c6a8", "desert": "#e3c98b",
    "tropical_rainforest": "#1f6f3f", "savanna": "#c9a227",
    "temperate_continental": "#7a9a5a", "temperate_oceanic": "#4f8f6a",
    "boreal": "#2f5d3a",
}


def _git_head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:      # noqa: BLE001
        return ""


def scaled(cfg: BiomeConfig, k: float) -> BiomeConfig:
    """阈值整体 ×k（k=0.8 / 1.2 ⇒ ±20% 敏感性）。温度阈值也按比例缩放。"""
    return BiomeConfig(
        enabled=True,
        ice_t=cfg.ice_t * k, tundra_t=cfg.tundra_t * k,
        desert_p=cfg.desert_p * k, rainforest_p=cfg.rainforest_p * k,
        tropical_t=cfg.tropical_t * k, temperate_t=cfg.temperate_t * k,
        continental_amp=cfg.continental_amp * k,
    )


def write_svg(path: Path, biome: np.ndarray) -> None:
    rows, cols = biome.shape
    cw, ch = 720 / cols, 360 / rows
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="440" '
             f'viewBox="0 0 760 440"><rect width="100%" height="100%" fill="#ffffff"/>',
             '<text x="20" y="24" font-size="14" fill="#111827">'
             'Biome classification (derived from annual T / P; thresholds = world-family params)'
             '</text>']
    for i in range(rows):
        for j in range(cols):
            name = BIOME_NAMES[int(biome[i, j])]
            parts.append(f'<rect x="{20 + j * cw:.2f}" y="{40 + i * ch:.2f}" '
                         f'width="{cw + 0.4:.2f}" height="{ch + 0.4:.2f}" '
                         f'fill="{COLORS[name]}"/>')
    for k, name in enumerate(BIOME_NAMES):
        y = 410 - (len(BIOME_NAMES) - 1 - k) * 0
        x = 20 + k * 92
        parts.append(f'<rect x="{x}" y="{y}" width="10" height="10" fill="{COLORS[name]}"/>')
        parts.append(f'<text x="{x + 14}" y="{y + 9}" font-size="9" fill="#111827">{name}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="s2")
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--svg", default=None)
    args = ap.parse_args()

    preset = DEVICE_PRESETS.get(args.device) if args.device else None
    rows = int(preset["rows"]) if preset else 120
    cols = int(preset["cols"]) if preset else 240
    world = SphereWorld(rows, cols)
    src = GCMTableSource(Path(args.table), year_ticks=87600)
    src.bootstrap(world)
    flat = np.arange(world.n_cells, dtype=np.int64)
    temp = src.temperature_c(flat, 0).reshape(rows, cols)
    precip = np.asarray(src.precip_mm_yr(), dtype=np.float64).reshape(rows, cols)
    areas = world.cell_area(flat).reshape(rows, cols)
    # 🔴 群系只在**陆地**上有意义：海洋用同一套气候量分类会造出"海上稀树草原"这种假读数。
    #   陆地掩码取世界自己的 EnvField（volume 海平面默认）⇒ 图与统计都只在陆地上比。
    lt = LightAndTemperature(world)
    land = EnvField(world, lt, EnvFieldConfig(enabled=True, c0b_enabled=True)) \
        .land_mask.reshape(rows, cols)

    base = BiomeConfig(enabled=True)
    out = {"git_head": _git_head(), "device": args.device, "grid": [rows, cols],
           "table": str(args.table),
           "mean_temp_c": round(float(np.average(temp, weights=areas)), 2),
           "mean_precip_mm": round(float(np.average(precip, weights=areas)), 1),
           "variants": {}}
    for tag, k in (("default", 1.0), ("low_20pct", 0.8), ("high_20pct", 1.2)):
        cfg = scaled(base, k)
        b = classify(temp, precip, thresholds=cfg.thresholds())
        b = np.where(land, b, np.int8(-1))          # 海洋 = -1（不参与统计、图上画灰）
        st = biome_stats(b, np.where(land, areas, 0.0))
        out["variants"][tag] = {"k": k, "dominant": st["dominant"],
                                "frac": st["frac"], "counts": st["counts"]}
        if tag == "default":
            out["biome_map"] = b.tolist()
            if args.svg:
                write_svg(Path(args.svg), b)
        print(f"[{tag:9s} k={k:.1f}] dominant={st['dominant']:22s} "
              f"tropical_rainforest={st['frac']['tropical_rainforest']:.3f} "
              f"desert={st['frac']['desert']:.3f} ice={st['frac']['ice']:.3f} "
              f"temperate={(st['frac']['temperate_oceanic'] + st['frac']['temperate_continental']):.3f}")

    print(json.dumps({k: v for k, v in out.items() if k != "biome_map"},
                     ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
