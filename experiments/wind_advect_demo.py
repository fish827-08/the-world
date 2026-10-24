"""A20.5 风平流开档验证：用 30 年表的真实风场推动气味场（证明"沿风向偏移"）。

做法
----
1. 读表 → 重映射到世界网格（s1 240×120）拿到 `u` 风场；
2. 造一个气味场（默认关 vs 开档），在同一个格注入一团"食物气味"；
3. 用 `dt = tick_seconds × update_every` 平流 → 比较浓度质心的位移与风向。

不设判据（A19 决策页口径）：只要求质心沿**平均风向**移动、且关档时位移为 0。

用法
----
    python experiments/wind_advect_demo.py --device s1 --table <npz> --tick-seconds 360
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

from simulation.config import SmellConfig  # noqa: E402
from tools.device_presets import DEVICE_PRESETS  # noqa: E402
from world.advect import centroid  # noqa: E402
from world.climate import GCMTableSource  # noqa: E402
from world.smell_field import SmellField  # noqa: E402
from world.sphere_world import SphereWorld  # noqa: E402


def _git_head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:      # noqa: BLE001
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="s1")
    ap.add_argument("--table", required=True)
    ap.add_argument("--tick-seconds", type=int, default=360, choices=(36, 360))
    ap.add_argument("--update-every", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    preset = DEVICE_PRESETS.get(args.device) if args.device else None
    rows = int(preset["rows"]) if preset else 120
    cols = int(preset["cols"]) if preset else 240
    world = SphereWorld(rows, cols)
    src = GCMTableSource(Path(args.table), year_ticks=87600)
    src.bootstrap(world)
    wind = src.wind()
    u = np.asarray(wind["u"], dtype=np.float64).reshape(rows, cols)
    v = np.asarray(wind["v"], dtype=np.float64).reshape(rows, cols) if wind.get("v") is not None \
        else np.zeros((rows, cols))

    # 在赤道附近注入一团气味（选一个有东风的格：u>0）
    eq = rows // 2
    c0 = cols // 3
    dt_s = float(args.tick_seconds) * float(args.update_every)

    out = {"git_head": _git_head(), "device": args.device, "grid": [rows, cols],
           "tick_seconds": args.tick_seconds, "dt_s": dt_s,
           "wind_at_source": {"u": round(float(u[eq, c0]), 2), "v": round(float(v[eq, c0]), 2)},
           "runs": {}}
    for tag, on in (("wind_off", False), ("wind_on", True)):
        cfg = SmellConfig(channels=("food",), update_every=int(args.update_every),
                          decay=1.0, inject_food=1.0, advect_with_wind=on)
        sm = SmellField(world, cfg)
        cell = eq * cols + c0
        sm.update({"food": (np.array([cell]), np.array([1.0]))}, tick=0)
        before = sm._S[0].reshape(rows, cols).copy()
        sm.set_wind(u, v, dt_s=dt_s)
        for t in range(1, 4):
            sm.update({}, tick=t)
        after = sm._S[0].reshape(rows, cols)
        r0, cc0 = centroid(before)
        r1, cc1 = centroid(after)
        d_col = ((cc1 - cc0 + cols / 2) % cols) - cols / 2     # 有符号位移（格）
        d_row = r1 - r0
        out["runs"][tag] = {"d_col_cells": round(float(d_col), 3),
                            "d_row_cells": round(float(d_row), 3),
                            "mass": round(float(after.sum()), 4)}
        print(f"[{tag:8s}] d_col={d_col:+.3f} 格  d_row={d_row:+.3f} 格  mass={after.sum():.4f}")

    # 期望位移（解析）：u·Δt / (R·cosφ·Δλ)，3 次 update ⇒ ×3
    lam = 2 * np.pi / cols
    phi = np.radians(-90 + (eq + 0.5) * 180 / rows)
    expect = float(u[eq, c0]) * dt_s * 3 / (6.371e6 * np.cos(phi) * lam)
    out["expected_d_col_cells"] = round(expect, 3)
    out["pass_off_zero"] = bool(abs(out["runs"]["wind_off"]["d_col_cells"]) < 1e-9)
    out["pass_on_direction"] = bool(out["runs"]["wind_on"]["d_col_cells"] * expect > 0
                                    and abs(out["runs"]["wind_on"]["d_col_cells"])
                                    > 0.5 * abs(expect))
    print(f"[expect] d_col≈{expect:+.3f} 格  pass_off_zero={out['pass_off_zero']} "
          f"pass_on_direction={out['pass_on_direction']}")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
