# -*- coding: utf-8 -*-
"""P0.3 汇率对照探针：45 s/tick（dt_s=150）vs 36 s/tick（dt_s=120）的开档读数。

用法（仓库根）：
    python tools/atmosphere_epoch_probe.py [--ticks 2000] [--rows 480] [--cols 960]

口径
----
* 用 `Atmosphere` **直跑**（不经 `SimConfig.__post_init__` 的汇率断言——那条断言会
  按设计拦住 45 s/tick 的旧纪元）。两条都跑同样的 **tick 数**（= 同样的主 tick），
  差别只在每个主 tick 的物理秒数（45 vs 36）。
* 装置：`_cap_land`（北陆南海）+ `pr_fine=1`，与 `tests/test_c4a2_moisture.py` 同源。
* 读数：温度/风/水汽判据；`precip/evap/sea2land/residence` 因与 `dt_s` 成正比而变，
  `t_awm`/`t_min`/`t_max` 是物理量、不随纪元变（除瞬态差异）。
* 输出 JSON 与屏幕对照表；JSON 落 `_rerun_logs/`（一次性读数件，不入主仓）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import AtmosphereConfig                       # noqa: E402
from tests.test_c4a2_moisture import _cap_land                       # noqa: E402
from world.atmosphere import Atmosphere                              # noqa: E402
from world.sphere_world import SphereWorld                           # noqa: E402

KEYS = ("t_awm", "t_min", "t_max", "u_max_ms", "u_rms_awm_ms",
        "cfl_max", "precip_mm_yr", "evap_mm_yr", "sea2land_flux_mm_yr",
        "residence_days_global", "bucket_awm_land_mm", "wb_imbalance_rel")


def run(dt_s: float, rows: int, cols: int, ticks: int) -> dict:
    a = Atmosphere(
        SphereWorld(rows, cols),
        AtmosphereConfig(enabled=True, coarse_factor=4, dt_s=float(dt_s),
                         moisture_on=True),
        _cap_land(rows, cols),
        pr_fine=np.ones(rows * cols, dtype=np.float64))
    n_adv = int(ticks) // 10          # every_ticks=10
    for k in range(1, n_adv + 1):
        a.advance(k * 10)
    pr = a.probe()
    out = {k: pr.get(k) for k in KEYS}
    out["dt_s"] = float(dt_s)
    out["tick_seconds"] = float(dt_s) * 3.0 / 10.0
    out["n_advance"] = n_adv
    out["steps"] = int(a.steps)
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--cols", type=int, default=960)
    args = ap.parse_args()
    old = run(150.0, args.rows, args.cols, args.ticks)
    new = run(120.0, args.rows, args.cols, args.ticks)
    print(f"{'读数':<28}{'45 s/tick (旧)':>18}{'36 s/tick (新)':>18}{'变化':>12}")
    for k in KEYS:
        o, n = old.get(k), new.get(k)
        try:
            chg = f"{(n / o - 1.0) * 100:+.2f}%" if o not in (None, 0) else "—"
        except TypeError:
            chg = "—"
        print(f"{k:<28}{str(o):>18}{str(n):>18}{chg:>12}")
    payload = {"ticks": args.ticks, "rows": args.rows, "cols": args.cols,
               "old_150": old, "new_120": new}
    out = Path("_rerun_logs") / "epoch_150_vs_120.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
