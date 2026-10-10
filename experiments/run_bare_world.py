"""无生物纯场装置（run_bare_world.py）——「纯斑块世界」M0 探针。

口径
----
* 设计稿 `docs/设计文档/设计-纯斑块世界-底层规则自组织-20261011.md` §四/§十五；
  四臂编排 = fish 直令（≤4 run、一半对照、单 run ≤2000 tick）。
* 臂定义（锁在 `ARMS`，预注册引用本表；同 seed 同参数，**只动规则开关**）：
    1 现状（对照）       env 全关（= 现役规则绝对基线）
    2 置换零模型（对照） 全规则 + H 空间置换（空间归因 null）
    3 地形（处理）       H→W→P；无河（runoff_gain=0）
    4 全规则（处理）     地形 + 河流汇流（runoff_gain=2.0）
  `water_sensitivity = cap_sensitivity = 1.0`（线性耦合 = 最小假设；AWM=1 归一由
  ENV 场自身保证、k_max=4.0 默认上钳）、`terrain_*`/`lapse_c`/`orographic`/`et_frac`
  全走 EnvFieldConfig 默认。
* 无生物：`initial_count` **构造后置改 0**（先例 `experiments/smell_perf_probe.py:60`；
  构造期断言 initial_count≥1 只在新构造时生效）。N=0 时 `_step_population` 早退、
  资源再生在同 tick 前序步骤照跑 ⇒ 场照常演化（设计稿 §2.2 已核，代码锚 3344/2895）。
  ⚠️ 不走 `eng.run()`（其默认 `stop_on_extinction=True` 会因 N=0 立即停）⇒ `eng.step()` 循环。
* 落盘（**不进主仓**，进数据仓 `the-world-data/bare_world/`）：一条 run =
  一个 npz（静态场 + 期初/期中/期末存量快照）+ 一个 CSV（样点读数）+ 一个
  manifest.json（臂定义/生效装置/env probe/计时/git 号）。run 结束 flush+fsync。
* 引擎侧零标签：本脚本不定义"斑块类型"，只出连续场（判型在观察侧，设计稿 §八）。

用法
----
  python experiments/run_bare_world.py --arm 4 --device s2 --seed 42 --ticks 2000
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import SimConfig            # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402
from tools.device_presets import add_device_arg, resolve_device  # noqa: E402

#: 四臂定义（预注册引用本表；键名与 EnvFieldConfig 字段对齐）
ARMS: dict[int, dict] = {
    1: {"name": "现状(对照)", "env": False, "shuffle": False,
        "runoff_gain": 0.0, "ws": 0.0, "cs": 0.0},
    2: {"name": "置换零模型(对照)", "env": True, "shuffle": True,
        "runoff_gain": 2.0, "ws": 1.0, "cs": 1.0},
    3: {"name": "地形(处理)", "env": True, "shuffle": False,
        "runoff_gain": 0.0, "ws": 1.0, "cs": 1.0},
    4: {"name": "全规则(处理)", "env": True, "shuffle": False,
        "runoff_gain": 2.0, "ws": 1.0, "cs": 1.0},
}


def build_cfg(arm: int, seed: int, rows: int, cols: int, patches: int,
              rgm: float) -> SimConfig:
    cfg = SimConfig(seed=seed)
    # 🔴 装置口径硬写为 s2 家族（R5.6：正式批必须显式 --device s2；这里的默认值
    #   与 DEVICE_PRESETS['s2'] 同步，忘传 --device 也不静默换装置）。
    cfg.world.rows, cfg.world.cols = int(rows), int(cols)
    cfg.simulation.use_sim_core = False       # env 需 Python 路径；四臂统一（可比性）
    cfg.resources.distribution = "patchy"
    cfg.resources.patch_count = int(patches)
    cfg.resources.bg_production_zero = True   # s2 口径：背景纯零产能
    cfg.resources.patch_regrowth_mult = float(rgm)
    cfg.population.initial_count = 0          # 🔴 构造后置改 0（先例 smell_perf_probe.py:60）
    spec = ARMS[arm]
    if spec["env"]:
        e = cfg.env_field
        e.enabled = True
        e.salt = 0
        e.shuffle = bool(spec["shuffle"])
        e.runoff_gain = float(spec["runoff_gain"])
        e.water_sensitivity = float(spec["ws"])
        e.cap_sensitivity = float(spec["cs"])
    return cfg


def _sample_row(tick: int, grid, cap_pos, pos) -> dict:
    s = grid[pos]
    sat = np.clip(s / cap_pos, 0.0, 1.0)
    return {
        "tick": tick,
        "stock_sum": float(grid.sum()),
        "stock_pos_mean": float(s.mean()),
        "stock_pos_std": float(s.std()),
        "stock_p50": float(np.percentile(s, 50.0)),
        "stock_p90": float(np.percentile(s, 90.0)),
        "sat_mean": float(sat.mean()),
        "sat_gt50_frac": float((sat > 0.5).mean()),
    }


def _git_head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def run_arm(arm: int, seed: int, ticks: int, sample: int, out_dir: Path,
            device_applied: dict, rows: int, cols: int, patches: int,
            rgm: float) -> dict:
    spec = ARMS[arm]
    cfg = build_cfg(arm, seed, rows, cols, patches, rgm)
    t0 = time.perf_counter()
    eng = SphereEngine(cfg)
    t_build = time.perf_counter() - t0

    res = eng.resources
    cap = res._capacity
    pos = cap > 0.0
    cap_pos = cap[pos]

    static: dict[str, np.ndarray] = {}
    env_probe = None
    if spec["env"]:
        ef = eng.env_field
        static = {"h": ef.h, "w": ef.w, "temp": ef.temp, "acc": ef.acc,
                  "river_mask": ef.river_mask}
        env_probe = ef.probe()

    snap_ticks = sorted({ticks // 2, ticks})
    snaps: dict[int, np.ndarray] = {0: res._grid.copy()}
    samples = [_sample_row(0, res._grid, cap_pos, pos)]
    t0 = time.perf_counter()
    for t in range(1, ticks + 1):
        eng.step()
        if t in snap_ticks:
            snaps[t] = res._grid.copy()
        if t % sample == 0 or t == ticks:
            samples.append(_sample_row(t, res._grid, cap_pos, pos))
    run_s = time.perf_counter() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"arm{arm}_s{seed}"
    npz_path = out_dir / f"{stem}.npz"
    csv_path = out_dir / f"{stem}.csv"
    man_path = out_dir / f"{stem}.manifest.json"

    payload: dict[str, np.ndarray] = {
        "capacity": cap, "patch_mask": res._patch_mask}
    payload.update(static)
    for t, arr in snaps.items():
        payload[f"stock_t{t}"] = arr
    np.savez_compressed(npz_path, **payload)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        f.write("arm,seed," + ",".join(samples[0].keys()) + "\n")
        for r in samples:
            f.write("%d,%d," % (arm, seed)
                    + ",".join(f"{v:.9g}" for v in r.values()) + "\n")
        f.flush()
        os.fsync(f.fileno())

    manifest = {
        "probe": "run_bare_world",
        "date": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "git_head": _git_head(),
        "arm": arm, "arm_name": spec["name"], "arm_spec": spec,
        "seed": seed, "ticks": ticks, "sample_every": sample,
        "world": {"rows": rows, "cols": cols,
                  "n_cells": rows * cols},
        "device_applied": device_applied,
        "use_sim_core": False,
        "sparse_fields": bool(cfg.simulation.sparse_fields),
        "pop_initial": 0, "patches": patches, "rgm": rgm,
        "n_patch_cells": int(pos.sum()),
        "env_probe": env_probe,
        "env_note": getattr(eng, "env_note", None),
        "timing": {"build_s": t_build, "run_s": run_s,
                   "ms_per_tick": run_s / ticks * 1000.0},
        "outputs": {"npz": npz_path.name, "csv": csv_path.name},
        "snap_ticks": [0] + snap_ticks,
    }
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    print(f"[arm{arm} {spec['name']}] ticks={ticks} build={t_build:.2f}s "
          f"run={run_s:.1f}s ({run_s / ticks * 1000.0:.1f} ms/tick) "
          f"patch_cells={int(pos.sum())}")
    if env_probe is not None:
        print(f"  env: w_mean={env_probe['w_mean']:.4g} u_p99={env_probe['u_p99']:.3g} "
              f"u_max={env_probe['u_max']:.3g} river_frac={env_probe['river_frac']:.4f} "
              f"flood_filled={env_probe['flood_filled']} "
              f"gf_max={env_probe.get('gf_max')} cf_max={env_probe.get('cf_max')}")
    print(f"  -> {npz_path}")
    return manifest


def main(argv=None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="无生物纯场装置（四臂协议 §十五）")
    ap.add_argument("--arm", type=int, choices=sorted(ARMS), required=True,
                    help="1 现状 | 2 置换零模型 | 3 地形 | 4 全规则")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--sample", type=int, default=50, help="CSV 样点间隔（tick）")
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--cols", type=int, default=960)
    ap.add_argument("--patches", type=int, default=1700)
    ap.add_argument("--rgm", type=float, default=1.195)
    ap.add_argument("--out", default=str(ROOT / "the-world-data" / "bare_world"))
    add_device_arg(ap)
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])
    device_applied = resolve_device(a, sys.argv[1:],
                                    logger=lambda m: print(m, file=sys.stderr))
    if a.ticks < 1 or a.sample < 1:
        raise SystemExit("🔴 --ticks/--sample 必须 ≥1")
    run_arm(a.arm, a.seed, a.ticks, a.sample, Path(a.out), device_applied,
            a.rows, a.cols, a.patches, a.rgm)


if __name__ == "__main__":
    main()
