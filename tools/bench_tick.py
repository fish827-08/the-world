# -*- coding: utf-8 -*-
"""B2.2 性能基准（P1.1 修测量）：固定 workload、7 次重复、中位 + IQR。

用法（仓库根）：
    python tools/bench_tick.py                          # 两组基线：off / on(moisture)
    python tools/bench_tick.py --baseline _rerun_logs/bench_baseline.json

口径
----
* 装置 **钉死 s2**（`DEVICE_PRESETS["s2"]`，不许改）；seed=42、ticks=500、pop=10k。
* **每次重复重建引擎**（同 seed ⇒ 同 workload 轨迹）⇒ 排除"跑久了种群变化"带来的漂移。
* 7 次重复取**中位**，报 **IQR**（q75−q25）作为离散度；另报相对基线 JSON 的 **%**。
* `--cpu N` 可选 CPU 亲和（需 psutil；Windows 上 `psutil.Process().cpu_affinity`）。
* 输出 JSON（`--out`，默认 `_rerun_logs/bench_tick.json`），便于跨 commit 对账。

验收（派工原文）：**同一 commit 两次独立测量的中位差 < 0.3 个百分点**；达不到就先改协议。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# 🔴 B5.1：单线程 BLAS 必须在 **import numpy 之前**钉死（import 时线程池已建）
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import AtmosphereConfig, EnvFieldConfig, SimConfig  # noqa: E402
from simulation.sphere_engine import SphereEngine                          # noqa: E402
from tools.device_presets import DEVICE_PRESETS                            # noqa: E402

P = DEVICE_PRESETS["s2"]
TICKS = 500
SEED = 42
WARMUP = 50
REPEATS = 7
CALIB_REF = 0.2          # 校准参考值（秒）；main 里用首次实测覆盖


def build(atm_on: bool, moisture: bool) -> SphereEngine:
    cfg = SimConfig(seed=SEED)
    cfg.world.rows, cfg.world.cols = P["rows"], P["cols"]
    cfg.resources.distribution = "patchy"
    cfg.resources.patch_count = P["patches"]
    cfg.resources.patch_regrowth_mult = P["rgm"]
    cfg.resources.bg_production_zero = True
    cfg.resources.bg_low_prod_frac = P["bg_low_prod_frac"]
    cfg.resources.bg_cap_mult = P["bg_low_cap_mult"]
    cfg.population.initial_count = P["pop"]
    cfg.population.max_count = max(P["pop"] * 2, 20000)
    cfg.simulation.use_sim_core = False
    cfg.env_field = EnvFieldConfig(enabled=True, c0b_enabled=True) if atm_on else EnvFieldConfig()
    cfg.atmosphere = AtmosphereConfig(enabled=atm_on, moisture_on=moisture)
    return SphereEngine(cfg)


def _calib(target_s: float = 0.2) -> float:
    """校准核：固定 200 ms 的 256×256 matmul，返回**CPU 时间**实际耗时（秒）。

    用途：把 ms/tick 归一化到"同一台机器、同一负载水平"——共享机上墙钟没有 0.3 点分辨率，
    但"这段时间里机器跑 matmul 快了多少"可以量出来（B5.1 裁定②）。
    """
    n = 256
    a = np.ones((n, n), dtype=np.float64)
    t0 = time.process_time()
    while time.process_time() - t0 < target_s:
        a @ a
    return time.process_time() - t0


def measure_once(atm_on: bool, moisture: bool, ticks: int) -> dict:
    """单次测量：墙钟 + **CPU 时间** + 归一化 CPU（ms/tick）。

    归一化 = cpu_ms_per_tick × (CALIB_REF / 本次校准耗时)：
    机器忙时校准核也变慢 ⇒ 比值抵消负载漂移。
    """
    c0 = _calib()
    eng = build(atm_on, moisture)
    for _ in range(WARMUP):
        eng.step()
    w0, p0 = time.perf_counter(), time.process_time()
    for _ in range(ticks):
        eng.step()
    wall = (time.perf_counter() - w0) / ticks * 1e3
    cpu = (time.process_time() - p0) / ticks * 1e3
    c1 = _calib()
    cal = 0.5 * (c0 + c1)
    return {"wall_ms": wall, "cpu_ms": cpu,
            "norm_ms": cpu * (CALIB_REF / cal), "calib_s": cal}


def summarize(samples: list[dict]) -> dict:
    out = {}
    for key, name in (("wall_ms", "wall"), ("cpu_ms", "cpu"), ("norm_ms", "norm")):
        vals = [s[key] for s in samples]
        med = statistics.median(vals)
        q1, q3 = np.percentile(vals, [25, 75])
        out[f"median_{name}_ms"] = med
        out[f"iqr_{name}_ms"] = float(q3 - q1)
        out[f"samples_{name}_ms"] = vals
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=TICKS)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--cpu", type=int, default=None, help="CPU 亲和（需 psutil）")
    ap.add_argument("--baseline", default=None, help="旧 JSON，报相对百分比")
    ap.add_argument("--out", default=str(ROOT / "_rerun_logs" / "bench_tick.json"))
    a = ap.parse_args()

    if a.cpu is not None:
        try:
            import psutil
            psutil.Process().cpu_affinity([a.cpu])
            print(f"[bench] CPU 亲和 -> {a.cpu}")
        except Exception as exc:  # noqa: BLE001
            print(f"[bench][警告] CPU 亲和设置失败（{exc}），继续无亲和测量",
                  file=sys.stderr)

    global CALIB_REF
    CALIB_REF = _calib()                    # 本次会话的参考校准（同机器同负载水平的锚）
    print(f"[bench] 校准参考 {CALIB_REF*1000:.1f} ms（256×256 matmul × N）")
    modes = {"off": (False, False), "on": (True, True)}
    raw: dict[str, list[dict]] = {k: [] for k in modes}
    # 🔴 交替测量（round-robin）：每轮先 off 再 on，机器负载漂移对两组**同权**。
    # 顺序测（先跑完一组再跑另一组）会被"这一小时机器变快了/慢了"直接污染
    # ——B2.2 第一次协议就是这么得到 off −14 %、on +5.9 % 的假差。
    for _ in range(a.repeats):
        for name, (atm_on, moist) in modes.items():
            raw[name].append(measure_once(atm_on, moist, a.ticks))
    out = {"timestamp": datetime.now().isoformat(timespec="seconds"),
           "host": platform.node(), "python": platform.python_version(),
           "device": "s2", "seed": SEED, "ticks": a.ticks, "repeats": a.repeats,
           "commit": _git_head(), "arms": {}}
    for name in modes:
        r = summarize(raw[name])
        out["arms"][name] = r
        print(f"[bench] {name}: 归一化 CPU {r['median_norm_ms']:.3f} ms/tick "
              f"(IQR {r['iqr_norm_ms']:.3f}) | CPU {r['median_cpu_ms']:.3f} | "
              f"墙钟 {r['median_wall_ms']:.3f}")

    if a.baseline and Path(a.baseline).exists():
        base = json.loads(Path(a.baseline).read_text(encoding="utf-8"))
        pct = {}
        for name, r in out["arms"].items():
            b = base.get("arms", {}).get(name, {}).get("median_norm_ms")
            if b:
                pct[name] = (r["median_norm_ms"] / b - 1.0) * 100.0
                print(f"[bench] {name} vs baseline: {pct[name]:+.2f}%")
        out["vs_baseline_pct"] = pct

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[bench] -> {p}")


def _git_head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


if __name__ == "__main__":
    main()
