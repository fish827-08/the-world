"""benchmark_sim_core：use_sim_core 性能基准（模块三 · 3.4 收尾）。

目的
----
量化"把热路径下沉到 Rust 到底快多少"。同一种子同一配置，跑两个引擎：
  - Python 路径（use_sim_core=False，默认）；
  - Rust 路径（use_sim_core=True：regrow + step_vectors stage1/stage2 全下沉）。
分别计时，再互相比对最终状态（种群/能量/资源总量），确认"提速但行为不变"。

用法
----
    py -m experiments.benchmark_sim_core [-n 2000] [-t 300] [-r 60] [-c 120] [-s 42]

依赖
----
- simulation.sphere_engine.SphereEngine / simulation.config.SimConfig
- sim_core（需要先构建：sim_core/ 下运行 .venv\\Scripts\\python -m maturin develop）
"""
from __future__ import annotations

import argparse
import time

from simulation.config import SimConfig
from simulation.sphere_engine import SphereEngine


def build_engine(seed: int, n: int, rows: int, cols: int, ticks: int, use_sim_core: bool):
    c = SimConfig(seed=seed)
    c.population.initial_count = n
    c.world.rows = rows
    c.world.cols = cols
    c.simulation.ticks = ticks
    c.simulation.use_sim_core = use_sim_core
    return SphereEngine(c)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-n", type=int, default=2000, help="初始个体数（默认 2000）")
    ap.add_argument("-t", type=int, default=300, help="运行 tick 数（默认 300）")
    ap.add_argument("-r", type=int, default=60, help="纬度网格数（默认 60）")
    ap.add_argument("-c", type=int, default=120, help="经度网格数（默认 120）")
    ap.add_argument("-s", type=int, default=42, help="随机种子（默认 42）")
    args = ap.parse_args()

    results = {}
    for label, use_rs in (("python", False), ("rust", True)):
        e = build_engine(args.s, args.n, args.r, args.c, args.t, use_rs)
        t0 = time.perf_counter()
        e.run(args.t)
        dt = time.perf_counter() - t0
        results[label] = {
            "seconds": dt,
            "per_tick_ms": dt / args.t * 1000.0,
            "population": e.alive_count(),
            "total_energy": float(e._energy.sum()),
            "total_resource": e.resources.total(),
            "extinct": e.extinct,
        }
        print(f"[{label:<6}] {args.t} ticks × {args.n} 初始个体: {dt:8.3f} s "
              f"({dt / args.t * 1000:7.3f} ms/tick)")

    py_, rs_ = results["python"], results["rust"]
    speedup = py_["seconds"] / rs_["seconds"] if rs_["seconds"] > 0 else float("inf")
    print(f"\n提速 {speedup:.2f}x（python {py_['seconds']:.3f}s → rust {rs_['seconds']:.3f}s）")

    # 行为一致性校验：加速的前提是结果逐位一致
    checks = {
        "population": py_["population"] == rs_["population"],
        "total_energy(位级)": py_["total_energy"] == rs_["total_energy"],
        "total_resource(位级)": py_["total_resource"] == rs_["total_resource"],
        "extinct": py_["extinct"] == rs_["extinct"],
    }
    ok = all(checks.values())
    for k, v in checks.items():
        print(f"  一致[{k}] = {v}  (py={py_.get(k.split('(')[0], None)}, "
              f"rs={rs_.get(k.split('(')[0], None)})")
    print("REGRESSION CHECK:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()