"""biosphere.parallel：多进程 seed/臂并行跑批基建（B16.0）。

通俗理解
--------
跑批的瓶颈是"一个 seed 一个臂占一个核串行跑"。本模块把**每个 (臂, seed)
作业**丢进进程池，14 核 ⇒ 理论 ~14×。两条硬纪律：

1. **并行不改变结果**：每个作业是独立模拟器（独立 seed、独立 RNG），
   进程之间零通信 ⇒ 并行结果与串行结果**逐位一致**（有测试钉住）；
2. **结果合并 + manifest**：每个作业返回一行纯 JSON-able 读数，合并成
   一个 manifest（含 workers / wall / 吞吐 / 逐作业耗时），读数可追溯。

用法（仓库/worktree 根）::

    python -m biosphere.parallel --bench --n 14 --ticks 1000
    python -m biosphere.experiments b16 --seeds 41-45 --ticks 5000 --parallel 14
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA = "biosphere-parallel-v0.1"


@dataclass(frozen=True)
class JobSpec:
    """一个跑批作业 = (臂, seed)（可 pickle ⇒ spawn 也能送进子进程）。

    Attributes:
        arm: 臂名（在 experiments 的臂表里查）。
        seed: 随机种子。
        ticks: 总 tick。
        tag: 文件/日志前缀（空 = 用 arm_seed 拼）。
        ckpt_dir: checkpoint 目录（空 = 不写）。
        ckpt_every: 每多少 tick 落一个 checkpoint（0 = 不写）。
        kind: 作业类型（"b16" = experiments.run_job；"b17_diag" = diagnose）。
        extra: 透传给臂构造的额外参数（纯标量）。
    """

    arm: str
    seed: int
    ticks: int = 1000
    tag: str = ""
    ckpt_dir: str = ""
    ckpt_every: int = 0
    kind: str = "b16"
    extra: Dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        return f"{self.arm}_s{self.seed}"


def _plain(obj: Any) -> Any:
    """把 numpy 标量/数组转成纯 Python（pickle + json 双友好）。"""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _run_one(job: JobSpec) -> Dict:
    """子进程体：跑一个作业，返回一行结果（异常也回行，不炸池）。"""
    t0 = time.perf_counter()
    try:                                   # 延迟导入防环（experiments 引本模块）
        if job.kind == "b17_diag":
            from biosphere.diagnose import run_diagnosis_job
            row = run_diagnosis_job(job)
        else:
            from biosphere.experiments import run_job
            row = run_job(job)
        err = ""
        ok = True
    except Exception as exc:               # pragma: no cover - 兜底
        row = {}
        err = f"{type(exc).__name__}: {exc}"
        ok = False
    return {
        "arm": job.arm, "seed": int(job.seed), "ticks": int(job.ticks),
        "label": job.label(), "ok": ok, "error": err,
        "wall_s": float(time.perf_counter() - t0),
        "row": _plain(row),
    }


def _default_workers(n_jobs: int) -> int:
    cpu = os.cpu_count() or 1
    return max(1, min(n_jobs, cpu))


def run_serial(jobs: Sequence[JobSpec]) -> List[Dict]:
    """串行跑（对照基准；并行结果须与它逐位一致）。"""
    return [_run_one(j) for j in jobs]


def run_parallel(jobs: Sequence[JobSpec], workers: Optional[int] = None,
                 ctx: Optional[str] = None) -> List[Dict]:
    """进程池并行跑（Windows 默认 spawn，Linux 默认 fork）。"""
    jobs = list(jobs)
    if not jobs:
        return []
    n_workers = int(workers) if workers else _default_workers(len(jobs))
    if ctx is None:
        ctx = "spawn" if os.name == "nt" else "fork"
    mp_ctx = mp.get_context(ctx)
    if n_workers <= 1:
        return run_serial(jobs)
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as pool:
        return list(pool.map(_run_one, jobs))


def manifest(jobs: Sequence[JobSpec], results: List[Dict], workers: int,
             wall_s: float, ctx: str, serial_wall_s: Optional[float] = None,
             extra: Optional[Dict] = None) -> Dict:
    """合并结果 → manifest（定键、sort_keys ⇒ 逐位可复现）。"""
    total_ticks = sum(int(j.ticks) for j in jobs)
    speedup = (serial_wall_s / wall_s) if (serial_wall_s and wall_s) else None
    out: Dict = {
        "schema": SCHEMA,
        "n_jobs": len(jobs),
        "workers": int(workers),
        "mp_context": ctx,
        "cpu_count": int(os.cpu_count() or 1),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "wall_s": float(wall_s),
        "serial_wall_s": (float(serial_wall_s)
                          if serial_wall_s is not None else None),
        "speedup_vs_serial": (float(speedup)
                              if speedup is not None else None),
        "throughput_jobs_per_s": (len(jobs) / wall_s) if wall_s else 0.0,
        "throughput_ticks_per_s": (total_ticks / wall_s) if wall_s else 0.0,
        "n_ok": sum(1 for r in results if r["ok"]),
        "n_failed": sum(1 for r in results if not r["ok"]),
        "jobs": [{"arm": j.arm, "seed": j.seed, "ticks": j.ticks}
                 for j in jobs],
        "wall_per_job": {r["label"]: r["wall_s"] for r in results},
        "results": results,
    }
    if extra:
        out["extra"] = extra
    return out


def write_manifest(path: Path, man: Dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(man, fh, sort_keys=True, indent=1)
        fh.write("\n")
    return path


def canonical_sha(obj: Any) -> str:
    """结果集合的指纹（并行 vs 串行逐位一致的判据）。"""
    import hashlib
    return hashlib.sha256(
        json.dumps(_plain(obj), sort_keys=True, ensure_ascii=False,
                   separators=(",", ":")).encode("utf-8")).hexdigest()


def bench(n_jobs: int = 14, ticks: int = 1000, arm: str = "b16_natural",
          seed0: int = 41, workers: Optional[int] = None) -> Dict:
    """小跑验证吞吐：同作业集串行 vs 并行，报加速比 + 结果指纹是否一致。"""
    jobs = [JobSpec(arm=arm, seed=seed0 + i, ticks=ticks)
            for i in range(n_jobs)]
    t0 = time.perf_counter()
    ser = run_serial(jobs)
    serial_wall = time.perf_counter() - t0
    t1 = time.perf_counter()
    par = run_parallel(jobs, workers=workers)
    wall = time.perf_counter() - t1
    man = manifest(jobs, par, workers or _default_workers(n_jobs), wall,
                   "spawn" if os.name == "nt" else "fork",
                   serial_wall_s=serial_wall)
    man["determinism"] = {
        "serial_sha": canonical_sha([r["row"] for r in ser]),
        "parallel_sha": canonical_sha([r["row"] for r in par]),
        "identical": canonical_sha([r["row"] for r in ser])
                     == canonical_sha([r["row"] for r in par]),
    }
    print(f"=== parallel bench: {n_jobs} jobs x {ticks} ticks ===")
    print(f"serial   {serial_wall:8.2f} s  ({n_jobs / serial_wall:6.3f} jobs/s)")
    print(f"parallel {wall:8.2f} s  ({n_jobs / wall:6.3f} jobs/s)  "
          f"speedup = {serial_wall / wall:.2f}x  "
          f"workers = {man['workers']} / cpu {man['cpu_count']}")
    print(f"determinism: serial==parallel = {man['determinism']['identical']}"
          f"  sha={man['determinism']['parallel_sha'][:16]}…")
    return man


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="B16.0 并行跑批基建自检")
    ap.add_argument("--bench", action="store_true", help="跑吞吐对比")
    ap.add_argument("--n", type=int, default=14, help="作业数（默认 14）")
    ap.add_argument("--ticks", type=int, default=1000)
    ap.add_argument("--arm", default="b16_natural")
    ap.add_argument("--seed0", type=int, default=41)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="results/biosphere/b16_parallel_bench.json")
    a = ap.parse_args(argv)
    if not a.bench:
        ap.print_help()
        return 0
    man = bench(n_jobs=a.n, ticks=a.ticks, arm=a.arm, seed0=a.seed0,
                workers=a.workers)
    p = write_manifest(Path(a.out), man)
    print(f"manifest: {p}")
    return 0


__all__ = [
    "SCHEMA", "JobSpec", "run_serial", "run_parallel", "manifest",
    "write_manifest", "canonical_sha", "bench",
]


if __name__ == "__main__":                # pragma: no cover
    raise SystemExit(main())
