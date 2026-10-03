"""本机硬件档案探针（[本地开发·性能线] 开工用；scratch，不入库）。

目的：量单 run 的 构造时间 / ms-per-tick / 峰值 RSS（Windows: psapi.PeakWorkingSetSize）。
每个配置**独立子进程**跑一次（峰值 RSS 只增不减，必须分进程）。

用法：
    .venv\\Scripts\\python.exe _trash_local/_perfline_hw_probe.py --rows 60  --cols 120  --pop 500   --ticks 60
    .venv\\Scripts\\python.exe _trash_local/_perfline_hw_probe.py --rows 480 --cols 960 --pop 5000  --ticks 30
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --- R98：Windows GBK 控制台兜底 ---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from simulation.config import SimConfig            # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402


class _PMC(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_psapi = ctypes.WinDLL("psapi", use_last_error=True)
_k32.GetCurrentProcess.restype = ctypes.c_void_p
_psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC), ctypes.c_ulong]
_psapi.GetProcessMemoryInfo.restype = ctypes.c_bool


def rss_mb() -> tuple[float, float]:
    """返回 (当前 RSS MB, 峰值 RSS MB)。Windows：psapi（显式 argtypes，防句柄截断）。"""
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(pmc)
    h = _k32.GetCurrentProcess()
    ok = _psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
    if not ok:
        raise OSError(f"GetProcessMemoryInfo 失败：err={ctypes.get_last_error()}")
    return (pmc.WorkingSetSize / 1e6, pmc.PeakWorkingSetSize / 1e6)


def make_cfg(seed: int, rows: int, cols: int, pop: int, sim_core: bool) -> SimConfig:
    c = SimConfig(seed=seed)
    c.world.rows, c.world.cols = rows, cols
    c.simulation.use_sim_core = bool(sim_core)
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = True
    c.resources.patch_regrowth_mult = 1.195
    c.resources.patch_count = max(4, int(round(30 * rows * cols / 7200)))
    if pop > 0:
        c.population.initial_count = pop
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description="本机硬件档案探针（性能线）")
    ap.add_argument("--rows", type=int, default=60)
    ap.add_argument("--cols", type=int, default=120)
    ap.add_argument("--pop", type=int, default=500)
    ap.add_argument("--ticks", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sim-core", action="store_true", help="走 Rust 路径（默认 Python）")
    a = ap.parse_args()

    rss0, _ = rss_mb()
    t0 = time.perf_counter()
    eng = SphereEngine(make_cfg(a.seed, a.rows, a.cols, a.pop, a.sim_core))
    t_build = time.perf_counter() - t0
    rss1, _ = rss_mb()
    nb_mb = eng._nb_table.nbytes / 1e6
    P = int(len(eng._flat))

    for _ in range(3):                     # 预热
        eng.step()
    print(f"[{'Rust' if a.sim_core else 'Py  '}] {a.rows}x{a.cols} pop={a.pop} "
          f"构造 {t_build:.2f}s，开始 {a.ticks} tick（进度 + ETA）：", flush=True)
    t0 = time.perf_counter()
    done = 0
    for i in range(a.ticks):
        if eng.extinct:
            break
        eng.step()
        done = i + 1
        if done % max(1, a.ticks // 10) == 0 or done == a.ticks:
            el = time.perf_counter() - t0
            eta = el / done * (a.ticks - done)
            print(f"    tick {done:>3}/{a.ticks}  已用 {el:>6.1f}s  ETA {eta:>6.1f}s", flush=True)
    dt = (time.perf_counter() - t0) / done

    _, peak = rss_mb()
    print(f"[{'Rust' if a.sim_core else 'Py  '}] {a.rows}x{a.cols} pop={a.pop:>5} "
          f"| N={P:>5} | 构造 {t_build:>5.2f}s | {dt*1e3:>7.2f} ms/tick | {1/dt:>7.1f} t/s "
          f"| nb_table {nb_mb:>7.1f} MB | RSS {rss0:>6.0f}->{rss1:>6.0f} 峰值 {peak:>7.0f} MB")


if __name__ == "__main__":
    main()
