"""T-C 证据②：默认配置短跑（≤2k tick）改动前后逐位对拍（一次性脚本，不入仓）。

口径（R230 T-C 派工）：
  · 全默认 `SimConfig(seed=42)`（**不覆写任何字段** ⇒ info_structure.enabled=False 等全默认）
  · 每 tick 把 (tick, N, sha(flat/genes/grid/marks/age), Σenergy, Σstomach) 串进 sha256 链
  · 末行附全局 np.random 状态指纹（证明默认路径不消费进程级全局流）

用法：
    .venv\\Scripts\\python.exe _trash_local/_tc_default_digest.py [TICKS]   # 默认 2000
改动前/后各跑一次，比较输出的 chain / flat_sum / energy / rng_draws / global_sha 五项。
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from simulation.config import SimConfig            # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def main() -> int:
    ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    cfg = SimConfig(seed=42)                      # 全默认，不覆写
    print(f"cfg: use_sim_core={cfg.simulation.use_sim_core} "
          f"sparse_fields={cfg.simulation.sparse_fields} "
          f"info_structure.enabled={cfg.info_structure.enabled} "
          f"rd.enabled={cfg.resource_dynamics.enabled} seed={cfg.seed}")
    e = SphereEngine(cfg)
    h = hashlib.sha256()
    t0 = time.perf_counter()
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
        h.update(repr((
            e._tick, len(e._id), sha(e._flat), sha(e._genes),
            sha(e.resources._grid), sha(e.signals._marks), sha(e.signals._age),
            round(float(e._energy.sum()), 6), round(float(e._stomach.sum()), 6),
        )).encode("ascii"))
    dt = time.perf_counter() - t0
    g = np.random.get_state()
    gsha = hashlib.sha256(np.ascontiguousarray(g[1]).tobytes()).hexdigest()[:16]
    print(f"ticks={e._tick} N={len(e._id)} chain={h.hexdigest()[:32]} "
          f"flat_sum={int(e._flat.sum())} energy={float(e._energy.sum()):.6f} "
          f"stomach={float(e._stomach.sum()):.6f} rng_draws={e.rng_draws} "
          f"global_sha={gsha} wall={dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
