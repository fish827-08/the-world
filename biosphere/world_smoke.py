"""biosphere.world_smoke：真实世界字段适配冒烟（B7；不跑生物）。

两条路（都**只读**世界侧对象，不改 `world/`、`simulation/`）：

1. **engine 路**（适配真源）：构造真实 `SphereEngine`（env + C0b 开），
   把 `eng.world / eng.env_field / eng.resources / eng.light` 交给
   `world_adapter.adapt_real_world` ⇒ `WorldView`，打印字段统计并做
   形状/不变量校验。这才是"接真实世界字段"的证明。
2. **npz 路**（run_bare_world 静态场）：读 `experiments/run_bare_world.py`
   产出的 npz（capacity/patch_mask/h/w/temp/acc/river_mask），打印数组
   形状与统计。npz **不含** surface_type/海平面/季节参数 ⇒ 只做统计与
   形状校验，不构造 WorldView（缺字段按 fail-loud 纪律宁缺勿造）。

用法::

    python -m biosphere.world_smoke engine --rows 120 --cols 240 --ticks 0
    python -m biosphere.world_smoke bio --rows 120 --cols 240 --ticks 1000
    python -m biosphere.world_smoke npz --path results/biosphere/bare_world/arm4_s42.npz

产出 s2 静态场（一条，可选；~10 s）::

    python experiments/run_bare_world.py --arm 4 --device s2 --seed 42 \\
        --ticks 2 --out results/biosphere/bare_world
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from biosphere.world_adapter import (
    SURFACE_LAND, SURFACE_OCEAN, SURFACE_RIVER, WATER_SURFACES,
    WorldView, adapt_real_resource, adapt_real_world,
)

#: surface_type 编码名（与 world/env_field.py 的 C0b 编码一致）
_SURFACE_NAMES = {
    0: "LAND", 1: "OCEAN", 2: "SHELF", 3: "RIVER",
    4: "LAKE_FRESH", 5: "LAKE_SALT",
}


def build_real_engine(rows: int = 120, cols: int = 240, patches: int = 60,
                      rgm: float = 1.195, seed: int = 42, ticks: int = 0,
                      season: bool = True):
    """构造真实引擎（env + C0b 开；无生物；只读用途）。

    season=True 时显式开季节（tilt=0.41 rad ≈ 23.5°、周期 400 tick），
    让 `WorldView.season_phase` 的读数非平凡（默认世界季节是关的）。
    """
    from simulation.config import SimConfig
    from simulation.sphere_engine import SphereEngine

    cfg = SimConfig(seed=seed)
    cfg.world.rows, cfg.world.cols = int(rows), int(cols)
    cfg.simulation.use_sim_core = False
    cfg.resources.distribution = "patchy"
    cfg.resources.patch_count = int(patches)
    cfg.resources.bg_production_zero = True
    cfg.resources.patch_regrowth_mult = float(rgm)
    cfg.population.initial_count = 0
    if season:
        cfg.light.tilt_rad = 0.41        # ≈ 地球黄赤交角
        cfg.light.season_period = 400
    e = cfg.env_field
    e.enabled = True
    e.c0b_enabled = True          # ⇒ surface_type / h_m / sea_level_m 就绪
    eng = SphereEngine(cfg)
    for _ in range(int(ticks)):
        eng.step()
    return eng


def smoke_engine(rows: int = 120, cols: int = 240, patches: int = 60,
                 seed: int = 42, ticks: int = 0) -> Dict:
    """engine 路：适配真实字段 + 统计 + 不变量校验（失败即抛）。"""
    eng = build_real_engine(rows=rows, cols=cols, patches=patches,
                            seed=seed, ticks=ticks)
    view = adapt_real_world(eng.world, eng.env_field, eng.resources, eng.light)

    # ── 形状校验 ──
    n = rows * cols
    assert view.n_cells == n == view.rows * view.cols
    for name in ("surface_type", "elevation_m", "depth_m", "patch_id"):
        assert len(getattr(view, name)) == n, f"{name} 长度不符"
    assert len(view.neighbors) == n

    # ── 不变量 ──
    # 注意：EnvField 的 ocean 判据是 h_m <= sea_level_m ⇒ 恰在海平面的
    # 边界格是"水深 0 的水生格"；故水生于 [0,∞)、陆地恒 0。
    water = view.water_mask()
    assert bool(np.all(view.depth_m >= 0.0)), "水深不能为负"
    assert bool(np.all(view.depth_m[~water] == 0.0)), "陆格水深必须 = 0"
    deep_frac = float(np.mean(view.depth_m[water] > 0.0)) if water.any() else 1.0
    assert deep_frac > 0.99, f"水深 > 0 的水生格比例过低: {deep_frac:.4f}"
    assert bool(np.all(view.elevation_m[~water] > 0.0)), "陆格海拔必须 > 0"
    assert int(view.patch_id.min()) >= -1
    phase0 = view.season_phase(0)
    phase1 = view.season_phase(max(1, view.season_period // 4))
    assert 0.0 <= phase0 < 1.0 and 0.0 <= phase1 < 1.0

    st = view.surface_type
    counts = {_SURFACE_NAMES.get(int(v), str(v)): int((st == v).sum())
              for v in np.unique(st)}
    n_patches = int(view.patch_id.max()) + 1
    stats: Dict = {
        "route": "engine",
        "world": {"rows": rows, "cols": cols, "n_cells": n},
        "surface_type_counts": counts,
        "water_fraction": float(np.mean(water)),
        "elevation_m": {"min": float(view.elevation_m.min()),
                        "max": float(view.elevation_m.max()),
                        "mean": float(view.elevation_m.mean())},
        "depth_m": {"max": float(view.depth_m.max()),
                    "mean_on_water": float(view.depth_m[water].mean())
                    if water.any() else 0.0,
                    "deep_fraction_on_water": deep_frac},
        "patches": {"n_patches": n_patches,
                    "patch_cells": int((view.patch_id >= 0).sum()),
                    "bg_cells": int((view.patch_id == -1).sum())},
        "season": {"tilt_rad": view.tilt_rad,
                   "period": view.season_period,
                   "phase_at_0": phase0, "phase_at_quarter": phase1},
        "neighbor_count_range": [int(min(len(x) for x in view.neighbors)),
                                 int(max(len(x) for x in view.neighbors))],
    }
    print("=== B7 world_smoke: engine route ===")
    print(json.dumps(stats, indent=1, sort_keys=True))
    print("invariants: shape OK / depth-water OK / elevation-land OK / "
          "patch_id >= -1 OK / season_phase in [0,1) OK")
    return stats


def _trait_stats(sim, individuals) -> Dict:
    """当前个体的控制器性状统计（空种群返回 None 而非 0）。"""
    if not individuals:
        return {"n": 0, "defense_mean": None, "defense_sd": None,
                "attack_mean": None, "attack_sd": None}
    cs = [sim.cache.get(ind.genome) for ind in individuals]
    d = np.array([c.defense for c in cs], dtype=np.float64)
    a = np.array([c.attack_drive for c in cs], dtype=np.float64)
    return {"n": int(d.size), "defense_mean": float(d.mean()),
            "defense_sd": float(d.std()), "attack_mean": float(a.mean()),
            "attack_sd": float(a.std())}


def smoke_bio(rows: int = 120, cols: int = 240, patches: int = 60,
              seeds: List[int] = None, ticks: int = 1000,
              init_pop: int = 48, out: Optional[Path] = None) -> Dict:
    """B17.3：把真实世界字段注入生物层，跑 ``ticks`` 个 tick。

    每个 seed 独立构造真实引擎（env+C0b 开）⇒ ``adapt_real_world`` +
    ``adapt_real_resource``（均只读快照）⇒ ``BiosphereSimulator`` 注入运行。
    产出的是**集成读数**（能否连通运行/存活/流量），不是生态判据：资源单位
    尚未与生物能量标定（A3 耦合点）。
    """
    from biosphere.simulator import BioMVPConfig, BiosphereSimulator

    seeds = [42, 43, 44] if seeds is None else [int(s) for s in seeds]
    per_seed: List[Dict] = []
    for seed in seeds:
        eng = build_real_engine(rows=rows, cols=cols, patches=patches,
                                seed=seed, ticks=0, season=True)
        view = adapt_real_world(eng.world, eng.env_field, eng.resources,
                                eng.light)
        res = adapt_real_resource(eng.resources)
        cfg = BioMVPConfig(rows=rows, cols=cols, seed=seed, ticks=ticks,
                           init_pop=init_pop)
        sim = BiosphereSimulator(cfg, world=view, resource=res)
        res0 = float(res.amount.sum())

        inds0 = list(sim.population.individuals)
        cells0 = np.array([ind.cell for ind in inds0], dtype=np.int64)
        on_patch0 = (view.patch_id[cells0] >= 0) if cells0.size else \
            np.zeros(0, dtype=bool)
        wet0 = ((view.surface_type[cells0] == SURFACE_LAND)
                & (view.soil_water[cells0] >= view.soil_water_min)
                if cells0.size else np.zeros(0, dtype=bool))
        init = {
            "alive": len(inds0),
            "energy_mean": float(np.mean([i.energy for i in inds0]))
            if inds0 else None,
            "on_patch_frac": float(on_patch0.mean()) if cells0.size else None,
            "wet_land_frac": float(wet0.mean()) if cells0.size else None,
            "traits": _trait_stats(sim, inds0),
        }

        log = sim.run(ticks)
        final = list(sim.population.individuals)
        alive_by_tick = {int(f.tick): int(f.uids.size) for f in log.frames}
        probes = [0, ticks // 4, ticks // 2, 3 * ticks // 4, ticks - 1]
        first_zero = next((t for t in sorted(alive_by_tick)
                           if alive_by_tick[t] == 0), None)
        land = view.surface_type == SURFACE_LAND
        water = view.water_mask()
        patch = view.patch_id >= 0
        row = {
            "seed": seed,
            "world": {
                "n_cells": int(view.n_cells),
                "land_cells": int(land.sum()),
                "land_frac": float(land.mean()),
                "water_fraction": float(water.mean()),
                "patch_cells": int(patch.sum()),
                "patch_frac": float(patch.mean()),
                "soil_water_on_land_mean": float(view.soil_water[land].mean())
                if land.any() else None,
                "wet_land_frac": float(
                    (view.soil_water[land] >= view.soil_water_min).mean())
                if land.any() else None,
                "season_period": int(view.season_period),
            },
            "initial": init,
            "final": {
                "alive": len(final),
                "energy_mean": float(np.mean([i.energy for i in final]))
                if final else None,
                "traits": _trait_stats(sim, final),
            },
            "alive_series": {str(t): alive_by_tick.get(t) for t in probes},
            "min_alive": int(min(alive_by_tick.values())) if alive_by_tick
            else 0,
            "max_alive": int(max(alive_by_tick.values())) if alive_by_tick
            else 0,
            "first_zero_tick": first_zero,
            "births": len(log.births),
            "deaths_total": len(log.deaths),
            "deaths_by_cause": dict(Counter(c for _u, c, _t in log.deaths)),
            "ledger": {
                "energy_in": float(log.energy_in),
                "energy_paid": float(log.energy_paid),
                "resource0_sum": res0,
                "resource_end_sum": float(res.amount.sum()),
                "resource_consumed_net": res0 - float(res.amount.sum()),
            },
        }
        per_seed.append(row)
        print(f"[bio] seed={seed} alive {init['alive']}→{len(final)} "
              f"births={len(log.births)} deaths={len(log.deaths)} "
              f"min={row['min_alive']}")

    summary = {
        "route": "bio",
        "params": {"rows": rows, "cols": cols, "patches": patches,
                   "seeds": seeds, "ticks": ticks, "init_pop": init_pop},
        "seeds": per_seed,
        "alive_end": {str(r["seed"]): r["final"]["alive"] for r in per_seed},
        "n_alive_end": sum(1 for r in per_seed if r["final"]["alive"] > 0),
        "all_alive_end": all(r["final"]["alive"] > 0 for r in per_seed),
    }
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                       encoding="utf-8")
    print("=== B17.3 world_smoke: bio route ===")
    print(json.dumps({k: summary[k] for k in
                      ("params", "alive_end", "n_alive_end",
                       "all_alive_end")}, indent=1, ensure_ascii=False))
    print("note: 集成冒烟读数（非生态判据）；资源单位未标定（A3 耦合点）")
    return summary


def smoke_npz(path: Path) -> Dict:
    """npz 路：run_bare_world 静态场的形状与统计（不构造 WorldView）。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"npz 不存在: {path}（先用 run_bare_world.py 产出）")
    data = np.load(path)
    expected = {"capacity", "patch_mask", "h", "w", "temp", "acc", "river_mask"}
    missing = expected - set(data.files)
    if missing:
        raise ValueError(f"npz 缺字段: {sorted(missing)}（期望 {sorted(expected)}）")
    n = int(data["h"].shape[0])
    stats: Dict = {"route": "npz", "path": str(path), "n_cells": n,
                   "arrays": {}}
    for key in sorted(data.files):
        arr = data[key]
        entry = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        if arr.dtype.kind in "fiu" and arr.size:
            entry["min"] = float(arr.min())
            entry["max"] = float(arr.max())
            entry["mean"] = float(arr.mean())
        stats["arrays"][key] = entry
    # 形状互证：所有逐格字段同长
    for key in expected:
        assert data[key].shape[0] == n, f"{key} 与 h 不同长"
    patch_cells = int(data["patch_mask"].sum())
    stats["patch_cells"] = patch_cells
    stats["patch_fraction"] = patch_cells / n
    stats["river_cells"] = int(data["river_mask"].sum())
    print("=== B7 world_smoke: npz route ===")
    print(json.dumps(stats, indent=1, sort_keys=True))
    print("shape check: all per-cell arrays length", n, "OK")
    return stats


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m biosphere.world_smoke",
        description="B7 真实世界字段适配冒烟（只读；不跑生物）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_eng = sub.add_parser("engine", help="构造真实引擎并适配 WorldView")
    p_eng.add_argument("--rows", type=int, default=120)
    p_eng.add_argument("--cols", type=int, default=240)
    p_eng.add_argument("--patches", type=int, default=60)
    p_eng.add_argument("--seed", type=int, default=42)
    p_eng.add_argument("--ticks", type=int, default=0)
    p_bio = sub.add_parser("bio", help="真世界字段上跑生物（B17.3 集成冒烟）")
    p_bio.add_argument("--rows", type=int, default=120)
    p_bio.add_argument("--cols", type=int, default=240)
    p_bio.add_argument("--patches", type=int, default=60)
    p_bio.add_argument("--seeds", type=str, default="42,43,44")
    p_bio.add_argument("--ticks", type=int, default=1000)
    p_bio.add_argument("--init-pop", type=int, default=48)
    p_bio.add_argument("--out", type=str,
                       default="results/biosphere/b17_3_smoke.json")
    p_npz = sub.add_parser("npz", help="run_bare_world 静态场统计")
    p_npz.add_argument("--path", type=str, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "engine":
        smoke_engine(rows=args.rows, cols=args.cols, patches=args.patches,
                     seed=args.seed, ticks=args.ticks)
    elif args.cmd == "bio":
        smoke_bio(rows=args.rows, cols=args.cols, patches=args.patches,
                  seeds=[int(s) for s in args.seeds.split(",") if s.strip()],
                  ticks=args.ticks, init_pop=args.init_pop,
                  out=Path(args.out))
    else:
        smoke_npz(Path(args.path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
