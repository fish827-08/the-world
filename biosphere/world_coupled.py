"""biosphere.world_coupled：B18 生物进真世界联调（A3 合成天气 + A18/A20 世界族）。

预注册：``docs/预注册/预注册-B18-生物进真世界联调-20261020.md``

⚠️ **天气源 = synthetic（A3 当前核，``placeholder=True``）**：本模块产出的
t2m/precip 是合成占位值，不是天气/生态结论。

驱动口径（A3 规定顺序）::

    source.on_tick(engine.tick + 1)     # 交换相位 t=25,50,... 推进 atmo
    engine.step()

B18 在每个交换相位把世界 ``ResourceField._capacity`` **只读同步**进生物资源
（只同步容量，不同步存量），并采样 A3 的 t2m/precip 陆格统计。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from biosphere.convergence import (
    K_EPS, K_N, K_T0, K_W,
    batch_k_convergence_verdict, k_convergence_verdict,
)
from biosphere.metrics import SimLog, metrics_json, run_digest
from biosphere.resolve import MAX_POP, ResolveConfig
from biosphere.simulator import BioMVPConfig, BiosphereSimulator
from biosphere.world_adapter import (
    SURFACE_LAND, adapt_real_resource, adapt_real_world, sample_climate,
    sync_real_capacity,
)

#: A20 s1 multiplicative seed42 容量锚（汇报-20261020-A20.md §4.1）
A20_S1_MULT_CAP_SUM: float = 247280.9
A20_S1_MULT_CAP_P50: float = 4.181

B18_ROWS: int = 120
B18_COLS: int = 240
B18_SEASON_PERIOD: int = 1200
B18_TILT_RAD: float = 0.409
B18_TICKS: int = 2000
B18_INIT_POP: int = 200
B18_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
B18_NLAT: int = 32
B18_NLON: int = 64
B18_NLEV: int = 8
B18_DT_ATMO: float = 900.0

#: 预注册路径（B19 收口：参数化；默认仍 B18 ⇒ 旧跑批行为不变）
B18_PREREG_PATH: str = "docs/预注册/预注册-B18-生物进真世界联调-20261020.md"
B19_PREREG_PATH: str = "docs/预注册/预注册-B19-解除MAX_POP上限-20261020.md"
B19B_PREREG_PATH: str = "docs/预注册/预注册-B19b-K收敛-20261021.md"
#: 单遍墙钟护栏（秒）；B19 预注册 §五 = 30 min
DEFAULT_MAX_PASS_S: float = 1800.0

#: B19b 2,000 步等价性基线（预注册 §5.2/§7.2）
B19_BASELINE_PATH: str = "results/biosphere/b19_uncapped.json"
B19_BASELINE_SHA256: str = (
    "df0498c13f756be8097fd33fabcce8a8dc057fac82a2694a03d0982649e84926")
B19B_EQUIV_AT: int = 2000
B19B_TMAX: int = 6000


class EquivalenceError(RuntimeError):
    """B19b 2,000 步等价性门失败（实现不合 ⇒ 整批中止）。"""

#: J3 判据阈值（预注册 §四）
U1_REL_TOL: float = 1e-6
U2_BAND: tuple[float, float] = (0.90, 1.10)
U5_BAND: tuple[float, float] = (0.25, 4.0)
#: 天气耦合交换节拍（世界 tick）＝ ``CouplingClock.k``（s1：dt_atmo=900 /
#: tick_seconds=36）。U3 的期望同步次数必须由跑批时限推出，不能写死某一档。
SYNC_EVERY_TICKS: int = 25


def build_weather_world(
    seed: int,
    *,
    rows: int = B18_ROWS,
    cols: int = B18_COLS,
    season_period: int = B18_SEASON_PERIOD,
    tilt_rad: float = B18_TILT_RAD,
    nlat: int = B18_NLAT,
    nlon: int = B18_NLON,
    nlev: int = B18_NLEV,
):
    """构造 B18 的 s1 世界 + A3 合成天气（已 ``bind``）。

    世界族参数锁定为 A18/A20 默认组合：NPP 乘性动态、海平面 volume(300 m)、
    季节 tilt=0.409/period=1200。返回 ``(engine, source)``；驱动顺序见模块头。
    """
    from atmo.coupling.config import AtmoConfig
    from atmo.coupling.synthetic import SyntheticCore
    from atmo.coupling.wiring import make_weather_engine
    from simulation.config import (
        EnvFieldConfig, NPPConfig, ResourceConfig, SeaLevelConfig, SimConfig,
        WorldConfig,
    )

    cfg = SimConfig(
        seed=int(seed),
        world=WorldConfig(rows=int(rows), cols=int(cols)),
        env_field=EnvFieldConfig(
            enabled=True,
            c0b_enabled=True,
            npp=NPPConfig(enabled=True, form="multiplicative",
                          dynamic=True, update_every=10),
            sea_level=SeaLevelConfig(mode="volume", total_water_m=300.0),
        ),
        resources=ResourceConfig(distribution="npp"),
    )
    # PopulationConfig 构造期要求 ≥1；世界要裸跑 ⇒ 构造后再置 0。
    cfg.population.initial_count = 0
    cfg.simulation.use_sim_core = False
    cfg.light.tilt_rad = float(tilt_rad)
    cfg.light.season_period = int(season_period)

    acfg = AtmoConfig(
        enabled=True, weather_source="atmo",
        nlat=int(nlat), nlon=int(nlon), nlev=int(nlev),
        dt_atmo=B18_DT_ATMO, seed=int(seed),
    )
    engine, source = make_weather_engine(cfg, acfg, SyntheticCore(acfg))
    return engine, source


def _sha256(arr) -> str:
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


def _land_stats(capacity: np.ndarray, amount: np.ndarray,
                land: np.ndarray) -> Dict:
    cap = np.asarray(capacity, dtype=np.float64).reshape(-1)[land]
    amt = np.asarray(amount, dtype=np.float64).reshape(-1)[land]
    return {
        "capacity_p50": float(np.percentile(cap, 50)),
        "capacity_p99": float(np.percentile(cap, 99)),
        "capacity_sum": float(cap.sum()),
        "amount_sum": float(amt.sum()),
    }


def _sample_dict(sample) -> Dict:
    return {
        "tick": int(sample.tick),
        "n_land": int(sample.n_land),
        "t2m_c_min": float(sample.t2m_c_min),
        "t2m_c_mean": float(sample.t2m_c_mean),
        "t2m_c_max": float(sample.t2m_c_max),
        "precip_mm_yr_min": float(sample.precip_mm_yr_min),
        "precip_mm_yr_mean": float(sample.precip_mm_yr_mean),
        "precip_mm_yr_max": float(sample.precip_mm_yr_max),
    }


def _git_head() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def load_b19_baseline(path: str = B19_BASELINE_PATH,
                      expected_sha256: str = B19_BASELINE_SHA256
                      ) -> tuple[Dict[int, Dict], Dict]:
    """读 B19 读数，返回 ``(seed → pass0, 元数据)``。

    B19b 的 2,000 步等价性门只对 pass0（B19 A/B 已 5/5 逐位一致）。
    ``expected_sha256`` 非空时校验文件指纹，防止拿错基线。
    """
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and sha != expected_sha256:
        raise RuntimeError(
            f"B19 基线 sha256 不符：{sha} != 预注册 {expected_sha256}（path={path}）")
    doc = json.loads(raw.decode("utf-8"))
    by_seed = {int(row["seed"]): row["passes"][0] for row in doc["seeds"]}
    meta = {
        "path": str(path),
        "sha256": sha,
        "expected_sha256": str(expected_sha256),
        "commit": str(doc.get("commit", "")),
        "schema": str(doc.get("schema", "")),
        "note": "B19 读数只作 B19b 的 2,000 步等价性基线；不重判、不改写",
    }
    return by_seed, meta


def _series_sha256(values: Sequence[int]) -> str:
    arr = np.asarray(list(values), dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _check_equivalence(
    series: Sequence[int],
    *,
    total_born: int,
    total_died: int,
    sync_count: int,
    sync_max_abs_diff: float,
    sync_hold_equal: bool,
    climate: Sequence[Dict],
    baseline: Dict,
    at_step: int = B19B_EQUIV_AT,
) -> Dict:
    """B19b 预注册 §5.2 的 2,000 步等价性检查（纯比较，不读 RNG）。

    返回逐项记录；任一项不等 ⇒ ``passed=False``（调用方立即中止）。
    """
    if int(at_step) != B19B_EQUIV_AT:
        raise ValueError(f"等价性检查点固定为 {B19B_EQUIV_AT}（收到 {at_step}）")
    pop = baseline["population"]
    expected_probes = pop["alive_probes"]
    probe_map = [(1, "0"), (501, "500"), (1001, "1000"),
                 (1501, "1500"), (2000, "1999")]
    mismatches: List[str] = []
    probes: Dict[str, Dict] = {}
    for s, key in probe_map:
        actual = int(series[s])
        expected = expected_probes.get(key)
        ok = expected is not None and actual == int(expected)
        probes[f"P[{s}]"] = {"actual": actual, "expected": expected,
                             "ok": bool(ok)}
        if not ok:
            mismatches.append(f"probe P[{s}] != B19 alive_probes[{key}]")

    checks = {
        "final_alive": {
            "actual": int(series[at_step]),
            "expected": int(pop["final_alive"]),
            "ok": int(series[at_step]) == int(pop["final_alive"]),
        },
        "total_born": {
            "actual": int(total_born),
            "expected": int(pop["total_born"]),
            "ok": int(total_born) == int(pop["total_born"]),
        },
        "total_died": {
            "actual": int(total_died),
            "expected": int(pop["total_died"]),
            "ok": int(total_died) == int(pop["total_died"]),
        },
        "sync_count": {
            "actual": int(sync_count),
            "expected": int(at_step // 25),
            "ok": int(sync_count) == int(at_step // 25),
        },
        "sync_max_abs_diff": {
            "actual": float(sync_max_abs_diff),
            "expected": 0.0,
            "ok": float(sync_max_abs_diff) == 0.0,
        },
        "sync_hold_equal": {
            "actual": bool(sync_hold_equal),
            "expected": True,
            "ok": bool(sync_hold_equal),
        },
    }
    if not checks["final_alive"]["ok"]:
        mismatches.append("final_alive != B19")
    if not checks["total_born"]["ok"]:
        mismatches.append("total_born != B19")
    if not checks["total_died"]["ok"]:
        mismatches.append("total_died != B19")
    for key in ("sync_count", "sync_max_abs_diff", "sync_hold_equal"):
        if not checks[key]["ok"]:
            mismatches.append(f"{key} != B19/协议")

    n_climate = int(at_step // 25) + 1
    expected_climate = baseline["weather"]["samples"][:n_climate]
    actual_climate = list(climate)[:n_climate]
    climate_ok = len(actual_climate) == n_climate \
        and len(expected_climate) == n_climate
    if climate_ok:
        for a, e in zip(actual_climate, expected_climate):
            if (int(a["tick"]), float(a["t2m_c_mean"]),
                    float(a["precip_mm_yr_mean"])) != (
                    int(e["tick"]), float(e["t2m_c_mean"]),
                    float(e["precip_mm_yr_mean"])):
                climate_ok = False
                break
    if not climate_ok:
        mismatches.append("climate samples[0..80] != B19")

    return {
        "checked": True,
        "at_step": int(at_step),
        "passed": not mismatches,
        "probes": probes,
        "checks": checks,
        "climate": {
            "n_expected": n_climate,
            "n_actual": len(actual_climate),
            "ok": bool(climate_ok),
        },
        "mismatches": mismatches,
    }


def run_pass(
    seed: int,
    *,
    ticks: int = B18_TICKS,
    init_pop: int = B18_INIT_POP,
    rows: int = B18_ROWS,
    cols: int = B18_COLS,
    season_period: int = B18_SEASON_PERIOD,
    tilt_rad: float = B18_TILT_RAD,
    nlat: int = B18_NLAT,
    nlon: int = B18_NLON,
    nlev: int = B18_NLEV,
    max_pop: int = MAX_POP,
    light: bool = False,
    series_every: int = 1,
    baseline: Optional[Dict] = None,
    equiv_at: int = B19B_EQUIV_AT,
    progress_every: int = 1000,
) -> Dict:
    """一次完整耦合运行（世界 + 天气 + 生物），返回单遍读数。

    ``max_pop`` 默认 = 冻结的 ``MAX_POP=512``（B18 逐位口径）；B19 传更大值
    解除上限混杂（s1 有 28,800 格 ⇒ 100000 不会绑定）。

    B19b ``light=True``：只关 ``SimLog`` 的帧/事件累积（纯观测，仿真路径
    不读取），保留出生/死亡/能量台账，逐步记录 ``P[s]``；跳过
    ``metrics_json``/``run_digest``。``baseline`` 给出 B19 pass0 时在
    ``equiv_at`` 步执行等价性门，失败即抛 :class:`EquivalenceError`。
    默认 ``light=False`` 的输出与 B18/B19 完全一致。
    """
    if int(series_every) < 1:
        raise ValueError("series_every 必须 ≥ 1")
    if baseline is not None:
        if not light:
            raise ValueError("baseline 等价性门只在 light=True 下使用")
        if int(series_every) != 1:
            raise ValueError("等价性门要求 series_every=1（逐步序列）")
        if int(ticks) < int(equiv_at):
            raise ValueError(
                f"ticks={ticks} < equiv_at={equiv_at}，无法执行等价性门")
    t_build0 = time.time()
    engine, source = build_weather_world(
        seed, rows=rows, cols=cols, season_period=season_period,
        tilt_rad=tilt_rad, nlat=nlat, nlon=nlon, nlev=nlev)
    build_s = time.time() - t_build0

    view = adapt_real_world(engine.world, engine.env_field,
                            engine.resources, engine.light)
    resource = adapt_real_resource(engine.resources)
    land = np.asarray(engine.env_field.land_mask, dtype=bool).reshape(-1)
    cap0 = np.asarray(engine.resources._capacity, dtype=np.float64).reshape(-1).copy()
    amount0_world = np.asarray(engine.resources._grid, dtype=np.float64).reshape(-1).copy()
    start = _land_stats(cap0, amount0_world, land)
    climate = [_sample_dict(sample_climate(source, land, 0))]

    cfg = BioMVPConfig(rows=int(rows), cols=int(cols), seed=int(seed),
                       ticks=int(ticks), init_pop=int(init_pop),
                       resolve=ResolveConfig(max_pop=int(max_pop)))
    sim = BiosphereSimulator(cfg, world=view, resource=resource)
    if light:
        # 纯观测累积关闭；出生/死亡/能量台账保留（仿真路径不读帧/事件）。
        sim.log = SimLog(capture_frames=False, capture_events=False)
        # 墓地只供 diagnose.py 的谱系诊断；6,000 tick 会累积约 20 万对象。
        sim.population.retain_dead = False
    init_alive = len(sim.population.individuals)
    init_energy = ([float(ind.energy) for ind in sim.population.individuals])

    sync_count = 0
    sync_max_abs = 0.0
    last_sync_cap: Optional[np.ndarray] = None
    series: Optional[List[int]] = [int(init_alive)] if light else None
    series_steps: Optional[List[int]] = [0] if light else None
    equivalence: Optional[Dict] = None
    t_run0 = time.time()
    for tick in range(int(ticks)):
        step = tick + 1
        exchanged = source.on_tick(engine.tick + 1)
        engine.step()
        if exchanged:
            sync_real_capacity(resource, engine.resources)
            diff = float(np.max(np.abs(
                np.asarray(resource.capacity, dtype=np.float64)
                - np.asarray(engine.resources._capacity, dtype=np.float64))))
            sync_max_abs = max(sync_max_abs, diff)
            sync_count += 1
            last_sync_cap = np.asarray(
                engine.resources._capacity, dtype=np.float64).reshape(-1).copy()
            climate.append(_sample_dict(
                sample_climate(source, land, engine.tick)))
        sim.step(tick)
        if light:
            if step % int(series_every) == 0 or step == int(ticks):
                series.append(int(len(sim.population.individuals)))
                series_steps.append(int(step))
            if baseline is not None and step == int(equiv_at):
                hold_equal = bool(
                    last_sync_cap is not None
                    and np.array_equal(
                        np.asarray(resource.capacity, dtype=np.float64),
                        last_sync_cap))
                equivalence = _check_equivalence(
                    series,
                    total_born=len(sim.log.births),
                    total_died=len(sim.log.deaths),
                    sync_count=sync_count,
                    sync_max_abs_diff=sync_max_abs,
                    sync_hold_equal=hold_equal,
                    climate=climate,
                    baseline=baseline,
                    at_step=int(equiv_at))
                equivalence["wall_s"] = time.time() - t_run0
                if not equivalence["passed"]:
                    raise EquivalenceError(
                        f"seed={seed} pass 在 s={equiv_at} 等价性失败："
                        + "; ".join(equivalence["mismatches"]))
            if progress_every and step % int(progress_every) == 0:
                print(f"[B19b] seed={seed} tick={step} "
                      f"alive={series[-1]} wall={time.time() - t_run0:.1f}s",
                      flush=True)
    run_s = time.time() - t_run0

    cap_end = np.asarray(engine.resources._capacity, dtype=np.float64).reshape(-1).copy()
    amount_end_world = np.asarray(engine.resources._grid, dtype=np.float64).reshape(-1).copy()
    end = _land_stats(cap_end, amount_end_world, land)
    # 生物侧容量自最后一次同步后不应再变（simulator 只读容量）；
    # 世界侧动态 NPP 可能在两次交换之间继续更新（零阶保持的另一面）。
    sync_hold_equal = bool(
        last_sync_cap is not None
        and np.array_equal(np.asarray(resource.capacity, dtype=np.float64),
                           last_sync_cap))
    world_drift_after_sync = bool(
        last_sync_cap is not None and not np.array_equal(last_sync_cap, cap_end))

    probes = sorted({0, ticks // 4, ticks // 2, 3 * ticks // 4, ticks - 1})
    if light:
        alive_series = [int(v) for v in series]
        final_alive = int(alive_series[-1])
        peak_alive = int(max(alive_series))
        min_alive = int(min(alive_series))
        zero_step = next((s for s in range(1, len(alive_series))
                          if alive_series[s] == 0), None)
        # 与完整路径同口径：first_zero_tick = 帧 tick（= 步数 s - 1）。
        first_zero = (int(zero_step) - 1) if zero_step is not None else None
        alive_probes = {
            str(t): (int(alive_series[t + 1])
                     if int(series_every) == 1
                     and t + 1 < len(alive_series) else None)
            for t in probes
        }
        total_born = int(len(sim.log.births))
        total_died = int(len(sim.log.deaths))
        death_causes = dict(Counter(c for _u, c, _t in sim.log.deaths))
        energy_final = float(math.fsum(
            float(ind.energy) for ind in sim.population.individuals))
        behaviors = None
        lineage = None
        action_counts = None
        digest = None
        series_sha = _series_sha256(series)
        verdict = None
        if int(series_every) == 1 and int(ticks) >= 2 * 500:
            verdict = k_convergence_verdict(series, t_max=int(ticks))
    else:
        metrics = metrics_json(sim.log, view, seed, ticks)
        pop = metrics["population"]
        alive_series = [int(v) for v in pop["alive_series"]]
        alive_probes = {
            str(t): (alive_series[t] if t < len(alive_series) else None)
            for t in probes
        }
        first_zero = next((i for i, v in enumerate(alive_series) if v == 0),
                          None)
        lineage = metrics["lineage"]
        final_alive = int(pop["final_alive"])
        peak_alive = int(pop["peak_alive"])
        min_alive = int(min(alive_series)) if alive_series else 0
        total_born = int(pop["total_born"])
        total_died = int(pop["total_died"])
        death_causes = dict(pop["death_causes"])
        energy_final = float(metrics["energy"]["final_total"])
        behaviors = dict(metrics["behaviors"]["counts"])
        lineage = {
            "n_individuals": int(lineage["n_individuals"]),
            "n_roots": int(lineage["n_roots"]),
            "max_generation": int(lineage["max_generation"]),
        }
        action_counts = dict(metrics["action_counts"])
        digest = run_digest(sim.log)
        series_sha = None
        verdict = None

    probe = source.probe()
    bridge = probe.get("bridge", {})
    params = {
        "rows": int(rows), "cols": int(cols), "ticks": int(ticks),
        "init_pop": int(init_pop), "season_period": int(season_period),
        "tilt_rad": float(tilt_rad),
        "nlat": int(nlat), "nlon": int(nlon), "nlev": int(nlev),
        "max_pop": int(max_pop),
    }
    if light:
        params.update({
            "light": True, "series_every": int(series_every),
            "equiv_at": int(equiv_at) if baseline is not None else None,
        })
    out = {
        "seed": int(seed),
        "params": params,
        "world": {
            "n_cells": int(view.n_cells),
            "land_cells": int(land.sum()),
            "land_frac": float(land.mean()),
            "surface_land_cells": int((view.surface_type == SURFACE_LAND).sum()),
            "patch_cells": int((view.patch_id >= 0).sum()),
            "season_period": int(view.season_period),
            "tilt_rad": float(view.tilt_rad),
        },
        "weather": {
            "source": "synthetic",
            "placeholder": bool(probe.get("synthetic", False)),
            "exchanges": int(bridge.get("exchanges", 0)),
            "k_ticks": int(bridge.get("k_ticks", 0)),
            "samples": climate,
        },
        "resource": {
            "regrow_rate": float(resource.regrow_rate),
            "start": start,
            "end": end,
            "sync_count": int(sync_count),
            "sync_max_abs_diff": float(sync_max_abs),
            "sync_hold_equal": sync_hold_equal,
            "world_drift_after_last_sync": world_drift_after_sync,
            "amount_sha256": _sha256(resource.amount),
            "capacity_sha256": _sha256(resource.capacity),
            "world_grid_sha256": _sha256(amount_end_world),
            "world_capacity_sha256": _sha256(cap_end),
        },
        "population": {
            "initial_alive": init_alive,
            "initial_energy_mean": (float(np.mean(init_energy))
                                    if init_energy else None),
            "final_alive": final_alive,
            "peak_alive": peak_alive,
            "min_alive": min_alive,
            "first_zero_tick": first_zero,
            "alive_probes": alive_probes,
            "total_born": total_born,
            "total_died": total_died,
            "death_causes": death_causes,
            "final_energy_total": energy_final,
            "final_energy_mean": (energy_final / final_alive
                                  if final_alive else None),
        },
        "behaviors": behaviors,
        "lineage": lineage,
        "action_counts": action_counts,
        "run_digest": digest,
        "atmo": {
            "digest": str(bridge.get("atmo_digest", "")),
            "synthetic": bool(probe.get("synthetic", False)),
        },
        "wall_s": {"build": build_s, "run": run_s,
                   "total": time.time() - t_build0},
    }
    if light:
        out["series"] = [int(v) for v in series]
        out["series_sha256"] = series_sha
        if int(series_every) != 1:
            out["series_steps"] = [int(s) for s in series_steps]
        out["equivalence"] = equivalence
        out["convergence"] = verdict
    return out


def _det_key(p: Dict) -> Dict:
    return {
        "run_digest": p["run_digest"],
        "amount_sha256": p["resource"]["amount_sha256"],
        "capacity_sha256": p["resource"]["capacity_sha256"],
        "world_grid_sha256": p["resource"]["world_grid_sha256"],
        "world_capacity_sha256": p["resource"]["world_capacity_sha256"],
        "final_alive": p["population"]["final_alive"],
        "total_born": p["population"]["total_born"],
        "total_died": p["population"]["total_died"],
        "climate": [(s["tick"], s["t2m_c_mean"], s["precip_mm_yr_mean"])
                    for s in p["weather"]["samples"]],
        "atmo_digest": p["atmo"]["digest"],
    }


def _det_key_light(p: Dict) -> Dict:
    """light 路径的 A/B 确定性键（预注册 §5.3：完整 P 序列 + 末窗计数）。"""
    return {
        "series_sha256": p["series_sha256"],
        "amount_sha256": p["resource"]["amount_sha256"],
        "capacity_sha256": p["resource"]["capacity_sha256"],
        "world_grid_sha256": p["resource"]["world_grid_sha256"],
        "world_capacity_sha256": p["resource"]["world_capacity_sha256"],
        "final_alive": p["population"]["final_alive"],
        "total_born": p["population"]["total_born"],
        "total_died": p["population"]["total_died"],
        "final_window_mean": (p["convergence"]["final_window_mean"]
                              if p["convergence"] else None),
        "climate": [(s["tick"], s["t2m_c_mean"], s["precip_mm_yr_mean"])
                    for s in p["weather"]["samples"]],
        "atmo_digest": p["atmo"]["digest"],
    }


def _judgements(per_seed: List[Dict], repeat: int, max_pop: int,
                ticks: int) -> Dict:
    anchor = A20_S1_MULT_CAP_SUM
    expected_sync = int(ticks) // SYNC_EVERY_TICKS
    n_alive = sum(1 for r in per_seed
                  if r["passes"][0]["population"]["final_alive"] > 0)
    j1 = n_alive >= 3

    det_fail = [r["seed"] for r in per_seed if not r["determinism"]["equal"]]
    j2 = repeat >= 2 and not det_fail

    peak_max = max((p["population"]["peak_alive"]
                    for r in per_seed for p in r["passes"]), default=0)
    cap_binding_seeds = [
        r["seed"] for r in per_seed
        if max(p["population"]["peak_alive"] for p in r["passes"]) >= max_pop]
    j4 = peak_max < max_pop

    u1_rows = []
    u2_rows = []
    u5_rows = []
    u3_rows = []
    for r in per_seed:
        p = r["passes"][0]
        rel = abs(p["resource"]["start"]["capacity_sum"] - anchor) / anchor
        u1_rows.append({"seed": r["seed"], "rel_diff": rel})
        cap_sum = p["resource"]["start"]["capacity_sum"]
        u2_rows.append({"seed": r["seed"], "capacity_sum": cap_sum,
                        "in_band": U2_BAND[0] * anchor <= cap_sum <= U2_BAND[1] * anchor})
        p50 = p["resource"]["end"]["capacity_p50"]
        u5_rows.append({"seed": r["seed"], "capacity_p50_end": p50,
                        "in_band": U5_BAND[0] * A20_S1_MULT_CAP_P50 <= p50
                                <= U5_BAND[1] * A20_S1_MULT_CAP_P50})
        u3_rows.append({
            "seed": r["seed"],
            "sync_count": p["resource"]["sync_count"],
            "sync_max_abs_diff": p["resource"]["sync_max_abs_diff"],
            "sync_hold_equal": p["resource"]["sync_hold_equal"],
            "ok": (p["resource"]["sync_count"] == expected_sync
                   and p["resource"]["sync_max_abs_diff"] == 0.0
                   and p["resource"]["sync_hold_equal"]),
        })

    seed42 = next((row for row in u1_rows if row["seed"] == 42), None)
    u1 = seed42 is not None and seed42["rel_diff"] <= U1_REL_TOL
    u2 = all(row["in_band"] for row in u2_rows)
    u3 = all(row["ok"] for row in u3_rows)
    u5 = all(row["in_band"] for row in u5_rows)
    return {
        "J1_ecological_gate": {
            "pass": bool(j1), "alive_end_seeds": int(n_alive),
            "n_seeds": len(per_seed), "rule": ">=3/5 alive at t=ticks-1",
        },
        "J2_determinism": {
            "pass": bool(j2), "repeat": int(repeat),
            "failed_seeds": det_fail, "rule": "all seeds A/B exact",
        },
        "J4_cap_not_binding": {
            "pass": bool(j4), "max_pop": int(max_pop),
            "peak_max": int(peak_max), "binding_seeds": cap_binding_seeds,
            "rule": "every pass peak_alive < max_pop",
        },
        "J3_units": {
            "pass": bool(u1 and u2 and u3 and u5),
            "U1_seed42_anchor": {
                "pass": bool(u1), "anchor": anchor,
                "rel_tol": U1_REL_TOL, "rows": u1_rows,
            },
            "U2_all_seeds_sum_band": {
                "pass": bool(u2), "band": list(U2_BAND), "rows": u2_rows,
            },
            "U3_capacity_sync": {
                "pass": bool(u3), "rows": u3_rows,
                "expected_sync_count": int(expected_sync),
            },
            "U4_eat_unit_ledger": {
                "pass": None,
                "covered_by": "tests/test_bio_b18.py::test_eat_one_to_one",
            },
            "U5_p50_order": {
                "pass": bool(u5), "band": list(U5_BAND),
                "anchor_p50": A20_S1_MULT_CAP_P50, "rows": u5_rows,
            },
        },
    }


def run_batch(
    seeds: Sequence[int] = B18_SEEDS,
    *,
    ticks: int = B18_TICKS,
    init_pop: int = B18_INIT_POP,
    repeat: int = 2,
    rows: int = B18_ROWS,
    cols: int = B18_COLS,
    season_period: int = B18_SEASON_PERIOD,
    tilt_rad: float = B18_TILT_RAD,
    nlat: int = B18_NLAT,
    nlon: int = B18_NLON,
    nlev: int = B18_NLEV,
    max_pop: int = MAX_POP,
    prereg: str = B18_PREREG_PATH,
    max_pass_s: float = DEFAULT_MAX_PASS_S,
    out: Optional[Path] = None,
    light: bool = False,
    series_every: int = 1,
    baseline_path: Optional[str] = B19_BASELINE_PATH,
    baseline_sha256: str = B19_BASELINE_SHA256,
    equiv_at: int = B19B_EQUIV_AT,
    max_pass_abort_s: float = 2700.0,
    max_batch_s: float = 18000.0,
) -> Dict:
    """B18/B19 判读批：每 seed 跑 ``repeat`` 遍，做判定并落 JSON。

    每遍结束即把墙钟追加到 ``<out>.passes.jsonl``（不依赖末次汇总）；
    ``wall_s.total > max_pass_s`` 时立即写告警并打印到 stderr。

    ``light=True`` = B19b 6,000 tick 长跑：逐步 ``P[s]``、每遍 2,000 步
    等价性门（对 B19 pass0）、A/B 全序列确定性、K 收敛纯函数判读；
    并启用单遍 >45 min / 整批投影 >5 h 的成本中止线。
    """
    if int(repeat) < 2:
        raise ValueError("B18 预注册要求 --repeat ≥2（确定性判据）")
    if light and int(series_every) != 1:
        raise ValueError("B19b 预注册 §1.3 要求每步采样：light 模式 series_every 必须=1")
    commit = _git_head()
    out_path = Path(out) if out is not None else None
    wall_log: Optional[Path] = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        wall_log = out_path.with_suffix(".passes.jsonl")
        wall_log.write_text("", encoding="utf-8")
    baseline_meta: Optional[Dict] = None
    baseline_by_seed: Optional[Dict[int, Dict]] = None
    if light:
        if baseline_path is None:
            # 逃生门：只给非正式短跑（oracle/接线自检）用。正式 T_max 长跑
            # 必须过 §5.2 等价性门，缺基线即拒绝。
            if int(ticks) == B19B_TMAX:
                raise ValueError(
                    f"正式 T_max={B19B_TMAX} 的 light 长跑必须给 B19 基线"
                    f"（预注册 §5.2 等价性门；baseline_path=None 只允许非正式短跑）")
        else:
            baseline_by_seed, baseline_meta = load_b19_baseline(
                baseline_path, baseline_sha256)
            missing = [int(s) for s in seeds if int(s) not in baseline_by_seed]
            if missing:
                raise ValueError(f"B19 基线缺少 seed: {missing}")
    wall_alerts: List[Dict] = []
    per_seed: List[Dict] = []
    tick_batch0 = time.time()
    n_total_passes = len(seeds) * int(repeat)
    n_done = 0
    max_total_s = 0.0
    for seed in seeds:
        seed = int(seed)
        passes = []
        for pass_idx in range(int(repeat)):
            try:
                p = run_pass(
                    seed, ticks=ticks, init_pop=init_pop, rows=rows, cols=cols,
                    season_period=season_period, tilt_rad=tilt_rad,
                    nlat=nlat, nlon=nlon, nlev=nlev,
                    max_pop=max_pop,
                    light=light, series_every=series_every,
                    baseline=(baseline_by_seed.get(seed)
                              if baseline_by_seed is not None else None),
                    equiv_at=equiv_at)
            except EquivalenceError as exc:
                print(f"[B19b][EQUIV-FAIL] {exc}", file=sys.stderr, flush=True)
                if out_path is not None:
                    fail = {
                        "schema": "biosphere-b19b-v1",
                        "status": "equivalence_failed",
                        "detail": str(exc),
                        "commit": commit, "prereg": str(prereg),
                        "baseline": baseline_meta,
                        "wall_log": str(wall_log) if wall_log else None,
                        "wall_alerts": wall_alerts,
                        "completed_passes": n_done,
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    out_path.write_text(
                        json.dumps(fail, indent=1, ensure_ascii=False),
                        encoding="utf-8")
                raise
            passes.append(p)
            n_done += 1
            max_total_s = max(max_total_s, float(p["wall_s"]["total"]))
            rec = {
                "seed": seed, "pass": pass_idx,
                "run_s": float(p["wall_s"]["run"]),
                "total_s": float(p["wall_s"]["total"]),
                "max_pass_s": float(max_pass_s),
                "over_budget": bool(p["wall_s"]["total"] > max_pass_s),
                "ticks": int(ticks), "max_pop": int(max_pop),
                "prereg": str(prereg), "commit": commit,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            if light:
                rec["light"] = True
                rec["series_sha256"] = p["series_sha256"]
                rec["equivalence_passed"] = bool(
                    p["equivalence"] and p["equivalence"]["passed"])
            if wall_log is not None:
                with wall_log.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
            if rec["over_budget"]:
                wall_alerts.append({"seed": seed, "pass": pass_idx,
                                    "total_s": rec["total_s"]})
                print(f"[B18][WALL-ALERT] seed={seed} pass={pass_idx} "
                      f"total={rec['total_s']:.1f}s "
                      f"> max_pass_s={max_pass_s:.0f}s",
                      file=sys.stderr, flush=True)
            if light:
                if float(p["wall_s"]["total"]) > float(max_pass_abort_s):
                    raise RuntimeError(
                        f"[B19b][COST-ABORT] seed={seed} pass={pass_idx} "
                        f"total={p['wall_s']['total']:.1f}s > "
                        f"{max_pass_abort_s:.0f}s（预注册 §6.2）")
                elapsed = time.time() - tick_batch0
                remaining = n_total_passes - n_done
                projected = elapsed + remaining * max_total_s
                print(f"[B19b][COST] done={n_done}/{n_total_passes} "
                      f"elapsed={elapsed:.1f}s max_pass={max_total_s:.1f}s "
                      f"projected={projected:.1f}s budget={max_batch_s:.0f}s",
                      flush=True)
                if projected > float(max_batch_s):
                    raise RuntimeError(
                        f"[B19b][COST-ABORT] 投影 {projected:.1f}s > "
                        f"{max_batch_s:.0f}s（预注册 §6.2）")
        keys = [(_det_key_light(p) if light else _det_key(p)) for p in passes]
        equal = all(k == keys[0] for k in keys[1:])
        per_seed.append({"seed": seed, "passes": passes,
                         "determinism": {"equal": bool(equal)}})
        p = passes[0]
        if light:
            conv = p.get("convergence") or {}
            print(f"[B19b] seed={seed} alive {p['population']['initial_alive']}→"
                  f"{p['population']['final_alive']} "
                  f"born={p['population']['total_born']} "
                  f"died={p['population']['total_died']} "
                  f"status={conv.get('status')} T_K={conv.get('T_K')} "
                  f"K={conv.get('K')} sync={p['resource']['sync_count']} "
                  f"det={'OK' if equal else 'FAIL'} "
                  f"wall={p['wall_s']['total']:.1f}s", flush=True)
        else:
            print(f"[B18] seed={seed} alive {p['population']['initial_alive']}→"
                  f"{p['population']['final_alive']} born={p['population']['total_born']} "
                  f"died={p['population']['total_died']} "
                  f"cap_p50_end={p['resource']['end']['capacity_p50']:.3f} "
                  f"sync={p['resource']['sync_count']} det={'OK' if equal else 'FAIL'} "
                  f"wall={p['wall_s']['total']:.1f}s", flush=True)

    params = {
        "seeds": [int(s) for s in seeds], "ticks": int(ticks),
        "init_pop": int(init_pop), "repeat": int(repeat),
        "rows": int(rows), "cols": int(cols),
        "season_period": int(season_period), "tilt_rad": float(tilt_rad),
        "nlat": int(nlat), "nlon": int(nlon), "nlev": int(nlev),
        "max_pop": int(max_pop),
    }
    if light:
        params.update({"light": True, "series_every": int(series_every),
                       "equiv_at": int(equiv_at)})
    summary = {
        "schema": "biosphere-b19b-v1" if light else "biosphere-b18-v1",
        "commit": commit,
        "prereg": str(prereg),
        "weather_note": "天气源=synthetic（A3 SyntheticCore，placeholder）；"
                        "不得当作天气/生态结论",
        "params": params,
        "anchor": {
            "source": "A20 s1 multiplicative seed42",
            "capacity_land_sum": A20_S1_MULT_CAP_SUM,
            "capacity_land_p50": A20_S1_MULT_CAP_P50,
        },
        "seeds": per_seed,
        "judgements": _judgements(per_seed, int(repeat), int(max_pop),
                                  int(ticks)),
    }
    if light:
        summary["baseline"] = baseline_meta
        eq_rows = []
        for row in per_seed:
            for i, p in enumerate(row["passes"]):
                eq = p.get("equivalence") or {}
                eq_rows.append({
                    "seed": row["seed"], "pass": i,
                    "checked": bool(eq.get("checked")),
                    "passed": bool(eq.get("passed")),
                    "mismatches": eq.get("mismatches"),
                    "wall_s": eq.get("wall_s"),
                })
        if baseline_by_seed is None:
            summary["equivalence"] = {
                "at_step": None, "all_passed": None, "passes": [],
                "note": "未执行等价性门（baseline_path=None；非正式短跑）",
            }
        else:
            summary["equivalence"] = {
                "at_step": int(equiv_at),
                "all_passed": (all(r["passed"] for r in eq_rows)
                               if eq_rows else False),
                "passes": eq_rows,
            }
        series_by_seed = {int(r["seed"]): r["passes"][0]["series"]
                          for r in per_seed}
        if int(ticks) < K_T0 + K_W:
            conv = {
                "params": {"W": K_W, "eps": K_EPS, "N": K_N, "t0": K_T0,
                           "t_max": int(ticks)},
                "per_seed": {}, "n_total": len(series_by_seed),
                "n_converged": 0, "n_not_converged": 0, "n_extinct": 0,
                "converged_seeds": [], "not_converged_seeds": [],
                "extinct_seeds": [], "edge_confirmation_seeds": [],
                "conclusion": "series_too_short",
                "K_cross_seed": None,
                "series_too_short": True,
                "note": (f"ticks={ticks} < t0+W={K_T0 + K_W}："
                         f"序列不足以算窗口均值，不做收敛判读"),
            }
        else:
            conv = batch_k_convergence_verdict(series_by_seed,
                                              t_max=int(ticks))
            conv["series_too_short"] = False
        det_fail = [int(r["seed"]) for r in per_seed
                    if not r["determinism"]["equal"]]
        conv["determinism_failed_seeds"] = det_fail
        if int(ticks) != B19B_TMAX:
            conv["formal_judgement"] = False
            _why = (f"仅 B19b 固定 T_max={B19B_TMAX} 判读；本次 ticks={ticks}")
            conv["note"] = (f"{conv['note']}；{_why}"
                            if conv.get("note") else _why)
            conv["conclusion"] = "not_formal_tmax"
            conv["K_cross_seed"] = None
        elif det_fail:
            conv["formal_judgement"] = False
            conv["note"] = "A/B 全序列不相等 ⇒ 整批无效，不报收敛结论（§5.3）"
            conv["conclusion"] = "invalid_determinism"
            conv["K_cross_seed"] = None
        else:
            conv["formal_judgement"] = True
        summary["convergence"] = conv
    summary["alive_end"] = {str(r["seed"]): r["passes"][0]["population"]["final_alive"]
                            for r in per_seed}
    summary["n_alive_end"] = int(sum(1 for r in per_seed
                                     if r["passes"][0]["population"]["final_alive"] > 0))
    summary["wall_log"] = str(wall_log) if wall_log is not None else None
    summary["wall_alerts"] = wall_alerts
    if out_path is not None:
        out_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                            encoding="utf-8")
    if light:
        print("=== B19b convergence ===")
        print(json.dumps({
            "conclusion": summary["convergence"]["conclusion"],
            "n_converged": summary["convergence"]["n_converged"],
            "n_total": summary["convergence"]["n_total"],
            "n_extinct": summary["convergence"]["n_extinct"],
            "K_cross_seed": summary["convergence"]["K_cross_seed"],
            "equivalence_all_passed": summary["equivalence"]["all_passed"],
        }, indent=1, ensure_ascii=False))
    else:
        print("=== B18 judgements ===")
        print(json.dumps(summary["judgements"], indent=1, ensure_ascii=False))
    print(f"note: 天气源=synthetic（A3 当前核）；读数不是天气/生态结论。"
          f" commit={summary['commit']}")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m biosphere.world_coupled",
        description="B18 生物进真世界联调（A3 synthetic 天气 + A18/A20 世界族）")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--ticks", type=int, default=B18_TICKS)
    parser.add_argument("--init-pop", type=int, default=B18_INIT_POP)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--rows", type=int, default=B18_ROWS)
    parser.add_argument("--cols", type=int, default=B18_COLS)
    parser.add_argument("--season-period", type=int, default=B18_SEASON_PERIOD)
    parser.add_argument("--tilt", type=float, default=B18_TILT_RAD)
    parser.add_argument("--nlat", type=int, default=B18_NLAT)
    parser.add_argument("--nlon", type=int, default=B18_NLON)
    parser.add_argument("--nlev", type=int, default=B18_NLEV)
    parser.add_argument("--max-pop", type=int, default=MAX_POP,
                        help="繁殖种群上限（B19 解除上限用 100000）")
    parser.add_argument("--prereg", type=str, default=B18_PREREG_PATH,
                        help="写进读数的预注册路径（B19 用 B19 预注册）")
    parser.add_argument("--max-pass-s", type=float, default=DEFAULT_MAX_PASS_S,
                        help="单遍墙钟告警线（秒；默认 1800 = B19 预注册 30 min）")
    parser.add_argument("--out", type=str,
                        default="results/biosphere/b18_weather.json")
    parser.add_argument("--light", action="store_true",
                        help="B19b 轻量长跑路径（只关观测累积，不改仿真）")
    parser.add_argument("--series-every", type=int, default=1,
                        help="P[s] 采样间隔（B19b 固定 1 = 每步）")
    parser.add_argument("--baseline", type=str, default=B19_BASELINE_PATH,
                        help="2,000 步等价性基线（B19 读数）；空串 = 不过门"
                             "（只允许非正式短跑，正式 T_max 会拒绝）")
    parser.add_argument("--baseline-sha256", type=str,
                        default=B19_BASELINE_SHA256,
                        help="基线 sha256（默认 = 预注册 §1.1 的 B19 指纹）")
    parser.add_argument("--equiv-at", type=int, default=B19B_EQUIV_AT,
                        help="等价性检查点（固定 2000）")
    parser.add_argument("--max-pass-abort-s", type=float, default=2700.0,
                        help="light 单遍中止线（秒；默认 2700 = 45 min）")
    parser.add_argument("--max-batch-s", type=float, default=18000.0,
                        help="light 整批投影中止线（秒；默认 18000 = 5 h）")
    args = parser.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.repeat < 2:
        parser.error("B18 预注册要求 --repeat ≥2（确定性判据）")
    run_batch(
        seeds, ticks=args.ticks, init_pop=args.init_pop, repeat=args.repeat,
        rows=args.rows, cols=args.cols, season_period=args.season_period,
        tilt_rad=args.tilt, nlat=args.nlat, nlon=args.nlon, nlev=args.nlev,
        max_pop=args.max_pop, prereg=args.prereg,
        max_pass_s=args.max_pass_s, out=Path(args.out),
        light=args.light, series_every=args.series_every,
        baseline_path=(args.baseline or None),
        baseline_sha256=args.baseline_sha256,
        equiv_at=args.equiv_at,
        max_pass_abort_s=args.max_pass_abort_s,
        max_batch_s=args.max_batch_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
