"""biosphere.diagnose：B17.1 生态门诊断（spread 注入 → 崩群路径；非判据）。

只回答三个问题（不做参数搜索、不跑 B16 判据）：

1. **初始性状**：``init_trait_spread`` 各档把 defense/attack_drive 推到哪
   （均值/方差/饱和比例），以及初始能量结构是否被改动（初代恒 60）；
2. **收支**：按 attack_drive / defense 分箱，统计每次攻击的成本、loot、
   反击、伤害与净能量（事件级账本）；
3. **崩群路径**：存活曲线、死亡原因、首批 100 tick 的能量流
   （攻击成本 vs 进食收入），判断是"参数（饱和）"还是"机制（收益结构）"。

用法（worktree 根）::

    python -m biosphere.diagnose --seeds 41-45 --ticks 5000 \\
        --spreads 0,0.5,1.0,1.5,2.0 --workers 14

输出 ``results/biosphere/b17_diag_summary_t<ticks>.json``（gitignore）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from biosphere.action import ACTION_BASE_COST, ActionKind
from biosphere.criteria import seed_gate
from biosphere.experiments import build_b16_config
from biosphere.parallel import (
    JobSpec, manifest, run_parallel, write_manifest,
)
from biosphere.simulator import BiosphereSimulator

#: 攻击基准成本（从动作模块读，不在诊断里重写常数）
ATTACK_COST: float = ACTION_BASE_COST[int(ActionKind.ATTACK)]
#: 分箱边界（左闭右开；最后一箱含 1.0）
BIN_EDGES: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.000001)
#: 早窗（能量流诊断）
EARLY_TICKS: int = 100
#: 存活曲线采样点
SERIES_SAMPLE: Tuple[int, ...] = (0, 50, 100, 200, 500, 1000, 2000, 3000, 4000)


def _parse_seeds(text: str) -> List[int]:
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _parse_floats(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _trait_stats(values: Sequence[float]) -> Dict:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    hist, _ = np.histogram(arr, bins=BIN_EDGES)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "sd": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "frac_ge_0p8": float(np.mean(arr >= 0.8)),
        "hist": [int(x) for x in hist],
    }


def _controller_traits(sim: BiosphereSimulator) -> Dict[int, Tuple[float, float]]:
    """uid → (attack_drive, defense)。性状由基因组决定、生命周期内不变。"""
    out: Dict[int, Tuple[float, float]] = {}
    for ind in list(sim.population.individuals) \
            + list(sim.population.graveyard):
        c = sim.cache.get(ind.genome)
        out[int(ind.uid)] = (float(c.attack_drive), float(c.defense))
    return out


def _bin_index(x: float) -> int:
    return min(len(BIN_EDGES) - 2, max(0, int(float(x) / 0.2)))


def _empty_bin() -> Dict:
    return {
        "n_attacks": 0, "n_hits": 0, "n_kills": 0,
        "cost": 0.0, "loot": 0.0, "counter": 0.0,
        "damage": 0.0, "net": 0.0, "exposure": 0,
    }


def _finalize_bin(d: Dict) -> Dict:
    n = d["n_attacks"]
    kills = d["n_kills"]
    exposure = d["exposure"]
    return {
        "n_attacks": int(n),
        "n_hits": int(d["n_hits"]),
        "n_kills": int(kills),
        "cost": float(d["cost"]),
        "loot": float(d["loot"]),
        "counter": float(d["counter"]),
        "damage": float(d["damage"]),
        "net": float(d["net"]),
        "exposure_ind_ticks": int(exposure),
        "attacks_per_ind_tick": (n / exposure) if exposure else 0.0,
        "net_per_attack": (d["net"] / n) if n else 0.0,
        "loot_per_kill": (d["loot"] / kills) if kills else 0.0,
        "kill_rate_per_attack": (kills / n) if n else 0.0,
        "counter_per_attack": (d["counter"] / n) if n else 0.0,
        "damage_per_hit": (d["damage"] / d["n_hits"])
                          if d["n_hits"] else 0.0,
    }


def _attack_ledger(log, traits: Dict[int, Tuple[float, float]],
                   max_tick: Optional[int] = None) -> Dict:
    """事件级攻击账本：总体 + attack_drive 分箱 + defense 分箱。"""
    exposure: Dict[int, int] = {}
    for f in log.frames:
        if max_tick is not None and int(f.tick) > max_tick:
            continue
        for u in f.uids:
            exposure[int(u)] = exposure.get(int(u), 0) + 1

    overall = _empty_bin()
    by_attack = {i: _empty_bin() for i in range(len(BIN_EDGES) - 1)}
    by_defense = {i: _empty_bin() for i in range(len(BIN_EDGES) - 1)}

    for e in log.events:
        if e.kind != "attack" or e.status == "actor_dead":
            continue
        if max_tick is not None and int(e.tick) > max_tick:
            continue
        actor = int(e.actor_id)
        atk, _def_a = traits.get(actor, (0.0, 0.0))
        abin = by_attack[_bin_index(atk)]
        hit = e.status in ("ok", "attack_survived")
        target = int(e.target_id)
        tgt_def = traits.get(target, (0.0, 0.0))[1] if target >= 0 else 0.0
        dbin = by_defense[_bin_index(tgt_def)] if (hit and target >= 0) else None
        damage = 0.0
        if hit:
            power = 4.0 + 8.0 * atk
            damage = power * (1.0 - 0.6 * tgt_def)
        for d in (overall, abin):
            d["n_attacks"] += 1
            d["cost"] += ATTACK_COST
            d["loot"] += float(e.loot)
            d["counter"] += float(e.counter_damage)
            d["net"] += float(e.actor_energy_delta)
            if hit:
                d["n_hits"] += 1
                d["damage"] += damage
            if e.status == "ok":
                d["n_kills"] += 1
        if dbin is not None:
            dbin["n_attacks"] += 1
            dbin["n_hits"] += 1
            dbin["damage"] += damage
            dbin["counter"] += float(e.counter_damage)
            if e.status == "ok":
                dbin["n_kills"] += 1

    for u, (atk, dfn) in traits.items():
        n = exposure.get(int(u), 0)
        overall["exposure"] += n
        by_attack[_bin_index(atk)]["exposure"] += n
        by_defense[_bin_index(dfn)]["exposure"] += n

    return {
        "overall": _finalize_bin(overall),
        "by_attack_bin": {f"{BIN_EDGES[i]:.1f}-{BIN_EDGES[i + 1]:.1f}":
                          _finalize_bin(by_attack[i])
                          for i in range(len(BIN_EDGES) - 1)},
        "by_defense_bin": {f"{BIN_EDGES[i]:.1f}-{BIN_EDGES[i + 1]:.1f}":
                           _finalize_bin(by_defense[i])
                           for i in range(len(BIN_EDGES) - 1)},
    }


def _early_flows(log, max_tick: int = EARLY_TICKS) -> Dict:
    """首批 max_tick 的能量流（每 ind·tick）：攻击成本 / loot / 反击 / 进食。"""
    exposure = sum(len(f.uids) for f in log.frames
                   if int(f.tick) <= max_tick)
    out = {"window_ticks": int(max_tick), "exposure_ind_ticks": int(exposure),
           "n_attacks": 0, "attack_cost": 0.0, "loot": 0.0,
           "counter": 0.0, "eat_gain": 0.0, "move_cost": 0.0,
           "other_action_cost": 0.0}
    for e in log.events:
        if int(e.tick) > max_tick:
            continue
        kind = e.kind
        if kind == "attack" and e.status != "actor_dead":
            out["n_attacks"] += 1
            out["attack_cost"] += ATTACK_COST
            out["loot"] += float(e.loot)
            out["counter"] += float(e.counter_damage)
        elif kind == "eat" and e.status == "ok":
            # EAT delta = -成本 + 进食所得
            out["eat_gain"] += float(e.actor_energy_delta) \
                + ACTION_BASE_COST[int(ActionKind.EAT)]
        elif kind == "move" and e.status != "actor_dead":
            out["move_cost"] += -float(e.actor_energy_delta)
        elif kind in ("share", "signal", "reproduce") \
                and e.status != "actor_dead":
            # actor 净支出（SHARE 含转移给目标的能量，按支出计）
            out["other_action_cost"] += -float(e.actor_energy_delta)
    if exposure:
        for key in ("attack_cost", "loot", "counter", "eat_gain",
                    "move_cost", "other_action_cost"):
            out[f"{key}_per_ind_tick"] = out[key] / exposure
            out[f"{key}_per_capita_tick"] = out[key] / exposure
    return out


def _series_sample(series: Sequence[int]) -> Dict[str, int]:
    return {str(t): int(series[t]) for t in SERIES_SAMPLE if t < len(series)}


def _energy_sample(log) -> Dict[str, Dict]:
    picks = {0, 50, 100, 200, 500, 1000, 2000, 3000, 4000}
    out: Dict[str, Dict] = {}
    for f in log.frames:
        if int(f.tick) in picks and f.energies.size:
            e = np.asarray(f.energies, dtype=np.float64)
            out[str(int(f.tick))] = {
                "n": int(e.size), "mean": float(e.mean()),
                "sd": float(e.std()), "min": float(e.min()),
                "max": float(e.max()),
            }
    return out


def _run_diagnosis(seed: int, spread: float, ticks: int) -> Dict:
    cfg = build_b16_config("b16_full", seed, ticks, spread=spread)
    sim = BiosphereSimulator(cfg)
    founders = sim.population.individuals
    init_traits = _controller_traits(sim)
    init = {
        "n": len(founders),
        "energy": _trait_stats([ind.energy for ind in founders]),
        "defense": _trait_stats([init_traits[ind.uid][1]
                                 for ind in founders]),
        "attack": _trait_stats([init_traits[ind.uid][0]
                                for ind in founders]),
    }
    log = sim.run()
    traits = _controller_traits(sim)
    alive_series = [int(len(f.uids)) for f in log.frames]
    peak = int(max(alive_series)) if alive_series else 0
    peak_tick = int(np.argmax(alive_series)) if alive_series else -1
    first_zero = next((i for i, n in enumerate(alive_series) if n == 0), -1)
    below_half = next((i for i in range(peak_tick, len(alive_series))
                       if alive_series[i] <= max(1, peak // 2)), -1)
    causes: Dict[str, int] = {}
    for _uid, cause, _tick in log.deaths:
        causes[str(cause)] = causes.get(str(cause), 0) + 1
    return {
        "schema": "biosphere-b17-diag-v0.1",
        "arm": "b16_full",
        "seed": int(seed),
        "ticks": int(ticks),
        "spread": float(spread),
        "initial": init,
        "population": {
            "final_alive": int(alive_series[-1]) if alive_series else 0,
            "peak_alive": peak,
            "peak_tick": peak_tick,
            "first_zero_tick": first_zero,
            "half_peak_tick": below_half,
            "death_causes": causes,
            "alive_series": alive_series,
            "alive_sample": _series_sample(alive_series),
            "gate": seed_gate(alive_series),
        },
        "energy_by_tick": _energy_sample(log),
        "early_flows": _early_flows(log),
        "ledger": _attack_ledger(log, traits),
        "early_ledger": _attack_ledger(log, traits, max_tick=EARLY_TICKS),
    }


def run_diagnosis_job(job: "JobSpec") -> Dict:
    """parallel._run_one 的诊断作业体。"""
    return _run_diagnosis(job.seed, float(job.extra.get("spread", 0.0)),
                          job.ticks)


def _aggregate(rows: List[Dict], spread: float) -> Dict:
    gates = {}
    for r in rows:
        gates[r["seed"]] = r["population"]["gate"]
    n_pass = sum(1 for g in gates.values() if g["passed"])
    pooled = {
        "n_attacks": sum(r["ledger"]["overall"]["n_attacks"] for r in rows),
        "n_kills": sum(r["ledger"]["overall"]["n_kills"] for r in rows),
        "cost": sum(r["ledger"]["overall"]["cost"] for r in rows),
        "loot": sum(r["ledger"]["overall"]["loot"] for r in rows),
        "counter": sum(r["ledger"]["overall"]["counter"] for r in rows),
        "damage": sum(r["ledger"]["overall"]["damage"] for r in rows),
        "net": sum(r["ledger"]["overall"]["net"] for r in rows),
    }
    n = pooled["n_attacks"]
    pooled["net_per_attack"] = pooled["net"] / n if n else 0.0
    pooled["loot_per_kill"] = (pooled["loot"] / pooled["n_kills"]
                               if pooled["n_kills"] else 0.0)
    pooled["kill_rate_per_attack"] = (pooled["n_kills"] / n) if n else 0.0
    pooled["cost_per_attack"] = pooled["cost"] / n if n else 0.0
    pooled["counter_per_attack"] = pooled["counter"] / n if n else 0.0
    deaths: Dict[str, int] = {}
    for r in rows:
        for k, v in r["population"]["death_causes"].items():
            deaths[k] = deaths.get(k, 0) + int(v)
    return {
        "spread": float(spread),
        "n_seeds": len(rows),
        "gate_passed": n_pass,
        "gate": "过门" if n_pass >= 4 else "未过（需 ≥4/5）",
        "per_seed_final_alive": {r["seed"]: r["population"]["final_alive"]
                                 for r in rows},
        "per_seed_first_zero_tick": {
            r["seed"]: r["population"]["first_zero_tick"] for r in rows},
        "per_seed_half_peak_tick": {
            r["seed"]: r["population"]["half_peak_tick"] for r in rows},
        "death_causes": deaths,
        "initial_defense_mean": [r["initial"]["defense"].get("mean", 0.0)
                                 for r in rows],
        "initial_attack_mean": [r["initial"]["attack"].get("mean", 0.0)
                                for r in rows],
        "initial_attack_frac_ge_0p8": [
            r["initial"]["attack"].get("frac_ge_0p8", 0.0) for r in rows],
        "initial_defense_frac_ge_0p8": [
            r["initial"]["defense"].get("frac_ge_0p8", 0.0) for r in rows],
        "initial_energy": {
            "mean": [r["initial"]["energy"].get("mean", 0.0) for r in rows],
            "sd": [r["initial"]["energy"].get("sd", 0.0) for r in rows],
        },
        "early_flows_mean": _mean_dict([r["early_flows"] for r in rows]),
        "early_ledger": _pool_ledger([r["early_ledger"] for r in rows]),
        "full_ledger": _pool_ledger([r["ledger"] for r in rows]),
        "ledger": pooled,
        "rows": rows,
    }


def _mean_dict(dicts: List[Dict]) -> Dict:
    keys = set()
    for d in dicts:
        keys.update(k for k, v in d.items() if isinstance(v, (int, float)))
    return {k: float(np.mean([d.get(k, 0.0) for d in dicts]))
            for k in sorted(keys)}


def _sum_bins(bins: List[Dict]) -> Dict:
    acc = _empty_bin()
    for d in bins:
        for k in ("n_attacks", "n_hits", "n_kills", "exposure"):
            acc[k] += int(d.get(k, 0))
        for k in ("cost", "loot", "counter", "damage", "net"):
            acc[k] += float(d.get(k, 0.0))
    return acc


def _pool_ledger(ledgers: List[Dict]) -> Dict:
    """跨 seed 汇总分箱账本（计数相加；浓度按 exposure 重算）。"""
    out = {"overall": _finalize_bin(_sum_bins([led["overall"]
                                               for led in ledgers])),
           "by_attack_bin": {}, "by_defense_bin": {}}
    for section in ("by_attack_bin", "by_defense_bin"):
        bins = set()
        for led in ledgers:
            bins.update(led[section].keys())
        for b in sorted(bins):
            out[section][b] = _finalize_bin(
                _sum_bins([led[section][b] for led in ledgers
                           if b in led[section]]))
    return out


def b17_diagnose(seeds: List[int], ticks: int, spreads: List[float],
                 out_dir: Path, workers: int = 1,
                 ckpt_every: int = 0) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [JobSpec(arm="b16_full", seed=s, ticks=ticks, kind="b17_diag",
                    ckpt_every=ckpt_every, extra={"spread": float(sp)})
            for sp in spreads for s in seeds]
    import time
    t0 = time.time()
    if workers > 1:
        res = run_parallel(jobs, workers=workers)
        meta = {"workers": workers, "wall_s": time.time() - t0,
                "mode": "parallel"}
    else:
        from biosphere.parallel import run_serial
        res = run_serial(jobs)
        meta = {"workers": 1, "wall_s": time.time() - t0, "mode": "serial"}
    bad = [r for r in res if not r["ok"]]
    if bad:
        raise RuntimeError(f"诊断作业失败: {bad[0]['label']}: {bad[0]['error']}")
    rows = [r["row"] for r in res]
    report: Dict = {
        "schema": "biosphere-b17-diag-summary-v0.1",
        "ticks": int(ticks), "seeds": list(seeds),
        "spreads": [float(x) for x in spreads],
        "parallel": meta, "arms": {},
        "note": "诊断用（非判据）；只读，不改参数、不跑 B16 五条判据",
    }
    for sp in spreads:
        sel = [r for r in rows if abs(float(r["spread"]) - float(sp)) < 1e-12]
        report["arms"][f"spread={float(sp):g}"] = _aggregate(sel, float(sp))
    out_path = out_dir / f"b17_diag_summary_t{ticks}.json"
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, sort_keys=True, indent=1)
        fh.write("\n")

    print(f"=== B17.1 生态门诊断：{ticks} ticks x {len(seeds)} seeds "
          f"x {len(spreads)} spread 档（{meta['mode']} "
          f"{meta.get('workers', 1)} 进程，{meta['wall_s']:.1f}s）===")
    for key, arm in report["arms"].items():
        print(f"\n[{key}] 生态门 {arm['gate_passed']}/{arm['n_seeds']} "
              f"{arm['gate']}")
        print(f"  death_causes={arm['death_causes']} "
              f"final_alive={arm['per_seed_final_alive']} "
              f"first_zero={arm['per_seed_first_zero_tick']}")
        print(f"  init attack mean={arm['initial_attack_mean']} "
              f"sat={arm['initial_attack_frac_ge_0p8']}")
        print(f"  init defense mean={arm['initial_defense_mean']} "
              f"sat={arm['initial_defense_frac_ge_0p8']}")
        print(f"  init energy mean={arm['initial_energy']['mean']} "
              f"sd={arm['initial_energy']['sd']}")
        ef = arm["early_flows_mean"]
        print(f"  early100: n_attacks={int(ef.get('n_attacks', 0))} "
              f"cost/ind·t={ef.get('attack_cost_per_ind_tick', 0):.3f} "
              f"eat/ind·t={ef.get('eat_gain_per_ind_tick', 0):.3f} "
              f"move/ind·t={ef.get('move_cost_per_ind_tick', 0):.3f}")
        led = arm["ledger"]
        print(f"  full ledger: attacks={led['n_attacks']} kills={led['n_kills']} "
              f"cost/atk={led['cost_per_attack']:.3f} "
              f"loot/kill={led['loot_per_kill']:.3f} "
              f"counter/atk={led['counter_per_attack']:.3f} "
              f"net/atk={led['net_per_attack']:+.3f}")
        for b, d in arm["full_ledger"]["by_attack_bin"].items():
            print(f"    atk {b}: n={d['n_attacks']:<6} "
                  f"kill_rate={d['kill_rate_per_attack']:.3f} "
                  f"loot/kill={d['loot_per_kill']:.2f} "
                  f"net/atk={d['net_per_attack']:+.3f} "
                  f"dmg/hit={d['damage_per_hit']:.2f}")
    print(f"\nsummary: {out_path}")
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m biosphere.diagnose",
        description="B17.1 生态门诊断（spread 梯度 + 攻击收支账本；非判据）")
    ap.add_argument("--seeds", type=str, default="41-45")
    ap.add_argument("--ticks", type=int, default=5000)
    ap.add_argument("--spreads", type=str, default="0,0.5,1.0,1.5,2.0")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=str, default="results/biosphere")
    args = ap.parse_args(argv)
    b17_diagnose(_parse_seeds(args.seeds), args.ticks,
                 _parse_floats(args.spreads), Path(args.out),
                 workers=args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
