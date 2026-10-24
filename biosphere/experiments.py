"""biosphere.experiments：B 线对照实验跑批器（B6 威胁对照 / B8 捕食双型）。

用法（仓库/worktree 根目录）::

    python -m biosphere.experiments control --seeds 42,43,44 --ticks 1000
    python -m biosphere.experiments dualtype --ticks 2000 --seed 42

- **control**（B6）：两臂对照——①`threat_sensing=True`（邻格威胁排斥 +
  受击记忆逃逸项）②`False`（个体完全看不到威胁），各 seed 跑 ticks，
  对比 fleeing/chasing 等标签与攻击结构；并打印「命中窗口」距离增量分布
  （受击后 3 tick 内与攻击者距离的变化）——用于判断
  `FLEE_DIST_GAIN_MIN=0.5` 是否结构性可达。
- **dualtype**（B8）：人工初始化高攻击型 / 低攻击型两类（只差攻击基因
  一个槽位），跑 ticks，验证 chasing 与 fleeing 同时出现。

输出：metrics JSON（定键 sort_keys）+ 终端 ASCII 表；落 `results/biosphere/`
（gitignore 覆盖）。同参数两次运行逐位一致（与 demo 同纪律）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from biosphere.genome import PARAM_SLOTS
from biosphere.checkpoint import load_checkpoint
from biosphere.metrics import (
    BEHAVIOR_LABELS, BehaviorLabeler, metrics_json, predation_json,
    run_digest,
)
from biosphere.parallel import (
    JobSpec, manifest, run_parallel, run_serial, write_manifest,
)
from biosphere.predation import PredationConfig, oracle_group
from biosphere.simulator import BioMVPConfig, BiosphereSimulator

#: 高/低攻击型的原始基因值（slot = PARAM_SLOTS["attack_drive"]）
ATTACK_HI_GENE: float = 3.0
ATTACK_LO_GENE: float = -3.0


def _summarize(sim, log, seed: int, ticks: int,
               threat_sensing: bool) -> Dict:
    labels = BehaviorLabeler(log).label_all(sim.world)
    counts = {lab: sum(1 for v in labels.values() if v[lab])
              for lab in BEHAVIOR_LABELS}
    ev = [e for e in log.events if e.kind == "attack"]
    n_kill = sum(1 for e in ev if e.status == "ok")
    n_wound = sum(1 for e in ev if e.status == "attack_survived")
    n_miss = sum(1 for e in ev if e.status == "attack_no_target")
    m = metrics_json(log, sim.world, seed=seed, ticks=ticks)
    m["attack_structure"] = {
        "attacks": len(ev), "kills": n_kill, "wounded": n_wound,
        "missed": n_miss,
        "lethality": (n_kill / (n_kill + n_wound)) if (n_kill + n_wound) else 0.0,
    }
    # flee/chase 窗口分布由 metrics_json 统一产出（B6/B10 永久仪器）
    assert "flee_gain_distribution" in m and "chase_gain_distribution" in m
    m["config"] = {"threat_sensing": bool(threat_sensing)}
    return m


def _run_arm(seed: int, ticks: int, threat_sensing: bool,
             ckpt_dir=None, resume: Optional[str] = None,
             ckpt_every: int = 500) -> Dict:
    if resume:
        sim = load_checkpoint(resume)
    else:
        sim = BiosphereSimulator(
            BioMVPConfig(ticks=ticks, seed=seed,
                         threat_sensing=threat_sensing))
    log = sim.run(ckpt_dir=ckpt_dir, ckpt_every=ckpt_every,
                  tag=f"b6_{'on' if threat_sensing else 'off'}_s{seed}")
    return _summarize(sim, log, seed, ticks, threat_sensing)


def control(seeds: List[int], ticks: int, out_dir: Path,
            ckpt_dir=None, resume: Optional[str] = None) -> Dict:
    """B6 两臂对照 + 命中窗口分布；返回汇总 dict（可 JSON 化）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict = {"ticks": ticks, "seeds": seeds, "arms": {}}
    for arm, ts in (("threat_on", True), ("threat_off", False)):
        rows = []
        for seed in seeds:
            m = _run_arm(seed, ticks, ts, ckpt_dir=ckpt_dir,
                         resume=resume if arm == "threat_on" else None)
            rows.append({
                "seed": seed,
                "fleeing": m["behaviors"]["counts"]["fleeing"],
                "chasing": m["behaviors"]["counts"]["chasing"],
                "foraging": m["behaviors"]["counts"]["foraging"],
                "grouping": m["behaviors"]["counts"]["grouping"],
                "signaling": m["behaviors"]["counts"]["signaling"],
                "final_alive": m["population"]["final_alive"],
                "max_generation": m["lineage"]["max_generation"],
                "attack_structure": m["attack_structure"],
                "flee_gain_distribution": m["flee_gain_distribution"],
                "chase_gain_distribution": m["chase_gain_distribution"],
            })
            out_path = out_dir / f"b6_control_{arm}_s{seed}_t{ticks}.json"
            with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(m, fh, sort_keys=True, indent=1)
                fh.write("\n")
        report["arms"][arm] = rows

    # 终端表（纯 ASCII）
    print(f"=== B6 control: {ticks} ticks x {len(seeds)} seeds ===")
    header = (f"{'arm':<12}{'seed':<6}{'fleeing':<9}{'chasing':<9}"
              f"{'foraging':<10}{'grouping':<10}{'signaling':<11}"
              f"{'alive':<7}{'kills':<7}{'wounded':<9}{'leth%':<7}"
              f"{'win':<6}{'gain>=0.5':<10}")
    print(header)
    for arm, rows in report["arms"].items():
        for r in rows:
            a = r["attack_structure"]
            g = r["flee_gain_distribution"]
            print(f"{arm:<12}{r['seed']:<6}{r['fleeing']:<9}{r['chasing']:<9}"
                  f"{r['foraging']:<10}{r['grouping']:<10}{r['signaling']:<11}"
                  f"{r['final_alive']:<7}{a['kills']:<7}{a['wounded']:<9}"
                  f"{a['lethality']*100:<6.1f}{g['n_window_available']:<6}"
                  f"{g['frac_gain_ge_0p5']*100:<9.1f}%")
    print("win = 命中窗口可用数（受击后 3tick 有帧）；gain>=0.5 = 距离增量达标签阈值比例")
    out_path = out_dir / f"b6_control_summary_t{ticks}.json"
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, sort_keys=True, indent=1)
        fh.write("\n")
    print(f"summary: {out_path}")
    return report


def dualtype(seed: int, ticks: int, out_dir: Path,
             n_pred: int = 20, n_prey: int = 40,
             ckpt_dir=None, resume: Optional[str] = None) -> Dict:
    """B8 捕食者-猎物双型对照（只差攻击基因一个槽位）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        sim = load_checkpoint(resume)
    else:
        sim = BiosphereSimulator(
            BioMVPConfig(ticks=ticks, seed=seed, init_pop=0))
        rng = np.random.default_rng(seed + 1)   # 放置独立流，不扰主 rng
        land = np.flatnonzero(sim.world.surface_type == 0)
        occupied = {ind.cell for ind in sim.population.individuals}
        free = [int(c) for c in land if int(c) not in occupied]
        picks = rng.choice(free, size=n_pred + n_prey, replace=False)
        slot = PARAM_SLOTS["attack_drive"]
        for i, cell in enumerate(sorted(int(c) for c in picks)):
            overrides = ({slot: ATTACK_HI_GENE} if i < n_pred
                         else {slot: ATTACK_LO_GENE})
            sim.spawn_founder(cell, sim.cfg.init_energy, overrides)
    log = sim.run(ckpt_dir=ckpt_dir, ckpt_every=ckpt_every,
                  tag=f"b8_dual_s{seed}")
    m = _summarize(sim, log, seed, ticks, threat_sensing=True)
    m["dualtype"] = {"n_predators": n_pred, "n_prey": n_prey,
                     "attack_hi_gene": ATTACK_HI_GENE,
                     "attack_lo_gene": ATTACK_LO_GENE}
    # 初代双型分解（uid < n_pred+n_prey）：两类创始者的行为标签计数
    labeler = BehaviorLabeler(log)
    founder_labels = {
        uid: labeler.label_individual(uid, sim.world)
        for uid in range(n_pred + n_prey) if uid in labeler._ticks_of
    }
    breakdown: Dict[str, Dict] = {}
    for name, group in (("predators", range(n_pred)),
                        ("prey", range(n_pred, n_pred + n_prey))):
        rows = [founder_labels[u] for u in group if u in founder_labels]
        breakdown[name] = {
            "n": len(rows),
            **{lab: sum(1 for r in rows if r[lab])
               for lab in BEHAVIOR_LABELS},
        }
    m["dualtype"]["founder_breakdown"] = breakdown
    out_path = out_dir / f"b8_dualtype_s{seed}_t{ticks}.json"
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(m, fh, sort_keys=True, indent=1)
        fh.write("\n")

    pop = m["population"]
    beh = m["behaviors"]["counts"]
    a = m["attack_structure"]
    print("=== B8 dualtype (predator hi-attack / prey lo-attack) ===")
    print(f"seed={seed} ticks={ticks} predators={n_pred} prey={n_prey}")
    print(f"final_alive={pop['final_alive']} born={pop['total_born']} "
          f"died={pop['total_died']} max_generation={m['lineage']['max_generation']}")
    print(f"behaviors(all): chasing={beh['chasing']} fleeing={beh['fleeing']} "
          f"foraging={beh['foraging']} grouping={beh['grouping']} "
          f"signaling={beh['signaling']}")
    for name, b in breakdown.items():
        extra = " ".join(f"{k}={v}" for k, v in b.items() if k != "n")
        print(f"behaviors({name} founders, n={b['n']}): {extra}")
    print(f"attacks={a['attacks']} kills={a['kills']} wounded={a['wounded']} "
          f"lethality={a['lethality']:.1%}")
    ch = m["chase_gain_distribution"]
    print(f"chase_gain: hits={ch['n_hits']} window={ch['n_window_available']} "
          f"drop>=0.5: {ch['frac_drop_ge_0p5']*100:.1f}% "
          f"already_adjacent@t-3: {ch['frac_already_adjacent_at_t3']*100:.1f}% "
          f"target_closed: {ch['frac_target_closed']*100:.1f}% "
          f"attacker_closed: {ch['frac_attacker_closed']*100:.1f}%")
    print(f"metrics_json: {out_path}")
    return m


def _w4_arm(seed: int, ticks: int, variant: str,
             ckpt_dir=None, resume: Optional[str] = None,
             ckpt_every: int = 500) -> Dict:
    """W4 单臂：variant ∈ full/no_adhesion/no_group_selection/full_mixing/no_signal。"""
    from biosphere.groups import W4Config
    kw = dict(enabled=True)
    if variant == "no_adhesion":
        kw["adhesion"] = False
    elif variant == "no_group_selection":
        kw["group_reproduction"] = False
    elif variant == "full_mixing":
        kw["full_mixing"] = True
    elif variant == "no_signal":
        kw["signal"] = False
    elif variant != "full":
        raise ValueError(f"未知 W4 臂: {variant}")
    if resume:
        sim = load_checkpoint(resume)
    else:
        sim = BiosphereSimulator(
            BioMVPConfig(ticks=ticks, seed=seed, w4=W4Config(**kw)))
    log = sim.run(ckpt_dir=ckpt_dir, ckpt_every=ckpt_every,
                  tag=f"b12_{variant}_s{seed}")
    return _summarize(sim, log, seed, ticks, threat_sensing=True)


#: W4 五臂（完整 + 四零模型）
W4_ARMS = ("full", "no_adhesion", "no_group_selection", "full_mixing",
           "no_signal")


def w4(seeds: List[int], ticks: int, out_dir: Path,
       ckpt_dir=None, resume: Optional[str] = None,
       ckpt_every: int = 500) -> Dict:
    """B12 W4b 五臂 × seeds：粘附链团（主张）/ 邻接团（对照）+ 生态门
    + 预注册组选择判据。"""
    from biosphere.criteria import (
        evaluate_ecological_gate, evaluate_group_selection_criterion,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict = {"ticks": ticks, "seeds": seeds, "arms": {}, "gates": {}}
    for arm in W4_ARMS:
        rows = []
        for seed in seeds:
            m = _w4_arm(seed, ticks, arm, ckpt_dir=ckpt_dir,
                        resume=resume if arm == "full" else None,
                        ckpt_every=ckpt_every)
            chain = m["groups"]["adhesion_chain"]
            ctrl = m["groups"]["adjacency_control"]
            traj = chain.get("adhesion_trajectory", {})
            repro = chain.get("repro_composition", {})
            cheaters = chain.get("cheaters", {})
            rows.append({
                "seed": seed,
                "final_alive": m["population"]["final_alive"],
                "born": m["population"]["total_born"],
                "died": m["population"]["total_died"],
                "alive_series": m["population"]["alive_series"],
                # 粘附链团（主张镜头）
                "n_multi_chain": chain["n_multi_chain_final"],
                "in_multi_chain": chain["in_multi_chain_final"],
                "max_chain_size": chain["max_chain_size_final"],
                "mean_chain_size": chain["mean_chain_size_final"],
                "chain_ratio_mean": chain["chain_ratio_mean"],
                "adhesion_gene_mean": traj.get("whole_run_mean", {}).get(
                    "adhesion_gene_mean", 0.0),
                "adhesion_gene_var": traj.get("whole_run_mean", {}).get(
                    "adhesion_gene_var", 0.0),
                "sticky_fraction": traj.get("whole_run_mean", {}).get(
                    "sticky_fraction", 0.0),
                "adhesion_traj_start": traj.get("start", {}).get(
                    "adhesion_gene_mean"),
                "adhesion_traj_mid": traj.get("mid", {}).get(
                    "adhesion_gene_mean"),
                "adhesion_traj_end": traj.get("end", {}).get(
                    "adhesion_gene_mean"),
                # 组繁殖构成
                "n_repro_events": repro.get("n_events", 0),
                "repro_high": repro.get("n_high_share_events", 0),
                "repro_low": repro.get("n_low_share_events", 0),
                "repro_rate_high": repro.get(
                    "repro_per_eligible_group_tick_high", 0.0),
                "repro_rate_low": repro.get(
                    "repro_per_eligible_group_tick_low", 0.0),
                "children_survive_high": chain.get(
                    "children_survival_by_bucket", {}).get(
                        "high_share", {}).get("survival_fraction", 0.0),
                "children_survive_low": chain.get(
                    "children_survival_by_bucket", {}).get(
                        "low_share", {}).get("survival_fraction", 0.0),
                # 作弊者
                "cheater_freq": cheaters.get("frequency", 0.0),
                "cheater_per_group_mean": cheaters.get(
                    "per_final_group_mean_fraction", 0.0),
                # 邻接团（对照镜头）
                "adj_multi": ctrl["n_multi_final"],
                "adj_in_multi": ctrl["n_in_multi_final"],
                "adj_max_size": ctrl["max_size_final"],
                "within_var": ctrl["within_var_final"],
                "lineage_survival": ctrl["lineage_survival"]["fraction"],
                "behaviors": m["behaviors"]["counts"],
            })
            out_path = out_dir / f"b12_w4b_{arm}_s{seed}_t{ticks}.json"
            with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(m, fh, sort_keys=True, indent=1)
                fh.write("\n")
        report["arms"][arm] = rows
        report["gates"][arm] = evaluate_ecological_gate(
            {r["seed"]: r["alive_series"] for r in rows})

    # 预注册判据：full vs 关组选择（逐 seed 配对，≥4/5 方向一致）
    crit = evaluate_group_selection_criterion(report["arms"]["full"],
                                              report["arms"]["no_group_selection"])
    report["criterion"] = crit
    report["criterion_note"] = (
        "仅当 full 与关组选择两臂生态门均过门时才判读；"
        f"生态门：full={report['gates']['full']['verdict']}，"
        f"关组选择={report['gates']['no_group_selection']['verdict']}")

    print(f"=== B12 W4b control: {ticks} ticks x {len(seeds)} seeds ===")
    header = (f"{'arm':<20}{'seed':<6}{'alive':<7}{'gate':<6}{'multiC':<8}"
              f"{'inC':<6}{'maxC':<6}{'chainR':<8}{'adhGene':<9}{'sticky':<8}"
              f"{'repro':<7}{'hi/lo':<8}{'cheat':<7}{'adjMax':<7}")
    print(header)
    for arm, rows in report["arms"].items():
        gate = report["gates"][arm]["per_seed"]
        for r in rows:
            g = "pass" if gate[r["seed"]]["passed"] else "FAIL"
            print(f"{arm:<20}{r['seed']:<6}{r['final_alive']:<7}{g:<6}"
                  f"{r['n_multi_chain']:<8}{r['in_multi_chain']:<6}"
                  f"{r['max_chain_size']:<6}{r['chain_ratio_mean']:<8.3f}"
                  f"{r['adhesion_gene_mean']:<9.3f}{r['sticky_fraction']:<8.3f}"
                  f"{r['n_repro_events']:<7}"
                  f"{str(r['repro_high']) + '/' + str(r['repro_low']):<8}"
                  f"{r['cheater_freq']:<7.2f}{r['adj_max_size']:<7}")
    print("multiC/inC/maxC = 粘附链团（主张镜头）；chainR=链团比例（时间平均）；")
    print("adhGene=粘附基因频率（全程均值）；sticky=粘附个体占比；")
    print("repro=组繁殖事件数；hi/lo=高/低共享团事件；cheat=作弊者频率（含死者）；")
    print("adjMax=邻接团最大团（对照镜头，密度混淆）")
    print()
    for arm, g in report["gates"].items():
        print(f"[gate] {arm:<20} {g['n_passed']}/{g['n_total']} "
              f"{g['verdict']}")
    print(f"[criterion] full vs 关组选择：chain {crit['n_chain_direction_ok']}"
          f"/{crit['n_seeds']}，adhesion {crit['n_adhesion_direction_ok']}"
          f"/{crit['n_seeds']}，both {crit['n_both_direction_ok']}"
          f"/{crit['n_seeds']}（需 ≥{crit['min_consistent_required']}）"
          f" ⇒ **{crit['verdict']}**")
    for d in crit["per_seed"]:
        print(f"  seed {d['seed']}: chainR full {d['full_chain_ratio']:.3f} vs "
              f"z2 {d['z2_chain_ratio']:.3f} ({d['chain_direction_ok']})；"
              f"adhGene full {d['full_adhesion_gene']:.3f} vs "
              f"z2 {d['z2_adhesion_gene']:.3f} ({d['adhesion_direction_ok']})")
    out_path = out_dir / f"b12_w4b_summary_t{ticks}.json"
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, sort_keys=True, indent=1)
        fh.write("\n")
    print(f"summary: {out_path}")
    return report


#: W4c 臂定义（预注册 §13.12）：每组 = 处理臂 + 同参数 z2 对照臂
W4C_ARMS: Dict[str, Dict] = {
    "thr40_full": {"group": "G1 门槛扫描", "role": "full",
                   "w4": {"group_repro_thresh_per_capita": 40.0}},
    "thr40_z2": {"group": "G1 门槛扫描", "role": "z2",
                 "w4": {"group_repro_thresh_per_capita": 40.0,
                        "group_reproduction": False}},
    "thr80_full": {"group": "G1 门槛扫描", "role": "full",
                   "w4": {}},
    "thr80_z2": {"group": "G1 门槛扫描", "role": "z2",
                 "w4": {"group_reproduction": False}},
    "thr160_full": {"group": "G1 门槛扫描", "role": "full",
                    "w4": {"group_repro_thresh_per_capita": 160.0}},
    "thr160_z2": {"group": "G1 门槛扫描", "role": "z2",
                  "w4": {"group_repro_thresh_per_capita": 160.0,
                         "group_reproduction": False}},
    "kin_full": {"group": "G2 亲缘结构", "role": "full",
                 "w4": {"kin_cohesion": True}},
    "kin_z2": {"group": "G2 亲缘结构", "role": "z2",
               "w4": {"kin_cohesion": True, "group_reproduction": False}},
    "benMid_full": {"group": "G3 粘附收益", "role": "full",
                    "w4": {"adhesion_benefit": 0.5}},
    "benMid_z2": {"group": "G3 粘附收益", "role": "z2",
                  "w4": {"adhesion_benefit": 0.5,
                         "group_reproduction": False}},
    "benStrong_full": {"group": "G3 粘附收益", "role": "full",
                       "w4": {"adhesion_benefit": 1.0}},
    "benStrong_z2": {"group": "G3 粘附收益", "role": "z2",
                     "w4": {"adhesion_benefit": 1.0,
                            "group_reproduction": False}},
}


def _w4c_arm(seed: int, ticks: int, arm: str, ckpt_dir=None,
             ckpt_every: int = 500) -> Dict:
    from biosphere.groups import W4Config
    spec = W4C_ARMS[arm]
    cfg = BioMVPConfig(ticks=ticks, seed=seed,
                       w4=W4Config(enabled=True, **spec["w4"]))
    sim = BiosphereSimulator(cfg)
    log = sim.run(ckpt_dir=ckpt_dir, ckpt_every=ckpt_every,
                  tag=f"b13_{arm}_s{seed}")
    return _summarize(sim, log, seed, ticks, threat_sensing=True)


def _w4c_row(m: Dict) -> Dict:
    chain = m["groups"]["adhesion_chain"]
    traj = chain.get("adhesion_trajectory", {})
    repro = chain.get("repro_composition", {})
    cheaters = chain.get("cheaters", {})
    return {
        "final_alive": m["population"]["final_alive"],
        "born": m["population"]["total_born"],
        "alive_series": m["population"]["alive_series"],
        "n_multi_chain": chain["n_multi_chain_final"],
        "in_multi_chain": chain["in_multi_chain_final"],
        "max_chain_size": chain["max_chain_size_final"],
        "chain_ratio_mean": chain["chain_ratio_mean"],
        "adhesion_gene_mean": traj.get("whole_run_mean", {}).get(
            "adhesion_gene_mean", 0.0),
        "adhesion_gene_start": traj.get("start", {}).get(
            "adhesion_gene_mean", 0.0),
        "adhesion_gene_end": traj.get("end", {}).get(
            "adhesion_gene_mean", 0.0),
        "sticky_fraction": traj.get("whole_run_mean", {}).get(
            "sticky_fraction", 0.0),
        "n_repro_events": repro.get("n_events", 0),
        "repro_high": repro.get("n_high_share_events", 0),
        "repro_low": repro.get("n_low_share_events", 0),
        "repro_rate_high": repro.get(
            "repro_per_eligible_group_tick_high", 0.0),
        "repro_rate_low": repro.get(
            "repro_per_eligible_group_tick_low", 0.0),
        "cheater_freq": cheaters.get("frequency", 0.0),
        "cheater_per_group_mean": cheaters.get(
            "per_final_group_mean_fraction", 0.0),
    }


def w4c(seeds: List[int], ticks: int, out_dir: Path, ckpt_dir=None,
        resume: Optional[str] = None, ckpt_every: int = 500) -> Dict:
    """B13 W4c：三旋钮 × 12 臂 × seeds；逐组预注册判据。"""
    from biosphere.criteria import (
        evaluate_ecological_gate, evaluate_group_selection_criterion,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict = {"ticks": ticks, "seeds": seeds, "arms": {}, "gates": {}}
    for arm, spec in W4C_ARMS.items():
        rows = []
        for seed in seeds:
            m = _w4c_arm(seed, ticks, arm, ckpt_dir=ckpt_dir,
                         ckpt_every=ckpt_every)
            row = {"seed": seed, "group": spec["group"], "role": spec["role"]}
            row.update(_w4c_row(m))
            rows.append(row)
            out_path = out_dir / f"b13_w4c_{arm}_s{seed}_t{ticks}.json"
            with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(m, fh, sort_keys=True, indent=1)
                fh.write("\n")
        report["arms"][arm] = rows
        report["gates"][arm] = evaluate_ecological_gate(
            {r["seed"]: r["alive_series"] for r in rows})

    # 逐组判据：full 臂 vs 同参数 z2 臂
    report["criteria"] = {}
    for arm, spec in W4C_ARMS.items():
        if spec["role"] != "full":
            continue
        z2 = arm.replace("_full", "_z2")
        crit = evaluate_group_selection_criterion(
            report["arms"][arm], report["arms"][z2])
        crit["group"] = spec["group"]
        crit["arm_full"] = arm
        crit["arm_z2"] = z2
        crit["gate_full"] = report["gates"][arm]["verdict"]
        crit["gate_z2"] = report["gates"][z2]["verdict"]
        report["criteria"][arm] = crit

    print(f"=== B13 W4c control: {ticks} ticks x {len(seeds)} seeds "
          f"x {len(W4C_ARMS)} arms ===")
    header = (f"{'arm':<16}{'seed':<6}{'alive':<7}{'gate':<6}{'multiC':<8}"
              f"{'maxC':<6}{'chainR':<8}{'adhS':<7}{'adhE':<7}{'sticky':<8}"
              f"{'repro':<7}{'hi/lo':<8}{'cheat':<7}")
    print(header)
    for arm, rows in report["arms"].items():
        gate = report["gates"][arm]["per_seed"]
        for r in rows:
            g = "pass" if gate[r["seed"]]["passed"] else "FAIL"
            print(f"{arm:<16}{r['seed']:<6}{r['final_alive']:<7}{g:<6}"
                  f"{r['n_multi_chain']:<8}{r['max_chain_size']:<6}"
                  f"{r['chain_ratio_mean']:<8.3f}"
                  f"{r['adhesion_gene_start']:<7.2f}{r['adhesion_gene_end']:<7.2f}"
                  f"{r['sticky_fraction']:<8.3f}{r['n_repro_events']:<7}"
                  f"{str(r['repro_high']) + '/' + str(r['repro_low']):<8}"
                  f"{r['cheater_freq']:<7.2f}")
    print()
    for arm, g in report["gates"].items():
        print(f"[gate] {arm:<16} {g['n_passed']}/{g['n_total']} {g['verdict']}")
    print()
    for arm, crit in report["criteria"].items():
        print(f"[criterion] {crit['group']} ({arm} vs {crit['arm_z2']}): "
              f"chain {crit['n_chain_direction_ok']}/{crit['n_seeds']}，"
              f"adhesion {crit['n_adhesion_direction_ok']}/{crit['n_seeds']}，"
              f"both {crit['n_both_direction_ok']}/{crit['n_seeds']}"
              f"（需 ≥{crit['min_consistent_required']}）⇒ "
              f"**{crit['verdict']}**"
              f"（gate: {crit['gate_full']} / {crit['gate_z2']}）")
        for d in crit["per_seed"]:
            print(f"  seed {d['seed']}: chainR {d['full_chain_ratio']:.3f} vs "
                  f"{d['z2_chain_ratio']:.3f} ({d['chain_direction_ok']})；"
                  f"adhGene {d['full_adhesion_gene']:.3f} vs "
                  f"{d['z2_adhesion_gene']:.3f} ({d['adhesion_direction_ok']})")
    out_path = out_dir / f"b13_w4c_summary_t{ticks}.json"
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, sort_keys=True, indent=1)
        fh.write("\n")
    print(f"summary: {out_path}")
    return report


def threshold_sensitivity(seed: int, ticks: int, out_dir: Path,
                           levels: Sequence[float] = (0.5, 1.0, 2.0)
                           ) -> Dict:
    """B14.1 阈值三档敏感性：**一次跑批、三次打标**（标签逻辑不变）。

    对同一份观测日志按 levels 缩放全部行为阈值（0.5×/1×/2×），报各档
    标签计数 + 无标签交叉验证；原始分布在 metrics JSON（不过阈值）。
    """
    from biosphere.metrics import BehaviorThresholds, metrics_json
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = BioMVPConfig(ticks=ticks, seed=seed)
    sim = BiosphereSimulator(cfg)
    log = sim.run()
    base = BehaviorThresholds()
    report: Dict = {"seed": seed, "ticks": ticks, "levels": list(levels),
                    "label_counts": {}, "crossval": {}}
    for lv in levels:
        th = BehaviorThresholds(
            forage_gain_min=base.forage_gain_min * lv,
            forage_speed_min=base.forage_speed_min * lv,
            flee_dist_gain_min=base.flee_dist_gain_min * lv,
            chase_dist_drop_min=base.chase_dist_drop_min * lv,
            group_nn_max=base.group_nn_max * lv,
        )
        m = metrics_json(log, sim.world, seed=seed, ticks=ticks,
                         thresholds=th)
        report["label_counts"][f"{lv:g}x"] = m["behaviors"]["counts"]
        if lv == 1.0:
            report["crossval"][f"{lv:g}x"] = m["crossval"]
            out_path = out_dir / f"b14_sensitivity_s{seed}_t{ticks}.json"
            with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(m, fh, sort_keys=True, indent=1)
                fh.write("\n")
    print(f"=== B14.1 threshold sensitivity: seed={seed} ticks={ticks} ===")
    header = (f"{'level':<7}" + "".join(f"{lab:<11}"
                                        for lab in BEHAVIOR_LABELS))
    print(header)
    for lv_key, counts in report["label_counts"].items():
        print(f"{lv_key:<7}" + "".join(f"{counts[lab]:<11}"
                                       for lab in BEHAVIOR_LABELS))
    print("口径注记：标签判定逻辑与历史一致；本表只缩放阈值数字（H1）。")
    print("无标签交叉验证（1× 档）:")
    for lab, d in report["crossval"].get("1x", {}).get("labels", {}).items():
        print(f"  {lab:<9} n={d['n_labeled']:<4} MI(cluster)={d['mi_with_clusters']:.3f} "
              f"MI(best:{d['mi_best_feature_name']})={d['mi_best_feature']:.3f} "
              f"⇒ {d['verdict']}")
    out_path = out_dir / f"b14_sensitivity_summary_s{seed}_t{ticks}.json"
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, sort_keys=True, indent=1)
        fh.write("\n")
    print(f"summary: {out_path}")
    return report


#: C3 v2 臂定义（预注册 §13.14；首轮记忆默认关）
C3_ARMS: Dict[str, Dict] = {
    "c3_full": {"desc": "完整档（徽章+强度+发射耗能；记忆关）", "c3": {}},
    "c3_no_strength": {"desc": "负对照：关强度不对称（strength 不进威力）",
                       "c3": {"strength_effect": False}},
    "c3_badge_reshuffle": {"desc": "负对照：徽章每代重随机（断遗传）",
                           "c3": {"badge_reshuffle": True}},
    "c3_memory": {"desc": "观察臂：开记忆（打斗结果学 response）",
                  "c3": {"memory_enabled": True}},
}


def _c3_arm(seed: int, ticks: int, arm: str, ckpt_dir=None,
            ckpt_every: int = 500) -> Dict:
    from biosphere.c3 import C3Config
    spec = C3_ARMS[arm]
    cfg = BioMVPConfig(ticks=ticks, seed=seed,
                       c3=C3Config(enabled=True, **spec["c3"]))
    sim = BiosphereSimulator(cfg)
    log = sim.run(ckpt_dir=ckpt_dir, ckpt_every=ckpt_every,
                  tag=f"b14_{arm}_s{seed}")
    m = _summarize(sim, log, seed, ticks, threat_sensing=True)
    # 徽章/强度/谱系互信息 + 反应轨迹（从终局种群 + 出生流水算）
    from biosphere.crossval import mutual_info
    from biosphere.develop import develop
    from biosphere.genome import PARAM_SLOTS
    inds = sim.population.individuals
    badges, strengths, roots = [], [], []
    child_set = {c for _p, c, _cell in log.births}
    for ind in inds:
        c = develop(ind.genome)
        badges.append(c.badge)
        strengths.append(c.strength)
        roots.append(0 if ind.parent < 0 or ind.parent not in child_set
                     else 1)
    k = int(max(badges)) + 1 if badges else 1
    badge_hist = {b: badges.count(b) for b in range(max(k, 1))}
    mi_strength = mutual_info(badges, [int(s * 4) for s in strengths]) \
        if badges else 0.0
    mi_lineage = mutual_info(badges, roots) if badges else 0.0
    # 各族徽章：载体 mean strength + 人口 mean response（终局）
    per_badge = {}
    for b in range(max(k, 1)):
        idx = [i for i, x in enumerate(badges) if x == b]
        if not idx:
            continue
        resp_mean = []
        for i in idx:
            rs = inds[i].response_state
            resp_mean.append(float(rs[b]) if rs is not None else 0.0)
        per_badge[b] = {
            "n": len(idx),
            "carrier_strength_mean": sum(strengths[i] for i in idx) / len(idx),
            "response_self_mean": sum(resp_mean) / len(resp_mean),
        }
    m["c3_detail"] = {
        "badge_histogram_final": badge_hist,
        "mi_badge_strength": mi_strength,
        "mi_badge_lineage": mi_lineage,
        "per_badge": per_badge,
        "n_final": len(inds),
    }
    return m


def c3(seeds: List[int], ticks: int, out_dir: Path, ckpt_dir=None,
       resume: Optional[str] = None, ckpt_every: int = 500) -> Dict:
    """B14.3 C3 v2 首轮预注册跑批：4 臂 × seeds × ticks + 逐条判据。"""
    from biosphere.criteria import evaluate_ecological_gate
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict = {"ticks": ticks, "seeds": seeds, "arms": {}, "gates": {}}
    for arm, spec in C3_ARMS.items():
        rows = []
        for seed in seeds:
            m = _c3_arm(seed, ticks, arm, ckpt_dir=ckpt_dir,
                        ckpt_every=ckpt_every)
            det = m["c3_detail"]
            c3m = m["c3"]
            ar = c3m["attacked_rate"]
            rows.append({
                "seed": seed, "desc": spec["desc"],
                "final_alive": m["population"]["final_alive"],
                "alive_series": m["population"]["alive_series"],
                "mi_badge_strength": det["mi_badge_strength"],
                "mi_badge_lineage": det["mi_badge_lineage"],
                "badge_hist": det["badge_histogram_final"],
                "attacked_all": ar["all"],
                "attacked_emit": ar["emitting"],
                "n_attacks": ar["n_attacks"],
            })
            out_path = out_dir / f"b14_c3_{arm}_s{seed}_t{ticks}.json"
            with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(m, fh, sort_keys=True, indent=1)
                fh.write("\n")
        report["arms"][arm] = rows
        report["gates"][arm] = evaluate_ecological_gate(
            {r["seed"]: r["alive_series"] for r in rows})

    # 判据（只对 full 臂判定；负对照臂作对照读数）
    full = report["arms"]["c3_full"]
    crit: Dict = {"mi_positive_seeds": 0, "response_alignment_seeds": 0,
                  "deterrence_seeds": 0, "n_seeds": len(full)}
    for r in full:
        if r["mi_badge_strength"] > 0.0:
            crit["mi_positive_seeds"] += 1
        ae = r["attacked_emit"]
        if ae["high"] < ae["low"]:
            crit["deterrence_seeds"] += 1
    crit["mi_badge_strength"] = {
        "mean": sum(r["mi_badge_strength"] for r in full) / len(full),
        "per_seed": {r["seed"]: r["mi_badge_strength"] for r in full}}
    crit["mi_badge_lineage"] = {
        "mean": sum(r["mi_badge_lineage"] for r in full) / len(full),
        "per_seed": {r["seed"]: r["mi_badge_lineage"] for r in full}}
    crit["deterrence_per_seed"] = {
        r["seed"]: {"high": r["attacked_emit"]["high"],
                    "low": r["attacked_emit"]["low"]} for r in full}
    report["criterion"] = crit

    print(f"=== B14.3 C3 v2 first round: {ticks} ticks x {len(seeds)} seeds "
          f"x {len(C3_ARMS)} arms ===")
    for arm, rows in report["arms"].items():
        gate = report["gates"][arm]
        print(f"[{arm}] {gate['n_passed']}/{gate['n_total']} "
              f"{gate['verdict']}")
        for r in rows:
            print("  s%d alive=%3d MI(str)=%.3f MI(lin)=%.3f "
                  "atk(all)=%s atk(emit)=%s" % (
                      r["seed"], r["final_alive"], r["mi_badge_strength"],
                      r["mi_badge_lineage"],
                      {k: round(v, 4) for k, v in r["attacked_all"].items()},
                      {k: round(v, 4) for k, v in r["attacked_emit"].items()}))
    print()
    print(f"[criterion] MI(badge,strength)>0: "
          f"{crit['mi_positive_seeds']}/{crit['n_seeds']}；"
          f"MI(badge,lineage)>0 均值 {crit['mi_badge_lineage']['mean']:.3f}；"
          f"威慑方向（强者发射被攻击率<弱者）: "
          f"{crit['deterrence_seeds']}/{crit['n_seeds']}")
    out_path = out_dir / f"b14_c3_summary_t{ticks}.json"
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, sort_keys=True, indent=1)
        fh.write("\n")
    print(f"summary: {out_path}")
    return report


#: B15.1 oracle 臂定义（诊断正对照；自然臂 = B14 首轮同参数复跑对照）
B15_ORACLE_ARMS: Dict[str, Dict] = {
    "oracle_response": {
        "desc": "oracle-response：response 按真实 strength 覆写（完美回避强徽章）+ 发射保底",
        "c3": {"oracle_response": True}},
    "oracle_emitter": {
        "desc": "oracle-emitter：只允许 strength≥0.5 者发射（弱者 SIGNAL 硬压）",
        "c3": {"oracle_emitter": True}},
    "natural": {
        "desc": "自然臂（与 B14 c3_full 同参数；同 tick 口径对照）",
        "c3": {}},
}


def _b15_arm(seed: int, ticks: int, arm: str, ckpt_dir=None,
             ckpt_every: int = 500) -> Dict:
    from biosphere.c3 import C3Config
    spec = B15_ORACLE_ARMS[arm]
    cfg = BioMVPConfig(ticks=ticks, seed=seed,
                       c3=C3Config(enabled=True, **spec["c3"]))
    sim = BiosphereSimulator(cfg)
    log = sim.run(ckpt_dir=ckpt_dir, ckpt_every=ckpt_every,
                  tag=f"b15_{arm}_s{seed}")
    return _summarize(sim, log, seed, ticks, threat_sensing=True)


def b15(seeds: List[int], ticks: int, out_dir: Path, ckpt_dir=None,
        resume: Optional[str] = None, ckpt_every: int = 500) -> Dict:
    """B15.1 oracle 正对照：3 臂 × seeds × ticks + 判据③ oracle 式。

    判据③（oracle 式）：发射者被攻击率 < 非发射者被攻击率（by_emitting）。
    oracle-response 把“回避”设为完美 ⇒ 若仍无不对称 ⇒ 因果链断在攻击端；
    oracle-emitter 把“发射”与“强者”绑定 ⇒ 检验发射侧是否影响被攻击。
    """
    from biosphere.criteria import evaluate_ecological_gate
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict = {"ticks": ticks, "seeds": seeds, "arms": {}, "gates": {}}
    for arm, spec in B15_ORACLE_ARMS.items():
        rows = []
        for seed in seeds:
            m = _b15_arm(seed, ticks, arm, ckpt_dir=ckpt_dir,
                         ckpt_every=ckpt_every)
            be = m["c3"]["by_emitting"]
            ar = m["c3"]["attacked_rate"]
            rows.append({
                "seed": seed,
                "final_alive": m["population"]["final_alive"],
                "alive_series": m["population"]["alive_series"],
                "attacked_emit": be["attacked_rate_emitting"],
                "attacked_non_emit": be["attacked_rate_non_emitting"],
                "n_attacks_emit": be["n_attacks_emitting"],
                "n_attacks_non_emit": be["n_attacks_non_emitting"],
                "exp_emit": be["exposure_emitting"],
                "exp_non_emit": be["exposure_non_emitting"],
                "attacked_rate_all": ar["all"],
                "attacked_rate_emitting_terc": ar["emitting"],
            })
            out_path = out_dir / f"b15_{arm}_s{seed}_t{ticks}.json"
            with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(m, fh, sort_keys=True, indent=1)
                fh.write("\n")
        report["arms"][arm] = rows
        report["gates"][arm] = evaluate_ecological_gate(
            {r["seed"]: r["alive_series"] for r in rows})

    # 判据③ oracle 式：逐 seed 看发射者被攻击率是否低于非发射者
    crit: Dict = {}
    for arm in B15_ORACLE_ARMS:
        ok = [r["seed"] for r in report["arms"][arm]
              if r["attacked_emit"] < r["attacked_non_emit"]]
        crit[arm] = {
            "n_direction_ok": len(ok),
            "n_seeds": len(seeds),
            "seeds_ok": ok,
            "mean_diff": sum(r["attacked_emit"] - r["attacked_non_emit"]
                             for r in report["arms"][arm]) / max(len(seeds), 1),
        }
    report["criterion3_oracle"] = crit

    print(f"=== B15.1 oracle control: {ticks} ticks x {len(seeds)} seeds "
          f"x {len(B15_ORACLE_ARMS)} arms ===")
    print(f"{'arm':<18}{'seed':<6}{'alive':<7}{'gate':<6}"
          f"{'atkEmit':<10}{'atkNon':<10}{'nE/nN':<12}{'diff':<10}")
    for arm, rows in report["arms"].items():
        gate = report["gates"][arm]["per_seed"]
        for r in rows:
            g = "pass" if gate[r["seed"]]["passed"] else "FAIL"
            print(f"{arm:<18}{r['seed']:<6}{r['final_alive']:<7}{g:<6}"
                  f"{r['attacked_emit']:<10.5f}{r['attacked_non_emit']:<10.5f}"
                  f"{str(r['n_attacks_emit']) + '/' + str(r['n_attacks_non_emit']):<12}"
                  f"{r['attacked_emit'] - r['attacked_non_emit']:<+10.5f}")
    print()
    for arm, g in report["gates"].items():
        print(f"[gate] {arm:<18} {g['n_passed']}/{g['n_total']} "
              f"{g['verdict']}")
    print()
    for arm, c in crit.items():
        print(f"[criterion3-oracle] {arm:<18} 发射者被攻击率<非发射者："
              f"{c['n_direction_ok']}/{c['n_seeds']} "
              f"（均值差 {c['mean_diff']:+.5f}）")
    out_path = out_dir / f"b15_oracle_summary_t{ticks}.json"
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, sort_keys=True, indent=1)
        fh.write("\n")
    print(f"summary: {out_path}")
    return report


#: B15.2 功效分析：样本量档（pop, rows, cols）——世界随 N 扩，密度近似恒定
B15_POWER_LEVELS: List[Dict] = [
    {"n": 100, "rows": 32, "cols": 64},
    {"n": 500, "rows": 32, "cols": 64},
    {"n": 2000, "rows": 64, "cols": 128},
]


def _c3_arm_sized(seed: int, ticks: int, arm: str, rows: int, cols: int,
                  init_pop: int, ckpt_dir=None, ckpt_every: int = 500) -> Dict:
    """C3 臂（世界尺寸/初始种群参数化；B15.2 功效分析用）。"""
    from biosphere.c3 import C3Config
    spec = C3_ARMS[arm]
    cfg = BioMVPConfig(ticks=ticks, seed=seed, rows=rows, cols=cols,
                       init_pop=init_pop,
                       c3=C3Config(enabled=True, **spec["c3"]))
    sim = BiosphereSimulator(cfg)
    log = sim.run(ckpt_dir=ckpt_dir, ckpt_every=ckpt_every,
                  tag=f"b15p_{arm}_n{init_pop}_s{seed}")
    m = _summarize(sim, log, seed, ticks, threat_sensing=True)
    from biosphere.crossval import mutual_info
    from biosphere.develop import develop
    inds = sim.population.individuals
    badges, strengths, roots = [], [], []
    child_set = {c for _p, c, _cell in log.births}
    for ind in inds:
        c = develop(ind.genome)
        badges.append(c.badge)
        strengths.append(c.strength)
        roots.append(0 if ind.parent < 0 or ind.parent not in child_set
                     else 1)
    k = int(max(badges)) + 1 if badges else 1
    badge_hist = {b: badges.count(b) for b in range(max(k, 1))}
    mi_strength = mutual_info(badges, [int(s * 4) for s in strengths]) \
        if badges else 0.0
    per_badge = {}
    for b in range(max(k, 1)):
        idx = [i for i, x in enumerate(badges) if x == b]
        if not idx:
            continue
        resp_mean = []
        for i in idx:
            rs = inds[i].response_state
            resp_mean.append(float(rs[b]) if rs is not None else 0.0)
        per_badge[b] = {
            "n": len(idx),
            "carrier_strength_mean": sum(strengths[i] for i in idx) / len(idx),
            "response_self_mean": sum(resp_mean) / len(resp_mean),
        }
    m["c3_detail"] = {
        "badge_histogram_final": badge_hist,
        "mi_badge_strength": mi_strength,
        "per_badge": per_badge,
        "n_final": len(inds),
    }
    return m


def _effect_concordance(per_badge: Dict) -> float:
    """判据②效应量：徽章对 (载体强度, 自身反应) 的 Kendall 一致度 ∈ [-1,1]。

    0 = 无对齐（随机）；+1 = 强者徽章被正反应（亲近方向）；
    −1 = 强者徽章被负反应（回避/威慑方向）。K<2 时不可定义 ⇒ None。
    """
    pts = [(d["carrier_strength_mean"], d["response_self_mean"])
           for d in per_badge.values()]
    if len(pts) < 2:
        return None
    conc = disc = 0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            ds = pts[i][0] - pts[j][0]
            dr = pts[i][1] - pts[j][1]
            if ds == 0.0 or dr == 0.0:
                continue
            if (ds > 0) == (dr > 0):
                conc += 1
            else:
                disc += 1
    if conc + disc == 0:
        return None
    return (conc - disc) / (conc + disc)


def _effect_deterrence(m: Dict) -> float:
    """判据③效应量：强者发射被攻击率 − 弱者发射被攻击率（负 = 威慑方向）。"""
    em = m["c3"]["attacked_rate"]["emitting"]
    return em["high"] - em["low"]


def _bootstrap_ci(values: List[float], n_boot: int = 10000,
                  alpha: float = 0.05, seed: int = 7) -> Dict:
    """per-seed 效应量的均值 + bootstrap 95% CI（seed 为重抽样单元）。"""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=np.float64)
    n = arr.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "lo": float(np.quantile(means, alpha / 2)),
        "hi": float(np.quantile(means, 1 - alpha / 2)),
        "n": n,
    }


def b15_power(seeds: List[int], ticks: int, out_dir: Path, ckpt_dir=None,
              ckpt_every: int = 500) -> Dict:
    """B15.2 功效分析：N ∈ {100,500,2000} × {full, no_strength} × seeds。

    产出判据②（response 符号一致度）与判据③（强弱发射被攻击率差）的
    效应量 vs N 曲线（均值 ± bootstrap 95% CI）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict = {"ticks": ticks, "seeds": seeds, "levels": {},
                    "note": "效应量=per-seed；CI=对 seed 重抽样 10000 次"}
    for lv in B15_POWER_LEVELS:
        n = lv["n"]
        for arm in ("c3_full", "c3_no_strength"):
            e2_list: List[float] = []
            e3_list: List[float] = []
            alive_list = []
            for seed in seeds:
                m = _c3_arm_sized(seed, ticks, arm, lv["rows"], lv["cols"],
                                  n, ckpt_dir=ckpt_dir, ckpt_every=ckpt_every)
                e2 = _effect_concordance(m["c3_detail"]["per_badge"])
                e3 = _effect_deterrence(m)
                e2_list.append(e2 if e2 is not None else float("nan"))
                e3_list.append(e3)
                alive_list.append(m["population"]["final_alive"])
                out_path = out_dir / f"b15p_{arm}_n{n}_s{seed}_t{ticks}.json"
                with open(out_path, "w", encoding="utf-8",
                          newline="\n") as fh:
                    json.dump(m, fh, sort_keys=True, indent=1)
                    fh.write("\n")
            key = f"n{n}"
            report["levels"].setdefault(key, {})[arm] = {
                "n": n,
                "effect2_concordance": _bootstrap_ci(e2_list),
                "effect3_deterrence": _bootstrap_ci(e3_list),
                "final_alive": alive_list,
                "per_seed_effect2": e2_list,
                "per_seed_effect3": e3_list,
            }
            print(f"[n={n:<5} {arm:<14}] eff2={_fmt_ci(report['levels'][key][arm]['effect2_concordance'])} "
                  f"eff3={_fmt_ci(report['levels'][key][arm]['effect3_deterrence'])} "
                  f"alive={alive_list}")
    print()
    print("eff2=判据②符号一致度（0=无对齐；负=强者徽章被回避=威慑方向）")
    print("eff3=判据③强者发射被攻击率−弱者发射被攻击率（负=威慑方向）")
    print("CI=对 seed 的 bootstrap 95% 区间（10000 次）")
    out_path = out_dir / f"b15_power_summary_t{ticks}.json"
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, sort_keys=True, indent=1)
        fh.write("\n")
    print(f"summary: {out_path}")
    return report


def _fmt_ci(ci: Dict) -> str:
    if ci["mean"] is None:
        return "n/a"
    return f"{ci['mean']:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}] (n={ci['n']})"


def b15_rejudge(seeds: List[int], ticks: int, out_dir: Path,
                ckpt_dir: Path) -> Dict:
    """B15.3 用新判据重判 B14 已有数据（恢复终局 checkpoint，不重跑）。

    逐臂 × seed：load t<ticks> checkpoint（= B14 那场模拟的终局，逐位一致）
    ⇒ 个体级记录 ⇒ 新判据：谱系条件 MI / 谱系内置换 z·p / K_eff；
    并与 B14 旧判据读数（汇总 JSON 的 c3_detail）并排输出。
    """
    from biosphere.criteria import (
        badge_keff, individual_records, lineage_conditional_mi,
        lineage_label_permutation,
    )
    from biosphere.checkpoint import load_checkpoint
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict = {"ticks": ticks, "seeds": seeds, "arms": {},
                    "note": "恢复 B14 终局 checkpoint（不重跑）；新判据=谱系条件"}
    for arm in ("c3_full", "c3_no_strength", "c3_badge_reshuffle",
                "c3_memory"):
        rows = []
        for seed in seeds:
            ckpt = ckpt_dir / f"ckpt_b14_{arm}_s{seed}_t{ticks}.pkl"
            old_path = out_dir / f"b14_c3_{arm}_s{seed}_t3000.json"
            old = {}
            if old_path.exists():
                old = json.load(open(old_path, encoding="utf-8")).get(
                    "c3_detail", {})
            row = {"seed": seed, "ckpt_exists": ckpt.exists()}
            if ckpt.exists():
                sim = load_checkpoint(ckpt)
                recs = individual_records(sim)
                cond = lineage_conditional_mi(recs)
                perm = lineage_label_permutation(recs)
                hist: Dict[int, int] = {}
                for r in recs:
                    hist[r["badge"]] = hist.get(r["badge"], 0) + 1
                row.update({
                    "n_final": len(recs),
                    "k_eff": badge_keff(hist),
                    "badge_hist": hist,
                    "mi_lineage_weighted": cond["mi_weighted"],
                    "n_roots_estimated": cond["n_roots_estimated"],
                    "perm_observed": perm["observed"],
                    "perm_z": perm["z"],
                    "perm_p": perm["p_one_sided"],
                    "old_mi_global": old.get("mi_badge_strength"),
                    "old_badge_hist": old.get("badge_histogram_final"),
                })
            rows.append(row)
            print(f"[{arm} s{seed}] ckpt={'Y' if row['ckpt_exists'] else 'N'} "
                  f"n={row.get('n_final', 0)} K_eff={row.get('k_eff', 0):.2f} "
                  f"MI_lin={row.get('mi_lineage_weighted', 0):.3f} "
                  f"perm_z={row.get('perm_z', 0):+.2f} "
                  f"p={row.get('perm_p', 1):.3f} "
                  f"(旧 MI 全局={row.get('old_mi_global')})")
        report["arms"][arm] = rows

    # 新判据汇总：逐臂报 K_eff 均值 / 置换显著 seed 数
    summary: Dict = {}
    for arm, rows in report["arms"].items():
        ok_rows = [r for r in rows if r["ckpt_exists"]]
        summary[arm] = {
            "k_eff_mean": (sum(r["k_eff"] for r in ok_rows) / len(ok_rows))
            if ok_rows else None,
            "perm_significant_seeds": sum(
                1 for r in ok_rows if r["perm_p"] < 0.05),
            "mi_lineage_mean": (sum(r["mi_lineage_weighted"]
                                    for r in ok_rows) / len(ok_rows))
            if ok_rows else None,
            "n_seeds_with_ckpt": len(ok_rows),
        }
    report["summary"] = summary
    print()
    for arm, s in summary.items():
        print(f"[summary] {arm:<22} K_eff均值={s['k_eff_mean']} "
              f"谱系内置换显著(p<0.05)={s['perm_significant_seeds']}"
              f"/{s['n_seeds_with_ckpt']} "
              f"谱系条件MI均值={s['mi_lineage_mean']}")
    out_path = out_dir / f"b15_rejudge_t{ticks}.json"
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, sort_keys=True, indent=1)
        fh.write("\n")
    print(f"summary: {out_path}")
    return report


# ── B16 捕食-猎物军备竞赛（owner 2026-10-19 定线；预注册 §13.16） ──────────

#: B16.1 oracle 正对照臂（**不跑演化**：性状钉死，只测收益差）
B16_ORACLE_ARMS: Dict[str, Dict] = {
    "b16_oracle_defense": {
        "desc": "oracle-defense：半数 defense=1 / 半数 =0（uid 奇偶）",
        "predation": {"oracle_defense": True}, "resolve": {}},
    "b16_oracle_attack": {
        "desc": "oracle-attack：半数 attack_drive=1 / 半数 =0（uid 奇偶）",
        "predation": {"oracle_attack": True}, "resolve": {}},
    "b16_natural": {
        "desc": "自然臂（同参数、性状不钉；oracle 的对照）",
        "predation": {}, "resolve": {}},
}

#: B16.3 军备竞赛臂（两个负对照：去 loot / 去 defense 收益）
B16_RACE_ARMS: Dict[str, Dict] = {
    "b16_full": {
        "desc": "完整档（loot 开 + 减伤开 + 反击开）",
        "predation": {"init_trait_spread": 2.0}, "resolve": {}},
    "b16_no_loot": {
        "desc": "负对照：掠食分成 = 0（攻击无能量收益）",
        "predation": {"init_trait_spread": 2.0},
        "resolve": {"attack_loot_frac": 0.0}},
    "b16_no_defense": {
        "desc": "负对照：defense 零收益（减伤关 + 反击关）",
        "predation": {"init_trait_spread": 2.0},
        "resolve": {"defense_mit": 0.0, "counter_defense_scale": 0.0}},
    "b16_no_counter": {
        "desc": "观察臂：只关反击（保留减伤；看减伤 alone 够不够）",
        "predation": {"init_trait_spread": 2.0},
        "resolve": {"counter_defense_scale": 0.0}},
}

B16_ARMS: Dict[str, Dict] = {**B16_ORACLE_ARMS, **B16_RACE_ARMS}

#: B17.2 军备竞赛二轮（命中即转移；spread 固定 2.0，B17.1 §5.2 选档）
B17_RACE_ARMS: Dict[str, Dict] = {
    "b17_full": {
        "desc": "主臂：contact_loot 开（命中存活即转移）",
        "predation": {"init_trait_spread": 2.0},
        "resolve": {"contact_loot": True}},
    "b17_off": {
        "desc": "负对照 1：只关 contact_loot（同 spread 同参数）",
        "predation": {"init_trait_spread": 2.0},
        "resolve": {"contact_loot": False}},
    "b17_no_loot": {
        "desc": "负对照 2：contact_loot 开但 loot_frac=0（无任何转移）",
        "predation": {"init_trait_spread": 2.0},
        "resolve": {"contact_loot": True, "attack_loot_frac": 0.0}},
    "b17_no_defense": {
        "desc": "负对照 3：defense 零收益（减伤关 + 反击关）",
        "predation": {"init_trait_spread": 2.0},
        "resolve": {"contact_loot": True, "defense_mit": 0.0,
                    "counter_defense_scale": 0.0}},
}

#: B17.2 oracle（contact_loot 开；oracle 只看因果链，不跑演化）
B17_ORACLE_ARMS: Dict[str, Dict] = {
    "b17_oracle_defense": {
        "desc": "oracle-defense（contact_loot 开）：半数 defense=1/0",
        "predation": {"oracle_defense": True},
        "resolve": {"contact_loot": True}},
    "b17_oracle_attack": {
        "desc": "oracle-attack（contact_loot 开）：半数 attack_drive=1/0",
        "predation": {"oracle_attack": True},
        "resolve": {"contact_loot": True}},
    "b17_natural": {
        "desc": "自然臂（contact_loot 开；oracle 的对照）",
        "predation": {}, "resolve": {"contact_loot": True}},
}

B17_ARMS: Dict[str, Dict] = {**B17_ORACLE_ARMS, **B17_RACE_ARMS}
#: 全部臂（build_b16_config 查表用；B16 历史臂不变）
ALL_ARMS: Dict[str, Dict] = {**B16_ARMS, **B17_ARMS}


def build_b16_config(arm: str, seed: int, ticks: int,
                     spread: Optional[float] = None) -> BioMVPConfig:
    """臂名 → BioMVPConfig（ predation 恒开：性状帧是判据数据源）。

    Args:
        spread: 覆盖臂表里的 ``init_trait_spread``（世界族参数；None = 用臂表值）。
    """
    from biosphere.resolve import ResolveConfig
    spec = ALL_ARMS[arm]
    pred = dict(spec["predation"])
    if spread is not None:
        pred["init_trait_spread"] = float(spread)
    return BioMVPConfig(
        ticks=ticks, seed=seed,
        predation=PredationConfig(enabled=True, **pred),
        resolve=ResolveConfig(**spec["resolve"]),
    )


def _b16_row(sim, log, arm: str, seed: int, ticks: int,
             spread: Optional[float] = None) -> Dict:
    """一次跑批 → 一行 B16 读数（纯 Python；可 pickle/json）。"""
    m = metrics_json(log, sim.world, seed=seed, ticks=ticks)
    pred = predation_json(log, sim.world)
    pop = m["population"]
    row: Dict = {
        "schema": "biosphere-b16-v0.1",
        "arm": arm, "seed": int(seed), "ticks": int(ticks),
        "desc": ALL_ARMS[arm]["desc"],
        "init_trait_spread": (ALL_ARMS[arm]["predation"]
                              .get("init_trait_spread", 0.0)
                              if spread is None else float(spread)),
        "population": {
            "final_alive": int(pop["final_alive"]),
            "peak_alive": int(pop["peak_alive"]),
            "total_born": int(pop["total_born"]),
            "total_died": int(pop["total_died"]),
            "death_causes": {str(k): int(v)
                             for k, v in pop["death_causes"].items()},
            "alive_series": [int(v) for v in pop["alive_series"]],
        },
        "predation": {k: v for k, v in pred.items() if k != "_per_uid"},
        "digest": run_digest(log),
    }
    contrast = _oracle_contrast(pred, arm)
    if contrast is not None:
        row["oracle"] = contrast
    row["_per_uid"] = pred.get("_per_uid", {})
    return row


def _oracle_contrast(pred: Dict, arm: str) -> Optional[Dict]:
    """oracle 臂的高/低组对比（B16.1 判据数据源）。

    - defense 臂：看**死于攻击的比例**（高防御应更低）与被攻击率；
    - attack 臂：看**攻击净能量收益**（loot − 成本 − 反击；高攻应更高）。
    """
    if not pred.get("enabled"):
        return None
    if "oracle_defense" in arm:
        mode = "defense"
    elif "oracle_attack" in arm:
        mode = "attack"
    else:
        return None
    per = pred.get("_per_uid", {})
    exposure = {int(u): int(v) for u, v in per.get("exposure", {}).items()}
    attacked = {int(u): int(v) for u, v in per.get("attacked", {}).items()}
    killed = {int(u): int(v) for u, v in per.get("killed", {}).items()}
    loot = {int(u): float(v) for u, v in per.get("loot", {}).items()}
    net = {int(u): float(v) for u, v in per.get("net_attack", {}).items()}
    born = {int(u): int(v) for u, v in per.get("births", {}).items()}
    groups: Dict[str, Dict] = {}
    for g, name in ((1, "high"), (0, "low")):
        uids = [u for u in exposure if oracle_group(u) == g]
        if not uids:
            groups[name] = {"n": 0}
            continue
        n = len(uids)
        exp = sum(exposure[u] for u in uids)
        n_kill = int(sum(killed.get(u, 0) for u in uids))
        n_atk = int(sum(attacked.get(u, 0) for u in uids))
        loot_sum = float(sum(loot.get(u, 0.0) for u in uids))
        net_sum = float(sum(net.get(u, 0.0) for u in uids))
        groups[name] = {
            "n_individuals": n,
            "exposure_ticks": int(exp),
            "n_killed": n_kill,
            # 主读数：**每单位存活时间**的死亡危害（killed/个体·tick）——
            # 不用 killed/个体：活得久的组分母相同但分子机会更多，会伪影
            "kill_hazard": (n_kill / exp) if exp else 0.0,
            "killed_per_individual": n_kill / n,
            "attack_hazard": (n_atk / exp) if exp else 0.0,
            "mean_lifespan": (exp / n),
            "loot_total": loot_sum,
            "loot_per_individual": loot_sum / n,
            "net_attack_per_individual": net_sum / n,
            "net_attack_per_exposure": (net_sum / exp) if exp else 0.0,
            "births_per_individual": (sum(born.get(u, 0) for u in uids) / n),
        }
    hi, lo = groups["high"], groups["low"]
    if not hi.get("n_individuals") or not lo.get("n_individuals"):
        return {"mode": mode, "groups": groups, "diff": None}
    key = ("kill_hazard" if mode == "defense"
           else "net_attack_per_individual")
    return {
        "mode": mode,
        "key_stat": key,
        "expect": "high<low（防御降低被掠食危害）" if mode == "defense"
                  else "high>low（攻击拿到正净收益）",
        "groups": groups,
        "diff_high_minus_low": float(hi[key] - lo[key]),
    }


def run_job(job: "JobSpec") -> Dict:
    """并行/串行共用的作业体（B16；parallel._run_one 调它）。"""
    spread = job.extra.get("spread") if job.extra else None
    cfg = build_b16_config(job.arm, job.seed, job.ticks, spread=spread)
    sim = BiosphereSimulator(cfg)
    log = sim.run(ckpt_dir=(job.ckpt_dir or None),
                  ckpt_every=(job.ckpt_every or 500),
                  tag=(job.tag or job.label()))
    return _b16_row(sim, log, job.arm, job.seed, job.ticks, spread=spread)


def _run_jobs(jobs: List["JobSpec"], workers: int) -> Tuple[List[Dict], Dict]:
    """按 workers 选择串行/并行跑一批作业，返回 (结果行, 并行元信息)。"""
    if workers <= 1:
        t0 = time.time()
        res = run_serial(jobs)
        meta = {"workers": 1, "wall_s": time.time() - t0, "mode": "serial"}
        return [r["row"] for r in res], meta
    t0 = time.time()
    res = run_parallel(jobs, workers=workers)
    meta = {"workers": workers, "wall_s": time.time() - t0,
            "mode": "parallel",
            "n_failed": sum(1 for r in res if not r["ok"])}
    bad = [r for r in res if not r["ok"]]
    if bad:
        print("❌ 并行作业失败：", [b["error"] for b in bad][:3])
    return [r["row"] for r in res], meta


def b16_oracle(seeds: List[int], ticks: int, out_dir: Path,
               workers: int = 1, ckpt_dir=None,
               ckpt_every: int = 500,
               arms: Optional[Dict[str, Dict]] = None,
               label: str = "b16") -> Dict:
    """B16.1/B17.2 oracle 正对照：3 臂 × seeds × ticks（不跑演化）。"""
    from biosphere.criteria import evaluate_ecological_gate
    arm_table = B16_ORACLE_ARMS if arms is None else arms
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [JobSpec(arm=arm, seed=s, ticks=ticks,
                    ckpt_dir=str(ckpt_dir) if ckpt_dir else "",
                    ckpt_every=ckpt_every)
            for arm in arm_table for s in seeds]
    rows, meta = _run_jobs(jobs, workers)
    by_arm: Dict[str, List[Dict]] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    report: Dict = {"schema": f"biosphere-{label}-oracle-v0.1", "ticks": ticks,
                    "seeds": seeds, "parallel": meta, "arms": {},
                    "gates": {}, "criterion": {}}
    for arm, rs in by_arm.items():
        report["arms"][arm] = rs
        report["gates"][arm] = evaluate_ecological_gate(
            {r["seed"]: r["population"]["alive_series"] for r in rs})
    # 判读：逐 seed 方向 + bootstrap CI（对 seed 重抽样）
    for arm in arm_table:
        if not (arm.endswith("oracle_defense") or arm.endswith("oracle_attack")):
            continue
        rs = by_arm.get(arm, [])
        diffs = [r["oracle"]["diff_high_minus_low"] for r in rs
                 if r.get("oracle") and r["oracle"].get("diff_high_minus_low")
                 is not None]
        if not diffs:
            continue
        if arm.endswith("defense"):
            ok = [r["seed"] for r in rs
                  if r["oracle"]["diff_high_minus_low"] < 0.0]
        else:
            ok = [r["seed"] for r in rs
                  if r["oracle"]["diff_high_minus_low"] > 0.0]
        report["criterion"][arm] = {
            "expect": rs[0]["oracle"]["expect"],
            "n_direction_ok": len(ok), "n_seeds": len(rs), "seeds_ok": ok,
            "n_high_net_positive": sum(
                1 for r in rs
                if r["oracle"]["groups"]["high"].get(
                    "net_attack_per_individual", 0.0) > 0.0),
            "mean_diff": _bootstrap_ci(diffs),
            "per_seed_diff": {r["seed"]: r["oracle"]["diff_high_minus_low"]
                              for r in rs},
        }
    out_path = out_dir / f"{label}_oracle_summary_t{ticks}.json"
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, sort_keys=True, indent=1)
        fh.write("\n")
    title = "B16.1" if label == "b16" else "B17.2"
    print(f"=== {title} oracle 正对照：{ticks} ticks × {len(seeds)} seeds "
          f"× {len(arm_table)} 臂（{meta['mode']} "
          f"{meta.get('workers', 1)} 进程，{meta['wall_s']:.1f}s）===")
    for arm in arm_table:
        gate = report["gates"][arm]
        print(f"[{arm}] 生态门 {gate['n_passed']}/{gate['n_total']} "
              f"{gate['verdict']}")
        for r in report["arms"][arm]:
            oc = r.get("oracle") or {}
            if oc:
                hi, lo = oc["groups"]["high"], oc["groups"]["low"]
                print("  s%-3d alive=%-4d hazard hi=%.5f lo=%.5f "
                      "diff=%+.5f | life %.0f/%.0f | atkhz %.4f/%.4f | "
                      "net/ind hi=%+.3f lo=%+.3f" % (
                          r["seed"], r["population"]["final_alive"],
                          hi.get("kill_hazard", 0.0),
                          lo.get("kill_hazard", 0.0),
                          oc["diff_high_minus_low"],
                          hi.get("mean_lifespan", 0.0),
                          lo.get("mean_lifespan", 0.0),
                          hi.get("attack_hazard", 0.0),
                          lo.get("attack_hazard", 0.0),
                          hi.get("net_attack_per_individual", 0.0),
                          lo.get("net_attack_per_individual", 0.0)))
            else:
                print("  s%-3d alive=%-4d（自然臂，无 oracle 分组）"
                      % (r["seed"], r["population"]["final_alive"]))
    print()
    for arm, c in report["criterion"].items():
        ci = c["mean_diff"]
        print(f"[{arm}] {c['expect']}：方向一致 {c['n_direction_ok']}/"
              f"{c['n_seeds']}；均值差 {ci['mean']:+.4f} "
              f"CI[{ci['lo']:+.4f},{ci['hi']:+.4f}]")
    print(f"summary: {out_path}")
    return report


def b17_oracle(seeds: List[int], ticks: int, out_dir: Path,
               workers: int = 1, ckpt_dir=None,
               ckpt_every: int = 500) -> Dict:
    """B17.2 oracle 前置门（contact_loot 开；不通过即停，不跑判据）。"""
    return b16_oracle(seeds, ticks, out_dir, workers=workers,
                      ckpt_dir=ckpt_dir, ckpt_every=ckpt_every,
                      arms=B17_ORACLE_ARMS, label="b17")


def b16(seeds: List[int], ticks: int, out_dir: Path, workers: int = 1,
        ckpt_dir=None, ckpt_every: int = 500,
        spread: Optional[float] = None,
        arms: Optional[Dict[str, Dict]] = None,
        label: str = "b16") -> Dict:
    """B16.3/B17.2 军备竞赛跑批：臂 × seeds × ticks + 五条判据逐条判定。

    Args:
        spread: 覆盖臂表的 ``init_trait_spread``（世界族参数；None = 用臂表值
            = 预注册口径 2.0）。**改它 = 脱离预注册口径**，只用于可行性探测。
        arms: 臂表（None = B16_RACE_ARMS；B17.2 传 B17_RACE_ARMS）。
        label: 报告前缀（b16/b17；同时决定输出文件名与判据臂名前缀）。
    """
    from biosphere.criteria import evaluate_ecological_gate
    arm_table = B16_RACE_ARMS if arms is None else arms
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = {} if spread is None else {"spread": float(spread)}
    jobs = [JobSpec(arm=arm, seed=s, ticks=ticks,
                    ckpt_dir=str(ckpt_dir) if ckpt_dir else "",
                    ckpt_every=ckpt_every, extra=extra)
            for arm in arm_table for s in seeds]
    rows, meta = _run_jobs(jobs, workers)
    by_arm: Dict[str, List[Dict]] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    report: Dict = {"schema": f"biosphere-{label}-race-v0.1", "ticks": ticks,
                    "seeds": seeds, "parallel": meta, "arms": {},
                    "gates": {}, "criterion": {},
                    "init_trait_spread": (2.0 if spread is None
                                          else float(spread)),
                    "note": ("spread=2.0 为预注册口径；其他值仅作可行性探测，"
                             "不进判据")}
    if label != "b16":
        report["note"] = ("B17.2：命中即转移（contact_loot）；"
                          "spread=2.0 由 B17.1 §5.2 规则选出")
    for arm, rs in by_arm.items():
        report["arms"][arm] = rs
        report["gates"][arm] = evaluate_ecological_gate(
            {r["seed"]: r["population"]["alive_series"] for r in rs})

    def _traj_end(r: Dict, key: str) -> Tuple[float, float]:
        tr = r["predation"]["trajectory"]
        return float(tr[0][key]), float(tr[-1][key])

    full = by_arm.get(f"{label}_full", [])
    no_loot = by_arm.get(f"{label}_no_loot", [])
    no_def = by_arm.get(f"{label}_no_defense", [])
    off = by_arm.get(f"{label}_off", [])
    d0 = {r["seed"]: _traj_end(r, "defense_mean") for r in full}
    a0 = {r["seed"]: _traj_end(r, "attack_mean") for r in full}
    crit: Dict = {}
    crit["①defense 上移"] = {
        "n_ok": sum(1 for s in d0 if d0[s][1] > d0[s][0]),
        "n_seeds": len(d0),
        "per_seed": {s: {"t0": d0[s][0], "t_end": d0[s][1],
                         "delta": d0[s][1] - d0[s][0]} for s in d0}}
    crit["②attack 上移"] = {
        "n_ok": sum(1 for s in a0 if a0[s][1] > a0[s][0]),
        "n_seeds": len(a0),
        "per_seed": {s: {"t0": a0[s][0], "t_end": a0[s][1],
                         "delta": a0[s][1] - a0[s][0]} for s in a0}}
    surv = {r["seed"]: r["predation"]["survival"] for r in full}
    crit["③高防御存活优势"] = {
        "n_ok": sum(1 for s in surv
                    if surv[s]["alive_mean_defense"]
                    > surv[s]["killed_mean_defense"]),
        "n_seeds": len(surv),
        "per_seed": {s: {"alive": surv[s]["alive_mean_defense"],
                         "killed": surv[s]["killed_mean_defense"]}
                     for s in surv},
        "lineage_spearman": {r["seed"]:
                             r["predation"]["lineage"]
                             ["spearman_defense_survival"] for r in full}}
    crit["④两性状同升或波动"] = {
        "n_ok": sum(1 for s in d0
                    if d0[s][1] > d0[s][0] and a0[s][1] > a0[s][0]),
        "n_seeds": len(d0)}
    bc = {r["seed"]: r["predation"]["final"]["bc_defense"] for r in full}
    crit["⑤二型化 BC>0.555"] = {
        "n_ok": sum(1 for s in bc if bc[s] > 0.555),
        "n_seeds": len(bc), "per_seed": bc}
    # 负对照：军备竞赛必须减弱/消失
    neg: Dict = {}
    controls = [("no_loot", no_loot, "defense_mean"),
                ("no_defense", no_def, "defense_mean")]
    if off:
        controls.append(("off", off, "defense_mean"))
    for name, arm_rows, key in controls:
        d = {r["seed"]: _traj_end(r, key) for r in arm_rows}
        neg[name] = {
            "n_rise": sum(1 for s in d if d[s][1] > d[s][0]),
            "n_seeds": len(d),
            "per_seed": {s: {"t0": d[s][0], "t_end": d[s][1],
                             "delta": d[s][1] - d[s][0]} for s in d}}
    report["criterion"] = crit
    report["negative_controls"] = neg

    if label == "b16":
        out_path = out_dir / (
            f"b16_race_summary_t{ticks}.json" if spread is None
            else f"b16_race_probe_spread{spread}_t{ticks}.json")
    else:
        out_path = out_dir / f"{label}_race_summary_t{ticks}.json"
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, sort_keys=True, indent=1)
        fh.write("\n")
    title = "B16.3 军备竞赛" if label == "b16" else "B17.2 军备竞赛（命中转移）"
    print(f"=== {title}：{ticks} ticks × {len(seeds)} seeds "
          f"× {len(arm_table)} 臂（{meta['mode']} "
          f"{meta.get('workers', 1)} 进程，{meta['wall_s']:.1f}s）===")
    for arm in arm_table:
        gate = report["gates"][arm]
        print(f"[{arm}] 生态门 {gate['n_passed']}/{gate['n_total']} "
              f"{gate['verdict']}")
        for r in report["arms"][arm]:
            tr = r["predation"]["trajectory"]
            fin = r["predation"]["final"]
            print("  s%-3d alive=%-4d def %.3f→%.3f atk %.3f→%.3f "
                  "BC_d=%.3f kills=%-5d loot/ind=%.3f" % (
                      r["seed"], r["population"]["final_alive"],
                      tr[0]["defense_mean"], tr[-1]["defense_mean"],
                      tr[0]["attack_mean"], tr[-1]["attack_mean"],
                      fin["bc_defense"],
                      r["predation"]["attacks"]["n_kills"],
                      r["predation"]["per_capita"]["loot_per_capita"]))
    print()
    for name, c in crit.items():
        print(f"[{name}] {c['n_ok']}/{c['n_seeds']} seed 方向一致")
    for name, c in neg.items():
        print(f"[负对照 {name}] defense 上升 {c['n_rise']}/{c['n_seeds']} seed")
    print(f"summary: {out_path}")
    return report


def b17(seeds: List[int], ticks: int, out_dir: Path, workers: int = 1,
        ckpt_dir=None, ckpt_every: int = 500) -> Dict:
    """B17.2 判据跑批：4 臂 × seeds × ticks（spread 固定 2.0）。

    与 B16 同一套五条判据；唯一机制差异 = ``contact_loot``（命中即转移）。
    """
    return b16(seeds, ticks, out_dir, workers=workers,
               ckpt_dir=ckpt_dir, ckpt_every=ckpt_every,
               arms=B17_RACE_ARMS, label="b17")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m biosphere.experiments",
        description="B 线对照实验跑批器（B6 威胁对照 / B8 捕食双型 / B11 W4）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def _add_common(p):
        p.add_argument("--out", type=str, default="results/biosphere")
        p.add_argument("--ckpt-dir", type=str, default="",
                       help="checkpoint 目录（默认：ticks>1000 时 "
                            "results/biosphere/ckpt，否则不写）")
        p.add_argument("--resume", type=str, default="",
                       help="从 checkpoint 续跑（pkl 路径；恢复步骤见汇报）")
        p.add_argument("--ckpt-every", type=int, default=500,
                       help="checkpoint 间隔（tick）；磁盘紧张时调大")

    p_ctrl = sub.add_parser("control", help="B6 两臂威胁对照 + 命中窗口分布")
    p_ctrl.add_argument("--seeds", type=str, default="42,43,44")
    p_ctrl.add_argument("--ticks", type=int, default=1000)
    _add_common(p_ctrl)
    p_dual = sub.add_parser("dualtype", help="B8 捕食者-猎物双型对照")
    p_dual.add_argument("--seed", type=int, default=42)
    p_dual.add_argument("--ticks", type=int, default=2000)
    p_dual.add_argument("--pred", type=int, default=20)
    p_dual.add_argument("--prey", type=int, default=40)
    _add_common(p_dual)
    p_w4 = sub.add_parser("w4", help="B11 W4 多细胞/合作对照（完整 + 四零模型）")
    p_w4.add_argument("--seeds", type=str, default="41,42,43,44,45")
    p_w4.add_argument("--ticks", type=int, default=3000)
    _add_common(p_w4)
    p_rej = sub.add_parser("b15-rejudge",
                           help="B15.3 新判据重判 B14 数据（恢复终局 checkpoint）")
    p_rej.add_argument("--seeds", type=str, default="41,42,43,44,45")
    p_rej.add_argument("--ticks", type=int, default=3000)
    _add_common(p_rej)
    p_b15p = sub.add_parser("b15-power",
                            help="B15.2 功效分析（N=100/500/2000 × 2 臂）")
    p_b15p.add_argument("--seeds", type=str, default="41,42,43,44,45,46,47,48,49,50")
    p_b15p.add_argument("--ticks", type=int, default=1000)
    _add_common(p_b15p)
    p_b15 = sub.add_parser("b15", help="B15.1 oracle 正对照（3 臂诊断）")
    p_b15.add_argument("--seeds", type=str, default="41,42,43,44,45")
    p_b15.add_argument("--ticks", type=int, default=1000)
    _add_common(p_b15)
    p_b16 = sub.add_parser("b16", help="B16.3 军备竞赛跑批（4 臂 × seeds）")
    p_b16.add_argument("--seeds", type=str, default="41,42,43,44,45")
    p_b16.add_argument("--ticks", type=int, default=5000)
    p_b16.add_argument("--workers", type=int, default=1,
                       help="并行进程数（1 = 串行；14 核机建议 10~14）")
    p_b16.add_argument("--spread", type=float, default=None,
                       help="覆盖 init_trait_spread（脱离预注册口径；"
                            "仅供可行性探测）")
    _add_common(p_b16)
    p_b16o = sub.add_parser("b16-oracle",
                            help="B16.1 oracle 正对照（3 臂；不跑演化）")
    p_b16o.add_argument("--seeds", type=str, default="41,42,43,44,45")
    p_b16o.add_argument("--ticks", type=int, default=2000)
    p_b16o.add_argument("--workers", type=int, default=1)
    _add_common(p_b16o)
    p_b17 = sub.add_parser("b17",
                           help="B17.2 军备竞赛二轮（命中转移；4 臂 × seeds）")
    p_b17.add_argument("--seeds", type=str, default="41,42,43,44,45")
    p_b17.add_argument("--ticks", type=int, default=5000)
    p_b17.add_argument("--workers", type=int, default=1)
    _add_common(p_b17)
    p_b17o = sub.add_parser("b17-oracle",
                            help="B17.2 oracle 前置门（3 臂；contact_loot 开）")
    p_b17o.add_argument("--seeds", type=str, default="41,42,43,44,45")
    p_b17o.add_argument("--ticks", type=int, default=2000)
    p_b17o.add_argument("--workers", type=int, default=1)
    _add_common(p_b17o)
    p_c3 = sub.add_parser("c3", help="B14.3 C3 v2 首轮预注册（4 臂对照）")
    p_c3.add_argument("--seeds", type=str, default="41,42,43,44,45")
    p_c3.add_argument("--ticks", type=int, default=3000)
    _add_common(p_c3)
    p_sens = sub.add_parser("sensitivity",
                            help="B14.1 行为阈值三档敏感性（一次跑批三次打标）")
    p_sens.add_argument("--seed", type=int, default=42)
    p_sens.add_argument("--ticks", type=int, default=1000)
    _add_common(p_sens)
    p_w4c = sub.add_parser("w4c", help="B13 W4c 三旋钮对照（预注册 §13.12）")
    p_w4c.add_argument("--seeds", type=str, default="41,42,43,44,45")
    p_w4c.add_argument("--ticks", type=int, default=3000)
    _add_common(p_w4c)
    args = parser.parse_args(argv)

    def _ckpt_dir_of(a) -> Optional[Path]:
        if a.ckpt_dir:
            return Path(a.ckpt_dir)
        # >1000 tick 跑批默认开 checkpoint（补充指令 5）
        return Path("results/biosphere/ckpt") if a.ticks > 1000 else None

    def _parse_seeds(text: str) -> List[int]:
        """支持 "41,42,43" 与 "41-45" 两种写法。"""
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

    if args.cmd == "control":
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        control(seeds, args.ticks, Path(args.out), ckpt_dir=_ckpt_dir_of(args),
                resume=args.resume or None)
    elif args.cmd == "dualtype":
        dualtype(args.seed, args.ticks, Path(args.out),
                 n_pred=args.pred, n_prey=args.prey,
                 ckpt_dir=_ckpt_dir_of(args), resume=args.resume or None)
    elif args.cmd == "b15-rejudge":
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        b15_rejudge(seeds, args.ticks, Path(args.out),
                    Path(args.ckpt_dir or "results/biosphere/ckpt"))
    elif args.cmd == "b15-power":
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        b15_power(seeds, args.ticks, Path(args.out),
                  ckpt_dir=_ckpt_dir_of(args), ckpt_every=args.ckpt_every)
    elif args.cmd == "b15":
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        b15(seeds, args.ticks, Path(args.out), ckpt_dir=_ckpt_dir_of(args),
            resume=args.resume or None, ckpt_every=args.ckpt_every)
    elif args.cmd == "b16":
        seeds = _parse_seeds(args.seeds)
        b16(seeds, args.ticks, Path(args.out), workers=args.workers,
            ckpt_dir=_ckpt_dir_of(args), ckpt_every=args.ckpt_every,
            spread=args.spread)
    elif args.cmd == "b16-oracle":
        seeds = _parse_seeds(args.seeds)
        b16_oracle(seeds, args.ticks, Path(args.out), workers=args.workers,
                   ckpt_dir=_ckpt_dir_of(args), ckpt_every=args.ckpt_every)
    elif args.cmd == "b17":
        seeds = _parse_seeds(args.seeds)
        b17(seeds, args.ticks, Path(args.out), workers=args.workers,
            ckpt_dir=_ckpt_dir_of(args), ckpt_every=args.ckpt_every)
    elif args.cmd == "b17-oracle":
        seeds = _parse_seeds(args.seeds)
        b17_oracle(seeds, args.ticks, Path(args.out), workers=args.workers,
                   ckpt_dir=_ckpt_dir_of(args), ckpt_every=args.ckpt_every)
    elif args.cmd == "c3":
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        c3(seeds, args.ticks, Path(args.out), ckpt_dir=_ckpt_dir_of(args),
           resume=args.resume or None, ckpt_every=args.ckpt_every)
    elif args.cmd == "sensitivity":
        threshold_sensitivity(args.seed, args.ticks, Path(args.out))
    elif args.cmd == "w4c":
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        w4c(seeds, args.ticks, Path(args.out), ckpt_dir=_ckpt_dir_of(args),
            resume=args.resume or None, ckpt_every=args.ckpt_every)
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        w4(seeds, args.ticks, Path(args.out), ckpt_dir=_ckpt_dir_of(args),
           resume=args.resume or None, ckpt_every=args.ckpt_every)
    return 0


if __name__ == "__main__":
    sys.exit(main())
