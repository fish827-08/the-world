"""A3 敏感性扫描：对隐式选择压审计清单 §三 的 4 个高影响参数做扫描。

参数维度（来自 隐式选择压审计清单.md §三）：
  1. predation.transfer_ratio  —— 捕食能量转移率（baseline 0.4，±20%/±50%）
  2. culture.trust_false       —— 信任下降幅度（baseline 0.1，对称化对照 0.05）
  3. pleasure.social_rpe/alone_rpe —— 社会效价（baseline +0.2/-0.1，对称化 ±0.15）
  4. organisms.niche_floor     —— 活性保底（baseline 0.4，→0.0/0.2/0.6）

用法：
  python experiments/sensitivity_scan.py <param> <value> <seed> [ticks]
  例：python experiments/sensitivity_scan.py transfer_ratio 0.2 42 60000

输出：results/sensitivity/<param>_<value>_<seed>/termination.csv
汇总：results/sensitivity/summary.csv（由 run_all.py 或手动汇总）

增量落盘：每 RECORD_INTERVAL tick 立即追加一行，防中断丢数据。
"""
import sys
import os
import time
import csv
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.config import SimConfig
from simulation.sphere_engine import SphereEngine
from simulation.genes import Gene
from core.lifecycle import DeathCause

RECORD_INTERVAL = 1000
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "sensitivity")

# 跟踪的基因列（终局基因均值）
TRACKED_GENES = [
    Gene.MOVE_PROB, Gene.METABOLIC, Gene.REPRO_THRESHOLD, Gene.LIFE_GENE,
    Gene.EAT_AMOUNT, Gene.STOMACH_CAP, Gene.MOVE_COST, Gene.PARENTAL_INVEST,
    Gene.PHOTOSYNTHESIS, Gene.HOMEOTHERM, Gene.FORAGE_NEIGHBOR, Gene.TEMP_PREF,
    Gene.REPRO_COOLDOWN, Gene.SOCIABILITY, Gene.PERCEPTION, Gene.SIGNAL_STRENGTH,
    Gene.AGGRESSION, Gene.ROOTING,
]


def apply_param_override(cfg: SimConfig, param: str, value: float) -> None:
    """将参数覆盖应用到 cfg。param 支持点号路径，如 'predation.transfer_ratio'。"""
    if param == "transfer_ratio":
        cfg.predation.transfer_ratio = value
    elif param == "trust_false":
        cfg.culture.trust_false = value
    elif param == "social_rpe":
        # 对称化：social_rpe 和 alone_rpe 同时设为 ±value
        cfg.pleasure.social_rpe = value
        cfg.pleasure.alone_rpe = -value
    elif param == "niche_floor":
        cfg.organisms.niche_floor = value
    elif param == "baseline":
        pass  # 不覆盖任何参数
    else:
        raise ValueError(f"未知参数: {param}")


def make_config(param: str, value: float, seed: int) -> SimConfig:
    """构建实验配置（复用 long_run_l4l5.py 的基线配置）。"""
    cfg = SimConfig(seed=seed)
    cfg.world.rows = 60
    cfg.world.cols = 120
    cfg.population.initial_count = 500
    cfg.resources.distribution = "patchy"
    cfg.simulation.use_sim_core = True
    apply_param_override(cfg, param, value)
    return cfg


def gene_means(engine: SphereEngine) -> dict[str, float]:
    """计算当前种群的跟踪基因均值。"""
    P = len(engine._id)
    if P == 0:
        return {g.name: 0.0 for g in TRACKED_GENES}
    return {g.name: float(engine._genes[:P, int(g)].mean()) for g in TRACKED_GENES}


def run_experiment(param: str, value: float, seed: int, ticks: int) -> dict:
    """跑单次实验，返回终局统计。"""
    cfg = make_config(param, value, seed)
    engine = SphereEngine(cfg)

    # 输出目录
    value_str = f"{value:.4f}".rstrip("0").rstrip(".")
    exp_dir = os.path.join(RESULTS_DIR, f"{param}_{value_str}_{seed}")
    os.makedirs(exp_dir, exist_ok=True)
    out_csv = os.path.join(exp_dir, "termination.csv")

    # 写配置快照
    config_snapshot = {
        "param": param, "value": value, "seed": seed, "ticks": ticks,
        "transfer_ratio": cfg.predation.transfer_ratio,
        "trust_false": cfg.culture.trust_false,
        "social_rpe": cfg.pleasure.social_rpe,
        "alone_rpe": cfg.pleasure.alone_rpe,
        "niche_floor": cfg.organisms.niche_floor,
    }
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2, ensure_ascii=False)

    # CSV 表头
    fieldnames = ["tick", "N", "max_gen", "sig_density", "trust_mean",
                  "valence_mean", "arousal_mean", "total_energy", "total_resource",
                  "pred_cum"] + [g.name for g in TRACKED_GENES]

    def append_row(row: dict):
        new_file = not os.path.exists(out_csv)
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if new_file:
                w.writeheader()
            w.writerow(row)

    print(f"=== 敏感性扫描: {param}={value}, seed={seed}, ticks={ticks} ===")
    print(f"输出: {out_csv}")
    t0 = time.time()

    for tick in range(1, ticks + 1):
        engine.step()
        if tick % RECORD_INTERVAL == 0:
            P = len(engine._id)
            row = {
                "tick": tick,
                "N": P,
                "max_gen": int(engine._generation[:P].max()) if P > 0 else 0,
                "sig_density": float((engine.signals._marks > 0).mean()),
                "trust_mean": float(engine._trust[:P].mean()) if P > 0 else 0.0,
                "valence_mean": float(engine._valence[:P].mean()) if P > 0 else 0.0,
                "arousal_mean": float(engine._arousal[:P].mean()) if P > 0 else 0.0,
                "total_energy": float(engine._energy[:P].sum()) if P > 0 else 0.0,
                "total_resource": float(engine.resources._grid.sum()),
                "pred_cum": int(engine._run_deaths.get(DeathCause.PREDATION, 0)),
            }
            row.update(gene_means(engine))
            append_row(row)
            elapsed = time.time() - t0
            rate = tick / elapsed if elapsed > 0 else 0
            print(f"  tick={tick:>6} N={P:>4} gen={row['max_gen']:>3} "
                  f"g16={row['AGGRESSION']:.3f} g15={row['SIGNAL_STRENGTH']:.3f} "
                  f"trust={row['trust_mean']:.3f} rate={rate:.1f}t/s")

            # 灭绝提前终止
            if P == 0:
                print(f"  种群灭绝于 tick={tick}，提前终止")
                break

    elapsed = time.time() - t0
    # 终局统计
    P = len(engine._id)
    final = {
        "param": param, "value": value, "seed": seed,
        "ticks_run": tick, "elapsed_s": round(elapsed, 1),
        "final_N": P,
        "final_max_gen": int(engine._generation[:P].max()) if P > 0 else 0,
        "extinct": P == 0,
    }
    final.update(gene_means(engine))
    if P > 0:
        final["trust_mean"] = float(engine._trust[:P].mean())
        final["valence_mean"] = float(engine._valence[:P].mean())
    else:
        final["trust_mean"] = 0.0
        final["valence_mean"] = 0.0

    # 写终局 JSON
    with open(os.path.join(exp_dir, "final.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"完成: {param}={value}, seed={seed}, {tick} ticks, {elapsed:.1f}s, "
          f"final_N={P}, max_gen={final['final_max_gen']}")
    return final


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    param = sys.argv[1]
    value = float(sys.argv[2])
    seed = int(sys.argv[3])
    ticks = int(sys.argv[4]) if len(sys.argv) > 4 else 60_000

    run_experiment(param, value, seed, ticks)


if __name__ == "__main__":
    main()
