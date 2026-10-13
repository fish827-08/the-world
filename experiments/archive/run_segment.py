#!/usr/bin/env python3
"""分段续跑脚本：从快照恢复 → 跑 N tick → 立即保存快照 → 增量写统计。

设计要点（针对云端杀进程）：
1. 每段结束立即 save_snapshot（即使后续被杀，进度不丢）
2. 每 RECORD_INTERVAL tick 立即追加一行统计到 CSV（不等到结束）
3. 快照存在则恢复，不存在则初始化新实验
4. pred_cum 跨段恢复：从统计 CSV 末行读取
5. 支持命令行参数指定每段 tick 数、快照路径、输出路径

用法：
    python3 experiments/run_segment.py --ticks 80000
    python3 experiments/run_segment.py --ticks 80000 --snapshot snap.npz --output result.csv
"""
import sys
import os
import time
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from simulation.config import SimConfig
from simulation.sphere_engine import SphereEngine
from core.lifecycle import DeathCause

# 默认参数
DEFAULT_TICKS = 40_000          # 每段跑多少 tick（~5 分钟 @130 tick/s，留安全余量）
RECORD_INTERVAL = 2000          # 统计记录间隔
SEED = 42
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SNAPSHOT = os.path.join(PROJECT_ROOT, "snapshot.npz")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "long_run_segment_result.csv")

CSV_HEADER = ["tick", "N", "sig_density", "g14", "g15", "g16", "g19",
              "trust", "cult_div", "valence", "arousal", "pred_cum",
              "total_energy", "rate", "elapsed_h"]


def _append_row(row, output_path):
    """立即落盘一行统计（防中途被杀丢数据）。"""
    new_file = not os.path.exists(output_path)
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(CSV_HEADER)
        w.writerow(row)


def _make_config(seed):
    """创建标准实验配置（与 long_run_l4l5.py 一致）。"""
    cfg = SimConfig(seed=seed)
    cfg.world.rows = 60
    cfg.world.cols = 120
    cfg.population.initial_count = 500
    cfg.resources.distribution = "patchy"
    cfg.simulation.use_sim_core = True
    # L10a 默认关闭（fruit.enabled=False），不改变基线生态
    return cfg


def main():
    parser = argparse.ArgumentParser(description="分段续跑实验")
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS,
                        help=f"本段跑多少 tick（默认 {DEFAULT_TICKS}）")
    parser.add_argument("--snapshot", type=str, default=DEFAULT_SNAPSHOT,
                        help="快照文件路径")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="统计输出 CSV 路径")
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子（仅新实验用）")
    args = parser.parse_args()

    t_start = time.time()

    # === 1. 恢复或初始化引擎 ===
    if os.path.exists(args.snapshot):
        print(f"[快照恢复] 从 {args.snapshot} 恢复...")
        e = SphereEngine.load_snapshot(args.snapshot)
        start_tick = e._tick
        print(f"  已恢复: tick={start_tick}, N={len(e._id)}, "
              f"max_gen={e._max_generation}")
    else:
        print(f"[新实验] 快照不存在，初始化新引擎 (seed={args.seed})...")
        cfg = _make_config(args.seed)
        e = SphereEngine(cfg)
        start_tick = 0
        print(f"  已初始化: tick=0, N={len(e._id)}")

    # === 1.5 跨段统计恢复：pred_cum 从 CSV 末行读取 ===
    predation_cum = 0
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) >= 2:
                last = lines[-1].strip().split(",")
                # CSV 列序: pred_cum index=11
                predation_cum = int(float(last[11]))
                print(f"[统计恢复] 从 CSV 末行恢复 pred_cum={predation_cum}")
        except (IndexError, ValueError) as exc:
            print(f"[统计恢复] 读取末行失败，pred_cum 从 0 开始: {exc}")

    # === 2. 跑 ticks ===
    target_tick = start_tick + args.ticks
    print(f"[运行] tick {start_tick} → {target_tick} (本段 {args.ticks} tick)")
    print(f"{'tick':>7} {'N':>5} {'sig':>6} {'g14':>5} {'g15':>5} "
          f"{'g16':>5} {'g19':>5} {'trust':>6} {'cult_div':>8} "
          f"{'val':>6} {'rate':>8}")
    sys.stdout.flush()

    for t in range(start_tick + 1, target_tick + 1):
        s = e.step()
        predation_cum += s.deaths_by_cause.get(DeathCause.PREDATION, 0)

        if e._extinct:
            print(f"\n*** 种群灭绝 at tick {t} ***")
            break

        if t % RECORD_INTERVAL == 0:
            P = len(e._id)
            if P == 0:
                break
            g = e._genes[:P]
            sig_density = int((e.signals._marks > 0).sum())
            trust_mean = float(e._trust[:P].mean())
            cult_div = float(e._interpret[:P].std(axis=0).mean())
            val_mean = float(e._valence[:P].mean())
            arous_mean = float(e._arousal[:P].mean())
            elapsed = time.time() - t_start
            rate = (t - start_tick) / elapsed if elapsed > 0 else 0

            print(f"{t:7d} {P:5d} {sig_density:6d} "
                  f"{g[:,14].mean():5.3f} {g[:,15].mean():5.3f} "
                  f"{g[:,16].mean():5.3f} {g[:,19].mean():5.3f} "
                  f"{trust_mean:6.3f} {cult_div:8.4f} "
                  f"{val_mean:6.3f} {rate:8.1f}")
            sys.stdout.flush()

            _append_row([
                t, P, sig_density,
                float(g[:, 14].mean()), float(g[:, 15].mean()),
                float(g[:, 16].mean()), float(g[:, 19].mean()),
                trust_mean, cult_div, val_mean, arous_mean,
                predation_cum, float(s.total_energy),
                round(rate, 1), round(elapsed / 3600, 4),
            ], args.output)

    # === 3. 立即保存快照（关键：防云端杀进程丢中间成果）===
    e.save_snapshot(args.snapshot)
    elapsed_total = time.time() - t_start
    final_rate = (e._tick - start_tick) / elapsed_total if elapsed_total > 0 else 0

    print(f"\n[段完成] tick={e._tick}, N={len(e._id)}, "
          f"max_gen={e._max_generation}, pred_cum={predation_cum}")
    print(f"  快照已保存: {args.snapshot}")
    print(f"  统计已追加: {args.output}")
    print(f"  本段耗时: {elapsed_total:.1f}s, 速率: {final_rate:.1f} tick/s")


if __name__ == "__main__":
    main()
