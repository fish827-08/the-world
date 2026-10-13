#!/usr/bin/env python3
"""长实验脚本：支持自定义世界大小、密度控制、总tick数。

设计要点：
1. 每段结束立即 save_snapshot（防中途被杀丢数据）
2. 每 RECORD_INTERVAL tick 立即追加一行统计到 CSV
3. 快照存在则恢复，不存在则初始化新实验
4. 密度控制：max_count = 格子数 × density_cap（默认45%，保证<50%）

用法：
    # 小世界100万tick
    python3 experiments/run_long_experiment.py --rows 60 --cols 120 --ticks 1000000 --tag small

    # 大世界100万tick
    python3 experiments/run_long_experiment.py --rows 200 --cols 400 --ticks 1000000 --tag large
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

RECORD_INTERVAL = 2000
SEED = 42
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_HEADER = ["tick", "N", "density", "sig_density", "g14", "g15", "g16", "g19",
              "trust", "cult_div", "valence", "arousal", "pred_cum",
              "total_energy", "rate", "elapsed_h", "max_gen"]


def _append_row(row, output_path):
    new_file = not os.path.exists(output_path)
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(CSV_HEADER)
        w.writerow(row)


def _make_config(args):
    cfg = SimConfig(seed=args.seed)
    cfg.world.rows = args.rows
    cfg.world.cols = args.cols
    n_cells = args.rows * args.cols
    cfg.population.initial_count = args.initial
    cfg.population.max_count = int(n_cells * args.density_cap)
    cfg.resources.distribution = "patchy"
    cfg.simulation.use_sim_core = True
    # D3 世代时间扫描（云端实验用，None 时保持默认）；寿命基准随
    # rotation_period 派生（_lifespan = day*(1+g3*7)），压缩于此同步提速世代
    if args.rotation_period:
        cfg.light.rotation_period = args.rotation_period
    # 战役参数：lifespan_mult 寿命基准缩放（不动昼夜，直接压寿命→世代缩短）
    if args.lifespan_mult != 1.0:
        cfg.organisms.lifespan_mult = args.lifespan_mult
    # 战役参数：stay_prob 停驻率（move_prob × (1-stay_prob)，等效扩世界）
    if args.stay_prob != 0.0:
        cfg.simulation.stay_prob = args.stay_prob
    # D1 零模型三开关
    cfg.neutral_genes = args.neutral_genes
    cfg.signal_disabled = args.signal_disabled
    cfg.signal_mode = args.signal_mode
    return cfg


def _stats(e, tick, rate, elapsed_h):
    p = e.alive_count()
    n_cells = e.world.n_cells
    density = p / n_cells
    g = e._genes[:p].mean(axis=0) if p > 0 else np.zeros(e._genes.shape[1])
    sig_density = int(e.signals.active_count()) if hasattr(e, 'signals') else 0
    trust = float(e._trust[:p].mean()) if p > 0 and hasattr(e, '_trust') else 0.0
    cult_div = float(e._interpret[:p].std(axis=0).mean()) if p > 0 and hasattr(e, '_interpret') else 0.0
    valence = float(e._valence[:p].mean()) if p > 0 and hasattr(e, '_valence') else 0.0
    arousal = float(e._arousal[:p].mean()) if p > 0 and hasattr(e, '_arousal') else 0.0
    total_energy = float(e._energy[:p].sum()) if p > 0 else 0.0
    max_gen = int(e._generation[:p].max()) if p > 0 else 0
    # D0 修复：pred_cum 从引擎全局计数实时取（此前恒 0），elapsed_h 跨段累计
    pred_cum = float(e.death_cause_totals().get(DeathCause.PREDATION, 0))
    return [tick, p, f"{density:.4f}", sig_density,
            f"{g[14]:.4f}" if len(g) > 14 else "0",
            f"{g[15]:.4f}" if len(g) > 15 else "0",
            f"{g[16]:.4f}" if len(g) > 16 else "0",
            f"{g[19]:.4f}" if len(g) > 19 else "0",
            f"{trust:.4f}", f"{cult_div:.6f}",
            f"{valence:.4f}", f"{arousal:.4f}",
            f"{pred_cum:.0f}", f"{total_energy:.1f}",
            f"{rate:.1f}", f"{elapsed_h:.4f}", max_gen]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=60)
    parser.add_argument("--cols", type=int, default=120)
    parser.add_argument("--ticks", type=int, default=1000000)
    parser.add_argument("--initial", type=int, default=500)
    parser.add_argument("--density-cap", type=float, default=0.45,
                        help="种群上限占格子数比例（默认45%%，保证小于50%%）")
    parser.add_argument("--tag", type=str, default="exp",
                        help="实验标签，用于命名快照和输出文件")
    parser.add_argument("--segment", type=int, default=0,
                        help="每段tick数，0表示一次性跑完（适合短实验）")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="随机种子（长实验 ≥5 seed 系综用，D1 统计纪律）")
    parser.add_argument("--rotation-period", type=int, default=0,
                        help="世界自转周期覆盖（D3 世代时间扫描：默认2400，压缩到300~600 提速世代）")
    # D1 零模型三开关（进 SimConfig.fingerprint，用于对照实验）
    parser.add_argument("--neutral-genes", action="store_true",
                        help="D1 基因断线：所有基因恒=0.5（读取侧冻结，写入侧照常）")
    parser.add_argument("--signal-disabled", action="store_true",
                        help="D1 不发信号：发射概率恒0（接收/解读照常）")
    parser.add_argument("--signal-mode", type=str, default="state",
                        choices=["state", "random", "evolved"],
                        help="D1 信号编码模式：state(现状)/random(独立rng随机)/evolved(D2码本暂未接线)")
    # 战役参数（千代战役用：不动昼夜直接压寿命 + 停驻等效扩世界）
    parser.add_argument("--lifespan-mult", type=float, default=1.0,
                        help="寿命基准缩放（1.0=旧行为；0.25=寿命缩到1/4→世代×4；范围(0.05,8]）")
    parser.add_argument("--stay-prob", type=float, default=0.0,
                        help="停驻率（0=旧行为每tick必移判定；0.8=移动概率×0.2，等效扩世界；范围[0,0.95)）")
    args = parser.parse_args()

    n_cells = args.rows * args.cols
    max_count = int(n_cells * args.density_cap)
    snapshot_path = os.path.join(PROJECT_ROOT, f"snapshot_{args.tag}.npz")
    output_path = os.path.join(PROJECT_ROOT, "experiments", f"long_{args.tag}.csv")

    print(f"=== 长实验: {args.tag} ===")
    print(f"世界: {args.rows}x{args.cols} = {n_cells}格")
    print(f"种群: 初始={args.initial}, 上限={max_count} (密度={args.density_cap*100:.0f}%)")
    print(f"目标: {args.ticks} tick")
    if args.neutral_genes or args.signal_disabled or args.signal_mode != "state":
        print(f"D1 零模型: neutral_genes={args.neutral_genes}, signal_disabled={args.signal_disabled}, signal_mode={args.signal_mode}")
    print(f"快照: {snapshot_path}")
    print(f"输出: {output_path}")

    # 初始化或恢复
    if os.path.exists(snapshot_path):
        # load_snapshot 是 classmethod，返回新实例
        e = SphereEngine.load_snapshot(snapshot_path)
        start_tick = e.tick
        print(f"从快照恢复: tick={start_tick}, N={e.alive_count()}")
    else:
        cfg = _make_config(args)
        e = SphereEngine(cfg)
        start_tick = 0
        print(f"新实验初始化: N={e.alive_count()}")

    # 从CSV恢复跨段累计 elapsed_h（此前只记本段，误导分析）
    elapsed_h_acc = 0.0
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            lines = f.readlines()
            if len(lines) > 1:
                last = lines[-1].strip().split(",")
                elapsed_h_acc = float(last[15]) if len(last) > 15 else 0.0

    t0 = time.time()
    ticks_this_run = 0
    segment_ticks = args.segment if args.segment > 0 else (args.ticks - start_tick)
    target = min(start_tick + segment_ticks, args.ticks)

    print(f"本次跑: {start_tick} -> {target} ({target-start_tick} tick)")

    for tick in range(start_tick, target):
        e.step()
        ticks_this_run += 1

        if (tick + 1) % RECORD_INTERVAL == 0:
            elapsed = time.time() - t0
            rate = ticks_this_run / elapsed if elapsed > 0 else 0
            elapsed_h = elapsed_h_acc + elapsed / 3600
            row = _stats(e, tick + 1, rate, elapsed_h)
            _append_row(row, output_path)
            print(f"  tick={tick+1}, N={e.alive_count()}, density={e.alive_count()/n_cells:.3f}, "
                  f"rate={rate:.0f} tick/s, g15={row[5]}, max_gen={row[16]}")

    # 段结束立即保存快照
    e.save_snapshot(snapshot_path)
    elapsed = time.time() - t0
    rate = ticks_this_run / elapsed if elapsed > 0 else 0
    print(f"\n=== 段完成 ===")
    print(f"tick={e.tick}, N={e.alive_count()}, rate={rate:.0f} tick/s")
    print(f"快照已保存: {snapshot_path}")

    if e.tick >= args.ticks:
        print(f"*** 实验完成: {args.ticks} tick ***")
    else:
        remaining = args.ticks - e.tick
        eta_min = remaining / rate / 60 if rate > 0 else float('inf')
        print(f"剩余: {remaining} tick, 预计还需 {eta_min:.1f} 分钟")


if __name__ == "__main__":
    main()
