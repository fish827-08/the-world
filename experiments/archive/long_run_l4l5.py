"""L4+L5 长实验：60 万（默认）/ 自定义 tick，观察语言涌现迹象。

记录指标（每 RECORD_INTERVAL tick）：
- 种群数量、信号密度
- g14(感知)/g15(信号)/g16(攻击)/g19(植物化) 均值
- trust 均值、文化多样性（解读表种群内 std 均值）
- valence/arousal 均值
- 累计捕食死亡数

落盘：每 RECORD_INTERVAL tick **立即追加**一行到结果 CSV
（进程被杀/断电也不会丢已记录数据——不等到结束时才写）。
用法：py experiments/long_run_l4l5.py [ticks]  （默认 600000）
"""
import sys
import os
import time
import csv
import numpy as np

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.config import SimConfig
from simulation.sphere_engine import SphereEngine
from core.lifecycle import DeathCause

TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 600_000
RECORD_INTERVAL = 2000
SEED = 42
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "long_run_l4l5_result.csv")


def _append_row(row):
    """立即落盘一行（防中途被杀丢数据）。"""
    new_file = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["tick", "N", "sig_density", "g14", "g15", "g16", "g19",
                        "trust", "cult_div", "valence", "arousal", "pred_cum",
                        "total_energy", "rate", "elapsed_h"])
        w.writerow(row)


def main():
    cfg = SimConfig(seed=SEED)
    cfg.world.rows = 60
    cfg.world.cols = 120
    cfg.population.initial_count = 500
    cfg.resources.distribution = "patchy"  # 斑块化（老写法 resources.patchy=True 是死代码）
    cfg.simulation.use_sim_core = True  # L6 Rust 下沉，N=5000 时 ~117 tick/s
    e = SphereEngine(cfg)

    print(f"=== L4+L5 长实验 ===")
    print(f"seed={SEED}, N0=500, patchy=True, ticks={TICKS}")
    print(f"record every {RECORD_INTERVAL} ticks")
    print(f"gene_count={cfg.genome.gene_count}")
    print()
    print(f"{'tick':>7} {'N':>5} {'sig':>6} {'g14':>5} {'g15':>5} {'g16':>5} {'g19':>5} "
          f"{'trust':>6} {'cult_div':>8} {'val':>6} {'arous':>6} {'pred_cum':>8} {'E':>8}")
    sys.stdout.flush()

    t0 = time.time()
    predation_cum = 0

    for t in range(1, TICKS + 1):
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
            # 文化多样性：解读表每个模式的种群内 std，取均值
            cult_div = float(e._interpret[:P].std(axis=0).mean())
            val_mean = float(e._valence[:P].mean())
            arous_mean = float(e._arousal[:P].mean())
            elapsed = time.time() - t0
            rate = t / elapsed if elapsed > 0 else 0
            print(f"{t:7d} {P:5d} {sig_density:6d} "
                  f"{g[:,14].mean():5.3f} {g[:,15].mean():5.3f} "
                  f"{g[:,16].mean():5.3f} {g[:,19].mean():5.3f} "
                  f"{trust_mean:6.3f} {cult_div:8.4f} "
                  f"{val_mean:6.3f} {arous_mean:6.3f} "
                  f"{predation_cum:8d} {s.total_energy:8.0f} "
                  f"({rate:.0f} tick/s, {elapsed/3600:.1f}h)")
            sys.stdout.flush()
            _append_row([t, P, sig_density,
                         float(g[:, 14].mean()), float(g[:, 15].mean()),
                         float(g[:, 16].mean()), float(g[:, 19].mean()),
                         trust_mean, cult_div, val_mean, arous_mean,
                         predation_cum, float(s.total_energy),
                         round(rate, 1), round(elapsed / 3600, 4)])

    elapsed = time.time() - t0
    print(f"\n=== 实验结束 ===")
    print(f"总耗时: {elapsed/3600:.2f} 小时 ({elapsed:.0f} 秒)")
    print(f"平均速度: {TICKS/elapsed:.0f} tick/s")
    print(f"最终种群: {len(e._id)}")
    print(f"累计捕食死亡: {predation_cum}")
    if len(e._id) > 0:
        g = e._genes[:len(e._id)]
        print(f"最终基因均值: g14={g[:,14].mean():.3f} g15={g[:,15].mean():.3f} "
              f"g16={g[:,16].mean():.3f} g19={g[:,19].mean():.3f}")
        print(f"最终 trust 均值: {e._trust[:len(e._id)].mean():.3f}")
        print(f"最终文化多样性: {e._interpret[:len(e._id)].std(axis=0).mean():.4f}")


if __name__ == "__main__":
    main()
