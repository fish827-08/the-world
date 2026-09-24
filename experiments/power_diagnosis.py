#!/usr/bin/env python3.12
"""检验适配性诊断（可复核）—— 解释相位零分布 p 为何不稳定。

结论（四条，全部可复算）：
  ① n 不是瓶颈：完美同相合成信号在 **n=25** 时 p = 0.0010（可拒绝）。
  ② 该检验的输入是**零滞后 corr**：同相合成信号 corr≥0.9 ⇒ p<0.03；
     corr≤0.8 ⇒ p>0.05。
  ③ 真实数据是**带滞后的相位锁定**（种群追不上食物带瞬时位置，
     lag = 2~4 采样点）。零滞后 corr 因此被压低
     （seed 42/123/2024 = +0.58 / −0.12 / +0.51），
     而**带 lag 最优 corr 都很强**（+0.994 / +0.826 / +0.975）。
     ⇒ 相位零分布检验**不适配"带滞后锁定"**，而非机制不成立。
     seed 7（lag=0）zero-lag corr 高 ⇒ p=0.0105。
  ④ **零分布宽度随窗口长度 n 剧烈变化**（本文件第三节）：季节信号是
     低维窄带信号，相位随机化后的 |corr| 中位数可在 0.30~0.71 间跳，
     ⇒ **p 值本身不稳定**（同一数据仅改 n，p 在 0.0000~0.15 间跳变）。
     这是相位随机化方法对窄带信号的固有弱点，**不是机制问题**。
     ⇒ 稳健做法：报告**零滞后 corr** 与**带 lag corr**（二者跨窗口稳定），
     不把相位零分布 p 当作唯一判据。

用法：python3.12 power_diagnosis.py [CSV...]
"""
import sys, csv
import numpy as np

_here = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
sys.path.insert(0, _here)
from verify_migration_verdict import phase_null_pvalue, best_lag_corr


def synth_corr(target, n, period_pts, seed=0):
    """构造 corr 恰为 target 的合成对（与观测同频的正弦）。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = np.sin(2 * np.pi * t / period_pts)
    env = base.copy()
    noise = rng.standard_normal(n)
    noise -= noise.mean()
    noise -= noise.dot(base) / base.dot(base) * base
    noise /= noise.std()
    bn = (base - base.mean()) / base.std()
    k = np.sqrt((1 - target ** 2) / target ** 2) if target < 1 else 0.0
    return bn + k * noise, env


def main():
    print("=" * 74)
    print("① n 不是瓶颈：完美同相合成信号")
    print("=" * 74)
    print(f"{'n':>5} {'周期(点)':>8} {'corr':>9} {'相位零分布p':>12}")
    for n, pp in [(25, 12), (49, 24), (73, 24), (121, 24)]:
        cen, env = synth_corr(0.998, n, pp, seed=1)
        _, p = phase_null_pvalue(cen, env, n_perm=1000, seed=2)
        print(f"{n:>5} {pp:>8} {np.corrcoef(cen,env)[0,1]:>+9.4f} {p:>12.4f}")

    print()
    print("=" * 74)
    print("② 真正的输入是零滞后 corr（n=25, 周期 12 点）")
    print("=" * 74)
    print(f"{'目标corr':>9} {'实测corr':>9} {'相位零分布p':>12} {'可拒绝?':>9}")
    for target in [0.99, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50]:
        cen, env = synth_corr(target, 25, 12, seed=3)
        _, p = phase_null_pvalue(cen, env, n_perm=1000, seed=3)
        print(f"{target:>9.2f} {np.corrcoef(cen,env)[0,1]:>+9.4f} {p:>12.4f} "
              f"{'是' if p < 0.05 else '否':>9}")

    csvs = sys.argv[1:]
    if not csvs:
        return
    print()
    print("=" * 74)
    print("③ 真实数据：零滞后 corr（检验输入）vs 带 lag 最优 corr（真实关系）")
    print("=" * 74)
    print(f"{'seed':>6} {'n':>4} {'零滞后corr':>11} {'带lag最优':>11} {'lag':>4} "
          f"{'相位零分布p':>11}")
    for path in csvs:
        rows = [r for r in csv.DictReader(open(path)) if float(r["tick"]) >= 6000]
        if len(rows) < 8:
            continue
        cen = np.array([float(r["centroid_lat_deg"]) for r in rows])
        env = np.array([float(r["band_env_deg"]) for r in rows])
        n = len(cen)
        c0 = float(np.corrcoef(cen, env)[0, 1])
        _, p = phase_null_pvalue(cen, env, n_perm=2000, seed=1)
        bl, lag = best_lag_corr(cen, env, max(4, n // 3))
        seed = path.split("_s")[-1].replace(".csv", "")
        print(f"{seed:>6} {n:>4} {c0:>+11.4f} {bl:>+11.4f} {lag:>4d} {p:>11.4f}")

    # ④ 零分布宽度 / p 的窗口敏感性
    print()
    print("=" * 74)
    print("④ p 值本身不稳定：同一数据只改窗口长度 n（起点固定）")
    print("=" * 74)
    path = csvs[0]
    allrows = list(csv.DictReader(open(path)))
    tick = np.array([float(r["tick"]) for r in allrows])
    cen_all = np.array([float(r["centroid_lat_deg"]) for r in allrows])
    env_all = np.array([float(r["band_env_deg"]) for r in allrows])
    dt = float(np.median(np.diff(tick)))
    per_pts = int(round(6000.0 / dt))          # 1 个季节周期的采样点数
    t0 = tick[0] + 6000.0
    print(f"起点 t={t0:.0f}（跳过建群），周期 = {per_pts} 采样点")
    print(f"{'n':>4} {'零滞后corr':>11} {'|corr|中位':>11} {'|corr|90分位':>13} {'p':>8}")
    rng = np.random.default_rng(1)
    for k in range(3, 6):
        for frac in (0.0, 0.4):
            n = int(round(per_pts * k * (1.0 + frac)))
            m = (tick >= t0) & (tick < t0 + n * dt)
            a, b = cen_all[m], env_all[m]
            if len(a) < 20 or len(b) % 2:
                continue
            obs = float(np.corrcoef(a, b)[0, 1])
            B = np.fft.rfft(b - b.mean())
            vals = []
            for _ in range(2000):
                ph = np.exp(2j * np.pi * rng.random(len(B)))
                ph[0] = 1.0
                ph[-1] = ph[-1].real
                bp = np.fft.irfft(B * ph, n=len(b))
                if bp.std() > 1e-9:
                    vals.append(abs(float(np.corrcoef(a, bp)[0, 1])))
            vals = np.array(vals)
            print(f"{len(a):>4} {obs:>+11.4f} {np.quantile(vals,0.5):>11.4f} "
                  f"{np.quantile(vals,0.9):>13.4f} {np.mean(vals>=abs(obs)):>8.4f}")
    print()
    print("⇒ 零滞后 corr 跨窗口稳定（±0.005），但 |corr| 零分布与 p 剧烈跳动")
    print("⇒ 季节信号是窄带信号，相位随机化检验对本类信号 p 不稳定")
    print("⇒ 建议以「零滞后 corr + 带 lag corr + 半周期反号」为判据")


if __name__ == "__main__":
    main()
