#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""R198 / 13.8 S2 判定：多 seed × 相邻采样配对，个体级 ρ(Δ|φ|, g23)。

为什么要重做（诚实记录）：
  * 单 seed 短窗（gap=40）给出 ρ≈+0.001~0.014（**符号对但极小**）；
  * 单 seed 长窗（gap=500/1000）给出 ρ≈−0.08~−0.14（**符号反**，n 仅 138~1461）；
  ⇒ 两者都**不可作为结论**：单 seed + 种群崩到百人量级 ⇒ 噪声压制信号。
  正确做法 = **多 seed 合并**（预注册 §5 明写「seed-stratified, merged」），
  并以**配对差**（同 seed 内有/无迁徙）作 Support-1，避免靠单条轨迹下结论。
"""
import sys
import json
import numpy as np

sys.path.insert(0, "/tmp/tw")
sys.path.insert(0, "/workspace/r198")
from simulation.genes import Gene              # noqa: E402
from s2_smoke import build, lat_of, spearman   # noqa: E402

SEEDS = [1, 2, 3, 4, 5, 6]
TICKS = 6000
GAP = 40          # 相邻采样间隔（tick）
GAINS = [2.0, 20.0, 50.0]


def run_one(seed, mig, gain, ticks=TICKS, gap=GAP):
    """单 seed：相邻采样配对 ⇒ 逐段 Δ|φ| 与段首 g23。"""
    _, eng = build(seed, mig, gain)
    prev = None
    D, G, disp0, dispN = [], [], None, None
    for _ in range(ticks):
        eng.step()
        if eng.tick % gap:
            continue
        ids = eng._id.copy()
        lat = lat_of(eng)
        g23 = eng._genes[:len(ids), Gene.MIGRATE_BIAS].copy()
        if prev is not None:
            common, i0, i1 = np.intersect1d(prev[0], ids, return_indices=True)
            if len(common):
                D.append(lat[i1] - prev[1][i0])
                G.append(prev[2][i0])
        prev = (ids, lat, g23)
    # Support-1 用：全种群平均 |φ| 的净位移（种群级，仅作辅助臂间对照）
    return {
        "D": np.concatenate(D) if D else np.zeros(0),
        "G": np.concatenate(G) if G else np.zeros(0),
        "alive_end": eng.alive_count(),
        "dphi_mean": float(np.concatenate(D).mean()) if D else None,
    }


def main():
    rep = {"seeds": SEEDS, "ticks": TICKS, "gap": GAP, "arms": {}}
    print("=" * 78)
    print(f"多 seed 合并：{len(SEEDS)} seed × {TICKS} tick，gap={GAP}")
    print("=" * 78)

    # ---- 空臂 A（有季节、无迁徙）＝基线 ----
    base = {}
    for s in SEEDS:
        r = run_one(s, False, 0.0)
        base[s] = r
        print(f"  [base ] seed={s} n={len(r['D']):6d} dphi_mean={r['dphi_mean']} alive={r['alive_end']}",
              flush=True)
    BD = np.concatenate([base[s]["D"] for s in SEEDS])
    rep["arms"]["mig_base"] = {
        "rho": None, "n": int(len(BD)),
        "dphi_mean": float(BD.mean()), "dphi_std": float(BD.std()),
        "dphi_abs_mean": float(np.abs(BD).mean()),
        "alive_end": [base[s]["alive_end"] for s in SEEDS],
    }
    print(f"  → base 合并 n={len(BD)} dphi_mean={BD.mean():.5f} "
          f"|dphi|_mean={np.abs(BD).mean():.5f}")

    # ---- 迁徙臂（3 档 gain）----
    for gain in GAINS:
        per = {}
        for s in SEEDS:
            r = run_one(s, True, gain)
            per[s] = r
        D = np.concatenate([per[s]["D"] for s in SEEDS])
        G = np.concatenate([per[s]["G"] for s in SEEDS])
        rho, p = spearman(D, G)
        # seed 分层：逐 seed 算 ρ，再看符号一致性（比合并 ρ 更能抗"单 seed 假象"）
        per_rho = []
        for s in SEEDS:
            if len(per[s]["D"]) >= 30:
                r_s, _ = spearman(per[s]["D"], per[s]["G"])
                per_rho.append(r_s)
        per_rho = [x for x in per_rho if np.isfinite(x)]
        hi = G > np.median(G)
        entry = {
            "rho_merged": float(rho), "p_merged": float(p), "n": int(len(D)),
            "dphi_mean": float(D.mean()), "dphi_std": float(D.std()),
            "dphi_abs_mean": float(np.abs(D).mean()),
            "hi_g23_dphi_mean": float(D[hi].mean()),
            "lo_g23_dphi_mean": float(D[~hi].mean()),
            "per_seed_rho": [float(x) for x in per_rho],
            "per_seed_positive": int(sum(1 for x in per_rho if x > 0)),
            "per_seed_n": len(per_rho),
            "alive_end": [per[s]["alive_end"] for s in SEEDS],
        }
        # Support-1：迁徙臂个体位移 > 空臂（逐 seed 配对）
        wins = 0
        tot = 0
        for s in SEEDS:
            a = np.abs(per[s]["D"]).mean() if len(per[s]["D"]) else None
            b = np.abs(base[s]["D"]).mean() if len(base[s]["D"]) else None
            if a is not None and b is not None:
                tot += 1
                wins += int(a > b)
        entry["support1_paired_wins"] = f"{wins}/{tot}"
        rep["arms"][f"mig_g{gain:g}"] = entry
        print(f"  gain={gain:5.1f} n={len(D):6d} rho={rho:+.4f} p={p:.2e} "
              f"|dphi|={np.abs(D).mean():.5f} hi={D[hi].mean():+.5f} lo={D[~hi].mean():+.5f} "
              f"逐seed正号={entry['per_seed_positive']}/{entry['per_seed_n']} "
              f"S1={entry['support1_paired_wins']} alive={entry['alive_end']}", flush=True)
        rep["arms"][f"mig_g{gain:g}"]["_base_abs_mean"] = float(np.abs(BD).mean())

    with open("/workspace/r198/s2_judge.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print("\n已写 /workspace/r198/s2_judge.json")


if __name__ == "__main__":
    main()
