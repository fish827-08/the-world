#!/usr/bin/env python3.12
"""迁徙判读汇总 —— 读多个 seed 的 summary.json，出三态判定 + 图。

判据（设计稿 §2.2）：
  有迁徙      : corr(centroid, band) 显著 > 0 且滞后小
  有迁徙但滞后: 相关显著但存在固定滞后
  无迁徙      : 相关不显著（阴性）

显著性：用**相位随机化零分布**（打乱 band 序列相位）估 p 值——
  比裸相关系数更稳（裸相关在半周期上会虚高）。
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_CJK = None
for _fn in ("Noto Sans CJK SC", "WenQuanYi Zen Hei", "Source Han Sans SC"):
    try:
        if any(_fn in f.name for f in font_manager.fontManager.ttflist):
            _CJK = _fn
            break
    except Exception:
        pass
if _CJK:
    plt.rcParams["font.sans-serif"] = [_CJK, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def phase_null_pvalue(a, b, n_perm=2000, seed=0):
    """相位随机化：保持 b 的功率谱，随机相位 ⇒ 破坏对齐但保留周期结构。"""
    rng = np.random.default_rng(seed)
    n = len(b)
    B = np.fft.rfft(b - b.mean())
    obs = np.corrcoef(a, b)[0, 1]
    cnt = 0
    for _ in range(n_perm):
        ph = np.exp(2j * np.pi * rng.random(len(B)))
        ph[0] = 1.0
        if n % 2 == 0:
            ph[-1] = ph[-1].real + 1j * 0  # 保 Nyquist 实
        b_perm = np.fft.irfft(B * ph, n=n) + b.mean()
        if b_perm.std() < 1e-9:
            continue
        r = np.corrcoef(a, b_perm)[0, 1]
        if abs(r) >= abs(obs):
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="/workspace/mig_s")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123, 2024])
    ap.add_argument("--period", type=int, default=6000)
    ap.add_argument("--out", default="/workspace/migration_result.png")
    ap.add_argument("--json-out", default="/workspace/migration_verdict.json")
    args = ap.parse_args()

    verdicts = []
    n = len(args.seeds)
    fig, axes = plt.subplots(n + 1, 1, figsize=(11, 2.8 * (n + 1)), sharex=False)

    for i, s in enumerate(args.seeds):
        js = f"{args.prefix}{s}.summary.json"
        if not os.path.exists(js):
            axes[i].text(0.5, 0.5, f"seed {s}: missing", ha="center",
                         transform=axes[i].transAxes)
            continue
        d = json.load(open(js))
        t = np.array([float(x) for x in d.get("centroid_series", [])])  # dummy len
        cl = np.array([float(x) for x in d["centroid_series"] if x != ""])
        be = np.array([float(x) for x in d["band_env_series"] if x != ""])
        br = np.array([float(x) for x in d["band_res_series"] if x != ""])
        N = d["N_series"]
        ticks = np.arange(1, len(cl) + 1) * (d["ticks_target"] // max(len(cl), 1))

        ax = axes[i]
        ax.plot(ticks, be, color="#c0392b", lw=1.4, label="band_env (illum)" if not _CJK else "光照带 band_env")
        ax.plot(ticks, cl, color="#2c6fbb", lw=1.6, label="centroid (pop)" if not _CJK else "种群质心 centroid")
        ax.plot(ticks, br, color="#27ae60", lw=1.2, ls="--", label="band_res (stock)" if not _CJK else "资源存量带 band_res")
        ax.axhline(0, color="0.7", lw=0.6, ls=":")
        # 周期分隔线
        for k in range(1, 4):
            ax.axvline(k * args.period, color="0.85", lw=0.8)
        ax.set_ylabel(f"seed {s}\nLat(deg)" if not _CJK else f"seed {s}\n纬度(°)", fontsize=9)
        se = d["centroid_vs_band_env"]
        sr = d["centroid_vs_band_res"]
        ax.set_title(f"seed {s}: N={d['final_N']} | vs_env r0={se['corr0']} best={se['best_corr']}@{se['best_lag']} | "
                     f"vs_res r0={sr['corr0']} best={sr['best_corr']}@{sr['best_lag']}",
                     fontsize=8, loc="left")
        ax.legend(fontsize=7, ncol=3, loc="upper right")

        # 显著性
        p_env = None
        if len(cl) >= 8 and cl.std() > 1e-9 and be.std() > 1e-9:
            _, p_env = phase_null_pvalue(cl, be, seed=s)
        verdicts.append({
            "seed": s, "final_N": d["final_N"],
            "vs_env_corr0": se["corr0"], "vs_env_best": se["best_corr"],
            "vs_env_lag": se["best_lag"], "vs_env_p": None if p_env is None else round(p_env, 4),
            "vs_res_corr0": sr["corr0"], "vs_res_best": sr["best_corr"], "vs_res_lag": sr["best_lag"],
        })

    # 末图：lag 曲线（⚠️ x 轴必须是 lag 索引，不是 tick）
    ax = axes[-1]
    for s in args.seeds:
        js = f"{args.prefix}{s}.summary.json"
        if not os.path.exists(js):
            continue
        d = json.load(open(js))
        for key, sty, lab in (("centroid_vs_band_env", "-o", "vs band_env"),
                              ("centroid_vs_band_res", "--s", "vs band_res")):
            cur = d.get(key, {}).get("curve", {})
            if cur:
                # 归一化 key（json 可能把 int key 存成 str）
                cc = {(int(k) if str(k).lstrip("-").isdigit() else k): v
                      for k, v in cur.items()}
                xs = sorted(k for k in cc if isinstance(k, int))
                ys = [cc[x] for x in xs]
                ax.plot(xs, ys, sty, ms=4, lw=1.2, label=f"seed {s} {lab}")
    ax.axvline(0, color="0.7", lw=0.7, ls=":")
    ax.axhline(0, color="0.6", lw=0.7, ls=":")
    ax.axhline(0.5, color="0.5", lw=0.6, ls="--")
    ax.set_xlabel("lag (in 250-tick log steps)" if not _CJK else "滞后（250 tick 为一格）")
    ax.set_ylabel("corr")
    ax.legend(fontsize=6, ncol=2)
    ax.set_title("Lag correlation curves" if not _CJK else "滞后相关曲线（正 lag = 质心滞后于带）", fontsize=9)

    fig.suptitle("Seasonal migration: population centroid vs seasonal band"
                 if not _CJK else "季节迁徙：种群质心 vs 季节带（tilt=23.44°, P=6000, temp_sens=6）",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(args.out, dpi=130)
    json.dump(verdicts, open(args.json_out, "w"), indent=2, ensure_ascii=False)

    print("=== 迁徙判读汇总 ===")
    for v in verdicts:
        sig = "显著" if (v["vs_env_p"] is not None and v["vs_env_p"] < 0.05) else "不显著"
        print(f"seed {v['seed']:>4}: N={v['final_N']:>5} | "
              f"vs_env r0={v['vs_env_corr0']} best={v['vs_env_best']}@lag{v['vs_env_lag']} p={v['vs_env_p']} [{sig}] | "
              f"vs_res r0={v['vs_res_corr0']} best={v['vs_res_best']}@lag{v['vs_res_lag']}")
    print(f"\nsaved {args.out} , {args.json_out}")


if __name__ == "__main__":
    main()
