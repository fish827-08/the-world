#!/usr/bin/env python3.12
"""R196 光驱动再生 —— 迁徙涌现图（多 seed）。

出图：3 seed × 2 面板
  上：centroid_lat(t) 与 band_env(t) / band_res(t) 时间序列（稳态段）
  下：滞后对齐曲线 corr(centroid[t+lag], band[t])
另出对照：R195（季节只动光）vs R196（光驱动再生）的 band_res 摆幅对比。
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体回退：Noto CJK 的 .ttc 在不同系统上注册的主名可能是 JP/SC/TC。
# 注意：matplotlib 的字体缓存在 .ttc 字体首次未被扫描时会过期，
# 如遇方框乱码请先清缓存：rm ~/.cache/matplotlib/fontlist-*.json
_cjk = None
for _f in font_manager.fontManager.ttflist:
    if "CJK" in _f.name or "WenQuanYi" in _f.name:
        _cjk = _f.name
        break
if _cjk:
    plt.rcParams["font.sans-serif"] = [_cjk, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEEDS = [42, 123, 2024, 7]
BASE = os.path.dirname(os.path.abspath(__file__))   # 同目录 CSV
WARM = 6000.0


def load(path, warm=WARM):
    rows = [r for r in csv.DictReader(open(path)) if float(r["tick"]) >= warm]
    if len(rows) < 5:
        return None
    return dict(
        tick=np.array([float(r["tick"]) for r in rows]),
        cen=np.array([float(r["centroid_lat_deg"]) for r in rows]),
        bres=np.array([float(r["band_res_deg"]) for r in rows]),
        benv=np.array([float(r["band_env_deg"]) for r in rows]),
        N=np.array([int(r["N"]) for r in rows]),
    )


def lag_curve(a, b, max_lag):
    lags, cs = [], []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[lag:], b[:len(b) - lag]
        else:
            x, y = a[:lag], b[-lag:]
        if len(x) < 4 or x.std() < 1e-9 or y.std() < 1e-9:
            continue
        lags.append(lag)
        cs.append(float(np.corrcoef(x, y)[0, 1]))
    return np.array(lags), np.array(cs)


def main():
    series = {}
    for s in SEEDS:
        d = load(f"{BASE}/migLR_s{s}.csv")
        if d:
            series[s] = d
    keys = list(series)
    n = len(keys)
    if n == 0:
        print("no data"); return

    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 7.6), squeeze=False)

    for i, s in enumerate(keys):
        d = series[s]
        t = d["tick"]
        ax = axes[0][i]
        ax.plot(t, d["benv"], color="#d4a017", lw=1.6, ls="--", label="光照带 band_env")
        ax.plot(t, d["bres"], color="#2e8b57", lw=2.0, label="食物存量带 band_res")
        ax.plot(t, d["cen"], color="#c0392b", lw=2.2, label="种群质心 centroid")
        ax.axhline(0, color="k", lw=0.6, alpha=0.3)
        ax.set_title(f"seed {s}   质心振幅 "
                     f"{d['cen'].max()-d['cen'].min():.0f}°", fontsize=11)
        ax.set_xlabel("tick")
        ax.set_ylabel("纬度 (°)")
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=8, loc="lower left")
        # 标注半周期自相关
        dt = float(np.median(np.diff(t)))
        hp = int(round((6000 / 2) / dt))
        if 2 <= hp < len(d["cen"]):
            ac = float(np.corrcoef(d["cen"][:-hp], d["cen"][hp:])[0, 1])
            ax.text(0.02, 0.96, f"autocorr@半周期 = {ac:+.3f}",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(fc="white", ec="#888", alpha=0.85, boxstyle="round"))

        ax2 = axes[1][i]
        bl = max(4, hp)
        lags, cs_env = lag_curve(d["cen"], d["benv"], bl)
        _, cs_res = lag_curve(d["cen"], d["bres"], bl)
        ax2.plot(lags, cs_res, "o-", color="#2e8b57", lw=2, ms=4, label="vs 存量带")
        ax2.plot(lags, cs_env, "s-", color="#d4a017", lw=2, ms=4, label="vs 光照带")
        ax2.axhline(0, color="k", lw=0.6, alpha=0.3)
        best = int(lags[int(np.argmax(cs_env))])
        ax2.axvline(best, color="#c0392b", ls=":", lw=1.5)
        ax2.set_title(f"滞后对齐   峰值 {cs_env.max():+.3f} @ lag{best}", fontsize=11)
        ax2.set_xlabel("滞后 lag（采样点）")
        ax2.set_ylabel("相关系数 r")
        ax2.grid(alpha=0.25)
        if i == 0:
            ax2.legend(fontsize=8)

    fig.suptitle("R196 光驱动再生 —— 种群随季节南北往返（稳态段 t≥6000）",
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = f"{BASE}/r196_migration_emergence.png"
    fig.savefig(out, dpi=140)
    print("saved", out)


if __name__ == "__main__":
    main()
