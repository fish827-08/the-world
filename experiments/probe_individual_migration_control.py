#!/usr/bin/env python3.12
"""个体级迁徙 —— 决定性对照（v2：相邻点配对法 + 三臂）

v1 失败原因：要求个体"全程 30 点全勤"⇒ 一个都没剩下（个体寿命 < 6000 tick）。
改为**相邻采样点配对**：只要某个体在 t_k 与 t_{k+1} 都存在，就计一次位移。
这正是"个体级迁徙"的正确度量，且样本量足够。

三臂：
  A. R196 光驱动再生 + 季节（待检验）
  B. 纯基线（光关 + 季节关）—— 测"随机游走本底"
  C. R195（光关 + 季节开）—— 食物带不动的对照

判据：
  ① |位移| 量级：A 是否显著高于 B？（若是随机游走本底，两者应相当）
  ② 定向效率 = |净位移| / 累计路径：随机游走 ≈ 1/√n，定向迁徙 ≈ 1
  ③ 与食物带移动的同向率：>0.5 且显著 ⇒ 跟随食物
"""
import sys
sys.path.insert(0, '/tmp/tw')
sys.path.insert(0, '/tmp/tw/experiments')
import importlib.util
import time
import numpy as np
from world.resource_field import ResourceField

_orig = ResourceField._regrowth_amount


def _patched(self, tick):
    g = _orig(self, tick)
    lam = getattr(self, "light_sensitivity", 0.0)
    if lam and lam > 0:
        ill = self.lt.illumination(np.arange(self.world.n_cells), tick)
        g = g * np.power(np.clip(ill, 0, 1), lam)
    return g


ResourceField._regrowth_amount = _patched

spec = importlib.util.spec_from_file_location(
    'a4', '/tmp/tw/experiments/a4_verify_capacity.py')
a4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a4)

DT = 250
N_SAMP = 24          # 24 × 250 = 6000 tick = 1 个完整周期
PREHEAT = 6000


def run(label, light_k, tilt):
    t_start = time.time()
    kw = {}
    if tilt is not None:
        kw = dict(tilt_deg=tilt, season_period=6000)
    if light_k:
        kw['light_sensitivity'] = light_k
    eng = a4.build("off", True, 42, 40000, **kw)
    for _ in range(PREHEAT):
        eng.step()

    def lat_of(flat):
        return np.rad2deg(eng.world.latitude_of(
            (flat // eng.world.cols).astype(np.int64)))

    r_idx = np.arange(eng.world.n_cells) // eng.world.cols
    la = np.rad2deg(eng.world.latitude_of(r_idx.astype(np.int64)))
    wc = np.cos(np.deg2rad(la))

    def band(v):
        ww = wc * np.maximum(v, 0.0)
        s = ww.sum()
        return float((ww * la).sum() / s) if s > 1e-12 else 0.0

    prev_ids = None
    prev_lat = None
    disp = []          # 相邻点位移（逐个体逐对）
    disp_bins = []
    net_first_last = {}   # id -> [首lat, 末lat]（用于定向效率，只取首末两次）
    cen, res, env = [], [], []

    for k in range(N_SAMP):
        for _ in range(DT):
            eng.step()
        ids = eng._id.copy()
        lat = lat_of(eng._flat.copy())
        cen.append(float(lat.mean()))
        res.append(band(eng.resources._grid.copy()))
        env.append(band(eng.light.illumination(
            np.arange(eng.world.n_cells), PREHEAT + (k + 1) * DT)))

        if prev_ids is not None:
            # 按 id 取交集（存在性配对，允许中间死亡/出生）
            common, ia, ib = np.intersect1d(
                prev_ids, ids, assume_unique=False, return_indices=True)
            d = lat[ib] - prev_lat[ia]
            disp.append(d)
            disp_bins.append(int(d.size))
            for j, i_id in enumerate(common.tolist()):
                if i_id in net_first_last:
                    net_first_last[i_id][1] = float(lat[ib[j]])
                else:
                    net_first_last[i_id] = [float(prev_lat[ia[j]]),
                                            float(lat[ib[j]])]
        prev_ids, prev_lat = ids, lat

    cen = np.array(cen); res = np.array(res); env = np.array(env)
    D = np.concatenate(disp) if disp else np.zeros(0)

    # 定向效率：用首末都出现的个体的净位移 / 路径
    # （路径用相邻位移按 id 累加，这里用简化下界：|净|/该个体出现次数*平均|步|）
    net = np.array([v[1] - v[0] for v in net_first_last.values()])
    cnt = np.array([len(v) for v in net_first_last.values()])

    print(f"\n【{label}】  快照配对总数 {D.size}（平均每次 {np.mean(disp_bins):.0f} 对）")
    print(f"  相邻点位移（每 {DT} tick）: 平均|Δ| = {np.abs(D).mean():.4f}°"
          f"  中位 {np.median(np.abs(D)):.4f}°  std {D.std():.4f}°")
    print(f"  向北比例 = {(D > 0).mean():.4f}  (0.5 = 无方向)")
    print(f"  首末净位移: 平均 |净| = {np.abs(net).mean():.3f}°"
          f"  均值 {net.mean():+.3f}°  std {net.std():.3f}°"
          f"  (n={net.size})")
    # 同向率 vs 食物带
    d_res = np.diff(res)
    agree = [(np.sign(dd) == np.sign(d_res[i])).mean()
             for i, dd in enumerate(disp)]
    agree = np.array(agree)
    tot = D.size
    z = (agree.mean() - 0.5) / (0.5 / np.sqrt(tot)) if tot else 0.0
    print(f"  与食物带同向率 = {agree.mean():.4f}  (z = {z:+.2f}, N={tot})")
    print(f"  群体质心摆幅 = {cen.max()-cen.min():.2f}°"
          f"  食物带摆幅 = {res.max()-res.min():.2f}°"
          f"  光照带摆幅 = {env.max()-env.min():.2f}°")
    print(f"  用时 {time.time()-t_start:.0f}s", flush=True)

    return dict(label=label, abs_disp=float(np.abs(D).mean()),
                med_disp=float(np.median(np.abs(D))),
                north=float((D > 0).mean()),
                net_abs=float(np.abs(net).mean()),
                net_mean=float(net.mean()), net_std=float(net.std()),
                agree=float(agree.mean()), z=float(z),
                cen_swing=float(cen.max() - cen.min()),
                res_swing=float(res.max() - res.min()),
                env_swing=float(env.max() - env.min()))


print("=" * 70)
print("个体级迁徙 —— 三臂对照（相邻点 id 配对, DT=250, 24 点 = 1 周期）")
print("=" * 70)
out = []
out.append(run("A. R196 光驱动 + 季节", 1.0, 23.44))
out.append(run("B. 纯基线（光关季节关）", None, None))
out.append(run("C. R195（光关 + 季节开）", None, 23.44))

print("\n" + "=" * 70)
print("对照汇总")
print("=" * 70)
hdr = f"{'臂':<24}{'|Δ|/点':>9}{'向北比':>8}{'|净|':>8}{'同向率':>8}{'z':>9}"
print(hdr); print("-" * len(hdr))
for r in out:
    print(f"{r['label']:<24}{r['abs_disp']:>9.4f}{r['north']:>8.4f}"
          f"{r['net_abs']:>8.3f}{r['agree']:>8.4f}{r['z']:>9.2f}")
print(f"\n随机游走理论向率 = 0.5；定向迁徙应 > 0.5 且 z 大正")

import json
with open('/tmp/ictrl_summary.json', 'w') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
