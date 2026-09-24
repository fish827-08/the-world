#!/usr/bin/env python3.12
"""个体级迁徙验证 v3 —— 按引擎原生稳定 ID（`_id`）配对，分段落盘。

v1 教训：用"数组长度相同 ⇒ 槽位可配对"判据是错的（同 tick 可同时死亡+出生，
长度持平但槽位已重排）⇒ 得到 52.5°/10tick 的物理不可能伪影。

v2 教训：12 分钟超时不够（种群 ~1900 时每 tick 代价远高于建群期）。

v3：跑满 2 周期稳态后，追踪 1 个完整周期，每 100 tick 采样（60 点），
    每 10 点落一次盘，保证部分结果可用。
"""
import sys
sys.path.insert(0, '/tmp/tw')
sys.path.insert(0, '/tmp/tw/experiments')
import importlib.util
import json
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

P_CYCLE = 6000
DT = 100
N_SAMP = P_CYCLE // DT      # 60 个采样点 / 1 周期

eng = a4.build("off", True, 42, 40000, tilt_deg=23.44, season_period=P_CYCLE)
print(f"世界 {eng.world.rows}x{eng.world.cols}  P={P_CYCLE}", flush=True)


def lat_of_flat(flat):
    rows = (flat // eng.world.cols).astype(np.int64)
    return np.rad2deg(eng.world.latitude_of(rows))


r_idx = np.arange(eng.world.n_cells) // eng.world.cols
lat_all = np.rad2deg(eng.world.latitude_of(r_idx.astype(np.int64)))
w_cos = np.cos(np.deg2rad(lat_all))


def band(v):
    ww = w_cos * np.maximum(v, 0.0)
    s = ww.sum()
    return float((ww * lat_all).sum() / s) if s > 1e-12 else 0.0


# ---- 稳态 2 周期（不采样，尽量快）----
t0 = time.time()
for t in range(2 * P_CYCLE):
    eng.step()
    if (t + 1) % P_CYCLE == 0:
        print(f"  预热 t={t+1} N={len(eng._id)} "
              f"({time.time()-t0:.0f}s)", flush=True)

# ---- 追踪 1 周期 ----
hist = {}
rows = []
for k in range(N_SAMP):
    for _ in range(DT):
        eng.step()
    t = 2 * P_CYCLE + (k + 1) * DT
    ids = eng._id.copy()
    lats = lat_of_flat(eng._flat.copy())
    for i, lat in zip(ids.tolist(), lats.tolist()):
        hist.setdefault(i, []).append((t, lat))
    rows.append((t, float(lats.mean()), len(ids),
                 band(eng.resources._grid.copy()),
                 band(eng.light.illumination(np.arange(eng.world.n_cells), t))))
    if (k + 1) % 10 == 0:
        print(f"  追踪 {k+1}/{N_SAMP}  t={t} N={len(ids)} "
              f"centroid={lats.mean():+.2f}", flush=True)
        with open('/tmp/iv3_traj.csv', 'w') as f:
            f.write("tick,centroid,band_res,band_env,N\n")
            for r in rows:
                f.write(f"{r[0]},{r[1]:.4f},{r[3]:.4f},{r[4]:.4f},{r[2]}\n")

t_arr = np.array([r[0] for r in rows])
cen = np.array([r[1] for r in rows])
nA = np.array([r[2] for r in rows])
res = np.array([r[3] for r in rows])
env = np.array([r[4] for r in rows])

print(f"\n{'='*62}")
print(f"采样 {len(t_arr)} 点 / 1 完整周期 (P={P_CYCLE})")
print(f"种群 N: {nA.min()} ~ {nA.max()}")

full = {i: v for i, v in hist.items() if len(v) == len(t_arr)}
print(f"全程存活个体 = {len(full)} / {len(hist)} 个曾出现")

summary = {"n_samples": int(len(t_arr)),
           "N": [int(nA.min()), int(nA.max())],
           "n_full_lived": len(full), "n_seen": len(hist),
           "centroid": [float(cen.min()), float(cen.max())],
           "band_res": [float(res.min()), float(res.max())],
           "band_env": [float(env.min()), float(env.max())],
           "corr_cen_res": float(np.corrcoef(cen, res)[0, 1]),
           "corr_cen_env": float(np.corrcoef(cen, env)[0, 1])}

if full:
    ids_full = np.array(list(full.keys()))
    M = np.array([[lat for _, lat in full[i]] for i in ids_full]).astype(np.float64)
    dM = np.diff(M, axis=1)
    absd = np.abs(dM)
    print(f"\n① 个体纬向位移（每 {DT} tick）:")
    print(f"   平均 |位移| = {absd.mean():.4f}°  (中位 {np.median(absd):.4f}°)")
    print(f"   位移 std    = {dM.std():.4f}°")
    print(f"   非零占比    = {(absd > 1e-9).mean():.3f}")
    print(f"   实测量级 vs 随机漫游对照见 ③")
    summary["indiv_abs_disp_mean"] = float(absd.mean())
    summary["indiv_abs_disp_median"] = float(np.median(absd))
    summary["indiv_disp_std"] = float(dM.std())

    h = len(t_arr) // 2
    d1 = M[:, h] - M[:, 0]
    d2 = M[:, -1] - M[:, h]
    flips = (np.sign(d1) * np.sign(d2) < 0)
    print(f"\n② 往返（半周期净位移方向）:")
    print(f"   前半净位移 均值 = {d1.mean():+.3f}°  (北向比 {(d1>0).mean():.3f})")
    print(f"   后半净位移 均值 = {d2.mean():+.3f}°  (北向比 {(d2>0).mean():.3f})")
    print(f"   掉头个体比例 = {flips.mean():.3f}   (0.5 = 无方向性)")
    print(f"   个体半周期净位移 |·| = {np.abs(d1).mean():.2f}° / "
          f"{np.abs(d2).mean():.2f}°")
    print(f"   群体质心半周期位移   = {cen[h]-cen[0]:+.2f}° / "
          f"{cen[-1]-cen[h]:+.2f}°")
    summary.update(flip_frac=float(flips.mean()),
                   d1_mean=float(d1.mean()), d2_mean=float(d2.mean()),
                   d1_abs_mean=float(np.abs(d1).mean()),
                   d2_abs_mean=float(np.abs(d2).mean()),
                   cen_d1=float(cen[h] - cen[0]),
                   cen_d2=float(cen[-1] - cen[h]))

    d_res = np.diff(res)
    agree = (np.sign(dM) == np.sign(d_res)[None, :])
    z = (agree.mean() - 0.5) / (0.5 / np.sqrt(agree.size))
    print(f"\n③ 个体位移方向 vs 食物带移动方向:")
    print(f"   同向比例 = {agree.mean():.3f}   (0.5 = 无关)")
    print(f"   同向显著性 z = {z:+.2f}")
    summary.update(agree_frac=float(agree.mean()), agree_z=float(z))

print(f"\n群体质心: {cen.min():.2f} ~ {cen.max():.2f}° (幅 {cen.max()-cen.min():.2f}°)")
print(f"食物带  : {res.min():.2f} ~ {res.max():.2f}° (幅 {res.max()-res.min():.2f}°)")
print(f"光照带  : {env.min():.2f} ~ {env.max():.2f}° (幅 {env.max()-env.min():.2f}°)")
print(f"零滞后 corr(质心,食物带) = {np.corrcoef(cen,res)[0,1]:+.4f}")
print(f"零滞后 corr(质心,光照带) = {np.corrcoef(cen,env)[0,1]:+.4f}")

np.save('/tmp/iv3_traj.npy', np.array(rows))
if full:
    np.save('/tmp/iv3_M.npy', M)
with open('/tmp/iv3_summary.json', 'w') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("\n写出 /tmp/iv3_traj.csv /tmp/iv3_summary.json")
