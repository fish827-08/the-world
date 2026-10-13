#!/usr/bin/env python3.12
"""迁徙判定 v3 —— 双检验（相位零分布 + 滞后对齐）多 seed 汇总。

为什么需要双检验（本脚本的存在理由）
------------------------------------
季节迁徙 = **相位锁定振荡**，不是简单相关。裸 corr 会踩两个坑：
  ① 半周期假阳性：只跑半个周期时质心与带的相关系数虚高（R192 已记录 r=0.997 假阳）
  ② 相位零分布检验在 n 小时**检验力不足**：质心是近完美正弦
     （autocorr@半周期≈-1），只有一个自由参数（相位）。
     相位随机化恰好专门打掉这一频率对齐 ⇒ 即使真实关系 r=0.99，
     n=49 时也算不出 p<0.05（"检验力不足" ≠ "无关系"）。

⇒ 判据必须同时报告：
  A. 相位零分布 p（保守，防假阳）
  B. 滞后对齐 best-lag corr（灵敏，抓真阳）
  C. 半周期反号 autocorr（**季节性迁徙的定义性特征**：往返）
  D. 主周期一致性（centroid 与 band 的主周期是否同为 1 个季节周期）
  E. 同号率 z（质心与带同处一个半球的频率是否高于随机）

三态判定：
  有迁徙        : D 成立 ∧ (C≈-1 往返) ∧ (B 高 ∧ A 显著)
  有迁徙（锁定）: D 成立 ∧ C≈-1 ∧ B 高 ∧ A 不显著（零滞后检验不适配带滞后锁定）
  无迁徙        : D 不成立 或 C 不反号 或 B 低
"""
import sys
import csv
import numpy as np


# 🔴 R98 纪律（F-R14/F-R15 同族）：中文 Windows 默认 **GBK** 控制台下，print 里的
#    emoji / 箭头（`⇒` `✅` `❌` 等）会抛 UnicodeEncodeError ⇒ 脚本 **rc=1 假失败**，
#    把批次退出码搅坏（数据其实无损）。入口强制 UTF-8，`errors="replace"` 兜底，
#    **绝不因编码丢结果**。守卫测试：`tests/test_r98_nonascii_print.py`。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def load(path, warm=6000.0):
    rows = [r for r in csv.DictReader(open(path)) if float(r["tick"]) >= warm]
    if len(rows) < 8:
        return None
    return {
        "tick": np.array([float(r["tick"]) for r in rows]),
        "N": np.array([int(r["N"]) for r in rows]),
        "cen": np.array([float(r["centroid_lat_deg"]) for r in rows]),
        "bres": np.array([float(r["band_res_deg"]) for r in rows]),
        "benv": np.array([float(r["band_env_deg"]) for r in rows]),
    }


def phase_null_pvalue(a, b, n_perm=2000, seed=0):
    """相位随机化零分布（保守检验）。"""
    rng = np.random.default_rng(seed)
    n = len(b)
    B = np.fft.rfft(b - b.mean())
    obs = float(np.corrcoef(a, b)[0, 1])
    cnt = 0
    for _ in range(n_perm):
        ph = np.exp(2j * np.pi * rng.random(len(B)))
        ph[0] = 1.0
        if n % 2 == 0:
            ph[-1] = ph[-1].real
        bp = np.fft.irfft(B * ph, n=n)
        if bp.std() < 1e-9:
            continue
        if abs(float(np.corrcoef(a, bp)[0, 1])) >= abs(obs):
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def best_lag_corr(a, b, max_lag):
    """在 ±max_lag 内扫 corr(a[t+lag], b[t])，返回 (best_r, best_lag)。"""
    best, bl = -2.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[lag:], b[:len(b) - lag]
        else:
            x, y = a[:lag], b[-lag:]
        if len(x) < 4 or x.std() < 1e-9 or y.std() < 1e-9:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        if c > best:
            best, bl = c, lag
    return (round(best, 4), bl)


def dominant_period_samples(x):
    xf = np.fft.rfft(x - x.mean())
    amp = np.abs(xf)
    k = int(np.argmax(amp[1:])) + 1
    return len(x) / k if k else float("nan")


def verdict_one(path, warm=6000.0, period=6000.0):
    d = load(path, warm)
    if d is None:
        return None
    cen, benv, bres = d["cen"], d["benv"], d["bres"]
    n = len(cen)
    # 采样间隔 → 半周期点数
    dt = float(np.median(np.diff(d["tick"])))
    half_pts = int(round((period / 2.0) / dt))
    per_pts = int(round(period / dt))

    # C. 半周期反号
    if 2 <= half_pts < n:
        ac_half = float(np.corrcoef(cen[:-half_pts], cen[half_pts:])[0, 1])
    else:
        ac_half = float("nan")
    if 2 <= per_pts < n:
        ac_full = float(np.corrcoef(cen[:-per_pts], cen[per_pts:])[0, 1])
    else:
        ac_full = float("nan")

    # D. 主周期
    pc, pe = dominant_period_samples(cen), dominant_period_samples(benv)

    # B. 滞后对齐
    bl_env, lag_env = best_lag_corr(cen, benv, half_pts)
    bl_res, lag_res = best_lag_corr(cen, bres, half_pts)

    # A. 相位零分布（零滞后，保守）
    _, p_env = phase_null_pvalue(cen, benv, seed=1)
    _, p_res = phase_null_pvalue(cen, bres, seed=1)
    # A′. 零滞后 corr（= 相位零分布检验的真正输入；跨窗口稳定，见 power_diagnosis.py）
    c0_env = float(np.corrcoef(cen, benv)[0, 1])
    c0_res = float(np.corrcoef(cen, bres)[0, 1])

    # E. 同号率
    same_env = float(np.mean(np.sign(cen) == np.sign(benv)))
    z_env = (same_env - 0.5) * 2 * np.sqrt(n)

    d.update(dict(n=n, half_pts=half_pts, per_pts=per_pts,
                  ac_half=ac_half, ac_full=ac_full,
                  period_cen=pc, period_env=pe,
                  bl_env=bl_env, lag_env=lag_env, bl_res=bl_res, lag_res=lag_res,
                  p_env=p_env, p_res=p_res, c0_env=c0_env, c0_res=c0_res,
                  same_env=same_env, z_env=z_env,
                  cen_amp=float(cen.max() - cen.min()),
                  env_amp=float(benv.max() - benv.min()),
                  res_amp=float(bres.max() - bres.min())))
    return d


def classify(r):
    """三态判定（见模块 docstring）。"""
    if not (np.isfinite(r["ac_half"]) and r["ac_half"] < -0.5):
        return "无迁徙（质心未往返）"
    period_ok = (abs(r["period_cen"] - r["period_env"]) / max(r["period_env"], 1e-9) < 0.35)
    if not period_ok:
        return "无迁徙（主周期不一致）"
    if r["bl_env"] >= 0.7 and r["p_env"] < 0.05:
        return "有迁徙（显著）"
    if r["bl_env"] >= 0.7:
        return "有迁徙（相位锁定；零滞后检验不适配）"
    return "无迁徙（相关性低）"


def main():
    ap_seeds = sys.argv[1:]
    print("=" * 78)
    print("迁徙判定 v3 —— 双检验多 seed 汇总")
    print("=" * 78)
    results = []
    for path in ap_seeds:
        r = verdict_one(path)
        if r is None:
            print(f"[skip] {path}（稳态段样本不足）")
            continue
        v = classify(r)
        results.append((path, r, v))
        print(f"\n{path}")
        print(f"  稳态 n={r['n']}  (半周期 {r['half_pts']} 点 / 全周期 {r['per_pts']} 点)")
        print(f"  振幅: centroid {r['cen_amp']:.1f}° | band_env {r['env_amp']:.1f}° "
              f"| band_res {r['res_amp']:.1f}°")
        print(f"  [C] autocorr@半周期 = {r['ac_half']:+.3f}   @全周期 = {r['ac_full']:+.3f}")
        print(f"  [D] 主周期: centroid {r['period_cen']:.1f} 点 vs band_env {r['period_env']:.1f} 点")
        print(f"  [B] best-lag corr: band_env r={r['bl_env']:+.3f}@lag{r['lag_env']} | "
              f"band_res r={r['bl_res']:+.3f}@lag{r['lag_res']}")
        print(f"  [A] 零滞后 corr: band_env {r['c0_env']:+.3f} | band_res {r['c0_res']:+.3f}")
        print(f"     相位零分布 p: band_env {r['p_env']:.4f} | band_res {r['p_res']:.4f}"
              f"   (⚠️ 窄带信号下此 p 不稳定，见 power_diagnosis.py)")
        print(f"  [E] 同号率 vs band_env = {r['same_env']:.3f} (z={r['z_env']:+.2f})")
        print(f"  ⇒ 判定: {v}")

    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    from collections import Counter
    for path, r, v in results:
        print(f"  {path.split('/')[-1]:<20} cen_amp={r['cen_amp']:5.1f}°  "
              f"ac_half={r['ac_half']:+.3f}  corr={r['bl_env']:+.3f}  → {v}")
    print("\n  判定分布:", dict(Counter(v for _, _, v in results)))


if __name__ == "__main__":
    main()
