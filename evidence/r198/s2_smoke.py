#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""R198 / 13.8 S2 冒烟：P1-P7 前置条件 + 3 档 gain 量级扫描。

口径要点（照抄 R196 的教训，不重犯）：
  * 个体配对**必须**用引擎原生 `_id`（槽位会被死亡压缩重排 ⇒ 禁止槽位键控）。
  * 采样用**相邻 tick 配对**（np.intersect1d on _id），不要求"全程存活"
    （R196 indiv_ctrl.py v1 的死因：要求 lifespan > 全程 ⇒ 样本 0）。
  * 主判据用 **个体级** ρ(Δ|φ|, g23)；**禁用种群质心**（R197 已证质心往返 ≠ 个体迁徙）。
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, "/tmp/tw")
from simulation.config import SimConfig            # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402
from simulation.genes import Gene                  # noqa: E402

TILT = 23.44 * np.pi / 180.0
SEASON = 6000


def build(seed, mig, gain, min_abs=0.0, use_core=False, max_count=3240):
    """构造与 a4 preset **同配置**的引擎。

    🔴 `max_count=3240` 必须与 S3 preset 的 `max-count=3240` 一致：
       默认 `SimConfig()` 无上限 ⇒ 种群冲到 ~5000 ⇒ 每 seed 样本量 ~40 万
       ⇒ 多 seed 判定会跑到超时（R198 实测：6 seed × 6000 tick 跑了 24 min 未完）。
       锁上限后既与正式批口径一致，又让判定在分钟内出结果。
    """
    cfg = SimConfig()
    cfg.seed = seed
    cfg.population.max_count = int(max_count)
    cfg.light.tilt_rad = TILT
    cfg.light.season_period = SEASON
    cfg.migration.enabled = mig
    cfg.migration.gain = gain
    cfg.migration.min_abs_anomaly = min_abs
    cfg.simulation.use_sim_core = use_core
    return cfg, SphereEngine(cfg)


def lat_of(eng):
    """个体所在格 |φ|（弧度）——用引擎预计算表，与机制同口径。"""
    return eng._lat_abs[eng._flat]


# ---------------------------------------------------------------- P1-P7
def preconditions(seed=1, gain=20.0):
    _, eng = build(seed, True, gain)
    lt, w = eng.light, eng.world
    cells = np.arange(w.n_cells, dtype=np.int64)
    out = {}

    # P1 δ(t) 在跑：取 4 个相位，看日长是否真的随 t 变
    pp = {t: lt.photoperiod(cells, t) for t in (0, 1500, 3000, 4500)}
    out["P1_delta_running"] = bool(
        pp[1500].max() - pp[1500].min() > 0.9
        and pp[4500].max() - pp[4500].min() > 0.9
        and abs(pp[0].max() - 0.5) < 1e-9            # t=0 ⇒ δ=0 ⇒ P≡0.5
    )
    out["P1_detail"] = {str(t): [round(float(v.min()), 6), round(float(v.max()), 6)]
                        for t, v in pp.items()}

    # P2 纬度结构：t=1500 时 N/S 均值应**异号**（且关于 0.5 对称）
    lat_rows = w.latitude_of(np.arange(w.rows, dtype=np.int64))
    p15 = pp[1500].reshape(w.rows, w.cols)[:, 0]
    N = float(p15[lat_rows > 0].mean())
    S = float(p15[lat_rows < 0].mean())
    out["P2_lat_structure"] = bool(N > 0.5 > S)
    out["P2_detail"] = {"north_mean": round(N, 6), "south_mean": round(S, 6),
                        "sum_minus_1": round(N + S - 1.0, 9)}

    # P3 无 NaN / 越界（全域、全周期抽样）
    bad = 0
    rng = 0.0, 1.0
    for t in range(0, SEASON + 1, SEASON // 24):
        v = lt.photoperiod(cells, t)
        if not np.isfinite(v).all():
            bad += 1
        if (v < 0.0).any() or (v > 1.0).any():
            bad += 1
    out["P3_no_nan_or_range"] = bool(bad == 0)
    out["P3_checked_ticks"] = len(range(0, SEASON + 1, SEASON // 24))

    # P4/P5/P7：跑一小段拿引擎读数
    for _ in range(200):
        eng.step()
    pr = eng.migration_probe()
    out["P4_zero_frac_lt_half"] = bool(pr["mig_zero_frac"] < 0.5)
    out["P5_flat_frac_clearly_lt_1"] = bool(pr["mig_flat_frac"] < 0.9)
    out["P7_gene_slot_23"] = bool(pr["migrate_gene_slot"] == 23)
    out["engine_detail"] = {k: pr[k] for k in
                            ("mig_flat_frac", "mig_zero_frac", "mig_skip_frac",
                             "mig_term_abs_mean", "mig_dec_n")}

    # P6 量级非僭越：觅食项上界 = perc · fr · 0.5（fr≤1）⇒ 上界 ≈ perc·0.5
    perc = float(eng._genes[:len(eng._id), Gene.PERCEPTION].mean())
    out["P6_forage_scale"] = round(perc * 0.5, 6)
    out["P6_mig_abs_mean"] = pr["mig_term_abs_mean"]
    out["P6_ratio_vs_forage"] = (round(pr["mig_term_abs_mean"] / (perc * 0.5), 6)
                                 if pr["mig_term_abs_mean"] is not None and perc else None)
    out["P6_not_usurping"] = bool(
        pr["mig_term_abs_mean"] is not None and pr["mig_term_abs_mean"] <= perc * 0.5)
    return out


def _rankdata(a):
    """平均秩（处理并列；与 scipy.stats.rankdata 同语义）。纯 numpy，无外部依赖。"""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman ρ + 双侧 p（正态近似，含并列校正）。

    🔴 自实现而非依赖 scipy：本沙箱 scipy 1.18 要求 numpy≥2，而引擎编译
       （sim_core.so）绑 numpy 1.26 ⇒ 装 scipy 会连带毁掉 numpy ⇒ 引擎跑不动。
       同一算法 10 行可写完 ⇒ **不为一个相关系数引入整条依赖链**（少一个坑）。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    sx = float(np.sqrt(np.sum(rx * rx)))
    sy = float(np.sqrt(np.sum(ry * ry)))
    if sx == 0.0 or sy == 0.0:
        return float("nan"), float("nan")
    rho = float(np.sum(rx * ry) / (sx * sy))
    if abs(rho) >= 1.0:
        return rho, 0.0
    # t 近似（df=n-2）：t = rho·sqrt((n-2)/(1-rho²))
    t = rho * np.sqrt((n - 2) / (1.0 - rho * rho))
    # 双侧 p：用 Student-t 的连分式太久 ⇒ 正态近似（n 大时足够；批内自查用）
    from math import erfc, sqrt
    z = t * sqrt((n - 1.0) / 1.0) if n > 2 else t
    p = erfc(abs(z) / sqrt(2.0))
    return rho, float(p)


# ---------------------------------------------------------------- 主判据
def individual_rho(seed, mig, gain, ticks=6000, sample_every=50):
    """个体级 ρ(Δ|φ|, g23)：**相邻采样配对**（不要求全程存活）。

    返回 (rho, p, n, detail)。**禁用种群质心**（R197 教训）。
    """
    _, eng = build(seed, mig, gain)
    prev_id, prev_lat, prev_g = None, None, None
    dphi_all, g_all = [], []
    for _ in range(ticks):
        eng.step()
        if eng.tick % sample_every:
            continue
        ids = eng._id.copy()
        lat = lat_of(eng)
        # ⚠️ `_genes` 与 `_id` **同一压缩节拍**（同一 keep 掩码）⇒ 行号 i ↔ _id[i]。
        #    绝不能用 `_genes[id]`（id≠行号，会越界）。
        g23 = eng._genes[:len(ids), Gene.MIGRATE_BIAS].copy()
        if prev_id is not None:
            common, i0, i1 = np.intersect1d(prev_id, ids, return_indices=True)
            if len(common):
                dphi_all.append(lat[i1] - prev_lat[i0])
                g_all.append(prev_g[i0])
        prev_id, prev_lat, prev_g = ids, lat, g23
    if not dphi_all:
        return None
    dphi = np.concatenate(dphi_all)
    g23 = np.concatenate(g_all)
    if len(dphi) < 30:
        return {"rho": None, "p": None, "n": int(len(dphi)),
                "note": "样本不足（种群过小/配对失败）"}
    rho, p = spearman(dphi, g23)
    hi = g23 > np.median(g23)
    return {
        "rho": float(rho), "p": float(p), "n": int(len(dphi)),
        "dphi_mean": float(dphi.mean()), "dphi_std": float(dphi.std()),
        "g23_mean": float(g23.mean()),
        "dphi_hi_g23": float(dphi[hi].mean()),
        "dphi_lo_g23": float(dphi[~hi].mean()),
        "alive_end": int(eng.alive_count()),
    }


def main():
    rep = {"stage": "S2", "tilt_deg": 23.44, "season_period": SEASON}

    print("=" * 72)
    print("P1–P7 前置条件（seed=1, gain=20）")
    print("=" * 72)
    rep["preconditions"] = preconditions(seed=1, gain=20.0)
    for k, v in rep["preconditions"].items():
        print(f"  {k:28s} = {v}")

    print()
    print("=" * 72)
    print("3 档 gain 量级扫描（seed=7, 4000 tick, 个体级 ρ）")
    print("=" * 72)
    rep["gain_scan"] = {}
    for gain in (2.0, 20.0, 50.0):
        r = individual_rho(7, True, gain, ticks=4000, sample_every=40)
        rep["gain_scan"][str(gain)] = r
        print(f"  gain={gain:5.1f} -> {r}")

    os.makedirs("/workspace/r198", exist_ok=True)
    with open("/workspace/r198/s2_smoke.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print("\n已写 /workspace/r198/s2_smoke.json")


if __name__ == "__main__":
    main()
