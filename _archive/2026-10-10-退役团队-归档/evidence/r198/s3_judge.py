#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""R198 / 13.8 S3 判读：按设计稿 §5 预注册判据（**不得自改**，R194 B8）。

判据原文（§5.1）：
  主 D_mig = 迁徙臂内，**个体级半季净纬向位移** Δφ_i 与 g23_i 的**个体级 ρ**，
             按 seed 分层后合并，**显著为正**。
              🔴 必须个体级，**禁用种群质心**（R196 教训：质心往返可被原地更替伪造）。
  支持 1 = 迁徙臂个体级位移幅度 > 空臂 A（有季节无迁徙），配对 seed。
  支持 2 = 高 g23 组的半季位移符号 与 sign(δ) 的一致率 > 0.5。
  第四态 = 观测 tick < 3×season_period ⇒ 判据不可执行（不判）。

"半季净纬向位移"口径（照 §5.1 字面）：
  · 半季窗口 = season_period/2 = 3000 tick；
  · **净**位移 = 窗口**首尾**的 |φ| 差 —— **不是**逐 40-tick Δ 的累加（后者被
    随机游走噪声稀释：S2 单 seed 短窗只读到 ρ≈0.001~0.014 即此因）；
  · 配对 = 窗口首存活 _id ∩ 窗口末存活 _id（引擎原生 id；**禁槽位键控**，R196 血泪）；
  · 18000 tick ⇒ 5 个不重叠半季窗口 ⇒ 5× 样本量。

为什么**在进程内重跑**而非读批产 CSV（据实说明）：
  · 判据要的是**个体级** `_id` + `g23` + `|φ|` 的三元配对；
  · `a4_verify_capacity.py` 落盘的 CSV 是**种群级聚合**（存活/均龄等），
    **不含**逐个体 id/基因/纬度 ⇒ 拿它算个体级 ρ 只能靠**槽位键控**伪造，
    正是 R196 教训禁止的做法；
  · ⇒ 个体级判据**必须**在进程内采快照。批产 CSV 仍照常落盘（供种群级
    `alive/世代/g23 均值` 等 §5.2 读数与归档），二者**同一随机种子、同一 preset 参数**，
    口径一致（`max_count=3240 / tilt=23.44 / season_period=6000`）。
"""
from __future__ import annotations

import sys
import os
import json
import math
import numpy as np

sys.path.insert(0, "/tmp/tw")
sys.path.insert(0, "/workspace/r198")
from simulation.config import SimConfig              # noqa: E402
from simulation.sphere_engine import SphereEngine    # noqa: E402
from simulation.genes import Gene                    # noqa: E402
from s2_smoke import lat_of, spearman                 # noqa: E402

SEASON = 6000
HALF = SEASON // 2          # 3000 tick = 半季
TICKS = 18000
SEEDS = [42, 7, 11, 13, 17, 23]
ARMS = {
    "mig_base":     dict(mig=False, gain=0.0,  season=True),
    "mig_g":        dict(mig=True,  gain=20.0, season=True),
    "mig_2g":       dict(mig=True,  gain=40.0, season=True),
    "mig_noseason": dict(mig=False, gain=0.0,  season=False),
}
TILT = float(np.deg2rad(23.44))


def run_arm(seed, mig, gain, season=True, ticks=TICKS, cap=None):
    """跑一个 run，返回**逐半季窗口**的明细（口径见模块 docstring）。

    `cap`：种群上限。**默认 None ⇒ 用 3240**（与 S3 preset 同口径）。
    §5.1 判据对种群规模**无要求**（只要求个体级配对 + `ticks ≥ 3×season_period`），
    而引擎步进成本随种群近似线性 ⇒ 判读可用较小 `cap` 换取墙钟（见 S3 报告 §5.5）。
    """
    c = SimConfig()
    c.seed = seed
    # 🔴 与 S3 preset 同口径（`max-count=3240`）：不锁上限 ⇒ 种群冲到 ~5000
    #    ⇒ 样本量爆炸 + 判定超时（R198 实测教训）。
    c.population.max_count = int(cap) if cap else 3240
    c.light.tilt_rad = TILT if season else 0.0
    c.light.season_period = SEASON if season else 0
    c.migration.enabled = bool(mig)
    c.migration.gain = float(gain)
    e = SphereEngine(c)

    snaps = {}
    for _ in range(ticks):
        e.step()
        t = e.tick
        if t % HALF == 0:
            ids = e._id.copy()
            snaps[t] = (ids, lat_of(e),
                        e._genes[:len(ids), Gene.MIGRATE_BIAS].copy())

    windows = []
    ks = sorted(snaps)
    for a, b in zip(ks[:-1], ks[1:]):
        ia, la, ga = snaps[a]
        ib, lb, _ = snaps[b]
        common, i0, i1 = np.intersect1d(ia, ib, return_indices=True)
        if len(common) < 5:
            continue
        windows.append({
            "t_start": int(a), "t_end": int(b), "n": int(len(common)),
            "D": lb[i1] - la[i0],        # 半季**净**纬向位移（|φ| 差）
            "G": ga[i0],                 # 窗口起始时刻的 g23
        })
    return {
        "windows": windows,
        "alive_end": e.alive_count(),
        "mig_probe": e.migration_probe(),
    }


def delta_sign(tick, period=SEASON):
    """sign(δ(t))，δ(t)=tilt·sin(2πt/P)。支持 2 用（追随夏天）。"""
    d = math.sin(2.0 * math.pi * (tick % period) / period)
    return 0 if abs(d) < 1e-9 else (1 if d > 0 else -1)


def _cat(runs, key):
    xs = [w[key] for r in runs for w in r["windows"]]
    return np.concatenate(xs) if xs else np.zeros(0)


def judge():
    rep = {"stage": "S3", "season_period": SEASON, "half_window": HALF,
           "ticks": TICKS, "seeds": SEEDS, "tilt_deg": 23.44,
           "criteria": "设计稿 §5.1（预注册，未修改；B8）",
           "forbid": "种群质心（R196 教训：质心往返可被原地更替伪造）"}

    # 第四态：观测长度不足 ⇒ 不判
    if TICKS < 3 * SEASON:
        rep["verdict"] = "第四态「纪元不足·判据不可执行」"
        with open("/workspace/r198/s3_judge.json", "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=2, default=str)
        return rep

    res = {}
    for arm, kw in ARMS.items():
        per = {}
        for s in SEEDS:
            per[s] = run_arm(s, **kw)
            nw = len(per[s]["windows"])
            nn = sum(w["n"] for w in per[s]["windows"])
            print(f"  [{arm:12s}] seed={s:3d} 窗口={nw} 样本={nn:5d} "
                  f"alive={per[s]['alive_end']}", flush=True)
        res[arm] = per
        D, G = _cat([per[s] for s in SEEDS], "D"), _cat([per[s] for s in SEEDS], "G")
        rho, p = (spearman(D, G) if len(D) > 30 else (None, None))
        # seed 分层 ρ（推断单位 = 世界种子 ⇒ 报分布，不做个体级伪重复检验）
        per_rho = []
        for s in SEEDS:
            d, g = _cat([per[s]], "D"), _cat([per[s]], "G")
            if len(d) >= 30:
                r, _ = spearman(d, g)
                if np.isfinite(r):
                    per_rho.append(float(r))
        rep.setdefault("arms", {})[arm] = {
            "n": int(len(D)),
            "dphi_abs_mean": float(np.abs(D).mean()) if len(D) else None,
            "dphi_mean": float(D.mean()) if len(D) else None,
            "rho_merged": rho, "p_merged": p,
            "per_seed_rho": per_rho,
            "per_seed_rho_mean": (float(np.mean(per_rho)) if per_rho else None),
            "per_seed_positive": int(sum(1 for x in per_rho if x > 0)),
            "per_seed_n": len(per_rho),
            "alive_end": [per[s]["alive_end"] for s in SEEDS],
        }
        a = rep["arms"][arm]
        print(f"  → {arm:12s} n={len(D):6d} |Δφ|={a['dphi_abs_mean']} "
              f"ρ={rho if rho is None else round(rho, 4)} "
              f"逐seed正号={a['per_seed_positive']}/{a['per_seed_n']}", flush=True)

    # ---- 主判据：迁徙臂 ρ 显著为正 ----
    def main_pass(arm):
        a = rep["arms"][arm]
        if a["rho_merged"] is None:
            return False
        return bool(a["rho_merged"] > 0
                    and (a["p_merged"] or 1.0) < 0.05
                    and a["per_seed_positive"] > a["per_seed_n"] / 2)

    # ---- 支持 1：迁徙臂位移幅度 > 空臂 A（配对 seed）----
    def sup1(arm):
        w, t, det = 0, 0, []
        for s in SEEDS:
            a = _cat([res[arm][s]], "D")
            b = _cat([res["mig_base"][s]], "D")
            if len(a) and len(b):
                t += 1
                win = np.abs(a).mean() > np.abs(b).mean()
                w += int(win)
                det.append({"seed": s, "arm_abs": float(np.abs(a).mean()),
                            "base_abs": float(np.abs(b).mean()), "win": bool(win)})
        return f"{w}/{t}", (t > 0 and w > t / 2), det

    # ---- 支持 2：高 g23 组半季位移符号 与 sign(δ) 一致率 > 0.5 ----
    def sup2(arm):
        hit, tot, det = 0, 0, []
        for s in SEEDS:
            for w in res[arm][s]["windows"]:
                sgn = delta_sign(w["t_end"])
                if sgn == 0:
                    continue
                hi = w["G"] > np.median(w["G"])
                if not hi.any():
                    continue
                sh = np.sign(w["D"][hi])
                agree = int(np.sum(sh == sgn))
                dis = int(np.sum(sh == -sgn))
                tot += 1
                if agree > dis:
                    hit += 1
                det.append({"seed": s, "t_end": w["t_end"], "sign_delta": sgn,
                            "agree": agree, "disagree": dis, "n_hi": int(hi.sum())})
        return f"{hit}/{tot}", (tot > 0 and hit > tot / 2), det

    for g, arm in (("g20", "mig_g"), ("g40", "mig_2g")):
        rep[f"main_{g}"] = main_pass(arm)
        s1s, s1p, s1d = sup1(arm)
        s2s, s2p, s2d = sup2(arm)
        rep[f"support1_{g}"] = s1s
        rep[f"support1_{g}_pass"] = s1p
        rep[f"support2_{g}"] = s2s
        rep[f"support2_{g}_pass"] = s2p
        rep[f"support1_{g}_detail"] = s1d
        rep[f"support2_{g}_detail"] = s2d
        # ---- 结论（§5.1 表）----
        if rep[f"main_{g}"] and s1p and s2p:
            v = "定向迁徙成立"
        elif rep[f"main_{g}"] and not (s1p and s2p):
            fails = []
            if not s1p:
                fails.append("支持1(幅度>空臂)")
            if not s2p:
                fails.append("支持2(方向一致率>0.5)")
            v = "证据不完整 —— 主判据过，但 " + "、".join(fails) + " 不过"
        else:
            v = "未成立"
        rep[f"verdict_{g}"] = v
        print(f"\n  【{g}】主={rep[f'main_{g}']} S1={s1s}({s1p}) S2={s2s}({s2p})")
        print(f"        ⇒ {v}\n", flush=True)

    os.makedirs("/workspace/r198", exist_ok=True)
    with open("/workspace/r198/s3_judge.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=str)
    print("已写 /workspace/r198/s3_judge.json")
    return rep


if __name__ == "__main__":
    judge()
