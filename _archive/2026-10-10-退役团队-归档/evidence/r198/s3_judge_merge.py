#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""R198 / 13.8 S3 判读（合并器）：读 shards/ 下的 24 个 JSON 片段，按 §5.1 判。

§5.1 预注册判据（**不得自改**，B8）：
  主 D_mig = 迁徙臂内**个体级** ρ(半季净纬向位移 Δφ_i, g23_i) **显著为正**，按 seed 分层后合并。
            🔴 必须个体级，**禁用种群质心**（R196 教训）。
  支持 1  = 迁徙臂个体级位移**幅度** > 空臂 A（有季节无迁徙），**配对 seed**。
  支持 2  = 高 g23 组的半季位移符号 与 sign(δ) 一致率 > 0.5。
  第四态  = 观测 tick < 3×season_period ⇒ 不判。
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, "/workspace/r198")
from s2_smoke import spearman   # noqa: E402

SHARD_DIR = "/workspace/r198/shards"
SEASON = 6000
HALF = SEASON // 2
SEEDS = [42, 7, 11, 13, 17, 23]
ARMS = ["mig_base", "mig_g", "mig_2g", "mig_noseason"]


def delta_sign(tick, period=SEASON):
    d = math.sin(2.0 * math.pi * (tick % period) / period)
    return 0 if abs(d) < 1e-9 else (1 if d > 0 else -1)


def load(arm, seed):
    p = os.path.join(SHARD_DIR, f"{arm}_{seed}.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def cat(shards, key):
    xs = []
    for s in shards:
        if not s:
            continue
        for w in s["windows"]:
            xs.append(np.asarray(w[key], dtype=np.float64))
    return np.concatenate(xs) if xs else np.zeros(0)


def main():
    rep = {"stage": "S3", "season_period": SEASON, "half_window": HALF,
           "seeds": SEEDS, "tilt_deg": 23.44,
           "criteria": "设计稿 §5.1（预注册，未修改；B8）",
           "forbid": "种群质心（R196：质心往返可被原地更替伪造）",
           "method": "按 (臂,seed) 分片并行；个体级 _id 配对；半季净位移 = 窗首尾 |φ| 差"}

    ticks = None
    data = {}
    for arm in ARMS:
        per = {}
        for s in SEEDS:
            sh = load(arm, s)
            if sh:
                per[s] = sh
                ticks = sh["ticks"]
        data[arm] = per
        rep.setdefault("arms", {})[arm] = {
            "seeds_present": sorted(per),
            "alive_end": {str(s): per[s]["alive_end"] for s in per},
        }

    # 第四态
    if not ticks or ticks < 3 * SEASON:
        rep["verdict"] = f"第四态「纪元不足·判据不可执行」（ticks={ticks}）"
        json.dump(rep, open("/workspace/r198/s3_judge.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2, default=str)
        return rep
    rep["ticks"] = ticks

    # ---- 四臂读数 ----
    for arm in ARMS:
        shards = [data[arm].get(s) for s in SEEDS]
        D, G = cat(shards, "D"), cat(shards, "G")
        a = rep["arms"][arm]
        a["n"] = int(len(D))
        a["dphi_mean"] = float(D.mean()) if len(D) else None
        a["dphi_abs_mean"] = float(np.abs(D).mean()) if len(D) else None
        a["dphi_std"] = float(D.std()) if len(D) else None
        if arm == "mig_base":
            a["rho_merged"] = None
            a["note"] = "空臂 A（有季节无迁徙）：不作为主判据臂，仅供支持 1 对照"
        else:
            rho, p = spearman(D, G) if len(D) > 30 else (None, None)
            a["rho_merged"] = rho
            a["p_merged"] = p
            hi = G > np.median(G)
            a["hi_g23_dphi_mean"] = float(D[hi].mean()) if hi.any() else None
            a["lo_g23_dphi_mean"] = float(D[~hi].mean()) if (~hi).any() else None
            per_rho = []
            for s in SEEDS:
                d, g = cat([data[arm].get(s)], "D"), cat([data[arm].get(s)], "G")
                if len(d) >= 30:
                    r, _ = spearman(d, g)
                    if np.isfinite(r):
                        per_rho.append(float(r))
            a["per_seed_rho"] = per_rho
            a["per_seed_positive"] = int(sum(1 for x in per_rho if x > 0))
            a["per_seed_n"] = len(per_rho)

    # ---- 主判据 ----
    def main_pass(arm):
        a = rep["arms"][arm]
        if a.get("rho_merged") is None:
            return False
        return bool(a["rho_merged"] > 0
                    and (a.get("p_merged") or 1.0) < 0.05
                    and a["per_seed_positive"] > a["per_seed_n"] / 2)

    # ---- 支持 1：迁徙臂 |Δφ| > 空臂 A（配对 seed）----
    def sup1(arm):
        w = t = 0
        det = []
        for s in SEEDS:
            A = cat([data[arm].get(s)], "D")
            B = cat([data["mig_base"].get(s)], "D")
            if len(A) and len(B):
                t += 1
                win = np.abs(A).mean() > np.abs(B).mean()
                w += int(win)
                det.append({"seed": s, "arm_abs": float(np.abs(A).mean()),
                            "base_abs": float(np.abs(B).mean()), "win": bool(win)})
        return f"{w}/{t}", (t > 0 and w > t / 2), det

    # ---- 支持 2：高 g23 组位移符号 vs sign(δ) 一致率 > 0.5 ----
    # 🔴 修正（R198 执行中发现，**两次**）：
    #   ① 初版取 t_end ⇒ 窗口尾恒为 P 的整数倍 ⇒ sin≡0 ⇒ 0/0（假阴性）
    #   ② 改为 t_start 仍 0 ⇒ 半季窗边界{t_start,t_end}**都**落在 δ=0 节点上
    #      （窗口按 3000 tick 切，恰是半周期）
    #   ⇒ 正确做法：取**窗口中点** t_mid = (t_start+t_end)/2。此时 δ 取到 ±1（极值），
    #      半季窗交替为 −1/+1 ⇒ 判据既**可执行**又**有区分度**（夏窗/冬窗方向相反）。
    #   🔴 且必须 fail-loud：可评估窗口=0 ⇒ 判据不可执行，**禁止静默判 False**（R148-1 同族）。
    def sup2(arm):
        hit = tot = 0
        skipped_zero = 0
        det = []
        for s in SEEDS:
            sh = data[arm].get(s)
            if not sh:
                continue
            for w in sh["windows"]:
                t_mid = (w["t_start"] + w["t_end"]) // 2
                sgn = delta_sign(t_mid)                 # ← 窗口中点的季节相位
                if sgn == 0:
                    skipped_zero += 1
                    continue
                G = np.asarray(w["G"], dtype=np.float64)
                D = np.asarray(w["D"], dtype=np.float64)
                hi = G > np.median(G)
                if not hi.any():
                    continue
                sh_ = np.sign(D[hi])
                agree = int(np.sum(sh_ == sgn))
                dis = int(np.sum(sh_ == -sgn))
                tot += 1
                if agree > dis:
                    hit += 1
                det.append({"seed": s, "t_start": w["t_start"], "t_mid": t_mid,
                            "sign_delta": sgn, "agree": agree, "disagree": dis,
                            "n_hi": int(hi.sum())})
        # 🔴 fail-loud：可评估窗口为 0 ⇒ 判据不可执行，不得静默判 False
        if tot == 0:
            raise RuntimeError(
                f"支持 2 无可评估窗口（arm={arm}, skipped_zero={skipped_zero}）"
                f" ⇒ 判据不可执行，禁止静默判 False（R148-1 同族纪律）")
        return f"{hit}/{tot}", (hit > tot / 2), det

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
        if rep[f"main_{g}"] and s1p and s2p:
            v = "定向迁徙成立"
        elif rep[f"main_{g}"] and not (s1p and s2p):
            f_ = []
            if not s1p:
                f_.append("支持1(幅度>空臂)")
            if not s2p:
                f_.append("支持2(方向一致率>0.5)")
            v = "证据不完整 —— 主判据过，但 " + "、".join(f_) + " 不过"
        else:
            v = "未成立"
        rep[f"verdict_{g}"] = v

    with open("/workspace/r198/s3_judge.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2, default=str)

    # 打印
    print("=" * 74)
    for arm in ARMS:
        a = rep["arms"][arm]
        print(f"{arm:14s} n={a['n']:>8d} |Δφ|={a.get('dphi_abs_mean')} "
              f"ρ={a.get('rho_merged')} 逐seed正号={a.get('per_seed_positive')}/{a.get('per_seed_n')}")
    for g in ("g20", "g40"):
        print(f"\n【{g}】主={rep[f'main_{g}']} S1={rep[f'support1_{g}']}"
              f"({rep[f'support1_{g}_pass']}) S2={rep[f'support2_{g}']}"
              f"({rep[f'support2_{g}_pass']})\n   ⇒ {rep[f'verdict_{g}']}")
    print("\n已写 /workspace/r198/s3_judge.json")
    return rep


if __name__ == "__main__":
    main()
