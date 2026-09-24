#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""R198 / 13.8 S3 判读 —— **按 (臂, seed) 分片并行**，每片独立进程写一个 JSON 片段。

为什么并行：单 run（18000 tick，N=3240）> 3.5 min ⇒ 24 runs 串行 > 85 min。
36 核可承受 24 路并发 ⇒ 墙钟 ≈ 单 run 时间（约 5–8 min）。

用法：
  python3.12 s3_judge_shard.py --arm mig_g --seed 42 --out /workspace/r198/shards/mig_g_42.json

判据本体（§5.1 预注册）在 s3_judge.py；本脚本只负责"跑一片 + 落盘"。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, "/tmp/tw")
sys.path.insert(0, "/workspace/r198")

import s3_judge as J   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(J.ARMS))
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--ticks", type=int, default=J.TICKS)
    ap.add_argument("--cap", type=int, default=None,
                    help="种群上限；缺省=3240（与 S3 preset 同口径）。"
                         "§5.1 对规模无要求，小 cap 可显著省墙钟")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    kw = dict(J.ARMS[a.arm])
    r = J.run_arm(a.seed, ticks=a.ticks, cap=a.cap, **kw)
    payload = {
        "arm": a.arm, "seed": a.seed, "ticks": a.ticks, "cap": a.cap,
        "season_period": J.SEASON, "half_window": J.HALF,
        "alive_end": r["alive_end"],
        "windows": [
            {"t_start": w["t_start"], "t_end": w["t_end"], "n": w["n"],
             "D": [float(x) for x in w["D"]],
             "G": [float(x) for x in w["G"]]}
            for w in r["windows"]
        ],
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, a.out)
    print(f"✅ {a.arm} seed={a.seed} windows={len(r['windows'])} "
          f"样本={sum(w['n'] for w in r['windows'])} alive={r['alive_end']}", flush=True)


if __name__ == "__main__":
    main()
