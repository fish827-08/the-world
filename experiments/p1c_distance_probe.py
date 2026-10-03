#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1-c 距离/可达性门探针（R351 派工 A4；**纯测量，不预设结论**）。

判据来源：`docs/设计总档案/AI-设计总档案-v1.4-定稿.md` §5.1「P1-c」
    - 记忆格到当前格的距离分布（中位 / 90 分位，单位=格；🔴 必须用**球面距离**）
    - 可直读比例（距离 ≤1 格的记忆格占比）
判读分支（定稿原文，本探针只出数字，**不代判**）：
    - 距离足够远（可直读比例低）⇒ `mem_bit` 构成私有信息 ⇒ M1 前提成立；
    - 距离都很近（可直读比例高）⇒ 🔴 停止 M1，先判根因（世界饱和 vs 记忆策略），
      ⚠️ **不得只改 `mem_bit` 判据** —— 回报天平。

装置口径（= `smell_rd_01` A0 基线档 + P1 链前提，单一变量说明见下）：
    480×960 / patches 1700 / pop 10000 / rgm 1.195 / bg_low 0.0/0.0（纯斑块 A 支）
    subpos 开（speed_max 0.125 / gain 0.125 / subdiv 80 / k 2.5 / max_count 30000）
    rd 关（M1 用 rd 关装置，R349/R350）＋ 无 smell（默认）
    - 相对 A0 的**唯一差异** = `signal_alphabet="8"`（P1-a 前提；`mem_bit` 通路所必需）。
    - `memory_v2` 保持 False（P1-b 前提，R6.7：字段在 `cfg.info_structure` 下，不是顶层）。
    - `info_structure.enabled` 保持默认关（不引入 D2 机制；= A0 口径）。
    - `sparse_fields` 默认关（本测量与稀疏化正交；需要与 A0 完全同旗标可用 `--sparse-fields` 复跑）。

测什么（每个采样 tick，对每个存活个体、每个有效记忆槽 `_work_memory[i,j] >= 0`）：
    d = 球面大圆距离(当前格中心, 记忆格中心)，单位 = 1 个纬度格宽
        （`world.subpos.sphere_dist_rows`，atan2 形式 ⇒ 极区自动正确；不直译 9×9）
    统计：中位 / p90 / p99 / 均值 / max；分段占比：d==0、d≤1、d≤1.5
    引擎真值对照（口径 B）：记忆格 ∈ Moore 8 邻（= 移动打分里 `mem_in_nb` 的口径，
        `sphere_engine.py:3171-3177`；极点格按相邻纬度带整行处理）
        ⇒ `frac_read_moore` = **引擎实际能直读**的比例
    ⚠️ 口径 A（定稿字面，"距离 ≤1 格"）与口径 B（引擎邻表）在纬度/对角处略有差；
        两者都报，判读用哪个由天平定。
    🔴 **d>0 子集（列前缀 `pos_`）**：`mem_bit=1` 的构造要求"存在一个 **≠ 当前格** 的
        记忆富食点"（`encode_signal_states` 的 `mem_hit`）⇒ d==0 的槽**不驱动 mem_bit**。
        故另报 d>0 子集的同族统计（`pos_n_slots` / `pos_d_med` / `pos_d_p90` /
        `pos_frac_dle1` / `pos_frac_read_moore`）—— 该子集才是"私有性"争点的直接读数。

用法（项目根目录）：
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m experiments.p1c_distance_probe \
        --seed 207 --ticks 3000 --sample 100 \
        --out _trash_local/p1c/p1c_s207_t3000.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# R98 纪律：非 ASCII 输出在 GBK 控制台会 rc=1 假失败，入口统一兜底。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from simulation.sphere_engine import SphereEngine          # noqa: E402
from experiments.steady_k_probe import make_cfg, apply_post_build  # noqa: E402
from world.subpos import sphere_dist_rows                  # noqa: E402
from tools.device_presets import add_device_arg, resolve_device    # noqa: E402

# subpos-family 常量（与 S1_BASE / s2_depletion_probe 逐位一致；装置字段走 CLI）
SPEED_MAX, GAIN, SUBDIV, K_TC, MAX_COUNT = 0.125, 0.125, 80, 2.5, 30000


def _pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if x.size else float("nan")


def _frac(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else float("nan")


def measure(e: SphereEngine) -> dict:
    """一次采样：全部存活个体 × 全部有效记忆槽的球面距离。"""
    n = int(len(e._id))
    if n == 0:
        return {"n_agents": 0, "n_slots": 0, "frac_empty": 1.0,
                "d_med": None, "d_p90": None, "d_p99": None,
                "d_mean": None, "d_max": None,
                "frac_d0": None, "frac_dle1": None, "frac_dle15": None,
                "frac_read_moore": None,
                "pos_n_slots": 0, "pos_d_med": None, "pos_d_p90": None,
                "pos_frac_dle1": None, "pos_frac_read_moore": None,
                "_d_pool": np.empty(0), "_pos_pool": np.empty(0)}

    flat = np.asarray(e._flat)[:n]
    wm = np.asarray(e._work_memory)[:n]
    valid = wm >= 0
    n_slots = int(valid.sum())
    n_cap = 4 * n
    if n_slots == 0:
        return {"n_agents": n, "n_slots": 0, "frac_empty": 1.0,
                "d_med": None, "d_p90": None, "d_p99": None,
                "d_mean": None, "d_max": None,
                "frac_d0": None, "frac_dle1": None, "frac_dle15": None,
                "frac_read_moore": None,
                "pos_n_slots": 0, "pos_d_med": None, "pos_d_p90": None,
                "pos_frac_dle1": None, "pos_frac_read_moore": None,
                "_d_pool": np.empty(0), "_pos_pool": np.empty(0)}

    ai, _sj = np.nonzero(valid)
    cur = flat[ai].astype(np.int64)
    mem = wm[ai, _sj].astype(np.int64)

    # ---- 口径 A：球面大圆距离（单位 = 1 纬度格宽）；格中心 = 整数索引 + 0.5
    world = e.world
    c_r, c_c = world.flat_to_rc(cur)
    m_r, m_c = world.flat_to_rc(mem)
    d = np.asarray(sphere_dist_rows(
        c_r + 0.5, c_c + 0.5, m_r + 0.5, m_c + 0.5, world), dtype=np.float64)

    # ---- 口径 B：引擎"可直读"= 记忆格 ∈ Moore 8 邻（极点格 = 相邻纬度带整行）
    cur_row = cur // world.cols
    is_pole = (cur_row == 0) | (cur_row == world.rows - 1)
    readable = (mem == cur)
    normal = ~is_pole
    if normal.any():
        nb = world._nb_table[cur[normal]]                 # (m,8)，普通格预计算邻表
        readable[normal] = readable[normal] | (nb == mem[normal, None]).any(axis=1)
    for i in np.flatnonzero(is_pole):
        band = world._pole_nb[0 if cur_row[i] == 0 else 1]
        readable[i] = readable[i] or bool((band == mem[i]).any())

    eps = 1e-9
    pos = d > eps
    return {
        "n_agents": n, "n_slots": n_slots,
        "frac_empty": float(1.0 - n_slots / n_cap),
        "d_med": _pct(d, 50), "d_p90": _pct(d, 90), "d_p99": _pct(d, 99),
        "d_mean": float(d.mean()), "d_max": float(d.max()),
        "frac_d0": _frac(d <= eps),
        "frac_dle1": _frac(d <= 1.0 + eps),
        "frac_dle15": _frac(d <= 1.5 + eps),
        "frac_read_moore": _frac(readable),
        "pos_n_slots": int(pos.sum()),
        "pos_d_med": _pct(d[pos], 50), "pos_d_p90": _pct(d[pos], 90),
        "pos_frac_dle1": _frac(d[pos] <= 1.0 + eps),
        "pos_frac_read_moore": _frac(readable[pos]),
        "_d_pool": d, "_pos_pool": d[pos],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="P1-c 距离/可达性门探针（纯测量）")
    ap.add_argument("--seed", type=int, default=207)
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--out", required=True, help="逐采样 CSV 路径（必填）")
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--cols", type=int, default=960)
    ap.add_argument("--patches", type=int, default=1700)
    ap.add_argument("--pop", type=int, default=10000)
    ap.add_argument("--rgm", type=float, default=1.195)
    ap.add_argument("--bg-low-prod-frac", type=float, default=0.0)
    ap.add_argument("--bg-low-cap-mult", type=float, default=0.0)
    ap.add_argument("--sparse-fields", action="store_true",
                    help="与 smell_rd_01 A0 完全同旗标复跑用（本测量与稀疏正交）")
    add_device_arg(ap)                      # R5.6：显式 --device（预设档防呆）
    a = ap.parse_args()
    resolve_device(a, sys.argv[1:])

    c, notes = make_cfg(
        a.seed, a.rows, a.cols, a.pop, a.patches, True,
        SPEED_MAX, GAIN, SUBDIV, K_TC, MAX_COUNT, a.rgm,
        bg_low_prod_frac=a.bg_low_prod_frac,
        bg_low_cap_mult=a.bg_low_cap_mult,
    )
    c.simulation.use_sim_core = False       # 纯 Python（本机无 .so；subpos 亦强制）
    c.simulation.sparse_fields = bool(a.sparse_fields)
    c.resource_dynamics.enabled = False     # 🔴 rd 关（M1 用 rd 关装置）
    c.signal_alphabet = "8"                 # 🔴 P1-a 前提（mem_bit 通路）
    # memory_v2 / info_structure.enabled 保持默认（False / False = A0 口径）

    e = SphereEngine(c)
    apply_post_build(e, notes)

    # ---- fail-loud 读回（R6.6 仪表先冒烟；任一不符立即中止）
    rb = {
        "signal_alphabet": getattr(e, "_alpha", None),
        "memory_v2": bool(getattr(c.info_structure, "memory_v2", None)),
        "info_structure_enabled": bool(getattr(c.info_structure, "enabled", None)),
        "rd_enabled": bool(getattr(getattr(e, "_rd", None), "enabled", False)),
        "use_sim_core": bool(getattr(c.simulation, "use_sim_core", True)),
        "subpos_enabled": bool(getattr(c.subpos, "enabled", False)),
        "sparse_cli": bool(a.sparse_fields),
        "rows": int(e.world.rows), "cols": int(e.world.cols),
        "patches": int(getattr(c.resources, "patch_count", -1)),
        "max_count": int(getattr(c.population, "max_count", -1)),
        "rgm": float(getattr(c.resources, "patch_regrowth_mult", -1.0)),
        "bg_low_prod_frac": float(getattr(c.resources, "bg_low_prod_frac", -1.0)),
        "bg_cap_mult": float(getattr(c.resources, "bg_cap_mult", -1.0)),
        "initial_pop": int(len(e._id)),
        "notes": str(notes),
    }
    bad = []
    if rb["signal_alphabet"] != "8":
        bad.append(f'signal_alphabet 读回 {rb["signal_alphabet"]!r} != "8"')
    if rb["memory_v2"] is not False:
        bad.append("memory_v2 读回 != False（P1-b 前提；R6.7 字段归属）")
    if rb["rd_enabled"]:
        bad.append("rd 应关未关")
    if rb["use_sim_core"]:
        bad.append("use_sim_core 应为 False（纯 Python）")
    if bad:
        raise SystemExit("🔴 P1-c 探针前置读回失败：\n  - " + "\n  - ".join(bad))

    print(f"# [P1-c] seed={a.seed} ticks={a.ticks} sample={a.sample} "
          f"device={a.rows}x{a.cols} patches={rb['patches']} pop0={rb['initial_pop']}")
    print(f"# [P1-c] 读回：alphabet={rb['signal_alphabet']} rd={rb['rd_enabled']} "
          f"memory_v2={rb['memory_v2']} info_enabled={rb['info_structure_enabled']} "
          f"sparse={a.sparse_fields} bg_low={rb['bg_low_prod_frac']}/{rb['bg_cap_mult']}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["tick", "n_agents", "n_slots", "frac_empty", "d_med", "d_p90",
            "d_p99", "d_mean", "d_max", "frac_d0", "frac_dle1", "frac_dle15",
            "frac_read_moore", "ms_per_tick_window",
            "pos_n_slots", "pos_d_med", "pos_d_p90", "pos_frac_dle1",
            "pos_frac_read_moore"]
    recs: list[dict] = []
    pool: list[np.ndarray] = []
    pos_pool: list[np.ndarray] = []
    t_prev = time.perf_counter()
    extinct_at = None
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        f.flush()
        os.fsync(f.fileno())
        for t in range(1, a.ticks + 1):
            e.step()
            if e.extinct:
                extinct_at = t
                break
            if t % a.sample == 0:
                now = time.perf_counter()
                ms_t = (now - t_prev) / a.sample * 1000.0
                t_prev = now
                rec = measure(e)
                pool.append(rec.pop("_d_pool"))
                pos_pool.append(rec.pop("_pos_pool"))
                rec["tick"] = t
                rec["ms_per_tick_window"] = round(ms_t, 2)
                recs.append(rec)
                w.writerow({k: (v if v is None else
                                (round(v, 4) if isinstance(v, float) else v))
                            for k, v in rec.items()})
                f.flush()
                os.fsync(f.fileno())       # R263：行级落盘，崩溃最多丢 1 个采样点
                print(f"  t={t:>6d} N={rec['n_agents']:>6d} slots={rec['n_slots']:>6d} "
                      f"med={rec['d_med'] if rec['d_med'] is None else round(rec['d_med'], 3)} "
                      f"p90={rec['d_p90'] if rec['d_p90'] is None else round(rec['d_p90'], 3)} "
                      f"≤1格={rec['frac_dle1'] if rec['frac_dle1'] is None else round(rec['frac_dle1'], 4)} "
                      f"可直读(Moore)={rec['frac_read_moore'] if rec['frac_read_moore'] is None else round(rec['frac_read_moore'], 4)} "
                      f"d>0可直读={rec['pos_frac_read_moore'] if rec['pos_frac_read_moore'] is None else round(rec['pos_frac_read_moore'], 4)} "
                      f"ms/t={rec['ms_per_tick_window']}", flush=True)

    dd = np.concatenate(pool) if pool else np.empty(0)
    pd = np.concatenate(pos_pool) if pos_pool else np.empty(0)
    eps = 1e-9
    pooled = {
        "n_samples": len(recs), "n_dist_pairs": int(dd.size),
        "d_med": _pct(dd, 50), "d_p90": _pct(dd, 90), "d_p99": _pct(dd, 99),
        "d_mean": float(dd.mean()) if dd.size else None,
        "d_max": float(dd.max()) if dd.size else None,
        "frac_d0": _frac(dd <= eps), "frac_dle1": _frac(dd <= 1.0 + eps),
        "frac_dle15": _frac(dd <= 1.5 + eps),
    }
    # 可直读比例池化：按每采样 n_slots 加权（从 recs 重算，等价于全配对混合）
    if dd.size:
        wsum = sum(r["n_slots"] for r in recs)
        pooled["frac_read_moore"] = float(
            sum(r["frac_read_moore"] * r["n_slots"] for r in recs) / wsum)
    else:
        pooled["frac_read_moore"] = None
    # d>0 子集池化（mem_bit 判据相关；与上面同口径，按 pos_n_slots 加权）
    if pd.size:
        wsum_pos = sum(r["pos_n_slots"] for r in recs)
        pooled["pos_n_dist_pairs"] = int(pd.size)
        pooled["pos_d_med"] = _pct(pd, 50)
        pooled["pos_d_p90"] = _pct(pd, 90)
        pooled["pos_frac_dle1"] = _frac(pd <= 1.0 + eps)
        pooled["pos_frac_read_moore"] = (
            float(sum(r["pos_frac_read_moore"] * r["pos_n_slots"]
                      for r in recs) / wsum_pos) if wsum_pos else None)
    else:
        pooled["pos_n_dist_pairs"] = 0
        pooled["pos_d_med"] = None
        pooled["pos_d_p90"] = None
        pooled["pos_frac_dle1"] = None
        pooled["pos_frac_read_moore"] = None

    summary = {
        "probe": "p1c_distance_probe", "argv": sys.argv[1:],
        "seed": a.seed, "ticks": a.ticks, "sample": a.sample,
        "extinct_at": extinct_at, "readback": rb, "per_sample": recs,
        "pooled": pooled,
        "notes": [
            "口径A = 定稿字面：球面大圆距离 ≤1.0 格（单位=1 纬度格宽）",
            "口径B = 引擎真值：记忆格 ∈ Moore 8 邻（行内 mem_in_nb 口径，含极点带特判）",
            "pos_* = d>0 子集（mem_bit 判据要求槽 ≠ 当前格；d==0 槽不驱动 mem_bit）",
            "纯测量，不预设结论；分支判读由天平据 §5.1 P1-c 条款执行",
        ],
    }
    sp = out.with_name(out.stem + ".summary.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())

    print("\n===== P1-c 池化（全部采样点、全部有效记忆槽）=====")
    print(f"  配对样本 n={pooled['n_dist_pairs']}（{pooled['n_samples']} 个采样点）")
    print(f"  距离（格，单位=1 纬度格宽）：中位={pooled['d_med']:.3f} "
          f"p90={pooled['d_p90']:.3f} p99={pooled['d_p99']:.3f} "
          f"均值={pooled['d_mean']:.3f} max={pooled['d_max']:.3f}")
    print(f"  分段占比：d==0 {pooled['frac_d0']:.4f}｜d≤1 {pooled['frac_dle1']:.4f}"
          f"（口径A）｜d≤1.5 {pooled['frac_dle15']:.4f}｜"
          f"可直读(Moore) {pooled['frac_read_moore']:.4f}（口径B）")
    if pooled.get("pos_n_dist_pairs"):
        print(f"  [d>0 子集（mem_bit 判据相关）] n={pooled['pos_n_dist_pairs']} "
              f"中位={pooled['pos_d_med']:.3f} p90={pooled['pos_d_p90']:.3f} "
              f"d≤1={pooled['pos_frac_dle1']:.4f}（口径A）｜"
              f"可直读(Moore)={pooled['pos_frac_read_moore']:.4f}（口径B）")
    print(f"  逐采样 CSV：{out}")
    print(f"  摘要 JSON：{sp}")
    if extinct_at is not None:
        print(f"  ⚠️ 灭绝于 tick={extinct_at}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
