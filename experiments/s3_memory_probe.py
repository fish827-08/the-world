#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3 批跑探针 —— 记忆 v2（egocentric）对**吸引子归属的偏置**。

任务来源：R255（天平）云归任务②「S3 批跑脚本准备」
  原文："配置照 S1/S0：patches=1700 + bg_low 0.4/0.05 + memory_v2 两臂，
        种子区间另定与已用 42–100 错开"。

判据基调（R255 §三，R254 裁定）：
  🔴 效应量 = 记忆对吸引子归属的**偏置**（on 臂相对 off 臂的移动方向 +
     记忆相关基因频率变化），**禁用"出走占比"绝对值**（对种子区间敏感、非可外推）。
  候选主判据：
    ① 记忆基因频率在 on 臂被选择抬升（多 seed 一致）
    ② "从未被访问斑块格占比"相对关臂下降
    ③ 留守型盆地内个体在斑块耗竭前的撤离时机（记忆应有的行为签名）

装置口径（实测可构造，见上板报告）：
  patchy + bg_production_zero=True + patches=1700 + bg_low 0.4/0.05 + memory_v2 两臂
  （= S0/S1/S2 同源装置；bg_low 仅在 bgzero 下有语义，二者不是冲突而是配套）
  ⚠️ memory_v2 要求 use_sim_core=False（fail-loud ①）⇒ 本探针强制 Python 路径。
  ⚠️ memory_v2 ∧ rd ∧ bgzero=True ⇒ 放行（fail-loud ② 只拦 ¬bgzero）。

两臂定义：
  arm="mem_on"  : memory_v2=True  + memory_gradient="orientation"
  arm="mem_off" : memory_v2=False + memory_gradient="none"（默认关档 = 回滚点）
  🔴 两臂 rd 均开（rd.enabled=True）、装置其余字段逐位一致 ⇒ 单变量。

输出：stdout 报告 + CSV（逐采样点）+ 同名 .summary.json。

🔴 v2（R264 派工，2026-09-29）：**逐 run flush + fsync + `--append` 断点续跑**。
  背景：S3 首轮（14:00）因 memv2 fail-loud 崩在 seed 104，双进程退出后**已完成的
  6 个 run 数据全丢** —— 根因是 v1 用 `with open(out,"w")`，所有行只在块正常退出时
  落盘，进程被杀 ⇒ 块缓冲全丢（R263 登记「块缓冲未刷」教训兑现）。
  v2 改法：`csv.DictWriter` + **每 run 结束 `flush()` + `os.fsync()`** ⇒ 崩溃最多丢
  1 个 run，已完成的 run 绝不丢；summary 同样每 run 原子重写；`--append` 跳过已完成
  `(seed, arm)` ⇒ 重启零成本。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

# R98 纪律：非 ASCII 输出（如 ⇒、【】）在 GBK 控制台会 rc=1 假失败，入口统一兜底。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from simulation.sphere_engine import SphereEngine
from experiments.steady_k_probe import make_cfg, apply_post_build
# 复用 S2 已验证的读数函数（斑块标注 / 饱和度 / L1-L2 分化）
from experiments.s2_depletion_probe import (
    _label_patches, patch_saturation, l2_variance_decomposition, l0_rd_state,
)

# S3 装置口径（= S0/S1/S2 逐字段同源；见板上 S0 预注册原文）
#   rows=480 cols=960 patches=1700 pop=10000 ticks=2000 sample=250
#   bg_low_prod_frac=0.4 bg_low_cap_mult=0.05（不挂 sparse）
#   subpos-family：speed_max/gain/subdiv/k/max_count 与 S1_BASE 对齐
#   （s2_depletion_probe.run_one 亦如此 → subpos=on + 这组参数）
# 唯一变量 = memory_v2 两臂；rd 两臂均开。
DEVICE = dict(
    rows=480, cols=960, patches=1700, pop=10000,
    speed_max=0.125, gain=0.125, subdiv=80, k=2.5, max_count=30000,
    ticks=2000, sample=250, rgm=1.195,
    bg_low_prod_frac=0.4, bg_low_cap_mult=0.05,
)
# 兼容别名（旧脚本引用 S1_BASE 的 subpos 家族参数）
S1_BASE = DEVICE

# 已用种子区间（R243 S0=45-52 / S1=53-76 / 新机复现=77-100）
SEEDS_USED = set(range(42, 101))


# ---------- v2：断点续跑辅助（R264） ----------

def _scan_done_runs(csv_path: str, ticks: int) -> set[tuple[str, str]]:
    """扫描已有 CSV，返回**已完成**的 `(seed, arm)` 集合。

    完成判据：该 run 存在 `tick == ticks` 的末行（说明跑到目标 tick 正常收尾）。
    只扫到部分行 ⇒ 视为**未完成**（由调用方整 run 重跑并先删残行）。
    """
    done: set[tuple[str, str]] = set()
    if not os.path.exists(csv_path):
        return done
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    if int(float(row.get("tick", -1))) == int(ticks):
                        done.add((str(row.get("seed")), str(row.get("arm"))))
                except (TypeError, ValueError):
                    continue
    except OSError as e:
        print(f"  ⚠️ 读取已有 CSV 失败（按空处理）：{e}", file=sys.stderr)
    return done


def _strip_incomplete(csv_path: str, done: set[tuple[str, str]]) -> int:
    """删掉**未完成** run 的残行（= 该 (seed,arm) 未在 done 集里），保留已完成行与表头。

    返回删除行数。用于 `--append`：保证重跑某 run 时不与其残行重复。
    """
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        keep, drop = [], 0
        for row in reader:
            key = (str(row.get("seed")), str(row.get("arm")))
            if key in done:
                keep.append(row)
            else:
                drop += 1
    if drop == 0 or not fieldnames:
        return 0
    tmp = csv_path + ".tmp_strip"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(keep)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, csv_path)
    return drop


def run_one(seed, rows, cols, pop, patches, ticks, sample, rgm, mem_on,
            bg_low_prod_frac=0.4, bg_low_cap_mult=0.05):
    """跑一个 S3 run（单 seed 单臂）。

    rd 恒开；mem_on 决定记忆 v2 开关（其余逐字段对齐 ⇒ 单变量）。
    """
    c, notes = make_cfg(
        seed, rows, cols, pop, patches, True,
        S1_BASE["speed_max"], S1_BASE["gain"], S1_BASE["subdiv"],
        S1_BASE["k"], S1_BASE["max_count"], rgm,
        bg_low_prod_frac=bg_low_prod_frac, bg_low_cap_mult=bg_low_cap_mult,
    )
    c.simulation.use_sim_core = False           # memory_v2 fail-loud ① 要求 Python 路径
    c.resource_dynamics.enabled = True          # S3 主线：rd 开（= S2 处理臂）
    # ---- 记忆两臂（单变量）----
    if mem_on:
        c.info_structure.memory_v2 = True
        c.info_structure.memory_gradient = "orientation"
    else:
        c.info_structure.memory_v2 = False
        c.info_structure.memory_gradient = "none"

    eng = SphereEngine(c)
    apply_post_build(eng, notes)

    init_cap_full = eng.resources._capacity.copy()
    labels = _label_patches(eng.world, eng.resources._patch_mask)
    uniq0 = np.unique(labels[labels >= 0])
    init_patch_cap = {int(lab): float(init_cap_full[labels == lab].sum())
                      for lab in uniq0}
    init_cap_sum = float(eng.resources._capacity.sum())

    n_patch_total = int(uniq0.size)

    # 参数一致性自检（防静默换档）
    assert bool(eng.config.resource_dynamics.enabled) is True, "rd 未开"
    assert bool(eng.config.resources.bg_production_zero) is True, "bgzero 未开"
    assert bool(eng.config.info_structure.memory_v2) is bool(mem_on), "memory_v2 未生效"

    # 🔴 R264 自检：S3 装置（bg_low ⇒ ¬bgzero）+ memv2 + rd 只有在**动态质心**开着才合法。
    #    若跑在旧 main（无该字段）或该字段被关 ⇒ 静态质心 + rd 搬移 = 质心静默失效（fail-loud
    #    在旧版会直接炸；新版 dynamic=False 时才炸）。此处显式告警，避免"跑了但结果无效"。
    if mem_on:
        _dyn = getattr(eng.config.info_structure, "memory_v2_dynamic_centroid", None)
        if _dyn is None:
            print("  🔴 警告：本 main 无 memory_v2_dynamic_centroid 字段 ⇒ 用的是**静态质心**；"
                  "rd 搬移会使质心静默失效（应升级到含 R264 动态化的版本）。",
                  file=sys.stderr)
        elif not bool(_dyn):
            print("  🔴 警告：memory_v2_dynamic_centroid=False ⇒ 静态质心 + rd 搬移 ⇒ "
                  "质心静默失效（与 R264 正解不符）。", file=sys.stderr)

    rows_out = []
    t_slice = time.perf_counter()
    last = 0
    stop = "ticks"
    visited_mask = np.zeros(eng.world.n_cells, dtype=bool)   # 是否曾被任何个体访问
    first_visit = {}                                          # patch_label -> tick
    patch_ever_visited = np.zeros(n_patch_total, dtype=bool)

    for t in range(1, ticks + 1):
        eng.step()
        flat = eng._flat
        visited_mask[flat] = True
        for p in np.unique(labels[flat]):
            if p >= 0 and not patch_ever_visited[p]:
                patch_ever_visited[p] = True
                first_visit[int(p)] = t

        if t % sample == 0 or t == ticks:
            dt = (time.perf_counter() - t_slice) / max(t - last, 1) * 1e3
            last = t
            t_slice = time.perf_counter()
            N = int(len(eng._flat))
            if N == 0:
                stop = "灭绝"
            cap_live = eng.resources._capacity
            prod = cap_live > 0
            cap_sum_live = float(cap_live[prod].sum())
            stock = float(eng.resources._grid[prod].sum())
            sat = stock / cap_sum_live if cap_sum_live > 0 else float("nan")
            cap_lost_frac = (1.0 - cap_sum_live / init_cap_sum
                             if init_cap_sum > 0 else float("nan"))
            ps = patch_saturation(eng.world, eng, labels, init_patch_cap)
            l2 = l2_variance_decomposition(eng.resources._grid, labels, visited_mask)
            l0 = l0_rd_state(eng)
            # ---- S3 记忆专项读数 ----
            # ② 从未被访问斑块格占比（on 臂应下降）
            never_visited_frac = 1.0 - (int(visited_mask[labels >= 0].sum())
                                        / max(1, int((labels >= 0).sum())))
            # ③ 留守型撤离时机：以背景停留占比 bg_resid_frac 的轨迹体现（见判读）
            bg_resid = float(visited_mask[labels < 0].sum()) / max(1, int((labels < 0).sum()))
            rows_out.append({
                "seed": seed, "arm": "mem_on" if mem_on else "mem_off", "tick": t,
                "pop": N, "global_sat": sat,
                "abs_food": stock, "abs_cap": cap_sum_live,
                "cap_lost_frac": cap_lost_frac,
                "ms_per_tick": dt,
                "n_patches": ps["n_patches"],
                "patch_sat_mean": ps["patch_sat_mean"],
                "patch_sat_var": ps["patch_sat_var"],
                "patch_sat_range": ps["patch_sat_range"],
                "patch_sat_init_var": ps["patch_sat_init_var"],
                "l2_visited_gini": l2["l2_visited_gini"],
                "l2_visited_frac": l2["l2_visited_frac"],
                "l2_between_share": l2["l2_between_share"],
                # ---- S3 专项 ----
                "never_visited_patch_cell_frac": round(never_visited_frac, 6),
                "bg_resid_frac": round(bg_resid, 6),
                "l1_visited_patch_n": int(patch_ever_visited.sum()),
                "l1_visited_patch_frac": (int(patch_ever_visited.sum()) / n_patch_total
                                          if n_patch_total else 0.0),
                "l0_peak_dead": l0.get("l0_peak_dead", 0),
                "l0_peak_rest": l0.get("l0_peak_rest", 0),
            })
            if stop == "灭绝":
                break
    return rows_out, stop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=DEVICE["rows"])
    ap.add_argument("--cols", type=int, default=DEVICE["cols"])
    ap.add_argument("--patches", type=int, default=DEVICE["patches"],
                    help="R255 口径：1700（S0/S1/S2 复现批同款）")
    ap.add_argument("--pop", type=int, default=DEVICE["pop"])
    ap.add_argument("--seeds", default="101,102,103",
                    help="与已用 42–100 错开（R255 要求）")
    ap.add_argument("--ticks", type=int, default=DEVICE["ticks"],
                    help="与 S2 复现批同口径（2000t，H1 已验证）")
    ap.add_argument("--sample", type=int, default=DEVICE["sample"])
    ap.add_argument("--rgm", type=float, default=DEVICE["rgm"])
    ap.add_argument("--bg-low-prod-frac", type=float, default=DEVICE["bg_low_prod_frac"])
    ap.add_argument("--bg-low-cap-mult", type=float, default=DEVICE["bg_low_cap_mult"])
    ap.add_argument("--arms", choices=("both", "on", "off"), default="both")
    ap.add_argument("--out", default="results/s3_memory.csv")
    ap.add_argument("--append", action="store_true",
                    help="断点续跑：追加到已存在的 --out，跳过已完成的 (seed,arm)，"
                         "并先删未完成 run 的残行（默认关 = 覆盖写，向后兼容）")
    a = ap.parse_args()

    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    overlap = sorted(set(seeds) & SEEDS_USED)
    if overlap:
        print(f"🔴 种子 {overlap} 与已用区间 42–100 重叠 ⇒ 违反 R255「另定错开」。中止。",
              file=sys.stderr)
        sys.exit(2)

    arms = ([True, False] if a.arms == "both" else [a.arms == "on"])
    header = ["seed", "arm", "tick", "pop", "global_sat",
              "abs_food", "abs_cap", "cap_lost_frac", "ms_per_tick",
              "n_patches", "patch_sat_mean", "patch_sat_var", "patch_sat_range",
              "patch_sat_init_var", "l2_visited_gini", "l2_visited_frac",
              "l2_between_share",
              "never_visited_patch_cell_frac", "bg_resid_frac",
              "l1_visited_patch_n", "l1_visited_patch_frac",
              "l0_peak_dead", "l0_peak_rest"]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out_json = os.path.splitext(a.out)[0] + ".summary.json"

    print(f"# S3 探针 v2：rows={a.rows} cols={a.cols} patches={a.patches} "
          f"pop={a.pop} rgm={a.rgm} ticks={a.ticks} seeds={seeds} "
          f"arms={['on' if m else 'off' for m in arms]} "
          f"bg_low_frac={a.bg_low_prod_frac} bg_low_mult={a.bg_low_cap_mult} "
          f"append={a.append}")

    # ---- v2：断点续跑准备（R264）----
    done = set()
    if a.append:
        done = _scan_done_runs(a.out, a.ticks)
        if done:
            dropped = _strip_incomplete(a.out, done)
            print(f"# 续跑：已完成 {len(done)} 个 (seed,arm)，清理未完成残行 {dropped} 行",
                  file=sys.stderr)
        else:
            print("# 续跑：无已完成 run（或 --out 不存在）⇒ 从头跑", file=sys.stderr)

    # 已完成 run 的 summary 继承（--append 时不让旧 summary 丢）
    summary = []
    if a.append and os.path.exists(out_json):
        try:
            with open(out_json, "r", encoding="utf-8") as jf:
                summary = list(json.load(jf).get("runs", []))
        except (OSError, ValueError) as e:
            print(f"  ⚠️ 旧 summary 读取失败（忽略）：{e}", file=sys.stderr)

    # 🔴 v2 核心：append 模式不重写表头；文件为空（新建/刚清空）时才写
    need_header = (not a.append) or (not os.path.exists(a.out)) \
        or (os.path.getsize(a.out) == 0)

    fout = open(a.out, "a" if a.append else "w", newline="", encoding="utf-8")
    try:
        w = csv.DictWriter(fout, fieldnames=header, extrasaction="ignore")
        if need_header:
            w.writeheader()
            fout.flush()
            os.fsync(fout.fileno())
        for sd in seeds:
            for mem_on in arms:
                arm = "mem_on" if mem_on else "mem_off"
                if a.append and (str(sd), arm) in done:
                    print(f"# skip seed={sd} arm={arm}（已完成，续跑跳过）",
                          file=sys.stderr)
                    continue
                t0 = time.time()
                rows_out, stop = run_one(
                    sd, a.rows, a.cols, a.pop, a.patches, a.ticks, a.sample,
                    a.rgm, mem_on, a.bg_low_prod_frac, a.bg_low_cap_mult)
                for r in rows_out:
                    w.writerow({h: r[h] for h in header})
                # 🔴 逐 run flush + fsync（R264 硬要求）：崩最多丢 1 run
                fout.flush()
                os.fsync(fout.fileno())
                print(f"# flushed seed={sd} arm={arm} rows={len(rows_out)}",
                      file=sys.stderr)

                last_r = rows_out[-1]
                summary = [s for s in summary
                           if not (str(s.get("seed")) == str(sd)
                                   and s.get("arm") == arm)]
                summary.append({
                    "seed": sd, "arm": arm,
                    "stop": stop, "K_final": last_r["pop"],
                    "bg_resid_frac": last_r["bg_resid_frac"],
                    "never_visited_patch_cell_frac":
                        last_r["never_visited_patch_cell_frac"],
                    "l1_visited_patch_frac": last_r["l1_visited_patch_frac"],
                    "cap_lost_frac": last_r["cap_lost_frac"],
                    "wall_s": round(time.time() - t0, 1),
                })
                # 🔴 summary 同样逐 run 原子重写（防同类丢失）
                tmp_json = out_json + ".tmp"
                with open(tmp_json, "w", encoding="utf-8") as jf:
                    json.dump({"meta": vars(a), "runs": summary}, jf,
                              ensure_ascii=False, indent=2)
                    jf.flush()
                    os.fsync(jf.fileno())
                os.replace(tmp_json, out_json)
                print(f"# done seed={sd} arm={'on' if mem_on else 'off'} "
                      f"stop={stop} K={last_r['pop']} wall={time.time()-t0:.0f}s",
                      file=sys.stderr)
    finally:
        fout.close()
    print(f"# 摘要 -> {out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
