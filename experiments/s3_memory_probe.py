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

🔴 v3（R287 T2 派工，2026-09-30，云归）：**续跑粒度 run 级 → sample 级**。
  背景：`--append` 的完成判据是"存在 tick==ticks 末行"，且行只在**整个 run 结束时**
  flush ⇒ 一旦在 run 中途被杀，该 run **所有采样行归零**（实测：seed 165 跑 49 min
  仍未落任何采样点）。60k 旗舰批 on 臂 25.6 h/run ⇒ 这种粒度等于"任何一次中断
  让 25.6 h 白跑"，长跑不可用。
  v3 改法：**每 `--save-every`（须 `== --sample`）个 tick 存一次引擎快照**（复用
  `steady_k_probe.save_ckpt` 三件套）+ **每次采样立刻 flush 行**；`--resume-sample`
  从快照的 `tick_saved` 接续、只丢最后不足一个采样周期的 tick。
  🔴 硬约束：**与从头跑逐位等价**（同机同树），不传新参数时**逐字节等于旧版**。

  🔴 v3 实现踩过的三个"静默错"坑（都在"清快照点之后的残行"这一步；测试
  `tests/test_r287_s3_sample_resume.py` 逐个钉住）：
    ① `os.replace` 换 inode，而 main 长期持有的 `fout` 指向孤儿
       ⇒ 续跑新行**一个字节都不落盘**（旧行完好、日志正常、无异常）；
    ② 与 `fout` 并存的第二个 `r+` 句柄 ⇒ 两个写位置互不知情
       ⇒ 把**别 run 的行**一起截掉；
    ③ 复用 `fout` 自己 `seek(0)` ⇒ `"a"` 模式下 O_APPEND 让 seek 失效
       ⇒ 重写的表头被追加到**文件尾**（表头重复）。
  ⇒ 定稿：**在打开 `fout` 之前**用独立句柄 `os.replace` 清行（无句柄持有期，最稳）。

⚠️ 已知缺口（v3 发现，**非本探针可修**，已上板）：引擎快照**不含** `_mem_v2_rng`
  （`sphere_engine.py:1362`，硬编码 `default_rng(20260928)`），而 `rng`/`_d2_rng`
  都存了。该流只在 `memory_noise=True` 时被消费（`sphere_engine.py:5319/5322`），
  而 `memory_noise` 引擎默认 False 且**全仓无调用点打开它** ⇒ 当前装置不可触发。
  一旦谁打开噪声档 ⇒ 续跑**不再逐位**（方位扰动流从种子起点重放）。
  本探针在 `--resume-sample` 下对该档 **fail-loud**（见 `_assert_snapshot_rng_coverage`）。

用法（v3）
----------
    # 首跑（开 sample 级快照）
    python -m experiments.s3_memory_probe --seeds 165 --ticks 60000 --sample 2000 \
        --save-every 2000 --snapshot-dir _rerun_logs/snap --out results/fl.csv
    # 中断后接续（快照存在 ⇒ 接续；不存在 ⇒ 当首跑）
    python -m experiments.s3_memory_probe --seeds 165 --ticks 60000 --sample 2000 \
        --resume-sample --save-every 2000 --snapshot-dir _rerun_logs/snap --out results/fl.csv
    # run 级续跑（旧口径，两者互斥）
    python -m experiments.s3_memory_probe ... --append --out results/fl.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# R98 纪律：非 ASCII 输出（如 ⇒、【】）在 GBK 控制台会 rc=1 假失败，入口统一兜底。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from simulation.sphere_engine import SphereEngine
from simulation.genes import Gene            # R275 T2：g22 记忆权重基因统计
from experiments.steady_k_probe import (
    make_cfg, apply_post_build, save_ckpt, load_ckpt,
)
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


# ---------- R275 T2：g22（记忆权重基因）活体统计 ----------

def _g22_stats(eng, N: int) -> dict:
    """取活体 `eng._genes[:N, Gene.MEMORY_WEIGHT]` 的均值/分位 + 按代分层（R280 S3.5b）。

    为什么用活体切片：`_genes` 的容量随种群浮动 ⇒ 只有前 N 行是活体（与 `eng._flat` 同序）。
    g22 仅被 memory_v2 路径消费（`sphere_engine.py:5374`，eff = cfg_gain × 2×g22）⇒
    在 mem_on 臂上若值被选择，应看到均值/分位漂移，mem_off 臂则为其漂变对照。

    R280 增补：按代分层（gen0..gen3+）+ 最新一代（max_gen），排除"老一代 g22≈0.5 稀释新生代信号"。
    核心指标 = g22_mean_latest − g22_mean_gen0。

    🔴 只读，不改任何行为；N=0（灭绝）⇒ 报 NaN 而非 0，避免与"基因全 0"混淆。
    某代无个体 ⇒ 该代均值 NaN，pop_by_gen 对应位 = 0。
    """
    _nan_cols = ("g22_mean_gen0", "g22_mean_gen1", "g22_mean_gen2",
                 "g22_mean_gen3p", "g22_mean_latest")
    _zero_cols = ("pop_gen0", "pop_gen1", "pop_gen2", "pop_gen3p", "max_gen")
    if N <= 0:
        out = {"g22_mean": float("nan"), "g22_p10": float("nan"),
               "g22_p90": float("nan")}
        for k in _nan_cols:
            out[k] = float("nan")
        for k in _zero_cols:
            out[k] = 0
        return out
    genes = getattr(eng, "_genes", None)
    if genes is None or genes.ndim != 2 or genes.shape[0] < N \
            or genes.shape[1] <= int(Gene.MEMORY_WEIGHT):
        out = {"g22_mean": float("nan"), "g22_p10": float("nan"),
               "g22_p90": float("nan")}
        for k in _nan_cols:
            out[k] = float("nan")
        for k in _zero_cols:
            out[k] = 0
        return out
    g22 = np.asarray(genes[:N, int(Gene.MEMORY_WEIGHT)], dtype=float)
    out = {
        "g22_mean": round(float(g22.mean()), 6),
        "g22_p10": round(float(np.percentile(g22, 10)), 6),
        "g22_p90": round(float(np.percentile(g22, 90)), 6),
    }
    # ---- R280：按代分层 ----
    gen_arr = getattr(eng, "_generation", None)
    if gen_arr is not None and gen_arr.shape[0] >= N:
        gens = np.asarray(gen_arr[:N], dtype=np.int64)
        max_gen = int(gens.max()) if N > 0 else 0
        out["max_gen"] = max_gen
        for gi, key in enumerate(("gen0", "gen1", "gen2")):
            mask = gens == gi
            cnt = int(mask.sum())
            out[f"pop_{key}"] = cnt
            out[f"g22_mean_{key}"] = round(float(g22[mask].mean()), 6) if cnt > 0 else float("nan")
        # gen3+ = 第3代及以上合并
        mask3p = gens >= 3
        cnt3p = int(mask3p.sum())
        out["pop_gen3p"] = cnt3p
        out["g22_mean_gen3p"] = round(float(g22[mask3p].mean()), 6) if cnt3p > 0 else float("nan")
        # 最新一代（max_gen）
        mask_latest = gens == max_gen
        out["g22_mean_latest"] = round(float(g22[mask_latest].mean()), 6) if mask_latest.sum() > 0 else float("nan")
    else:
        for k in _nan_cols:
            out[k] = float("nan")
        for k in _zero_cols:
            out[k] = 0
    return out


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


# ---------- v3：sample 级续跑辅助（R287 T2） ----------

def _resume_tag(seed, mem_on) -> str:
    """本探针的快照 tag（与 `steady_k_probe.run_tag` 的命名空间刻意区分开）。

    `steady_k_probe.run_tag` 是 `w{rows}x{cols}_p{patches}_s{seed}_k{k}_rgm{rgm}[_rd]`；
    本探针另起 `s3_s{seed}_{arm}` ⇒ 两套工具共用一个 `--snapshot-dir` 时**不会互顶**。
    """
    return f"s3_s{seed}_{'mem_on' if mem_on else 'mem_off'}"


def _probe_sidecar_path(snapshot_dir, tag) -> Path:
    """探针侧车（**引擎快照不含的探针累积量**）。扁平放 `snapshot_dir` 下 ⇒ 天然在
    `.gitignore` 的 `_rerun_logs/snap/*.npz` 白名单内，零 ignore 改动。"""
    return Path(snapshot_dir) / f"{tag}.probe.npz"


def _save_probe_sidecar(path, *, visited_mask, patch_ever_visited, first_visit,
                        init_cap_full, init_patch_cap, init_cap_sum,
                        n_patch_total) -> None:
    """存探针侧车。

    🔴 为什么必须存：这些量**不在引擎快照里**（引擎只认自己的状态），而采样读数
    直接依赖它们（`never_visited_frac` / `bg_resid_frac` / `l1_visited_patch_frac` /
    `cap_lost_frac`）。不存 ⇒ 续跑后 `cap_lost_frac` 会从 0 重算、`visited_mask` 从零开始
    ⇒ **数值静默错**（比崩更糟）。

    `first_visit` 是 `dict{label: tick}` ⇒ 拆成两个平行数组存（**不用 pickle**：
    保持 npz 纯数值、跨机可读、无需 `allow_pickle` 的隐式依赖）。
    """
    labels = np.fromiter(first_visit.keys(), dtype=np.int64,
                         count=len(first_visit))
    ticks = np.fromiter(first_visit.values(), dtype=np.int64,
                        count=len(first_visit))
    pcap_lab = np.fromiter(init_patch_cap.keys(), dtype=np.int64,
                           count=len(init_patch_cap))
    pcap_val = np.fromiter(init_patch_cap.values(), dtype=np.float64,
                           count=len(init_patch_cap))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(p),
        visited_mask=np.asarray(visited_mask, dtype=bool),
        patch_ever_visited=np.asarray(patch_ever_visited, dtype=bool),
        first_visit_labels=labels, first_visit_ticks=ticks,
        init_cap_full=np.asarray(init_cap_full, dtype=np.float64),
        init_patch_cap_labels=pcap_lab, init_patch_cap_vals=pcap_val,
        init_cap_sum=np.array(float(init_cap_sum)),
        n_patch_total=np.array(int(n_patch_total)),
    )


def _load_probe_sidecar(path):
    """读探针侧车 ⇒ 与 `_save_probe_sidecar` 同构的 dict。缺文件 ⇒ 抛 FileNotFoundError。"""
    d = np.load(str(path), allow_pickle=False)
    return {
        "visited_mask": d["visited_mask"].copy(),
        "patch_ever_visited": d["patch_ever_visited"].copy(),
        "first_visit": {int(k): int(v) for k, v in
                        zip(d["first_visit_labels"], d["first_visit_ticks"])},
        "init_cap_full": d["init_cap_full"].copy(),
        "init_patch_cap": {int(k): float(v) for k, v in
                           zip(d["init_patch_cap_labels"], d["init_patch_cap_vals"])},
        "init_cap_sum": float(d["init_cap_sum"]),
        "n_patch_total": int(d["n_patch_total"]),
    }


def _config_fingerprint_check(eng, seed, rows, cols, pop, patches, rgm,
                              mem_on, bg_low_prod_frac, bg_low_cap_mult,
                              weight_gene) -> None:
    """fail-loud 逐字段核对（**慢**，但只在续跑时跑一次）。

    为什么不用 `config.fingerprint()` 一把比：本探针在 `make_cfg` 之后**又改了**三个
    字段（`use_sim_core=False` / `resource_dynamics.enabled` / `memory_v2`+`gradient`
    [+ `memory_weight_gene`]）—— 快照存的指纹是"改完之后的"，直接比也能中；但一旦将来
    谁在 `run_one` 里多改一个字段，指纹比对的**报错信息**只会说"指纹不等"，无法定位。
    逐字段核对给出的报错能直接指到字段 ⇒ 排查成本从"翻配置"降到"读一行报错"。
    """
    cfg = eng.config
    got = {
        "seed": int(cfg.seed),
        "rows": int(cfg.world.rows),
        "cols": int(cfg.world.cols),
        "patches": int(cfg.resources.patch_count),
        "rgm": float(cfg.resources.patch_regrowth_mult),
        "use_sim_core": bool(cfg.simulation.use_sim_core),
        "rd_enabled": bool(cfg.resource_dynamics.enabled),
        "bg_production_zero": bool(cfg.resources.bg_production_zero),
        "memory_v2": bool(cfg.info_structure.memory_v2),
        "memory_gradient": str(cfg.info_structure.memory_gradient),
        "memory_weight_gene": bool(getattr(cfg.info_structure,
                                           "memory_weight_gene", False)),
    }
    want = {
        "seed": int(seed), "rows": int(rows), "cols": int(cols),
        "patches": int(patches), "rgm": float(rgm),
        "use_sim_core": False, "rd_enabled": True, "bg_production_zero": True,
        "memory_v2": bool(mem_on),
        "memory_gradient": "orientation" if mem_on else "none",
        "memory_weight_gene": bool(weight_gene and mem_on),
    }
    diff = {k: (want[k], got[k]) for k in want if got.get(k) != want[k]}
    if diff:
        raise ValueError(
            f"🔴 续跑装置不一致（快照 vs 命令行）：{diff}"
            f"（格式 字段: (命令行, 快照)）⇒ 用同一套参数再来，"
            f"否则续出来的 run 不是同一个 run")
    # pop 不在 config 字段里（初始种群数），只能核对快照里的活体规模是否"合理"：
    # 不核对——engine 的 pop 由生灭决定，与 `--pop` 无关（`--pop` 只影响初始 N）。
    _ = pop


def _assert_snapshot_rng_coverage(eng) -> None:
    """fail-loud：`memory_noise=True` 时引擎快照**不覆盖** `_mem_v2_rng` ⇒ 续跑不逐位。

    这是 v3 发现的**引擎级缺口**（`sphere_engine.py:1362` 硬编码 `default_rng(20260928)`，
    `save_snapshot` 存了 `rng`/`_d2_rng` 但漏它）。当前装置 `memory_noise=False`
    ⇒ 不触发。**宁可炸也不要静默产出"看着逐位、实则不逐位"的结果。**
    """
    if bool(getattr(eng, "_mem_noise", False)):
        raise ValueError(
            "🔴 memory_noise=True 但引擎快照不含 `_mem_v2_rng`"
            "（sphere_engine.py:1362 的独立流）⇒ 从快照续跑**不逐位**。"
            "请改用 --append（run 级续跑）或先补引擎快照键。")


def _strip_rows_after(csv_path: str, key: tuple[str, str], start_tick: int) -> int:
    """只删 **本 run** 的 `tick > start_tick` 行（其余 run 与其余采样点原样保留）。

    🔴 为什么**不能**调用 `_strip_incomplete`：它的语义是"未完成 run 的行**全删**"
    —— 对 `--resume-sample` 而言，要保留的恰恰是那批采样点（它们有效且已完成），
    调它会**删掉续跑的起点行**。

    🔴 为什么**不用 `os.replace`**（`_strip_incomplete` 用的是它）：本函数由 main 在
    `fout` **长期打开**期间调用。`os.replace(tmp, csv_path)` 会把 `csv_path` 换成一个
    **新 inode**，而 `fout` 仍指向已被 unlink 的旧 inode ⇒ 后续 `writerow` 全部写进
    **孤儿 inode**、一个字节都不落盘（本实现第一版就栽在这，现象=续跑后旧行完好、
    新行全无）。⇒ 改为**原地重写**（`r+` + `truncate(0)`），inode 不变、句柄继续有效。
    ⚠️ 前提：调用方已 `flush()` + `fsync()`（否则会把自己的缓冲盖掉）。
    """
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        keep, drop = [], 0
        for row in reader:
            if (str(row.get("seed")), str(row.get("arm"))) != key:
                keep.append(row)
                continue
            try:
                tk = int(float(row.get("tick", -1)))
            except (TypeError, ValueError):
                keep.append(row)
                continue
            if tk > int(start_tick):
                drop += 1
            else:
                keep.append(row)
    if drop == 0 or not fieldnames:
        return 0
    # 🔴 原地重写（inode 不变）
    with open(csv_path, "r+", newline="", encoding="utf-8") as f:
        f.seek(0)
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(keep)
        f.truncate()
        f.flush()
        os.fsync(f.fileno())
    return drop


def _last_sample_tick(csv_path: str, key: tuple[str, str]) -> int:
    """本 run 已落盘的**最大**采样 tick（无行 ⇒ -1）。"""
    last = -1
    if not os.path.exists(csv_path):
        return last
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (str(row.get("seed")), str(row.get("arm"))) != key:
                continue
            try:
                last = max(last, int(float(row.get("tick", -1))))
            except (TypeError, ValueError):
                continue
    return last


def run_one(seed, rows, cols, pop, patches, ticks, sample, rgm, mem_on,
            bg_low_prod_frac=0.4, bg_low_cap_mult=0.05, weight_gene=False,
            save_every=0, snapshot_dir=None, resume_sample=False,
            prior_rows=None, out_path=None):
    """跑一个 S3 run（单 seed 单臂），可选 **sample 级续跑**。

    rd 恒开；mem_on 决定记忆 v2 开关（其余逐字段对齐 ⇒ 单变量）。
    weight_gene（R275 T2）：在 **mem_on 臂**额外打开 `memory_weight_gene`
    （g22 被 13.11 路径消费）⇒ S3.5 记忆基因演化。默认 False ⇒ 逐位等于旧版。

    v3（R287 T2）新增三个续跑参数，**默认全关 ⇒ 逐字节等于旧版行为**：
      · `save_every > 0` + `snapshot_dir` ⇒ 每 N tick 存快照 + 侧车（N 须 == sample）
      · `resume_sample=True` ⇒ 从 `snapshot_dir/<tag>.snapshot.npz` 接续；
        `ticks` 仍是**绝对目标 tick**
    🔴 `run_one` **不自己动 `out_path`**：清残行（`_strip_rows_after`）由 main 在
      "`fout` 已 flush 且尚未开始本 run" 的窗口里做 —— 若在这里 `os.replace` 整个
      CSV 文件，会把 `fout` 缓冲里**还没落盘的行**一起覆盖掉（本实现的第一版就栽在这）。
    """
    tag = _resume_tag(seed, mem_on)
    start_tick = 0
    if resume_sample:
        _err = _fail_save_every(save_every, sample, snapshot_dir)
        if _err:
            raise ValueError(_err)
        snap_path = Path(snapshot_dir) / f"{tag}.snapshot.npz"
        if not snap_path.exists():
            raise FileNotFoundError(
                f"🔴 --resume-sample 但快照不存在：{snap_path}"
                f"（首跑请**不带** --resume-sample，并给 --save-every/--snapshot-dir）")
        eng, meta, start_tick = load_ckpt(snap_path)
        _assert_snapshot_rng_coverage(eng)
        # 🔴 装置一致性三重核对（文件名 / meta / config 逐字段）
        if str(meta.get("tag")) != tag:
            raise ValueError(
                f"🔴 快照 tag 不符：meta={meta.get('tag')!r} vs 期望 {tag!r}"
                f"（--snapshot-dir 里混进了别的 run 的产物？）")
        if meta.get("sample") is not None and int(meta["sample"]) != int(sample):
            raise ValueError(
                f"🔴 采样节拍不一致：快照 sample={int(meta['sample'])} vs "
                f"命令行 --sample={int(sample)} ⇒ 行网格会静默错位，用同一个 --sample 再来")
        _config_fingerprint_check(
            eng, seed, rows, cols, pop, patches, rgm, mem_on,
            bg_low_prod_frac, bg_low_cap_mult, weight_gene)
        side = _load_probe_sidecar(_probe_sidecar_path(snapshot_dir, tag))
        init_cap_full = side["init_cap_full"]
        labels = _label_patches(eng.world, eng.resources._patch_mask)
        n_patch_total = side["n_patch_total"]
        init_patch_cap = side["init_patch_cap"]
        init_cap_sum = side["init_cap_sum"]
        visited_mask = side["visited_mask"]
        patch_ever_visited = side["patch_ever_visited"]
        first_visit = side["first_visit"]
        # 🔴 世界格数变了 ⇒ 侧车对齐失效（换了世界尺寸/掩码）
        if visited_mask.shape != (eng.world.n_cells,) \
                or patch_ever_visited.shape != (n_patch_total,):
            raise ValueError(
                f"🔴 侧车形状与世界不符：visited_mask{visited_mask.shape} vs "
                f"({eng.world.n_cells},) / patch_ever_visited{patch_ever_visited.shape} "
                f"vs ({n_patch_total},) ⇒ --rows/--cols/--patches 是否被改过？")
        # 🔴 主循环从 `_tick` 接续（load 已恢复 `_tick`）
        start_tick = int(eng._tick)
        rows_out = []
    else:
        (init_cap_full, labels, n_patch_total, init_patch_cap, init_cap_sum,
         visited_mask, patch_ever_visited, first_visit,
         rows_out), eng = _build_fresh_run(
            seed, rows, cols, pop, patches, rgm, mem_on,
            bg_low_prod_frac, bg_low_cap_mult, weight_gene)
    return _loop_impl(eng, seed, mem_on, rows_out, start_tick, ticks, sample,
                      labels, n_patch_total, init_cap_full, init_patch_cap,
                      init_cap_sum, visited_mask, patch_ever_visited, first_visit,
                      save_every, snapshot_dir, tag)


def _fail_save_every(save_every, sample, snapshot_dir):
    """`--resume-sample` 的前置校验（返回错误串或 None）。"""
    if not save_every:
        return ("🔴 --resume-sample 需要 --save-every > 0（否则没有快照可续）")
    if save_every != sample:
        return (f"🔴 --save-every({save_every}) 必须 == --sample({sample})："
                f"采样节拍决定行网格，快照节拍必须与它同步，"
                f"否则续跑落下的行网格与连续跑**不一致**（静默错位）")
    if not snapshot_dir:
        return "🔴 --resume-sample 需要 --snapshot-dir"
    return None


def _peek_start_tick(snapshot_dir, tag) -> int:
    """只读快照 meta 的 `tick_saved`（不构造引擎 ⇒ 便宜）。缺文件 ⇒ 0。"""
    p = Path(snapshot_dir) / f"{tag}.snapshot.meta.json"
    if not p.exists():
        return 0
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return int(json.load(fh).get("tick_saved", 0))
    except (OSError, ValueError, TypeError):
        return 0


def _build_fresh_run(seed, rows, cols, pop, patches, rgm, mem_on,
                     bg_low_prod_frac, bg_low_cap_mult, weight_gene):
    """首跑：建引擎 + 探针初始累积量（与旧版 `run_one` 逐行等价）。"""
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
        # 🔴 R275 T2：S3.5 处理臂 —— 打开 g22 消费（仅 mem_on 臂；config 会 fail-loud
        #   拦住 "weight_gene ∧ ¬memory_v2" ⇒ 此处天然只在处理臂生效）
        if weight_gene:
            c.info_structure.memory_weight_gene = True
    else:
        c.info_structure.memory_v2 = False
        c.info_structure.memory_gradient = "none"
        # 对照臂恒不消费 g22（纯漂变）⇒ weight_gene 不生效，显式断言防静默错配
        if weight_gene:
            print("  ⚠️ weight_gene=True 但本臂 memory_v2=False ⇒ g22 不被消费"
                  "（= 漂变对照，符合 S3.5 设计）。", file=sys.stderr)

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

    visited_mask = np.zeros(eng.world.n_cells, dtype=bool)   # 是否曾被任何个体访问
    first_visit = {}                                          # patch_label -> tick
    patch_ever_visited = np.zeros(n_patch_total, dtype=bool)
    return ((init_cap_full, labels, n_patch_total, init_patch_cap, init_cap_sum,
             visited_mask, patch_ever_visited, first_visit, []), eng)


def _loop(seed, mem_on, rows_out, start_tick, ticks, sample, labels,
          n_patch_total, init_cap_full, init_patch_cap, init_cap_sum,
          visited_mask, patch_ever_visited, first_visit,
          save_every, snapshot_dir, tag):
    """采样主循环（首跑 / 续跑**共用**）。

    🔴 续跑时 `rows_out` 从**空**开始（本段新行），旧行已在 CSV 里、由 main 的
    `--resume-sample` 分支原样保留 ⇒ 不把旧行读进内存再写一遍（避免重写时抖精度）。

    🔴 落盘顺序 = **先写行 + fsync，再存快照 + 侧车**：快照里的 `tick_saved` 是
    "哪个 tick 的行已落盘"的**唯一权威**。反过来的话，崩在"存完快照、行还没落"
    之间 ⇒ 续跑会从 `tick_saved` 之后开始 ⇒ **静默丢一个采样点**。
    """
    raise NotImplementedError("内部函数，首跑/续跑一律经 `run_one` 调用")


def _loop_impl(eng, seed, mem_on, rows_out, start_tick, ticks, sample, labels,
               n_patch_total, init_cap_full, init_patch_cap, init_cap_sum,
               visited_mask, patch_ever_visited, first_visit,
               save_every, snapshot_dir, tag):
    """采样主循环（首跑 / 续跑共用）——见 `_loop` 的落盘顺序说明。"""
    t_slice = time.perf_counter()
    last = start_tick
    stop = "ticks"

    for t in range(start_tick + 1, ticks + 1):
        eng.step()
        flat = eng._flat
        visited_mask[flat] = True
        for p in np.unique(labels[flat]):
            if p >= 0 and not patch_ever_visited[p]:
                patch_ever_visited[p] = True
                first_visit[int(p)] = t

        sampled = (t % sample == 0) or (t == ticks)
        if sampled:
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
                # ---- R275 T2：g22 活体统计（S3.5 记忆基因演化）----
                **_g22_stats(eng, N),
            })
            if stop == "灭绝":
                break

        # 🔴 v3：快照节拍。**必须 `save_every == sample`**（由 `_fail_save_every` 在
        #    resume 侧把关；首跑侧在 main 里把关）⇒ 快照点恒是采样点，
        #    "最后落盘的行" 与 "快照 tick" 一一对应，续跑边界无歧义。
        if save_every and sampled and t < ticks:
            save_ckpt(eng, snapshot_dir, tag,
                      {"tag": tag, "sample": int(sample), "ticks": int(ticks),
                       "seed": int(seed),
                       "arm": "mem_on" if mem_on else "mem_off",
                       "weight_gene": bool(getattr(eng, "_mem_weight_gene", False))})
            _save_probe_sidecar(
                _probe_sidecar_path(snapshot_dir, tag),
                visited_mask=visited_mask,
                patch_ever_visited=patch_ever_visited,
                first_visit=first_visit,
                init_cap_full=init_cap_full,
                init_patch_cap=init_patch_cap,
                init_cap_sum=init_cap_sum,
                n_patch_total=n_patch_total)
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
    ap.add_argument("--weight-gene", action="store_true",
                    help="R275 T2 / S3.5：处理臂打开 `memory_weight_gene`（g22 被消费）；"
                         "默认关 = 逐位等于旧版 S3（固定权重批）")
    ap.add_argument("--out", default="results/s3_memory.csv")
    # ---- v3（R287 T2）：两种续跑粒度**互斥**（对"残行归属"的判定相反）----
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--append", action="store_true",
                     help="run 级续跑：追加到已存在的 --out，跳过已完成的 (seed,arm)，"
                          "并先删未完成 run 的**全部**残行（默认关 = 覆盖写，向后兼容）")
    grp.add_argument("--resume-sample", action="store_true",
                     help="sample 级续跑（R287 T2）：从 --snapshot-dir 里本 run 的快照"
                          "接续（只丢不足一个采样周期）；须配 --save-every == --sample。"
                          "与 --append 互斥：本模式**只删快照点之后**的行，保留之前的采样点")
    ap.add_argument("--save-every", type=int, default=0,
                    help="每 N tick 存一次快照 + 探针侧车（默认 0 = 关；"
                         "用 --resume-sample 时**必须 == --sample**）")
    ap.add_argument("--snapshot-dir", default="",
                    help="快照目录（默认 _rerun_logs/snap/，已在 .gitignore 白名单）")
    ap.add_argument("--keep-ckpt", action="store_true",
                    help="run 正常收尾后**保留**快照（默认收尾即清，防过期快照把下次"
                         "跑偏进“接续”分支）。**测试/演练断点场景时用它**")
    a = ap.parse_args()

    # ---- v3 前置校验：快照节拍必须与采样节拍同步（否则续跑行网格静默错位）----
    if a.save_every and a.save_every != a.sample:
        print(f"🔴 --save-every({a.save_every}) 必须 == --sample({a.sample})"
              f"（快照点须落在采样点上，否则续跑边界有歧义）。中止。", file=sys.stderr)
        sys.exit(2)
    if a.resume_sample and not a.snapshot_dir:
        print("🔴 --resume-sample 需要 --snapshot-dir（快照写哪儿）。中止。",
              file=sys.stderr)
        sys.exit(2)
    if not a.snapshot_dir:
        a.snapshot_dir = os.path.join("_rerun_logs", "snap")

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
              "l0_peak_dead", "l0_peak_rest",
              # ---- R275 T2：g22 活体统计（S3.5 记忆基因演化）----
              "g22_mean", "g22_p10", "g22_p90",
              # ---- R280 S3.5b：按代分层 g22（排除老一代稀释）----
              "g22_mean_gen0", "g22_mean_gen1", "g22_mean_gen2", "g22_mean_gen3p",
              "g22_mean_latest", "pop_gen0", "pop_gen1", "pop_gen2", "pop_gen3p",
              "max_gen"]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out_json = os.path.splitext(a.out)[0] + ".summary.json"

    print(f"# S3 探针 v3：rows={a.rows} cols={a.cols} patches={a.patches} "
          f"pop={a.pop} rgm={a.rgm} ticks={a.ticks} seeds={seeds} "
          f"arms={['on' if m else 'off' for m in arms]} "
          f"bg_low_frac={a.bg_low_prod_frac} bg_low_mult={a.bg_low_cap_mult} "
          f"weight_gene={a.weight_gene} append={a.append} "
          f"resume_sample={a.resume_sample} save_every={a.save_every}")

    # ---- v2：run 级断点续跑准备（R264）----
    done = set()
    if a.append:
        done = _scan_done_runs(a.out, a.ticks)
        if done:
            dropped = _strip_incomplete(a.out, done)
            print(f"# 续跑：已完成 {len(done)} 个 (seed,arm)，清理未完成残行 {dropped} 行",
                  file=sys.stderr)
        else:
            print("# 续跑：无已完成 run（或 --out 不存在）⇒ 从头跑", file=sys.stderr)

    # ---- v3：sample 级续跑准备（R287 T2）----
    # 逐 run 看快照存在与否：**有快照 ⇒ 接续；无快照 ⇒ 就当首跑**（同一批里
    # "跑完的 run 已被清掉快照" 与 "还没跑的 run 没快照" 都落到首跑分支，
    # 语义正确且不需要额外开关）。
    resume_map: dict[tuple[str, str], bool] = {}
    if a.resume_sample:
        os.makedirs(a.snapshot_dir, exist_ok=True)
        for sd in seeds:
            for mem_on in arms:
                arm = "mem_on" if mem_on else "mem_off"
                _tag = _resume_tag(sd, mem_on)
                resume_map[(str(sd), arm)] = \
                    os.path.exists(os.path.join(a.snapshot_dir,
                                                f"{_tag}.snapshot.npz"))
        _n = sum(1 for v in resume_map.values() if v)
        print(f"# sample 级续跑：{_n}/{len(resume_map)} 个 run 有快照 ⇒ 接续；"
              f"其余首跑（快照目录 {a.snapshot_dir}）", file=sys.stderr)
    else:
        for sd in seeds:
            for mem_on in arms:
                resume_map[(str(sd), "mem_on" if mem_on else "mem_off")] = False

    # 已完成 run 的 summary 继承（两种续跑都不让旧 summary 丢）
    summary = []
    if (a.append or a.resume_sample) and os.path.exists(out_json):
        try:
            with open(out_json, "r", encoding="utf-8") as jf:
                summary = list(json.load(jf).get("runs", []))
        except (OSError, ValueError) as e:
            print(f"  ⚠️ 旧 summary 读取失败（忽略）：{e}", file=sys.stderr)

    # 🔴 v2 核心：追加模式不重写表头；文件为空（新建/刚清空）时才写
    _append_mode = a.append or a.resume_sample
    need_header = (not _append_mode) or (not os.path.exists(a.out)) \
        or (os.path.getsize(a.out) == 0)

    # 🔴 v3：sample 级续跑的"清残行"**必须在打开 `fout` 之前**做。
    #   踩过三个坑（全是"静默错"，只有对拍才发现）：
    #     ① 用 `os.replace` ⇒ CSV 换 inode，而 `fout` 指向孤儿 ⇒ 后续行全丢；
    #     ② 与 `fout` 并存的第二个 `r+` 句柄 ⇒ 写位置互不知情 ⇒ 截掉别 run 的行；
    #     ③ 复用 `fout` 自己 `seek(0)` ⇒ **`"a"` 模式下 O_APPEND 让 seek 失效**，
    #        重写的表头被追加到文件尾 ⇒ 表头重复。
    #   ⇒ 顺序改为：**先清（无句柄持有）→ 再打开 `fout`**。
    if a.resume_sample:
        for sd in seeds:
            for mem_on in arms:
                if not resume_map.get((str(sd), "mem_on" if mem_on else "mem_off")):
                    continue
                _arm = "mem_on" if mem_on else "mem_off"
                _st = _peek_start_tick(a.snapshot_dir, _resume_tag(sd, mem_on))
                _dd = _strip_rows_after(a.out, (str(sd), _arm), _st)
                if _dd:
                    print(f"# 续跑 seed={sd} arm={_arm}：清掉快照点({_st})之上的"
                          f"{_dd} 行残行", file=sys.stderr)

    fout = open(a.out, "a" if _append_mode else "w", newline="", encoding="utf-8")
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
                _res = resume_map.get((str(sd), arm), False)
                # v3：续跑时"本段新行"从空开始，旧行留在 CSV 里（已在上面的
                # 清行段处理过）。**不能用 --append 的 prior_rows 合并**：
                # 那会把旧行读进来再重写一遍 ⇒ 浮点 repr 往返抖精度。
                t0 = time.time()
                rows_out, stop = run_one(
                    sd, a.rows, a.cols, a.pop, a.patches, a.ticks, a.sample,
                    a.rgm, mem_on, a.bg_low_prod_frac, a.bg_low_cap_mult,
                    a.weight_gene,
                    save_every=a.save_every, snapshot_dir=a.snapshot_dir,
                    resume_sample=_res, prior_rows=None, out_path=a.out)
                for r in rows_out:
                    w.writerow({h: r[h] for h in header})
                    # 🔴 v3：**逐行 flush + fsync**（原来只在 run 末 flush）——
                    #    这是"sample 级"的另一半：行与快照同节拍落盘，
                    #    否则崩在 run 中途仍然是"行全丢、快照白存"。
                    fout.flush()
                    os.fsync(fout.fileno())
                print(f"# flushed seed={sd} arm={arm} rows={len(rows_out)}"
                      f"{' (接续自快照)' if _res else ''}", file=sys.stderr)

                if not rows_out:
                    print(f"  ⚠️ seed={sd} arm={arm} 本段无可落盘行 ⇒ 跳过 summary",
                          file=sys.stderr)
                    continue
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
                # 🔴 v3：run **正常收尾** ⇒ 清掉快照 + 侧车。否则该 run 下次仍会被
                #    判为"有快照 ⇒ 接续"，从过期快照重跑一段（数据虽等价但白烧机时）。
                #    只清"跑到目标 / 灭绝"的，**不清超时/异常留下的**（那些要留着续）。
                if a.save_every and stop in ("ticks", "灭绝") and not a.keep_ckpt:
                    _drop_ckpt(a.snapshot_dir, _resume_tag(sd, mem_on))
    finally:
        fout.close()
    print(f"# 摘要 -> {out_json}", file=sys.stderr)


def _drop_ckpt(snapshot_dir, tag) -> None:
    """删掉该 run 的快照三件套 + 探针侧车（run 正常收尾后的清理；缺文件静默）。"""
    base = Path(snapshot_dir)
    for p in (base / f"{tag}.snapshot.npz",
              base / f"{tag}.snapshot.rngstate.pkl",
              base / f"{tag}.snapshot.meta.json",
              base / f"{tag}.probe.npz"):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            print(f"  ⚠️ 清理快照失败（无害）：{p} — {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
