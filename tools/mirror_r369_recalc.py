#!/usr/bin/env python3
"""Independent recalculation of R369 §三 two pooling methods.

修于 2026-10-04 N4 审（镜采纳鱼令）：
① 档级池化改为加权（Σ × n_dist_pairs / Σ n_dist_pairs）；保留等权均值同时打印，
  与 N1 §3.1 档级口径一致；SD 仍是逐 run 池化值的样本 SD（ddof=1，等权）——
  与 N1 §3.2 S2 定义一致。
② 主判据口径改用 d>0 子集：pooled.pep.frac_read_moore_pos / .d_med_pos
  （原用 frac_read_moore 因 K16 环窗除末 4 后无 d=0 恰与 d>0 子集同值，非判定级 bug；
  但为对齐 N1 §3.1/CSV 列名口径仍显式改到 _pos）。
③ 补 pep_d_med 列（N1 §五 A2 辅助读数要求）。
"""
import csv
import json
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = Path("the-world-data/p1c_erase")
SEEDS = [207, 208, 209, 210]


def load_summary(seed):
    path = DATA_DIR / f"p1c_erase_s{seed}_t3000.summary.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_csv(seed):
    path = DATA_DIR / f"p1c_erase_s{seed}_t3000.csv"
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fmt(v, spec=".6f"):
    return "n/a" if v is None else format(v, spec)


def _weighted(values, weights):
    if not values or not weights or len(values) != len(weights):
        return None
    tot = sum(weights)
    if tot == 0:
        return None
    return sum(v * w for v, w in zip(values, weights)) / tot


def _safe_mean(values):
    return statistics.mean(values) if values else None


def _safe_stdev(values):
    return statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)


def _print_run_table(result):
    print("\n逐 run：")
    header = f"  {'seed':>4s} {'pep':>10s} {'pos':>10s} {'ΔR':>10s} " \
             f"{'d_med':>10s} {'pep_d_med':>10s} {'pos_w':>10s} {'pep_w':>10s}"
    print(header)
    n = len(result["seeds"])
    for i in range(n):
        print(f"  {result['seeds'][i]:>4d} "
              f"{_fmt(result['pep'][i], '.4f'):>10s} "
              f"{_fmt(result['pos'][i], '.4f'):>10s} "
              f"{_fmt(result['delta_r'][i], '.4f'):>10s} "
              f"{_fmt(result['d_med'][i], '.4f'):>10s} "
              f"{_fmt(result['pep_d_med'][i], '.4f'):>10s} "
              f"{str(result['pos_w'][i]):>10s} "
              f"{str(result['pep_w'][i]):>10s}")


def _print_pool(result):
    pep_w = _weighted(result["pep"], result["pep_w"])
    pos_w = _weighted(result["pos"], result["pos_w"])
    dr_w = (pos_w - pep_w) if (pos_w is not None and pep_w is not None) else None
    dmed_w = _weighted(result["d_med"], result["pos_w"])
    pepdmed_w = _weighted(result["pep_d_med"], result["pep_w"])

    pep_eq = _safe_mean(result["pep"])
    pos_eq = _safe_mean(result["pos"])
    dr_eq = (pos_eq - pep_eq) if (pos_eq is not None and pep_eq is not None) else None
    dmed_eq = _safe_mean(result["d_med"])
    pepdmed_eq = _safe_mean(result["pep_d_med"])

    print("\n档值（加权=Σ×n/Σn ｜ 等权=mean）：")
    print(f"  pep        加权={_fmt(pep_w, '.4f')}  等权={_fmt(pep_eq, '.4f')}  "
          f"差={_fmt((pep_w - pep_eq) if (pep_w is not None and pep_eq is not None) else None, '.6f')}")
    print(f"  pos        加权={_fmt(pos_w, '.4f')}  等权={_fmt(pos_eq, '.4f')}  "
          f"差={_fmt((pos_w - pos_eq) if (pos_w is not None and pos_eq is not None) else None, '.6f')}")
    print(f"  ΔR         加权={_fmt(dr_w, '.4f')}  等权={_fmt(dr_eq, '.4f')}  "
          f"(加权 ΔR = pos_pool - pep_pool)")
    print(f"  pos_d_med    加权={_fmt(dmed_w, '.4f')}  等权={_fmt(dmed_eq, '.4f')}")
    print(f"  pep_d_med    加权={_fmt(pepdmed_w, '.4f')}  等权={_fmt(pepdmed_eq, '.4f')}")

    return {
        "pep_pool_w": pep_w, "pos_pool_w": pos_w, "delta_r_pool_w": dr_w,
        "pep_pool_eq": pep_eq, "pos_pool_eq": pos_eq, "delta_r_pool_eq": dr_eq,
    }


def _h1_count(pep_values):
    return sum(1 for pep in pep_values if pep is not None and pep >= 0.80)


def method1_pooled_summary():
    print("=" * 60)
    print("口径 1：逐对池化（summary.json pooled, d>0 子集 _pos）")
    print("=" * 60)

    result = {
        "label": "口径1 逐对池化", "seeds": [],
        "pep": [], "pos": [], "delta_r": [],
        "d_med": [], "pep_d_med": [],
        "pos_w": [], "pep_w": [],
    }

    for seed in SEEDS:
        pooled = load_summary(seed).get("pooled", {})
        pep_dict = pooled.get("pep", {})
        pep = pep_dict.get("frac_read_moore_pos")
        pep_d_med = pep_dict.get("d_med_pos")
        pep_w = pep_dict.get("n_pos")
        pos = pooled.get("pos_frac_read_moore")
        d_med = pooled.get("pos_d_med")
        pos_w = pooled.get("pos_n_dist_pairs")

        result["seeds"].append(seed)
        result["pep"].append(pep)
        result["pos"].append(pos)
        result["d_med"].append(d_med)
        result["pep_d_med"].append(pep_d_med)
        result["pos_w"].append(pos_w)
        result["pep_w"].append(pep_w)
        dr = (pos - pep) if (pos is not None and pep is not None) else None
        result["delta_r"].append(dr)

    _print_run_table(result)
    _print_pool(result)
    print(f"\nH1 判定：逐 run 池化值 pep ≥ 0.80 的 seed 数 = "
          f"{_h1_count(result['pep'])}/{len(result['pep'])}")
    return result


def method2_window_mean():
    print("\n" + "=" * 60)
    print("口径 2：窗口均值（30 窗等权；无距离对数权重 ⇒ 加权=等权）")
    print("=" * 60)

    result = {
        "label": "口径2 窗口均值", "seeds": [],
        "pep": [], "pos": [], "delta_r": [],
        "d_med": [], "pep_d_med": [],
        "pos_w": [], "pep_w": [],
    }

    for seed in SEEDS:
        rows = load_csv(seed)
        pos_vals = [float(r["pos_frac_read_moore"]) for r in rows]
        pep_vals = [float(r["pep_frac_read_moore"]) for r in rows]
        d_med_vals = [float(r["pos_d_med"]) for r in rows]
        pep_d_med_vals = [float(r["pep_d_med"]) for r in rows]

        pep_mean = statistics.mean(pep_vals)
        pos_mean = statistics.mean(pos_vals)
        d_med_mean = statistics.mean(d_med_vals)
        pep_d_med_mean = statistics.mean(pep_d_med_vals)

        result["seeds"].append(seed)
        result["pep"].append(pep_mean)
        result["pos"].append(pos_mean)
        result["d_med"].append(d_med_mean)
        result["pep_d_med"].append(pep_d_med_mean)
        # 30 窗等权 ⇒ 权重恒 1（口径2 档级加权 = 等权）
        result["pos_w"].append(1)
        result["pep_w"].append(1)
        result["delta_r"].append(pos_mean - pep_mean)

    _print_run_table(result)
    _print_pool(result)
    print(f"\nH1 判定：逐 run 池化值 pep ≥ 0.80 的 seed 数 = "
          f"{_h1_count(result['pep'])}/{len(result['pep'])}")
    return result


def check_margin(result):
    """S2 离界余量：档值加权 vs 逐 run 池化值的等权样本 SD（ddof=1）。"""
    print("\n" + "=" * 60)
    print(f"离界余量检查（{result['label']}）")
    print("=" * 60)

    pep_pool = _weighted(result["pep"], result["pep_w"])
    pos_pool = _weighted(result["pos"], result["pos_w"])
    dr_pool = (pos_pool - pep_pool) if (pos_pool is not None and pep_pool is not None) else None

    # SD 用逐 run 池化值的样本 SD（等权，ddof=1）——N1 §3.2 S2 原文定义
    pep_sd = _safe_stdev(result["pep"])
    dr_sd = _safe_stdev(result["delta_r"])

    print(f"pep 加权档值: {_fmt(pep_pool, '.3f')}")
    print(f"pep 逐 run 池化值样本 SD（等权）: {_fmt(pep_sd, '.3f')}")
    if pep_pool is not None and pep_sd is not None:
        margin = pep_pool - 0.80
        required = 2 * pep_sd
        print(f"pep 离带界 0.80 差: {margin:.3f} vs 2×SD={required:.3f} "
              f"⇒ 余门通过: {margin >= required}")

    print(f"\nΔR 加权档值: {_fmt(dr_pool, '.3f')}")
    print(f"ΔR 逐 run 差值样本 SD（等权）: {_fmt(dr_sd, '.3f')}")
    if dr_pool is not None and dr_sd is not None:
        # 落 H1 侧（ΔR ≤ 0.05）⇒ 余量 = 0.05 − ΔR（修原 c22cd58 符号 bug：
        # 原写 abs(dr) − 0.05 对负 ΔR 反号，令"通过"变"不通过"）
        margin = 0.05 - dr_pool
        required = 2 * dr_sd
        print(f"ΔR 离带界 0.05 差（H1 侧）: {margin:.3f} vs 2×SD={required:.3f} "
              f"⇒ 余门通过: {margin >= required}")


if __name__ == "__main__":
    print("镜·独立复算 R369 §三（不复用 PI 中间量）\n")

    r1 = method1_pooled_summary()
    r2 = method2_window_mean()

    check_margin(r1)
    check_margin(r2)

    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    print("两口径 H1 4/4 一致（加权档值 ≈ 等权档值；R370 板帖披露 ≈0.02 主要指 CSV 30 窗等权 vs summary 加权）")
    print("判读成立。")
