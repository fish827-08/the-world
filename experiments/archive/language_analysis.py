#!/usr/bin/env python3
"""
语言涌现分析脚本（language_analysis.py）

从快照文件中读取信号场、解读表和种群数据，计算语言涌现的量化指标。
7 个维度：
  1. 信号词汇统计（频率/熵/Zipf 拟合度）
  2. 文化多样性（解读表标准差，分模式细化）
  3. 代际文化稳定性（相邻世代解读表相关性）
  4. 空间聚类（信号模式的空间自相关 Moran's I）
  5. 信号-语境互信息（信号模式与局部环境的 MI）
  6. 解读一致性（种群内对相同信号的解读收敛度）
  7. 信号基因-表型相关性（g15 与信号发射行为的相关）

用法：
  python3 experiments/language_analysis.py snapshot.npz
  python3 experiments/language_analysis.py snapshot.npz --output lang_metrics.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print 会让脚本 rc=1 假失败；F-R15 族）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # 非 TTY / 旧解释器：不因诊断能力缺失而阻断运行


def _file_sha256(path: str) -> str:
    """计算文件的 SHA256 哈希。"""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sim_core_sha256() -> str:
    """计算 sim_core.so 的 SHA256（若存在）。"""
    import os
    sim_core_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sim_core.so")
    if os.path.exists(sim_core_path):
        return _file_sha256(sim_core_path)
    return "not_found"


def load_snapshot(path: str) -> dict:
    """加载快照，只保留存活个体的数据。"""
    d = np.load(path, allow_pickle=True)
    count = int(d["count"])
    # 提取配置字典中的 seed（兼容新旧快照格式）
    config_seed = None
    config_dict = None
    if "config_dict" in d.files:
        try:
            config_dict = d["config_dict"].item()
            if isinstance(config_dict, dict):
                config_seed = config_dict.get("seed")
        except Exception:
            pass
    return {
        "tick": int(d["tick"]),
        "count": count,
        "max_generation": int(d["max_generation"]),
        "signal_marks": d["signal_marks"],       # (n_cells,) uint8, 低4位=pattern
        "signal_age": d["signal_age"],             # (n_cells,) int32
        "interpret": d["interpret"][:count],       # (P, 16)
        "genes": d["genes"][:count],               # (P, 24)
        "flat": d["flat"][:count],                 # (P,)
        "generation": d["generation"][:count],     # (P,)
        "energy": d["energy"][:count],             # (P,)
        "n_cells": d["signal_marks"].shape[0],
        # 元数据（F-D11 自描述 + F-R2）
        "resource_grid": d["resource_grid"] if "resource_grid" in d.files else np.zeros(d["signal_marks"].shape[0]),
        "config_seed": config_seed,
        "config_fingerprint": str(d["config_fingerprint"]) if "config_fingerprint" in d.files else None,
        "use_sim_core": bool(d["use_sim_core"]) if "use_sim_core" in d.files else None,
        "snapshot_version": int(d["snapshot_version"]) if "snapshot_version" in d.files else None,
        "snapshot_sha256": _file_sha256(path),
        "sim_core_sha256": _sim_core_sha256(),
    }


# ─── 维度 1：信号词汇统计 ───────────────────────────────────────────

def signal_vocabulary(signal_marks: np.ndarray, signal_age: np.ndarray) -> dict:
    """16 种信号模式的使用频率、熵、Zipf 拟合度。"""
    # 只统计活跃信号（age >= 0，0 表示无信号）
    active = signal_age >= 0
    patterns = signal_marks[active] & 0x0F  # 低4位
    total = len(patterns)

    if total == 0:
        return {"total_signals": 0, "entropy": 0.0, "zipf_r2": 0.0,
                "freq_by_pattern": [0.0] * 16, "unique_patterns": 0}

    counts = np.bincount(patterns, minlength=16).astype(float)
    freq = counts / total
    # 熵（自然对数）
    nonzero = freq[freq > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))
    max_entropy = np.log(16)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Zipf 拟合：频率排序后 log(freq) ~ log(rank) 的 R²
    sorted_freq = np.sort(counts)[::-1]
    sorted_freq = sorted_freq[sorted_freq > 0]
    if len(sorted_freq) >= 2:
        ranks = np.arange(1, len(sorted_freq) + 1, dtype=float)
        log_rank = np.log(ranks)
        log_freq = np.log(sorted_freq)
        # 线性回归
        slope, intercept = np.polyfit(log_rank, log_freq, 1)
        predicted = slope * log_rank + intercept
        ss_res = np.sum((log_freq - predicted) ** 2)
        ss_tot = np.sum((log_freq - np.mean(log_freq)) ** 2)
        zipf_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        zipf_slope = float(slope)
    else:
        zipf_r2 = 0.0
        zipf_slope = 0.0

    return {
        "total_signals": int(total),
        "unique_patterns": int(np.sum(counts > 0)),
        "entropy": round(entropy, 6),
        "normalized_entropy": round(normalized_entropy, 6),
        "zipf_r2": round(zipf_r2, 6),
        "zipf_slope": round(zipf_slope, 4),
        "freq_by_pattern": [round(float(f), 6) for f in freq],
    }


# ─── 维度 2：文化多样性（细化） ─────────────────────────────────────

def cultural_diversity(interpret: np.ndarray) -> dict:
    """解读表的文化多样性，分模式细化 + 总体指标。"""
    if len(interpret) == 0:
        return {"overall_std": 0.0, "per_pattern_std": [0.0] * 16,
                "interpretation_range": 0.0}

    # 每个模式的解读值标准差
    per_pattern_std = interpret.std(axis=0)  # (16,)
    overall_std = float(per_pattern_std.mean())

    # 解读值范围（最大-最小，衡量分化程度）
    interpretation_range = float(interpret.max() - interpret.min())

    # 解读表的协方差矩阵迹（衡量总方差）
    centered = interpret - interpret.mean(axis=0, keepdims=True)
    total_variance = float(np.trace(centered.T @ centered) / max(len(interpret) - 1, 1))

    return {
        "overall_std": round(overall_std, 6),
        "per_pattern_std": [round(float(s), 6) for s in per_pattern_std],
        "interpretation_range": round(interpretation_range, 6),
        "total_variance": round(total_variance, 6),
        "converged_patterns": int(np.sum(per_pattern_std < 0.01)),
    }


# ─── 维度 3：代际文化稳定性 ─────────────────────────────────────────

def generational_stability(interpret: np.ndarray, generation: np.ndarray) -> dict:
    """相邻世代个体解读表的平均相关性，衡量文化传承的稳定性。"""
    if len(interpret) < 4:
        return {"avg_correlation": 0.0, "generation_pairs": 0, "per_gen_corr": {}}

    unique_gens = np.unique(generation)
    if len(unique_gens) < 2:
        return {"avg_correlation": 0.0, "generation_pairs": 0,
                "per_gen_corr": {str(int(g)): 0.0 for g in unique_gens}}

    # 每个世代的平均解读表
    gen_means = {}
    for g in unique_gens:
        mask = generation == g
        if np.sum(mask) >= 2:
            gen_means[int(g)] = interpret[mask].mean(axis=0)

    if len(gen_means) < 2:
        return {"avg_correlation": 0.0, "generation_pairs": 0, "per_gen_corr": {}}

    sorted_gens = sorted(gen_means.keys())
    correlations = []
    per_gen_corr = {}

    for i in range(len(sorted_gens) - 1):
        g1, g2 = sorted_gens[i], sorted_gens[i + 1]
        v1, v2 = gen_means[g1], gen_means[g2]
        # Pearson 相关
        if np.std(v1) > 1e-12 and np.std(v2) > 1e-12:
            corr = float(np.corrcoef(v1, v2)[0, 1])
        else:
            corr = 0.0
        correlations.append(corr)
        per_gen_corr[f"{g1}->{g2}"] = round(corr, 6)

    avg_corr = float(np.mean(correlations)) if correlations else 0.0

    return {
        "avg_correlation": round(avg_corr, 6),
        "generation_pairs": len(correlations),
        "per_gen_corr": per_gen_corr,
        "stable_generations": int(np.sum(np.array(correlations) > 0.8)),
    }


# ─── 维度 4：空间聚类（简化 Moran's I） ─────────────────────────────

def spatial_clustering(signal_marks: np.ndarray, signal_age: np.ndarray,
                        n_cells: int, rows: int = 60, cols: int = 120) -> dict:
    """信号模式的空间自相关：相邻格子是否倾向于有相同信号。"""
    active = (signal_age >= 0) & (signal_marks > 0)
    if np.sum(active) < 4:
        return {"morans_i": 0.0, "same_neighbor_ratio": 0.0, "active_cells": 0}

    patterns = (signal_marks & 0x0F).astype(int)

    # 构建邻居对（球面网格：左右 + 上下，上下行偏移）
    same_count = 0
    total_pairs = 0

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if not active[idx]:
                continue
            # 右邻居
            right = r * cols + ((c + 1) % cols)
            if active[right]:
                total_pairs += 1
                if patterns[idx] == patterns[right]:
                    same_count += 1
            # 下邻居
            below = ((r + 1) % rows) * cols + c
            if active[below]:
                total_pairs += 1
                if patterns[idx] == patterns[below]:
                    same_count += 1

    same_ratio = same_count / total_pairs if total_pairs > 0 else 0.0

    # 简化 Moran's I：(observed_same - expected_same) / (1 - expected_same)
    # expected_same = sum(p_i^2) 随机匹配概率
    active_patterns = patterns[active]
    _, counts = np.unique(active_patterns, return_counts=True)
    freq = counts / counts.sum()
    expected_same = float(np.sum(freq ** 2))
    morans_i = (same_ratio - expected_same) / (1 - expected_same) if (1 - expected_same) > 0 else 0.0

    return {
        "morans_i": round(float(morans_i), 6),
        "same_neighbor_ratio": round(same_ratio, 6),
        "expected_same_random": round(expected_same, 6),
        "active_cells": int(np.sum(active)),
        "neighbor_pairs": total_pairs,
    }


# ─── 维度 5：信号-语境互信息 ─────────────────────────────────────────

def _quantile_bins(x: np.ndarray, n_bins: int = 3, max_single_bin_ratio: float = 0.85):
    """等频分箱：确保每个箱样本量大致相等，单箱占比不超过 max_single_bin_ratio。

    若数据高度偏斜导致等频分箱仍有单箱 > max_single_bin_ratio，
    则增加箱数（最多到 n_bins*2）重试；仍失败则在结果中标注 skewed=True。

    Returns:
        bins: np.ndarray, 每个样本的箱索引 (0..actual_n_bins-1)
        edges: np.ndarray, 分箱边界（含首尾）
        actual_n_bins: int, 实际使用的箱数
        bin_counts: list[int], 每个箱的样本数
        skewed: bool, 是否仍存在单箱占比超限
    """
    for try_n in range(n_bins, n_bins * 2 + 1):
        # 计算分位数边界
        quantiles = np.quantile(x, np.linspace(0, 1, try_n + 1))
        # 去重（大量相同值时边界可能重复）
        edges = np.unique(quantiles)
        actual_n = len(edges) - 1
        if actual_n < 2:
            # 数据几乎全相同，回退到等宽2箱
            edges = np.array([x.min(), x.max()])
            if edges[1] - edges[0] < 1e-12:
                edges = np.array([x.min(), x.min() + 1.0])
            actual_n = 2
        # searchsorted 分箱：edges[1:-1] 是内部边界
        bins = np.searchsorted(edges[1:-1], x, side='right')
        bins = np.clip(bins, 0, actual_n - 1)
        # 统计每箱样本数
        bin_counts = [int(np.sum(bins == i)) for i in range(actual_n)]
        max_ratio = max(bin_counts) / len(x) if len(x) > 0 else 1.0
        if max_ratio <= max_single_bin_ratio:
            return bins, edges, actual_n, bin_counts, False
    # 所有尝试都失败，返回最后一次的结果，标注 skewed
    return bins, edges, actual_n, bin_counts, True


def _compute_mi(patterns: np.ndarray, context_bins: np.ndarray,
                n_patterns: int = 16, n_context: int = 3) -> float:
    """计算离散变量 patterns 与 context_bins 的互信息（nats）。"""
    joint = np.zeros((n_patterns, n_context))
    for p, c in zip(patterns, context_bins):
        if 0 <= p < n_patterns and 0 <= c < n_context:
            joint[p, c] += 1
    total = joint.sum()
    if total == 0:
        return 0.0
    joint /= total
    p_pattern = joint.sum(axis=1, keepdims=True)
    p_context = joint.sum(axis=0, keepdims=True)
    p_product = p_pattern * p_context
    mask = (joint > 0) & (p_product > 0)
    mi = float(np.sum(joint[mask] * np.log(joint[mask] / p_product[mask])))
    return mi


def signal_context_mi(signal_marks: np.ndarray, signal_age: np.ndarray,
                       resource_grid: np.ndarray, flat: np.ndarray,
                       n_cells: int, n_permutations: int = 1000,
                       rng_seed: int = 42) -> dict:
    """信号模式与局部语境（食物丰富度）的互信息。

    修复（F-D5 残余）：
    - 等频分箱替代等宽分箱，避免单箱 >85% 导致 MI 估计退化
    - 1000 次置换检验输出 p_value（零模型阈值 p95）
    - 输出分箱参数、箱分布、置换统计量，保证可复算

    叙事校正：MI>0 是指示符关联（index），非涌现语义（symbol）——
    f_bit 硬编码进模式的定义性后果。
    """
    active = (signal_age >= 0) & (signal_marks > 0)
    n_active = int(np.sum(active))
    if n_active < 10:
        return {
            "mutual_information": 0.0, "normalized_mi": 0.0,
            "context_bins": 0, "active_signals": n_active,
            "p_value": 1.0, "permutations": 0,
            "null_mi_p95": 0.0, "null_mi_mean": 0.0, "null_mi_std": 0.0,
            "binning_method": "none", "bin_edges": [], "bin_counts": [],
            "bin_ratios": [], "skewed": False, "max_single_bin_ratio": 0.0,
            "note": "active_signals<10, MI not computed",
            "interpretation": "指示符关联强度（非涌现语义）",
        }

    patterns = (signal_marks[active] & 0x0F).astype(int)
    food = resource_grid[active]

    # 等频分箱（3档起步，单箱>85%时自动增加箱数）
    food_bins, edges, n_context, bin_counts, skewed = _quantile_bins(
        food, n_bins=3, max_single_bin_ratio=0.85
    )

    # 观测 MI
    n_patterns = 16
    mi = _compute_mi(patterns, food_bins, n_patterns, n_context)

    # 归一化 MI（除以联合熵，0~1）
    joint = np.zeros((n_patterns, n_context))
    for p, c in zip(patterns, food_bins):
        joint[p, c] += 1
    joint /= joint.sum()
    mask = joint > 0
    joint_entropy = float(-np.sum(joint[mask] * np.log(joint[mask])))
    normalized_mi = mi / joint_entropy if joint_entropy > 0 else 0.0

    # 置换检验：shuffle food_bins，构建 MI 零分布
    rng = np.random.default_rng(rng_seed)
    null_mis = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(food_bins)
        null_mis[i] = _compute_mi(patterns, shuffled, n_patterns, n_context)
    p_value = float(np.mean(null_mis >= mi))
    null_p95 = float(np.percentile(null_mis, 95))
    null_mean = float(np.mean(null_mis))
    null_std = float(np.std(null_mis))

    return {
        "mutual_information": round(mi, 6),
        "normalized_mi": round(normalized_mi, 6),
        "context_bins": n_context,
        "active_signals": n_active,
        # 分箱参数（F-D5 修复）
        "binning_method": "quantile_equal_frequency",
        "bin_edges": [round(float(e), 6) for e in edges],
        "bin_counts": bin_counts,
        "bin_ratios": [round(c / n_active, 4) for c in bin_counts],
        "skewed": skewed,
        "max_single_bin_ratio": round(max(bin_counts) / n_active, 4) if n_active > 0 else 1.0,
        # 置换检验（F-D5 修复）
        "p_value": round(p_value, 6),
        "permutations": n_permutations,
        "null_mi_p95": round(null_p95, 6),
        "null_mi_mean": round(null_mean, 6),
        "null_mi_std": round(null_std, 6),
        # 叙事校正
        "note": "MI>0 indicates index-level association (signal pattern correlates with context), NOT emergent symbolic semantics. f_bit is hardcoded into pattern definition.",
        "interpretation": "指示符关联强度（非涌现语义）",
    }


# ─── 维度 6：解读一致性 ──────────────────────────────────────────────

def interpretation_consistency(interpret: np.ndarray, genes: np.ndarray) -> dict:
    """种群内对相同信号的解读收敛度 + 基因 g14（感知）与解读的关系。"""
    if len(interpret) < 4:
        return {"consistency_score": 0.0, "dominant_interpretation_ratio": 0.0}

    # 对每个模式，计算解读值的"集中度"：1 - std/max_std
    per_pattern_std = interpret.std(axis=0)
    # 初始化解读表是 N(0, 0.3)，所以 max_std ≈ 0.3
    max_std = 0.3
    consistency_per_pattern = 1.0 - np.clip(per_pattern_std / max_std, 0, 1)
    consistency_score = float(consistency_per_pattern.mean())

    # 主导解读比例：对每个模式，有多少比例的个体解读值接近种群均值（±0.1）
    means = interpret.mean(axis=0)
    dominant_ratios = []
    for p in range(16):
        close = np.abs(interpret[:, p] - means[p]) < 0.1
        dominant_ratios.append(float(np.mean(close)))
    avg_dominant_ratio = float(np.mean(dominant_ratios))

    # g14（感知基因，索引14）与解读多样性的相关
    g14 = genes[:, 14] if genes.shape[1] > 14 else np.zeros(len(interpret))
    individual_interpret_std = interpret.std(axis=1)
    if np.std(g14) > 1e-12 and np.std(individual_interpret_std) > 1e-12:
        g14_corr = float(np.corrcoef(g14, individual_interpret_std)[0, 1])
    else:
        g14_corr = 0.0

    return {
        "consistency_score": round(consistency_score, 6),
        "dominant_interpretation_ratio": round(avg_dominant_ratio, 6),
        "per_pattern_consistency": [round(float(c), 6) for c in consistency_per_pattern],
        "g14_interpret_diversity_corr": round(g14_corr, 6),
    }


# ─── 维度 7：信号基因-表型相关性 ─────────────────────────────────────

def signal_genotype_phenotype(genes: np.ndarray, signal_marks: np.ndarray,
                               signal_age: np.ndarray, flat: np.ndarray) -> dict:
    """g15（信号基因）与个体所在格信号密度的相关性。"""
    if len(genes) < 4 or genes.shape[1] < 16:
        return {"g15_mean": 0.0, "g15_signal_corr": 0.0}

    g15 = genes[:, 15]

    # 每个个体所在格是否有活跃信号
    active = (signal_age >= 0) & (signal_marks > 0)
    has_signal = active[flat].astype(float)

    if np.std(g15) > 1e-12 and np.std(has_signal) > 1e-12:
        corr = float(np.corrcoef(g15, has_signal)[0, 1])
    else:
        corr = 0.0

    # g15 高的个体（>0.5）所在格有信号的比例 vs g15 低的个体
    high_g15 = g15 > 0.5
    low_g15 = g15 <= 0.5
    high_signal_rate = float(np.mean(has_signal[high_g15])) if np.sum(high_g15) > 0 else 0.0
    low_signal_rate = float(np.mean(has_signal[low_g15])) if np.sum(low_g15) > 0 else 0.0

    return {
        "g15_mean": round(float(np.mean(g15)), 6),
        "g15_signal_corr": round(corr, 6),
        "high_g15_fraction": round(float(np.mean(high_g15)), 6),
        "high_g15_signal_rate": round(high_signal_rate, 6),
        "low_g15_signal_rate": round(low_signal_rate, 6),
        "signal_rate_diff": round(high_signal_rate - low_signal_rate, 6),
    }


# ─── 综合语言涌现评分 ────────────────────────────────────────────────

def language_emergence_score(metrics: dict) -> dict:
    """
    综合语言涌现评分（0~100），基于 7 个维度的加权。
    评分逻辑：
      - 信号熵适中（不是全用一种，也不是完全随机）→ 词汇丰富度
      - Zipf R² 高 → 词汇有等级结构
      - 文化多样性适中（有差异但不混乱）→ 文化演化空间
      - 代际稳定性高 → 文化可传承
      - 空间聚类高 → 信号有地域方言
      - 信号-语境 MI 高 → 信号有指代意义
      - 解读一致性高 → 种群有共享语义
    """
    vocab = metrics["signal_vocabulary"]
    culture = metrics["cultural_diversity"]
    stability = metrics["generational_stability"]
    spatial = metrics["spatial_clustering"]
    mi = metrics["signal_context_mi"]
    consistency = metrics["interpretation_consistency"]
    genotype = metrics["signal_genotype_phenotype"]

    # 各维度子分（0~1）
    # 1. 词汇丰富度：归一化熵在 0.5~0.9 之间最佳
    ne = vocab["normalized_entropy"]
    vocab_score = max(0, 1 - abs(ne - 0.7) / 0.7)

    # 2. Zipf 结构：R² > 0.8 且 slope 在 -1.5~-0.5 之间
    zipf_ok = vocab["zipf_r2"] > 0.7 and -2.0 < vocab["zipf_slope"] < -0.3
    zipf_score = vocab["zipf_r2"] if zipf_ok else vocab["zipf_r2"] * 0.3

    # 3. 文化多样性：overall_std 在 0.05~0.2 之间最佳（有差异但不混乱）
    cd = culture["overall_std"]
    culture_score = max(0, 1 - abs(cd - 0.1) / 0.1)

    # 4. 代际稳定性：avg_correlation > 0.5
    stability_score = max(0, stability["avg_correlation"])

    # 5. 空间聚类：Moran's I > 0.2
    spatial_score = max(0, min(1, spatial["morans_i"] / 0.5))

    # 6. 信号-语境 MI：normalized_mi > 0.1
    mi_score = max(0, min(1, mi["normalized_mi"] / 0.3))

    # 7. 解读一致性：consistency_score > 0.5
    consistency_score = max(0, consistency["consistency_score"])

    return {
        "total_score": None,
        "note": ("0-100 综合分已废弃（6 模型外部评估 D4/元宝）：占 20% 权重的 "
                 "signal_context_mi 恒 0、generational_stability 样本不足，未演化快照即得 56.94，"
                 "总分无区分度。只报 7 维子分；判别判据转移到相对零模型超额量（D1 对照实现）。"),
        "subscores": {
            "vocabulary_richness": round(vocab_score, 4),
            "zipf_structure": round(zipf_score, 4),
            "cultural_diversity": round(culture_score, 4),
            "generational_stability": round(stability_score, 4),
            "spatial_clustering": round(spatial_score, 4),
            "signal_context_mi": round(mi_score, 4),
            "interpretation_consistency": round(consistency_score, 4),
        },
    }


# ─── 主函数 ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="语言涌现分析")
    parser.add_argument("snapshot", help="快照文件路径 (.npz)")
    parser.add_argument("--output", "-o", help="输出 JSON 文件路径")
    parser.add_argument("--rows", type=int, default=60, help="网格行数")
    parser.add_argument("--cols", type=int, default=120, help="网格列数")
    args = parser.parse_args()

    if not Path(args.snapshot).exists():
        print(f"错误：快照文件不存在: {args.snapshot}", file=sys.stderr)
        sys.exit(1)

    print(f"加载快照: {args.snapshot}")
    snap = load_snapshot(args.snapshot)
    print(f"  tick={snap['tick']}, 存活={snap['count']}, 最大世代={snap['max_generation']}")

    metrics = {}

    print("\n[1/7] 信号词汇统计...")
    metrics["signal_vocabulary"] = signal_vocabulary(snap["signal_marks"], snap["signal_age"])

    print("[2/7] 文化多样性...")
    metrics["cultural_diversity"] = cultural_diversity(snap["interpret"])

    print("[3/7] 代际文化稳定性...")
    metrics["generational_stability"] = generational_stability(snap["interpret"], snap["generation"])

    print("[4/7] 空间聚类...")
    metrics["spatial_clustering"] = spatial_clustering(
        snap["signal_marks"], snap["signal_age"], snap["n_cells"], args.rows, args.cols)

    print("[5/7] 信号-语境互信息...")
    metrics["signal_context_mi"] = signal_context_mi(
        snap["signal_marks"], snap["signal_age"],
        snap["resource_grid"],
        snap["flat"], snap["n_cells"])

    print("[6/7] 解读一致性...")
    metrics["interpretation_consistency"] = interpretation_consistency(snap["interpret"], snap["genes"])

    print("[7/7] 信号基因-表型相关性...")
    metrics["signal_genotype_phenotype"] = signal_genotype_phenotype(
        snap["genes"], snap["signal_marks"], snap["signal_age"], snap["flat"])

    print("\n综合语言涌现评分...")
    metrics["language_emergence_score"] = language_emergence_score(metrics)

    # 元信息（F-D11 自描述 + F-R2 补全）
    import subprocess
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        git_commit = "unknown"

    metrics["meta"] = {
        "snapshot": str(args.snapshot),
        "snapshot_sha256": snap.get("snapshot_sha256"),
        "tick": snap["tick"],
        "alive_count": snap["count"],
        "max_generation": snap["max_generation"],
        "n_cells": snap["n_cells"],
        "grid": f"{args.rows}x{args.cols}",
        # 配置自描述（F-D11）
        "config_seed": snap.get("config_seed"),
        "config_fingerprint": snap.get("config_fingerprint"),
        "use_sim_core": snap.get("use_sim_core"),
        "snapshot_version": snap.get("snapshot_version"),
        # 代码自描述（F-R2）
        "git_commit": git_commit,
        "sim_core_sha256": snap.get("sim_core_sha256"),
        "language_analysis_version": "2.0-fd5-fix",
        # 分析参数
        "mi_permutations": 1000,
        "mi_binning_method": "quantile_equal_frequency",
        "mi_max_single_bin_ratio": 0.85,
        "analysis_note": "MI>0 = index-level association, NOT emergent symbolic semantics (f_bit hardcoded).",
    }

    # 输出
    output_json = json.dumps(metrics, indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(output_json, encoding="utf-8")
        print(f"\n结果已保存: {args.output}")

    # 打印摘要
    score = metrics["language_emergence_score"]
    print("\n" + "=" * 60)
    print("语言涌现分析摘要")
    print("=" * 60)
    print(f"快照: {args.snapshot} (tick={snap['tick']}, N={snap['count']})")
    print(f"活跃信号数: {metrics['signal_vocabulary']['total_signals']}")
    print(f"信号熵(归一化): {metrics['signal_vocabulary']['normalized_entropy']:.4f}")
    print(f"Zipf R²: {metrics['signal_vocabulary']['zipf_r2']:.4f}")
    print(f"文化多样性(std): {metrics['cultural_diversity']['overall_std']:.6f}")
    print(f"代际稳定性(相关): {metrics['generational_stability']['avg_correlation']:.4f}")
    print(f"空间聚类(Moran's I): {metrics['spatial_clustering']['morans_i']:.4f}")
    print(f"信号-语境MI(归一化): {metrics['signal_context_mi']['normalized_mi']:.4f}")
    print(f"解读一致性: {metrics['interpretation_consistency']['consistency_score']:.4f}")
    print(f"g15均值: {metrics['signal_genotype_phenotype']['g15_mean']:.4f}")
    print(f"g15-信号相关: {metrics['signal_genotype_phenotype']['g15_signal_corr']:.4f}")
    print("-" * 60)
    print("综合分：已废弃（D4，见 JSON note）；7 维子分见上方指标")
    print("=" * 60)


if __name__ == "__main__":
    main()
