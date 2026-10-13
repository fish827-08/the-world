#!/usr/bin/env python3
"""批量重算所有归档快照的语言指标（F-D5 修复后）。

用法：
  python experiments/recalc_all_lang_metrics.py
  python experiments/recalc_all_lang_metrics.py --snapshot-dir data/snapshots --output-dir experiments/lang_metrics_v2
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# 导入 language_analysis 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from language_analysis import load_snapshot  # noqa: E402


# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print 会让脚本 rc=1 假失败；F-R15 族）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # 非 TTY / 旧解释器：不因诊断能力缺失而阻断运行


def analyze_snapshot_full(snap: dict, rows: int = 60, cols: int = 120) -> dict:
    """完整分析一个快照，返回所有7维指标 + meta。"""
    metrics = {}

    # 延迟导入各维度函数
    from language_analysis import (
        signal_vocabulary, cultural_diversity, generational_stability,
        spatial_clustering, signal_context_mi, interpretation_consistency,
        signal_genotype_phenotype, language_emergence_score,
    )

    metrics["signal_vocabulary"] = signal_vocabulary(snap["signal_marks"], snap["signal_age"])
    metrics["cultural_diversity"] = cultural_diversity(snap["interpret"])
    metrics["generational_stability"] = generational_stability(snap["interpret"], snap["generation"])
    metrics["spatial_clustering"] = spatial_clustering(
        snap["signal_marks"], snap["signal_age"], snap["n_cells"], rows, cols)
    metrics["signal_context_mi"] = signal_context_mi(
        snap["signal_marks"], snap["signal_age"],
        snap["resource_grid"], snap["flat"], snap["n_cells"])
    metrics["interpretation_consistency"] = interpretation_consistency(snap["interpret"], snap["genes"])
    metrics["signal_genotype_phenotype"] = signal_genotype_phenotype(
        snap["genes"], snap["signal_marks"], snap["signal_age"], snap["flat"])
    metrics["language_emergence_score"] = language_emergence_score(metrics)

    # meta
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
        "snapshot_sha256": snap.get("snapshot_sha256"),
        "tick": snap["tick"],
        "alive_count": snap["count"],
        "max_generation": snap["max_generation"],
        "n_cells": snap["n_cells"],
        "config_seed": snap.get("config_seed"),
        "config_fingerprint": snap.get("config_fingerprint"),
        "use_sim_core": snap.get("use_sim_core"),
        "snapshot_version": snap.get("snapshot_version"),
        "git_commit": git_commit,
        "sim_core_sha256": snap.get("sim_core_sha256"),
        "language_analysis_version": "2.0-fd5-fix",
        "mi_permutations": 1000,
        "mi_binning_method": "quantile_equal_frequency",
        "mi_max_single_bin_ratio": 0.85,
        "analysis_note": "MI>0 = index-level association, NOT emergent symbolic semantics (f_bit hardcoded).",
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="批量重算所有归档快照的语言指标")
    parser.add_argument("--snapshot-dir", default="data/snapshots", help="快照目录")
    parser.add_argument("--output-dir", default="experiments/lang_metrics_v2", help="输出目录")
    parser.add_argument("--rows", type=int, default=60, help="网格行数")
    parser.add_argument("--cols", type=int, default=120, help="网格列数")
    args = parser.parse_args()

    snapshot_dir = Path(args.snapshot_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshots = sorted(snapshot_dir.glob("*.npz"))
    print(f"找到 {len(snapshots)} 个快照，输出目录: {output_dir}")
    print("=" * 70)

    results = []
    for i, snap_path in enumerate(snapshots):
        name = snap_path.stem
        output_path = output_dir / f"{name}_lang_metrics.json"

        t0 = time.time()
        try:
            snap = load_snapshot(str(snap_path))
            metrics = analyze_snapshot_full(snap, args.rows, args.cols)
            metrics["meta"]["snapshot"] = str(snap_path)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            mi = metrics["signal_context_mi"]
            elapsed = time.time() - t0
            print(f"[{i+1}/{len(snapshots)}] {name}: "
                  f"N={snap['count']}, tick={snap['tick']}, "
                  f"MI={mi['mutual_information']:.4f} (norm={mi['normalized_mi']:.4f}), "
                  f"p={mi['p_value']:.4f}, bins={mi['context_bins']}, "
                  f"max_bin_ratio={mi['max_single_bin_ratio']:.2f}, "
                  f"{elapsed:.1f}s ✓")

            results.append({
                "snapshot": name,
                "alive_count": snap["count"],
                "tick": snap["tick"],
                "mi": mi["mutual_information"],
                "normalized_mi": mi["normalized_mi"],
                "p_value": mi["p_value"],
                "bins": mi["context_bins"],
                "max_bin_ratio": mi["max_single_bin_ratio"],
                "skewed": mi["skewed"],
                "status": "ok",
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[{i+1}/{len(snapshots)}] {name}: ERROR ({elapsed:.1f}s) - {e}")
            results.append({"snapshot": name, "status": "error", "error": str(e)})

    # 汇总
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    ok_count = sum(1 for r in results if r["status"] == "ok")
    err_count = sum(1 for r in results if r["status"] == "error")
    print(f"成功: {ok_count}/{len(results)}, 失败: {err_count}/{len(results)}")

    # 保存汇总
    summary_path = output_dir / "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "total": len(results),
                    "ok": ok_count, "error": err_count}, f, indent=2, ensure_ascii=False)
    print(f"汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
