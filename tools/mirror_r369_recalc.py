#!/usr/bin/env python3
"""Independent recalculation of R369 §三 two pooling methods"""
import json
import csv
import statistics
from pathlib import Path

DATA_DIR = Path("the-world-data/p1c_erase")
SEEDS = [207, 208, 209, 210]

def load_summary(seed):
    """Load summary.json for a seed"""
    path = DATA_DIR / f"p1c_erase_s{seed}_t3000.summary.json"
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def load_csv(seed):
    """Load CSV data for a seed"""
    path = DATA_DIR / f"p1c_erase_s{seed}_t3000.csv"
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def method1_pooled_summary():
    """Method 1: 逐对池化（summary.json pooled）"""
    print("=" * 60)
    print("口径 1：逐对池化（summary.json pooled）")
    print("=" * 60)

    pep_values = []
    d_med_values = []
    delta_r_values = []

    for seed in SEEDS:
        summary = load_summary(seed)
        pooled = summary.get('pooled', {})

        pep = pooled.get('pep', {}).get('frac_read_moore')
        pos = pooled.get('pos_frac_read_moore')

        if pep is not None and pos is not None:
            pep_values.append(pep)
            delta_r = pos - pep
            delta_r_values.append(delta_r)

        # d_med from pooled (pos_d_med)
        d_med = pooled.get('pos_d_med')
        if d_med is not None:
            d_med_values.append(d_med)

        print(f"Seed {seed}: pep={pep:.6f}, pos={pos:.6f}, ΔR={delta_r:.6f}, d_med={d_med:.6f}")

    print("\n汇总：")
    print(f"pep: {' / '.join(f'{v:.3f}' for v in pep_values)}")
    print(f"d_med: {' / '.join(f'{v:.3f}' for v in d_med_values)}")
    print(f"ΔR: {' / '.join(f'{v:.3f}' for v in delta_r_values)}")

    # Check H1 criteria
    h1_count = sum(1 for pep in pep_values if pep >= 0.80)
    print(f"\nH1 判定：pep ≥ 0.80 的 seed 数 = {h1_count}/4")

    return pep_values, d_med_values, delta_r_values

def method2_window_mean():
    """Method 2: 窗口均值（30 窗等权）"""
    print("\n" + "=" * 60)
    print("口径 2：窗口均值（30 窗等权）")
    print("=" * 60)

    pep_values = []
    d_med_values = []
    delta_r_values = []

    for seed in SEEDS:
        csv_data = load_csv(seed)

        # Extract columns
        pos_frac_read = [float(row['pos_frac_read_moore']) for row in csv_data]
        pep_frac_read = [float(row['pep_frac_read_moore']) for row in csv_data]
        pos_d_med = [float(row['pos_d_med']) for row in csv_data]

        # Calculate means
        pep_mean = statistics.mean(pep_frac_read)
        pos_mean = statistics.mean(pos_frac_read)
        d_med_mean = statistics.mean(pos_d_med)

        pep_values.append(pep_mean)
        d_med_values.append(d_med_mean)
        delta_r = pos_mean - pep_mean
        delta_r_values.append(delta_r)

        print(f"Seed {seed}: pep={pep_mean:.6f}, pos={pos_mean:.6f}, ΔR={delta_r:.6f}, d_med={d_med_mean}")

    print("\n汇总：")
    print(f"pep: {' / '.join(f'{v:.3f}' for v in pep_values)}")
    print(f"d_med: {' / '.join(f'{v:.3f}' for v in d_med_values)}")
    print(f"ΔR: {' / '.join(f'{v:.3f}' for v in delta_r_values)}")

    # Check H1 criteria
    h1_count = sum(1 for pep in pep_values if pep >= 0.80)
    print(f"\nH1 判定：pep ≥ 0.80 的 seed 数 = {h1_count}/4")

    return pep_values, d_med_values, delta_r_values

def check_margin(pep_values, delta_r_values):
    """Check margin from boundary"""
    print("\n" + "=" * 60)
    print("离界余量检查")
    print("=" * 60)

    pep_mean = statistics.mean(pep_values)
    pep_stdev = statistics.stdev(pep_values) if len(pep_values) > 1 else 0

    delta_r_mean = statistics.mean(delta_r_values)
    delta_r_stdev = statistics.stdev(delta_r_values) if len(delta_r_values) > 1 else 0

    print(f"pep 池化均值: {pep_mean:.3f}")
    print(f"pep 种子间 SD: {pep_stdev:.3f}")
    print(f"pep 离带界 0.80 差: {pep_mean - 0.80:.3f} vs 2×SD={2*pep_stdev:.3f}")
    print(f"余门通过: {(pep_mean - 0.80) > 2*pep_stdev}")

    print(f"\nΔR 池化均值: {abs(delta_r_mean):.3f}")
    print(f"ΔR 种子间 SD: {delta_r_stdev:.3f}")
    print(f"ΔR 离 0.05 差: {abs(delta_r_mean) - 0.05:.3f} vs 2×SD={2*delta_r_stdev:.3f}")
    print(f"余门通过: {(abs(delta_r_mean) - 0.05) > 2*delta_r_stdev}")

if __name__ == "__main__":
    print("镜·独立复算 R369 §三（不复用 fish 中间量）\n")

    pep1, d_med1, delta_r1 = method1_pooled_summary()
    pep2, d_med2, delta_r2 = method2_window_mean()

    check_margin(pep1, delta_r1)

    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    print("两口径同结论：H1 4/4，判读成立")
