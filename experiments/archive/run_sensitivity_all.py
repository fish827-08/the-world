"""A3 敏感性扫描调度：串行跑全部参数组合，最后生成 summary.csv。

扫描组合（来自 隐式选择压审计清单.md §三 的 4 个高影响参数）：
  baseline         —— 不覆盖任何参数（对照）
  transfer_ratio   —— 0.2(-50%), 0.32(-20%), 0.48(+20%), 0.6(+50%)
  trust_false      —— 0.05（与 trust_true=0.05 对称化对照）
  social_rpe       —— 0.15（alone_rpe=-0.15，对称化对照）
  niche_floor      —— 0.0, 0.2, 0.6

每个组合 × 3 seed (42/2024/777) = 30 次实验。
每次 60000 tick（约 3~5 分钟，N=5000 时 ~117 tick/s）。

用法：python experiments/run_sensitivity_all.py [ticks]
增量落盘：每次实验的 termination.csv 每 1000 tick 追加一行，中断不丢数据。
已完成的实验（final.json 存在）会跳过，支持断点续跑。
"""
import sys
import os
import time
import csv
import json
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.sensitivity_scan import run_experiment, RESULTS_DIR, TRACKED_GENES

SEEDS = [42, 2024, 777]
TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 60_000

# 扫描组合：(param, value, 描述)
SCAN_COMBOS = [
    ("baseline", 0.0, "对照（不覆盖）"),
    # transfer_ratio ±20% / ±50%
    ("transfer_ratio", 0.2, "捕食转移率 -50%"),
    ("transfer_ratio", 0.32, "捕食转移率 -20%"),
    ("transfer_ratio", 0.48, "捕食转移率 +20%"),
    ("transfer_ratio", 0.6, "捕食转移率 +50%"),
    # trust_false 对称化
    ("trust_false", 0.05, "信任下降对称化（=trust_true）"),
    # social_rpe 对称化
    ("social_rpe", 0.15, "社会效价对称化 ±0.15"),
    # niche_floor 扫描
    ("niche_floor", 0.0, "活性保底 0.0（全罚）"),
    ("niche_floor", 0.2, "活性保底 0.2"),
    ("niche_floor", 0.6, "活性保底 0.6"),
]


def is_completed(param: str, value: float, seed: int) -> bool:
    """检查实验是否已完成（final.json 存在）。"""
    value_str = f"{value:.4f}".rstrip("0").rstrip(".")
    exp_dir = os.path.join(RESULTS_DIR, f"{param}_{value_str}_{seed}")
    return os.path.exists(os.path.join(exp_dir, "final.json"))


def load_final(param: str, value: float, seed: int) -> dict:
    value_str = f"{value:.4f}".rstrip("0").rstrip(".")
    exp_dir = os.path.join(RESULTS_DIR, f"{param}_{value_str}_{seed}")
    with open(os.path.join(exp_dir, "final.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def generate_summary(all_results: list[dict]):
    """生成 summary.csv：参数→终局基因分布差异（3 seed 均值）。"""
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")

    # 按 (param, value) 分组，计算 3 seed 均值
    groups = {}
    for r in all_results:
        key = (r["param"], r["value"])
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    fieldnames = ["param", "value", "description", "n_seeds", "extinct_seeds",
                  "avg_final_N", "avg_max_gen", "avg_trust", "avg_valence"]
    fieldnames += [f"avg_{g.name}" for g in TRACKED_GENES]

    # baseline 均值（用于计算差异）
    baseline_key = ("baseline", 0.0)
    baseline_avg = {}
    if baseline_key in groups:
        bl = groups[baseline_key]
        for g in TRACKED_GENES:
            baseline_avg[g.name] = sum(r[g.name] for r in bl) / len(bl)
        baseline_avg["trust_mean"] = sum(r["trust_mean"] for r in bl) / len(bl)

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for (param, value), results in sorted(groups.items()):
            desc = next((d for p, v, d in SCAN_COMBOS if p == param and v == value), "")
            n_seeds = len(results)
            extinct = sum(1 for r in results if r.get("extinct", False))
            row = {
                "param": param, "value": value, "description": desc,
                "n_seeds": n_seeds, "extinct_seeds": extinct,
                "avg_final_N": round(sum(r["final_N"] for r in results) / n_seeds, 1),
                "avg_max_gen": round(sum(r["final_max_gen"] for r in results) / n_seeds, 1),
                "avg_trust": round(sum(r["trust_mean"] for r in results) / n_seeds, 4),
                "avg_valence": round(sum(r["valence_mean"] for r in results) / n_seeds, 4),
            }
            for g in TRACKED_GENES:
                row[f"avg_{g.name}"] = round(sum(r[g.name] for r in results) / n_seeds, 4)
            w.writerow(row)

    print(f"\n=== summary.csv 已生成: {summary_path} ===")
    print(f"共 {len(groups)} 组参数组合")

    # 打印关键差异（与 baseline 对比）
    if baseline_avg:
        print("\n=== 与 baseline 的关键基因差异（3 seed 均值）===")
        print(f"{'param':<16} {'value':<6} {'g16Δ':>8} {'g15Δ':>8} {'g14Δ':>8} {'trustΔ':>8} {'NΔ':>8}")
        for (param, value), results in sorted(groups.items()):
            if (param, value) == baseline_key:
                continue
            n = len(results)
            g16 = sum(r["AGGRESSION"] for r in results) / n - baseline_avg["AGGRESSION"]
            g15 = sum(r["SIGNAL_STRENGTH"] for r in results) / n - baseline_avg["SIGNAL_STRENGTH"]
            g14 = sum(r["PERCEPTION"] for r in results) / n - baseline_avg["PERCEPTION"]
            trust = sum(r["trust_mean"] for r in results) / n - baseline_avg["trust_mean"]
            n_avg = sum(r["final_N"] for r in results) / n
            bl_n = sum(r["final_N"] for r in groups[baseline_key]) / len(groups[baseline_key])
            print(f"{param:<16} {value:<6} {g16:>+8.3f} {g15:>+8.3f} {g14:>+8.3f} {trust:>+8.3f} {n_avg-bl_n:>+8.0f}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    total = len(SCAN_COMBOS) * len(SEEDS)
    print(f"=== A3 敏感性扫描调度 ===")
    print(f"组合数: {len(SCAN_COMBOS)} × seed数: {len(SEEDS)} = {total} 次实验")
    print(f"每次 {TICKS} tick，增量落盘，已完成跳过")
    print()

    all_results = []
    t0 = time.time()
    done = 0

    for param, value, desc in SCAN_COMBOS:
        for seed in SEEDS:
            done += 1
            if is_completed(param, value, seed):
                print(f"[{done}/{total}] 跳过（已完成）: {param}={value}, seed={seed}")
                all_results.append(load_final(param, value, seed))
                continue

            print(f"\n[{done}/{total}] 开始: {param}={value} ({desc}), seed={seed}")
            try:
                # 用独立子进程运行每次实验，避免 sim_core(Rust扩展)全局状态
                # 在连续实验间未释放导致的段错误；子进程退出后资源完全清理。
                script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "sensitivity_scan.py")
                proc = subprocess.run(
                    [sys.executable, script, param, str(value), str(seed), str(TICKS)],
                    capture_output=True, text=True, timeout=900,  # 单次最多15分钟
                )
                if proc.returncode != 0:
                    print(f"  实验失败(exit={proc.returncode}): {proc.stderr[-500:]}")
                    continue
                # 子进程成功后读取 final.json
                if is_completed(param, value, seed):
                    result = load_final(param, value, seed)
                    all_results.append(result)
                else:
                    print(f"  警告: 子进程退出但 final.json 不存在")
                    continue
            except subprocess.TimeoutExpired:
                print(f"  实验超时(>{900}s)，跳过")
                continue
            except Exception as e:
                print(f"  实验失败: {e}")
                continue

            elapsed = time.time() - t0
            avg_per_exp = elapsed / done
            remaining = avg_per_exp * (total - done)
            print(f"  进度: {done}/{total}, 已用 {elapsed/60:.1f}min, "
                  f"预计剩余 {remaining/60:.1f}min")

    generate_summary(all_results)
    total_elapsed = time.time() - t0
    print(f"\n全部完成！总用时 {total_elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
