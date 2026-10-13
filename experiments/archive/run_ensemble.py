#!/usr/bin/env python3
"""种子系综调度器（S3 多进程 seed 系综，性能优化路线与触发条件 §S3）。

职责
----
- 每 seed 用一个子进程跑 experiments/run_long_experiment.py（绕 GIL，多核线性铺开）
- 并发数 = min(--jobs, seed 数)；子进程默认 sys.executable（保证与父进程同一解释器 / 同一 sim_core）
- 全部结束后汇总每 seed 的终局指标（g14/g15/g16/g19/trust/cult_div/max_gen/N）
  与跨 seed 的 mean±std 汇总表，写入 results/ensemble/<name>/summary.csv
- 写 manifest.json（commit hash + 参数 + 时间戳），复现追踪（G5/manifest 扩展要求）

用法
----
    # 5 个 seed 并发跑小世界 300k tick，每 seed 各出一个 long_<tag>_s<N>.csv
    python experiments/run_ensemble.py --rows 60 --cols 120 --ticks 300000 \
        --segment 50000 --tag ens_a --seeds 42 43 44 45 46 --jobs 4

    # 云端 D1 四臂对照批量（主实验用，对照组按启动说明补齐对应 CLI 开关后复用本条）
    python experiments/run_ensemble.py --rows 60 --cols 120 --ticks 300000 \
        --segment 50000 --tag zero_main --seeds 42 43 44 --jobs 3

依赖
----
    仅 stdlib（subprocess/multiprocessing 并发用 subprocess + Semaphore 同源实现，
    避免 spawn 重入）：Windows/Linux 通用；run_long_experiment.py 已具 if __name__ guard。
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "experiments" / "run_long_experiment.py"
OUT_DIR = PROJECT_ROOT / "results" / "ensemble"

# 汇总关注的终局列（与 run_long_experiment.CSV_HEADER 对齐）
SUMMARY_COLUMNS = {
    "N": "N", "g14": "g14", "g15": "g15", "g16": "g16", "g19": "g19",
    "trust": "trust", "cult_div": "cult_div", "max_gen": "max_gen",
    "sig_density": "sig_density", "tick": "tick",
}


def _commit_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _last_row(csv_path: Path) -> dict:
    """读 CSV 最后一行 → dict（列名 → 值，数值转 float）。文件不存在返回 {}。"""
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return {k: (float(v) if v not in ("", None) and k in SUMMARY_COLUMNS else v)
            for k, v in rows[-1].items()}


def _summary_stats(rows: list[dict]) -> dict:
    """跨 seed 汇总：mean / std（仅数值列）。"""
    stats: dict = {"n_seeds": len(rows)}
    for col in SUMMARY_COLUMNS.values():
        try:
            vals = [float(r.get(col)) for r in rows if r.get(col) not in (None, "")]
        except (TypeError, ValueError):
            continue
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
        stats[col + "_mean"] = round(mean, 4)
        stats[col + "_std"] = round(var ** 0.5, 4)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=60)
    parser.add_argument("--cols", type=int, default=120)
    parser.add_argument("--ticks", type=int, default=300000)
    parser.add_argument("--initial", type=int, default=500)
    parser.add_argument("--density-cap", type=float, default=0.45)
    parser.add_argument("--segment", type=int, default=50000)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                        help="seed 列表，如 --seeds 42 43 44")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4,
                        help="并发进程数（默认 = 逻辑核数）")
    parser.add_argument("--rotation-period", type=int, default=0)
    # D1 零模型三开关（透传给 run_long_experiment.py）
    parser.add_argument("--neutral-genes", action="store_true")
    parser.add_argument("--signal-disabled", action="store_true")
    parser.add_argument("--signal-mode", type=str, default="state",
                        choices=["state", "random", "evolved"])
    # 战役参数（透传给 run_long_experiment.py）
    parser.add_argument("--lifespan-mult", type=float, default=1.0,
                        help="寿命基准缩放（1.0=旧行为；范围(0.05,8]）")
    parser.add_argument("--stay-prob", type=float, default=0.0,
                        help="停驻率（0=旧行为；范围[0,0.95)）")
    args = parser.parse_args()

    tag = args.tag
    csv_paths = [PROJECT_ROOT / "experiments" / f"long_{tag}_s{s}.csv" for s in args.seeds]

    # 并发调度：Semaphore 控制同时运行的子进程数
    import threading
    import queue
    sem = threading.Semaphore(max(1, args.jobs))
    results = {}
    errors = {}

    def run_one(seed: int):
        out = PROJECT_ROOT / "experiments" / f"long_{tag}_s{seed}.csv"
        # 快照由 run_long_experiment 自管（存在则续跑），此处只组装命令并循环到跑满
        cmd_base = [sys.executable, str(RUNNER),
                    "--rows", str(args.rows), "--cols", str(args.cols),
                    "--ticks", str(args.ticks), "--initial", str(args.initial),
                    "--density-cap", str(args.density_cap),
                    "--segment", str(args.segment),
                    "--tag", f"{tag}_s{seed}", "--seed", str(seed)]
        if args.rotation_period:
            cmd_base += ["--rotation-period", str(args.rotation_period)]
        if args.neutral_genes:
            cmd_base += ["--neutral-genes"]
        if args.signal_disabled:
            cmd_base += ["--signal-disabled"]
        if args.signal_mode != "state":
            cmd_base += ["--signal-mode", args.signal_mode]
        if args.lifespan_mult != 1.0:
            cmd_base += ["--lifespan-mult", str(args.lifespan_mult)]
        if args.stay_prob != 0.0:
            cmd_base += ["--stay-prob", str(args.stay_prob)]
        with sem:
            t0 = time.time()
            rc, err = 0, ""
            while True:
                proc = subprocess.run(cmd_base, capture_output=True,
                                      text=True, encoding="utf-8", errors="replace")
                if proc.returncode != 0:
                    rc, err = proc.returncode, (proc.stderr or proc.stdout)[-2000:]
                    break
                # 从 stdout "本次跑: {start} -> {target} ({n} tick)" 取本段到达 tick
                reached = 0
                for line in reversed(proc.stdout.splitlines()):
                    if "本次跑" in line:
                        try:
                            reached = int(line.split("->")[1].split("(")[0].strip())
                        except (IndexError, ValueError):
                            reached = 0
                        break
                # segment>0 且未跑满 → 快照续跑下一段；否则（segment=0 一次性或已满）收工
                if args.segment > 0 and reached < args.ticks:
                    continue
                break
            dt = time.time() - t0
            if rc:
                errors[seed] = err
            results[seed] = {
                "seed": seed, "wall_s": round(dt, 1),
                "rc": rc if rc else 0, "csv": str(out.relative_to(PROJECT_ROOT)),
            }
            print(f"  [seed {seed}] rc={rc or 0} wall={dt:.1f}s "
                  f"{'OK' if rc == 0 else 'FAIL'}", flush=True)

    threads = [threading.Thread(target=run_one, args=(s,)) for s in args.seeds]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    out_dir = OUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # 每 seed 终局 + 跨 seed 汇总
    per_seed = []
    for seed in args.seeds:
        row = dict(results[seed])
        last = _last_row(csv_paths[args.seeds.index(seed)])
        if last:
            for col in SUMMARY_COLUMNS.values():
                row.setdefault(col, last.get(col))
        per_seed.append(row)
    summary = {"meta": {
        "tag": tag, "rows": args.rows, "cols": args.cols, "ticks": args.ticks,
        "segment": args.segment, "density_cap": args.density_cap,
        "rotation_period": args.rotation_period, "seeds": args.seeds,
        "lifespan_mult": args.lifespan_mult, "stay_prob": args.stay_prob,
        "commit": _commit_hash(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "interpreter": sys.executable,
    }}
    summary["per_seed"] = per_seed
    summary["stats"] = _summary_stats([
        {c: r.get(c) for c in SUMMARY_COLUMNS.values()} for r in per_seed if r.get("rc") == 0
    ] or [])

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    # 人类可读 CSV 汇总
    csv_path = out_dir / "summary.csv"
    fieldnames = ["seed", "wall_s", "rc", "tick", "N",
                  "g14", "g15", "g16", "g19", "trust", "cult_div", "max_gen"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_seed:
            w.writerow({k: r.get(k, "") for k in fieldnames})
        if summary.get("stats"):
            st = summary["stats"]
            mean_row = {"seed": "mean", "N": st.get("N_mean"),
                        "g14": st.get("g14_mean"), "g15": st.get("g15_mean"),
                        "g16": st.get("g16_mean"), "g19": st.get("g19_mean"),
                        "trust": st.get("trust_mean"),
                        "cult_div": st.get("cult_div_mean"),
                        "max_gen": st.get("max_gen_mean"),
                        "tick": st.get("tick_mean")}
            w.writerow({k: mean_row.get(k, "") for k in fieldnames})

    print(f"\n=== 系综完成: {tag} ({len(args.seeds)} seeds, "
          f"{len(errors)} 失败) ===")
    print(f"每 seed 数据: experiments/long_{tag}_s*.csv")
    print(f"汇总: {csv_path} / {summary_path}")
    if errors:
        print("失败详情:")
        for seed, msg in errors.items():
            print(f"  seed {seed}: {msg[-400:]}")
        sys.exit(1)


if __name__ == "__main__":
    main()