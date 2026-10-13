#!/usr/bin/env python3
"""R19 验收批（R24 口径）：4 机制主臂 + 3 机制对照臂，5 seed × 60k。

口径（所有者裁定 R24，2026-09-13）：
- 主臂：4 机制全开（学习瓶颈 + 任意性码本 + 信息不对称 + Steels 对齐）
- 对照臂：3 机制（关闭任意性码本，其余三机制开）
  依据：码本关闭时 codebook_conv 恒 1.000（常数，无信息量）
- 5 seed：42, 43, 44, 45, 46
- 60,000 tick
- 统一参数：uniform 资源 / N0=200 / Python 路径（use_sim_core=False）

验收条件（E-1）：
- 多数 seed 存活且 N>0 稳定持续 50k
- 先过 seed 哨兵：异 seed 终局三元组不得相同

用法：
  python experiments/run_r19_acceptance_batch.py [--seeds 42,43] [--ticks 60000]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print 会让脚本 rc=1 假失败；F-R15 族）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # 非 TTY / 旧解释器：不因诊断能力缺失而阻断运行

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RUN_SCRIPT = ROOT / "experiments" / "run_d2_experiment.py"

# R24 口径：两大臂配置
ARMS = {
    "r19_4mech": [  # 主臂：4 机制全开
        "--learning-bottleneck",
        "--arbitrary-codebook",
        "--info-asymmetry",
        "--steels-alignment",
    ],
    "r19_3mech": [  # 对照臂：3 机制（关码本）
        "--learning-bottleneck",
        "--info-asymmetry",
        "--steels-alignment",
    ],
}

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_TICKS = 60000


def run_one(arm: str, extra_args: list, seed: int, ticks: int) -> dict:
    """运行单个实验，返回结果摘要。"""
    tag = f"{arm}_s{seed}"
    cmd = [
        sys.executable, str(RUN_SCRIPT),
        "--tag", tag,
        "--seed", str(seed),
        "--ticks", str(ticks),
        "--log-interval", "2000",
    ] + extra_args

    print(f"\n{'='*60}")
    print(f"启动: {tag} (seed={seed}, ticks={ticks})")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=False)
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else f"FAIL(rc={result.returncode})"
    print(f"\n{tag} 完成: {status}, 耗时 {elapsed/60:.1f} 分钟")

    return {
        "tag": tag,
        "arm": arm,
        "seed": seed,
        "ticks": ticks,
        "status": status,
        "elapsed_min": round(elapsed / 60, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="R19 验收批（R24 口径）")
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)),
                        help="逗号分隔的 seed 列表（默认 42,43,44,45,46）")
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS,
                        help="每个实验的 tick 数（默认 60000）")
    parser.add_argument("--arms", type=str, default=",".join(ARMS.keys()),
                        help="逗号分隔的臂名（默认 r19_4mech,r19_3mech）")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    arms = [a.strip() for a in args.arms.split(",")]

    print(f"R19 验收批（R24 口径）")
    print(f"  臂: {arms}")
    print(f"  seeds: {seeds}")
    print(f"  ticks: {args.ticks}")
    print(f"  总实验数: {len(arms) * len(seeds)}")
    print(f"  预计总耗时: ~{len(arms) * len(seeds) * 4} 小时（~4 t/s, Python 路径）")

    results = []
    for arm in arms:
        if arm not in ARMS:
            print(f"跳过未知臂: {arm}")
            continue
        for seed in seeds:
            r = run_one(arm, ARMS[arm], seed, args.ticks)
            results.append(r)

    # 汇总
    print(f"\n{'='*60}")
    print("R19 验收批汇总")
    print(f"{'='*60}")
    print(f"{'tag':<20} {'status':<10} {'elapsed_min':>12}")
    print("-" * 45)
    for r in results:
        print(f"{r['tag']:<20} {r['status']:<10} {r['elapsed_min']:>12.1f}")

    ok_count = sum(1 for r in results if r["status"] == "OK")
    print(f"\n完成: {ok_count}/{len(results)} 成功")

    # seed 哨兵检查提示
    print("\n⚠️  验收前请检查：")
    print("  1. seed 哨兵：异 seed 终局三元组(N, g15, g14)不得相同")
    print("  2. 生态门：多数 seed 存活且 N>0 稳定持续 50k")
    print("  3. manifest 包含 git_commit / sim_core_sha256 / switches（D-3）")


if __name__ == "__main__":
    main()
