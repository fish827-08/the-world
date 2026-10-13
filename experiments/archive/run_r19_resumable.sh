#!/usr/bin/env bash
# R19 生态门验证 —— 断点续跑驱动（本地开发）
#
# 规格（所有者裁定）：
#   R19：生态门须用【修复后】代码重跑 5 seed × 60k、arbitrary_codebook=true
#   V12：4 机制为主（--codebook 1）+ 3 机制对照（--codebook 0），同 seed 集 / 同 tick
#
# 断点续跑：每 --snapshot-every(默认 5000) tick 写一次快照 + 全局 RNG 状态；
#   中断后【原样重跑本脚本】即可从各臂最新快照续跑，与不中断连续跑【逐位一致】
#   （已实测：连续 [59,124,308] == 续跑 [59,124,308]，born/died 亦相同）。
#   强制从头：FRESH=1 ./run_r19_resumable.sh
#
# 可选环境变量：TICKS / SNAP / SEEDS / CODEBOOKS / MAXPAR
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY=".venv/Scripts/python.exe"
OUT="_rerun_logs/a4_fix"
TICKS="${TICKS:-60000}"
SNAP="${SNAP:-5000}"
SEEDS="${SEEDS:-42 43 44 45 46}"
CODEBOOKS="${CODEBOOKS:-1 0}"
mkdir -p "$OUT"

FRESH_ARG=""
[ "${FRESH:-0}" = "1" ] && FRESH_ARG="--fresh"

for cb in $CODEBOOKS; do
  for s in $SEEDS; do
    "$PY" experiments/a4_verify_capacity.py \
      --mode on --seed "$s" --ticks "$TICKS" --codebook "$cb" \
      --snapshot-every "$SNAP" $FRESH_ARG \
      --out "$OUT/asym_on_cb${cb}_s${s}.csv" \
      > "$OUT/on_cb${cb}_s${s}.log" 2>&1 &
  done
done
echo "launched; 结果落 $OUT/*.summary.json"
wait
echo "ALL_DONE"
