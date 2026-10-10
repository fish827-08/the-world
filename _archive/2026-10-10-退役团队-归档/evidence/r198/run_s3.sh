#!/usr/bin/env bash
# R198 / 13.8 S3 正式批启动脚本（4 臂 × 6 seed × 18000 tick）
#
# 依据：设计稿 §6「S3 正式批」+ §5 判据（预注册，B8 不得自改）
#   A = mig_base     （有季节、无迁徙 = 空臂 A）
#   B = mig_g        （gain=20，S2 推荐档）
#   C = mig_2g       （gain=40，剂量—响应上档）
#   D = mig_noseason （无季节、无迁徙 = 纯基线）
#
# 🔴 跑批前必做：
#   1. S2 闸门已过（P1–P7 全过 + 量级不越权）—— 见 /workspace/r198/S2-报告.md
#   2. `python tools/self_audit.py all && python tools/self_audit.py digest` 全绿
#   3. 确认 `sim_core.so` 与 Rust 源码同步（本任务强制 Python 路径，不开 core）
#
# 用法：bash /workspace/r198/run_s3.sh [并发数，默认 6]
set -euo pipefail
cd /tmp/tw

JOBS="${1:-6}"
echo "=== 13.8 S3 正式批：4 臂 × 6 seed × 18000 tick（并发 ${JOBS}）==="

for P in mig_base mig_g mig_2g mig_noseason; do
  echo "--- 启动 preset: ${P} ---"
  python3.12 experiments/batch_runner.py --preset "${P}" --concurrency "${JOBS}" \
    --skip-existing \
    2>&1 | tail -20
done

echo "=== 全部完成；产物在 _rerun_logs/mig_*/ ==="
ls -la _rerun_logs/mig_base/ _rerun_logs/mig_g/ _rerun_logs/mig_2g/ \
       _rerun_logs/mig_noseason/ 2>/dev/null | head -40
