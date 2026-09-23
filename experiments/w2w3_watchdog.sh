#!/bin/bash
# w2w3full（波2+3完整5臂）实验 watchdog（v2：flock单实例 + 日志落盘）
# 用法: TIME_LIMIT=540 bash experiments/w2w3_watchdog.sh ["A_s42 B_s7 ..."]
# 不传run列表则自动跑全部未完成run

cd /home/user/.doubao/agent_mode/workspace/digital-life-sphere

# ---- 单实例锁：定时任务延迟后可能并发触发，flock防多实例互踩 ----
LOCK_FD=200
eval "exec ${LOCK_FD}>/tmp/w2w3_watchdog.lock"
if ! flock -n ${LOCK_FD}; then
  echo "[$(date '+%H:%M:%S')] 已有watchdog实例在跑，本次退出（避免并发）"
  exit 0
fi

TIME_LIMIT=${TIME_LIMIT:-540}
PYTHON="/opt/python3.12/bin/python3"
SCRIPT="experiments/a4_verify_capacity.py"
OUTDIR="_rerun_logs/w2w3full"
SNAPDIR="_rerun_logs/w2w3full_snap"
RUNLOG="$OUTDIR/watchdog.log"
COMMON="--mode on --arm main --ticks 8000 --max-count 3240 \
  --snapshot-every 2000 --snapshot-dir $SNAPDIR \
  --distribution patchy --signal-alphabet 16 \
  --soft-cap-target 0.6 --energy-cap true \
  --forage-tradeoff-k 0.0 --log-interval 1000 --codebook 1 --no-measure"

mkdir -p "$OUTDIR" "$SNAPDIR"

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RUNLOG"; }

# 未传run列表则自动扫描全部未完成
if [ -z "$1" ]; then
  RUNS=""
  for arm in A B C D E; do
    for seed in 42 7 11 13; do
      run="${arm}_s${seed}"
      if [ -f "$OUTDIR/${run}.summary.json" ]; then
        :
      else
        RUNS="$RUNS $run"
      fi
    done
  done
else
  RUNS="$1"
fi

log "=== watchdog 启动（TIME_LIMIT=${TIME_LIMIT}s）待跑:${RUNS:-无} ==="

for run in $RUNS; do
  arm=$(echo "$run" | cut -d'_' -f1)
  seed=$(echo "$run" | sed 's/.*s//')

  if [ -f "$OUTDIR/${run}.summary.json" ]; then
    log "[skip] $run 已完成"
    continue
  fi

  case $arm in
    A) ARM_FLAGS="" ;;
    B) ARM_FLAGS="--resource-dynamics-enabled" ;;
    C) ARM_FLAGS="--resource-dynamics-enabled --perception-span 2" ;;
    D) ARM_FLAGS="--resource-dynamics-enabled --perception-span 2 --cell-occupancy-cap-enabled" ;;
    E) ARM_FLAGS="--resource-dynamics-enabled --perception-span 2 --cell-occupancy-cap-enabled --corpse-enabled --wound-enabled" ;;
  esac

  log "[run] $run 开始（arm=$arm seed=$seed）"
  timeout --signal=TERM --kill-after=15 "$TIME_LIMIT" \
    $PYTHON -u $SCRIPT $COMMON --seed "$seed" $ARM_FLAGS \
    --out "$OUTDIR/${run}.csv" >> "$RUNLOG" 2>&1
  RC=$?

  if [ -f "$OUTDIR/${run}.summary.json" ]; then
    log "[done] $run 完成（rc=$RC）"
  elif [ $RC -eq 124 ] || [ $RC -eq 143 ]; then
    tick=$(tail -1 "$OUTDIR/${run}.csv" 2>/dev/null | cut -d',' -f1)
    log "[paused] $run 窗口到期，停在 tick=${tick:-0}/8000，下次续跑"
  else
    tick=$(tail -1 "$OUTDIR/${run}.csv" 2>/dev/null | cut -d',' -f1)
    log "[ERROR] $run 异常退出 rc=$RC tick=${tick:-0}（详见 $RUNLOG，不静默）"
  fi
done

DONE_N=$(ls $OUTDIR/*.summary.json 2>/dev/null | wc -l)
log "=== watchdog 结束：已完成 ${DONE_N}/20 ==="
