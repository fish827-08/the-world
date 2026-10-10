#!/usr/bin/env bash
# R198 / 13.8 S3 个体级判读分片 —— **按 4 核配额控并发**（cpu.max=4.0）
set -uo pipefail
cd /tmp/tw
mkdir -p /workspace/r198/shards
CAP="${1:-800}"
JOBS="${2:-4}"
: > /workspace/r198/shards/_list.txt
for arm in mig_base mig_g mig_2g mig_noseason; do
  for seed in 42 7 11 13 17 23; do
    echo "$arm $seed" >> /workspace/r198/shards/_list.txt
  done
done

running=0
while read -r arm seed; do
  out="/workspace/r198/shards/${arm}_${seed}.json"
  [ -s "$out" ] && continue
  python3.12 /workspace/r198/s3_judge_shard.py --arm "$arm" --seed "$seed" \
      --cap "$CAP" --out "$out" > "/workspace/r198/shards/${arm}_${seed}.log" 2>&1 &
  running=$((running+1))
  if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running-1)); fi
done < /workspace/r198/shards/_list.txt
wait
echo "全部完成：$(ls /workspace/r198/shards/*.json 2>/dev/null | wc -l)/24"
