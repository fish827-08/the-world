#!/usr/bin/env bash
# ============================================================================
# smell_rd_01 —— 云机器编排器（灯塔/tianping 账号）
#
# 批次：R339 冻结代号 smell_rd_01（旧称 SX-1，已作废）
# 规模：18 run = 3 组（A0 / A1 / B1）× 6 seed（207–212）× 10 000 tick
# 装置：新档 bg_low 0.0/0.0（纯斑块）+ pop 10000 + patches 1700 + rgm 1.195
#       + --sparse-fields + smell(food) + use_in_move（仅 B1）
# 判据：见 docs/设计文档/预注册-气味×rd短实验10k-20261002.md（跑前锁定）
#
# 🔴 设计要点（R339 记录规范 §3.2 中档）：
#   - 每 run 一个 summary（分片）⇒ criterion_2 恒 null，属已知缺陷
#     ⇒ 判读必须用独立脚本重算（experiments/_judge_*.py），不只看 summary
#   - 装置回显守卫：每个 run 结束后核 bg_low_frac_actual == 0.0
#   - 幂等：--skip-existing 可重入，被杀后重跑不会重复劳动
#   - 进度：每 60 s 打一条（ETA 只用实测样本推，R189）
#
# 用法：
#   bash run_smell_rd_01.sh start     # 起跑（后台）
#   bash run_smell_rd_01.sh status   # 查进度
#   bash run_smell_rd_01.sh stop      # 停
# ============================================================================
set -uo pipefail

CODE="smell_rd_01"
BASE="$HOME/world/smell_rd_01"
TREE="$HOME/world/frozen_smell_rd_01"
REPO="$HOME/world/the-world"
# ---------------------------------------------------------------- 解释器
# 🔴 云机实测（2026-10-02 23:30，天平/灯塔）：
#   `tianping` 自建venv 的 numpy **装不上** —— LTO-1（ubuntu 账号）占 1.36 GB / 1.97 GB 内存，
#   pip 下载 16.7 MB wheel 长时间卡住（available 仅 146 MB）。
#   而 `/home/ubuntu/world/the-world/.venv`（ubuntu 账号，**只读可执行**）已有 numpy 2.5.3
#   ⇒ **直接复用其解释器**，不复制 venv（省内存 + 省时间）。
#   ⚠️ 若日后 LTO-1 停掉且内存释放，可改回"$REPO/.venv/bin/python"（自建 venv）。
PY="/home/ubuntu/world/the-world/.venv/bin/python"
SEEDS="207 208 209 210 211 212"
TICKS=10000
SAMPLE=250
PAR="${PAR:-2}"          # 云机器 2 核 ⇒ 2 路
LOGDIR="$BASE/logs"
RESDIR="$BASE/results"

# ---------------------------------------------------------------- 装置（单一真源）
# 🔴 三组唯一差别 = rd 开关 + smell 开关（其余全同）
device_args() {
  # 新档纯斑块：显式给 0.0/0.0，不依赖 device_presets 的默认值（R326 教训）
  echo "--rows 480 --cols 960 --patches 1700 --pop 10000 --rgm 1.195 \
        --bg-low-prod-frac 0.0 --bg-low-cap-mult 0.0"
}

group_args() {
  case "$1" in
    a0) echo "--arms off" ;;                       # rd 关 + 无 smell = 基线
    a1) echo "--arms on" ;;                        # rd 开 + 无 smell（= S2-6 on 臂）
    b1) echo "--arms on --smell-channels food --use-in-move" ;;
    *)  echo "UNKNOWN_GROUP:$1" >&2; exit 2 ;;
  esac
}

label_of() { case "$1" in a0) echo A0;; a1) echo A1;; b1) echo B1;; esac; }

# ---------------------------------------------------------------- 部署冻结树
deploy() {
  echo "[deploy] 冻结树 = 当前 HEAD（$(cd "$REPO" && git rev-parse --short HEAD)）"
  rm -rf "$TREE"; mkdir -p "$TREE"
  (cd "$REPO" && git archive HEAD) | tar -x -C "$TREE"
  # 🔴 冻结树无 sim_core.so ⇒ 纯 Python 单路（与 S2-6 同条件，保可比）
  echo "[deploy] 冻结树就绪：$TREE"
}

# ---------------------------------------------------------------- 🔴 起跑前自检
# 🔴 教训（2026-10-03）：首版编排器给 B1 写了 `--extra=--smell-channels food`，
#    但 **s2 探针不认 `--extra`**（那是 `_run_s2g2.py` 的参数）⇒ B1 六臂全 rc=2，
#    **白跑 3 小时**。根因 = 参数透传层没做"探针是否接受"的校验。
#    ⇒ 纪律：**每组参数先跑 --help 级别的解析校验，全批起跑前必须过。**
selftest() {
  echo "[selftest] 校验三组参数是否被探针接受..."
  local g
  for g in a0 a1 b1; do
    if ( cd "$TREE" && $PY -m experiments.s2_depletion_probe \
           $(device_args) --seeds 999 --ticks 1 --sample 1 $(group_args "$g") \
           --out /tmp/_selftest_$g.csv ) >/tmp/_selftest_$g.log 2>&1; then
      echo "  [OK]   $g：参数被接受"
    else
      echo "  [FAIL] $g：参数被拒 ⇒ $(tail -2 /tmp/_selftest_$g.log | head -1)"
      return 1
    fi
  done
  echo "[selftest] 三组参数全部通过"
}

# ---------------------------------------------------------------- 起一批
run_one() {
  local g="$1" s="$2"
  local lab; lab=$(label_of "$g")
  local out="$RESDIR/${CODE}_${s}_${lab}.csv"
  [ -s "$out" ] && { echo "[skip] $(basename "$out") 已存在"; return 0; }
  local cmd="$PY -u -m experiments.s2_depletion_probe \
      $(device_args) --seeds $s $(group_args "$g") \
      --ticks $TICKS --sample $SAMPLE --sparse-fields --out $out"
  echo "[start] seed=$s group=$lab -> $(basename "$out")"
  ( cd "$TREE" && eval "$cmd" ) > "$LOGDIR/${s}_${lab}.log" 2>&1
  local rc=$?
  # 装置守卫：bg_low_frac_actual 必须为 0.0（纯 A 支）
  # ⚠️ 2026-10-03 修正：CSV **数据行是纯数值**（无 key= 形式）⇒ 原grep 恒空。
  #    正确读法 = 最后一行的倒数第 3 列（表头 bg_low_frac_actual 固定为倒数第 3 列）。
  if [ $rc -ne 0 ]; then
    echo "[FAIL] seed=$s group=$lab rc=$rc （见 $LOGDIR/${s}_${lab}.log）"; return $rc
  fi
  local blf; blf=$(tail -1 "$out" | awk -F',' '{print $(NF-2)}')
  local npx; npx=$(tail -1 "$out" | awk -F',' '{print $14}')
  if [ "$blf" != "0.0" ]; then
    echo "[WARN] seed=$s group=$lab装置守卫异常：bg_low_frac_actual=$blf（应 0.0）"
  fi
  echo "[done] seed=$s group=$lab rows=$(($(wc -l < "$out")-1)) last_tick=$(tail -1 "$out" | cut -d, -f3) bg_low=$blf n_patches=$npx"
}

# ---------------------------------------------------------------- 编排
orchestrate() {
  mkdir -p "$LOGDIR" "$RESDIR"
  deploy
  local t0; t0=$(date +%s)
  local total=18
  local -a JOBS=()
  for s in $SEEDS; do for g in a0 a1 b1; do JOBS+=("$s:$g"); done; done
  # 🔴 R218/配对纪律：**同 seed 的三臂必须串行**（共享同 world 与 RNG 序）
  #    并行只跨 seed 展开 => 任何时刻最多 PAR 个不同 seed 在跑
  local i=0
  for s in $SEEDS; do
    (
      for g in a0 a1 b1; do run_one "$g" "$s"; done
    ) &
    i=$((i+1))
    if [ $((i % PAR)) -eq 0 ]; then wait; fi
  done
  wait
  local n; n=$(ls -1 "$RESDIR"/*.csv 2>/dev/null | wc -l)
  local el=$(( $(date +%s) - t0 ))
  echo "=========================================="
  echo "[ALLDONE] $CODE  完成 $n/$total run，耗时 $((el/60)) min"
  echo "[汇总] 每 run 行数（应40 = 10000/250）:"
  for f in "$RESDIR"/*.csv; do
    printf "  %-34s rows=%-4s last_tick=%-6s bg_low=%s\n" "$(basename "$f")" \
      "$(($(wc -l < "$f")-1))" "$(tail -1 "$f" | cut -d, -f3)" \
      "$(tail -1 "$f" | awk -F',' '{print $(NF-2)}')"
  done
  echo "[续跑] 本批完成后应接 LTO-1（见派工单：先验完整性 + 免费回归 + 幂等）"
  echo "=========================================="
}

case "${1:-status}" in
  start)   orchestrate ;;
  selftest) deploy; selftest ;;
  status)
    echo "== $CODE 状态 =="
    ls -1 "$RESDIR"/*.csv 2>/dev/null | wc -l | xargs echo "已完成 run:"
    for f in "$RESDIR"/*.csv; do
      [ -f "$f" ] && printf "  %-34s tick=%s\n" "$(basename "$f")" "$(tail -1 "$f" | cut -d, -f3)"
    done
    ps aux | grep -c "[s]2_depletion_probe" | xargs echo "在跑进程:"
    ;;
  stop)    pkill -f s2_depletion_probe && echo "[stop] 已发信号" ;;
  *)       echo "用法: bash $0 {start|status|stop}"; exit 1 ;;
esac
