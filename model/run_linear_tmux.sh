#!/usr/bin/env bash
# 在 tmux 后台用 因子库_final 全部因子训练 Ridge 线性模型
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
RESEARCH="$(cd "$ROOT/.." && pwd)"

SESSION="${SESSION:-linear_train}"
LOG="$ROOT/output/linear_full_run.log"
WORKERS="${WORKERS:-1}"
REFRESH_CACHE="${REFRESH_CACHE:-0}"

FACTOR_ROOT="${FACTOR_ROOT:-$RESEARCH/因子库_final}"
REGISTRY="${REGISTRY:-$FACTOR_ROOT/factor_registry.yaml}"
LABEL_ROOT="${LABEL_ROOT:-$RESEARCH/data/label}"
OUTPUT="${OUTPUT:-$ROOT/output/linear}"

mkdir -p "$ROOT/output"

CACHE_ARGS=()
if [[ "$REFRESH_CACHE" == "1" ]]; then
  CACHE_ARGS+=(--refresh-cache)
fi

RUN_CMD="cd \"$ROOT\" && python3 run_linear.py \
  --factor-root \"$FACTOR_ROOT\" \
  --registry \"$REGISTRY\" \
  --label-root \"$LABEL_ROOT\" \
  --output \"$OUTPUT\" \
  --workers \"$WORKERS\" \
  ${CACHE_ARGS[*]:-} \
  2>&1 | tee -a \"$LOG\""

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' 已存在。"
  echo "  查看: tmux attach -t $SESSION"
  echo "  结束: tmux kill-session -t $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" "$RUN_CMD"

echo "已在 tmux 启动线性模型训练: $SESSION"
echo "  因子库: $FACTOR_ROOT"
echo "  registry: $REGISTRY"
echo "  workers = $WORKERS"
echo "  refresh_cache = $REFRESH_CACHE"
echo "  进入: tmux attach -t $SESSION"
echo "  退出: Ctrl-b d"
echo "  日志: tail -f $LOG"
