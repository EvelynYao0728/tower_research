#!/usr/bin/env bash
# 分钟频多因子 LGBM 策略全流程（Expanding Window CV + 测试评估 + Lag 衰减 + 图表）
#
# 用法：
#   ./run_lgbm_strategy_tmux.sh                         # tmux 后台跑（默认）
#   USE_TMUX=0 ./run_lgbm_strategy_tmux.sh              # 前台跑
#   WORKERS=2 ./run_lgbm_strategy_tmux.sh               # 指定并行加载进程数
#   SESSION=my_lgbm ./run_lgbm_strategy_tmux.sh         # 自定义 tmux 会话名
#   OPTUNA_TRIALS=20 ./run_lgbm_strategy_tmux.sh        # 启用 Optuna 调参
#
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
RESEARCH="$(cd "$SRC/../.." && pwd)"

SESSION="${SESSION:-lgbm_strategy}"
USE_TMUX="${USE_TMUX:-1}"
WORKERS="${WORKERS:-4}"
DATA_PATH="${DATA_PATH:-$RESEARCH}"
OUTPUT_PATH="${OUTPUT_PATH:-$SRC/../output/lgbm_strategy}"
LAGS="${LAGS:-0 1 2 3}"
N_FOLDS="${N_FOLDS:-5}"
GAP_MONTHS="${GAP_MONTHS:-1}"
OPTUNA_TRIALS="${OPTUNA_TRIALS:-20}"

LOG="$OUTPUT_PATH/full_run.log"

_run_pipeline() {
  mkdir -p "$OUTPUT_PATH"

  echo "============================================================"
  echo "LGBM Strategy Pipeline  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "  data_path   : $DATA_PATH"
  echo "  output_path : $OUTPUT_PATH"
  echo "  workers     : $WORKERS"
  echo "  lags        : $LAGS"
  echo "  n_folds     : $N_FOLDS"
  echo "  gap_months  : $GAP_MONTHS"
  echo "  optuna      : $OPTUNA_TRIALS trials"
  echo "  log         : $LOG"
  echo "============================================================"

  cd "$SRC"

  local -a cmd=(
    python3 run_pipeline.py
    --data_path "$DATA_PATH"
    --output_path "$OUTPUT_PATH"
    --lags $LAGS
    --n_folds "$N_FOLDS"
    --gap_months "$GAP_MONTHS"
    --workers "$WORKERS"
  )

  if [[ "$OPTUNA_TRIALS" -gt 0 ]]; then
    cmd+=(--optuna_trials "$OPTUNA_TRIALS")
  fi

  "${cmd[@]}" 2>&1 | tee "$LOG"

  echo ""
  echo "============================================================"
  echo "FINISHED  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Output: $OUTPUT_PATH"
  echo "Log:    $LOG"
  echo "============================================================"
}

if [[ "$USE_TMUX" == "1" ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists."
    echo "Attach:  tmux attach -t $SESSION"
    echo "Or kill:  tmux kill-session -t $SESSION"
    exit 1
  fi
  RUN_CMD="cd \"$SRC\" && USE_TMUX=0 WORKERS=\"$WORKERS\" DATA_PATH=\"$DATA_PATH\" OUTPUT_PATH=\"$OUTPUT_PATH\" LAGS=\"$LAGS\" N_FOLDS=\"$N_FOLDS\" GAP_MONTHS=\"$GAP_MONTHS\" OPTUNA_TRIALS=\"$OPTUNA_TRIALS\" bash \"$SRC/run_lgbm_strategy_tmux.sh\""
  tmux new-session -d -s "$SESSION" "$RUN_CMD"
  echo "Started tmux session: $SESSION"
  echo "Attach:  tmux attach -t $SESSION"
  echo "Log:     $LOG"
else
  _run_pipeline
fi
