#!/usr/bin/env bash
# 示例：跑 simple_factors 下所有因子（不指定 -c 即全部）
set -euo pipefail
cd "$(dirname "$0")/.."

python -m single_factor_bt.cli \
    -f /home/yzyao.25/research/data/simple_factors \
    -l /home/yzyao.25/research/data/label \
    --trade-date-csv /home/yzyao.25/research/data/trade_date.csv \
    --label-col ex_log_ret_10m
