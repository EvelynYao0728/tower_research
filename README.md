# 高频因子研究与回测

NYSE TAQ 分钟级因子库、单因子回测与线性模型训练流水线。

## 项目结构

```
research/
├── 因子库_final/              # 28 因子库 + 回测结果 + 因子计算
│   ├── compute/               # 28 因子统一计算（B/C/D/E + MB/S + O）
│   ├── factor_registry.yaml
│   ├── FACTORS_CONSTRUCTION.md
│   ├── output/                # 单因子回测结果
│   └── run_all_backtests.py
├── base_data_process/         # simple_factors / label 预处理
├── backtest/                  # 单因子回测引擎
├── model/
│   ├── linear/                # Ridge 线性模型
│   └── src/                   # 分钟频多因子 LGBM 策略
└── data/
    └── trade_date.csv
```

## 28 个入库因子

详见 [`因子库_final/FACTORS_CONSTRUCTION.md`](因子库_final/FACTORS_CONSTRUCTION.md)。

| 族 | 数量 | 模块 |
|----|------|------|
| B/C/D/E | 8 | `因子库_final/compute/factors.py` |
| S + MB | 15 | `因子库_final/compute/factors.py` |
| O | 5 | `因子库_final/compute/factors.py` |

## 数据预处理

```bash
# simple_factors（MB 族上游）
python -m base_data_process.run_simple_factors --all-2026

# label
python -m base_data_process.run_label --all-2026
```

## 因子计算

```bash
# 全量生成（断点续跑，输出到 因子库_final/<factor_name>/）
python 因子库_final/compute/run.py --all

# 指定日期 / 因子族
python 因子库_final/compute/run.py --dates 20260102
python 因子库_final/compute/run.py --all --groups bcde
python 因子库_final/compute/run.py --all --groups mb
python 因子库_final/compute/run.py --all --groups o
```

## 回测

```bash
python 因子库_final/run_all_backtests.py
```

## 模型

```bash
python model/run_linear.py
python model/src/run_pipeline.py
```

## 依赖

Python 3.10+：pandas, pyarrow, polars, numpy, scikit-learn, numba, tqdm, matplotlib。
