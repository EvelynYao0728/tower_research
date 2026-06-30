# single_factor_bt — 单因子横截面回测

按 `单因子回测系统设计说明` 实现的高性能、向量化、多进程的单因子回测工具。

* 输入：因子文件/目录 + label 文件/目录（支持 `parquet` 与 `feather/.fea`，长表 / 宽表自动识别）
* 计算：IC、RankIC、ICIR、RankICIR、整体多空收益、每组组内多空（top20% / bot20%）
* 输出：每个因子单独子目录（`output/<factor_name>/`） + 根目录唯一 `output/summary.csv`（累积合并所有跑过的因子）
* 安全：硬编码拒绝任何写入 `/home/yzyao.25/research/public` 的尝试

> **2026-05 更新**：核心计算已对齐 `public/backtest_new`（RankIC 修正、分层 D1/D10 语义、日历跨度年化、可选 PIT universe）。

---

## 安装

```bash
cd /home/yzyao.25/research/backtest
pip install -r requirements.txt
# 或：pip install -e .  之后用 `single-factor-bt -h`
```

依赖：`numpy pandas pyarrow scipy matplotlib tqdm polars`

---

## 快速开始

```bash
cd /home/yzyao.25/research/backtest

# 1) 默认读 data/simple_factors 与 data/label；默认用 data/trade_date.csv 限定交易日（文件不存在则忽略）
python -m single_factor_bt.cli

# 2) 只跑指定因子
python -m single_factor_bt.cli -c imbalance_mean

# 3) 显式指定数据路径（与默认等价示例）
python -m single_factor_bt.cli \
    -f /home/yzyao.25/research/data/simple_factors \
    -l /home/yzyao.25/research/data/label \
    --trade-date-csv /home/yzyao.25/research/data/trade_date.csv
```

输出目录：

```
output/
├── summary.csv                         # 所有跑过的因子汇总，每行一个因子，4 位小数
├── imbalance_mean/
│   ├── metrics_per_minute.csv          # 每分钟横截面指标 + 各 decile 收益
│   ├── decile_inner_long_short.csv     # 题目 f) 的「每组组内多空」表，每行一个时间
│   ├── decile_inner_long_short.png     # 题目 f) 要求的曲线图
│   ├── long_short_overall.png          # 整体 多/空/多空 累计收益
│   └── _cache/                         # 每天分片缓存
├── imbalance_std/
│   └── ...
└── ...
```

`summary.csv` **累计合并**：每次运行只刷新本次涉及的因子行，其它因子保留。

---

## CLI 参数（标准 argparse）

```text
usage: single-factor-bt [-h] [-f PATH] [-l PATH] [-o DIR] [-c COL]
                        [--label-col COL] [-n K] [--inner-q Q]
                        [--sample-every-n-minutes M]
                        [--session-start HHMM] [--session-end HHMM]
                        [-w N] [--no-cache] [--trade-date-csv FILE] [-v]
```

| 选项 | 含义 |
|---|---|
| `-f, --feature` | **因子**文件或目录（默认 `/home/yzyao.25/research/data/simple_factors`） |
| `-l, --label` | **label** 文件或目录（默认 `/home/yzyao.25/research/data/label`） |
| `-o, --output` | 输出根目录（默认 `/home/yzyao.25/research/backtest/output`） |
| `-c, --factor-col` | 因子列名。**不指定时回测 feature 文件中所有可用因子列** |
| `--label-col` | 收益率列名；不传时按 `ex_log_ret_10m → log_ret_10m → ex_ret_10m → ret_10m` 自动探测 |
| `-n, --n-groups` | 分组数，默认 10 |
| `--inner-q` | 组内 top/bot 比例，默认 0.20 |
| `--sample-every-n-minutes` | 采样步长，默认 1 |
| `--session-start/--session-end` | 交易时段（HHMM 整数，默认 930~1550） |
| `-w, --workers` | 进程数，默认 `min(8, ncpu-1)` |
| `--no-cache` | 关闭按天缓存 |
| `--trade-date-csv` | 限定交易日范围（默认 `data/trade_date.csv`，不存在则忽略） |
| `--universe-file` | 可选 PIT universe JSON（默认路径 `data/2025_universe.json`，不存在则跳过） |
| `--no-universe` | 关闭 PIT 过滤（**默认**） |
| `-v, --verbose` | 打印更详细信息 |

### 年化系数（日历跨度）

| 列 | 含义 |
|---|---|
| `n_samples` | 有效横截面总数 |
| `n_days` | 实际有数据的交易日数 |
| `biz_days_in_span` | `[date_min, date_max]` 内工作日数 |
| `obs_per_year` | `n_samples / (biz_days_in_span / 252)` |

收益/波动/Sharpe 均用 `obs_per_year` 年化，不再硬编码 9828。

> 完整 help：`python -m single_factor_bt.cli -h`

---

## 数值精度

| 列类别 | 小数位数 |
|---|---|
| `IC, RankIC, IC_mean, IC_std, ICIR, RankIC_mean, RankIC_std, RankICIR` | **4** |
| `long_ret, short_ret, long_short_ret, decile_*_ret, decile_*_inner_long_short, *_mean` | 6 |
| `n_samples` / `factor` 名 | 不变 |

---

## 设计说明（与原文档对应）

| 文档要求 | 实现 |
|---|---|
| 1.a IC | `corr(factor, ret)` per minute（向量化） |
| 1.b RankIC | `corr(rank(factor), rank(ret))`，**两边都 rank** |
| 1.c ICIR / 1.d RankICIR | `mean / std`，`ddof=1` |
| 1.e 多/空/多空 | 因子升序 10 等分；`decile_1`=最低；多=D10；空=D1；多空=(D10−D1)/2 |
| 1.e.iii 分层质量 | `decile_1_ret`…`decile_10_ret`、`monotonicity_score`、`top_bottom_spread` 写入 `summary.csv` |
| 1.e.i 卖点 10 分钟 | 直接使用 label 的 `*_10m` 收益列 |
| 1.e.ii 因子为空剔除 | `valid_f = isfinite(factor)` 后再 rank/分组 |
| 1.f 组内多空 | 每组 `top inner_q` 多 + `bot inner_q` 空 |
| 2.a/b/c 数据要求 | 自动识别长表（按日分片）和宽表（`Datetime, A...Z`） |
| 3.a/b 输出 | `decile_inner_long_short.{csv,png}` + `summary.csv` |

### 性能优化

* PyArrow 列裁剪只读所需列
* 每天独立进程（`spawn` + `ProcessPoolExecutor`）
* **一次 IO 同时算多个因子**：每个 worker 读当天 parquet 一次，project 出所有请求的 factor cols，对 label merge 一次，逐 factor pivot+计算
* 横截面计算全部 numpy 向量化（`bincount` 做 group-mean、`pandas.DataFrame.rank(axis=1)` 做横截面 rank）
* 因子值/收益使用 `float32`，ticker 转 `category`
* 每天结果落盘 `output/<factor>/_cache/<date>.parquet`，下次同因子秒级命中
* 安全保护 `safety.assert_safe_output` 禁止写入 `research/public/`

### 测试

```bash
PYTHONPATH=. python tests/test_engine_small.py
```

---

## 目录结构

```
backtest/
├── single_factor_bt/
│   ├── cli.py        # argparse 入口  (-f feature, -l label, -c factor-col 可选)
│   ├── engine.py     # 多进程驱动 + 多因子单次 IO
│   ├── io_utils.py   # parquet/feather + 长/宽表自动识别
│   ├── metrics.py    # 向量化横截面指标
│   ├── plot.py       # matplotlib 出图
│   ├── universe.py   # 可选 PIT membership 过滤
│   └── safety.py     # 写路径硬保护
├── scripts/run_example.sh
├── tests/test_engine_small.py
├── output/           # 回测结果（本地生成）
├── pyproject.toml
└── requirements.txt
```
