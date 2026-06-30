# 线性截面收益预测模型 · 完整技术文档

**文档路径**：`model/linear/LINEAR_MODEL.md`  
**代码入口**：`model/run_linear.py`  
**一键训练**：`model/run_linear_tmux.sh`  
**最新全量产物**：`model/output/linear/`（配置快照见 `run_config.json`）  
**最后更新**：2026-06-15  

---

## 一、模型定位与目标

本模型是一个 **截面（cross-sectional）多因子线性回归器**，用 `因子库_final` 中全部可用因子的分钟级观测，线性组合预测每只股票在未来 10 分钟的对数超额收益。

| 项目 | 说明 |
|------|------|
| 算法 | Ridge 回归（L2 正则化最小二乘） |
| 求解方式 | 闭式 Gram 矩阵解（无需迭代优化） |
| 正则强度 | 在验证集上按 **Rank IC** 网格搜索 `alpha` |
| 预测标签 | `ex_log_ret_10m`：未来 10 分钟 mid-price 对数超额收益 |
| 样本粒度 | 股票 × 分钟（long-format panel） |
| 与 LGB 关系 | 共用 `data.py` 数据管道与 `linear/evaluate.py` 评估框架；LGB 的评估/出图模块即源于此包 |
| 与 LSTM 关系 | LSTM 使用时序窗口；线性模型将每个 `(date, ticker, minute)` 视为独立样本 |

**设计意图**：作为 **可解释基线（interpretable baseline）**。系数符号与大小直接反映因子对 forward return 的线性边际贡献；同时提供与 LGB 同口径的 IC / 分位组合指标，便于对比非线性增益。

**模型形式**（中心化后拟合，等价于带截距）：

\[
\hat{y}_i = \beta_0 + \sum_{f=1}^{F} \beta_f \cdot z_{i,f}
\]

其中 \(z_{i,f}\) 为经 winsorize + 截面 z-score 后的因子值，\(\beta_0\) 为截距（最新 run 约 \(-10^{-7}\)，接近 0）。

---

## 二、代码结构与调用链

```
model/
├── run_linear.py           # CLI 入口
├── run_linear_tmux.sh      # tmux 后台全量训练
├── config.py               # LinearModelConfig 默认超参
├── data.py                 # 因子/标签加载、预处理、三路切分
├── linear/
│   ├── trainer.py          # Ridge 拟合、alpha 搜索、保存 artifact
│   ├── evaluate.py         # IC / 分位组合 / 回归指标（LGB 复用）
│   ├── plot.py             # 系数图 + 评估图（英文标签）
│   ├── LINEAR_MODEL.md     # 本文档
│   └── __init__.py
```

**一次完整运行的流水线**：

1. `discover_factor_names()`：从 registry + 目录扫描因子列表  
2. `available_dates()`：label 与全部因子 parquet 日期交集  
3. `split_dates()`：按时间 **60% / 20% / 20%** 切 train / valid / test  
4. `load_and_split_panels()`：分 split 加载并预处理 panel  
5. `to_xy()`：提取 `(X, y)`，丢弃 label 或任一因子非有限的行  
6. `_fit_ridge()`：在 valid 上按 Rank IC 选 `alpha`，再在 train+valid 上 refit  
7. `save_model_artifacts()`：系数表、meta  
8. `build_evaluation_report()`：train / valid / test 截面 IC、分位收益  
9. `generate_all_plots()`：系数图 + valid/test 评估图（**不含 train 出图**）  
10. 写入 `run_config.json`

---

## 三、数据来源

### 3.1 路径（默认值）

| 配置项 | 默认路径 |
|--------|----------|
| `factor_root` | `/home/yzyao.25/research/因子库_final` |
| `label_root` | `/home/yzyao.25/research/data/label` |
| `trade_date_csv` | `/home/yzyao.25/research/data/trade_date.csv` |
| `registry` | `因子库_final/factor_registry.yaml` |
| `output_dir` | `/home/yzyao.25/research/model/output/linear` |
| `cache_dir` | `/home/yzyao.25/research/model/cache/panels` |

> 仅使用 `factor_root` 目录因子（与 LGB v1 相同），不加载 `new_feature_1` panel 宽表。

### 3.2 数据格式

**Long parquet 列**：`date, sym_root, sym_suffix, minute, <factor_or_label>`

**合并键**：`(date, minute, ticker)`，其中 `ticker = sym_root[.sym_suffix]`

**交易时段**（默认）：

| 参数 | 值 |
|------|-----|
| `session_start` | 931（9:31 含） |
| `session_end` | 1559（15:59 含） |

### 3.3 标签

- **列名**：`ex_log_ret_10m`
- **含义**：未来 10 分钟 mid-price 对数超额收益
- **预处理**：**不做** winsorize / z-score，保持原始数值用于训练与评估

### 3.4 因子列表（最新 run：37 个）

来源：`output/linear/run_config.json`（registry 顺序）：

| 族 | 因子 |
|----|------|
| B | `B4_cross_dup_price_spread_bps`, `B5_trade_dup_vwap_premium` |
| C | `C3_oir`, `C5_layer_stacking_x_flow_signed_1m`, `C8_spread_asymm_vol` |
| D | `D3_nbbo_size_imb`, `D5_spread_diff_mean` |
| E | `E6_trade_position_skew` |
| M | `F4_intraday_return`, `F5_close_location_value`, `F11_liquidity_adjusted_return`, `F12_micro_price_pressure` |
| S | `imbalance_mean`, `imbalance_pos_ratio`, `imbalance_skew` |
| N | `N1_microprice_deviation_bps`, `N11_phantom_asymm`, `N7_bid_lead_excess` |
| MB | `clv`, `clv_x_imb`, `dist_from_5m_high`, `dist_from_5m_low`, `imb_current`, `imb_trend`, `liq_adj_ret`, `open_mean_dev`, `range_pos_5m`, `ret_10m_past`, `ret_1m`, `ret_5m`, `spread_x_imb`, `vol_adj_imb` |
| T | `avg_trade_size`, `count_ofi`, `ofi_1m`, `ofi_5m_avg`, `vwap_mid_dev` |

---

## 四、样本构建与预处理

### 4.1 日级合并

与 LGB 相同：对每个交易日 inner-join 全部因子 + label，仅保留 **所有因子与 label 同时非空** 的 `(ticker, minute)`。

### 4.2 因子截面预处理

**`cross_section_winsorize_zscore`**（仅作用于因子，不作用于 label）：

| 步骤 | 参数 | 说明 |
|------|------|------|
| Winsorize | `winsorize_quantile = 0.01` | 每分钟截面 1%–99% 缩尾 |
| Z-Score | 同 minute 截面 | \(z = (x - \mu) / \sigma\)，样本 < 2 或 std < 1e-12 时置 0 |
| 最小截面 | ≥ 5 只股票 | 才做 winsorize |

### 4.3 训练矩阵（`to_xy`）

- 丢弃 label 非有限或 **任一因子 NaN** 的行  
- `X`：`float32`，shape `(n, 37)`  
- `y`：`float32`，raw label  

### 4.4 Panel 缓存

与 LGB 共用 `model/cache/panels/<fingerprint>/`，指纹由 `factor_cols + label_col + winsorize_q + session` 决定。

---

## 五、训练 / 验证 / 测试划分

| 参数 | 值 | 说明 |
|------|-----|------|
| `train_ratio` | **0.60** | 前 60% 交易日 |
| `valid_ratio` | **0.20** | 中间 20% |
| test | **剩余 ~20%** | 末段 held-out |

**最新全量 run 日期**（`model_meta.csv`）：

| Split | 起止日期 | 天数 | 样本行数 | 用途 |
|-------|----------|------|----------|------|
| Train | 2025-01-02 ~ 2025-08-08 | 150 | 27,060,288 | 拟合 Ridge |
| Valid | 2025-08-11 ~ 2025-10-20 | 50 | 8,718,632 | **选择 `alpha`** |
| Test | 2025-10-21 ~ 2025-12-31 | 42 | 1,443,191 | **最终报告（无参与调参）** |

> 划分严格按时间顺序，无 shuffle。

**与 LGB 的切分差异**（对比时注意）：

| 模型 | 划分 | 末段评估集 | 末段日期 |
|------|------|------------|----------|
| **Linear** | 60/20/20 | **Test**（未参与 alpha 选择后的 refit 评估） | 2025-10-21 ~ 2025-12-31 |
| **LGB v1/v2** | 80/20 | **Valid**（同时用于 early stopping） | 2025-10-21 ~ 2025-12-31 |

Linear 的 **Test** 与 LGB 的 **Valid** 时间段相同，可直接对比 Rank IC；但 Linear Test 是真正的 held-out（alpha 仅在更早的 valid 段上选），LGB valid 参与了 early stopping。

---

## 六、Ridge 回归与 Alpha 选择

### 6.1 超参数

| 参数 | 值 | CLI | 说明 |
|------|-----|-----|------|
| `ridge_alphas` | `1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000` | ❌ 未暴露 | 对数均匀网格，共 8 档 |
| **选中 alpha** | **1000.0** | — | 最新全量 run |
| `fit_intercept` | True | — | 显式截距 |
| 样本权重 | 无 | — | 所有 `(ticker, minute)` 等权 |
| 因子去冗余 | 无 | — | 37 因子全部进入 |

### 6.2 求解算法（`trainer.py`）

**Step 1 — 中心化**：

\[
\tilde{X} = X - \bar{X}, \quad \tilde{y} = y - \bar{y}
\]

**Step 2 — Gram 矩阵闭式解**（对每个候选 `alpha`）：

\[
\beta = (\tilde{X}^T \tilde{X} + \alpha I)^{-1} \tilde{X}^T \tilde{y}
\]

使用 `np.linalg.solve`，无需 sklearn 迭代，适合 3700 万级样本（仅计算 \(F \times F\) 的 Gram 矩阵，\(F=37\)）。

**Step 3 — Alpha 选择**：在 **valid** 集上计算预测 vs label 的 **全局 Spearman Rank IC**（非截面 minute 均值 IC），取 Rank IC 最高的 `alpha`。

**Step 4 — Refit**：用选定的 `alpha`，在 **train + valid 合并数据** 上重新估计 \(\beta\)，作为最终模型。

```python
# 伪代码
for alpha in ridge_alphas:
    weights = solve(Gram, xy, alpha)
    score = spearmanr(predict(valid), y_valid)
best_alpha = argmax(score)
refit on train ∪ valid with best_alpha
```

### 6.3 为何选 alpha=1000（强正则）？

| 现象 | 含义 |
|------|------|
| 选中网格最大值 1000 | 8 档中最强 L2 正则最优 |
| 37 因子高度冗余 | Gram 矩阵条件数大，需强 shrinkage |
| `C3_oir` 与 `D3_nbbo_size_imb` 系数几乎相同 | 共线性导致 Ridge 无法区分，强正则稳定系数 |

强正则 → 系数偏小、模型保守，有利于泛化但可能欠拟合非线性交互。

### 6.4 截距

最新 run：`intercept ≈ -1.05 × 10⁻⁷`（`model_meta.csv`），可视为 0。符合 z-score 因子 + 中心化拟合的预期。

---

## 七、评估体系

与 LGB 共用 `linear/evaluate.py` + `backtest/single_factor_bt/metrics.py`。

### 7.1 配置

| 参数 | 值 |
|------|-----|
| `n_deciles` | 10 |
| `annualization_days` | 252 |

### 7.2 截面指标（每个 `(date, minute)` 切片）

| 指标 | 定义 |
|------|------|
| **IC** | 该分钟 prediction vs `ex_log_ret_10m` 的 Pearson 相关 |
| **RankIC** | Spearman 秩相关 |
| **分位组合** | D1~D10 等权收益，long/short/L-S |

### 7.3 全量 run 评估结果

来源：`output/linear/evaluation_summary.csv`

| Split | n_rows | n_days | IC | ICIR | RankIC | RankICIR | RMSE | Hit Rate | L/S Ret (ann.) | Sharpe |
|-------|--------|--------|-----|------|--------|----------|------|----------|----------------|--------|
| **train** | 27,060,288 | 150 | 0.0519 | 0.826 | 0.0534 | 0.829 | 0.00217 | 51.83% | 16.39 | 175.25 |
| **valid** | 8,718,632 | 50 | 0.0635 | 1.097 | 0.0644 | 1.096 | 0.00196 | 52.16% | 19.80 | 255.26 |
| **test** | 1,443,191 | 42 | 0.0468 | 0.390 | **0.0458** | 0.390 | 0.00213 | 51.58% | 12.75 | 82.96 |

**解读要点**：

- **Test Rank IC ≈ 4.6%**：held-out 末段 42 天，与单因子相比仍显著，但低于 valid（6.4%）→ valid 用于选 alpha 带来一定乐观偏差  
- **Test 与 LGB valid 对比**（同时间段 2025-10-21 ~ 2025-12-31）：
  - Linear test Rank IC **0.0458** vs LGB v1 valid **0.0543** vs LGB v2 valid **0.0526**  
  - LGB 非线性模型在该时段仍有 ~0.7–0.9 pp Rank IC 优势  
- **Sharpe**：Test 83 vs LGB v2 valid 101；线性多空仍可用但弱于 GBDT  
- 评估报告含 train 数值，但 **出图仅 valid + test**（`plot.py` 设计）

---

## 八、因子系数（可解释性）

保存路径：`output/linear/coefficients.csv`  
按 **|coefficient|** 降序排列。

**Top 10（最新 run）**：

| 排名 | 因子 | 系数 | 方向解读 |
|------|------|------|----------|
| 1 | `liq_adj_ret` | +0.000199 | 流动性调整收益 ↑ → 预测收益 ↑ |
| 2 | `F11_liquidity_adjusted_return` | −0.000176 | 与 `liq_adj_ret` 同源，符号相反（共线性） |
| 3 | `count_ofi` | +0.000052 | 成交流方向 ↑ → 预测收益 ↑ |
| 4 | `B4_cross_dup_price_spread_bps` | +0.000045 | 并发成交溢价 ↑ → 预测收益 ↑ |
| 5 | `N1_microprice_deviation_bps` | +0.000041 | 微观价格偏高 → 预测收益 ↑ |
| 6 | `F4_intraday_return` | −0.000039 | 短期动量反转 |
| 7 | `imbalance_skew` | −0.000037 | 不平衡偏度 ↑ → 预测收益 ↓ |
| 8 | `N11_phantom_asymm` | −0.000035 | 虚假报价不对称 ↑ → 预测收益 ↓ |
| 9 | `ret_1m` | +0.000031 | 分钟收益（与 F4 部分重叠） |
| 10 | `clv` | −0.000028 | CLV 偏高 → 预测收益 ↓ |

**共线性示例**：

- `C3_oir` 与 `D3_nbbo_size_imb`：系数均为 **−1.873×10⁻⁵**（精确到数值误差）  
- `imb_current` 与 `imbalance_mean`：系数均为 **8.027×10⁻⁷**  

Ridge 在冗余因子间 **分配相同 shrinkage 系数**，无法像 LGB gain 或去冗余那样选出代表因子。这是线性模型使用全量 37 因子时的固有局限。

---

## 九、输出产物

### 9.1 目录结构

```
model/output/linear/
├── coefficients.csv           # 37 因子 Ridge 系数
├── model_meta.csv             # alpha、截距、样本量、日期范围
├── evaluation_summary.csv     # train / valid / test 汇总
├── run_config.json            # 完整可复现配置
├── train/
│   ├── minute_metrics.csv
│   └── daily_metrics.csv
├── valid/
│   ├── minute_metrics.csv
│   └── daily_metrics.csv
├── test/
│   ├── minute_metrics.csv
│   └── daily_metrics.csv
└── figures/
    ├── coefficients.png       # 系数条形图
    ├── metrics_summary.png    # Test 集指标卡片
    ├── valid/                 # ic_distribution, cumulative_ic, daily_ic,
    │                          # decile_returns, calibration, residuals
    └── test/                  # 同上 6 张
```

> 无 `lgb_model.txt` 类持久化文件；模型完全由 `coefficients.csv` + `intercept` + `alpha` 定义。推理时按相同预处理 pipeline 生成 z-score 因子后线性组合即可。

### 9.2 `model_meta.csv`（最新 run）

| 字段 | 值 |
|------|-----|
| `label_col` | ex_log_ret_10m |
| `alpha` | 1000.0 |
| `intercept` | −1.05e-07 |
| `n_factors` | 37 |
| `train_rows` / `valid_rows` / `test_rows` | 27060288 / 8718632 / 1443191 |

---

## 十、命令行用法

### 10.1 基本训练

```bash
cd /home/yzyao.25/research/model
python3 run_linear.py -w 1
```

### 10.2 常用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--factor-root` | 因子库_final | 因子根目录 |
| `--label-root` | data/label | 标签目录 |
| `--label-col` | ex_log_ret_10m | 标签列 |
| `--train-ratio` | 0.60 | 训练集比例 |
| `--valid-ratio` | 0.20 | 验证集比例（test = 剩余） |
| `--max-days` | None | 限制天数（快速试跑） |
| `-w`, `--workers` | min(16, cpu) | 并行加载天数 |
| `--session-start` / `--end` | 931 / 1559 | 交易时段 |
| `--no-cache` / `--refresh-cache` | — | 缓存控制 |
| `--skip-eval` / `--skip-plots` | — | 跳过评估或出图 |

> 当前 CLI **未暴露** `ridge_alphas` 网格；修改需编辑 `config.py` 中 `LinearModelConfig.ridge_alphas`。

### 10.3 tmux 后台全量训练

```bash
cd /home/yzyao.25/research/model
./run_linear_tmux.sh
# 或：WORKERS=2 REFRESH_CACHE=1 ./run_linear_tmux.sh
# 日志：output/linear_full_run.log
# 会话：tmux attach -t linear_train
```

最新全量 run：**workers=1**，总耗时约 **2850 秒**（~47 分钟）。

---

## 十一、与 LGB / LSTM 的对比

| 维度 | Linear Ridge | LGB v2 | LSTM |
|------|--------------|--------|------|
| 模型容量 | 线性 | GBDT 非线性 | RNN 非线性 + 时序 |
| 因子数 | 37（全量） | 21（去冗余） | 因子库 + panel |
| 划分 | 60/20/20（有 test） | 80/20（无 test） | 80/20 |
| 超参选择 | valid Rank IC → alpha | valid RMSE → early stop | valid loss |
| Label 处理 | raw | winsorize（训练） | winsorize |
| 样本权重 | 无 | 截面等权 | 无 |
| 可解释性 | **系数** | gain | 弱 |
| 末段 Rank IC | test **0.046** | valid **0.053** | （见 LSTM 文档） |
| 训练速度 | ~47 min | ~12 min | 更慢 |

**何时用线性模型**：

- 需要 **系数符号/大小** 做因子经济学解释  
- 作为 **强正则化基线**，量化 GBDT 的非线性增量  
- 快速验证新因子入库后的 **线性边际贡献**

**线性模型的局限**：

- 无法捕捉因子间阈值/交互（GBDT 的核心优势）  
- 37 因子共线性下系数不稳定、难单独解读  
- 无因子去冗余、无 label 缩尾、无截面权重（LGB v2 已具备）  
- Valid 选 alpha 导致 valid 指标偏乐观，**以 test 为准**

---

## 十二、已知限制与注意事项

1. **Alpha 选择偏差**：`alpha=1000` 在 valid 上最优，valid Rank IC (6.4%) > test (4.6%) 部分源于此。  
2. **共线性**：冗余因子系数相同或相反，不代表独立经济含义。  
3. **Inner join 样本损失**：与 LGB 相同，37 因子全齐才保留。  
4. **截面标准化按日 per-minute**：与 LGB 相同，valid/test 未严格使用 train-only 统计量。  
5. **无模型二进制文件**：部署需保存 `coefficients.csv` + 预处理 pipeline 代码。  
6. **评估出图跳过 train**：`evaluation_summary.csv` 含 train，但 `figures/train/` 不存在。  
7. **Sharpe 口径**：分钟级 L-S 累计年化，宜用于相对对比。

---

## 十三、复现 checklist

- [ ] Python 环境含 `scikit-learn`, `scipy`, `pandas`, `pyarrow`, `matplotlib`, `tqdm`, `pyyaml`  
- [ ] `因子库_final` 37 因子 + 250 日 parquet 齐全  
- [ ] `data/label` 与 `trade_date.csv` 对齐  
- [ ] 运行：`python3 run_linear.py -w 1` 或 `./run_linear_tmux.sh`  
- [ ] 核对：`alpha=1000`，test RankIC ≈ 0.046，Sharpe ≈ 83  
- [ ] 产物：`output/linear/run_config.json`

---

## 十四、关键源码索引

| 主题 | 文件 | 函数 / 类 |
|------|------|-----------|
| 默认超参 | `model/config.py` | `LinearModelConfig` |
| Ridge 拟合 | `model/linear/trainer.py` | `_fit_ridge`, `_ridge_solve_from_gram`, `train_linear_model` |
| Alpha 搜索 | `model/linear/trainer.py` | `_mean_rank_ic` |
| 预处理 | `model/data.py` | `_apply_cross_section_preprocess`, `split_dates` |
| 评估 | `model/linear/evaluate.py` | `build_evaluation_report` |
| 出图 | `model/linear/plot.py` | `plot_coefficients`, `generate_all_plots` |
| CLI | `model/run_linear.py` | `main` |

---

*本文档与 `output/linear/run_config.json`、`model_meta.csv`、`evaluation_summary.csv`、`coefficients.csv` 同步描述 2026-06 最新全量训练配置与结果。*
