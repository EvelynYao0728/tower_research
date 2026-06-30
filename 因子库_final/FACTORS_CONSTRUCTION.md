# 因子库_final · 因子构造与经济学含义

**整理日期**：2026-06-27  
**适用范围**：本文件覆盖 [`因子库_final`](.) 已入库的 **28** 个因子（因子 parquet 由 `compute/run.py` 生成；`output/` 为回测结果）。  
**数据格式**：长表 parquet，列 `date, sym_root, sym_suffix, minute, <因子名>`  
**预测目标（回测 label）**：未来 10 分钟 mid-price 对数超额收益 `ex_log_ret_10m`

本文档聚焦各因子的**计算公式、构造思路、经济学含义**，不含回测 RankIC/Sharpe 排名。  
**计算实现**：[`compute/factors.py`](./compute/factors.py)（28 因子统一入口）

---

## 一、命名体系（本库实际族别）

| 族别 | 前缀 / 命名 | 含义 | 本库数量 |
|------|-------------|------|----------|
| **B** | `B4_`, `B5_` | 成交流进阶（纳秒并发成交） | 2 |
| **C** | `C3_`, `C5_`, `C8_` | 报价结构（OIR、层叠、价差波动） | 3 |
| **D** | `D3_`, `D5_` | NBBO 动态（全国最优价竞争） | 2 |
| **E** | `E6_` | 微观结构（trade × quote 联合） | 1 |
| **S** | `imbalance_mean` | 基础报价深度统计 | 1 |
| **MB** | `clv`, `ret_1m` 等 | 分钟指标衍生（basic_feature_1） | 14 |
| **O** | `odd_lot_ofi` 等 | 正交 TAQ 因子 | 5 |

> 目录名 = parquet 列名 = 回测 `factor` 名。

---

## 二、因子索引

| 因子名 | 族 | 中文名 |
|--------|-----|--------|
| `B4_cross_dup_price_spread_bps` | B | 并发 VWAP vs 中间价偏差 |
| `B5_trade_dup_vwap_premium` | B | 并发 VWAP 溢价 |
| `C3_oir` | C | 挂单量不平衡 (OIR) |
| `C5_layer_stacking_x_flow_signed_1m` | C | 报价层叠 × 方向 |
| `C8_spread_asymm_vol` | C | 报价端非对称波动率 |
| `D3_nbbo_size_imb` | D | NBBO 量不对称 |
| `D5_spread_diff_mean` | D | 本地 vs NBBO 价差差 |
| `E6_trade_position_skew` | E | 成交价位置偏斜 |
| `imbalance_mean` | S | 分钟深度不平衡均值 |
| `ret_1m` | MB | 分钟 bar 收益 |
| `ret_5m` | MB | 5 分钟价格动量 |
| `ret_10m_past` | MB | 10 分钟历史动量 |
| `clv` | MB | Close Location Value |
| `range_pos_5m` | MB | 5 分钟区间位置 |
| `dist_from_5m_high` | MB | 距 5 分钟高点距离 |
| `dist_from_5m_low` | MB | 距 5 分钟低点距离 |
| `open_mean_dev` | MB | 开盘价相对均值偏离 |
| `imb_current` | MB | 当前分钟深度不平衡 |
| `imb_trend` | MB | 不平衡动量 |
| `spread_x_imb` | MB | 价差 × 不平衡 |
| `liq_adj_ret` | MB | 流动性调整收益 |
| `vol_adj_imb` | MB | 波动率调整不平衡 |
| `clv_x_imb` | MB | CLV × 不平衡交叉 |
| `odd_lot_ofi` | O | 小单订单流不平衡 |
| `mid_trade_share` | O | 中间价成交笔数占比 |
| `mid_trade_vol_share` | O | 中间价成交量占比 |
| `signed_jump_var` | O | 有方向跳跃方差 |
| `realized_skew_tick` | O | tick 级实现偏度 |

---

## 三、B 族 · 成交流进阶

> **数据来源**：NYSE Daily TAQ **Trades**  
> **计算入口**：[`compute/factors.py`](./compute/factors.py) · `compute_bcde_factors`

### B4 · `B4_cross_dup_price_spread_bps`

**中文名**：同纳秒并发成交 vs 中间价偏差

**公式**

\[
\text{B4} = \frac{\text{dup\_vwap} - \text{mid\_quote}}{\text{mid\_price}} \times 10{,}000 \quad \text{(bps)}
\]

**构造思路**

1. 识别同纳秒并发组：`(sym, time_m, time_m_nano)` 完全相同的多笔成交。
2. 组内 VWAP：\(\text{dup\_vwap} = \sum(p_i s_i) / \sum s_i\)。
3. 取同时刻 NBBO 中间价 \(\text{mid\_quote} = (\text{best\_bid}+\text{best\_ask})/2\)。
4. 仅在并发组内计算偏差，分钟聚合（均值）。

**经济学含义**

- 同一纳秒内多笔成交反映算法在同一信号触发时刻的快速响应（HFT 同步反应）。
- **并发 VWAP 高于中间价** → 买方密集主动成交 → 知情交易者急于建仓 → 短期看涨。
- **低于中间价** → 恐慌/被动卖出主导 → 短期看跌。
- 捕捉纳秒级「算法同步反应」中的信息性质，而非普通 tick 噪声。

---

### B5 · `B5_trade_dup_vwap_premium`

**中文名**：同纳秒并发 VWAP 溢价

**公式**

\[
\text{B5} = \frac{\text{dup\_vwap}}{\text{trade\_vwap\_1m}}
\]

其中 `trade_vwap_1m` 为该分钟全部成交的 VWAP。

**构造思路**

与 B4 共用 `dup_vwap` 基础；再除以整分钟 VWAP 得到相对溢价。

**经济学含义**

- 溢价 **> 1** → 买方愿意以高于分钟均价成交 → 私有信息驱动的抢筹，紧迫性高。
- 与 B4 同源、相关性较高：B4 是绝对 bps 偏差，B5 是相对溢价；组合使用时建议检验正交性。

---

## 四、C 族 · 报价结构

> **数据来源**：TAQ **Quotes**（`bid/ask/bidsiz/asksiz` 等）

### C3 · `C3_oir`

**中文名**：挂单量不平衡 (Order Imbalance Ratio)

**公式**

\[
\text{OIR} = \frac{\text{bidsiz} - \text{asksiz}}{\text{bidsiz} + \text{asksiz}} \in [-1, 1]
\]

**构造思路**

对单所报价或分钟聚合后的 bid/ask 挂单量计算不平衡度；本库实现可基于 NBBO 或本地 quote 聚合。

**经济学含义**

- 最直接的**报价端供需信号**：买方挂单量显著多于卖方 → 买压大 → 价格上涨预期强。
- 单所 `bidsiz/asksiz` 仅为该交易所手数；全国代表性见 D3。

---

### C5 · `C5_layer_stacking_x_flow_signed_1m`

**中文名**：报价层叠 × 方向

**公式**

\[
\text{C5} = \text{layer\_stacking\_1m} \times \text{sign}(\text{quote\_flow})
\]

\[
\text{layer\_stacking\_1m} = \text{mean}(\mathbb{1}[\Delta\text{bidsiz}>0 \land \text{ask 价格/量稳定}])
\]

**构造思路**

1. 逐条 quote 计算 bid 量变化 \(\Delta\text{bidsiz}\)。
2. **Layer 条件**：bid 量增加，且 ask 侧价格与量完全不变。
3. 分钟内 layer 事件占比 × 报价斜率/订单流方向符号。

**经济学含义**

- 买方**悄悄堆积挂单而不推价** → 机构隐形建仓的典型微观特征。
- ask 侧稳定确保不是双向波动，而是单纯买方积累。
- 属于「结构强度 × 方向」交叉因子范式。

---

### C8 · `C8_spread_asymm_vol`

**中文名**：报价端非对称波动率

**公式**

\[
\text{C8} = \text{std}(\Delta\text{bid}) - \text{std}(\Delta\text{ask}) \quad \text{（分钟内）}
\]

**构造思路**

分别统计分钟内 best_bid 与 best_ask 价格变化的波动率，取差值。

**经济学含义**

- **bid 波动远高于 ask** → 买方在快速调价 → 可能有买方知情交易推动。
- 两侧对称 → 正常做市；不对称 → **信息不对称**信号。
- 无方向符号，需与其他方向因子配合使用。

---

## 五、D 族 · NBBO 动态

> **数据来源**：TAQ Quotes（`best_bid/ask`, `best_bidsiz/asksiz`, `natbbo_ind` 等）

### D3 · `D3_nbbo_size_imb`

**中文名**：NBBO 量不对称

**公式**

\[
\text{D3} = \frac{\text{best\_bidsiz} - \text{best\_asksiz}}{\text{best\_bidsiz} + \text{best\_asksiz}}
\]

**构造思路**

在全国最优买卖价（NBBO）对应的挂单量上计算 OIR。

**经济学含义**

- 与 C3 类似，但使用**设定 NBBO 的那一侧交易所**的最优量，全国代表性更强。
- 非全市场挂单汇总，而是 SIP 广播的最优价深度。

---

### D5 · `D5_spread_diff_mean`

**中文名**：本地 vs NBBO 价差差

**公式**

\[
\text{D5} = \text{mean}\big((\text{ask}-\text{bid}) - (\text{best\_ask}-\text{best\_bid})\big)
\]

**构造思路**

逐条 quote 计算本交易所价差与 NBBO 价差之差，分钟取均值。

**经济学含义**

- **本地价差 > NBBO 价差** → 存在跨所套利空间 → 可能吸引 ISO 扫单。
- 差值越大，本所流动性越差，成交更可能来自被动方。
- 使用时需注意符号与经济直觉的对应关系。

---

## 六、E 族 · 微观结构

> **数据来源**：Trades × Quotes 时间对齐

### E6 · `E6_trade_position_skew`

**中文名**：成交价位置偏斜 (Trade Position Skew)

**公式**

\[
\text{position} = \frac{\text{price} - \text{best\_bid}}{\text{best\_ask} - \text{best\_bid}} \in [0,1]
\]

\[
\text{E6} = \text{mean}(\text{position}) - 0.5
\]

**构造思路**

对每笔成交，用同时刻 NBBO 计算其在 bid-ask 区间中的相对位置；分钟均值减 0.5 中性化。

**经济学含义**

- 成交价系统性偏向 **ask 侧**（→ 0.5）→ 买方主动推高价格。
- 偏向 **bid 侧** → 卖方主动压价。
- 比 Tick Rule 更连续：不只判断方向，还度量「推力大小」。

---

## 七、S 族 · 基础报价统计

> **数据来源**：`data/simple_factors` 或 TAQ quote 分钟聚合

### S1 · `imbalance_mean`

**中文名**：分钟 NBBO 深度不平衡均值

**公式**

\[
\text{imbalance} = \frac{\text{bid\_size} - \text{ask\_size}}{\text{bid\_size} + \text{ask\_size}}
\]

\[
\text{imbalance\_mean} = \text{time-weighted mean}(\text{imbalance})
\]

**构造思路**

对分钟内每条 quote 快照计算买卖深度不平衡，再按报价停留时间加权求均值。

**经济学含义**

- 分钟内报价停留时间加权的平均买卖深度不平衡。
- 持续正不平衡 → 买方深度占优 → 与 C3/D3 同类，但含**时间权重**与 simple_factors 管线特征。

---

## 八、MB 族 · 分钟指标衍生

> **数据来源**：`base_data_process/simple_factors`（由 TAQ quote 分钟聚合）  
> **计算入口**：[`compute/factors.py`](./compute/factors.py) · `compute_mb_factors`

**符号约定**：\(O=\) `mid_first`，\(C=\) `mid_last`，\(H/L=\) `mid_max/min`，\(\bar{m}=\) `mid_mean`，\(\bar{s}=\) `spread_mean`，\(\varepsilon=10^{-8}\)。

---

### MB1 · `ret_1m`

**中文名**：分钟 bar 收益

**公式**

\[
\text{ret\_1m} = \log\frac{C}{O + \varepsilon}
\]

**构造思路**

用分钟首 tick 与末 tick 的中间价计算对数收益。

**经济学含义**

- 分钟内中间价对数收益，经典**短期动量/反转**载体。
- 在 10 分钟预测 horizon 上常表现为短期反转效应。

---

### MB2 · `ret_5m`

**中文名**：5 分钟价格动量

**公式**

\[
\text{ret\_5m} = \log\frac{C[t]}{C[t-5] + \varepsilon}
\]

**构造思路**

当前分钟收盘价相对 5 分钟前收盘价的对数收益。

**经济学含义**

- 5 分钟 lookback 的中期分钟动量。
- 在 10 分钟 horizon 上，5 分钟涨跌幅往往有均值回归倾向。

---

### MB3 · `ret_10m_past`

**中文名**：10 分钟历史动量

**公式**

\[
\text{ret\_10m\_past} = \log\frac{C[t]}{C[t-10] + \varepsilon}
\]

**构造思路**

lookback 与预测 horizon（10 分钟）对齐的历史动量。

**经济学含义**

- 过去 10 分钟涨得越多，未来 10 分钟越可能回调（反转效应）。

---

### MB7 · `clv`

**中文名**：Close Location Value（收盘位置）

**公式**

\[
\text{clv} = \frac{C - L}{H - L + \varepsilon}
\]

**构造思路**

分钟收盘价在分钟最高/最低价区间中的相对位置。

**经济学含义**

- 接近 **1** → 分钟收在高位 → 买方控制该分钟价格路径。
- 接近 **0** → 卖方控制。

---

### MB11 · `range_pos_5m`

**中文名**：5 分钟区间位置

**公式**

\[
\text{range\_pos\_5m} = \frac{C[t] - \min(L, t-4:t)}{\max(H, t-4:t) - \min(L, t-4:t) + \varepsilon}
\]

**构造思路**

当前价格在过去 5 分钟价格区间中的归一化位置（0=区间底，1=区间顶）。

**经济学含义**

- 处于区间高位时未来收益往往偏低（反转）。

---

### MB12 · `dist_from_5m_high`

**中文名**：距 5 分钟高点距离

**公式**

\[
\text{dist\_from\_5m\_high} = \frac{\max(H, t-4:t) - C[t]}{\bar{m} + \varepsilon}
\]

**构造思路**

当前价距过去 5 分钟高点的相对回落幅度（以分钟均价归一化）。

**经济学含义**

- 度量价格从近期高点的回撤程度。
- 可从高点回落后的反弹/延续弱势等角度解读。

---

### MB13 · `dist_from_5m_low`

**中文名**：距 5 分钟低点距离

**公式**

\[
\text{dist\_from\_5m\_low} = \frac{C[t] - \min(L, t-4:t)}{\bar{m} + \varepsilon}
\]

**构造思路**

当前价距过去 5 分钟低点的相对距离（以分钟均价归一化）。

**经济学含义**

- 远离低点的股票，常与 `range_pos_5m` 一致反映「高位反转」逻辑。

---

### MB15 · `open_mean_dev`

**中文名**：开盘价相对均值偏离

**公式**

\[
\text{open\_mean\_dev} = \frac{O - \bar{m}}{\bar{m} + \varepsilon}
\]

**构造思路**

分钟开盘价相对分钟内 tick 均价的偏离。

**经济学含义**

- 负值通常表示开盘后价格上行（开盘低于均价）。
- 信号较弱但方向相对稳定。

---

### MC1 · `imb_current`

**中文名**：当前分钟深度不平衡

**公式**

\[
\text{imb\_current} = \text{imbalance\_mean}[t]
\]

**构造思路**

直接取当前分钟的 `imbalance_mean` 快照值。

**经济学含义**

- 与 S1 `imbalance_mean` **数值同源**；保留此命名是为了与 `imb_trend` 等衍生因子形成一致前缀。
- 衡量当前截面的买卖深度压力水平。

---

### MC4 · `imb_trend`

**中文名**：不平衡动量

**公式**

\[
\text{imb\_trend} = \text{imb\_current} - \text{mean}(\text{imbalance\_mean}, t-4:t)
\]

**构造思路**

当前买压相对过去 5 分钟均值的边际变化。

**经济学含义**

- 正值 → 买压正在增强 → 短期看涨。
- 与 S1 的水平信号互补：S1 看幅度，本因子看**边际变化**。

---

### MD1 · `spread_x_imb`

**中文名**：价差 × 不平衡

**公式**

\[
\text{spread\_x\_imb} = \bar{s} \times \text{imb\_current}
\]

**构造思路**

分钟平均买卖价差与当前深度不平衡的乘积。

**经济学含义**

- 流动性成本（spread）与买卖深度压力的**联合方向信号**。
- 宽价差 × 强不平衡 → 「带成本的买卖意愿」，信号更可靠。

---

### MD2 · `liq_adj_ret`

**中文名**：流动性调整收益

**公式**

\[
\text{liq\_adj\_ret} = \frac{\text{ret\_1m}}{\bar{s}/\bar{m} + \varepsilon}
\]

**构造思路**

将分钟收益除以相对买卖价差，过滤宽价差下的虚假波动。

**经济学含义**

- 经流动性成本调整后的收益，更能反映「真实」价格变动。
- 宽价差环境下的 raw return 噪声被削弱。

---

### MD7 · `vol_adj_imb`

**中文名**：波动率调整不平衡

**公式**

\[
\text{vol\_adj\_imb} = \frac{\text{imb\_current}}{\text{mid\_std}/\bar{m} + \varepsilon}
\]

**构造思路**

用分钟内 tick 波动率（`mid_std/mid_mean`）归一化不平衡。

**经济学含义**

- 高波动期的不平衡信号被折扣 → 提高**信噪比**。
- 与 `spread_x_imb` 类似，属于对 raw imbalance 的 regime 调整。

---

### MD8 · `clv_x_imb`

**中文名**：CLV × 不平衡交叉

**公式**

\[
\text{clv\_x\_imb} = \text{clv} \times \text{imb\_current}
\]

**构造思路**

价格路径位置（CLV）与深度压力的显式交叉项。

**经济学含义**

- 当 CLV 与 imbalance 同向时信号更强（例如收在高位且买压大）。
- 树模型可自动学习交互，线性模型需显式构造此类交叉。

---

## 九、O 族 · 正交 TAQ 因子

> **数据来源**：NYSE Daily TAQ Quotes + Trades  
> **计算入口**：[`compute/factors.py`](./compute/factors.py) · `compute_o_factors`

**符号约定**：\(r_t = \log(m_t/m_{t-1})\) 为 tick 级 mid 对数收益；Lee-Ready 方向 \(q_i \in \{+1,-1\}\)；\(\varepsilon = 10^{-9}\)。

---

### O3 · `signed_jump_var`

**中文名**：有方向跳跃方差

**公式**

\[
RV^+ = \sum_t r_t^2 \mathbb{1}[r_t>0], \quad RV^- = \sum_t r_t^2 \mathbb{1}[r_t<0]
\]

\[
\text{signed\_jump\_var} = RV^+ - RV^-
\]

**构造思路**

将 tick 级 mid 对数收益按正负分拆，分别累加平方和，再取差。

**经济学含义**

- 捕捉收益路径的**二阶矩方向**：可出现 ret_1m ≈ 0 但 signed_jump_var 极大（剧烈不对称震荡）。
- 与报价不平衡偏度（S3 类）机制完全不同。
- 分钟级常呈短期反转效应。

**文献**：Patton & Sheppard (2015); Barndorff-Nielsen et al. (2010)

---

### O4 · `odd_lot_ofi`

**中文名**：小单订单流不平衡

**公式**

odd-lot 定义：size < 100 股。

\[
\text{odd\_lot\_ofi} = \frac{V^{odd}_{buy} - V^{odd}_{sell}}{V^{odd}_{buy} + V^{odd}_{sell} + \varepsilon}
\]

**构造思路**

仅对小于 100 股的小单成交，按 Lee-Ready 方向划分买卖量，计算订单流不平衡。

**经济学含义**

- 常规 OFI 由大单主导；odd_lot_ofi 捕捉**散户/碎片化算法**方向。
- 与 institutional flow 因子形成互补。

**文献**：O'Hara, Yao, Ye (2014, J. Finance)

---

### O6 · `mid_trade_share` / `mid_trade_vol_share`

**中文名**：恰在 mid 成交的笔数/成交量占比

**公式**

\[
\text{mid\_trade}_i = \mathbb{1}\left[|p_i - m_i| < \tfrac{1}{2}\cdot\text{tick}\right]
\]

\[
\text{mid\_trade\_share} = \frac{N_{mid}}{N_{total}}, \quad
\text{mid\_trade\_vol\_share} = \frac{V_{mid}}{V_{total}}
\]

**构造思路**

识别成交价落在 mid 附近（半个 tick 以内）的成交，分别统计笔数占比与成交量占比。

**经济学含义**

- 中间价成交 ≈ midpoint peg / 隐藏单 → **未展示**的真实流动性需求。
- 与 E6（连续位置 0~1）互补：E6 度量连续位置，本因子专门计数「恰在中间」事件。
- 单独方向预测力弱，适合作为 regime / 交叉特征。

**文献**：Buti, Rindi, Werner (2017, JFE)

---

### O10 · `realized_skew_tick`

**中文名**：tick 级实现偏度

**公式**

\[
\text{RS} = \frac{\sqrt{N}\sum_t r_t^3}{(RV^+ + RV^-)^{3/2} + \varepsilon}
\]

**构造思路**

用 tick 级 mid 收益的三阶矩，除以实现波动率的标准化形式。

**经济学含义**

- 收益序列**三阶矩**；与 imbalance 分布偏度完全不同。
- 正偏（右尾厚）后短期收益往往偏低（反转）。

**文献**：Amaya et al. (2015, JFE)

---

## 十、因子关系速查（本库内）

| 因子 A | 因子 B | 关系 |
|--------|--------|------|
| `imb_current` | `imbalance_mean` | 数值同源，命名不同 |
| `clv` | `F5_close_location_value` | 公式相同，管线不同（F5 未入库） |
| `liq_adj_ret` | `F11_liquidity_adjusted_return` | 同类流动性调整收益（F11 未入库） |
| `B4` | `B5` | 同源 dup_vwap，B4 为 bps 偏差，B5 为相对溢价 |
| `C3` | `D3` | 同为 OIR，C3 本所/聚合，D3 为 NBBO 最优量 |
| `mid_trade_share` | `E6` | 离散 mid 成交占比 vs 连续成交位置 |
| `signed_jump_var` | `ret_1m` | 二阶矩方向 vs 一阶动量 |
| `odd_lot_ofi` | `ofi_1m` | 小单流 vs 全量 OFI（ofi_1m 未入库） |

---

## 十一、参考文档

| 文档 | 路径 |
|------|------|
| 因子计算实现 | [`compute/factors.py`](./compute/factors.py) |
| 因子生成 CLI | [`compute/run.py`](./compute/run.py) |
| simple_factors 预处理 | [`base_data_process/compute_simple_factors.py`](../base_data_process/compute_simple_factors.py) |
