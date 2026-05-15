# HZBPlus ASP 股票参数寻优回测

本项目用于研究一个基于成交额趋势斜率的 A 股单股票择时策略，并提供两套实现：

1. 原始 `rqalpha_plus` 回测脚本，用于单只股票或少量股票对照验证。
2. 新增的本地数据缓存 + `vectorbt` 向量化回测脚本，用于全量中国 A 股批量参数寻优。

当前推荐主流程是：

```text
先用 prepare_cn_stock_data.py 从 RQData 拉取并缓存行情数据
再用 ASP_stk_all_bestpara_vbt.py 直接读取本地缓存进行 vectorbt 参数寻优
```

这样做可以避免每次回测都重复请求 RQData，便于复现实验，也更适合后续上传 Git 管理和长期迭代。

## 文件说明

```text
.
|-- ASP_stk_single.py              # 原始单只股票 rqalpha_plus 回测
|-- ASP_stk_all(1).py              # 原始多股票 rqalpha_plus 批量回测，固定参数
|-- ASP_stk_all_bestpara.py        # 原始多股票 rqalpha_plus 参数寻优
|-- prepare_cn_stock_data.py       # 从 RQData 拉取全市场行情并缓存到本地
|-- ASP_stk_all_bestpara_vbt.py    # 读取本地缓存，用 vectorbt 做全市场参数寻优
|-- data/                          # 本地行情缓存目录，运行 prepare 后生成
`-- results/                       # 回测结果目录，运行 vbt 脚本后生成
```

## 策略逻辑

策略名称暂称为 `ASP`。核心思想是：用成交额 `total_turnover` 的线性回归斜率判断资金活跃度趋势。

对每只股票，在每个交易日计算最近 `N` 日成交额的线性回归斜率：

```text
slope = LINEARREG_SLOPE(total_turnover, timeperiod=N)
```

其中 `N` 就是需要寻优的参数 `LONGPERIOD`。

交易规则：

```text
如果 slope 从负数转为非负数：
    认为成交额趋势转强，产生买入信号

如果 slope 从正数转为负数：
    认为成交额趋势转弱，产生卖出信号

其他情况：
    保持原仓位不变
```

用代码表达就是：

```python
entries = (slope >= 0) & (slope.shift(1) < 0)
exits = (slope < 0) & (slope.shift(1) > 0)
```

原始 `rqalpha_plus` 版本中，买入时使用：

```python
order_target_percent(stock, 0.99)
```

卖出时使用：

```python
order_target_percent(stock, 0)
```

`vectorbt` 版本中，为了避免使用当天成交额生成信号后又在当天成交，默认将信号向后移动 1 个交易日：

```text
signal_lag = 1
```

因此推荐使用 `--price-field open`，即用次日开盘价近似执行信号。

## 环境要求

项目主要依赖：

```text
pandas
numpy
numba
vectorbt
rqdatac
rqalpha_plus
talib
tqdm
```

如果使用 WSL 中的 `rq` conda 环境，可以先激活：

```bash
conda activate rq
```

如果 `vectorbt` 尚未安装：

```bash
pip install vectorbt
```

如果希望额外保存 parquet 文件，建议安装：

```bash
pip install pyarrow
```

没有安装 `pyarrow` 也没关系，`prepare_cn_stock_data.py` 会自动跳过 parquet，只保存 CSV。

## 第一步：准备本地数据缓存

使用 [prepare_cn_stock_data.py](./prepare_cn_stock_data.py) 从 RQData 拉取数据并保存到本地。

默认下载区间：

```text
start_date = 2025-04-01
end_date   = 2026-04-01
as_of_date = 2026-03-01
fields     = open close total_turnover
```

默认运行：

```bash
python prepare_cn_stock_data.py
```

脚本会完成以下工作：

1. 调用 `rqdatac.all_instruments(type="CS", market="cn")` 获取中国 A 股股票列表。
2. 过滤上市满一年且状态为 `Active` 的股票。
3. 拉取指定日期区间的 `open`、`close`、`total_turnover`。
4. 将不同字段分别保存为矩阵 CSV。
5. 保存一份 `cn_stock_cache_meta.json` 记录数据来源、日期、字段和文件路径。

生成文件示例：

```text
data/
├── cn_stock_universe_20260301.csv
├── cn_stock_open_20250401_20260401.csv
├── cn_stock_close_20250401_20260401.csv
├── cn_stock_total_turnover_20250401_20260401.csv
└── cn_stock_cache_meta.json
```

矩阵 CSV 的格式是：

```text
date,000001.XSHE,000002.XSHE,600000.XSHG,...
2025-04-01,12.34,8.76,10.01,...
2025-04-02,12.50,8.80,10.12,...
```

### 下载 2020 年至今的数据

如果要下载 2020 年到当前日期的数据，例如到 `2026-04-29`：

```bash
python prepare_cn_stock_data.py \
  --start-date 2020-01-01 \
  --end-date 2026-04-29 \
  --as-of-date 2026-04-29 \
  --chunk-size 500 \
  --no-parquet
```

参数说明：

```text
--start-date     行情开始日期
--end-date       行情结束日期
--as-of-date     股票池截面日期，用于判断上市满一年和 Active 状态
--output-dir     数据保存目录，默认 data
--fields         需要下载的字段，默认 open close total_turnover
--chunk-size     每次向 RQData 请求的股票数量
--no-parquet     只保存 CSV，不尝试保存 parquet
```

如果不加 `--no-parquet`，但环境中没有 `pyarrow` 或 `fastparquet`，可能看到类似提示：

```text
Skipped parquet ... Unable to find a usable engine
```

这不是致命错误。只要看到 CSV 和 `cn_stock_cache_meta.json` 已写出，就可以继续后续回测。

## 第二步：使用 vectorbt 参数寻优

使用 [ASP_stk_all_bestpara_vbt.py](./ASP_stk_all_bestpara_vbt.py) 读取本地缓存，并对每只股票寻找最优 `LONGPERIOD`。

默认寻优范围：

```text
period = 5 ~ 35
```

默认运行：

```bash
python ASP_stk_all_bestpara_vbt.py
```

推荐使用次日开盘价执行信号：

```bash
python ASP_stk_all_bestpara_vbt.py \
  --price-field open \
  --batch-size 500 \
  --workers 4 \
  --save-param-matrix
```

### 使用 2020 年至今的数据做寻优

如果已经用 prepare 下载了 `2020-01-01` 到 `2026-04-29` 的缓存，则回测命令也要指定同样的日期：

```bash
python ASP_stk_all_bestpara_vbt.py \
  --start-date 2020-01-01 \
  --end-date 2026-04-29 \
  --price-field open \
  --period-start 5 \
  --period-end 120 \
  --batch-size 200 \
  --workers 4 \
  --save-param-matrix
```

参数说明：

```text
--data-dir            本地数据缓存目录，默认 data
--results-dir         回测结果输出目录，默认 results
--start-date          读取缓存的开始日期，必须和缓存文件名一致
--end-date            读取缓存的结束日期，必须和缓存文件名一致
--period-start        参数寻优起点
--period-end          参数寻优终点
--batch-size          每个 batch 处理的股票数量
--workers             并行处理 batch 的进程数量
--price-field         使用 open 或 close 作为 vectorbt 回测价格
--init-cash           初始资金，默认 1000000
--fees                手续费率，默认 0.000025
--slippage            滑点，默认 0
--signal-lag          信号延迟执行天数，默认 1
--save-param-matrix   是否保存每只股票每个 period 的收益矩阵
```

## vectorbt 版本的加速方式

`ASP_stk_all_bestpara_vbt.py` 做了两层加速。

第一层是按股票分批：

```text
全市场股票 -> 多个 batch -> 每个 batch 独立回测并输出 part 文件
```

例如：

```text
results/
├── asp_vbt_bestpara_part_001_20260429.csv
├── asp_vbt_bestpara_part_002_20260429.csv
└── asp_vbt_bestpara_all_20260429.csv
```

第二层是对 period 参数进行向量化处理。

脚本没有在 Python 层对每个 period 逐个创建回测，而是使用 `vectorbt.IndicatorFactory` 一次性生成所有 period 的斜率和信号矩阵，然后一次性构建 `vbt.Portfolio.from_signals`。

最终得到的收益矩阵结构类似：

```text
order_book_id,5,6,7,8,...,35
000001.XSHE,0.12,0.10,0.08,...,0.15
000002.XSHE,-0.03,0.01,0.05,...,0.02
```

每只股票会选择 `total_return` 最大的 period 作为最优参数。

## 回测结果说明

主结果文件：

```text
results/asp_vbt_bestpara_all_YYYYMMDD.csv
```

核心字段：

```text
order_book_id              股票代码
symbol                     股票名称
best_longperiod            最优成交额斜率窗口
total_return               最优参数下策略总收益
benchmark_total_return     该股票买入持有收益
excess_return              策略收益 - 买入持有收益
sharpe                     夏普比率
max_drawdown               最大回撤
total_trades               交易次数
```

如果运行时添加 `--save-param-matrix`，还会输出：

```text
results/asp_vbt_param_return_matrix_YYYYMMDD.csv
```

这个文件保存每只股票在每个 period 下的收益，适合后续分析参数稳定性，而不是只看最优值。

## 注意事项

1. `vectorbt` 版本默认使用 `signal_lag=1`，避免未来函数。
2. 如果使用 `--price-field open`，可以理解为用次日开盘价执行信号。
3. 如果使用 `--price-field close`，可以理解为用次日收盘价执行信号。
4. 参数寻优结果属于样本内最优，容易过拟合。后续建议拆分训练期和测试期。
5. 当前版本没有精确模拟涨跌停、停牌无法成交等微观交易约束。
6. 全市场长周期数据较大，建议从小 batch、小 workers 试跑，再逐步放大。
7. 如果机器内存不足，优先降低 `--batch-size`，其次降低 `--workers`。

## 建议运行流程

短周期测试：

```bash
python prepare_cn_stock_data.py --no-parquet

python ASP_stk_all_bestpara_vbt.py \
  --price-field open \
  --period-start 5 \
  --period-end 35 \
  --batch-size 100 \
  --workers 1
```

全市场正式回测：

```bash
python prepare_cn_stock_data.py \
  --start-date 2020-01-01 \
  --end-date 2026-04-29 \
  --as-of-date 2026-04-29 \
  --chunk-size 500 \
  --no-parquet

python ASP_stk_all_bestpara_vbt.py \
  --start-date 2020-01-01 \
  --end-date 2026-04-29 \
  --price-field open \
  --period-start 5 \
  --period-end 120 \
  --batch-size 200 \
  --workers 4 \
  --save-param-matrix
```

## 后续可扩展方向

可以继续扩展以下功能：

1. 增加训练期和测试期拆分，降低样本内过拟合。
2. 增加更多参数维度，例如信号延迟、斜率阈值、价格趋势过滤。
3. 加入停牌、涨跌停、ST 股票等交易约束。
4. 增加行业、市值、中证指数成分股等分组统计。
5. 将 CSV 缓存升级为 parquet，提高读取速度并降低磁盘占用。
