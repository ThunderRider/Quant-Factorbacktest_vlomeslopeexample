# Intraday Momentum Strategy Reproduction Spec

This document is a reproducible implementation brief for the SPY intraday
momentum strategy and the later exit-strategy/parameter-optimization variants
described in:

- "Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF
  (SPY)" by Zarattini, Aziz, and Barbon.
- "Improvements to Intraday Momentum Strategies Using Parameter Optimization
  and Different Exit Strategies" by Adam Maroy.

The goal is to give another Codex project enough detail to implement, test, and
extend the strategy without re-reading the papers.

Important caveat: the improved paper heavily optimizes parameters. Treat the
reported parameter sets as reproduction targets first, not as deployable
parameters. A real strategy must be validated with walk-forward tests,
transaction costs, slippage, parameter perturbation, and strict out-of-sample
evaluation.

---

## 1. Strategy Summary

The strategy is an intraday breakout/momentum system.

For each trading day and intraday timestamp, calculate how much the ETF usually
moves away from the daily open at the same intraday time over the previous `L`
days. This rolling same-time move is the "noise area". If today's price breaks
above the upper noise boundary, enter long. If today's price breaks below the
lower noise boundary, enter short.

The improved paper keeps this general entry idea and mainly tests better exit
rules:

- VWAP exit.
- Boundary exit with a separate exit multiplier.
- Boundary plus VWAP trailing exit.
- Ladder stop-loss/take-profit exits.
- VWAP plus Ladder.
- Boundary plus Ladder.

---

## 2. Data Frequency And Time Conventions

### 2.1 SPY / US Market

Use regular trading hours only:

- Timezone: `America/New_York`.
- Regular session: `09:30:00` to `16:00:00`.
- Exclude pre-market and post-market.
- Handle early-close days by using the exchange session close for that date.

Recommended bar frequency:

- Original SPY reproduction: 1-minute bars are enough.
- Improved paper reproduction: use 1-second bars if available, because the paper
  optimized on 1-second data. If only 1-minute data is available, mark results as
  an approximation.

Required intraday columns:

```text
timestamp
symbol
open
high
low
close
volume
vwap_optional
```

Required daily columns:

```text
date
symbol
open
high
low
close
adj_close_optional
volume
dividend_amount_optional
split_factor_optional
```

### 2.2 China Market / CSI 300 ETF

For 510300.SH, 159919.SZ, or another CSI 300 ETF:

- Timezone: `Asia/Shanghai`.
- Trading session: `09:30:00-11:30:00`, `13:00:00-15:00:00`.
- Treat the lunch break as a gap. Do not interpolate or create synthetic returns
  across the break.
- Intraday minute index should be session-relative from 1 to 240:
  - Morning: 120 minutes.
  - Afternoon: 120 minutes.

China-specific trading constraint:

- Domestic equity ETFs are normally not directly same-day round-trip T+0
  instruments.
- For a realistic ETF implementation, use a base holding/overlay structure:
  sell only from previous-day available holdings and do not sell shares bought
  today.
- For pure signal research, a theoretical T+0 long/short backtest is acceptable,
  but label it clearly as non-executable on ordinary ETF cash trading.

---

## 3. SPY Data Request Module

Keep SPY data acquisition separate from the strategy engine. The strategy engine
should consume standardized local files or DataFrames and should not depend on a
specific vendor.

### 3.1 Required Interface

Implement a data module with these functions or equivalent methods:

```python
def fetch_spy_intraday(
    start_date: str,
    end_date: str,
    bar_size: str = "1min",
    adjusted: bool = True,
    regular_hours_only: bool = True,
) -> pd.DataFrame:
    """Return SPY intraday OHLCV bars in America/New_York timezone."""


def fetch_spy_daily(
    start_date: str,
    end_date: str,
    adjusted: bool = True,
) -> pd.DataFrame:
    """Return SPY daily OHLCV plus dividends/splits if available."""


def load_or_fetch_spy_data(
    start_date: str,
    end_date: str,
    bar_size: str = "1min",
    cache_dir: str = "data/spy",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached data if present; otherwise fetch, validate, cache, and return."""
```

### 3.2 Acceptable Data Vendors

Use one of the following, depending on project access:

- Polygon.io aggregates.
- Nasdaq Data Link / historical intraday vendor.
- FirstRate Data or another paid historical minute provider.
- Interactive Brokers historical bars, if the date range is short enough.
- Alpaca or IEX Cloud, if coverage and adjustment handling are sufficient.

Do not mix vendors inside one backtest unless a reconciliation step is added.

### 3.3 Adjustment Rules

For backtesting, use split-adjusted prices. Dividend treatment matters because
SPY has ex-dividend gaps that can distort the boundary calculation.

Recommended handling:

1. Use split-adjusted intraday OHLCV.
2. Use split-adjusted daily OHLCV.
3. If a cash dividend occurs on date `d`, adjust the previous close used by the
   day `d` boundary:

```text
prev_close_for_boundary[d] = previous_daily_close[d] - dividend_amount[d]
```

If the vendor already provides total-return adjusted intraday prices, document
that choice and use the same adjustment consistently for daily and intraday
data.

### 3.4 Data Validation

Before running the strategy:

- Ensure timestamps are timezone-aware.
- Remove duplicate bars.
- Keep only regular-hours bars.
- Confirm each session has the expected number of bars, except early closes.
- Check that `open`, `high`, `low`, `close` are positive.
- Check that `high >= max(open, close)` and `low <= min(open, close)`.
- For missing intraday bars, either:
  - drop the whole day for strict reproduction, or
  - forward-fill prices only for VWAP/state continuity and mark the day as
    approximate.
- Compute intraday VWAP if not provided:

```text
typical_price = (high + low + close) / 3
dollar_volume = typical_price * volume
vwap[d,t] = cumulative_sum(dollar_volume within day up to t)
            / cumulative_sum(volume within day up to t)
```

If volume is zero for a bar, keep the previous valid VWAP for that day.

---

## 4. Core Feature Definitions

Let:

```text
d = trading date
t = intraday bar index or timestamp within date d
L = lookback_days
P[d,t] = intraday close at date d and time t
O[d] = daily/session open at date d
C_prev[d] = previous session close adjusted for dividend if needed
```

### 4.1 Same-Time Absolute Move

For each previous day `d-i` and the same intraday time `t`:

```text
move[d-i,t] = abs(P[d-i,t] / O[d-i] - 1)
```

Then:

```text
sigma_move[d,t] = mean(move[d-1,t], move[d-2,t], ..., move[d-L,t])
```

Rules:

- Use only days strictly before `d`.
- Match the same intraday bar index.
- If fewer than `L` valid prior observations exist, do not trade.
- Do not include current-day information in `sigma_move`.

### 4.2 Entry Boundaries

Use a volatility multiplier `VM_entry`.

```text
upper_entry[d,t] = max(O[d], C_prev[d]) * (1 + VM_entry * sigma_move[d,t])
lower_entry[d,t] = min(O[d], C_prev[d]) * (1 - VM_entry * sigma_move[d,t])
```

The `max(open, previous close)` and `min(open, previous close)` anchors reduce
false signals caused by overnight gaps.

### 4.3 Exit Boundaries

Some variants use a separate exit multiplier `VM_exit`.

```text
upper_exit[d,t] = max(O[d], C_prev[d]) * (1 + VM_exit * sigma_move[d,t])
lower_exit[d,t] = min(O[d], C_prev[d]) * (1 - VM_exit * sigma_move[d,t])
```

Usually `VM_exit < VM_entry`, so the exit boundary is closer to the open.

### 4.4 Daily Volatility For Position Sizing

Compute close-to-close daily returns:

```text
ret_daily[d] = close_daily[d] / close_daily[d-1] - 1
sigma_daily[d] = rolling_std(ret_daily, window=15, using data up to d-1)
```

For day `d`, use only information known before the session starts.

---

## 5. Position Sizing

Original-style volatility targeting:

```text
target_daily_volatility = parameter, for example 0.02 in the original SPY paper
max_leverage = 4
gross_exposure_fraction[d] = min(max_leverage, target_daily_volatility / sigma_daily[d])
notional[d] = equity_before_day[d] * gross_exposure_fraction[d]
shares[d] = floor(notional[d] / execution_price)
```

Rules:

- Recalculate `shares[d]` once per day before the first trade.
- Use the same share count for all intraday entries that day unless explicitly
  implementing dynamic intraday resizing.
- If `sigma_daily[d]` is missing or zero, do not trade.
- For long-only China ETF overlay, cap shares by cash and available holdings.

---

## 6. Entry Logic

### 6.1 Entry Schedule

Entries are only checked on scheduled bars.

Parameters:

```text
start_trade_after_open_minutes
trade_frequency_minutes
exit_trades_before_close_minutes
```

Example:

```text
start_trade_after_open_minutes = 30
trade_frequency_minutes = 30
exit_trades_before_close_minutes = 5
```

Then evaluate entry at 10:00, 10:30, 11:00, and so on, but never inside the final
`exit_trades_before_close_minutes` of the session.

For China, skip the lunch break. If a scheduled time falls inside the break,
move it to the next valid bar or simply exclude it; document the chosen rule.

### 6.2 Signal Conditions

Base breakout signal:

```text
long_signal[d,t] = close[d,t] > upper_entry[d,t]
short_signal[d,t] = close[d,t] < lower_entry[d,t]
```

VWAP-filtered signal, used by the original public SPY reproduction and most
practical variants:

```text
long_signal[d,t] = close[d,t] > upper_entry[d,t] and close[d,t] > vwap[d,t]
short_signal[d,t] = close[d,t] < lower_entry[d,t] and close[d,t] < vwap[d,t]
```

Recommended default:

- Use VWAP filter for SPY reproduction.
- For China ETF research, test both with and without VWAP filter.

### 6.3 Entry State Rules

At each scheduled entry bar:

- If flat and `long_signal`, enter long.
- If flat and `short_signal`, enter short.
- If long and `short_signal`, close long and optionally reverse to short.
- If short and `long_signal`, close short and optionally reverse to long.
- If already in the same direction, keep holding. Do not pyramid unless a
  separate pyramiding rule is explicitly added.

Use a config flag:

```text
allow_reversal = True or False
```

For paper-style long/short reproduction, set `allow_reversal = True`.
For China ETF cash overlay, set `allow_reversal = False` unless the overlay
inventory logic explicitly supports it.

### 6.4 Execution Price

Avoid lookahead. Two acceptable modes:

```text
execution_mode = "signal_bar_close"  # paper-style approximation
execution_mode = "next_bar_open"     # more conservative implementation
```

For exact paper-style reproduction, use signal bar close if the source code did
so. For realistic execution, use next bar open or next bar VWAP with slippage.

---

## 7. Exit Logic Variants

All variants must force-close open positions before session close:

```text
forced_exit_time = session_close - exit_trades_before_close_minutes
```

If no stop/exit rule triggers earlier, close at the first valid bar at or after
`forced_exit_time`.

Unless otherwise stated:

- Entry checks happen only on scheduled entry bars.
- Exit checks happen every valid bar after entry.
- For bar data, trigger boundary/VWAP exits using close price.
- For ladder stops/take-profits, use high/low.

### 7.1 Variant A: Opposite Signal / End Of Day

This is the simplest baseline.

Long position:

- Exit or reverse if a scheduled `short_signal` occurs.
- Otherwise exit at forced exit time.

Short position:

- Exit or reverse if a scheduled `long_signal` occurs.
- Otherwise exit at forced exit time.

### 7.2 Variant B: VWAP Exit

Long position:

```text
exit_long = close[d,t] <= vwap[d,t]
```

Short position:

```text
exit_short = close[d,t] >= vwap[d,t]
```

After exit, remain flat until the next scheduled entry check.

### 7.3 Variant C: Boundary With Different Exit

Uses `VM_exit`.

Long position:

```text
exit_long = close[d,t] <= upper_exit[d,t]
```

Short position:

```text
exit_short = close[d,t] >= lower_exit[d,t]
```

### 7.4 Variant D: Boundary With Different Exit And VWAP

Long position:

```text
exit_line_long = max(upper_exit[d,t], vwap[d,t])
exit_long = close[d,t] <= exit_line_long
```

Short position:

```text
exit_line_short = min(lower_exit[d,t], vwap[d,t])
exit_short = close[d,t] >= exit_line_short
```

### 7.5 Variant E: Boundary And VWAP

This is close to the original public SPY "current band + VWAP" stop.

Long position:

```text
exit_line_long = max(upper_entry[d,t], vwap[d,t])
exit_long = close[d,t] <= exit_line_long
```

Short position:

```text
exit_line_short = min(lower_entry[d,t], vwap[d,t])
exit_short = close[d,t] >= exit_line_short
```

### 7.6 Variant F: Ladder Stop-Loss And Take-Profit

The improved paper reports ladder parameters as price differences from entry.
This is scale-sensitive. If reproducing QQQ from the paper, use the values
directly. If porting to SPY or China ETFs, also test normalized versions based
on price percentage or intraday ATR.

Parameters:

```text
stop_loss_ladder_step_0_diff
stop_loss_ladder_step_1_diff
take_profit_ladder_step_0_diff
take_profit_ladder_step_1_diff
take_profit_fraction_step_0 = 0.50
```

For a long position:

```text
sl0 = entry_price + stop_loss_ladder_step_0_diff
tp0 = entry_price + take_profit_ladder_step_0_diff
sl1 = entry_price + stop_loss_ladder_step_1_diff
tp1 = entry_price + take_profit_ladder_step_1_diff
```

For a short position, multiply the difference by `-1`:

```text
sl0 = entry_price - stop_loss_ladder_step_0_diff
tp0 = entry_price - take_profit_ladder_step_0_diff
sl1 = entry_price - stop_loss_ladder_step_1_diff
tp1 = entry_price - take_profit_ladder_step_1_diff
```

State machine:

1. At entry, all shares are in ladder step 0.
2. If step 0 stop-loss is hit, close 100% of the position.
3. If step 0 take-profit is hit, close 50% of the position and move the
   remaining 50% to step 1.
4. In step 1, if stop-loss is hit, close the remaining position.
5. In step 1, if take-profit is hit, close the remaining position.
6. If no ladder event occurs, close remaining position at forced exit time.

Bar trigger rules:

- Long stop-loss: `low <= sl`.
- Long take-profit: `high >= tp`.
- Short stop-loss: `high >= sl`.
- Short take-profit: `low <= tp`.

If stop-loss and take-profit are both touched in the same bar, use conservative
ordering:

- For long: assume stop-loss first.
- For short: assume stop-loss first.

With 1-second data this ambiguity should be rare. With 1-minute data, log the
number of ambiguous bars.

### 7.7 Variant G: VWAP And Ladder

Combine Variant B and Variant F.

Exit if either:

- VWAP exit triggers, or
- the active ladder stop-loss/take-profit triggers, or
- forced exit time is reached.

If multiple exits trigger in the same bar, use conservative execution for the
current position:

- Long: choose the lowest plausible exit price.
- Short: choose the highest plausible exit price.

### 7.8 Variant H: Boundary And Ladder

Combine Variant E and Variant F.

Exit if either:

- Boundary/VWAP line triggers, or
- the active ladder stop-loss/take-profit triggers, or
- forced exit time is reached.

---

## 8. Reproduction Parameter Sets

### 8.1 Original SPY Baseline

Use this first to verify the engine.

```yaml
symbol: SPY
bar_size: 1min
lookback_days: 14
volatility_multiplier_entry: 1.0
entry_vwap_filter: true
target_daily_volatility: 0.02
daily_vol_window: 15
max_leverage: 4.0
start_trade_after_open_minutes: 30
trade_frequency_minutes: 30
exit_trades_before_close_minutes: 5
exit_variant: boundary_and_vwap
allow_reversal: true
execution_mode: signal_bar_close
transaction_cost_bps_one_way: 0.0
slippage_bps_one_way: 0.0
```

Also run a more realistic version:

```yaml
execution_mode: next_bar_open
transaction_cost_bps_one_way: 0.5
slippage_bps_one_way: 0.5
```

### 8.2 Improved Paper Candidate Sets

The improved paper optimizes over 2014-10-01 to 2024-10-01 and mainly reports
best sets for QQQ. Use these as candidate reproduction configs, then re-optimize
inside your own walk-forward framework.

```yaml
vwap_1:
  symbol: QQQ
  exit_variant: vwap
  lookback_days: 2
  volatility_multiplier_entry: 1.03
  target_daily_volatility: 0.013
  start_trade_after_open_minutes: 12
  trade_frequency_minutes: 45
  exit_trades_before_close_minutes: 13

vwap_2:
  symbol: QQQ
  exit_variant: vwap
  lookback_days: 2
  volatility_multiplier_entry: 0.85
  target_daily_volatility: 0.014
  start_trade_after_open_minutes: 12
  trade_frequency_minutes: 45
  exit_trades_before_close_minutes: 13

boundary_different_exit_1:
  symbol: QQQ
  exit_variant: boundary_different_exit
  lookback_days: 5
  volatility_multiplier_entry: 1.29
  volatility_multiplier_exit: 0.78
  target_daily_volatility: 0.016
  start_trade_after_open_minutes: 7
  trade_frequency_minutes: 35
  exit_trades_before_close_minutes: 16

boundary_different_exit_and_vwap_1:
  symbol: QQQ
  exit_variant: boundary_different_exit_and_vwap
  lookback_days: 4
  volatility_multiplier_entry: 1.14
  volatility_multiplier_exit: 0.35
  target_daily_volatility: 0.030
  start_trade_after_open_minutes: 10
  trade_frequency_minutes: 50
  exit_trades_before_close_minutes: 31

ladder_1:
  symbol: QQQ
  exit_variant: ladder
  lookback_days: 4
  volatility_multiplier_entry: 1.28
  target_daily_volatility: 0.064
  start_trade_after_open_minutes: 42
  trade_frequency_minutes: 23
  exit_trades_before_close_minutes: 30
  stop_loss_ladder_step_0_diff: -0.40
  stop_loss_ladder_step_1_diff: 0.16
  take_profit_ladder_step_0_diff: 2.55
  take_profit_ladder_step_1_diff: 3.68

ladder_2:
  symbol: QQQ
  exit_variant: ladder
  lookback_days: 4
  volatility_multiplier_entry: 1.08
  target_daily_volatility: 0.054
  start_trade_after_open_minutes: 42
  trade_frequency_minutes: 23
  exit_trades_before_close_minutes: 30
  stop_loss_ladder_step_0_diff: -0.40
  stop_loss_ladder_step_1_diff: -0.27
  take_profit_ladder_step_0_diff: 2.12
  take_profit_ladder_step_1_diff: 20.32

vwap_and_ladder_1:
  symbol: QQQ
  exit_variant: vwap_and_ladder
  lookback_days: 4
  volatility_multiplier_entry: 1.33
  target_daily_volatility: 0.025
  start_trade_after_open_minutes: 42
  trade_frequency_minutes: 46
  exit_trades_before_close_minutes: 30
  stop_loss_ladder_step_0_diff: -0.41
  stop_loss_ladder_step_1_diff: -0.28
  take_profit_ladder_step_0_diff: 2.20
  take_profit_ladder_step_1_diff: 37.08

vwap_and_ladder_2:
  symbol: QQQ
  exit_variant: vwap_and_ladder
  lookback_days: 4
  volatility_multiplier_entry: 1.33
  target_daily_volatility: 0.025
  start_trade_after_open_minutes: 42
  trade_frequency_minutes: 46
  exit_trades_before_close_minutes: 30
  stop_loss_ladder_step_0_diff: -0.65
  stop_loss_ladder_step_1_diff: 2.87
  take_profit_ladder_step_0_diff: 11.72
  take_profit_ladder_step_1_diff: 21.64

boundary_and_ladder_1:
  symbol: QQQ
  exit_variant: boundary_and_ladder
  lookback_days: 4
  volatility_multiplier_entry: 1.33
  target_daily_volatility: 0.025
  start_trade_after_open_minutes: 42
  trade_frequency_minutes: 46
  exit_trades_before_close_minutes: 30
  stop_loss_ladder_step_0_diff: -0.54
  stop_loss_ladder_step_1_diff: 1.73
  take_profit_ladder_step_0_diff: 3.22
  take_profit_ladder_step_1_diff: 4.35

boundary_and_ladder_2:
  symbol: QQQ
  exit_variant: boundary_and_ladder
  lookback_days: 4
  volatility_multiplier_entry: 1.33
  target_daily_volatility: 0.025
  start_trade_after_open_minutes: 42
  trade_frequency_minutes: 46
  exit_trades_before_close_minutes: 30
  stop_loss_ladder_step_0_diff: -0.54
  stop_loss_ladder_step_1_diff: 2.47
  take_profit_ladder_step_0_diff: 3.96
  take_profit_ladder_step_1_diff: 5.09
```

If exact full-paper tables are available in the target project, verify these
values against the PDF before using them as final reproduction constants.

---

## 9. Backtest Accounting

### 9.1 Trade PnL

For each fill:

```text
gross_pnl = direction * shares * (exit_price - entry_price)
cost = shares * entry_price * cost_rate + shares * exit_price * cost_rate
net_pnl = gross_pnl - cost
```

Where:

```text
direction = +1 for long
direction = -1 for short
```

For partial exits, track each lot separately or keep an average entry price and
remaining shares.

### 9.2 Equity Curve

Start with a fixed initial equity, for example:

```text
initial_equity = 100000
```

Update equity after every closed trade. For intraday mark-to-market equity, use
current close and open position.

### 9.3 Metrics

Report:

- Total return.
- CAGR.
- Annualized volatility.
- Sharpe ratio.
- Sortino ratio.
- Max drawdown.
- Calmar ratio.
- Win rate.
- Average win/loss.
- Profit factor.
- Average daily turnover.
- Number of trades.
- Long trades and short trades separately.
- Exposure time.
- Return by year.
- Return by month.
- Slippage/cost sensitivity.

Annualization:

```text
daily_sharpe = mean(daily_returns) / std(daily_returns) * sqrt(252)
```

Use daily returns from end-of-day equity.

---

## 10. No-Lookahead Checklist

The implementation must satisfy:

- `sigma_move[d,t]` uses only days before `d`.
- `sigma_daily[d]` uses only daily returns before `d`.
- `vwap[d,t]` uses only bars up to and including `t`.
- Entry uses only completed bar data.
- If using `next_bar_open`, fills happen after the signal bar.
- Corporate action adjustments are known at the appropriate date.
- Forced exits use the actual session close for early-close days.
- Parameter optimization never uses test-period performance.

---

## 11. China CSI 300 ETF Adaptation

Recommended research instruments:

- 510300.SH: Huatai-PineBridge CSI 300 ETF.
- 159919.SZ: Harvest CSI 300 ETF.
- Optional comparison: 510330.SH, 510310.SH, 515330.SH.

### 11.1 Theoretical Signal Research Version

Purpose: test whether the intraday boundary signal has predictive value.

Config:

```yaml
symbol: 510300.SH
bar_size: 1min
market: China
session_segments:
  - ["09:30", "11:30"]
  - ["13:00", "15:00"]
lookback_days_grid: [4, 8, 14, 20]
volatility_multiplier_entry_grid: [0.8, 1.0, 1.2, 1.5]
volatility_multiplier_exit_grid: [0.3, 0.5, 0.8, 1.0]
start_trade_after_open_minutes_grid: [10, 15, 30, 45]
trade_frequency_minutes_grid: [5, 10, 15, 30]
exit_trades_before_close_minutes_grid: [5, 10, 15]
exit_variants:
  - vwap
  - boundary_and_vwap
  - boundary_different_exit_and_vwap
  - ladder
  - vwap_and_ladder
entry_vwap_filter: true
execution_mode: next_bar_open
transaction_cost_bps_one_way: 1.0
slippage_bps_one_way: 1.0
```

Run both:

- Long-only.
- Theoretical long-short.

The long-short version is for alpha diagnosis, not direct ETF cash execution.

### 11.2 Tradable ETF Base-Holding Overlay Version

Purpose: approximate executable intraday "do T" trading with a previous-day ETF
inventory.

State variables:

```text
base_position_shares
available_to_sell_today
cash
overlay_position
shares_bought_today
```

Rules:

- At day start, `available_to_sell_today = shares_held_from_previous_day`.
- Long overlay:
  - Buy extra shares when long signal triggers.
  - Exit by selling from `available_to_sell_today`.
  - Do not sell shares bought today unless the market/instrument supports it.
- Short overlay:
  - Sell available base shares first when short signal triggers.
  - Exit by buying back shares later.
  - This is not true shorting; it is reducing base exposure intraday.
- Cap every sell by `available_to_sell_today`.
- Cap every buy by available cash.
- At end of day, update tomorrow's available shares according to exchange
  settlement rules.

Evaluate overlay alpha separately:

```text
total_etf_pnl = base_holding_pnl + overlay_pnl
strategy_alpha = overlay_pnl
```

This separation is important. Otherwise the beta exposure of the base CSI 300
ETF holding can hide whether the intraday signal actually works.

### 11.3 China Parameter Normalization

Do not directly reuse QQQ ladder dollar values on 510300.SH. The price scale is
different.

Test normalized ladder distances:

```text
diff_price = entry_price * diff_pct
```

or:

```text
diff_price = intraday_ATR_lookback * multiplier
```

Suggested initial ladder percentage grid:

```yaml
stop_loss_ladder_step_0_pct_grid: [-0.0010, -0.0015, -0.0020, -0.0030]
stop_loss_ladder_step_1_pct_grid: [-0.0005, 0.0000, 0.0005, 0.0010]
take_profit_ladder_step_0_pct_grid: [0.0010, 0.0015, 0.0020, 0.0030, 0.0040]
take_profit_ladder_step_1_pct_grid: [0.0020, 0.0030, 0.0050, 0.0080, 0.0100]
```

---

## 12. Optimization Framework

Use walk-forward optimization rather than one full-sample search.

Recommended split:

```text
train_window = 2 or 3 years
validation_window = 6 months
test_window = 6 months
roll_forward = 6 months
```

Objective:

```text
maximize:
  0.6 * validation_sharpe
  + 0.3 * validation_cagr
  - 0.1 * validation_max_drawdown_penalty
```

Reject parameter sets if:

- trades per year are too low.
- max drawdown exceeds threshold.
- turnover is unrealistic.
- performance comes from only one year.
- long side and short side are both unstable, unless intentionally long-only.

Use Optuna or grid/random search. Store every trial with full config, date range,
metrics, and random seed.

---

## 13. Suggested Project Structure

```text
project/
  data/
    spy/
    csi300_etf/
  src/
    data_spy.py
    data_china_etf.py
    sessions.py
    features.py
    signals.py
    exits.py
    position_sizing.py
    backtest.py
    metrics.py
    optimize.py
  configs/
    spy_baseline.yaml
    qqq_improved_candidates.yaml
    csi300_etf_research.yaml
    csi300_etf_overlay.yaml
  notebooks/
    01_validate_spy_reproduction.ipynb
    02_test_csi300_etf_signal.ipynb
  reports/
```

---

## 14. Implementation Pseudocode

```python
for date in trading_days:
    day_bars = intraday[intraday.date == date]

    if not has_enough_history(date):
        continue

    sigma_move = compute_same_time_noise(date, lookback_days)
    entry_bands = compute_entry_bands(date, sigma_move, vm_entry)
    exit_bands = compute_exit_bands(date, sigma_move, vm_exit)  # optional
    vwap = compute_intraday_vwap(day_bars)

    shares = compute_daily_position_size(date, equity, sigma_daily, config)
    position = flat
    ladder_state = None

    for bar in day_bars:
        if position != flat:
            exit_event = evaluate_exit_variant(
                bar=bar,
                position=position,
                vwap=vwap[bar.time],
                entry_band=entry_bands[bar.time],
                exit_band=exit_bands[bar.time],
                ladder_state=ladder_state,
                config=config,
            )

            if exit_event:
                execute_exit(exit_event)
                update_position_and_ladder_state()
                if position == flat:
                    continue

        if is_forced_exit_bar(bar, config):
            close_all_positions()
            break

        if is_scheduled_entry_bar(bar, config) and position_allows_entry():
            signal = evaluate_entry_signal(
                close=bar.close,
                upper_entry=entry_bands[bar.time].upper,
                lower_entry=entry_bands[bar.time].lower,
                vwap=vwap[bar.time],
                config=config,
            )

            if signal:
                if position_is_opposite(signal) and config.allow_reversal:
                    close_current_position()
                    open_new_position(signal, shares)
                elif position == flat:
                    open_new_position(signal, shares)

    mark_end_of_day_equity()
```

---

## 15. Minimum Acceptance Tests

Create tests for:

- Same-time move uses only prior days.
- Boundary formulas match hand-calculated examples.
- VWAP calculation is cumulative within each day and resets at next day.
- Entry schedule handles US regular session and China lunch break.
- Forced exit happens before close.
- Long and short VWAP exits trigger correctly.
- Boundary different exit uses `VM_exit`, not `VM_entry`.
- Ladder step 0 partial take-profit closes exactly 50%.
- Ladder step 1 closes the remaining position.
- Same-bar stop-loss/take-profit ambiguity uses conservative ordering.
- China overlay never sells more than previous-day available shares.

---

## 16. Source Links

- Maroy paper abstract:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5095349
- Concretum public SPY Python reproduction:
  https://concretumgroup.com/python-backtesting-beat-the-market-an-effective-intraday-momentum-strategy-for-the-sp500-etf-spy/
- Original paper text mirror:
  https://studylib.net/doc/27870742/concretum-bands
- Shanghai Stock Exchange trading rules:
  https://www.sse.com.cn/lawandrules/sselawsrules2025/fund/trading/c/c_20260424_10817739.shtml

