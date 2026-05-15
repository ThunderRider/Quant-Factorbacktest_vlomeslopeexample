import argparse
import concurrent.futures
import json
import multiprocessing as mp
from datetime import date
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
import vectorbt as vbt


DEFAULT_DATA_DIR = "data_rq"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2026-04-29"
DEFAULT_PERIOD_START = 5
DEFAULT_PERIOD_END = 35


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize ASP turnover-slope parameters with vectorbt."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--period-start", type=int, default=DEFAULT_PERIOD_START)
    parser.add_argument("--period-end", type=int, default=DEFAULT_PERIOD_END)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of stocks to process per vectorbt batch.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel batch workers. Use 1 to run serially.",
    )
    parser.add_argument(
        "--price-field",
        default="close",
        choices=("open", "close"),
        help="Cached price matrix used by vectorbt as execution/valuation price.",
    )
    parser.add_argument("--init-cash", type=float, default=1_000_000)
    parser.add_argument("--fees", type=float, default=0.000025)
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument(
        "--signal-lag",
        type=int,
        default=1,
        help="Shift generated signals forward by N bars before execution.",
    )
    parser.add_argument(
        "--save-param-matrix",
        action="store_true",
        help="Also save each stock's total return for every period.",
    )
    return parser.parse_args()


def read_cache_meta(data_dir):
    meta_path = Path(data_dir) / "cn_stock_cache_meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def matrix_path(data_dir, field, start_date, end_date):
    stem = f"cn_stock_{field}_{start_date.replace('-', '')}_{end_date.replace('-', '')}"
    parquet_path = Path(data_dir) / f"{stem}.parquet"
    if parquet_path.exists():
        return parquet_path
    raise FileNotFoundError(f"Missing cached {field} parquet matrix: {parquet_path}")


def read_matrix(data_dir, field, start_date, end_date):
    path = matrix_path(data_dir, field, start_date, end_date)
    matrix = pd.read_parquet(path)
    matrix.index = pd.to_datetime(matrix.index)
    matrix = matrix.sort_index().sort_index(axis=1)
    return matrix


def read_universe_symbols(data_dir, meta):
    universe_file = meta.get("universe_file")
    if universe_file and Path(universe_file).exists():
        universe_path = Path(universe_file)
    else:
        candidates = sorted(Path(data_dir).glob("cn_stock_universe_*.csv"))
        if not candidates:
            return {}
        universe_path = candidates[-1]

    universe = pd.read_csv(universe_path)
    if "order_book_id" not in universe.columns or "symbol" not in universe.columns:
        return {}
    return universe.set_index("order_book_id")["symbol"].to_dict()


@nb.njit(cache=True)
def linearreg_slope_nb(turnover, window):
    rows, cols = turnover.shape
    out = np.empty((rows, cols), dtype=np.float64)
    out[:, :] = np.nan

    x_mean = (window - 1) / 2.0
    denominator = 0.0
    for i in range(window):
        x_diff = i - x_mean
        denominator += x_diff * x_diff

    for col in range(cols):
        for row in range(window - 1, rows):
            y_sum = 0.0
            valid = True
            for i in range(window):
                value = turnover[row - window + 1 + i, col]
                if np.isnan(value):
                    valid = False
                    break
                y_sum += value

            if not valid:
                continue

            y_mean = y_sum / window
            numerator = 0.0
            for i in range(window):
                value = turnover[row - window + 1 + i, col]
                numerator += (i - x_mean) * (value - y_mean)
            out[row, col] = numerator / denominator

    return out


ASPIndicator = vbt.IndicatorFactory(
    class_name="ASPIndicator",
    short_name="asp",
    input_names=["turnover"],
    param_names=["window"],
    output_names=["slope"],
).from_apply_func(linearreg_slope_nb, keep_pd=False, to_2d=True)


def build_all_period_signals(turnover, periods, signal_lag):
    indicator = ASPIndicator.run(turnover, window=periods)
    slope = indicator.slope
    prev_slope = slope.shift(1)
    entries = (slope >= 0) & (prev_slope < 0)
    exits = (slope < 0) & (prev_slope > 0)
    if signal_lag:
        entries = entries.shift(signal_lag)
        exits = exits.shift(signal_lag)
    return entries, exits


def buy_and_hold_return(price):
    first = price.bfill().iloc[0]
    last = price.ffill().iloc[-1]
    return (last / first) - 1


def expand_price_to_periods(price, periods):
    return pd.concat(
        [price] * len(periods),
        axis=1,
        keys=periods,
        names=["asp_window", None],
    )


def run_all_period_backtest(price, turnover, periods, init_cash, fees, slippage, signal_lag):
    entries, exits = build_all_period_signals(turnover, periods, signal_lag)
    expanded_price = expand_price_to_periods(price, periods)
    entries = entries.reindex_like(expanded_price).fillna(False)
    exits = exits.reindex_like(expanded_price).fillna(False)

    portfolio = vbt.Portfolio.from_signals(
        close=expanded_price,
        entries=entries,
        exits=exits,
        init_cash=init_cash,
        fees=fees,
        slippage=slippage,
        size=np.inf,
        accumulate=False,
        freq="1D",
    )
    total_return = portfolio.total_return().unstack(level=0)
    sharpe = portfolio.sharpe_ratio().unstack(level=0)
    max_drawdown = portfolio.max_drawdown().unstack(level=0)
    total_trades = portfolio.trades.count().unstack(level=0)
    return {
        "total_return": portfolio.total_return(),
        "total_return_matrix": total_return,
        "sharpe_matrix": sharpe,
        "max_drawdown_matrix": max_drawdown,
        "total_trades_matrix": total_trades,
    }


def optimize_batch(price, turnover, periods, symbols, init_cash, fees, slippage, signal_lag):
    benchmark_return = buy_and_hold_return(price)
    metrics = run_all_period_backtest(
        price, turnover, periods, init_cash, fees, slippage, signal_lag
    )
    total_return_matrix = metrics["total_return_matrix"]
    best_period = total_return_matrix.idxmax(axis=1)
    best_total_return = total_return_matrix.max(axis=1)

    rows = []
    for stock in total_return_matrix.index:
        period = int(best_period.loc[stock]) if pd.notna(best_period.loc[stock]) else None
        row = {
            "order_book_id": stock,
            "symbol": symbols.get(stock, ""),
            "best_longperiod": period,
            "total_return": best_total_return.loc[stock],
            "benchmark_total_return": benchmark_return.get(stock, np.nan),
            "excess_return": best_total_return.loc[stock]
            - benchmark_return.get(stock, np.nan),
        }
        if period is not None:
            row.update(
                {
                    "sharpe": metrics["sharpe_matrix"].at[stock, period],
                    "max_drawdown": metrics["max_drawdown_matrix"].at[stock, period],
                    "total_trades": metrics["total_trades_matrix"].at[stock, period],
                }
            )
        rows.append(row)

    return pd.DataFrame(rows), total_return_matrix


def run_batch_task(task):
    (
        batch_no,
        columns,
        price,
        turnover,
        periods,
        symbols,
        init_cash,
        fees,
        slippage,
        signal_lag,
        results_dir,
        run_stamp,
        save_param_matrix,
    ) = task

    print(f"Batch {batch_no}: {len(columns)} stocks")
    batch_price = price.loc[:, columns]
    batch_turnover = turnover.loc[:, columns]

    batch_result, batch_param_matrix = optimize_batch(
        batch_price,
        batch_turnover,
        periods,
        symbols,
        init_cash,
        fees,
        slippage,
        signal_lag,
    )
    part_path = Path(results_dir) / f"asp_vbt_bestpara_part_{batch_no:03d}_{run_stamp}.csv"
    batch_result.to_csv(part_path, index=False, encoding="utf-8-sig")

    param_part_path = None
    if save_param_matrix:
        batch_param_matrix.index.name = "order_book_id"
        param_part_path = (
            Path(results_dir) / f"asp_vbt_param_return_part_{batch_no:03d}_{run_stamp}.csv"
        )
        batch_param_matrix.to_csv(param_part_path, encoding="utf-8-sig")

    return batch_no, batch_result, batch_param_matrix if save_param_matrix else None, part_path, param_part_path


def iter_batches(columns, batch_size):
    for start in range(0, len(columns), batch_size):
        yield start // batch_size + 1, columns[start : start + batch_size]


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    meta = read_cache_meta(data_dir)
    symbols = read_universe_symbols(data_dir, meta)

    price = read_matrix(data_dir, args.price_field, args.start_date, args.end_date)
    turnover = read_matrix(data_dir, "total_turnover", args.start_date, args.end_date)
    common_columns = price.columns.intersection(turnover.columns)
    price = price[common_columns]
    turnover = turnover[common_columns]

    periods = list(range(args.period_start, args.period_end + 1))
    run_stamp = date.today().strftime("%Y%m%d")
    all_results = []
    all_param_matrices = []

    print(
        f"Loaded {len(common_columns)} stocks, {len(price)} bars, "
        f"periods {periods[0]}-{periods[-1]}, workers {args.workers}"
    )

    tasks = [
        (
            batch_no,
            list(columns),
            price.loc[:, columns],
            turnover.loc[:, columns],
            periods,
            symbols,
            args.init_cash,
            args.fees,
            args.slippage,
            args.signal_lag,
            str(results_dir),
            run_stamp,
            args.save_param_matrix,
        )
        for batch_no, columns in iter_batches(common_columns, args.batch_size)
    ]

    if args.workers <= 1:
        completed = [run_batch_task(task) for task in tasks]
    else:
        ctx = mp.get_context("spawn")
        completed = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=ctx,
        ) as executor:
            futures = {executor.submit(run_batch_task, task): task[0] for task in tasks}
            for future in concurrent.futures.as_completed(futures):
                batch_no = futures[future]
                try:
                    completed.append(future.result())
                except Exception as exc:
                    print(f"Batch {batch_no} failed: {type(exc).__name__}: {exc}")
                    raise

    for batch_no, batch_result, batch_param_matrix, part_path, param_part_path in sorted(
        completed, key=lambda item: item[0]
    ):
        print(f"Wrote {part_path}")
        if param_part_path:
            print(f"Wrote {param_part_path}")
        all_results.append(batch_result)
        if batch_param_matrix is not None:
            all_param_matrices.append(batch_param_matrix)

    result = pd.concat(all_results, ignore_index=True)
    result_path = results_dir / f"asp_vbt_bestpara_all_{run_stamp}.csv"
    result.to_csv(result_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {result_path}")

    if args.save_param_matrix and all_param_matrices:
        param_matrix = pd.concat(all_param_matrices, axis=0)
        param_path = results_dir / f"asp_vbt_param_return_matrix_{run_stamp}.csv"
        param_matrix.to_csv(param_path, encoding="utf-8-sig")
        print(f"Wrote {param_path}")


if __name__ == "__main__":
    main()
