import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt

from ASP_stk_all_bestpara_vbt import linearreg_slope_nb, read_matrix


DEFAULT_DATA_DIR = "data_rq"
DEFAULT_RESULTS_DIR = "results/moderate_volume_q3"
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2026-04-29"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backtest moderate positive volume-slope Q3 factor with vectorbt."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--target-quantile", type=int, default=3)
    parser.add_argument("--price-field", choices=("open", "close"), default="open")
    parser.add_argument("--signal-lag", type=int, default=1)
    parser.add_argument("--init-cash", type=float, default=1_000_000)
    parser.add_argument("--fees", type=float, default=0.000025)
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument(
        "--min-names",
        type=int,
        default=30,
        help="Drop dates with fewer selected stocks than this threshold.",
    )
    return parser.parse_args()


def calc_slope_frame(turnover, window):
    values = turnover.to_numpy(dtype=float)
    slope = linearreg_slope_nb(values, window)
    return pd.DataFrame(slope, index=turnover.index, columns=turnover.columns)


def calc_daily_quantiles(factor, quantiles):
    def qcut_row(row):
        valid = row.dropna()
        out = pd.Series(np.nan, index=row.index)
        if len(valid) < quantiles * 10:
            return out
        try:
            out.loc[valid.index] = pd.qcut(
                valid,
                quantiles,
                labels=False,
                duplicates="drop",
            ).astype(float) + 1
        except ValueError:
            pass
        return out

    return factor.apply(qcut_row, axis=1)


def build_target_weights(slope, quantiles, target_quantile, min_names):
    q = calc_daily_quantiles(slope, quantiles)
    selected = (q == target_quantile) & (slope > 0)
    counts = selected.sum(axis=1)
    selected = selected.where(counts >= min_names, False)
    weights = selected.div(selected.sum(axis=1).replace(0, np.nan), axis=0)
    return weights.fillna(0.0), q, selected


def calc_basic_stats(portfolio, benchmark_returns):
    stats = portfolio.stats()
    portfolio_returns = portfolio.returns()
    benchmark_total_return = (1 + benchmark_returns.fillna(0)).prod() - 1
    excess_total_return = portfolio.total_return() - benchmark_total_return
    return pd.Series(
        {
            "total_return": portfolio.total_return(),
            "benchmark_total_return": benchmark_total_return,
            "excess_total_return": excess_total_return,
            "annualized_return": stats.get("Annualized Return [%]", np.nan),
            "annualized_volatility": portfolio_returns.std() * np.sqrt(252),
            "sharpe": portfolio.sharpe_ratio(),
            "max_drawdown": portfolio.max_drawdown(),
            "calmar": stats.get("Calmar Ratio", np.nan),
        }
    )


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    price = read_matrix(args.data_dir, args.price_field, args.start_date, args.end_date)
    close = read_matrix(args.data_dir, "close", args.start_date, args.end_date)
    turnover = read_matrix(args.data_dir, "total_turnover", args.start_date, args.end_date)

    common = price.columns.intersection(turnover.columns).intersection(close.columns)
    price = price[common]
    close = close[common]
    turnover = turnover[common]

    print(f"Loaded {len(common)} stocks and {len(price)} bars")
    print(
        f"Strategy: slope window={args.window}, "
        f"Q{args.target_quantile}/{args.quantiles}, slope > 0"
    )

    slope = calc_slope_frame(turnover, args.window)
    weights, quantiles, selected = build_target_weights(
        slope,
        args.quantiles,
        args.target_quantile,
        args.min_names,
    )

    if args.signal_lag:
        weights = weights.shift(args.signal_lag).fillna(0.0)

    portfolio = vbt.Portfolio.from_orders(
        close=price,
        size=weights,
        size_type="targetpercent",
        init_cash=args.init_cash,
        fees=args.fees,
        slippage=args.slippage,
        freq="1D",
    )

    benchmark_returns = close.mean(axis=1).pct_change()
    stats = calc_basic_stats(portfolio, benchmark_returns)

    selected_counts = selected.sum(axis=1)
    turnover_series = weights.diff().abs().sum(axis=1) / 2

    run_stamp = date.today().strftime("%Y%m%d")
    prefix = f"moderate_volume_q{args.target_quantile}_w{args.window}_{run_stamp}"

    portfolio.value().to_csv(results_dir / f"{prefix}_equity_curve.csv", encoding="utf-8-sig")
    portfolio.returns().to_csv(results_dir / f"{prefix}_returns.csv", encoding="utf-8-sig")
    weights.to_csv(results_dir / f"{prefix}_target_weights.csv", encoding="utf-8-sig")
    stats.to_csv(results_dir / f"{prefix}_stats.csv", encoding="utf-8-sig")

    diagnostics = pd.DataFrame(
        {
            "selected_count": selected_counts,
            "target_weight_turnover": turnover_series,
        }
    )
    diagnostics.to_csv(results_dir / f"{prefix}_diagnostics.csv", encoding="utf-8-sig")

    print("\nStats:")
    print(stats)
    print("\nDiagnostics:")
    print(diagnostics.describe())
    print(f"\nWrote results to {results_dir}")


if __name__ == "__main__":
    main()
