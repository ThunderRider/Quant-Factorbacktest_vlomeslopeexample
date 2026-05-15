import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2026-04-29"
DEFAULT_AS_OF_DATE = "2026-04-29"
DEFAULT_OUTPUT_DIR = "data_rq"
DEFAULT_CONFIG_PATH = "config.yaml"
PRICE_FIELDS = ("open", "close", "total_turnover")
DEFAULT_TURNOVER_FIELD = "today"
DEFAULT_MARKET_CAP_FACTOR = "market_cap"


class DataFetchError(RuntimeError):
    """Raised when a required RQData dataset cannot be fetched cleanly."""


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fetch China A-share price, turnover, market-cap, and industry data "
            "from RQData/RiceQuant and cache it locally."
        )
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--as-of-date", default=DEFAULT_AS_OF_DATE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Number of stocks to request per RQData call.",
    )
    parser.add_argument(
        "--storage",
        choices=("parquet", "csv", "both"),
        default="parquet",
        help="Matrix cache format. Default: parquet.",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write CSV matrix files when --storage parquet is used.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refetch data and overwrite existing matrix cache files.",
    )
    parser.add_argument(
        "--rq-username",
        default=None,
        help="RQData username. Overrides config.yaml and environment/default config.",
    )
    parser.add_argument(
        "--rq-password",
        default=None,
        help=(
            "RQData password. Prefer RQDATAC2_CONF/RQDATAC_CONF if you do not "
            "want it in shell history."
        ),
    )
    parser.add_argument(
        "--rq-uri",
        default=None,
        help="RQData URI, for example rqdata://username:password@host:port.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=(
            "YAML config path for RQData credentials. Default: config.yaml. "
            "Supported keys: rq_api.license/key, rq_api.username/password, rq_api.uri."
        ),
    )
    parser.add_argument(
        "--turnover-field",
        default=DEFAULT_TURNOVER_FIELD,
        help=(
            "Field passed to rqdatac.get_turnover_rate(). "
            "Default: today (daily turnover rate)."
        ),
    )
    parser.add_argument(
        "--market-cap-factor",
        default=DEFAULT_MARKET_CAP_FACTOR,
        help=(
            "Factor passed to rqdatac.get_factor() for market capitalization. "
            "Common choices: market_cap, market_cap_3, a_share_market_val_3, "
            "a_share_market_val_in_circulation."
        ),
    )
    return parser.parse_args()


def init_rqdatac(rqdatac, args):
    config = load_rq_config(args.config)
    uri = args.rq_uri or config.get("uri")
    username = args.rq_username or config.get("username") or config.get("license")
    password = args.rq_password or config.get("password") or config.get("key")

    if uri:
        rqdatac.init(uri=uri)
        return "config_uri" if not args.rq_uri else "cli_uri"
    if username or password:
        if not username or not password:
            raise SystemExit(
                "RQData credentials are incomplete. Provide both username/password "
                "or both rq_api.license/rq_api.key in config.yaml."
            )
        rqdatac.init(username=username, password=password)
        return "config_license_key" if not args.rq_username else "cli_username_password"

    rqdatac.init()
    return "default_config"


def load_rq_config(config_path):
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.exists():
        return {}

    try:
        import yaml  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise SystemExit("Reading config.yaml requires PyYAML. Install pyyaml in the rq environment.") from exc

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"{path} must contain a YAML mapping.")

    rq_api = raw.get("rq_api", raw)
    if rq_api is None:
        return {}
    if not isinstance(rq_api, dict):
        raise SystemExit("rq_api in config.yaml must be a YAML mapping.")

    return {
        key: value
        for key, value in rq_api.items()
        if key in {"license", "key", "username", "password", "uri"} and value
    }


def infer_rq_login_label(args):
    if args.rq_uri:
        return "cli_uri"
    if args.rq_username:
        return "cli_username_password"

    config = load_rq_config(args.config)
    if config.get("uri"):
        return "config_uri"
    if (config.get("license") and config.get("key")) or (
        config.get("username") and config.get("password")
    ):
        return "config_license_key"
    return "default_config"


def import_rqdatac():
    try:
        import rqdatac  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise SystemExit(
            "rqdatac is not installed in this Python environment. "
            "Activate the RiceQuant/RQData environment or install rqdatac, "
            "then run this script again."
        ) from exc
    return rqdatac


def date_token(date_text):
    return date_text.replace("-", "")


def matrix_stem(field, start_date, end_date):
    return f"cn_stock_{field}_{date_token(start_date)}_{date_token(end_date)}"


def matrix_paths(output_dir, field, start_date, end_date):
    stem = matrix_stem(field, start_date, end_date)
    return {
        "parquet": output_dir / f"{stem}.parquet",
        "csv": output_dir / f"{stem}.csv",
    }


def requested_formats(args):
    if args.storage == "both":
        return ("parquet", "csv")
    if args.storage == "csv":
        return ("csv",)
    if args.write_csv:
        return ("parquet", "csv")
    return ("parquet",)


def preferred_cache_path(output_dir, field, start_date, end_date, formats):
    paths = matrix_paths(output_dir, field, start_date, end_date)
    for fmt in formats:
        if paths[fmt].exists():
            return paths[fmt]
    return None


def read_matrix(path):
    if path.suffix == ".parquet":
        matrix = pd.read_parquet(path)
    else:
        matrix = pd.read_csv(path, index_col=0, parse_dates=True)
    matrix.index = pd.to_datetime(matrix.index).tz_localize(None)
    matrix.index.name = "date"
    matrix.columns = matrix.columns.astype(str)
    return matrix.sort_index().sort_index(axis=1)


def write_matrix(matrix, output_dir, field, start_date, end_date, formats):
    paths = matrix_paths(output_dir, field, start_date, end_date)
    written = {}
    for fmt in formats:
        path = paths[fmt]
        if fmt == "parquet":
            try:
                matrix.to_parquet(path)
            except ImportError as exc:
                raise RuntimeError(
                    "Parquet output requires pyarrow or fastparquet. "
                    "Install pyarrow or rerun with --storage csv."
                ) from exc
            except ValueError as exc:
                message = str(exc).lower()
                if "parquet" in message or "pyarrow" in message or "fastparquet" in message:
                    raise RuntimeError(
                        "Parquet output requires pyarrow or fastparquet. "
                        "Install pyarrow or rerun with --storage csv."
                    ) from exc
                raise
        elif fmt == "csv":
            matrix.to_csv(path, encoding="utf-8-sig")
        else:
            raise ValueError(f"Unsupported storage format: {fmt}")
        print(f"Wrote {path}")
        written[fmt] = str(path)
    return written


def coerce_listed_date(series):
    text = series.astype(str)
    valid = (text > "19900101") & (text < "21000101")
    parsed = pd.to_datetime(text.where(valid), errors="coerce")
    return parsed


def get_stock_universe(rqdatac, as_of_date):
    cutoff = datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=365)

    all_stocks = rqdatac.all_instruments(type="CS", market="cn").copy()
    if "order_book_id" not in all_stocks.columns:
        raise DataFetchError("rqdatac.all_instruments() did not return order_book_id.")
    if "listed_date" not in all_stocks.columns:
        raise DataFetchError("rqdatac.all_instruments() did not return listed_date.")
    if "status" not in all_stocks.columns:
        raise DataFetchError("rqdatac.all_instruments() did not return status.")

    all_stocks["listed_date"] = coerce_listed_date(all_stocks["listed_date"])
    universe = all_stocks[
        (all_stocks["listed_date"].notna())
        & (all_stocks["listed_date"] < cutoff)
        & (all_stocks["status"] == "Active")
    ].copy()
    universe = universe.sort_values("order_book_id").reset_index(drop=True)
    return universe


def find_multiindex_level(index, preferred_names, values=None):
    if not isinstance(index, pd.MultiIndex):
        return None

    for name in preferred_names:
        if name in index.names:
            return name

    if values:
        value_set = set(values)
        for level_number in range(index.nlevels):
            sample = index.get_level_values(level_number)
            sample_values = set(sample.astype(str)[: min(100, len(sample))])
            if sample_values & value_set:
                return index.names[level_number] or level_number

    return None


def normalize_matrix(data, field, order_book_ids):
    if data is None:
        return pd.DataFrame()

    if isinstance(data, pd.Series):
        series = data
        if not isinstance(series.index, pd.MultiIndex):
            raise DataFetchError(f"{field} returned a Series without a MultiIndex.")
        return series_to_matrix(series, field, order_book_ids)

    if not isinstance(data, pd.DataFrame):
        raise DataFetchError(f"{field} returned unsupported type: {type(data).__name__}")

    if data.empty:
        return pd.DataFrame()

    if {"order_book_id", "date", field}.issubset(data.columns):
        return pivot_tabular_frame(data, field, "date", order_book_ids)
    if {"order_book_id", "trading_date", field}.issubset(data.columns):
        return pivot_tabular_frame(data, field, "trading_date", order_book_ids)

    if isinstance(data.index, pd.MultiIndex):
        if field in data.columns:
            series = data[field]
        elif len(data.columns) == 1:
            series = data.iloc[:, 0]
        else:
            raise DataFetchError(
                f"{field} returned a MultiIndex DataFrame without a clear field column."
            )
        return series_to_matrix(series, field, order_book_ids)

    if isinstance(data.index, pd.DatetimeIndex):
        matrix = data.copy()
        return finalize_matrix(matrix, order_book_ids)

    try:
        converted = data.copy()
        converted.index = pd.to_datetime(converted.index)
        return finalize_matrix(converted, order_book_ids)
    except Exception:
        pass

    try:
        converted = data.T.copy()
        converted.index = pd.to_datetime(converted.index)
        return finalize_matrix(converted, order_book_ids)
    except Exception as exc:
        raise DataFetchError(f"Could not normalize {field} into a date x stock matrix.") from exc


def pivot_tabular_frame(data, field, date_column, order_book_ids):
    matrix = data.pivot(index=date_column, columns="order_book_id", values=field)
    return finalize_matrix(matrix, order_book_ids)


def series_to_matrix(series, field, order_book_ids):
    stock_level = find_multiindex_level(
        series.index,
        preferred_names=("order_book_id", "asset", "instrument", "symbol"),
        values=order_book_ids,
    )
    if stock_level is None:
        raise DataFetchError(f"Could not identify stock level in {field} MultiIndex.")

    matrix = series.unstack(stock_level)
    return finalize_matrix(matrix, order_book_ids)


def finalize_matrix(matrix, order_book_ids):
    matrix = matrix.copy()
    matrix.index = pd.to_datetime(matrix.index).tz_localize(None)
    matrix.index.name = "date"
    matrix.columns = matrix.columns.astype(str)
    ordered_columns = [stock for stock in order_book_ids if stock in matrix.columns]
    matrix = matrix.loc[:, ordered_columns]
    return matrix.sort_index().sort_index(axis=1)


def combine_pieces(pieces, field):
    pieces = [piece for piece in pieces if piece is not None and not piece.empty]
    if not pieces:
        raise DataFetchError(f"No non-empty data returned for {field}.")
    matrix = pd.concat(pieces, axis=1)
    matrix = matrix.loc[:, ~matrix.columns.duplicated()]
    matrix.index = pd.to_datetime(matrix.index).tz_localize(None)
    matrix.index.name = "date"
    matrix.columns = matrix.columns.astype(str)
    return matrix.sort_index().sort_index(axis=1)


def fetch_price_matrices(rqdatac, order_book_ids, start_date, end_date, chunk_size):
    pieces_by_field = {field: [] for field in PRICE_FIELDS}
    total = len(order_book_ids)

    for start in range(0, total, chunk_size):
        chunk = order_book_ids[start : start + chunk_size]
        print(f"Fetching price fields: {start + 1}-{min(start + chunk_size, total)} / {total}")
        data = rqdatac.get_price(
            chunk,
            start_date=start_date,
            end_date=end_date,
            frequency="1d",
            fields=list(PRICE_FIELDS),
            adjust_type="pre",
            expect_df=True,
        )
        if data is None or getattr(data, "empty", False):
            continue
        for field in PRICE_FIELDS:
            pieces_by_field[field].append(normalize_matrix(data, field, chunk))

    return {field: combine_pieces(pieces, field) for field, pieces in pieces_by_field.items()}


def fetch_factor_matrix(rqdatac, order_book_ids, factor_name, start_date, end_date, chunk_size):
    pieces = []
    total = len(order_book_ids)
    for start in range(0, total, chunk_size):
        chunk = order_book_ids[start : start + chunk_size]
        print(f"Fetching factor {factor_name}: {start + 1}-{min(start + chunk_size, total)} / {total}")
        data = rqdatac.get_factor(
            chunk,
            factor=factor_name,
            start_date=start_date,
            end_date=end_date,
            expect_df=True,
            market="cn",
        )
        if data is None or getattr(data, "empty", False):
            continue
        pieces.append(normalize_matrix(data, factor_name, chunk))

    return combine_pieces(pieces, factor_name)


def fetch_turnover_rate_matrix(rqdatac, order_book_ids, turnover_field, start_date, end_date, chunk_size):
    pieces = []
    total = len(order_book_ids)
    for start in range(0, total, chunk_size):
        chunk = order_book_ids[start : start + chunk_size]
        print(
            f"Fetching turnover_rate.{turnover_field}: "
            f"{start + 1}-{min(start + chunk_size, total)} / {total}"
        )
        data = rqdatac.get_turnover_rate(
            chunk,
            start_date=start_date,
            end_date=end_date,
            fields=turnover_field,
            expect_df=True,
            market="cn",
        )
        if data is None or getattr(data, "empty", False):
            continue
        pieces.append(normalize_matrix(data, turnover_field, chunk))

    return combine_pieces(pieces, f"turnover_rate.{turnover_field}")


def numeric_summary(matrix):
    values = matrix.to_numpy(dtype=float, copy=False)
    finite = np.isfinite(values)
    finite_count = int(finite.sum())
    total_count = int(values.size)
    if finite_count == 0:
        return {
            "shape": list(matrix.shape),
            "finite_count": 0,
            "total_count": total_count,
            "finite_ratio": 0.0,
            "min": None,
            "mean": None,
            "median": None,
            "negative_count": 0,
            "negative_ratio": 0.0,
            "positive_ratio": 0.0,
        }

    finite_values = values[finite]
    negative_count = int((finite_values < 0).sum())
    positive_count = int((finite_values > 0).sum())
    return {
        "shape": list(matrix.shape),
        "finite_count": finite_count,
        "total_count": total_count,
        "finite_ratio": finite_count / total_count if total_count else 0.0,
        "min": float(np.min(finite_values)),
        "mean": float(np.mean(finite_values)),
        "median": float(np.median(finite_values)),
        "negative_count": negative_count,
        "negative_ratio": negative_count / finite_count,
        "positive_ratio": positive_count / finite_count,
    }


def validate_real_factor(matrix, logical_name, max_negative_ratio):
    summary = numeric_summary(matrix)
    if summary["finite_count"] == 0:
        raise DataFetchError(f"{logical_name} contains no finite values.")
    if summary["negative_ratio"] > max_negative_ratio:
        raise DataFetchError(
            f"{logical_name} has too many negative values "
            f"({summary['negative_ratio']:.2%}); refusing to cache likely proxy/bad data."
        )
    if logical_name == "market_cap" and summary["positive_ratio"] < 0.95:
        raise DataFetchError(
            f"{logical_name} has too few positive values "
            f"({summary['positive_ratio']:.2%}); refusing to cache bad market-cap data."
        )
    return summary


def align_and_validate(matrices):
    required = ("close", "turnover_rate", "market_cap")
    common_index = matrices[required[0]].index
    common_columns = matrices[required[0]].columns
    for field in required[1:]:
        common_index = common_index.intersection(matrices[field].index)
        common_columns = common_columns.intersection(matrices[field].columns)

    if common_index.empty:
        raise DataFetchError("close, turnover_rate, and market_cap have no common dates.")
    if common_columns.empty:
        raise DataFetchError("close, turnover_rate, and market_cap have no common stocks.")

    return {
        "common_dates": int(len(common_index)),
        "common_stocks": int(len(common_columns)),
        "first_common_date": common_index.min().strftime("%Y-%m-%d"),
        "last_common_date": common_index.max().strftime("%Y-%m-%d"),
    }


def ensure_output_matrix(
    field,
    fetcher,
    args,
    output_dir,
    formats,
    logical_name=None,
    max_negative_ratio=None,
):
    cache_path = preferred_cache_path(output_dir, field, args.start_date, args.end_date, formats)
    if cache_path is not None and not args.force:
        print(f"Using cached {field}: {cache_path}")
        matrix = read_matrix(cache_path)
        if logical_name is not None and max_negative_ratio is not None:
            summary = validate_real_factor(matrix, logical_name, max_negative_ratio)
        else:
            summary = numeric_summary(matrix)
        existing = {cache_path.suffix.lstrip("."): str(cache_path)}
        paths = matrix_paths(output_dir, field, args.start_date, args.end_date)
        missing_formats = [fmt for fmt in formats if not paths[fmt].exists()]
        if missing_formats:
            existing.update(write_matrix(matrix, output_dir, field, args.start_date, args.end_date, missing_formats))
    else:
        matrix = fetcher()
        if logical_name is not None and max_negative_ratio is not None:
            summary = validate_real_factor(matrix, logical_name, max_negative_ratio)
        else:
            summary = numeric_summary(matrix)
        existing = write_matrix(matrix, output_dir, field, args.start_date, args.end_date, formats)

    return matrix, existing, summary


def write_failure_meta(output_dir, args, error):
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": "rqdatac",
        "status": "failed",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "as_of_date": args.as_of_date,
        "output_dir": str(output_dir),
        "storage": args.storage,
        "write_csv": args.write_csv,
        "rq_login": infer_rq_login_label(args),
        "turnover_source": "get_turnover_rate",
        "turnover_field": args.turnover_field,
        "market_cap_factor": args.market_cap_factor,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    meta_path = output_dir / "cn_stock_cache_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote failure metadata to {meta_path}")


def main():
    args = parse_args()
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be a positive integer.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = requested_formats(args)
    rqdatac = import_rqdatac()

    try:
        rq_login = init_rqdatac(rqdatac, args)

        universe = get_stock_universe(rqdatac, args.as_of_date)
        if universe.empty:
            raise DataFetchError("Stock universe is empty after Active/listed-one-year filtering.")

        universe_path = output_dir / f"cn_stock_universe_{date_token(args.as_of_date)}.csv"
        universe.to_csv(universe_path, index=False, encoding="utf-8-sig")
        print(f"Wrote {universe_path} with {len(universe)} stocks")

        order_book_ids = universe["order_book_id"].astype(str).tolist()
        files = {}
        summaries = {}

        fetched_price_matrices = None
        price_matrices = {}

        def fetch_all_prices_once():
            nonlocal fetched_price_matrices
            if fetched_price_matrices is None:
                fetched_price_matrices = fetch_price_matrices(
                    rqdatac,
                    order_book_ids,
                    args.start_date,
                    args.end_date,
                    args.chunk_size,
                )
            return fetched_price_matrices

        for price_field in PRICE_FIELDS:
            matrix, written, summary = ensure_output_matrix(
                price_field,
                fetcher=lambda field=price_field: fetch_all_prices_once()[field],
                args=args,
                output_dir=output_dir,
                formats=formats,
            )
            files[price_field] = written
            summaries[price_field] = summary
            price_matrices[price_field] = matrix

        turnover_matrix, written, summary = ensure_output_matrix(
            "turnover_rate",
            fetcher=lambda: fetch_turnover_rate_matrix(
                rqdatac,
                order_book_ids,
                args.turnover_field,
                args.start_date,
                args.end_date,
                args.chunk_size,
            ),
            args=args,
            output_dir=output_dir,
            formats=formats,
            logical_name="turnover_rate",
            max_negative_ratio=0.001,
        )
        files["turnover_rate"] = written
        summaries["turnover_rate"] = summary

        market_cap_matrix, written, summary = ensure_output_matrix(
            "market_cap",
            fetcher=lambda: fetch_factor_matrix(
                rqdatac,
                order_book_ids,
                args.market_cap_factor,
                args.start_date,
                args.end_date,
                args.chunk_size,
            ),
            args=args,
            output_dir=output_dir,
            formats=formats,
            logical_name="market_cap",
            max_negative_ratio=0.001,
        )
        files["market_cap"] = written
        summaries["market_cap"] = summary

        validation = align_and_validate(
            {
                "close": price_matrices["close"],
                "turnover_rate": turnover_matrix,
                "market_cap": market_cap_matrix,
            }
        )

        meta = {
            "source": "rqdatac",
            "status": "ok",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "as_of_date": args.as_of_date,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "price_fields": list(PRICE_FIELDS),
            "turnover_source": "get_turnover_rate",
            "turnover_field": args.turnover_field,
            "market_cap_factor": args.market_cap_factor,
            "stock_count": len(order_book_ids),
            "chunk_size": args.chunk_size,
            "storage": args.storage,
            "write_csv": args.write_csv,
            "rq_login": rq_login,
            "universe_file": str(universe_path),
            "files": files,
            "summaries": summaries,
            "alignment": validation,
        }
        meta_path = output_dir / "cn_stock_cache_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {meta_path}")
        print("Data quality checks passed.")

    except Exception as exc:
        write_failure_meta(output_dir, args, exc)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
