import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import rqdatac


DEFAULT_START_DATE = "2025-04-01"
DEFAULT_END_DATE = "2026-04-01"
DEFAULT_AS_OF_DATE = "2026-03-01"
DEFAULT_FIELDS = ("open", "close", "total_turnover")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch China A-share data from RQData and cache it locally."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--as-of-date", default=DEFAULT_AS_OF_DATE)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument(
        "--fields",
        nargs="+",
        default=list(DEFAULT_FIELDS),
        help="RQData price fields to cache.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Number of stocks to request per RQData call.",
    )
    parser.add_argument(
        "--no-parquet",
        action="store_true",
        help="Only write CSV files. By default the script also tries parquet.",
    )
    return parser.parse_args()


def get_stock_universe(as_of_date):
    end_date = datetime.strptime(as_of_date, "%Y-%m-%d")
    one_year_ago = end_date - timedelta(days=365)

    all_stocks = rqdatac.all_instruments(type="CS", market="cn")
    all_stocks = all_stocks[
        (all_stocks["listed_date"].astype(str) > "19900101")
        & (all_stocks["listed_date"].astype(str) < "21000101")
    ].copy()
    all_stocks["listed_date"] = pd.to_datetime(all_stocks["listed_date"])

    universe = all_stocks[
        (all_stocks["listed_date"] < one_year_ago)
        & (all_stocks["status"] == "Active")
    ].copy()
    universe = universe.sort_values("order_book_id")
    return universe


def fetch_field_matrix(order_book_ids, field, start_date, end_date, chunk_size):
    pieces = []
    total = len(order_book_ids)
    for start in range(0, total, chunk_size):
        chunk = order_book_ids[start : start + chunk_size]
        print(
            f"Fetching {field}: {start + 1}-{min(start + chunk_size, total)} / {total}"
        )
        data = rqdatac.get_price(
            chunk,
            start_date=start_date,
            end_date=end_date,
            frequency="1d",
            fields=[field],
            adjust_type="pre",
            expect_df=True,
        )
        if data.empty:
            continue

        if field in data.columns:
            series = data[field]
        else:
            # Some RQData versions return a single-column frame without the field name.
            series = data.iloc[:, 0]

        matrix = series.unstack("order_book_id")
        pieces.append(matrix)

    if not pieces:
        return pd.DataFrame()

    result = pd.concat(pieces, axis=1)
    result = result.loc[:, ~result.columns.duplicated()]
    return result.sort_index().sort_index(axis=1)


def write_matrix(matrix, output_dir, field, start_date, end_date, write_parquet):
    stem = f"cn_stock_{field}_{start_date.replace('-', '')}_{end_date.replace('-', '')}"
    csv_path = output_dir / f"{stem}.csv"
    matrix.to_csv(csv_path, encoding="utf-8-sig")
    print(f"Wrote {csv_path}")

    parquet_path = None
    if write_parquet:
        parquet_path = output_dir / f"{stem}.parquet"
        try:
            matrix.to_parquet(parquet_path)
            print(f"Wrote {parquet_path}")
        except Exception as exc:
            parquet_path = None
            print(f"Skipped parquet for {field}: {type(exc).__name__}: {exc}")

    return csv_path, parquet_path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rqdatac.init()

    universe = get_stock_universe(args.as_of_date)
    universe_path = output_dir / f"cn_stock_universe_{args.as_of_date.replace('-', '')}.csv"
    universe.to_csv(universe_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {universe_path} with {len(universe)} stocks")

    order_book_ids = universe["order_book_id"].tolist()
    files = {}
    for field in args.fields:
        matrix = fetch_field_matrix(
            order_book_ids,
            field,
            args.start_date,
            args.end_date,
            args.chunk_size,
        )
        csv_path, parquet_path = write_matrix(
            matrix,
            output_dir,
            field,
            args.start_date,
            args.end_date,
            write_parquet=not args.no_parquet,
        )
        files[field] = {
            "csv": str(csv_path),
            "parquet": str(parquet_path) if parquet_path else None,
            "shape": list(matrix.shape),
        }

    meta = {
        "source": "rqdatac",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": args.as_of_date,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "fields": args.fields,
        "stock_count": len(order_book_ids),
        "chunk_size": args.chunk_size,
        "universe_file": str(universe_path),
        "files": files,
    }
    meta_path = output_dir / "cn_stock_cache_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
