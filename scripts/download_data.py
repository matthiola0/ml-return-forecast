"""Fetch S&P 500 OHLCV via qtools and cache to data/raw/.

Universe is *current* S&P 500 constituents — survivorship bias is acknowledged
in README.md. Price data is cached once in qtools' own parquet cache
(~/.qtools_cache) and then re-exported here so downstream scripts have a
deterministic path to read from.

Usage:
    python scripts/download_data.py                       # default: 2015-01-01 → 2025-07-31
    python scripts/download_data.py --start 2010-01-01
"""
from __future__ import annotations

import argparse
from pathlib import Path

from qtools.data.loaders.us import get_sp500_constituents, get_us_prices

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

# Matches ml-cross-sectional so cross-repo comparisons (05_vs_ranking) use the
# same universe and sample window.
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2025-07-31"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching S&P 500 constituents from Wikipedia...")
    tickers = get_sp500_constituents()
    print(f"  {len(tickers)} tickers")

    print(f"Downloading OHLCV {args.start} → {args.end} (adjusted)...")
    df = get_us_prices(tickers, start=args.start, end=args.end, adjust=True)
    print(f"  {len(df):,} rows · {df['symbol'].nunique()} unique symbols")

    out = RAW_DIR / f"sp500_ohlcv_{args.start}_{args.end}.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
