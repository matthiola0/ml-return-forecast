"""Fetch macro time series from FRED + S&P index from Yahoo, cache to data/raw/.

Series pulled:
    VIXCLS          — CBOE VIX close
    DGS10           — 10-Year Treasury constant-maturity yield
    T10Y2Y          — 10Y minus 2Y term slope
    BAA10Y          — Moody's BAA corporate yield minus 10Y Treasury (credit spread)
    FEDFUNDS        — Federal funds effective rate (monthly, used for FOMC change dummy)
    ^GSPC          — S&P 500 index close (for rolling 3M/12M return and beta)

Note on credit spread: ICE's BAMLH0A0HYM2 (HY OAS) was the first pick, but
FRED's public CSV endpoint only returns ~2 years for that series due to an
ICE licensing change. BAA10Y is a reasonable IG credit-stress proxy that
goes back to the 1980s.

All series are re-indexed onto a daily business-day calendar and forward-filled
so downstream feature code can align by date without worrying about each
series' native frequency (VIX is daily, FEDFUNDS is monthly, etc.).

FRED's public CSV endpoint is used directly (no API key, no pandas-datareader —
the latter is broken under pandas 3.x as of April 2026).

Usage:
    python scripts/download_macro.py                        # default window
    python scripts/download_macro.py --start 2010-01-01
"""
from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

DEFAULT_START = "2014-01-01"   # a year earlier than DEFAULT_START in download_data.py
                               # so rolling-window features have warmup room
DEFAULT_END = "2025-07-31"

FRED_SERIES = {
    "vix": "VIXCLS",
    "ust_10y": "DGS10",
    "term_slope": "T10Y2Y",
    "credit_spread": "BAA10Y",
    "fed_funds": "FEDFUNDS",
}


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}&cosd={start}&coed={end}"


def fetch_fred(start: str, end: str) -> pd.DataFrame:
    frames = []
    for nice_name, fred_code in FRED_SERIES.items():
        print(f"  FRED {fred_code} → {nice_name}")
        url = FRED_CSV_URL.format(code=fred_code, start=start, end=end)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        # FRED CSV columns are: observation_date,<CODE> — missing values as "."
        s = pd.read_csv(StringIO(resp.text), parse_dates=["observation_date"], na_values=["."])
        s = s.set_index("observation_date")
        s.columns = [nice_name]
        s[nice_name] = pd.to_numeric(s[nice_name], errors="coerce")
        frames.append(s)
    return pd.concat(frames, axis=1, sort=False)


def fetch_sp500_index(start: str, end: str) -> pd.Series:
    print("  Yahoo ^GSPC")
    df = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False)
    # yfinance returns a MultiIndex when a single ticker is passed in newer versions;
    # normalise to a plain Series.
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.name = "sp500_close"
    return close


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching macro {args.start} → {args.end}")
    fred = fetch_fred(args.start, args.end)
    sp500 = fetch_sp500_index(args.start, args.end)

    # Align onto daily business-day index, forward-fill sticky series
    # (fed_funds updates monthly; credit spread has occasional blanks).
    idx = pd.bdate_range(args.start, args.end)
    macro = fred.reindex(idx).ffill()
    macro["sp500_close"] = sp500.reindex(idx).ffill()
    macro.index.name = "date"

    # quick sanity print
    print(f"  {len(macro):,} business days · {macro.shape[1]} series")
    print(macro.tail(3).round(3))

    out = RAW_DIR / f"macro_{args.start}_{args.end}.parquet"
    macro.to_parquet(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
