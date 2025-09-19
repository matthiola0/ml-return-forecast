"""Macro-level features derived from the FRED + S&P 500 snapshot.

Input: the `macro_*.parquet` frame produced by `scripts/download_macro.py`,
indexed by business-day date with columns:
    vix, ust_10y, term_slope, credit_spread, fed_funds, sp500_close

All derived features are daily-frequency, same index as the input. The output
is a wide DataFrame (one row per date) that is broadcast across all symbols
by the assembly layer — macro features are by construction identical for
every stock on the same date.

Look-ahead guard: every series is lagged by one business day so that the
forecast at date `t` only uses macro information that was already public by
the open of `t`. This is conservative (VIX close is published after the
close, but other series like BAA10Y have a T+1 lag) and simplifies the story.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MACRO_SOURCE_COLS = ("vix", "ust_10y", "term_slope", "credit_spread", "fed_funds", "sp500_close")


def build_macro_features(macro: pd.DataFrame) -> pd.DataFrame:
    """Expand the raw macro snapshot into derived features.

    Returns a wide DataFrame indexed by date. Columns are the macro feature
    names consumed by the model.
    """
    missing = set(MACRO_SOURCE_COLS) - set(macro.columns)
    if missing:
        raise ValueError(f"macro frame missing columns: {sorted(missing)}")

    out = pd.DataFrame(index=macro.index)

    # Level features (already standardised by nature of the series)
    out["vix"] = macro["vix"]
    out["vix_chg_20d"] = macro["vix"] - macro["vix"].shift(20)
    out["ust_10y"] = macro["ust_10y"]
    out["ust_10y_chg_20d"] = macro["ust_10y"] - macro["ust_10y"].shift(20)
    out["term_slope"] = macro["term_slope"]
    out["credit_spread"] = macro["credit_spread"]

    # Market trailing returns — absolute return prediction needs to know where
    # the market just was.
    sp500 = macro["sp500_close"]
    out["sp500_ret_3m"] = sp500 / sp500.shift(63) - 1
    out["sp500_ret_12m"] = sp500 / sp500.shift(252) - 1
    out["sp500_vol_60d"] = np.log(sp500 / sp500.shift(1)).rolling(60).std()

    # FOMC activity proxy: running count of fed-funds moves in the last 6
    # months. Crude but captures rate-cycle intensity without having to
    # hard-code meeting dates.
    ff_change = macro["fed_funds"].diff().abs().fillna(0.0)
    out["fed_funds_move_6m"] = (ff_change > 0).rolling(126).sum()

    # One-business-day lag applied to every column — look-ahead guard.
    out = out.shift(1)

    return out


MACRO_FEATURE_COLS: tuple[str, ...] = (
    "vix",
    "vix_chg_20d",
    "ust_10y",
    "ust_10y_chg_20d",
    "term_slope",
    "credit_spread",
    "sp500_ret_3m",
    "sp500_ret_12m",
    "sp500_vol_60d",
    "fed_funds_move_6m",
)
