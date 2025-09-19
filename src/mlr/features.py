"""Assemble the full feature matrix: stock + macro + exposure.

Pipeline:
    wide OHLCV     ─┐
    wide macro     ─┤─> long feature matrix (date, symbol, features..., target)
    market index   ─┤
    sector map     ─┘

Exposure features added at this layer (not in features_stock / features_macro
because they need both panels):
- `beta_252d`: rolling 252-day beta of stock return vs S&P 500 return
- `sector_*`:  GICS sector one-hot dummies

Target:
- `fwd_ret_21d`: close[t+21] / close[t] - 1, raw continuous value.
- Optional: `fwd_ret_21d_residual` = fwd_ret_21d - market_fwd_ret_21d (market
  leg hedged out), used when `build_feature_matrix(..., residualize=True)`.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from mlr.features_macro import MACRO_FEATURE_COLS, build_macro_features
from mlr.features_stock import STOCK_FEATURE_COLS, build_stock_features


# ---------------------------------------------------------------------------
# exposure features
# ---------------------------------------------------------------------------

def rolling_beta(
    stock_returns: pd.DataFrame,
    market_returns: pd.Series,
    window: int = 252,
) -> pd.DataFrame:
    """252-day rolling beta of each stock vs market return.

    Shapes:
        stock_returns  (dates × symbols) — simple daily returns
        market_returns (dates,)          — simple daily returns
        returns        (dates × symbols) — β_t per stock
    """
    aligned = stock_returns.reindex(market_returns.index)
    market_var = market_returns.rolling(window).var()

    # Cov(r_i, r_m) column-wise: rolling mean of r_i * r_m minus product of
    # rolling means. Vectorised across symbols.
    mkt_mean = market_returns.rolling(window).mean()
    stk_mean = aligned.rolling(window).mean()
    cross_mean = aligned.mul(market_returns, axis=0).rolling(window).mean()
    cov = cross_mean.sub(stk_mean.mul(mkt_mean, axis=0), axis=0)

    beta = cov.div(market_var, axis=0)
    return beta


# ---------------------------------------------------------------------------
# sector map
# ---------------------------------------------------------------------------

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def fetch_sp500_sectors(cache_path: Path | None = None) -> pd.DataFrame:
    """Return DataFrame with columns ['symbol', 'sector'].

    If `cache_path` is given and the file exists, load from there; otherwise
    hit Wikipedia and write the CSV for future calls. Sector is the GICS
    sector (11 in total).
    """
    if cache_path is not None and cache_path.exists():
        return pd.read_csv(cache_path)

    # Wikipedia rejects urllib's default user-agent with 403, so fetch via
    # requests with a browser UA and hand the HTML to pd.read_html.
    resp = requests.get(
        WIKI_SP500_URL,
        headers={"User-Agent": "Mozilla/5.0 (research script)"},
        timeout=30,
    )
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    df = tables[0][["Symbol", "GICS Sector"]].rename(
        columns={"Symbol": "symbol", "GICS Sector": "sector"}
    )
    # yfinance uses "-" in tickers (BRK.B → BRK-B); Wikipedia uses ".".
    df["symbol"] = df["symbol"].str.replace(".", "-", regex=False)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)

    return df


# ---------------------------------------------------------------------------
# target
# ---------------------------------------------------------------------------

def forward_return(close: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """close[t+horizon] / close[t] - 1, raw."""
    return close.shift(-horizon) / close - 1


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def build_feature_matrix(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    macro: pd.DataFrame,
    sectors: pd.DataFrame,
    horizon: int = 21,
    residualize: bool = False,
) -> pd.DataFrame:
    """Assemble the full long-format feature matrix.

    Returns columns:
        date, symbol,
        <stock features...>, <macro features...>,
        beta_252d, sector_<A>, sector_<B>, ...,
        fwd_ret_21d, [fwd_ret_21d_residual if residualize]
    """
    # 1. per-stock technical features (long)
    stock = build_stock_features(close, volume)

    # 2. macro features (wide → to be merged by date)
    macro_feats = build_macro_features(macro)

    # 3. rolling beta vs ^GSPC (wide → long)
    market_close = macro["sp500_close"].reindex(close.index).ffill()
    market_ret = market_close.pct_change()
    stock_returns = close.pct_change()
    beta = rolling_beta(stock_returns, market_ret, window=252)
    beta_long = beta.stack().rename("beta_252d")
    beta_long.index.names = ["date", "symbol"]
    beta_long = beta_long.reset_index()

    # 4. sector one-hot (static, per symbol)
    sector_dummies = pd.get_dummies(
        sectors.set_index("symbol")["sector"], prefix="sector", dtype="int8"
    ).reset_index()

    # 5. forward target (wide → long)
    fwd = forward_return(close, horizon=horizon)
    fwd_long = fwd.stack().rename(f"fwd_ret_{horizon}d")
    fwd_long.index.names = ["date", "symbol"]
    fwd_long = fwd_long.reset_index()

    # --- merge everything on (date, symbol) / date / symbol ------------------

    out = stock.merge(beta_long, on=["date", "symbol"], how="left")
    out = out.merge(macro_feats.reset_index().rename(columns={"index": "date"}),
                    on="date", how="left")
    out = out.merge(sector_dummies, on="symbol", how="left")
    out = out.merge(fwd_long, on=["date", "symbol"], how="left")

    if residualize:
        market_fwd = (market_close.shift(-horizon) / market_close - 1).rename("market_fwd")
        out = out.merge(
            market_fwd.reset_index().rename(columns={market_fwd.index.name or "index": "date"}),
            on="date",
            how="left",
        )
        out[f"fwd_ret_{horizon}d_residual"] = (
            out[f"fwd_ret_{horizon}d"] - out["market_fwd"]
        )
        out = out.drop(columns=["market_fwd"])

    return out


# Column groups exported so downstream code doesn't have to sniff the frame.
STOCK_COLS = STOCK_FEATURE_COLS
MACRO_COLS = MACRO_FEATURE_COLS
EXPOSURE_COLS = ("beta_252d",)


def model_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the ordered list of model-input columns from an assembled frame.

    Non-features excluded: date, symbol, fwd_*.
    """
    return [
        c for c in df.columns
        if c not in ("date", "symbol")
        and not c.startswith("fwd_")
    ]
