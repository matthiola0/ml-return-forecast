"""Per-stock feature engineering.

Inputs are wide DataFrames (index=date, columns=symbol) of adjusted close and
volume. All feature helpers preserve that shape so they can be stacked into a
long-format feature matrix downstream.

Design difference vs ml-cross-sectional's `features.py`:
- No cross-sectional normalization. Raw levels are kept because the regression
  target is absolute forward return, not cross-sectional rank — the model has
  to learn absolute magnitudes, so stripping cross-sectional means would erase
  the signal it needs.
- Classic factor signals with a baked-in sign flip (low_volatility, size_adv)
  are dropped — the regressor should learn the sign from the data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# per-stock technical features (raw, no sign flip, no cross-sectional norm)
# ---------------------------------------------------------------------------

def momentum_12_1(close: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """Total return over `lookback` days, skipping the last `skip` — classic 12-1."""
    return close.shift(skip) / close.shift(skip + lookback) - 1


def reversal_1w(close: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Past 1-week return (no sign flip; let the regressor learn direction)."""
    return close / close.shift(lookback) - 1


def horizon_return(close: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Plain cumulative return over `lookback` days (no skip)."""
    return close / close.shift(lookback) - 1


def realised_vol(close: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Rolling std of daily log returns (scale is relative)."""
    return np.log(close / close.shift(1)).rolling(lookback).std()


def rsi(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Wilder's RSI per symbol."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd_hist(close: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD histogram = (EMA_fast − EMA_slow) − signal-EMA-of-that."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return line - sig


def volume_zscore(volume: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Rolling z-score of log(1+volume) per symbol.

    Kept per-symbol rather than cross-sectional because absolute return
    prediction needs per-stock liquidity regime shifts, not relative ranking.
    """
    v = np.log1p(volume.astype("float64"))
    mean = v.rolling(lookback).mean()
    std = v.rolling(lookback).std()
    return (v - mean) / std.replace(0, np.nan)


# ---------------------------------------------------------------------------
# assembly — wide panels → long-format feature frame
# ---------------------------------------------------------------------------

STOCK_FEATURE_BUILDERS: dict[str, callable] = {
    "mom_12_1":      lambda c, v: momentum_12_1(c, 252, 21),
    "reversal_1w":   lambda c, v: reversal_1w(c, 5),
    "ret_21d":       lambda c, v: horizon_return(c, 21),
    "ret_63d":       lambda c, v: horizon_return(c, 63),
    "ret_126d":      lambda c, v: horizon_return(c, 126),
    "ret_252d":      lambda c, v: horizon_return(c, 252),
    "vol_20d":       lambda c, v: realised_vol(c, 20),
    "vol_60d":       lambda c, v: realised_vol(c, 60),
    "rsi_14":        lambda c, v: rsi(c, 14),
    "macd_hist":     lambda c, v: macd_hist(c),
    "volume_z_60":   lambda c, v: volume_zscore(v, 60),
}


def build_stock_features(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    """Compute per-stock features and return a long-format DataFrame.

    Columns: [date, symbol, <features...>]. No target column — target and
    macro/exposure features are attached by the assembly layer.
    """
    long_frames = []
    for name, build in STOCK_FEATURE_BUILDERS.items():
        wide = build(close, volume)
        s = wide.stack().rename(name)
        s.index.names = ["date", "symbol"]
        long_frames.append(s)

    return pd.concat(long_frames, axis=1).reset_index()


STOCK_FEATURE_COLS: tuple[str, ...] = tuple(STOCK_FEATURE_BUILDERS.keys())
