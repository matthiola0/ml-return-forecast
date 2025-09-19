"""Walk-forward cross-validation helpers.

Same annual-expanding-window design as ml-cross-sectional. Each OOS year Y
uses every row with date < Jan 1 Y for training and every row with date in
year Y for test. No purging/embargo — with a 21-day target and annual re-fit
the look-ahead risk is already dominated by other sources of noise.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


def walk_forward_years(
    dates: pd.Series,
    first_oos_year: int,
    last_oos_year: int,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield `(oos_year, train_mask, test_mask)` one year at a time.

    Masks are boolean arrays aligned to `dates`. Skips any year where either
    side would be empty.
    """
    d = pd.to_datetime(dates)
    years = d.dt.year.values
    for oos_year in range(first_oos_year, last_oos_year + 1):
        train_mask = years < oos_year
        test_mask = years == oos_year
        if not train_mask.any() or not test_mask.any():
            continue
        yield oos_year, train_mask, test_mask
