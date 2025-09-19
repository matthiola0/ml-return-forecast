"""Regression-model wrappers with a unified fit/predict contract.

All wrappers take a long-format feature matrix `X`, a continuous target `y`
(typically `fwd_ret_21d`), and a matching `dates` series. They emit a scalar
prediction per row at predict time — the predicted forward return in the
same units as the target.

Design contrast with ml-cross-sectional:
- Target is absolute return, not cross-sectional rank. cs-zscore helpers are
  dropped; linear models use a single global `StandardScaler` fitted on the
  training fold so regularisation is meaningful.
- No decile binning, no pairwise ranking loss — LightGBM / XGBoost use their
  `*Regressor` variants with RMSE loss.
- `HistMeanBaseline` replaces the handmade equal-weight baseline: for each
  symbol, predict its trailing mean daily return scaled up by the horizon.
  This is the "zero-skill" bar that any learned model must beat.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# base contract
# ---------------------------------------------------------------------------

class BaseRegressor:
    """All wrappers take (X, y, dates, symbols) so the baseline can group by
    symbol. Learned models ignore `symbols` — it's passed for uniformity."""

    name: str = "base"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series,
        symbols: pd.Series,
    ) -> "BaseRegressor":
        raise NotImplementedError

    def predict(
        self,
        X: pd.DataFrame,
        dates: pd.Series,
        symbols: pd.Series,
    ) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# linear baselines
# ---------------------------------------------------------------------------

@dataclass
class LinearRegressorModel(BaseRegressor):
    kind: str = "ridge"   # "ridge" | "lasso"
    alpha: float = 1.0
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"linear_{self.kind}"
        self._scaler = StandardScaler()
        self._model = Ridge(alpha=self.alpha) if self.kind == "ridge" else Lasso(
            alpha=self.alpha, max_iter=5000
        )
        self._train_median: pd.Series | None = None

    def fit(self, X, y, dates, symbols):
        mask = y.notna()
        X_train = X.loc[mask]
        # sklearn can't take NaN — impute with training-fold median only.
        self._train_median = X_train.median(numeric_only=True)
        Xv = X_train.fillna(self._train_median).values
        Xv = self._scaler.fit_transform(Xv)
        self._model.fit(Xv, y.loc[mask].values)
        return self

    def predict(self, X, dates, symbols):
        Xv = X.fillna(self._train_median).values
        Xv = self._scaler.transform(Xv)
        return self._model.predict(Xv)


# ---------------------------------------------------------------------------
# LightGBM regressor
# ---------------------------------------------------------------------------

@dataclass
class LGBMRegressorModel(BaseRegressor):
    n_estimators: int = 400
    num_leaves: int = 31
    learning_rate: float = 0.05
    min_child_samples: int = 50
    random_state: int = 42
    name: str = "lgbm_regressor"

    def __post_init__(self):
        self._model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            random_state=self.random_state,
            verbose=-1,
            objective="regression",
        )

    def fit(self, X, y, dates, symbols):
        min_feats = max(1, X.shape[1] // 2)
        mask = (X.notna().sum(axis=1) >= min_feats) & y.notna()
        self._model.fit(X.loc[mask].values, y.loc[mask].values)
        return self

    def predict(self, X, dates, symbols):
        return self._model.predict(X.values)


# ---------------------------------------------------------------------------
# XGBoost regressor
# ---------------------------------------------------------------------------

@dataclass
class XGBRegressorModel(BaseRegressor):
    n_estimators: int = 400
    max_depth: int = 5
    learning_rate: float = 0.05
    random_state: int = 42
    name: str = "xgb_regressor"

    def __post_init__(self):
        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            tree_method="hist",
            random_state=self.random_state,
            verbosity=0,
            objective="reg:squarederror",
        )

    def fit(self, X, y, dates, symbols):
        min_feats = max(1, X.shape[1] // 2)
        mask = (X.notna().sum(axis=1) >= min_feats) & y.notna()
        self._model.fit(X.loc[mask].values, y.loc[mask].values)
        return self

    def predict(self, X, dates, symbols):
        return self._model.predict(X.values)


# ---------------------------------------------------------------------------
# naive historical-mean baseline
# ---------------------------------------------------------------------------

@dataclass
class HistMeanBaseline(BaseRegressor):
    """Predict per-symbol in-sample mean forward return.

    Zero-skill baseline — any learned model with non-trivial macro / exposure
    signal must clear this. Falls back to the global mean for symbols unseen
    in training.
    """
    name: str = "hist_mean"
    _per_symbol: dict[str, float] = field(default_factory=dict, init=False)
    _global: float = field(default=0.0, init=False)

    def fit(self, X, y, dates, symbols):
        mask = y.notna()
        df = pd.DataFrame(
            {"sym": np.asarray(symbols)[mask.values], "y": y.loc[mask].values}
        )
        self._per_symbol = df.groupby("sym")["y"].mean().to_dict()
        self._global = float(df["y"].mean())
        return self

    def predict(self, X, dates, symbols):
        syms = np.asarray(symbols)
        return np.array([self._per_symbol.get(s, self._global) for s in syms])
