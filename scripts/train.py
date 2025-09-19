"""Walk-forward training loop for the regression models.

For each OOS year in [first, last], train on all prior data and predict the
OOS year. Writes one long-format parquet with columns:
    date, symbol, y_true, pred_<model_1>, pred_<model_2>, ...

Usage:
    python scripts/train.py                            # newest features, OOS 2020-2024
    python scripts/train.py --first 2021 --last 2024
    python scripts/train.py --features data/processed/features_....parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mlr.features import model_feature_cols
from mlr.model import (
    BaseRegressor,
    HistMeanBaseline,
    LGBMRegressorModel,
    LinearRegressorModel,
    XGBRegressorModel,
)
from mlr.validation import walk_forward_years

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = ROOT / "reports" / "predictions"

TARGET_COL = "fwd_ret_21d"


def build_model_roster() -> list[BaseRegressor]:
    return [
        LinearRegressorModel(kind="ridge", alpha=1.0),
        LinearRegressorModel(kind="lasso", alpha=1e-4),
        LGBMRegressorModel(),
        XGBRegressorModel(),
        HistMeanBaseline(),
    ]


def pick_latest_features() -> Path:
    candidates = sorted(PROC_DIR.glob("features_*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No processed feature files in {PROC_DIR}")
    return candidates[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--first", type=int, default=2020, help="first OOS year (inclusive)")
    ap.add_argument("--last", type=int, default=2024, help="last OOS year (inclusive)")
    ap.add_argument("--target", default=TARGET_COL)
    args = ap.parse_args()

    features_path = args.features or pick_latest_features()
    print(f"Loading {features_path}")
    df = pd.read_parquet(features_path)
    df["date"] = pd.to_datetime(df["date"])

    feat_cols = model_feature_cols(df)
    X = df[feat_cols]
    y = df[args.target]
    dates = df["date"]
    symbols = df["symbol"]
    print(f"  {len(df):,} rows · {len(feat_cols)} features · target={args.target}")
    print(f"  y-NaN: {y.isna().sum():,} (end-of-panel / missing forwards)")

    results = df[["date", "symbol", args.target]].rename(columns={args.target: "y_true"})
    results = results.reset_index(drop=True)

    for fold_year, train_mask, test_mask in walk_forward_years(dates, args.first, args.last):
        n_tr = train_mask.sum()
        n_te = test_mask.sum()
        y_tr_valid = y[train_mask].notna().sum()
        print(f"\nFold {fold_year}: train={n_tr:,} ({y_tr_valid:,} labelled)  test={n_te:,}")

        Xtr, ytr, dtr, str_ = (
            X.loc[train_mask], y.loc[train_mask],
            dates.loc[train_mask], symbols.loc[train_mask],
        )
        Xte, yte, dte, ste = (
            X.loc[test_mask], y.loc[test_mask],
            dates.loc[test_mask], symbols.loc[test_mask],
        )

        for model in build_model_roster():
            col = f"pred_{model.name}"
            if col not in results.columns:
                results[col] = pd.NA

            model.fit(Xtr, ytr, dtr, str_)
            preds = model.predict(Xte, dte, ste)
            results.loc[test_mask, col] = preds
            # quick in-fold sanity: MAE vs target mean |y|
            mask = yte.notna().values
            if mask.any():
                mae = (abs(preds[mask] - yte.values[mask])).mean()
                ir = pd.Series(preds[mask]).corr(pd.Series(yte.values[mask]))
                print(f"  {model.name:20s}  MAE={mae:.4f}  pearson={ir:+.3f}")

    # Drop rows with no OOS predictions (every year outside the requested range).
    pred_cols = [c for c in results.columns if c.startswith("pred_")]
    oos_mask = results[pred_cols].notna().any(axis=1)
    results = results.loc[oos_mask]

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = PRED_DIR / f"oos_{args.first}_{args.last}.parquet"
    results.to_parquet(out, index=False)
    print(f"\nWrote {out}  ({len(results):,} rows)")


if __name__ == "__main__":
    main()
