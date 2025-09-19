"""Assemble the full feature matrix from cached OHLCV + macro.

Inputs:
    data/raw/sp500_ohlcv_<start>_<end>.parquet  (from download_data.py)
    data/raw/macro_<start>_<end>.parquet        (from download_macro.py)

Output:
    data/processed/features_<start>_<end>_h<horizon>[_residual].parquet

Usage:
    python scripts/build_features.py                     # newest raw files, h=21
    python scripts/build_features.py --horizon 21 --residual
    python scripts/build_features.py --ohlcv data/raw/foo.parquet --macro data/raw/bar.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mlr.features import build_feature_matrix, fetch_sp500_sectors, model_feature_cols

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
SECTOR_CACHE = RAW_DIR / "sp500_sectors.csv"


def pick_latest(prefix: str) -> Path:
    candidates = sorted(RAW_DIR.glob(f"{prefix}*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No {prefix}*.parquet in {RAW_DIR}.")
    return candidates[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ohlcv", type=Path, default=None)
    ap.add_argument("--macro", type=Path, default=None)
    ap.add_argument("--horizon", type=int, default=21, help="forward-return horizon in trading days")
    ap.add_argument("--residual", action="store_true", help="also emit fwd_ret_Nd_residual (stock minus market)")
    ap.add_argument("--min-coverage", type=float, default=0.5,
                    help="drop rows with fewer than this fraction of model features present")
    args = ap.parse_args()

    ohlcv_path = args.ohlcv or pick_latest("sp500_ohlcv_")
    macro_path = args.macro or pick_latest("macro_")
    print(f"OHLCV  : {ohlcv_path}")
    print(f"Macro  : {macro_path}")

    ohlcv = pd.read_parquet(ohlcv_path)
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    close = ohlcv.pivot(index="date", columns="symbol", values="close").sort_index()
    volume = ohlcv.pivot(index="date", columns="symbol", values="volume").sort_index()
    print(f"  wide panel: {close.shape[0]} dates · {close.shape[1]} symbols")

    macro = pd.read_parquet(macro_path)
    macro.index = pd.to_datetime(macro.index)

    print("Fetching GICS sector map (cached)...")
    sectors = fetch_sp500_sectors(cache_path=SECTOR_CACHE)
    print(f"  {len(sectors)} rows · {sectors['sector'].nunique()} sectors")

    print(f"Assembling feature matrix (horizon={args.horizon}, residualize={args.residual})...")
    features = build_feature_matrix(
        close=close,
        volume=volume,
        macro=macro,
        sectors=sectors,
        horizon=args.horizon,
        residualize=args.residual,
    )

    # Drop rows with too few features — learned from ml-cross-sectional bug
    # where `how='all'` leaked label-only rows into training.
    feat_cols = model_feature_cols(features)
    min_feats = max(1, int(args.min_coverage * len(feat_cols)))
    before = len(features)
    features = features.dropna(subset=feat_cols, thresh=min_feats)
    dropped = before - len(features)
    print(f"  {len(features):,} rows ({dropped:,} dropped: <{min_feats}/{len(feat_cols)} features)")

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    stem = ohlcv_path.stem.replace("sp500_ohlcv_", "")
    suffix = "_residual" if args.residual else ""
    out = PROC_DIR / f"features_{stem}_h{args.horizon}{suffix}.parquet"
    features.to_parquet(out, index=False)
    print(f"Wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
