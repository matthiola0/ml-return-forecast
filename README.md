# ml-return-forecast

> **Languages**: **English** · [繁體中文](docs/README.zh-TW.md)

I built this as the harder companion to [`ml-cross-sectional`](https://github.com/matthiola0/ml-cross-sectional). That repo learned cross-sectional rankings; this one tries to predict the *absolute* 21-day forward return — which forces the model to know where the market is going, not just which stocks beat the median. The answer turns out to be mostly no, and the way it fails is the interesting part.

I rebuilt the feature pipeline from scratch: dropped the cross-sectional z-scoring, added macro and beta features, switched the target to a continuous 21-day return. Low-level indicators overlap with the ranking repo but nothing else does.

## Research question

`ml-cross-sectional` showed that cross-sectional ranking can be learned from price-volume features alone, because relative comparison nets out market direction. The harder question is whether the same features can predict the absolute forward return. That requires knowing where the market is going, not just which stocks beat the median. Notebooks 04 and 05 say: barely, and with worse portfolio-construction properties than the ranker.

## Headline result

| Model           | OOS MAE | OOS Pearson |
|-----------------|--------:|------------:|
| linear_ridge    | 0.079   | +0.09       |
| linear_lasso    | 0.078   | +0.10       |
| lgbm_regressor  | 0.089   | +0.13       |
| xgb_regressor   | 0.092   | +0.11       |
| hist_mean       | 0.074   | +0.00       |

_Full OOS 2020-2024. `hist_mean` is a per-symbol training-period mean. Note that it **wins on MAE** — any learned model has to be judged on Pearson/IC, not MAE, because the target is fat-tailed enough that a constant prediction already has a strong absolute error._

Vs. ranking (notebook 05): picking top-20 each month from `xgb_regressor` vs `ml-cross-sectional`'s `xgb_ranker` (same model family, different target), the baskets overlap only **≈ 0.19 by Jaccard** on average over 60 rebalances — roughly 3–4 names in common out of 20. The regressor takes a deeper 2022 drawdown (−29.6% vs −22.7%). Over the full window the two equity curves alternate leadership by year, and mean basket beta is actually slightly *higher* for the ranker (1.43 vs 1.32). "Ranking is more robust" holds structurally and in 2022 specifically, but it isn't a universal beta story.

## Method

- **Universe.** Current S&P 500 constituents (502 names), 2015-01 to
  2025-07. Survivorship is acknowledged — results are an upper bound.
- **Target.** `fwd_ret_21d = close[t+21] / close[t] - 1`, raw continuous.
- **Features (33 cols).**
  - **Stock (11):** `mom_12_1`, `reversal_1w`, `ret_{21,63,126,252}d`,
    `vol_{20,60}d`, `rsi_14`, `macd_hist`, `volume_z_60`.
  - **Macro (10), all lagged one business day:** VIX level + 20d change,
    10Y yield + 20d change, term slope (10Y−2Y), BAA credit spread
    (Moody's BAA − 10Y), S&P 3M / 12M trailing return + 60d vol, 6M
    fed-funds move count.
  - **Exposure (12):** 252d rolling beta vs ^GSPC, 11 GICS sector dummies.
- **Models.** Ridge / Lasso on standardised + median-imputed features;
  LightGBM & XGBoost regressors with RMSE loss; per-symbol HistMean as the
  zero-skill bar.
- **Validation.** Annual expanding-window walk-forward, OOS 2020–2024.
- **Evaluation.** MAE / RMSE / direction accuracy / Pearson / Spearman IC;
  threshold-long strategy with 5 bps one-way costs; Jaccard + signal
  correlation vs `ml-cross-sectional`.

## Notebook tour

| # | Notebook | What it shows |
|---|---|---|
| 01 | [`01_regression_eda.ipynb`](notebooks/01_regression_eda.ipynb) | Target σ ≈ 0.08, fat tails, per-stock R² vs market ≈ 0.3 — why macro matters |
| 02 | [`02_training_walkforward.ipynb`](notebooks/02_training_walkforward.ipynb) | Cross-model table + year-by-year + MAE vs VIX regime |
| 03 | [`03_error_analysis.ipynb`](notebooks/03_error_analysis.ipynb) | Per-sector MAE, high/low-VIX split, worst 20 predictions |
| 04 | [`04_threshold_strategy.ipynb`](notebooks/04_threshold_strategy.ipynb) | Long when `pred > τ`; τ sweep; net equity curves vs SPX |
| 05 | [`05_vs_ranking.ipynb`](notebooks/05_vs_ranking.ipynb) | **Head-to-head with `ml-cross-sectional`**: daily Spearman, top-20 Jaccard, drawdown behaviour |

## Failure modes

Absolute-return regression has three structural disadvantages against ranking, and this study makes all three visible.

Market beta dominates the target: the median S&P 500 stock has R² ≈ 0.3 against the contemporaneous market return. A model without explicit macro/beta features learns the market more than the stock; one that does carry them inherits macro look-ahead risks.

Target distribution is fat-tailed. Squared-loss regressors get distorted by outlier months (COVID, 2022). Per-fold MAE swings ~40% with regime — the headline row is misleading; notebook 02's year breakdown is the honest read.

Thresholds don't beat quantiles. The τ sweep in notebook 04 doesn't improve monotonically: the top-predicted names aren't reliably better than the bulk of positively-predicted names because the regressor's "magnitude" is noisy. Quintile-based ranking is more robust by construction.

These are the quantitative version of the industry folklore that signal research is dominated by ranking. This repo exists to make that folklore measurable.

## Limitations

Universe is current S&P 500 — delisted or removed names are invisible, same survivorship caveat as `ml-cross-sectional`. Credit spread uses `BAA10Y` (Moody's BAA − 10Y) instead of the originally planned `BAMLH0A0HYM2`: FRED's public endpoint for the ICE HY OAS series was restricted to ~2 years after a licensing change, so I switched to an IG proxy that covers the full window. Macro series are lagged one business day; monthly series like `FEDFUNDS` are forward-filled, so the look-ahead guard is conservative but not airtight. GICS sector is the current assignment, not a point-in-time mapping.

## Reproducing

```bash
conda create -n ml-return-forecast python=3.13
conda activate ml-return-forecast
pip install -e .
# register the kernel so nbconvert executes notebooks in the right env
python -m ipykernel install --user --name ml-return-forecast

# data (writes to data/raw/)
python scripts/download_data.py
python scripts/download_macro.py

# features (writes to data/processed/)
python scripts/build_features.py

# train OOS 2020-2024
python scripts/train.py   # writes reports/predictions/oos_2020_2024.parquet

# re-execute notebooks
python -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=ml-return-forecast \
  --inplace notebooks/*.ipynb
```

## Layout

```
ml-return-forecast/
├── data/
│   ├── raw/            # sp500_ohlcv_*.parquet, macro_*.parquet, sp500_sectors.csv
│   └── processed/      # features_*.parquet
├── notebooks/          # 01–05, executed
├── reports/
│   └── predictions/    # oos_2020_2024.parquet
├── scripts/
│   ├── download_data.py
│   ├── download_macro.py
│   ├── build_features.py
│   └── train.py
└── src/mlr/
    ├── features_stock.py
    ├── features_macro.py
    ├── features.py         # assembly + beta + sector + target
    ├── model.py            # 4 wrapper classes, 5 model instantiations
    └── validation.py       # walk_forward_years
```

## References

**Cross-sectional absolute-return regression (direct benchmark)**
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine
  learning. *Review of Financial Studies*, 33(5), 2223–2273.
  [doi:10.1093/rfs/hhaa009](https://doi.org/10.1093/rfs/hhaa009) — predicts
  absolute monthly US equity returns with 94 firm characteristics plus 8
  macro predictors, comparing linear / tree / neural models. This repo is a
  scaled-down version of the same setup (21 stock + 10 macro + 12 exposure
  features, 21-day horizon, Ridge / Lasso / LGBM / XGB), and its pairing
  with [`ml-cross-sectional`](https://github.com/matthiola0/ml-cross-sectional)
  is the direct ranking-vs-regression comparison GKX does not make
  explicitly.

**Macro predictability of returns (why macro doesn't save the regression)**
- Welch, I., & Goyal, A. (2008). A comprehensive look at the empirical
  performance of equity premium prediction. *Review of Financial Studies*,
  21(4), 1455–1508.
  [doi:10.1093/rfs/hhm014](https://doi.org/10.1093/rfs/hhm014) — finds that
  the canonical macro predictor set (term spread, credit spread, dividend
  yield, etc.) offers almost no reliable *out-of-sample* forecast power for
  the **aggregate** market premium. The variables I feed as per-stock
  macro features here (VIX, 10Y, term slope, BAA credit spread, S&P
  trailing return/vol) are drawn from the same pool. I use them
  cross-sectionally rather than to time the index, but notebook 02's
  year-by-year MAE swings and notebook 04's flat threshold sweep are
  consistent with the W&G finding that these series carry less forward
  information than their contemporaneous correlation suggests.

**Validation methodology**
- López de Prado, M. (2018). *Advances in financial machine learning*.
  Wiley. Chapter 7 argues for purging + embargo (with CPCV as the
  recommended scheme) in financial cross-validation. I use plain annual
  expanding-window walk-forward with **no purging** — the same deliberate
  deviation I made in `ml-cross-sectional`, justifiable at a 21-day target horizon and
  annual retrain where fold-to-fold IC noise dominates leakage, but a
  design choice a production setup should revisit.
