# ml-return-forecast

> **Languages**: [English](../README.md) · **繁體中文**

[`ml-cross-sectional`](https://github.com/matthiola0/ml-cross-sectional) 學的是 cross-sectional 排序；這個 repo 我做進階版，嘗試預測 *絕對* 21 日 forward return —— 強迫模型不只判斷哪些股票贏過 median，還要知道市場往哪去。答案大致是 *no*，而失敗的方式才是有趣的部分。

Feature pipeline 從頭重建：去掉 cross-sectional z-scoring、加上 macro 與 beta 特徵、target 改成連續 21 日報酬。底層指標跟 ranking repo 重疊，其餘都不同。

## 研究問題

`ml-cross-sectional` 已證明從價量特徵就能學到 cross-sectional *排序*，因為相對比較把市場方向 net out。我問的是更難的版本：相同特徵能不能預測 *絕對* forward return？這要求模型知道 **市場將往哪去**，不只是 **哪些股票贏過 median** —— 所以特徵設計必須改。Notebook 04 / 05 的答案是 *勉強能、且 portfolio construction 性質比 ranker 差*。

## Headline 結果

| 模型 | OOS MAE | OOS Pearson |
|---|---:|---:|
| linear_ridge | 0.079 | +0.09 |
| linear_lasso | 0.078 | +0.10 |
| lgbm_regressor | 0.089 | +0.13 |
| xgb_regressor | 0.092 | +0.11 |
| hist_mean | 0.074 | +0.00 |

_完整 OOS 2020-2024、所有 S&P 500 名字。`hist_mean` 是 per-symbol 訓練期均值 —— 注意它在 MAE **勝出**。任何 learned model 必須以 Pearson / IC 評斷而非 MAE，因為 target heavy-tailed 程度足以讓常數預測獲得很強的絕對誤差。_

**vs 排序（notebook 05）**：每月分別從 `xgb_regressor` 與 `ml-cross-sectional` 的 `xgb_ranker` 挑 top-20，60 次 rebalance 平均 Jaccard 重合僅 **≈ 0.19** —— 約 3-4 檔重合 / 20 檔。Regressor top-20 在 2022 drawdown 較深（−29.6% vs −22.7% intra-year），但 2020-2024 全期兩條 equity curve 年年互換領先 —— 實測 ranker basket beta（1.43）反而 *略高* 於 regressor（1.32），所以 *ranking 較 robust* 只成立於 2022 regime + 結構性論證 (a)，不是普遍 beta-concentration 故事。

## 方法

- **股票池**：當前 S&P 500（502 檔），2015-01 到 2025-07。承認 survivorship —— 結果是上界。
- **Target**：`fwd_ret_21d = close[t+21] / close[t] - 1`，原始連續值。
- **特徵（33 欄）**：
  - **Stock (11)**：`mom_12_1`、`reversal_1w`、`ret_{21,63,126,252}d`、`vol_{20,60}d`、`rsi_14`、`macd_hist`、`volume_z_60`。
  - **Macro (10)，全 lag 1 個交易日**：VIX level + 20 日變化、10Y 殖利率 + 20 日變化、term slope (10Y−2Y)、BAA 信用利差（Moody's BAA − 10Y）、S&P 3M / 12M trailing return + 60 日 vol、6 個月 fed-funds 變動次數。
  - **Exposure (12)**：252 日 rolling beta vs ^GSPC、11 個 GICS sector dummies。
- **模型**：Ridge / Lasso（標準化 + 中位數補值）；LightGBM & XGBoost regressor（RMSE loss）；per-symbol HistMean 當 zero-skill bar。
- **驗證**：年度 expanding-window walk-forward、OOS 2020-2024。
- **評估**：MAE / RMSE / direction accuracy / Pearson / Spearman IC；threshold-long 策略含 5 bps 單邊成本；vs `ml-cross-sectional` 的 Jaccard + signal correlation。

## Notebook 導覽

| # | Notebook | 內容 |
|---|---|---|
| 01 | [`01_regression_eda.ipynb`](../notebooks/01_regression_eda.ipynb) | Target σ ≈ 0.08、fat tails、per-stock R² vs market ≈ 0.3 —— 為什麼 macro 重要 |
| 02 | [`02_training_walkforward.ipynb`](../notebooks/02_training_walkforward.ipynb) | 跨模型表 + 逐年 + MAE vs VIX regime |
| 03 | [`03_error_analysis.ipynb`](../notebooks/03_error_analysis.ipynb) | per-sector MAE、高 / 低 VIX 切片、最差 20 個預測 |
| 04 | [`04_threshold_strategy.ipynb`](../notebooks/04_threshold_strategy.ipynb) | 當 `pred > τ` 做多；τ sweep；淨值 vs SPX |
| 05 | [`05_vs_ranking.ipynb`](../notebooks/05_vs_ranking.ipynb) | **與 `ml-cross-sectional` 正面對照**：daily Spearman、top-20 Jaccard、drawdown 行為 |

## 失效模式

絕對報酬迴歸相對於 ranking 有 3 個結構性劣勢：

1. **市場 beta 主導 target**。中位股票 21 日報酬對同期市場報酬的 R² ≈ 0.3。沒明確帶 macro / beta 特徵的模型在學市場而非個股；帶了又繼承 macro look-ahead 風險。
2. **Target 分布 fat-tailed**。Squared-loss regressor over-fit 異常月份（COVID、2022）。逐 fold MAE 隨 regime 擺動 40% —— 看 notebook 02 逐年表，不是 headline 那一行。
3. **Threshold 打不過 quantile**。Notebook 04 的 τ sweep 不單調改善：top 預測股票並非可靠優於所有正預測股票，因為 regressor 給的 magnitude 雜訊大。正確的 ranker（`ml-cross-sectional`）改用 top-quintile / long-short，結構上 robust。

合起來，這就是業界俚語 *signal research is dominated by ranking* 的數值版本 —— 做這個 repo 就是要把它量化。

## 限制

- **存活者偏差**：股票池是 *當前* S&P 500，2015-2024 期間下市或剔除的不可見。
- **信用利差替代**：FRED 公開 CSV `BAMLH0A0HYM2`（HY OAS）授權變更後只回傳 ~2 年；改用 `BAA10Y`（Moody's BAA − 10Y）—— 合理 IG-spread proxy，覆蓋整個視窗。
- **Macro look-ahead**：每個 macro series 都 lag 1 個交易日。部分 series（如 `FEDFUNDS`）是月頻 forward-fill —— look-ahead guard 保守但非滴水不漏。
- **Timing convention**：預測假設 t 日收盤執行（同日 close-to-close frame），所以 `beta_252d` 用 t 為止未 lag 的報酬。Macro series 釋出 cadence 與股價不同，加 1 個交易日 lag 作為額外安全 margin 而非為了對齊此 frame。
- **Sector snapshot**：GICS sector 為 *當前* assignment，非 point-in-time mapping。

## 重現

```bash
conda create -n ml-return-forecast python=3.13
conda activate ml-return-forecast
pip install -e .
# 註冊 kernel 確保 nbconvert 用對 env
python -m ipykernel install --user --name ml-return-forecast

# 資料（寫入 data/raw/）
python scripts/download_data.py
python scripts/download_macro.py

# 特徵（寫入 data/processed/）
python scripts/build_features.py

# 訓練 OOS 2020-2024
python scripts/train.py   # 寫 reports/predictions/oos_2020_2024.parquet

# 重新執行 notebooks
python -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=ml-return-forecast \
  --inplace notebooks/*.ipynb
```

## 結構

```
ml-return-forecast/
├── data/
│   ├── raw/            # sp500_ohlcv_*.parquet, macro_*.parquet, sp500_sectors.csv
│   └── processed/      # features_*.parquet
├── notebooks/          # 01–05，已執行
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
    ├── features.py         # 組裝 + beta + sector + target
    ├── model.py            # 4 個 wrapper class、5 個模型實例
    └── validation.py       # walk_forward_years
```

## 參考文獻

**橫斷面絕對報酬迴歸（直接 benchmark）**
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine
  learning. *Review of Financial Studies*, 33(5), 2223–2273.
  [doi:10.1093/rfs/hhaa009](https://doi.org/10.1093/rfs/hhaa009) — 用 94 個公司特徵 + 8 個 macro predictor 預測月頻美股絕對報酬，比較 linear / tree / neural model。本 repo 是同設計的縮小版（21 個個股 + 10 個 macro + 12 個 exposure 特徵、21 日 horizon、Ridge / Lasso / LGBM / XGB），與 [`ml-cross-sectional`](https://github.com/matthiola0/ml-cross-sectional) 配對就是 GKX 沒做的 ranking-vs-regression 直接對照。

**Macro 對報酬的預測力（為什麼 macro 救不了 regression）**
- Welch, I., & Goyal, A. (2008). A comprehensive look at the empirical
  performance of equity premium prediction. *Review of Financial Studies*,
  21(4), 1455–1508.
  [doi:10.1093/rfs/hhm014](https://doi.org/10.1093/rfs/hhm014) — 發現經典 macro predictor 集合（term spread、credit spread、dividend yield 等）對 **總體** 市場 premium 幾乎沒有可靠 *樣本外* 預測力。我在這個 repo 把同一池變數（VIX、10Y、term slope、BAA 信用利差、S&P trailing return/vol）當 per-stock macro 特徵餵進去 —— 橫斷面使用而非擇時。Notebook 02 逐年 MAE 擺動與 notebook 04 平緩 threshold sweep 都符合 W&G 的論點：這些 series 帶的 forward 訊息少於同期相關所暗示的程度。

**驗證方法論**
- López de Prado, M. (2018). *Advances in financial machine learning*.
  Wiley. 第 7 章主張金融 cross-validation 應用 purging + embargo（推薦 CPCV）。我用普通年度 expanding-window walk-forward **無 purging** —— 跟 `ml-cross-sectional` 同樣有意偏離，在 21 日 target + 年度重訓下 fold-to-fold IC noise 主導 leakage 影響，可接受；但 production 應該重新檢視這個選擇。
