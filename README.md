# DRW - Crypto Market Prediction

Predicting short-term cryptocurrency price movements from proprietary trading features and public market data, for the [DRW - Crypto Market Prediction](https://www.kaggle.com/competitions/drw-crypto-market-prediction) Kaggle competition hosted by DRW Trading (Cumberland).

**Result: Top 100 of ~1,000 teams** on the private leaderboard.

---

## The Problem

DRW provided a year of minute-level crypto market data and challenged competitors to turn it into a single directional signal predicting future price movements. The defining difficulty is an extremely **low signal-to-noise ratio**. Price moves are driven by a tangle of liquidity, order flow, and structural effects, with no single feature carrying much predictive power. Submissions are scored by **Pearson correlation** between predictions and the true label.

## The Data

| | Rows | Columns | Period |
|---|---|---|---|
| Train | 525,886 | 786 | 2023-03-01 → 2024-02-29 (minute-level) |
| Test | 538,150 | 786 | After train; timestamps masked & shuffled |

- **5 microstructure features**: `bid_qty`, `ask_qty`, `buy_qty`, `sell_qty`, `volume`
- **780 anonymized proprietary features**: `X1` … `X780`
- **`label`**: anonymized price movement (mean ≈ 0.04, std ≈ 1.01, ~51% positive, heavy-tailed)
- No missing values in either split.

## Approach

### 1. Exploratory analysis: quantifying the noise

The EDA established just how weak the signal is, which shaped every later decision:

- **No single feature is strongly predictive.** Not one of the 780 X-features exceeds |correlation| 0.1 with the label; the strongest, `X752`, sits at just **0.0906**. Only 32 features clear |corr| > 0.05.
- **Heavy redundancy in the feature block.** Of ~304k feature pairs, 459 have |r| > 0.95 and 112 have |r| > 0.99, including many *perfect duplicates* at r = 1.00. The 780 dimensions carry far less than 780 dimensions' worth of information.
- **Derived microstructure signals** (order imbalance, bid/ask imbalance) correlated near-zero with the label.
- No constant, near-constant, or sparse columns to clean.

### 2. Feature engineering: cutting redundancy

Given the redundancy above, the 780 X-features were collapsed to a compact, less-collinear set:

- **Correlation clustering**: hierarchical clustering with complete linkage on distance `1 − |corr|`, cut so every within-cluster pair has |r| ≥ 0.6. Each cluster is represented by its **medoid** (the most central member). This reduced 780 → **109 representatives (14%)**.
- **Target relevance filter**: dropped representatives with |corr(feature, label)| ≤ 0.01, leaving **69**.
- **Final feature set: 74** (69 representatives + 5 microstructure features).

### 3. Modeling

Three models were built on the same out-of-fold + test-bagging structure so their scores are directly comparable.

| # | Model | Validation | Notes |
|---|---|---|---|
| 1 | **Base XGBoost** | 5-fold `KFold` | Tuned gradient-boosted trees; baseline only, since plain K-fold *leaks* on this autocorrelated series |
| 2 | **Purged-CV XGBoost** | `PurgedGroupTimeSeriesSplit` | 6 time-ordered groups (~2 months each), 1 group purged on each side to kill AR(1) boundary leakage |
| 3 | **Ridge regression** | Same purged CV | `StandardScaler` fit per fold + `RidgeCV` (leave-one-out α selection) |

The move from plain K-fold to a **purged, grouped time-series split** was the key methodological fix: the series is strongly autocorrelated (φ ≈ 0.98), so naive cross-validation trains on the future and produces optimistic scores that don't track the leaderboard.

## Results

**Cross-validation (purged OOF Pearson):**

| Model | OOF Pearson |
|---|---|
| Purged-CV XGBoost | **0.1002** |
| Ridge regression | 0.0940 |

**Final leaderboard submissions:**

| Submission | Model | Private | Public |
|---|---|---|---|
| `submission_ridge.csv` | Ridge regression | **0.08737** | 0.06954 |
| `submission2.csv` | Purged-CV XGBoost | 0.07465 | 0.07842 |
| `submission.csv` | Base XGBoost | 0.07186 | 0.07909 |

### Key takeaway: the public/private flip

The Ridge submission had the **lowest public score but the highest private score**. The XGBoost models looked stronger on the public leaderboard, then fell behind on the private set. In a regime this noisy, the simple, heavily-regularized linear model generalized best to unseen future data, while the more expressive trees overfit signal that wasn't really there. Trusting robust cross-validation over the public leaderboard is what held up when it counted.

## Repository Structure

```
.
├── DRW_competition.ipynb     # Full pipeline: EDA → feature engineering → modeling → submissions
├── README.md
└── environment.yml           # Reproducible conda environment
```

## Reproducing

> **Data is not included.** Kaggle competition data can't be redistributed, so download `train.parquet`, `test.parquet`, and `sample_submission.csv` from the [competition data page](https://www.kaggle.com/competitions/drw-crypto-market-prediction/data) and place them in the repo root.


Core dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `scipy`, `shap`, `matplotlib`, `seaborn`, `pyarrow`.

## What I'd Explore Next

- **Blend** the linear and tree predictions: the OOF arrays are kept around precisely so Ridge and XGBoost can be combined, and they make uncorrelated errors.
- **Normalize the microstructure features**, which show a meaningful train→test distribution shift.
- **Sample-weighting toward recent data**, since the competition hints that relevance decays over the training window.
