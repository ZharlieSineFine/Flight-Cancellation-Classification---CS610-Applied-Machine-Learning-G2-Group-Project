# Flight Cancellation Prediction — CS610 Applied Machine Learning (G2)

> **Can we predict whether a U.S. domestic flight will be cancelled _before_ it departs?**

This repository contains the full end-to-end machine learning pipeline for our CS610 group project: from raw data exploration through feature engineering, model training, hyperparameter tuning, and final evaluation with SHAP interpretability — all built on a **3.23 million flight** dataset (2019–2023) enriched with daily weather observations.

---


## Problem Statement

Flight cancellations are costly for airlines and disruptive for passengers, yet they are relatively rare (~2.6% of all flights). We frame this as a **binary classification** task — predicting `CANCELLED` (0 or 1) using only information available **before departure** (scheduled times, airline, route, weather forecasts).

### Issues to address

| Issue | Our Approach |
|---|---|
| **Severe class imbalance** (97.4% vs 2.6%) | Cost-sensitive learning (`class_weight='balanced'`, `scale_pos_weight`) — no SMOTE/oversampling |
| **Temporal distribution shift** (COVID-19 spike in 2020) | Strict temporal train/val/test split + `IS_COVID` ablation study |
| **Data leakage risk** (post-departure columns reveal outcome) | Explicit feature classification — only pre-departure features retained |
| **Evaluation under imbalance** | **PR-AUC** as primary metric; **F2 score** as secondary (recall-weighted) |

---

## Dataset

**Source:** Our final dataset is built on the [Initial dataset](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023), amended with the U.S. DOT flight records (September to December 2023) and appended pre-merged daily weather observations from [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api). Our final dataset is available on [Kaggle](https://www.kaggle.com/datasets/huskydawg/flight-cancellation2019-2023-full-with-weather).

| Attribute | Value |
|---|---|
| Total flights | ~3.23 million |
| Time range | January 2019 – December 2023 |
| Raw columns | 44 (32 flight + 12 weather) |
| Cancellation rate | ~2.6% |
| File format | Parquet (Git LFS for the main dataset) |

### Feature Categories

| Category | Examples | Action |
|---|---|---|
| **Target** | `CANCELLED` | Predict |
| **Pre-departure** | `FL_DATE`, `AIRLINE_CODE`, `ORIGIN`, `DEST`, `CRS_DEP_TIME`, `CRS_ELAPSED_TIME`, `DISTANCE` | Keep — engineer & encode |
| **Weather** | `origin_temp_mean_c`, `origin_snowfall_cm`, `dest_wind_speed_max_kmh`, … (12 total) | Keep — impute & scale |
| **Post-departure (leakage)** | `DEP_DELAY`, `ARR_TIME`, `CANCELLATION_CODE`, `DIVERTED`, … | **Drop** |
| **Redundant identifiers** | `AIRLINE`, `FL_NUMBER`, `ORIGIN_CITY`, … | **Drop** |
| **Engineered** | `IS_COVID` (binary: Mar 2020 – Jun 2021) | Ablation study (with/without) |

### Temporal Split

| Split | Date Range | Rows | Purpose |
|---|---|---|---|
| **Train** | 2019-01-01 → 2022-06-30 | ~2.0M | Model training (includes COVID period) |
| **Validation** | 2022-07-01 → 2022-12-31 | ~0.5M | Threshold tuning, early stopping |
| **Test** | 2023-01-01 → 2023-12-31 | ~0.7M | Final unseen evaluation |

---

## Project Pipeline

```
Raw CSV/Parquet (3.23M flights, 44 cols)
        │
        ▼
┌─────────────────────────────┐
│   Exploratory Data Analysis │  ← Saelin
│   (saelin_eda.ipynb)        │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────────┐
│   Data Engineering & Preproc    │  ← Charlie
│   (charlie_data_pipeline.ipynb) │
│                                 │
│   • Leakage-clean columns       │
│   • Cyclical time encoding      │
│   • OOF target encoding         │
│   • Median imputation           │
│   • Standard scaling            │
│   • Two versions: ±IS_COVID     │
└─────────────┬───────────────────┘
              │
    ┌─────────┼─────────┬─────────────┐
    ▼         ▼         ▼             ▼
 Baseline   LR & DT   Random      XGBoost     Neural Net
 (Route)    (Saelin)   Forest      (Malik)     MLP (Hong Yuan)
                       (Amelia)
    │         │         │             │             │
    └─────────┴─────────┴──────┬──────┴─────────────┘
                               │
                               ▼
                ┌──────────────────────────┐
                │  Final Evaluation &      │  ← Charlie
                │  SHAP Interpretability   │
                │  (charlie_final_eval…)   │
                └──────────────────────────┘
```

### Preprocessing Details

All features are preprocessed in a **leakage-safe** order (split before fit):

1. **Column filtering** — post-departure and redundant columns dropped
2. **Temporal features** — `MONTH`, `DAY_OF_WEEK`, `DEP_HOUR` → cyclical sin/cos pairs; `IS_WEEKEND` and `IS_COVID` binary flags
3. **Target encoding** — `ORIGIN`, `DEST`, `AIRLINE_CODE` encoded using 5-fold out-of-fold target encoding (fit on training data only)
4. **Imputation** — missing values filled with median (fit on training data only)
5. **Scaling** — all features standardised to mean=0, std=1 (fit on training data only)

---

## Models

All models are trained on both feature versions (`with_covid` / `no_covid`) and evaluated on the held-out 2023 test set.

### Route Baseline (Non-ML)

Historical cancellation rate lookup per ORIGIN→DEST route. No ML — just a sanity-check floor that all real models must beat.

### Logistic Regression

Linear baseline using `class_weight='balanced'` and `solver='saga'`. Regularisation strength (C) tuned on validation PR-AUC. Decision threshold tuned for F2.

### Decision Tree

First non-linear model — captures conditional patterns (e.g., "February + ORD + regional carrier = high risk"). `max_depth` tuned on validation with constraints (`min_samples_split`, `min_samples_leaf`) to control overfitting on 2M+ rows.

### Random Forest

Ensemble of decision trees with `class_weight='balanced'`. Hyperparameters tuned via `RandomizedSearchCV` (100 iterations, 3-fold stratified CV, scored on PR-AUC) on a 400K stratified sample for efficiency. Best configuration retrained on the full training set.

### XGBoost

Gradient-boosted trees with `scale_pos_weight` for imbalance handling. Tuning explored 7+ configurations per feature version, varying:
- Tree depth (4–6), learning rate (0.02–0.05), n_estimators (1500–3000)
- Regularisation (γ, L1/α, L2/λ), subsampling ratios
- Class weight multipliers (0.8×, 1×, 1.2× of the default ratio)

All experiments logged in CSV trackers with full parameter snapshots.

### Neural Network / MLP

Keras MLP with BatchNorm and Dropout. 10 configurations tested across hidden layer architectures (`[64,32]` to `[512,256,128]`), optimizers (Adam, AdamW), learning rate schedules (constant, exponential decay), regularisation strategies (uniform dropout vs descending dropout), and batch sizes. Early stopping on validation PR-AUC.

---

## Results

Final test-set performance (2023 data), ranked by **PR-AUC**:

| Model | Dataset | PR-AUC | F2 | ROC-AUC | Precision | Recall |
|---|---|---|---|---|---|---|
| Neural Network | with_covid | **0.0775** | 0.200 | 0.715 | 0.080 | 0.321 |
| Random Forest | with_covid | 0.0742 | 0.192 | 0.695 | 0.074 | 0.320 |
| XGBoost | with_covid | 0.0739 | 0.184 | 0.709 | 0.066 | 0.332 |
| Decision Tree | with_covid | 0.0591 | 0.169 | 0.700 | 0.060 | 0.306 |
| XGBoost | no_covid | 0.0506 | 0.131 | 0.655 | 0.038 | 0.342 |
| Neural Network | no_covid | 0.0494 | 0.141 | 0.646 | 0.044 | 0.320 |
| Random Forest | no_covid | 0.0473 | 0.140 | 0.656 | 0.043 | 0.329 |
| Logistic Regression | with_covid | 0.0450 | 0.098 | 0.580 | 0.023 | 0.506 |
| Logistic Regression | no_covid | 0.0408 | 0.090 | 0.550 | 0.021 | 0.494 |
| Decision Tree | no_covid | 0.0363 | 0.133 | 0.600 | 0.035 | 0.439 |
| Route Baseline | — | 0.0297 | 0.116 | 0.601 | 0.028 | 0.561 |

### Key Takeaways

- **All ML models beat the route baseline**, confirming that pre-departure features carry genuine predictive signal.
- **`with_covid` models consistently outperform `no_covid`** — the `IS_COVID` flag helps, especially for tree-based models.
- **Neural Network (with_covid)** achieved the highest PR-AUC (0.0775), though the margin over Random Forest and XGBoost is narrow.
- **PR-AUC values are low in absolute terms** — this reflects the inherent difficulty of predicting a ~2.6% event from pre-departure information alone.
