## Temporal Split

| Split     | Date Range               | Purpose                                |
| --------- | ------------------------ | -------------------------------------- |
| **Train** | 2019-01-01 to 2022-06-30 | Model training (includes COVID period) |
| **Val**   | 2022-07-01 to 2022-12-31 | Threshold tuning, early stopping       |
| **Test**  | 2023-01-01 to 2023-12-31 | Final unseen evaluation                |

## Files

### Feature matrices (two versions for COVID ablation study)

| File                         | Description                                         |
| ---------------------------- | --------------------------------------------------- |
| `X_train_with_covid.parquet` | Training features, 25 columns (includes `IS_COVID`) |
| `X_val_with_covid.parquet`   | Validation features, 25 columns                     |
| `X_test_with_covid.parquet`  | Test features, 25 columns                           |
| `X_train_no_covid.parquet`   | Training features, 24 columns (no `IS_COVID`)       |
| `X_val_no_covid.parquet`     | Validation features, 24 columns                     |
| `X_test_no_covid.parquet`    | Test features, 24 columns                           |

### Target arrays (shared across both versions)

| File              | Description                                   |
| ----------------- | --------------------------------------------- |
| `y_train.parquet` | Training labels — column `CANCELLED` (0 or 1) |
| `y_val.parquet`   | Validation labels                             |
| `y_test.parquet`  | Test labels                                   |

## How to use

```python
import pandas as pd
import joblib

# Pick a version
X_train = pd.read_parquet('artifacts/X_train_with_covid.parquet')
X_val   = pd.read_parquet('artifacts/X_val_with_covid.parquet')
X_test  = pd.read_parquet('artifacts/X_test_with_covid.parquet')

y_train = pd.read_parquet('artifacts/y_train.parquet')['CANCELLED'].values
y_val   = pd.read_parquet('artifacts/y_val.parquet')['CANCELLED'].values
y_test  = pd.read_parquet('artifacts/y_test.parquet')['CANCELLED'].values

```

## Preprocessing applied

All features have been:

1. **Leakage-cleaned** — post-departure columns dropped
2. **Target-encoded** — `ORIGIN`, `DEST`, `AIRLINE_CODE` encoded using out-of-fold (5-fold) target encoding on training data
3. **Temporally engineered** — `MONTH`, `DAY_OF_WEEK`, `DEP_HOUR` converted to cyclical sin/cos pairs; `IS_WEEKEND` and `IS_COVID` binary flags derived
4. **Imputed** — missing values filled with median (fit on training data only)
5. **Scaled** — all features standardised to mean=0, std=1 (fit on training data only)
