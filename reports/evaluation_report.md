# QueueFlow model training report

Training date: 2026-11-09T07:40:25.551922+08:00

## Data quality and provenance

- Combined CSV: 2028 rows; 2027 usable records with observed department codes.
- Audit-only origins: `{'original_unverified': 1027, 'generated': 1000}`. Matching generated transactions remain in training.
- All observed codes were retained and normalized to uppercase: `['BSA', 'BSHM', 'CAF', 'CAS', 'CBM', 'CCJE', 'CCS', 'CHS', 'COE', 'CTED', 'ELEMENTARY', 'GRADE 1', 'GRADE 9', 'GRADUATE STUDIES', 'JHS']`. No department aliases were inferred.
- Invalid or missing target/timestamp/department rows: 1; exact duplicate rows removed: 0.
- All valid nonnegative service durations retained, including the 8-minute maximum. No records are independently verified real observations.

## ETS validation comparison

| Model | Validation MAE (min) | RMSE (min) | R² | n |
|---|---:|---:|---:|---:|
| Global median | 0.761 | 0.978 | -0.002 | 285 |
| Department median | 0.762 | 0.995 | -0.036 | 285 |
| RandomForest | 0.784 | 1.016 | -0.081 | 285 |
| GradientBoosting | 0.742 | 0.985 | -0.016 | 285 |
| HistGradientBoosting | 0.779 | 1.003 | -0.054 | 285 |
| XGBoost | 0.758 | 0.985 | -0.015 | 285 |

Selected ML candidate: **GradientBoosting** by chronological validation MAE. It improves validation MAE slightly over both medians; test MAE is **0.673 min** (n=355). On the generated-only test block, global median MAE is 0.700 min and department median MAE is 0.685 min. The advantage is small and test R² is below zero, so evidence for useful generalization is weak.

## EWT experiment

EWT regressor validation MAE: 1.187 min; test MAE: 1.323 min. The uniform-service FIFO baseline test MAE is 0.641 min, materially better on this split. EWT labels are unverified; queue composition and busy-state estimates in the extended file are simulated. Keep ML EWT experimental and use FIFO simulation operationally.

## Extended dataset inspection

`queueflow_extended_1000.csv` has 1,000 simulated rows and 22 columns. It adds service start/end timestamps, busy window count, ordered departments ahead, workload ahead, and remaining busy-window durations. The ordered department list is missing for 423 rows. These generated scenarios were not merged into combined data or treated as independent ETS samples.

## Interpretation

All reported test records are generated; the original records are provenance-unverified, not independently verified real observations. Metrics describe supplied data, not verified real-world accuracy. The model artifact is refit on all 2,027 eligible combined rows after holdout scoring, including generated records; only the six required department codes plus observed codes were retained, without inferred aliases.

## Retraining

Run `python src/train_models.py` after validating new real records and updating a reviewed training manifest. Current script preserves origin for audit and excludes it from model features. Archive model artifacts and compare new candidates against a separate real-only chronological holdout before deployment.
