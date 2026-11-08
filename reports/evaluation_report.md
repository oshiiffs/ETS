# Model Evaluation Report — SOP 3

## Result

No accuracy evaluation is reported. The only supplied CSV has no verifiable provenance, no event timestamps to confirm its `waiting_time_minutes`, and no historical counter state. Treating its rows as real measurements would violate the project's data-integrity rules. Thus there is no valid held-out ticket set and no defensible MAE, RMSE, tolerance percentage, signed error, queue-group MAE, per-day MAE, or prediction-interval coverage to report.

## Candidate status

| Candidate | Status | Reason |
|---|---|---|
| Constant 2-minute baseline | Not scored | Requires verified actual wait outcomes. |
| Existing FIFO engine | Not replayed | Historical active counter state, residual transaction durations, and closures are absent. |
| Calibrated FIFO | Not fit | No verified training targets or faithful engine estimates. |
| Additional regression/ML | Not fit | No defensible training set; live queue state and provenance are unverified. |

Every one of the 1,028 source rows is excluded from accuracy reporting. The `reports/per_ticket_predictions.csv` deliverable is a header-only schema, not evaluation output. No model JSON is exported or described as production-ready. The lightweight Python predictor is an integration reference only and has no empirically learned coefficients.

## Acceptance decision

No new method meets acceptance criteria 1–4. In particular, 0 verified records at 10+ people ahead means the long-queue criterion is unverified. Retain the existing FIFO engine as the operational baseline. Its replacement should be considered only after a chronological complete-day holdout demonstrates lower MAE than both it and the constant baseline, no worse MAE at 10+ people ahead, and acceptable deployment behavior.

## Reproducible evaluation procedure after data collection

1. Validate raw observations with `src/preprocess.py`; preserve invalid rows with flags.
2. Produce FIFO estimates by replaying exact prediction-time counter state. Do not approximate absent state and call it production replay.
3. Supply a validated file with `date`, `actual_wait_min`, `people_ahead`, and `fifo_prediction_min` to `src/evaluate.py`.
4. The script splits whole dates chronologically, fits a simple affine calibration on training dates, evaluates the same held-out rows for all candidates, and builds a nominal 80% residual interval from training residuals only.
5. Interpret the held-out interval coverage as an observed result; the nominal 80% target is not guaranteed. Add a separate CSV and report only after valid observations exist.

At least three distinct verified observation days are required by the collection plan; a larger history is strongly preferable for day-level holdout and calibration.
