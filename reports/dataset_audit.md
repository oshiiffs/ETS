# Dataset Audit Report

Audit date: 2026-11-08 (Asia/Manila)

## Inventory and provenance

The project directory contains one dataset file, `Datasetforqueueflow.csv` (112,873 bytes), and an empty `dataset/` directory. No Excel, JSON, database, or other source dataset was present. The CSV is preserved at `data/raw/Datasetforqueueflow.csv`.

The CSV has 1,028 rows and 16 columns: `date`, `transaction_number`, `cashier_window`, `department`, `ticket_creation_datetime`, `waiting_time_minutes`, `service_time_minutes`, `day_of_week`, `hour`, `minute`, `is_weekday`, `hour_sin`, `hour_cos`, `people_ahead`, `queue_length`, and `active_cashier_windows`.

Dates span 2026-10-02 through 2026-10-19, across 14 distinct calendar dates. Counts per date range from 7 to 205. These are distinct dates in a file, not verified observation days: there is no observation log, collector identity, or origin marker. There is no `record_origin` field. The file's provenance and whether rows are real, synthetic, or mixed cannot be verified. No row is certified as a real cashier observation; none is used as measured ground truth or in reported accuracy.

## Fields and quality checks

| Field group | Findings |
|---|---|
| Ticket timestamp | 1,028/1,028 parse as timestamps; `date` agrees with timestamp date on all rows. |
| Waiting duration | Present in all 1,028 rows; range 0–45 min; mean 3.477 min; median 2 min. It has no called timestamp, so the value cannot be independently recomputed or verified as actual elapsed waiting. |
| Service duration | 1,027 values present; range 0–8 min; mean 1.868 min; median 2 min. One missing value; no start/finish timestamps are supplied to verify it. Sixteen rows report 0 min, which needs source clarification. |
| Queue length | `people_ahead` is populated for all rows, but its measurement method and capture time are undocumented. Counts: 749 rows at 0, 271 at 1–4, 8 at 5–9, and 0 at 10+. It cannot be certified as an actual measured queue count. |
| Counters | `active_cashier_windows` values range 3–5. There is no counter state history, busy/idle state, remaining service duration, or closure schedule. |
| Duplicates | No exact duplicate rows. `transaction_number` repeats in 200 rows; likely ticket-number reuse is possible, but no unique ticket key is documented. |
| Outliers | Waiting duration maximum is 45 min and is retained for source review; it is not removed as an outlier. |
| Derived columns | Day-of-week, hour, minute, weekday flag, cyclic time, and queue length can be derived from other values. Their consistency beyond the date check has not been established. Department and cyclic time are not permitted model inputs for this project. |

No timestamp pairs exist for `called_time` and `finished_time`; therefore actual waiting and service times cannot be recomputed. The supplied numeric duration fields are unverified outcomes, not timestamps. All rows remain preserved in the raw copy; missing values and suspicious values are not filled.

## Suitability and decision

This dataset is not suitable for training or a defensible accuracy evaluation. Record origin, measured outcomes, and the relationship between queue counts and live prediction time are unknown. It also lacks the counter-state and closure history required to reproduce the FIFO engine. There are zero verifiable real observation days and zero verifiable measured ticket outcomes. Although 14 dates appear in the file, this does not meet the minimum of three *verified* observation days.

The 0–7 queue-count distribution also contains no examples at 10 or more people ahead. The acceptance requirement for long queues would remain unverified even if row origin were later established.

## Required collection

Use `data/observation_template.csv` for manual or automatic logs. Preserve one row per ticket and record a unique ticket ID, date, window, ticket creation time, actual people ahead at that time, counters open, actual call time, actual completion time, predicted estimate/model version if available, and an origin value (`manual_observation` or `system_log`). Log counter busy/idle status and elapsed or remaining transaction time at prediction time, plus scheduled closures/lunch intervals, to support faithful engine replay. Collect at least three separate observed days, including a busy day, and retain raw exports unchanged. Do not populate missing queue counts from ticket number or inference.
