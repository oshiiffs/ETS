# Capstone Research Summary — SOP 3

## Verified findings

The folder contains a single 1,028-row CSV with 14 calendar dates. The file includes reported waiting/service durations, `people_ahead`, and a count of active windows, but contains no call/completion timestamps, record-origin field, or historical cashier-state log. One service value is missing, 16 rows show zero service time, and `people_ahead` never exceeds 7. The supplied timestamps parse and match their date column. The source's real/synthetic origin cannot be verified.

## Experimental results

No training or accuracy experiment was validly performed. No MAE/RMSE or held-out performance claim is made. No estimator can be shown to outperform the constant or existing engine on measured real-world tickets from the available evidence.

## Assumptions and limitations

`waiting_time_minutes` and `service_time_minutes` are treated as unverified supplied values, not measured ground truth. Distinct dates are not assumed to be independently verified observation days. A rough FIFO reference can accept live state, but estimating residual duration as usual service duration minus elapsed time is an explicit approximation, and the production system's cashier closures cannot be faithfully replayed without logged schedules/state.

## Recommendation

Retain the existing FIFO queue engine. Collect at least three separate days of provenance-marked observations (including a busy day), with actual ticket, call, and completion timestamps and measured queue lengths. Log full per-counter state and closures at prediction time. Then compare the constant 2-minute baseline, faithful FIFO replay, and training-only calibrated FIFO on identical tickets from later complete days. Report the 10+ queue group only after enough such observations exist. Export a model JSON only if all predeclared acceptance criteria pass.
