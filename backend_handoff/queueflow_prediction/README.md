# QueueFlow backend prediction package

Copy the entire `queueflow_prediction` folder into the other repository. Keep its internal structure unchanged. It contains the prediction code, FIFO simulator, and trained model artifacts.

## Install dependencies

From the destination repository root:

```powershell
python -m pip install -r queueflow_prediction/requirements.txt
```

Use Python 3.10 or newer. The scikit-learn version is pinned to the version recorded when these models were trained, which helps avoid model serialization incompatibility.

## Service-time estimate

```python
from queueflow_prediction.src.predict import predict_service_time

seconds = predict_service_time(
    department="CCS",
    transaction_time="10:30",
    day_of_week=0,  # Monday=0 through Sunday=6
)
```

This returns predicted service duration in seconds.

## Department-aware FIFO waiting estimate

```python
from queueflow_prediction.src.predict import predict_waiting_time

seconds = predict_waiting_time(
    queue_students=[{"department": "CCS"}, {"department": "CBM"}],
    active_cashier_windows=2,
    busy_cashier_states=[{"residual_seconds": 75}],
    transaction_time="10:30",
    day_of_week=0,
)
```

`queue_students` must be ordered from the front of the queue. Each entry needs a supported `department` code. Supply one busy state per busy window, using the remaining service duration in `residual_seconds`; idle windows need no state entry. The function returns `None` when no windows are active. The current interface does not account for future closures or new arrivals.

Supported department codes are listed in `models/model_metadata.json`. Unknown or missing codes raise `ValueError`.

## Artifacts

- `src/predict.py`: prediction API and model loading.
- `src/queue_simulator.py`: FIFO queue calculation.
- `models/ets_model.joblib`: service-time model required by the predictions above.
- `models/ewt_model.joblib`: optional experimental direct waiting-time model, available through `predict_ewt_regression`.
- `models/model_metadata.json`: model version, supported departments, and recorded evaluation summary.

The model data includes generated records and records whose real/synthetic origin could not be verified. Treat predictions as a prototype until checked against real backend observations.
