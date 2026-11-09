"""Backend-safe deterministic predictor with no third-party runtime dependencies."""
from datetime import datetime
import json
from pathlib import Path


DEFAULT_MODEL = {"model": "fifo_constant_service", "version": "0.1.0",
                 "service_multiplier": 1.0, "intercept_min": 0.0}


def load_model(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix.lower() in {".joblib", ".pkl"}:
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("Loading trained models requires joblib") from exc
        return joblib.load(path)
    with open(path, encoding="utf-8") as f:
        model = json.load(f)
    if model.get("model") != "fifo_constant_service":
        raise ValueError("Unsupported model type")
    return model


_ROOT = Path(__file__).resolve().parents[1]
_ETS_PATH = _ROOT / "models" / "ets_model.joblib"
_EWT_PATH = _ROOT / "models" / "ewt_model.joblib"
DEPARTMENT_CODES = {x: x for x in ("CCS", "CBM", "CHS", "CCJE", "COE", "CTED", "CAF", "CAS",
    "GRADUATE STUDIES", "JHS", "BSHM", "BSA", "ELEMENTARY", "GRADE 9", "GRADE 1")}


def _department(value):
    code = str(value or "").strip().upper()
    if code not in DEPARTMENT_CODES:
        raise ValueError(f"Unsupported or missing department: {value!r}")
    return code


def _load_pipeline(path):
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found: {path}. Run python src/train_models.py")
    artifact = load_model(path)
    return artifact["pipeline"]


def predict_service_time(department="CCS", transaction_time=None, day_of_week=0,
                        cashier_window="", model=None, model_path=None) -> float:
    """Predict an individual's service duration, returned in seconds."""
    import pandas as pd
    import math
    dep = _department(department)
    if isinstance(day_of_week, str):
        names = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,"saturday":5,"sunday":6}
        day_of_week = names.get(day_of_week.strip().lower(), day_of_week)
    day = int(day_of_week)
    if not 0 <= day <= 6: raise ValueError("day_of_week must be 0..6 or a weekday name")
    if transaction_time is None: hour = 12
    elif hasattr(transaction_time, "hour"): hour = int(transaction_time.hour)
    else:
        parts = str(transaction_time).split(":")
        hour = int(parts[0])
        if len(parts) > 1 and not 0 <= int(parts[1]) <= 59: raise ValueError("invalid transaction_time")
    if not 0 <= hour <= 23: raise ValueError("transaction_time hour must be 0..23")
    pipe = model or _load_pipeline(Path(model_path) if model_path else _ETS_PATH)
    row = pd.DataFrame([{"department":dep,"cashier_window":str(cashier_window or ""),"hour":hour,"day_of_week":day}])
    prediction = float(pipe.predict(row)[0])
    return max(0.0, prediction * 60.0)


def predict_waiting_time(queue_students, active_cashier_windows, busy_cashier_states=None,
                         transaction_time=None, day_of_week=0, ets_model=None) -> float | None:
    """Department-aware FIFO simulation; returns seconds, or None with no active windows.

    Busy states accept residual_seconds (preferred) or elapsed_seconds. When only
    elapsed progress is known, residual service uses predicted mean duration minus
    elapsed, a documented approximation that is not a survival-model estimate.
    """
    from .queue_simulator import simulate_department_wait_seconds
    if active_cashier_windows == 0: return None
    model = ets_model or _load_pipeline(_ETS_PATH)
    return simulate_department_wait_seconds(queue_students, active_cashier_windows,
        busy_cashier_states or [], lambda student: predict_service_time(
            student.get("department"), transaction_time, day_of_week,
            student.get("cashier_window", ""), model=model))


def predict_ewt_regression(features, model=None, model_path=None) -> float:
    """Experimental EWT estimate in seconds; operational estimates should use FIFO."""
    import pandas as pd
    pipe = model or _load_pipeline(Path(model_path) if model_path else _EWT_PATH)
    return max(0.0, float(pipe.predict(pd.DataFrame([features]))[0]) * 60.0)


def predict_wait_minutes(people_ahead: int, counters_open: int,
                         busy_elapsed_min: list[float], minutes_until_open: float,
                         service_min: float, now: datetime,
                         model: dict | None = None) -> float:
    """Return a deterministic queue estimate; current counter state is mandatory."""
    if isinstance(people_ahead, bool) or not isinstance(people_ahead, int) or people_ahead < 0:
        raise ValueError("people_ahead must be a nonnegative integer")
    if isinstance(counters_open, bool) or not isinstance(counters_open, int) or counters_open < 0:
        raise ValueError("counters_open must be a nonnegative integer")
    if service_min < 0 or minutes_until_open < 0 or any(v < 0 for v in busy_elapsed_min):
        raise ValueError("time inputs must be nonnegative")
    if minutes_until_open > 0:
        raise ValueError("Scheduled counter reopenings are not represented by this interface")
    if counters_open == 0:
        raise ValueError("Cannot estimate without an open counter; future capacity is unknown")
    if len(busy_elapsed_min) > counters_open:
        raise ValueError("busy_elapsed_min cannot exceed counters_open")
    params = model or DEFAULT_MODEL
    # Empty active counters are idle; listed counters get estimated residual service.
    slots = [0.0] * (counters_open - len(busy_elapsed_min))
    slots += [max(0.0, service_min - float(e)) for e in busy_elapsed_min]
    slots.sort()
    rate = float(params.get("service_multiplier", 1.0)) * service_min
    bias = float(params.get("intercept_min", 0.0))
    for _ in range(people_ahead):
        slots.sort()
        slots[0] += rate
    return max(0.0, float(slots[0] + bias))
