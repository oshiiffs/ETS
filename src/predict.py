"""Backend-safe deterministic predictor with no third-party runtime dependencies."""
from datetime import datetime
import json
from pathlib import Path


DEFAULT_MODEL = {"model": "fifo_constant_service", "version": "0.1.0",
                 "service_multiplier": 1.0, "intercept_min": 0.0}


def load_model(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        model = json.load(f)
    if model.get("model") != "fifo_constant_service":
        raise ValueError("Unsupported model type")
    return model


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
