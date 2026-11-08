"""Deterministic lightweight FIFO estimate for explicitly supplied live state.

This is a reference approximation, not a claim to reproduce the production engine.
The application must supply accurate busy durations and closures before using it.
"""
from datetime import datetime


def simulate_wait_minutes(people_ahead: int, counters_open: int,
                          busy_elapsed_min: list[float],
                          minutes_until_open: float, service_min: float,
                          now: datetime) -> float:
    if isinstance(people_ahead, bool) or not isinstance(people_ahead, int) or people_ahead < 0:
        raise ValueError("people_ahead must be a nonnegative integer")
    if isinstance(counters_open, bool) or not isinstance(counters_open, int) or counters_open < 0:
        raise ValueError("counters_open must be a nonnegative integer")
    if service_min < 0 or minutes_until_open < 0 or any(x < 0 for x in busy_elapsed_min):
        raise ValueError("time inputs must be nonnegative")
    if minutes_until_open > 0:
        raise ValueError("Scheduled counter reopenings are not represented by this interface")
    if counters_open == 0:
        raise ValueError("No active counters; future capacity is unknown")
    if len(busy_elapsed_min) > counters_open:
        raise ValueError("busy_elapsed_min cannot exceed counters_open")
    # Each idle counter is available now; each busy counter becomes available after
    # max(0, usual service duration minus elapsed duration). This residual estimate
    # is a simplifying assumption pending measured residual-time data.
    slots = [0.0] * (counters_open - len(busy_elapsed_min))
    slots.extend(max(0.0, service_min - float(elapsed)) for elapsed in busy_elapsed_min)
    for _ in range(people_ahead):
        slots.sort()
        slots[0] += service_min
    slots.sort()
    return float(max(0.0, slots[0]))
