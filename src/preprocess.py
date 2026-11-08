"""Validate cashier observations and derive actual wait/service durations."""
import csv
from datetime import datetime, timedelta
from pathlib import Path

REQUIRED = ["date", "ticket_number", "window", "ticket_time", "people_ahead",
            "windows_open", "called_time", "finished_time"]
OPTIONAL = ["predicted_wait_min", "wait_model_version", "record_origin"]
OUTPUT = REQUIRED + ["actual_wait_min", "actual_service_min", "duration_valid",
                     "validation_note"] + OPTIONAL


def parse_timestamp(value: str):
    value = (value or "").strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def clean_rows(rows):
    """Preserve every input row and flag invalid or missing values; never impute."""
    result = []
    for source in rows:
        row = {k: (source.get(k, "") or "").strip() for k in OUTPUT}
        issues = []
        for field in REQUIRED:
            if not row[field]:
                issues.append(f"missing_{field}")
        for field in ("people_ahead", "windows_open"):
            try:
                n = int(row[field])
                if n < 0:
                    issues.append(f"negative_{field}")
            except (ValueError, TypeError):
                issues.append(f"invalid_{field}")
        ticket, called, finished = (parse_timestamp(row[k]) for k in
                                    ("ticket_time", "called_time", "finished_time"))
        # If an event's local clock time is earlier, treat it as next-day only when
        # this is a plausible midnight crossing (<24 h); otherwise flag it.
        if ticket and called and called < ticket:
            if ticket - called >= timedelta(hours=12):
                called += timedelta(days=1)
                row["called_time"] = called.isoformat(sep=" ")
                issues.append("called_time_crossed_midnight")
            else:
                issues.append("called_before_ticket")
        if called and finished and finished < called:
            if called - finished >= timedelta(hours=12):
                finished += timedelta(days=1)
                row["finished_time"] = finished.isoformat(sep=" ")
                issues.append("finished_time_crossed_midnight")
            else:
                issues.append("finished_before_called")
        wait = (called - ticket).total_seconds() / 60 if ticket and called and called >= ticket else None
        service = (finished - called).total_seconds() / 60 if called and finished and finished >= called else None
        valid = wait is not None and service is not None and not any(
            x in issues for x in ("called_before_ticket", "finished_before_called"))
        row["actual_wait_min"] = f"{wait:.6f}" if wait is not None else ""
        row["actual_service_min"] = f"{service:.6f}" if service is not None else ""
        row["duration_valid"] = "true" if valid else "false"
        row["validation_note"] = ";".join(issues)
        result.append(row)
    return result


def clean_file(source: str | Path, destination: str | Path):
    with open(source, encoding="utf-8-sig", newline="") as f:
        rows = clean_rows(csv.DictReader(f))
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT)
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("destination")
    args = p.parse_args()
    rows = clean_file(args.source, args.destination)
    print(f"Preserved {len(rows)} rows with validity flags in {args.destination}")
