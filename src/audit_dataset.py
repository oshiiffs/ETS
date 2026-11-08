"""Standard-library inventory and quality summary for CSV files."""
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path
import argparse


def audit(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"path": str(path), "rows": 0, "columns": [], "days": 0}
    cols = list(rows[0])
    dates = [r.get("date", "") for r in rows if r.get("date", "")]
    missing = {c: sum(not (r.get(c) or "").strip() for r in rows) for c in cols}
    parsed = []
    for r in rows:
        try:
            parsed.append(datetime.fromisoformat(r["ticket_creation_datetime"]))
        except (KeyError, ValueError, TypeError):
            pass
    result = {"path": str(path), "rows": len(rows), "columns": cols,
              "days": len(set(dates)), "date_min": min(dates) if dates else None,
              "date_max": max(dates) if dates else None, "missing": missing,
              "exact_duplicate_rows": len(rows) - len({tuple(sorted(r.items())) for r in rows}),
              "invalid_ticket_timestamps": len(rows) - len(parsed),
              "ticket_date_mismatches": sum(r.get("date") != (r.get("ticket_creation_datetime") or "")[:10]
                                               for r in rows),
              "department_counts": dict(Counter(r.get("department", "") for r in rows)),
              "window_counts": dict(Counter(r.get("cashier_window", "") for r in rows))}
    for col in ("waiting_time_minutes", "service_time_minutes", "people_ahead", "active_cashier_windows"):
        vals = []
        for r in rows:
            try: vals.append(float(r[col]))
            except (KeyError, ValueError, TypeError): pass
        result[col] = {"valid_n": len(vals), "min": min(vals) if vals else None,
                       "max": max(vals) if vals else None,
                       "mean": sum(vals) / len(vals) if vals else None}
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", type=Path)
    args = p.parse_args()
    import json
    print(json.dumps([audit(x) for x in args.paths], indent=2))
