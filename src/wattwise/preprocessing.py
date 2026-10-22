from __future__ import annotations
import pandas as pd


def clean_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    before = {
        "rows": len(frame),
        "missing": int(frame.isna().sum().sum()),
        "duplicates": int(frame.duplicated(subset="date").sum()),
    }
    out = frame.copy()
    # Mixed-format parsing handles common CSV fixtures where dates and datetimes coexist.
    try:
        out["date"] = pd.to_datetime(out["date"], errors="coerce", format="mixed")
    except TypeError:  # pandas < 2.0
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = (
        out.dropna(subset=["date", "Appliances"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
    )
    num = out.select_dtypes(include="number").columns
    out[num] = out[num].replace([float("inf"), -float("inf")], pd.NA).ffill()
    # Leading gaps have no past observation to impute from. Drop those rows
    # instead of backfilling from the future (which leaks information).
    leading_gap_rows = int(out[num].isna().any(axis=1).sum())
    out = out.dropna(subset=list(num))
    report = {
        "before": before,
        "after": {
            "rows": len(out),
            "missing": int(out.isna().sum().sum()),
            "duplicates": int(out.duplicated(subset="date").sum()),
        },
        "leading_rows_dropped_without_history": leading_gap_rows,
        "decisions": "Chronological ordering; duplicate timestamps keep last; numeric missing values are forward-filled from past only, and leading rows without history are dropped. Extremes are retained as potentially real use events.",
    }
    return out.reset_index(drop=True), report
