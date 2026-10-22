from __future__ import annotations
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = {"date", "Appliances", "lights"}


def validate_frame(frame: pd.DataFrame, min_rows: int = 2) -> dict:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if len(frame) < min_rows:
        raise ValueError(f"Expected at least {min_rows} rows, received {len(frame)}")
    parsed = pd.to_datetime(frame["date"], errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{int(parsed.isna().sum())} invalid timestamps found")
    if not pd.api.types.is_numeric_dtype(frame["Appliances"]):
        raise ValueError("Appliances target must be numeric")
    if not np.isfinite(frame["Appliances"].to_numpy(dtype=float)).all():
        raise ValueError("Appliances target must contain finite numeric values")
    if (frame["Appliances"] < 0).any():
        raise ValueError("Appliances contains negative energy values")
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicates": int(frame.duplicated().sum()),
    }
