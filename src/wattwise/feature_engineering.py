from __future__ import annotations
import numpy as np
import pandas as pd


def make_features(
    frame: pd.DataFrame,
    lags=(1, 2, 3, 6, 12, 144),
    rolling=(3, 6, 12, 144),
    drop_warmup=True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """At t, features use data through t; y is observed Appliances at t+1."""
    d = frame.sort_values("date").reset_index(drop=True).copy()
    ts = pd.to_datetime(d["date"])
    x = d.drop(columns=["date", "Appliances"]).copy()
    x["appliances_current"] = d["Appliances"]
    for lag in lags:
        if lag > 0:
            x[f"appliances_lag_{lag}"] = d["Appliances"].shift(lag)
    for window in rolling:
        hist = d["Appliances"].shift(1)
        x[f"appliances_roll_mean_{window}"] = hist.rolling(
            window, min_periods=window
        ).mean()
        x[f"appliances_roll_std_{window}"] = hist.rolling(
            window, min_periods=window
        ).std()
    # Recent changes can help predict a one-step transition; every difference uses t and t-1 only.
    change_columns = [
        c
        for c in [
            "Appliances",
            "lights",
            "T1",
            "RH_1",
            "T2",
            "RH_2",
            "T3",
            "RH_3",
            "T6",
            "RH_6",
            "T_out",
            "RH_out",
            "Windspeed",
        ]
        if c in d
    ]
    for column in change_columns:
        feature_name = (
            "appliances_change_1" if column == "Appliances" else f"{column}_change_1"
        )
        x[feature_name] = d[column].diff()
    hour = ts.dt.hour + ts.dt.minute / 60
    x["hour"] = hour
    x["day_of_week"] = ts.dt.day_of_week
    x["is_weekend"] = (ts.dt.day_of_week >= 5).astype(int)
    x["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    weekday_position = ts.dt.day_of_week + hour / 24
    x["week_sin"] = np.sin(2 * np.pi * weekday_position / 7)
    x["week_cos"] = np.cos(2 * np.pi * weekday_position / 7)
    x["minute_of_day"] = ts.dt.hour * 60 + ts.dt.minute
    y = d["Appliances"].shift(-1).rename("target_next")
    valid = y.notna() & x.notna().all(axis=1)
    if not drop_warmup:
        valid = y.notna()
    return x.loc[valid].astype(float), y.loc[valid].astype(float), ts.loc[valid]


def chronological_split(X, y, dates, train=0.70, valid=0.15):
    n = len(X)
    a = int(n * train)
    b = int(n * (train + valid))
    return {
        "train": (X.iloc[:a], y.iloc[:a], dates.iloc[:a]),
        "validation": (X.iloc[a:b], y.iloc[a:b], dates.iloc[a:b]),
        "test": (X.iloc[b:], y.iloc[b:], dates.iloc[b:]),
    }
