import pandas as pd
from wattwise.preprocessing import clean_data


def test_basic_preprocessing():
    d = pd.DataFrame(
        {
            "date": ["2021-01-01", "2021-01-01 00:10"],
            "Appliances": [10, 20],
            "lights": [0, 1],
        }
    )
    out, _ = clean_data(d)
    assert len(out) == 2 and out.Appliances.tolist() == [10, 20]


def test_leading_missing_values_are_dropped_not_backfilled():
    d = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=3, freq="10min"),
            "Appliances": [10, 20, 30],
            "lights": [None, 2, 3],
        }
    )
    out, report = clean_data(d)
    assert out.lights.tolist() == [2, 3]
    assert report["leading_rows_dropped_without_history"] == 1
