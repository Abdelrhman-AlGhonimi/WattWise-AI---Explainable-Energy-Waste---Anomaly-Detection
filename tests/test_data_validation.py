import pandas as pd
import pytest
from wattwise.validation import validate_frame
from wattwise.preprocessing import clean_data


def test_validation_and_schema_error():
    d = pd.DataFrame(
        {"date": ["2020-01-01", "2020-01-02"], "Appliances": [1, 2], "lights": [0, 0]}
    )
    assert validate_frame(d)["rows"] == 2
    with pytest.raises(ValueError):
        validate_frame(d.drop(columns="lights"))


def test_validation_rejects_nonfinite_target():
    d = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "Appliances": [1, float("inf")],
            "lights": [0, 0],
        }
    )
    with pytest.raises(ValueError, match="finite"):
        validate_frame(d)


def test_clean_sorts_deduplicates_and_fills():
    d = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-01", "2020-01-01"],
            "Appliances": [2, 4, 3],
            "lights": [1, 1, 2],
        }
    )
    out, report = clean_data(d)
    assert out.date.is_monotonic_increasing and not out.date.duplicated().any()
    assert report["after"]["duplicates"] == 0
