from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest import requires_data
from zuri_kreislauf.data.loader import KHKW_INT_COLUMNS, _parse_swiss_number

pytestmark = requires_data


def test_khkw_schema_and_index(khkw_df):
    assert list(khkw_df.columns) == list(KHKW_INT_COLUMNS)
    assert isinstance(khkw_df.index, pd.DatetimeIndex)
    assert khkw_df.index.is_monotonic_increasing
    assert khkw_df.index.is_unique


def test_khkw_no_nulls_and_positive_throughput(khkw_df):
    assert khkw_df.isna().sum().sum() == 0
    assert (khkw_df["Kehrichtdurchsatz"] > 0).all()


def test_khkw_covers_expected_range(khkw_df):
    assert khkw_df.index.min() <= pd.Timestamp("2020-01-01")
    assert len(khkw_df) >= 60  # at least 5 years of monthly data


def test_bioabfall_calendar_schema(bioabfall_df):
    assert set(bioabfall_df.columns) == {"PLZ", "Abholdatum"}
    assert bioabfall_df["PLZ"].dtype == "int64"
    assert pd.api.types.is_datetime64_any_dtype(bioabfall_df["Abholdatum"])


def test_bioabfall_includes_biogas_zuerich_plz(bioabfall_df):
    assert (bioabfall_df["PLZ"] == 8064).any()


def test_elog_kennzahlen_schema(elog_df):
    assert set(elog_df.columns) == {"Datum", "Beschreibung", "Masseinheit", "Wert", "Intervall"}
    assert pd.api.types.is_numeric_dtype(elog_df["Wert"])


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("92'056", 92056.0),
        ("4828.7", 4828.7),
        (100, 100.0),
    ],
)
def test_parse_swiss_number(raw, expected):
    assert _parse_swiss_number(raw) == expected
