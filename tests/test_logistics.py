from __future__ import annotations

import pandas as pd

from zuri_kreislauf.logistics.collection import collections_per_month, plz_schedule


def _toy_calendar():
    return pd.DataFrame(
        {
            "PLZ": [8064, 8064, 8064, 8001],
            "Abholdatum": pd.to_datetime(["2026-01-05", "2026-01-19", "2026-02-02", "2026-01-05"]),
        }
    )


def test_collections_per_month_counts_by_plz():
    result = collections_per_month(_toy_calendar())
    assert result.loc[pd.Period("2026-01", freq="M"), 8064] == 2
    assert result.loc[pd.Period("2026-02", freq="M"), 8064] == 1
    assert result.loc[pd.Period("2026-01", freq="M"), 8001] == 1


def test_plz_schedule_computes_gap_in_days():
    result = plz_schedule(_toy_calendar(), 8064)
    assert len(result) == 3
    assert result["days_since_previous"].iloc[0] != result["days_since_previous"].iloc[0]  # NaN
    assert result["days_since_previous"].iloc[1] == 14
    assert result["days_since_previous"].iloc[2] == 14
