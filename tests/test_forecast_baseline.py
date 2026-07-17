from __future__ import annotations

import numpy as np
import pandas as pd

from zuri_kreislauf.forecast.baseline import seasonal_naive_forecast


def _seasonal_series(n=36) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=n, freq="MS")
    values = 100 + 10 * np.sin(2 * np.pi * idx.month / 12)
    return pd.Series(values, index=idx)


def test_forecast_matches_same_month_last_year():
    s = _seasonal_series(36)
    fc = seasonal_naive_forecast(s, horizon=12)
    assert len(fc) == 12
    np.testing.assert_allclose(fc.to_numpy(), s.iloc[-12:].to_numpy(), rtol=1e-9)


def test_forecast_falls_back_when_no_full_season_available():
    s = _seasonal_series(6)  # shorter than season_length=12
    fc = seasonal_naive_forecast(s, horizon=3)
    assert len(fc) == 3
    assert (fc == s.iloc[-1]).all()


def test_forecast_index_is_contiguous_monthly():
    s = _seasonal_series(24)
    fc = seasonal_naive_forecast(s, horizon=5)
    expected_index = pd.date_range(s.index[-1] + pd.DateOffset(months=1), periods=5, freq="MS")
    pd.testing.assert_index_equal(fc.index, expected_index)
