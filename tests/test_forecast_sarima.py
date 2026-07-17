from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from zuri_kreislauf.forecast.sarima import fit_and_forecast


def _seasonal_series(n=48) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(0)
    values = 100 + 10 * np.sin(2 * np.pi * idx.month / 12) + rng.normal(0, 0.5, n)
    return pd.Series(values, index=idx)


@pytest.mark.slow
def test_forecast_shape_and_index():
    s = _seasonal_series()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = fit_and_forecast(s, horizon=6)
    assert len(fc.mean) == 6
    expected_index = pd.date_range(s.index[-1] + pd.DateOffset(months=1), periods=6, freq="MS")
    pd.testing.assert_index_equal(fc.mean.index, expected_index)


@pytest.mark.slow
def test_confidence_interval_brackets_mean():
    s = _seasonal_series()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = fit_and_forecast(s, horizon=6)
    assert (fc.ci_lower <= fc.mean).all()
    assert (fc.mean <= fc.ci_upper).all()
