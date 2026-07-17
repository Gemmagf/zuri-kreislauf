from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zuri_kreislauf.forecast.backtest import expanding_window_backtest, summarize_backtest


def _seasonal_series(n=48) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(0)
    values = 100 + 10 * np.sin(2 * np.pi * idx.month / 12) + rng.normal(0, 0.5, n)
    return pd.Series(values, index=idx)


@pytest.mark.slow
def test_backtest_produces_one_row_per_fold():
    s = _seasonal_series(36)
    min_train = 24
    horizon = 1
    results = expanding_window_backtest(s, min_train_size=min_train, horizon=horizon)
    assert len(results) == len(s) - min_train - horizon + 1
    assert set(results.columns) == {"train_size", "actual", "sarima_pred", "naive_pred"}


def test_summarize_backtest_computes_mape_and_mae():
    results = pd.DataFrame(
        {
            "actual": [100.0, 200.0],
            "sarima_pred": [110.0, 190.0],
            "naive_pred": [90.0, 220.0],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="MS"),
    )
    summary = summarize_backtest(results)
    assert set(summary.index) == {"sarima", "seasonal_naive"}
    np.testing.assert_allclose(summary.loc["sarima", "MAE"], 10.0)
    np.testing.assert_allclose(summary.loc["seasonal_naive", "MAE"], 15.0)
