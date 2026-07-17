"""Expanding-window backtest: SARIMA vs. seasonal-naive.

A random train/test split would let the model "see the future" through
seasonally adjacent months and overstate accuracy. Expanding-window
(train on everything up to month t, forecast t+1..t+h, advance t) is the
honest way to evaluate a monthly time series this short.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from zuri_kreislauf.forecast.baseline import seasonal_naive_forecast
from zuri_kreislauf.forecast.sarima import fit_and_forecast


def _mape(actual: pd.Series, predicted: pd.Series) -> float:
    return float((np.abs((actual - predicted) / actual)).mean() * 100)


def _mae(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.abs(actual - predicted).mean())


def expanding_window_backtest(
    series: pd.Series,
    min_train_size: int,
    horizon: int = 1,
) -> pd.DataFrame:
    """Roll an expanding-origin backtest and return one row per fold with
    SARIMA and seasonal-naive point forecasts vs. the actual.

    Each fold trains on series[:t] and forecasts the single month at
    t + horizon (a fresh 1..horizon forecast is fit each fold; only the
    `horizon`-th step is scored, so results are comparable across horizons).
    """
    rows = []
    n = len(series)
    for t in range(min_train_size, n - horizon + 1):
        train = series.iloc[:t]
        target_date = series.index[t + horizon - 1]
        actual = series.iloc[t + horizon - 1]

        naive = seasonal_naive_forecast(train, horizon).iloc[-1]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                sarima = fit_and_forecast(train, horizon).mean.iloc[-1]
            except Exception:
                sarima = np.nan

        rows.append(
            {
                "target_month": target_date,
                "train_size": t,
                "actual": actual,
                "sarima_pred": sarima,
                "naive_pred": naive,
            }
        )
    return pd.DataFrame(rows).set_index("target_month")


def summarize_backtest(results: pd.DataFrame) -> pd.DataFrame:
    """MAPE/MAE for SARIMA and the seasonal-naive baseline over all folds."""
    valid = results.dropna(subset=["sarima_pred"])
    summary = pd.DataFrame(
        {
            "MAPE_%": [
                _mape(valid["actual"], valid["sarima_pred"]),
                _mape(valid["actual"], valid["naive_pred"]),
            ],
            "MAE": [
                _mae(valid["actual"], valid["sarima_pred"]),
                _mae(valid["actual"], valid["naive_pred"]),
            ],
        },
        index=["sarima", "seasonal_naive"],
    )
    return summary
