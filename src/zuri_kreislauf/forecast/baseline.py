"""Seasonal-naive baseline: this month's forecast is the same month last year.

The bar any fitted model (SARIMA, GBM, ...) has to clear before it's worth
using at all — with only 78 monthly observations, a model that can't beat
this is not adding value, however sophisticated it looks.
"""

from __future__ import annotations

import pandas as pd


def seasonal_naive_forecast(history: pd.Series, horizon: int, season_length: int = 12) -> pd.Series:
    """Forecast `horizon` steps ahead using the value `season_length` steps back.

    Falls back to repeating the last observed value once fewer than
    `season_length` matching lags are available.
    """
    future_index = pd.date_range(
        history.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS"
    )
    values = []
    extended = history.copy()
    for date in future_index:
        lag_date = date - pd.DateOffset(months=season_length)
        value = extended.loc[lag_date] if lag_date in extended.index else extended.iloc[-1]
        values.append(value)
        extended.loc[date] = value
    return pd.Series(values, index=future_index, name="seasonal_naive")
