"""SARIMA load forecasting for monthly plant throughput.

n=78 rules out anything data-hungry (see project brief §4) — SARIMA with a
small, fixed order is the right amount of model for a single strong, stable
calendar-seasonal pattern (a September maintenance shutdown, present in all
six years of history) and no significant multi-year trend.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Grid search over (p,d,q)x(P,D,Q,12) in {0,1}^6 by AIC picked
# (1,1,1)x(1,1,1,12) as the minimum (AIC 961.7), but (0,1,1)x(0,1,1,12) — the
# classic Box-Jenkins "airline model" — is within 1 AIC point (962.6) with
# half the parameters. We use the simpler model: on 78 points, a ~1-point AIC
# gain isn't worth the extra parameters.
DEFAULT_ORDER = (0, 1, 1)
DEFAULT_SEASONAL_ORDER = (0, 1, 1, 12)


@dataclass
class SarimaForecast:
    mean: pd.Series
    ci_lower: pd.Series
    ci_upper: pd.Series


def fit_and_forecast(
    history: pd.Series,
    horizon: int,
    order: tuple[int, int, int] = DEFAULT_ORDER,
    seasonal_order: tuple[int, int, int, int] = DEFAULT_SEASONAL_ORDER,
    alpha: float = 0.05,
) -> SarimaForecast:
    """Fit SARIMA on `history` and forecast `horizon` months ahead with a
    (1 - alpha) confidence interval.
    """
    model = SARIMAX(
        history,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    pred = fitted.get_forecast(steps=horizon)
    ci = pred.conf_int(alpha=alpha)
    return SarimaForecast(
        mean=pred.predicted_mean,
        ci_lower=ci.iloc[:, 0],
        ci_upper=ci.iloc[:, 1],
    )
