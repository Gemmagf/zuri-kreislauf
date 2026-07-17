"""IsolationForest as a second, model-free anomaly detector.

Cross-checks the parametric SPC monitor (spc.py) with a non-parametric one —
agreement between the two raises confidence in a flag; disagreement is worth
narrating rather than silently trusting either one.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def run_iforest(
    ratios: pd.DataFrame,
    columns: tuple[str, ...],
    contamination: float = 0.1,
    random_state: int = 0,
) -> pd.DataFrame:
    """Returns a DataFrame indexed like `ratios` with `score` (higher = more
    anomalous) and `flag` (bool) columns.
    """
    x = ratios[list(columns)].dropna()
    z = StandardScaler().fit_transform(x)

    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(z)
    # decision_function: higher = more normal. Flip sign so higher = more anomalous.
    raw_score = -model.decision_function(z)
    is_outlier = model.predict(z) == -1

    return pd.DataFrame(
        {"score": raw_score, "flag": is_outlier},
        index=x.index,
    )
