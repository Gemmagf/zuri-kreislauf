from __future__ import annotations

import numpy as np
import pandas as pd

from zuri_kreislauf.detection.iforest import run_iforest

COLUMNS = ("a", "b", "c")


def _synthetic_ratios(n=60, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="MS")
    data = rng.normal(size=(n, 3))
    df = pd.DataFrame(data, index=idx, columns=list(COLUMNS))
    df.iloc[-1] = [10, 10, 10]
    return df


def test_output_shape_and_dtypes():
    ratios = _synthetic_ratios()
    result = run_iforest(ratios, COLUMNS)
    assert list(result.columns) == ["score", "flag"]
    assert len(result) == len(ratios)
    assert result["flag"].dtype == bool


def test_injected_outlier_has_high_score():
    ratios = _synthetic_ratios()
    result = run_iforest(ratios, COLUMNS, contamination=0.05)
    assert result["score"].idxmax() == ratios.index[-1]
    assert result["flag"].iloc[-1]
