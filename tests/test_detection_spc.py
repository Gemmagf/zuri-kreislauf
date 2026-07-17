from __future__ import annotations

import numpy as np
import pandas as pd

from zuri_kreislauf.detection.spc import run_spc, top_contributors

COLUMNS = ("a", "b", "c")


def _synthetic_ratios(n=60, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="MS")
    data = rng.normal(size=(n, 3))
    df = pd.DataFrame(data, index=idx, columns=list(COLUMNS))
    # Inject one obvious outlier month.
    df.iloc[-1] = [10, 10, 10]
    return df


def test_t2_contributions_sum_to_t2():
    ratios = _synthetic_ratios()
    result = run_spc(ratios, COLUMNS, alpha=0.01)
    recombined = result.t2_contributions.sum(axis=1)
    np.testing.assert_allclose(recombined.to_numpy(), result.t2.to_numpy(), rtol=1e-8)


def test_injected_outlier_is_flagged():
    ratios = _synthetic_ratios()
    result = run_spc(ratios, COLUMNS, alpha=0.01)
    assert result.flags.iloc[-1]


def test_top_contributors_returns_k_features():
    ratios = _synthetic_ratios()
    result = run_spc(ratios, COLUMNS, alpha=0.01)
    top = top_contributors(result.t2_contributions, ratios.index[-1], k=2)
    assert len(top) == 2
    assert top.is_monotonic_decreasing
