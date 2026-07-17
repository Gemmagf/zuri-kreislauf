from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from zuri_kreislauf.decision.cost import heat_seasonal_baseline, monthly_impact, reagent_baselines


@pytest.fixture
def toy_khkw():
    idx = pd.date_range("2024-01-01", periods=6, freq="MS")
    return pd.DataFrame(
        {
            "Kehrichtdurchsatz": [20000] * 6,
            "Ammoniakverbrauch": [70] * 6,
            "Natronlaugeverbrauch": [30] * 6,
            "Salzsaeureverbrauch": [60] * 6,
            "Kalkverbrauch": [25] * 6,
            "Stromproduktion": [8000, 8000, 8000, 8000, 8000, 0],  # last month: zero output
            "Waermeabsatz": [40000] * 6,
            "Abtransp_Restprodukte": [4000] * 6,
        },
        index=idx,
    )


@pytest.fixture
def toy_ratios(toy_khkw):
    from zuri_kreislauf.data.preprocess import build_intensity_ratios

    return build_intensity_ratios(toy_khkw)


def test_reagent_baselines_use_only_normal_months(toy_ratios):
    normal_mask = pd.Series(True, index=toy_ratios.index)
    baselines = reagent_baselines(toy_ratios, normal_mask)
    assert baselines["electricity_kwh_per_t"] == pytest.approx(
        toy_ratios["electricity_kwh_per_t"].median()
    )


def test_heat_seasonal_baseline_indexed_by_month(toy_ratios):
    normal_mask = pd.Series(True, index=toy_ratios.index)
    baseline = heat_seasonal_baseline(toy_ratios, normal_mask)
    assert set(baseline.index) == set(toy_ratios.index.month)


def test_electricity_outage_month_shows_cost_and_co2(toy_khkw, toy_ratios):
    normal_mask = pd.Series(True, index=toy_ratios.index)
    normal_mask.iloc[-1] = False  # the zero-production month is the anomaly, excluded from baseline
    impact = monthly_impact(toy_khkw, toy_ratios, normal_mask)

    outage_month = toy_ratios.index[-1]
    assert impact.loc[outage_month, "electricity_chf"] > 0
    assert impact.loc[outage_month, "electricity_co2_kg"] > 0

    normal_month = toy_ratios.index[0]
    assert impact.loc[normal_month, "electricity_chf"] == pytest.approx(0.0, abs=1e-6)


def test_total_columns_sum_components(toy_khkw, toy_ratios):
    normal_mask = pd.Series(True, index=toy_ratios.index)
    impact = monthly_impact(toy_khkw, toy_ratios, normal_mask)

    reagent_cols = [
        "ammonia_kg_per_t_chf",
        "naoh_kg_per_t_chf",
        "hcl_kg_per_t_chf",
        "lime_kg_per_t_chf",
    ]
    expected_total_chf = impact[[*reagent_cols, "electricity_chf", "heat_chf"]].sum(axis=1)
    np.testing.assert_allclose(impact["total_chf"].to_numpy(), expected_total_chf.to_numpy())

    expected_total_co2 = impact[["electricity_co2_kg", "heat_co2_kg"]].sum(axis=1)
    np.testing.assert_allclose(impact["total_co2_kg"].to_numpy(), expected_total_co2.to_numpy())
