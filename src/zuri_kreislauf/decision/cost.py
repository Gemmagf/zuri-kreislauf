"""Translate process-intensity deviations (Layer 2) into CHF and tCO2eq.

The "expected" value for each ratio is the median observed in *normal*
months (as judged by the SPC/IsolationForest flags) — so a flagged month's
economic impact is measured against what this plant actually, recently
achieves, not an external benchmark. Heat is compared against its own
calendar-month baseline since it is strongly seasonal; everything else
uses a single across-year baseline.
"""

from __future__ import annotations

import pandas as pd

from zuri_kreislauf.decision.constants import (
    AMMONIA_PRICE_CHF_PER_KG,
    ELECTRICITY_EMISSION_FACTOR_KG_CO2_PER_KWH,
    ELECTRICITY_PRICE_CHF_PER_KWH,
    GAS_EMISSION_FACTOR_KG_CO2_PER_KWH,
    HCL_PRICE_CHF_PER_KG,
    HEAT_PRICE_CHF_PER_KWH,
    LIME_PRICE_CHF_PER_KG,
    NAOH_PRICE_CHF_PER_KG,
)

REAGENT_PRICES_CHF_PER_KG = {
    "ammonia_kg_per_t": AMMONIA_PRICE_CHF_PER_KG,
    "naoh_kg_per_t": NAOH_PRICE_CHF_PER_KG,
    "hcl_kg_per_t": HCL_PRICE_CHF_PER_KG,
    "lime_kg_per_t": LIME_PRICE_CHF_PER_KG,
}


def reagent_baselines(ratios: pd.DataFrame, normal_mask: pd.Series) -> pd.Series:
    """Median of each reagent/electricity ratio over non-flagged months."""
    columns = [*REAGENT_PRICES_CHF_PER_KG, "electricity_kwh_per_t"]
    return ratios.loc[normal_mask, columns].median()


def heat_seasonal_baseline(ratios: pd.DataFrame, normal_mask: pd.Series) -> pd.Series:
    """Median heat_kwh_per_t per calendar month, over non-flagged months only."""
    normal = ratios.loc[normal_mask]
    return normal["heat_kwh_per_t"].groupby(normal.index.month).median()


def monthly_impact(
    khkw: pd.DataFrame,
    ratios: pd.DataFrame,
    normal_mask: pd.Series,
) -> pd.DataFrame:
    """One row per month: CHF and kg-CO2eq impact of each ratio's deviation
    from its normal-month baseline, plus totals.

    Sign convention: positive = cost / extra emissions (i.e. reagent
    overconsumption or energy-yield shortfall vs. the baseline);
    negative = saving / avoided emissions.
    """
    t = khkw["Kehrichtdurchsatz"].astype(float)
    baselines = reagent_baselines(ratios, normal_mask)
    heat_baseline_by_month = heat_seasonal_baseline(ratios, normal_mask)

    impact = pd.DataFrame(index=ratios.index)

    for col, price in REAGENT_PRICES_CHF_PER_KG.items():
        deviation_kg_per_t = ratios[col] - baselines[col]
        impact[f"{col}_chf"] = deviation_kg_per_t * t * price

    elec_deviation_kwh_per_t = baselines["electricity_kwh_per_t"] - ratios["electricity_kwh_per_t"]
    elec_shortfall_kwh = elec_deviation_kwh_per_t * t
    impact["electricity_chf"] = elec_shortfall_kwh * ELECTRICITY_PRICE_CHF_PER_KWH
    impact["electricity_co2_kg"] = elec_shortfall_kwh * ELECTRICITY_EMISSION_FACTOR_KG_CO2_PER_KWH

    expected_heat = ratios.index.month.map(heat_baseline_by_month)
    heat_shortfall_kwh = (
        pd.Series(expected_heat, index=ratios.index) - ratios["heat_kwh_per_t"]
    ) * t
    impact["heat_chf"] = heat_shortfall_kwh * HEAT_PRICE_CHF_PER_KWH
    impact["heat_co2_kg"] = heat_shortfall_kwh * GAS_EMISSION_FACTOR_KG_CO2_PER_KWH

    reagent_chf_cols = [f"{col}_chf" for col in REAGENT_PRICES_CHF_PER_KG]
    impact["total_chf"] = impact[[*reagent_chf_cols, "electricity_chf", "heat_chf"]].sum(axis=1)
    impact["total_co2_kg"] = impact[["electricity_co2_kg", "heat_co2_kg"]].sum(axis=1)

    return impact
