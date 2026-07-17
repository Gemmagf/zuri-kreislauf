"""Build normalised process-intensity ratios from the raw KHKW plant data.

Everything downstream (anomaly detection, decision layer) operates on these
ratios rather than raw tonnages/MWh, since throughput itself varies with
collection volume and heat demand is strongly seasonal — the ratios isolate
process efficiency from those confounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RATIO_COLUMNS = (
    "electricity_kwh_per_t",
    "heat_kwh_per_t",
    "ammonia_kg_per_t",
    "naoh_kg_per_t",
    "hcl_kg_per_t",
    "lime_kg_per_t",
    "residual_fraction",
)

# Features fed to the multivariate monitor (Layer 2). Uses the seasonally
# adjusted heat ratio instead of the raw one to avoid the two heat columns
# being near-collinear (raw heat is dominated by calendar month).
MONITORING_COLUMNS = (
    "electricity_kwh_per_t",
    "heat_kwh_per_t_sa",
    "ammonia_kg_per_t",
    "naoh_kg_per_t",
    "hcl_kg_per_t",
    "lime_kg_per_t",
    "residual_fraction",
)


def build_intensity_ratios(khkw: pd.DataFrame) -> pd.DataFrame:
    """One row per month, the ratios in RATIO_COLUMNS plus a seasonally
    deseasonalised heat ratio (`heat_kwh_per_t_sa`) via a monthly-mean baseline.
    """
    t = khkw["Kehrichtdurchsatz"].astype(float)
    ratios = pd.DataFrame(index=khkw.index)

    # MWh -> kWh per tonne: source columns are already MWh, tonnes are tonnes.
    ratios["electricity_kwh_per_t"] = khkw["Stromproduktion"] * 1000 / t
    ratios["heat_kwh_per_t"] = khkw["Waermeabsatz"] * 1000 / t

    # Reagents are reported in tonnes per month (confirmed via the dataset's
    # sszFields metadata: "Ammoniak [t]", "Natronlauge [t]", "Salzsäure [t]",
    # "Kalk [t]" — NOT kg, despite the "*_kg_per_t" naming below). Convert
    # t -> kg so the ratio reads in the industry-standard kg-reagent-per-
    # tonne-waste unit.
    ratios["ammonia_kg_per_t"] = khkw["Ammoniakverbrauch"] * 1000 / t
    ratios["naoh_kg_per_t"] = khkw["Natronlaugeverbrauch"] * 1000 / t
    ratios["hcl_kg_per_t"] = khkw["Salzsaeureverbrauch"] * 1000 / t
    ratios["lime_kg_per_t"] = khkw["Kalkverbrauch"] * 1000 / t

    ratios["residual_fraction"] = khkw["Abtransp_Restprodukte"] / t

    # Heat is dominated by seasonal demand (winter >> summer); express it as a
    # ratio to that month's across-year mean so a July anomaly isn't masked by
    # July always being low, and a January anomaly isn't hidden by January
    # always being high.
    month_of_year = ratios.index.month
    monthly_baseline = ratios["heat_kwh_per_t"].groupby(month_of_year).transform("mean")
    ratios["heat_kwh_per_t_sa"] = ratios["heat_kwh_per_t"] / monthly_baseline

    return ratios


def summarize_ratios(ratios: pd.DataFrame) -> pd.DataFrame:
    """Descriptive stats (mean, std, cv, min, max) per ratio column."""
    stats = ratios.agg(["mean", "std", "min", "max"]).T
    stats["cv"] = stats["std"] / stats["mean"].replace(0, np.nan)
    return stats
