"""Light-touch view of the bio-waste collection calendar (Layer 3).

Deliberately kept small (per project brief): this is context for when/where
feedstock arrives, not a routing or scheduling optimiser.
"""

from __future__ import annotations

import pandas as pd


def collections_per_month(calendar: pd.DataFrame) -> pd.DataFrame:
    """Collection count per PLZ per calendar month."""
    monthly = calendar.assign(month=calendar["Abholdatum"].dt.to_period("M"))
    return monthly.pivot_table(
        index="month", columns="PLZ", values="Abholdatum", aggfunc="count"
    ).fillna(0)


def plz_schedule(calendar: pd.DataFrame, plz: int) -> pd.DataFrame:
    """Sorted collection dates for one PLZ, with weekday name and the gap
    in days since the previous collection.
    """
    subset = calendar.loc[calendar["PLZ"] == plz, ["Abholdatum"]].sort_values("Abholdatum")
    subset = subset.reset_index(drop=True)
    subset["weekday"] = subset["Abholdatum"].dt.day_name()
    subset["days_since_previous"] = subset["Abholdatum"].diff().dt.days
    return subset
