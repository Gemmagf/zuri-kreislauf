"""Project paths and data-source registry.

Economic/emission constants used by the decision layer (Layer 4) live in
`zuri_kreislauf.decision.constants`, each with its own source comment — kept
separate from paths so assumptions are easy to audit in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


@dataclass(frozen=True)
class DataSource:
    """One open-data CSV: where it comes from and where it's cached."""

    name: str
    url: str
    filename: str
    retrieved: str  # ISO date this URL/schema was last verified against the brief

    @property
    def path(self) -> Path:
        return RAW_DIR / self.filename


KHKW = DataSource(
    name="erz_abfallmenge_energie_khkw",
    url=(
        "https://data.stadt-zuerich.ch/dataset/erz_abfallmenge_energie_khkw/"
        "download/erz_abfallmenge_energie_khkw.csv"
    ),
    filename="erz_abfallmenge_energie_khkw.csv",
    retrieved="2026-07-17",
)

BIOABFALL_CALENDAR = DataSource(
    name="entsorgungskalender_bioabfall",
    url=(
        "https://data.stadt-zuerich.ch/dataset/entsorgungskalender_bioabfall/"
        "download/entsorgungskalender_bioabfall_2026.csv"
    ),
    filename="entsorgungskalender_bioabfall.csv",
    retrieved="2026-07-17",
)

ELOG_KENNZAHLEN = DataSource(
    name="erz_elog_kennzahlen",
    url="https://data.stadt-zuerich.ch/dataset/erz_elog_kennzahlen/download/erz_elog_kennzahlen.csv",
    filename="erz_elog_kennzahlen.csv",
    retrieved="2026-07-17",
)

ALL_SOURCES = (KHKW, BIOABFALL_CALENDAR, ELOG_KENNZAHLEN)

# Werdhölzli / Biogas Zürich AG postcode — used to filter the collection calendar
# for the site-level view in Layer 3.
BIOGAS_ZUERICH_PLZ = 8064


def ensure_dirs() -> None:
    """Create all standard project directories if missing."""
    for d in (RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
