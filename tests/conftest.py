from __future__ import annotations

import pytest

from zuri_kreislauf.config import ALL_SOURCES
from zuri_kreislauf.data.loader import load_bioabfall_calendar, load_elog_kennzahlen, load_khkw


def _data_available() -> bool:
    return all(source.path.exists() for source in ALL_SOURCES)


requires_data = pytest.mark.skipif(
    not _data_available(), reason="raw CSVs not cached — run `make data` first"
)


@pytest.fixture(scope="session")
def khkw_df():
    return load_khkw()


@pytest.fixture(scope="session")
def bioabfall_df():
    return load_bioabfall_calendar()


@pytest.fixture(scope="session")
def elog_df():
    return load_elog_kennzahlen(pivot=False)
