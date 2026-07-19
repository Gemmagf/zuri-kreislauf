"""CLI entry point for `make data` — see zuri_kreislauf.data.fetch for the logic."""

from __future__ import annotations

import sys

from zuri_kreislauf.data.fetch import download

if __name__ == "__main__":
    download(force="--force" in sys.argv)
