"""Fetch and cache the City of Zurich open-data CSVs used by this project.

Idempotent: skips any file already present under data/raw/. The source URLs
302-redirect, so we must follow redirects explicitly. Lives in the installed
package (rather than scripts/) so both the CLI entry point and the deployed
Streamlit app — which has no shell access to run `make data` — can call it.
"""

from __future__ import annotations

import requests

from zuri_kreislauf.config import ALL_SOURCES, ensure_dirs


def download(force: bool = False) -> None:
    ensure_dirs()
    for source in ALL_SOURCES:
        if source.path.exists() and not force:
            print(f"[skip] {source.name} -> {source.path} (already cached)")
            continue
        print(f"[fetch] {source.name} <- {source.url}")
        response = requests.get(source.url, allow_redirects=True, timeout=30)
        response.raise_for_status()
        source.path.write_bytes(response.content)
        print(f"[ok]   {source.name} -> {source.path} ({len(response.content):,} bytes)")


def ensure_downloaded() -> None:
    """Fetch any missing source file; no-op if everything is already cached."""
    if all(source.path.exists() for source in ALL_SOURCES):
        return
    download()
