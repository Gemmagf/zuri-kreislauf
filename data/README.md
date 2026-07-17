# Data

Raw CSVs are fetched by `scripts/download_data.py` (or `make data`) and cached under
`data/raw/`. They are not committed — the download script hits open, unauthenticated
City of Zurich / opendata.swiss endpoints and re-downloading is cheap and reproducible.

Re-running the script does not re-fetch files that already exist locally, so it is safe
to run repeatedly without hammering the source servers.

See the main README's "Data provenance" section for dataset descriptions, licenses and
retrieval dates.
