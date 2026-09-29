# Züri-Kreislauf → merged into the Swiss Data Cockpit

This project has been folded into the **Swiss Data Cockpit** as its asset drill-down level
(*Switzerland › Energy & Climate › Zurich › KVA Hagenholz*) and is archived here.

- Live view: https://gemmagf.github.io/swiss-governance-dashboard/dashboard_real.html#v=asset&a=hagenholz&mo=2021-11
- Code: https://github.com/Gemmagf/swiss-governance-dashboard — package `src/asset/` (this repo's
  `src/zuri_kreislauf/`, MIT notice kept), tests in `tests/`, exporter `src/pipeline/export_asset.py`,
  documentation in the cockpit README (section *Asset drill-down (KVA Hagenholz)*).

The forecasting (SARIMA vs seasonal naive), the Hotelling T²/Q + IsolationForest monitor, the CHF/tCO₂eq
decision layer and the turbine-outage episode (Sep 2021 – Apr 2022) are unchanged; their results are now
precomputed from the same City of Zurich open data and embedded in a single static HTML file — no Streamlit.

This repository is kept read-only for reference (history, notebooks-era layout, original README in git history).
