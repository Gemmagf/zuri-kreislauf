# Züri-Kreislauf — Zurich Waste-to-Energy Analytics

Forecasting, process-efficiency monitoring and a CHF/tCO2eq decision layer for the
**KVA Hagenholz** waste-to-energy plant (operated by ERZ, Entsorgung + Recycling Zürich),
built entirely on open City of Zurich data. The goal: show that a handful of monthly
operating figures — waste burned, energy sold, reagents consumed — are enough to forecast
load, catch real operational events, and translate them into money and carbon, provided the
statistical method is matched to how little data there actually is.

A live example of what this surfaces: for eight months (Sep 2021–Apr 2022) the plant's
generator turbine went offline. Electricity output cratered while heat sales rose — a classic
combined-heat-and-power trade-off — and the multivariate monitor in Layer 2 flags exactly this
window from the ratios alone, with `electricity_kwh_per_t` as the dominant driver every single
month. Layer 4 prices it at roughly **CHF 18.8M** in lost electricity value and **~9,000
tCO2eq** in displaced grid emissions, partly offset by **~CHF 8.8M** in extra heat revenue.
See [Results](#results) below.

## Architecture

```
data.stadt-zuerich.ch (3 CSVs)
        │  scripts/download_data.py  (cached under data/raw/, idempotent)
        ▼
┌─────────────────────┐
│ data/loader.py       │  parse, dtype, Swiss-thousands-separator fix
│ data/preprocess.py   │  build 7 process-intensity ratios (kWh/t, kg-reagent/t, ...)
└──────────┬───────────┘
           │
   ┌───────┼────────────────────┬───────────────────────┐
   ▼                            ▼                        ▼
Layer 1: forecast/       Layer 2: detection/       Layer 3: logistics/
SARIMA vs.               Hotelling T2/Q +          bio-waste collection
seasonal-naive,          IsolationForest on         calendar by PLZ
expanding-window         the 7 ratios, T2
backtest                 contribution attribution
   │                            │
   └────────────┬───────────────┘
                ▼
        Layer 4: decision/
        flagged-month deviations × sourced CHF/tCO2eq
        constants → cost.py
                │
                ▼
        app/streamlit_app.py  (bilingual EN/DE dashboard)
```

## Results

Numbers below are written by `scripts/run_all.py` to `artifacts/results.json` — run it
yourself (`make all`) rather than trusting this table blindly.

**Forecast backtest** (expanding-window, 78-month series, SARIMA (0,1,1)(0,1,1)[12] vs.
seasonal-naive baseline):

| Horizon | SARIMA MAPE | Naive MAPE | SARIMA MAE (t) | Naive MAE (t) |
|---|---|---|---|---|
| 1 month | 12.56% | 12.49% | 2,298 | 2,197 |
| 3 months | 13.74% | 12.66% | 2,440 | 2,194 |
| 6 months | 15.80% | **11.39%** | 2,972 | **2,081** |

**The seasonal-naive baseline is competitive with, and at longer horizons clearly beats,
the fitted SARIMA model.** This is reported as a finding, not hidden: with 78 monthly
points, the added complexity of SARIMA is not earning its keep, especially past a 1-month
horizon. The dashboard and this README lead with that conclusion rather than the SARIMA
numbers alone.

**Anomaly detection** (Hotelling T2/Q on 7 intensity ratios, α=0.01):
- T2 UCL 22.31, Q UCL 8.06
- 20/78 months flagged by T2/Q, 8/78 by IsolationForest, 7 flagged by both
- Dominant, explainable episode: **Sep 2021–Apr 2022**, the turbine-outage window above

**Decision layer** (flagged-month deviations vs. each ratio's own normal-month baseline):
- Turbine outage (8 months): **CHF 18.84M** electricity value lost, **9,022 tCO2eq**
  extra grid-side emissions, **CHF 8.78M** offsetting extra heat revenue
- Sum across all 78 months: **CHF 12.66M** net, **–8,703 tCO2eq** net (a large negative
  because many non-outage months score as savings against the baseline — see
  [Limitations](#assumptions--limitations) on what this total does and doesn't mean)

## Data provenance

All three datasets are open CSV, no authentication, retrieved **2026-07-17**. Download URLs
302-redirect — `scripts/download_data.py` follows redirects and caches under `data/raw/`
(not committed; re-run `make data` after a clean clone).

| Dataset | Rows | Range | Use |
|---|---|---|---|
| [erz_abfallmenge_energie_khkw](https://data.stadt-zuerich.ch/dataset/erz_abfallmenge_energie_khkw) | 78 | 2020-01 → 2026-06 | Primary: monthly KVA Hagenholz operating data |
| [entsorgungskalender_bioabfall](https://data.stadt-zuerich.ch/dataset/entsorgungskalender_bioabfall) | 1,253 | 2026 (full year) | Bio-waste collection calendar by PLZ |
| [erz_elog_kennzahlen](https://data.stadt-zuerich.ch/dataset/erz_elog_kennzahlen) | 1,427 | 2023-01 → 2026-04 | City-wide waste/recycling KPIs (context) |

Column units for the primary dataset were **not** obvious from the CSV header alone — e.g.
`Ammoniakverbrauch` looks like it could be kg or tonnes. Confirmed via the dataset's CKAN
`sszFields` metadata (fetched from `data.stadt-zuerich.ch`'s API) that all four reagent
columns and drinking-water consumption are reported in **tonnes**, not kilograms; this
project converts to kg/t for the intensity ratios. Getting this wrong would have made every
downstream ratio and CHF figure off by 1000×, so it's called out here explicitly.

## Assumptions & Limitations

**n=78.** The primary series is 78 monthly observations. This ruled out any deep-learning
detector or forecaster — see [Layer 1's results](#results), where even a lightly-parameterised
SARIMA doesn't clearly beat a seasonal-naive baseline. Classical multivariate SPC (Hotelling
T2/Q with a robust covariance estimate) was chosen for anomaly detection for the same reason:
it has known control limits derived analytically for small samples, rather than requiring
enough data to learn a decision boundary.

**Decision-layer constants are a mix of solid and rough sources** — see
[`src/zuri_kreislauf/decision/constants.py`](src/zuri_kreislauf/decision/constants.py) for
every value with its citation:
- **Well-sourced**: Zurich district-heat tariff (11.5 Rp/kWh, city of Zurich, Oct 2025),
  Swiss industrial electricity price (23.6 Rp/kWh, ElCom 2026), Swiss electricity
  consumption-mix carbon intensity (113 gCO2eq/kWh, 2024), natural-gas combustion emission
  factor (0.201 kgCO2/kWh, BAFU, 2024 data).
- **Rougher**: the four reagent unit prices (ammonia, caustic soda, hydrochloric acid, lime)
  are European bulk commodity-market prices retrieved 2026-07-17, not ERZ's actual
  procurement prices (not public). The ammonia solution price in particular is a planning
  estimate, not a fetched market quote — flagged as such in the source code.

**The CHF/tCO2eq decision layer ranks and explains episodes; it is not an audited total.**
Each ratio's "expected" value is the median of non-flagged months — a simple, transparent
baseline, but one estimated from as few as ~58 "normal" months once anomalies are excluded.
The all-history sum (CHF 12.66M, –8,703 tCO2eq) is useful to see that most of the swing comes
from one real event, not as a certified annual cost figure.

**No COVID-19 level shift was found.** The brief expected one; 2020's monthly-throughput mean
(20,854 t) is in line with every other year in the series. Reported as a (non-)finding rather
than forced into the model.

**Layer 3 is intentionally light** — a collection-frequency view by postcode, not a routing
or scheduling optimiser, per scope.

## Reproducing this

```bash
make install   # create .venv, install package + dev/app extras
make data      # fetch and cache the 3 CSVs (idempotent)
make test      # pytest — data-contract tests skip if data isn't cached yet
make all       # data + scripts/run_all.py -> artifacts/results.json
make app       # streamlit run app/streamlit_app.py (EN/DE toggle)
```

Requires Python 3.11+. CI (`.github/workflows/ci.yml`) runs ruff + pytest on 3.11 and 3.12.

## Tech stack

Python, pandas/numpy/scipy, scikit-learn (IsolationForest, robust covariance, PCA),
statsmodels (SARIMAX), a from-scratch Kourti–MacGregor-style T2 contribution decomposition
(no SHAP dependency — see [Layer 2](#architecture)), Streamlit + Plotly for the dashboard,
pytest + ruff + GitHub Actions for CI, hatchling for packaging.

## License

MIT — see [LICENSE](LICENSE).
