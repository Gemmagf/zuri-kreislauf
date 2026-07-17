"""Run every layer end-to-end and write a summary to artifacts/results.json.

This is the single reproducible entry point referenced by `make all` /
the project brief's "Reproducible: make data && make all" requirement.
"""

from __future__ import annotations

import json
import warnings

from zuri_kreislauf.config import ARTIFACTS_DIR, ensure_dirs
from zuri_kreislauf.data.loader import load_khkw
from zuri_kreislauf.data.preprocess import MONITORING_COLUMNS, build_intensity_ratios
from zuri_kreislauf.decision.cost import monthly_impact
from zuri_kreislauf.detection.iforest import run_iforest
from zuri_kreislauf.detection.spc import run_spc
from zuri_kreislauf.forecast.backtest import expanding_window_backtest, summarize_backtest


def main() -> None:
    warnings.filterwarnings("ignore")
    ensure_dirs()

    khkw = load_khkw()
    ratios = build_intensity_ratios(khkw)

    # --- Layer 2: anomaly detection ---
    spc = run_spc(ratios, MONITORING_COLUMNS, alpha=0.01)
    iforest = run_iforest(ratios, MONITORING_COLUMNS, contamination=0.1)
    normal_mask = ~spc.flags.reindex(ratios.index, fill_value=False)

    # --- Layer 1: forecast backtest (horizons 1/3/6 months) ---
    series = khkw["Kehrichtdurchsatz"].asfreq("MS")
    backtest_summary = {}
    for h in (1, 3, 6):
        results = expanding_window_backtest(series, min_train_size=48, horizon=h)
        backtest_summary[f"horizon_{h}m"] = (
            summarize_backtest(results).round(2).to_dict(orient="index")
        )

    # --- Layer 4: decision layer ---
    impact = monthly_impact(khkw, ratios, normal_mask)
    outage_window = impact.loc["2021-09":"2022-04"]

    results = {
        "data": {
            "n_months": len(khkw),
            "date_range": [str(khkw.index.min().date()), str(khkw.index.max().date())],
        },
        "forecast_backtest": backtest_summary,
        "anomaly_detection": {
            "t2_ucl": round(float(spc.t2_ucl), 2),
            "q_ucl": round(float(spc.q_ucl), 2),
            "n_months_flagged_spc": int(spc.flags.sum()),
            "n_months_flagged_iforest": int(iforest["flag"].sum()),
            "n_months_flagged_both": int(
                (spc.flags.reindex(iforest.index) & iforest["flag"]).sum()
            ),
        },
        "decision_layer": {
            "total_chf_all_months": round(float(impact["total_chf"].sum()), 0),
            "total_tco2eq_all_months": round(float(impact["total_co2_kg"].sum() / 1000), 1),
            "turbine_outage_2021_09_to_2022_04": {
                "electricity_chf_lost": round(float(outage_window["electricity_chf"].sum()), 0),
                "electricity_tco2eq": round(
                    float(outage_window["electricity_co2_kg"].sum() / 1000), 1
                ),
                "heat_chf_offset": round(float(outage_window["heat_chf"].sum()), 0),
            },
        },
    }

    out_path = ARTIFACTS_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
