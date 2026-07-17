"""Züri-Kreislauf dashboard — bilingual (DE/EN) Streamlit app.

Run with `make app` or `streamlit run app/streamlit_app.py`.
"""

from __future__ import annotations

import warnings

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from zuri_kreislauf.data.loader import load_bioabfall_calendar, load_khkw
from zuri_kreislauf.data.preprocess import MONITORING_COLUMNS, build_intensity_ratios
from zuri_kreislauf.decision.cost import monthly_impact
from zuri_kreislauf.detection.spc import run_spc, top_contributors
from zuri_kreislauf.forecast.baseline import seasonal_naive_forecast
from zuri_kreislauf.forecast.sarima import fit_and_forecast
from zuri_kreislauf.logistics.collection import collections_per_month

warnings.filterwarnings("ignore")

T = {
    "en": {
        "title": "Züri-Kreislauf — Waste-to-Energy Analytics",
        "subtitle": "Forecasting, process monitoring and a CHF/CO2eq decision layer for the "
        "KVA Hagenholz waste-to-energy plant, built on open City of Zurich data.",
        "lang_label": "Language",
        "tab_overview": "Overview",
        "tab_forecast": "Forecast",
        "tab_anomaly": "Anomaly detection",
        "tab_decision": "Decision layer (CHF / CO2)",
        "tab_logistics": "Collection logistics",
        "throughput": "Monthly waste throughput",
        "kpi_months": "Months of data",
        "kpi_range": "Date range",
        "kpi_last_throughput": "Latest month throughput",
        "forecast_header": "SARIMA vs. seasonal-naive baseline",
        "forecast_note": "Backtested with expanding-window validation. On this 78-month "
        "series, the seasonal-naive baseline is competitive with — and at longer "
        "horizons beats — the fitted SARIMA model. That's a real, honest finding, "
        "not a bug: with this little data, added model complexity isn't clearly "
        "paying for itself.",
        "horizon": "Backtest horizon (months)",
        "mape": "MAPE (%)",
        "mae": "MAE (t)",
        "next_forecast": "Next-6-month forecast (SARIMA, 95% CI)",
        "anomaly_header": "Hotelling T2 / Q monitor over 7 process-intensity ratios",
        "anomaly_note": "Flags months where energy yield, reagent consumption or residual "
        "fraction deviate jointly from the plant's own historical pattern.",
        "select_month": "Inspect a flagged month",
        "top_drivers": "Top contributing features",
        "decision_header": "Monthly CHF and tCO2eq impact vs. normal-month baseline",
        "decision_note": "Positive = cost / extra emissions vs. this plant's own recent "
        "normal-operation baseline. All prices and emission factors are documented, "
        "sourced constants — see the README.",
        "total_chf": "Total CHF impact (all months)",
        "total_co2": "Total tCO2eq impact (all months)",
        "logistics_header": "Bio-waste collection frequency by postcode (PLZ)",
        "logistics_note": "PLZ 8064 (Werdhölzli) is Biogas Zürich AG's own neighbourhood.",
    },
    "de": {
        "title": "Züri-Kreislauf — Waste-to-Energy Analytics",
        "subtitle": "Prognose, Prozessüberwachung und eine CHF/CO2eq-Entscheidungsebene für "
        "die Kehrichtverwertungsanlage Hagenholz, basierend auf offenen Daten der Stadt Zürich.",
        "lang_label": "Sprache",
        "tab_overview": "Übersicht",
        "tab_forecast": "Prognose",
        "tab_anomaly": "Anomalieerkennung",
        "tab_decision": "Entscheidungsebene (CHF / CO2)",
        "tab_logistics": "Sammellogistik",
        "throughput": "Monatlicher Kehrichtdurchsatz",
        "kpi_months": "Monate mit Daten",
        "kpi_range": "Zeitraum",
        "kpi_last_throughput": "Durchsatz letzter Monat",
        "forecast_header": "SARIMA vs. saisonale Naiv-Baseline",
        "forecast_note": "Backtest mit expandierendem Zeitfenster. Bei dieser 78-monatigen "
        "Reihe ist die saisonale Naiv-Baseline konkurrenzfähig mit — und bei längeren "
        "Horizonten besser als — das SARIMA-Modell. Das ist ein ehrliches Ergebnis: bei "
        "so wenig Daten lohnt sich zusätzliche Modellkomplexität nicht eindeutig.",
        "horizon": "Backtest-Horizont (Monate)",
        "mape": "MAPE (%)",
        "mae": "MAE (t)",
        "next_forecast": "Prognose nächste 6 Monate (SARIMA, 95%-KI)",
        "anomaly_header": "Hotelling-T2-/Q-Überwachung über 7 Prozessintensitäts-Kennzahlen",
        "anomaly_note": "Markiert Monate, in denen Energieausbeute, Reagenzienverbrauch "
        "oder Reststoffanteil gemeinsam vom historischen Muster der Anlage abweichen.",
        "select_month": "Markierten Monat untersuchen",
        "top_drivers": "Wichtigste Einflussgrössen",
        "decision_header": "Monatliche CHF- und tCO2eq-Auswirkung vs. Normalbetrieb-Baseline",
        "decision_note": "Positiv = Kosten / zusätzliche Emissionen gegenüber dem eigenen, "
        "jüngsten Normalbetrieb der Anlage. Alle Preise und Emissionsfaktoren sind "
        "dokumentierte, belegte Konstanten — siehe README.",
        "total_chf": "CHF-Gesamtwirkung (alle Monate)",
        "total_co2": "tCO2eq-Gesamtwirkung (alle Monate)",
        "logistics_header": "Bioabfall-Sammelhäufigkeit nach Postleitzahl (PLZ)",
        "logistics_note": "PLZ 8064 (Werdhölzli) ist der Standort von Biogas Zürich AG.",
    },
}


@st.cache_data
def get_data():
    khkw = load_khkw()
    ratios = build_intensity_ratios(khkw)
    calendar = load_bioabfall_calendar()
    return khkw, ratios, calendar


@st.cache_data
def get_spc(ratios: pd.DataFrame):
    return run_spc(ratios, MONITORING_COLUMNS, alpha=0.01)


@st.cache_data
def get_forecast(series: pd.Series, horizon: int = 6):
    sarima = fit_and_forecast(series, horizon)
    naive = seasonal_naive_forecast(series, horizon)
    return sarima, naive


def main():
    st.set_page_config(page_title="Züri-Kreislauf", layout="wide")

    lang = st.sidebar.selectbox("Language / Sprache", options=["en", "de"], index=0)
    t = T[lang]

    st.title(t["title"])
    st.caption(t["subtitle"])

    khkw, ratios, calendar = get_data()
    spc = get_spc(ratios)
    normal_mask = ~spc.flags.reindex(ratios.index, fill_value=False)
    impact = monthly_impact(khkw, ratios, normal_mask)

    tab_overview, tab_forecast, tab_anomaly, tab_decision, tab_logistics = st.tabs(
        [t["tab_overview"], t["tab_forecast"], t["tab_anomaly"], t["tab_decision"], t["tab_logistics"]]
    )

    with tab_overview:
        col1, col2, col3 = st.columns(3)
        col1.metric(t["kpi_months"], len(khkw))
        col2.metric(t["kpi_range"], f"{khkw.index.min():%Y-%m} → {khkw.index.max():%Y-%m}")
        col3.metric(t["kpi_last_throughput"], f"{khkw['Kehrichtdurchsatz'].iloc[-1]:,.0f} t")

        st.subheader(t["throughput"])
        fig = go.Figure()
        fig.add_scatter(x=khkw.index, y=khkw["Kehrichtdurchsatz"], mode="lines+markers")
        st.plotly_chart(fig, use_container_width=True)

    with tab_forecast:
        st.subheader(t["forecast_header"])
        st.write(t["forecast_note"])

        series = khkw["Kehrichtdurchsatz"].asfreq("MS")
        sarima, naive = get_forecast(series)

        fig = go.Figure()
        fig.add_scatter(x=series.index, y=series, name="actual", mode="lines")
        fig.add_scatter(x=sarima.mean.index, y=sarima.mean, name="SARIMA forecast", mode="lines")
        fig.add_scatter(
            x=list(sarima.mean.index) + list(sarima.mean.index[::-1]),
            y=list(sarima.ci_upper) + list(sarima.ci_lower[::-1]),
            fill="toself",
            fillcolor="rgba(99,110,250,0.15)",
            line={"color": "rgba(255,255,255,0)"},
            name="95% CI",
        )
        fig.add_scatter(
            x=naive.index, y=naive, name="seasonal-naive", mode="lines", line={"dash": "dot"}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(t["next_forecast"])

    with tab_anomaly:
        st.subheader(t["anomaly_header"])
        st.write(t["anomaly_note"])

        fig = go.Figure()
        fig.add_scatter(x=spc.t2.index, y=spc.t2, name="T2", mode="lines+markers")
        fig.add_hline(y=spc.t2_ucl, line_dash="dash", annotation_text="UCL")
        st.plotly_chart(fig, use_container_width=True)

        flagged_months = spc.flags[spc.flags].index
        if len(flagged_months):
            month = st.selectbox(t["select_month"], options=list(flagged_months)[::-1])
            st.write(t["top_drivers"])
            st.dataframe(top_contributors(spc.t2_contributions, month, k=5).rename("contribution"))

    with tab_decision:
        st.subheader(t["decision_header"])
        st.write(t["decision_note"])

        col1, col2 = st.columns(2)
        col1.metric(t["total_chf"], f"{impact['total_chf'].sum():,.0f} CHF")
        col2.metric(t["total_co2"], f"{impact['total_co2_kg'].sum() / 1000:,.1f} t")

        fig = go.Figure()
        fig.add_bar(x=impact.index, y=impact["total_chf"], name="CHF")
        st.plotly_chart(fig, use_container_width=True)

    with tab_logistics:
        st.subheader(t["logistics_header"])
        st.caption(t["logistics_note"])
        freq = collections_per_month(calendar)
        if 8064 in freq.columns:
            fig = go.Figure()
            fig.add_bar(x=freq.index.astype(str), y=freq[8064], name="PLZ 8064")
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
