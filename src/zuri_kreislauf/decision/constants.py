"""Named economic and emission constants for the decision layer (Layer 4).

Every number here is either a cited public figure or an explicitly labelled
estimate — never a bare number in the calculation code. Retrieval date for
every source is 2026-07-17.
"""

from __future__ import annotations

# --- FX ---
# ECB/market spot, mid-July 2026 (~0.92-0.93 range observed). Used only to
# convert EUR-denominated commodity prices below into CHF.
EUR_TO_CHF = 0.93

# --- Energy tariffs ---

# District heating (Fernwärme) sold by ERZ/ewz in the city of Zurich.
# Average all-in price 11.5 Rp./kWh as of October 2025 (city of Zurich /
# Preisüberwacher reporting on the Wärmeverbund KVA+Holz tariff).
# Source: https://www.stadt-zuerich.ch (Fernwärmetarif 2027 coverage, Oct 2025 figure)
HEAT_PRICE_CHF_PER_KWH = 0.115

# Industrial electricity price, "C4" consumption profile (500 MWh/yr), 2026
# average across Swiss grid operators.
# Source: ElCom, https://www.strompreis.elcom.admin.ch (2026 tariff dataset)
ELECTRICITY_PRICE_CHF_PER_KWH = 0.236

# --- Emission factors ---

# Swiss electricity CONSUMPTION mix (Verbrauchsmix, includes imports) — the
# right factor for "what does a kWh not produced here cost in emissions
# elsewhere", since Switzerland is a large net electricity importer.
# Domestic PRODUCTION alone is much cleaner (~30 g/kWh, mostly hydro+nuclear)
# but that is not the relevant counterfactual for lost plant output.
# Source: energie-umwelt.ch / BFE Elektrizitätsstatistik 2024 commentary,
# consumption-mix figure of 113 gCO2eq/kWh for 2024.
ELECTRICITY_EMISSION_FACTOR_KG_CO2_PER_KWH = 0.113

# Natural gas combustion, heating-value basis — used as the marginal
# heat-supply counterfactual for district-heat shortfalls (i.e. a gas boiler
# is what would run harder if the KVA supplied less heat).
# Source: BAFU, "CO2-Emissionsfaktoren des Treibhausgasinventars der Schweiz",
# factsheet dated 01-2025 (2024 data): 0.201 kg CO2/kWh (Heizwert-basis).
GAS_EMISSION_FACTOR_KG_CO2_PER_KWH = 0.201

# --- Reagent unit prices ---
# All four are European bulk commodity-chemical market prices (not an ERZ
# procurement price, which isn't public), retrieved 2026-07-17 and converted
# EUR -> CHF at the rate above. These are the weakest-sourced constants in
# this file — treat kg/t deviations converted to CHF via these prices as
# order-of-magnitude, not exact.

# Caustic soda (NaOH), liquid bulk, Europe: ~EUR 480-540/t (early-2025
# assessments) -> midpoint ~EUR 510/t.
# Source: elchemy.com Caustic Soda Price Index, chemanalyst.com
NAOH_PRICE_CHF_PER_KG = 510 * EUR_TO_CHF / 1000

# Hydrochloric acid (HCl, ~30%), Europe: reported in the ~USD 106-174/t range
# through late 2025 (soft market); treated here as EUR-equivalent given
# EUR/USD proximity in this period -> midpoint ~EUR 140/t.
# Source: chemanalyst.com, imarcgroup.com Hydrochloric Acid Price Trend
HCL_PRICE_CHF_PER_KG = 140 * EUR_TO_CHF / 1000

# Quicklime, Europe, Q1 2026: ~USD 145-170/t across France/Netherlands ->
# treated as EUR-equivalent, midpoint ~EUR 157/t.
# Source: imarcgroup.com Quicklime Pricing Report Q1 2026
LIME_PRICE_CHF_PER_KG = 157 * EUR_TO_CHF / 1000

# Ammonia solution (~25%, used for SNCR NOx reduction) — no Zurich- or
# even Europe-specific bulk price for the *diluted solution* was found
# (only anhydrous NH3 spot prices, a different product). This is a rough
# planning estimate based on typical European water/wastewater-treatment
# reagent procurement ranges (~EUR 400-600/t of solution), NOT a fetched
# market quote. Flagged explicitly as the least reliable price in this file.
AMMONIA_PRICE_CHF_PER_KG = 500 * EUR_TO_CHF / 1000  # ASSUMPTION, not a sourced quote
