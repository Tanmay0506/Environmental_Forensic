"""Rule-based interpretation of environmental indicator changes.

Interpretations are hypotheses supported by observed indicator direction and
cross-indicator consistency. They are not proof of causation.
"""
from __future__ import annotations
from typing import Any
import math
import pandas as pd


def _num(value: Any) -> float | None:
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _row(indicator: str, change: str, value: Any, unit: str, interpretation: str,
         possible_reasons: list[str], evidence: list[str], confidence: str,
         caveat: str) -> dict[str, Any]:
    return {"Indicator": indicator, "Direction": change, "Observed change": value,
            "Unit": unit, "Interpretation": interpretation,
            "Possible explanations (not confirmed causes)": "; ".join(possible_reasons),
            "Supporting evidence": "; ".join(evidence) if evidence else "No corroborating indicator available",
            "Evidence level": confidence, "Caveat": caveat}


def build_change_interpretation(*, historical: dict[str, Any] | None,
                                climate: dict[str, Any] | None,
                                landuse: dict[str, Any] | None,
                                target_year: int | None = None) -> dict[str, Any]:
    """Create plain-language explanations from Modules 1–3 outputs."""
    historical, climate, landuse = historical or {}, climate or {}, landuse or {}
    year = int(target_year or historical.get("comparison_year") or landuse.get("comparison_year") or 0)
    items: list[dict[str, Any]] = []
    cross_evidence: list[str] = []

    # Module 1: NDVI and mapped vegetation change
    delta = _num(historical.get("mean_delta"))
    decline = _num(historical.get("decline_pct"))
    gain = _num(historical.get("gain_pct"))
    if delta is not None:
        if delta <= -0.05:
            direction, interpretation = "Decreased", "Average vegetation greenness declined materially between the selected observations."
            reasons = ["rainfall deficit or soil-water stress", "heat stress", "crop/season timing differences", "grazing, fire, pests, harvest or land conversion"]
        elif delta >= 0.05:
            direction, interpretation = "Increased", "Average vegetation greenness increased between the selected observations."
            reasons = ["higher rainfall or irrigation availability", "crop growth or seasonal timing", "vegetation recovery or planting", "different atmospheric or residual cloud conditions"]
        else:
            direction, interpretation = "Little average change", "Mean NDVI change is small; local patches may still have gained or lost vegetation."
            reasons = ["stable conditions", "opposing local gains and declines that average out", "change smaller than the chosen threshold"]
        evidence = []
        if decline is not None: evidence.append(f"Mapped decline covers {decline:.1f}% of valid area")
        if gain is not None: evidence.append(f"Mapped gain covers {gain:.1f}% of valid area")
        if delta < -0.05 and decline is not None and decline >= 10: cross_evidence.append("NDVI decline is accompanied by a meaningful mapped decline area")
        items.append(_row("Vegetation greenness (mean NDVI)", direction, round(delta, 4), "NDVI", interpretation, reasons, evidence, "Moderate indicator evidence", "NDVI measures spectral greenness, not biomass, species health, or vegetation area directly. Compare similar seasons and cloud quality."))
    else:
        items.append(_row("Vegetation greenness (mean NDVI)", "Unavailable", None, "NDVI", "Module 1 has no usable mean NDVI change value.", ["Run Module 1 for two comparable years"], [], "Insufficient data", "No direction is inferred without a measured change."))

    # Module 2: annual climate anomalies for selected year
    annual = climate.get("annual_anomalies")
    climate_row = None
    if isinstance(annual, pd.DataFrame) and not annual.empty and "year" in annual.columns:
        matches = annual[pd.to_numeric(annual["year"], errors="coerce") == year] if year else annual.iloc[0:0]
        if not matches.empty: climate_row = matches.iloc[-1]
    if climate_row is not None:
        rain = _num(climate_row.get("rainfall_anomaly_pct"))
        temp = _num(climate_row.get("temperature_anomaly_c"))
        soil = _num(climate_row.get("topsoil_moisture_anomaly"))
        if rain is not None:
            direction = "Below baseline" if rain <= -10 else "Above baseline" if rain >= 10 else "Near baseline"
            reasons = (["reduced precipitation during the year", "monsoon timing or distribution differences"] if rain <= -10 else ["higher precipitation than baseline", "monsoon timing or distribution differences"] if rain >= 10 else ["annual rainfall is within ±10% of the selected baseline"])
            evidence = []
            if soil is not None and ((rain < 0 and soil < 0) or (rain > 0 and soil > 0)): evidence.append("Topsoil-moisture anomaly has the same direction")
            if temp is not None and temp >= 1 and rain <= -10: evidence.append("Warm anomaly co-occurs with rainfall deficit")
            items.append(_row("Annual rainfall", direction, round(rain, 2), "% vs baseline", "Annual rainfall was lower/higher than the selected baseline." if abs(rain) >= 10 else "Annual rainfall was relatively close to the selected baseline.", reasons, evidence, "Moderate climate indicator evidence", "Annual totals can hide dry spells, intense rainfall, and seasonal shifts. Baseline period strongly affects anomaly."))
            if rain <= -10: cross_evidence.append("Annual rainfall deficit may contribute to vegetation or soil-moisture stress")
        if temp is not None:
            direction = "Warmer" if temp >= 0.5 else "Cooler" if temp <= -0.5 else "Near baseline"
            reasons = (["regional warming or a warmer-than-baseline year", "cloud cover and circulation differences"] if temp >= .5 else ["cooler-than-baseline conditions", "cloud cover and circulation differences"] if temp <= -.5 else ["annual mean temperature is close to baseline"])
            evidence = []
            if rain is not None and temp >= .5 and rain <= -10: evidence.append("Warmer conditions coincide with rainfall deficit")
            items.append(_row("Mean air temperature (ERA5-Land)", direction, round(temp, 3), "°C anomaly", "Annual mean air temperature was warmer/cooler than the selected baseline." if abs(temp) >= .5 else "Annual mean air temperature was close to baseline.", reasons, evidence, "Moderate climate indicator evidence", "This is reanalysis air temperature near the AOI centroid, not satellite land-surface temperature or a field thermometer."))
            if temp >= 1: cross_evidence.append("A positive temperature anomaly may increase evaporative demand")
        if soil is not None:
            direction = "Drier" if soil <= -.02 else "Wetter" if soil >= .02 else "Near baseline"
            reasons = (["rainfall deficit", "higher evaporative demand", "irrigation or soil drainage changes"] if soil <= -.02 else ["greater recent soil-water availability", "rainfall or irrigation differences"] if soil >= .02 else ["topsoil moisture anomaly is small relative to baseline"])
            evidence = []
            if rain is not None and ((soil < 0 and rain < 0) or (soil > 0 and rain > 0)): evidence.append("Rainfall and topsoil-moisture anomalies agree")
            items.append(_row("Topsoil moisture (0–7 cm)", direction, round(soil, 4), "m³/m³ anomaly", "Modelled topsoil moisture was lower/higher than the selected baseline." if abs(soil) >= .02 else "Modelled topsoil moisture was close to baseline.", reasons, evidence, "Moderate climate indicator evidence", "ERA5-Land estimates are grid-cell values and may not represent field soil moisture or deeper root-zone water."))
            if soil <= -.02: cross_evidence.append("Negative topsoil-moisture anomaly supports a soil-water-stress hypothesis")
    else:
        for name, unit in [("Annual rainfall", "% vs baseline"), ("Mean air temperature (ERA5-Land)", "°C anomaly"), ("Topsoil moisture (0–7 cm)", "m³/m³ anomaly")]:
            items.append(_row(name, "Unavailable", None, unit, f"No climate anomaly was found for investigation year {year or 'selected'}.", ["Run Module 2 and ensure the selected year is included"], [], "Insufficient data", "No change direction is inferred without a matching annual anomaly."))

    # Module 3: class area changes and transitions
    class_area = landuse.get("class_area")
    if isinstance(class_area, dict) and class_area:
        for cls, info in class_area.items():
            change_ha = _num(info.get("change_ha")) if isinstance(info, dict) else None
            if change_ha is None: continue
            direction = "Expanded" if change_ha > 0.05 else "Contracted" if change_ha < -0.05 else "Approximately stable"
            explanations = {
                "Vegetation": (["vegetation recovery, planting or wetter conditions", "crop-cycle or seasonal differences"] if change_ha > .05 else ["vegetation stress, harvest, grazing or fire", "conversion to another cover class"]),
                "Water proxy": (["greater water extent or temporary inundation", "seasonal water-level differences"] if change_ha > .05 else ["reduced water extent or seasonal drying", "water-level or classification differences"]),
                "Built-up / impervious proxy": (["possible built-up/impervious expansion", "bare soil or bright surfaces misclassified as built-up"] if change_ha > .05 else ["possible reduced mapped built-up area", "classification or image-timing differences"]),
                "Bare soil / sparse cover": (["exposed soil, harvesting, degradation or construction", "sparse vegetation or moisture differences"] if change_ha > .05 else ["vegetation cover recovery or seasonal greening", "classification differences"]),
            }
            why = explanations.get(cls, ["land-cover proxy change", "seasonal or classification differences"])
            if cls == "Vegetation" and change_ha < -.05: cross_evidence.append("Module 3 shows a reduction in mapped vegetation proxy area")
            items.append(_row(f"Land-cover proxy: {cls}", direction, round(change_ha, 3), "ha change", f"The mapped {cls.lower()} proxy area changed by {change_ha:+.2f} ha.", why, ["Two-date Sentinel-2 threshold-based class comparison"], "Screening-level evidence", "These are spectral proxy classes, not a validated land-cover map. Confirm with reference imagery/data before claiming real conversion."))
    else:
        items.append(_row("Land-cover class areas", "Unavailable", None, "ha change", "Module 3 class-area statistics are unavailable.", ["Run Module 3 for the same AOI and comparable seasonal dates"], [], "Insufficient data", "No land-cover direction is inferred."))

    if not cross_evidence:
        cross_evidence.append("No strong cross-indicator pattern could be established from the currently available outputs; inspect individual indicators and data coverage.")
    summary = ("Several indicators are consistent with environmental stress; treat the explanations as hypotheses, not confirmed causes." if any("deficit" in x.lower() or "stress" in x.lower() or "reduction" in x.lower() for x in cross_evidence) else "Interpret indicators individually; available outputs do not establish a single dominant cause.")
    return {"investigation_year": year, "baseline_context": {
                "vegetation_baseline_year": historical.get("baseline_year"),
                "vegetation_comparison_year": historical.get("comparison_year"),
                "climate_baseline_start_year": climate.get("baseline_start_year"),
                "climate_baseline_end_year": climate.get("baseline_end_year"),
                "landuse_baseline_year": landuse.get("baseline_year"),
                "landuse_comparison_year": landuse.get("comparison_year")},
            "summary": summary, "indicators": pd.DataFrame(items),
            "cross_indicator_evidence": cross_evidence,
            "limitations": ["Possible reasons are rule-based hypotheses, not causal proof.", "Different sensors, seasonal windows, baselines, and spatial resolutions can produce apparent disagreements.", "NDVI is spectral greenness; it does not directly measure vegetation area or biomass.", "ERA5-Land climate values are grid-cell estimates, not field measurements.", "Land-cover outputs are threshold-based spectral proxies and may confuse built-up and bare soil.", "If the baseline/comparison periods are not seasonally comparable, interpretation confidence should be reduced."]}
