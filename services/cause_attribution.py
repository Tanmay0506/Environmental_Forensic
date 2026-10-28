"""Evidence-based environmental cause-attribution forensics.

This module combines outputs from historical vegetation, climate stress, and
land-use transition analyses. Scores rank evidence support; they are NOT
probabilities and do not prove causation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from pyproj import CRS as PyprojCRS


def _bounded_score(value: float, thresholds: list[tuple[float, float]], default: float = 0.0) -> float:
    """Return score from first descending threshold met."""
    if not np.isfinite(value):
        return default
    for threshold, score in thresholds:
        if value <= threshold:
            return float(score)
    return float(default)


def _raster_overlap_evidence(historical: dict[str, Any], landuse: dict[str, Any]) -> dict[str, Any]:
    """Align land-use change classes to historical vegetation classes and measure overlap."""
    result = {
        "available": False,
        "decline_pixels": 0,
        "decline_area_ha": float(historical.get("decline_area_ha", 0.0) or 0.0),
        "decline_to_builtup_ha": None,
        "decline_to_bare_ha": None,
        "decline_to_water_ha": None,
        "builtup_overlap_pct": None,
        "bare_overlap_pct": None,
        "water_overlap_pct": None,
        "message": "Spatial overlap could not be calculated from the available raster outputs.",
    }
    hist_path = historical.get("class_path")
    land_path = landuse.get("class_path")
    if not hist_path or not land_path or not Path(hist_path).exists() or not Path(land_path).exists():
        return result

    with rasterio.open(hist_path) as hist_src, rasterio.open(land_path) as lu_src:
        hist = hist_src.read(1)
        lu_aligned = np.zeros((hist_src.height, hist_src.width), dtype="uint8")
        reproject(
            source=rasterio.band(lu_src, 1), destination=lu_aligned,
            src_transform=lu_src.transform, src_crs=lu_src.crs, src_nodata=lu_src.nodata,
            dst_transform=hist_src.transform, dst_crs=hist_src.crs,
            dst_nodata=0, resampling=Resampling.nearest,
        )
        valid = (hist == 2) & np.isin(lu_aligned, [1, 2, 3, 4, 5])
        decline_pixels = int(np.count_nonzero(valid))
        if decline_pixels == 0:
            result.update(available=True, decline_pixels=0,
                          decline_to_builtup_ha=0.0, decline_to_bare_ha=0.0,
                          decline_to_water_ha=0.0, builtup_overlap_pct=0.0,
                          bare_overlap_pct=0.0, water_overlap_pct=0.0,
                          message="No overlapping pixels were labelled as vegetation decline by Module 1. This does not imply no environmental change occurred.")
            return result

        crs = PyprojCRS.from_user_input(hist_src.crs) if hist_src.crs else None
        if crs is None or not crs.is_projected:
            result["message"] = "Historical raster CRS is not projected; pixel-area overlap was not reported."
            return result
        axes = crs.axis_info
        unit_factor = float(axes[0].unit_conversion_factor) if axes else 1.0
        transform = hist_src.transform
        pixel_area_m2 = abs(transform.a * transform.e - transform.b * transform.d) * unit_factor**2
        area_ha = pixel_area_m2 / 10000.0
        # Module 3 codes: 2 vegetation->built-up proxy, 3 vegetation->bare,
        # 4 vegetation->water, 5 other transition, 1 unchanged proxy class.
        builtup_px = int(np.count_nonzero(valid & (lu_aligned == 2)))
        bare_px = int(np.count_nonzero(valid & (lu_aligned == 3)))
        water_px = int(np.count_nonzero(valid & (lu_aligned == 4)))
        result.update({
            "available": True,
            "decline_pixels": decline_pixels,
            "decline_area_ha": decline_pixels * area_ha,
            "decline_to_builtup_ha": builtup_px * area_ha,
            "decline_to_bare_ha": bare_px * area_ha,
            "decline_to_water_ha": water_px * area_ha,
            "builtup_overlap_pct": builtup_px / decline_pixels * 100.0,
            "bare_overlap_pct": bare_px / decline_pixels * 100.0,
            "water_overlap_pct": water_px / decline_pixels * 100.0,
            "message": "Land-use transition raster was nearest-neighbour aligned to the historical vegetation-change raster. Overlap inherits the uncertainty of both input classifications.",
        })
    return result


def _climate_evidence(climate: dict[str, Any], target_year: int) -> dict[str, Any]:
    annual = climate.get("annual_anomalies")
    output = {"available": False, "year_used": None, "rainfall_anomaly_pct": None,
              "temperature_anomaly_c": None, "soil_moisture_anomaly": None,
              "rainfall_score": None, "temperature_score": None, "soil_score": None,
              "message": "Climate results are not available. Run Module 2 first."}
    if not isinstance(annual, pd.DataFrame) or annual.empty or "year" not in annual:
        return output
    exact = annual[annual["year"].astype(int) == int(target_year)]
    if exact.empty:
        return output
    row = exact.iloc[-1]
    rain = pd.to_numeric(pd.Series([row.get("rainfall_anomaly_pct")]), errors="coerce").iloc[0]
    temp = pd.to_numeric(pd.Series([row.get("temperature_anomaly_c")]), errors="coerce").iloc[0]
    soil = pd.to_numeric(pd.Series([row.get("topsoil_moisture_anomaly")]), errors="coerce").iloc[0]
    rain_score = _bounded_score(rain, [(-30,100),(-20,80),(-10,60),(-5,35)])
    temp_score = 100 if pd.notna(temp) and temp >= 2.0 else 80 if pd.notna(temp) and temp >= 1.5 else 60 if pd.notna(temp) and temp >= 1.0 else 35 if pd.notna(temp) and temp >= 0.5 else 0
    soil_score = 100 if pd.notna(soil) and soil <= -0.05 else 80 if pd.notna(soil) and soil <= -0.03 else 60 if pd.notna(soil) and soil <= -0.02 else 35 if pd.notna(soil) and soil < 0 else 0
    output.update({"available": True, "year_used": int(row["year"]),
                   "rainfall_anomaly_pct": float(rain) if pd.notna(rain) else None,
                   "temperature_anomaly_c": float(temp) if pd.notna(temp) else None,
                   "soil_moisture_anomaly": float(soil) if pd.notna(soil) else None,
                   "rainfall_score": rain_score, "temperature_score": temp_score,
                   "soil_score": soil_score,
                   "message": "Climate indicators represent ERA5-Land grid-cell estimates near the AOI centroid, not field-level measurements."})
    return output


def build_cause_attribution(
    *, historical: dict[str, Any] | None,
    climate: dict[str, Any] | None,
    landuse: dict[str, Any] | None,
    target_year: int | None = None,
) -> dict[str, Any]:
    """Build transparent, explainable evidence scores from completed module results."""
    historical = historical or {}
    climate = climate or {}
    landuse = landuse or {}
    year = int(target_year or historical.get("comparison_year") or landuse.get("comparison_year") or 0)
    climate_ev = _climate_evidence(climate, year) if climate else _climate_evidence({}, year)
    overlap = _raster_overlap_evidence(historical, landuse) if historical and landuse else {
        "available": False, "decline_area_ha": float(historical.get("decline_area_ha", 0.0) or 0.0),
        "decline_to_builtup_ha": None, "decline_to_bare_ha": None, "decline_to_water_ha": None,
        "builtup_overlap_pct": None, "bare_overlap_pct": None, "water_overlap_pct": None,
        "message": "Run both Modules 1 and 3 to calculate spatial overlap."}

    scores: list[dict[str, Any]] = []
    if climate_ev["available"]:
        drought = (0.60 * climate_ev["rainfall_score"] + 0.25 * climate_ev["soil_score"] + 0.15 * climate_ev["temperature_score"])
        heat = (0.70 * climate_ev["temperature_score"] + 0.15 * climate_ev["rainfall_score"] + 0.15 * climate_ev["soil_score"])
        scores.append({"Hypothesis": "Drought / rainfall deficit", "Evidence score (0–100)": round(drought, 1),
                       "Evidence status": "Climate indicators available", "Basis": "Rainfall anomaly (60%), topsoil moisture anomaly (25%), temperature anomaly (15%)."})
        scores.append({"Hypothesis": "Heat stress", "Evidence score (0–100)": round(heat, 1),
                       "Evidence status": "Climate indicators available", "Basis": "Temperature anomaly (70%), rainfall anomaly (15%), topsoil moisture anomaly (15%)."})
    else:
        scores.extend([
            {"Hypothesis": "Drought / rainfall deficit", "Evidence score (0–100)": np.nan, "Evidence status": "Insufficient climate data", "Basis": "Run Module 2 and include the vegetation comparison year."},
            {"Hypothesis": "Heat stress", "Evidence score (0–100)": np.nan, "Evidence status": "Insufficient climate data", "Basis": "Run Module 2 and include the vegetation comparison year."},
        ])

    if overlap.get("available") and overlap.get("decline_pixels", 0) > 0:
        built_pct = float(overlap.get("builtup_overlap_pct") or 0.0)
        bare_pct = float(overlap.get("bare_overlap_pct") or 0.0)
        def overlap_score(pct: float) -> float:
            return 100.0 if pct >= 50 else 80.0 if pct >= 30 else 60.0 if pct >= 15 else 35.0 if pct >= 5 else 10.0 if pct > 0 else 0.0
        scores.append({"Hypothesis": "Vegetation replaced by built-up land", "Evidence score (0–100)": round(overlap_score(built_pct), 1),
                       "Evidence status": "Spatial overlap measured", "Basis": f"{overlap.get('decline_to_builtup_ha', 0):.2f} ha of vegetation-decline pixels overlap Module 3 built-up-transition proxy ({built_pct:.1f}% of decline pixels)."})
        scores.append({"Hypothesis": "Vegetation replaced by bare / sparse cover", "Evidence score (0–100)": round(overlap_score(bare_pct), 1),
                       "Evidence status": "Spatial overlap measured", "Basis": f"{overlap.get('decline_to_bare_ha', 0):.2f} ha of vegetation-decline pixels overlap Module 3 bare-soil transition proxy ({bare_pct:.1f}% of decline pixels)."})
    else:
        scores.extend([
            {"Hypothesis": "Vegetation replaced by built-up land", "Evidence score (0–100)": np.nan, "Evidence status": "Insufficient spatial evidence", "Basis": "Run Modules 1 and 3 with the same AOI and years to measure raster overlap."},
            {"Hypothesis": "Vegetation replaced by bare / sparse cover", "Evidence score (0–100)": np.nan, "Evidence status": "Insufficient spatial evidence", "Basis": "Run Modules 1 and 3 with the same AOI and years to measure raster overlap."},
        ])

    table = pd.DataFrame(scores)
    if not table.empty:
        table = table.sort_values("Evidence score (0–100)", ascending=False, na_position="last").reset_index(drop=True)
        table.insert(0, "Rank", np.arange(1, len(table) + 1))

    return {
        "target_year": year,
        "vegetation": {
            "baseline_year": historical.get("baseline_year"),
            "comparison_year": historical.get("comparison_year"),
            "decline_area_ha": historical.get("decline_area_ha"),
            "gain_area_ha": historical.get("gain_area_ha"),
            "mean_delta": historical.get("mean_delta"),
        },
        "climate": climate_ev,
        "landuse_overlap": overlap,
        "hypotheses": table,
        "limitations": [
            "Evidence scores rank support from the available indicators; they are not probabilities and do not prove causation.",
            "Climate data are from a regional reanalysis grid near the AOI centroid, so local field conditions may differ.",
            "Land-use classes are spectral proxies; built-up surfaces can be confused with bare soil.",
            "Raster overlap is only as reliable as the input masks, raster alignment, seasonal matching, and classifications.",
            "Unmodelled causes, crop cycles, fire, grazing, pests, irrigation changes, and sensor artefacts may explain some vegetation changes.",
        ],
        "generated_from": {
            "historical_available": bool(historical),
            "climate_available": bool(climate_ev["available"]),
            "landuse_available": bool(landuse),
            "spatial_overlap_available": bool(overlap.get("available")),
        },
    }
