"""Historical climate stress analysis using Open-Meteo's ERA5-Land archive.

This module retrieves daily reanalysis values for the centroid of a user-selected
area. These are regional/grid-cell estimates, not field-scale measurements.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
import requests
from shapely.geometry import shape

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _centroid_coordinates(aoi_geojson: dict[str, Any]) -> tuple[float, float]:
    geom = shape(aoi_geojson.get("geometry", aoi_geojson))
    if geom.is_empty or not geom.is_valid:
        geom = geom.buffer(0)
    if geom.is_empty:
        raise ValueError("The confirmed study area is empty or invalid.")
    centroid = geom.centroid
    return float(centroid.y), float(centroid.x)


def fetch_climate_series(
    *, aoi_geojson: dict[str, Any], start_date: date, end_date: date,
    timeout: int = 90,
) -> dict[str, Any]:
    """Fetch daily precipitation, temperatures and soil moisture for AOI centroid."""
    if start_date > end_date:
        raise ValueError("Start date must be on or before end date.")
    if end_date > date.today():
        raise ValueError("End date cannot be in the future.")

    # Open-Meteo ERA5/ERA5-Land archive data typically has about a 5-day delay.
    # Use a 7-day safety margin to avoid HTTP 400 errors for not-yet-published dates.
    requested_end_date = end_date
    available_end_date = date.today() - timedelta(days=7)
    effective_end_date = min(end_date, available_end_date)
    if start_date > effective_end_date:
        raise ValueError(
            "The selected period is too recent for the historical climate archive. "
            f"Choose an end date on or before {available_end_date.isoformat()}."
        )

    lat, lon = _centroid_coordinates(aoi_geojson)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": effective_end_date.isoformat(),
        "daily": ",".join([
            "precipitation_sum", "temperature_2m_mean", "temperature_2m_max",
            "soil_moisture_0_to_7cm_mean", "soil_moisture_7_to_28cm_mean",
            "soil_moisture_28_to_100cm_mean",
        ]),
        "timezone": "Asia/Kolkata",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "models": "era5_land",
    }
    try:
        response = requests.get(ARCHIVE_URL, params=params, timeout=timeout)
        if not response.ok:
            # Surface Open-Meteo's actual validation reason instead of only "400 Bad Request".
            try:
                error_payload = response.json()
                reason = error_payload.get("reason") or error_payload.get("message")
            except (ValueError, AttributeError):
                reason = response.text[:500]
            raise RuntimeError(
                f"Open-Meteo returned HTTP {response.status_code}: "
                f"{reason or response.reason}. Request end date: {effective_end_date.isoformat()}."
            )
        payload = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not retrieve historical climate data: {exc}") from exc
    if payload.get("error"):
        raise RuntimeError(payload.get("reason", "Climate API returned an error."))
    daily = payload.get("daily") or {}
    if not daily.get("time"):
        raise RuntimeError("The climate API returned no daily observations for this period.")

    frame = pd.DataFrame({
        "date": pd.to_datetime(daily["time"], errors="coerce"),
        "rainfall_mm": daily.get("precipitation_sum", []),
        "temperature_mean_c": daily.get("temperature_2m_mean", []),
        "temperature_max_c": daily.get("temperature_2m_max", []),
        "soil_moisture_0_7cm": daily.get("soil_moisture_0_to_7cm_mean", []),
        "soil_moisture_7_28cm": daily.get("soil_moisture_7_to_28cm_mean", []),
        "soil_moisture_28_100cm": daily.get("soil_moisture_28_to_100cm_mean", []),
    })
    if frame.empty:
        raise RuntimeError("No climate observations were returned.")
    for column in frame.columns:
        if column != "date":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    frame["month_start"] = frame["date"].dt.to_period("M").dt.to_timestamp()

    monthly = frame.groupby("month_start", as_index=False).agg(
        rainfall_mm=("rainfall_mm", "sum"),
        temperature_mean_c=("temperature_mean_c", "mean"),
        temperature_max_c=("temperature_max_c", "max"),
        soil_moisture_0_7cm=("soil_moisture_0_7cm", "mean"),
        soil_moisture_7_28cm=("soil_moisture_7_28cm", "mean"),
        soil_moisture_28_100cm=("soil_moisture_28_100cm", "mean"),
        days_observed=("date", "count"),
    )
    monthly["year"] = monthly["month_start"].dt.year
    monthly["month"] = monthly["month_start"].dt.month

    annual = frame.groupby("year", as_index=False).agg(
        rainfall_mm=("rainfall_mm", "sum"),
        temperature_mean_c=("temperature_mean_c", "mean"),
        temperature_max_c=("temperature_max_c", "max"),
        soil_moisture_0_7cm=("soil_moisture_0_7cm", "mean"),
        soil_moisture_7_28cm=("soil_moisture_7_28cm", "mean"),
        soil_moisture_28_100cm=("soil_moisture_28_100cm", "mean"),
        days_observed=("date", "count"),
    )
    return {
        "daily": frame,
        "monthly": monthly,
        "annual": annual,
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": effective_end_date.isoformat(),
        "requested_end_date": requested_end_date.isoformat(),
        "date_range_note": (
            f"Requested through {requested_end_date.isoformat()}; "
            f"historical archive data used through {effective_end_date.isoformat()} "
            "to account for the archive publication delay."
            if effective_end_date != requested_end_date
            else "Requested end date was within the historical archive availability window."
        ),
        "source": "Open-Meteo Historical Weather API, ERA5-Land reanalysis",
        "source_url": "https://open-meteo.com/en/docs/historical-weather-api",
        "resolution_note": "ERA5-Land native grid is approximately 9 km. Values represent a model grid cell near the AOI centroid, not field-level measurements.",
    }


def add_climate_anomalies(
    annual: pd.DataFrame, baseline_start_year: int, baseline_end_year: int
) -> pd.DataFrame:
    """Add rainfall and temperature anomalies relative to baseline annual means."""
    if baseline_start_year > baseline_end_year:
        raise ValueError("Baseline start year must not exceed baseline end year.")
    baseline = annual[
        annual["year"].between(baseline_start_year, baseline_end_year)
    ]
    if baseline.empty:
        raise ValueError("No observations fall inside the selected baseline period.")
    rainfall_ref = baseline["rainfall_mm"].mean()
    temp_ref = baseline["temperature_mean_c"].mean()
    soil_ref = baseline["soil_moisture_0_7cm"].mean()
    result = annual.copy()
    result["rainfall_anomaly_mm"] = result["rainfall_mm"] - rainfall_ref
    result["rainfall_anomaly_pct"] = (
        (result["rainfall_mm"] - rainfall_ref) / rainfall_ref * 100
        if pd.notna(rainfall_ref) and rainfall_ref != 0 else float("nan")
    )
    result["temperature_anomaly_c"] = result["temperature_mean_c"] - temp_ref
    result["topsoil_moisture_anomaly"] = result["soil_moisture_0_7cm"] - soil_ref
    return result
