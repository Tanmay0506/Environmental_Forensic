"""Terrain analysis utilities for the Environmental Forensics application.

Uses the Open-Meteo Elevation API to obtain elevation samples.
Results are preliminary and should not be treated as a high-resolution DEM.
"""

from __future__ import annotations

from math import cos, radians
from typing import Any

import numpy as np
import requests
from shapely.geometry import shape


ELEVATION_API_URL = "https://api.open-meteo.com/v1/elevation"
REQUEST_TIMEOUT_SECONDS = 30


def _create_grid(
    geometry: dict[str, Any],
    grid_size: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a regular latitude/longitude sample grid inside an AOI."""

    if not isinstance(geometry, dict):
        raise ValueError("The selected area must be a GeoJSON geometry.")

    polygon = shape(geometry)

    if polygon.is_empty or not polygon.is_valid:
        raise ValueError("The selected area is empty or invalid.")

    if polygon.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("Select a polygon or rectangle on the map.")

    min_lon, min_lat, max_lon, max_lat = polygon.bounds

    if max_lon - min_lon > 2 or max_lat - min_lat > 2:
        raise ValueError(
            "The selected area is too large for this preliminary terrain "
            "analysis. Select a smaller area."
        )

    if grid_size < 3 or grid_size > 30:
        raise ValueError("grid_size must be between 3 and 30.")

    latitudes = np.linspace(min_lat, max_lat, grid_size)
    longitudes = np.linspace(min_lon, max_lon, grid_size)

    sample_points = [
        (float(lon), float(lat))
        for lat in latitudes
        for lon in longitudes
        if polygon.covers(shape_point(lon, lat))
    ]

    if len(sample_points) < 9:
        raise ValueError(
            "The selected area contains too few sample points. "
            "Select a larger polygon."
        )

    return (
        np.array([point[1] for point in sample_points]),
        np.array([point[0] for point in sample_points]),
    )


def shape_point(longitude: float, latitude: float):
    """Create a Shapely point in longitude, latitude order."""
    from shapely.geometry import Point

    return Point(longitude, latitude)


def analyze_terrain(
    geometry: dict[str, Any],
    grid_size: int = 15,
) -> dict[str, Any]:
    """Sample elevation and calculate preliminary terrain statistics."""

    latitudes, longitudes = _create_grid(geometry, grid_size=grid_size)

    response = requests.get(
        ELEVATION_API_URL,
        params={
            "latitude": ",".join(f"{value:.6f}" for value in latitudes),
            "longitude": ",".join(f"{value:.6f}" for value in longitudes),
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    payload = response.json()
    elevations = np.asarray(payload.get("elevation", []), dtype=float)

    if elevations.size != latitudes.size:
        raise ValueError(
            "The elevation service returned an unexpected number of samples."
        )

    valid = np.isfinite(elevations)

    if valid.sum() < 9:
        raise ValueError("Not enough valid elevation samples were returned.")

    latitudes = latitudes[valid]
    longitudes = longitudes[valid]
    elevations = elevations[valid]

    # Estimate local terrain derivatives from neighbouring sample points.
    # This is a coarse approximation, not a substitute for a proper DEM.
    unique_lats = np.unique(latitudes)
    unique_lons = np.unique(longitudes)

    slope_values: list[float] = []
    aspect_values: list[float] = []

    elevation_grid = np.full(
        (len(unique_lats), len(unique_lons)),
        np.nan,
        dtype=float,
    )

    lat_lookup = {value: i for i, value in enumerate(unique_lats)}
    lon_lookup = {value: i for i, value in enumerate(unique_lons)}

    for lat, lon, elev in zip(latitudes, longitudes, elevations):
        elevation_grid[lat_lookup[lat], lon_lookup[lon]] = elev

    for row in range(1, len(unique_lats) - 1):
        for col in range(1, len(unique_lons) - 1):
            window = elevation_grid[
                row - 1 : row + 2,
                col - 1 : col + 2,
            ]

            if not np.isfinite(window).all():
                continue

            latitude = float(unique_lats[row])

            north_south_distance = (
                abs(unique_lats[row + 1] - unique_lats[row - 1])
                * 111_320
            )
            east_west_distance = (
                abs(unique_lons[col + 1] - unique_lons[col - 1])
                * 111_320
                * max(cos(radians(latitude)), 0.01)
            )

            dz_dx = (
                window[1, 2] - window[1, 0]
            ) / max(east_west_distance, 1.0)

            dz_dy = (
                window[0, 1] - window[2, 1]
            ) / max(north_south_distance, 1.0)

            slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))

            # Aspect is the downslope compass direction, clockwise from north.
            aspect = (
                np.degrees(np.arctan2(-dz_dx, -dz_dy)) + 360
            ) % 360

            slope_values.append(float(slope))
            aspect_values.append(float(aspect))

    return {
        "source": "Open-Meteo Elevation API",
        "method": "Regular geographic sampling",
        "sample_count": int(elevations.size),
        "elevation_min_m": float(np.min(elevations)),
        "elevation_max_m": float(np.max(elevations)),
        "elevation_mean_m": float(np.mean(elevations)),
        "elevation_range_m": float(np.max(elevations) - np.min(elevations)),
        "slope_mean_degrees": (
            float(np.mean(slope_values)) if slope_values else None
        ),
        "slope_max_degrees": (
            float(np.max(slope_values)) if slope_values else None
        ),
        "aspect_mean_degrees": (
            float(np.mean(aspect_values)) if aspect_values else None
        ),
        "latitudes": latitudes.tolist(),
        "longitudes": longitudes.tolist(),
        "elevations_m": elevations.tolist(),
        "slope_values_degrees": slope_values,
        "aspect_values_degrees": aspect_values,
        "warning": (
            "Preliminary, coarsely sampled terrain estimates. "
            "Use a validated DEM for detailed slope, aspect, erosion, "
            "or hydrological analysis."
        ),
    }
