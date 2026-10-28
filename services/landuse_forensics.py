
"""Indicative two-date land-cover transition screening from Sentinel-2 bands.

Expected band order: B04 (red), B08 (NIR), B03 (green), B11 (SWIR),
SCL (scene classification layer).

These threshold-based classes are proxies, not a validated land-cover product.
"""
from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from pyproj import CRS as PyprojCRS

from services.copernicus import download_sentinel2_bands
from services.historical_vegetation import _find_best_scene


CLASS_NAMES = {
    0: "Invalid / masked",
    1: "Vegetation",
    2: "Water proxy",
    3: "Built-up / impervious proxy",
    4: "Bare soil / sparse cover",
}


def _read_bands(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read the five expected raster bands."""
    with rasterio.open(path) as src:
        if src.count < 5:
            raise ValueError(
                "Expected at least 5 bands in order "
                "B04, B08, B03, B11, SCL; "
                f"got {src.count}."
            )

        bands = src.read([1, 2, 3, 4, 5]).astype("float32")
        profile = src.profile.copy()
        nodata = src.nodata

        if nodata is not None:
            reflectance = bands[:4]
            reflectance[reflectance == nodata] = np.nan
            bands[:4] = reflectance

        return bands, profile


def _align_bands(source_path: Path, ref_profile: dict) -> np.ndarray:
    """Align comparison raster bands to the baseline raster grid."""
    with rasterio.open(source_path) as src:
        if src.count < 5:
            raise ValueError(
                "Comparison raster has fewer than five expected bands."
            )

        height = ref_profile["height"]
        width = ref_profile["width"]

        # Raster arrays are 3D: (band, row, column).
        out = np.full(
            (5, height, width),
            np.nan,
            dtype="float32",
        )

        # Continuous reflectance bands: bilinear resampling.
        for i in range(1, 5):
            band = src.read(i).astype("float32")

            if src.nodata is not None:
                band[band == src.nodata] = np.nan

            reproject(
                source=band,
                destination=out[i - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=np.nan,
                dst_transform=ref_profile["transform"],
                dst_crs=ref_profile["crs"],
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )

        # SCL is categorical: use nearest-neighbour resampling.
        scl = src.read(5).astype("float32")
        scl_nodata = src.nodata

        if scl_nodata is not None:
            scl[scl == scl_nodata] = np.nan

        reproject(
            source=scl,
            destination=out[4],
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )

        return out


def _normalise_reflectance(arr: np.ndarray) -> np.ndarray:
    """Handle common reflectance scales: 0-1 or 0-10000."""
    out = arr.copy().astype("float32")
    finite = out[np.isfinite(out)]

    if finite.size and np.nanpercentile(finite, 95) > 2.0:
        out[:4] /= 10000.0

    return out


def _classify(
    bands: np.ndarray,
    vegetation_threshold: float,
    builtup_ndbi_threshold: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Classify pixels into indicative land-cover proxy classes."""
    b = _normalise_reflectance(bands)

    red, nir, green, swir, scl = b
    eps = 1e-6

    ndvi = (nir - red) / (nir + red + eps)
    ndbi = (swir - nir) / (swir + nir + eps)
    ndwi = (green - nir) / (green + nir + eps)

    valid = (
        np.isfinite(red)
        & np.isfinite(nir)
        & np.isfinite(green)
        & np.isfinite(swir)
        & np.isfinite(scl)
    )

    # SCL classes to mask:
    # 0 no data, 1 defective, 3 cloud shadow, 8/9 cloud,
    # 10 cirrus, 11 snow/ice.
    scl_integer = np.nan_to_num(scl, nan=0).astype("int16")
    bad_scl = np.isin(
        scl_integer,
        [0, 1, 3, 8, 9, 10, 11],
    )
    valid &= ~bad_scl

    classes = np.zeros(ndvi.shape, dtype="uint8")

    vegetation = valid & (ndvi >= vegetation_threshold)

    water = (
        valid
        & ~vegetation
        & (ndwi > 0.12)
        & (ndvi < 0.20)
    )

    builtup = (
        valid
        & ~vegetation
        & ~water
        & (ndbi >= builtup_ndbi_threshold)
        & (ndvi < 0.25)
    )

    bare = valid & ~vegetation & ~water & ~builtup

    classes[vegetation] = 1
    classes[water] = 2
    classes[builtup] = 3
    classes[bare] = 4

    return classes, {
        "valid": valid,
        "ndvi": ndvi,
        "ndbi": ndbi,
        "ndwi": ndwi,
    }


def _pixel_area_m2(profile: dict) -> float:
    """Calculate pixel area using the raster CRS unit conversion."""
    crs = profile.get("crs")

    if crs is None:
        raise ValueError(
            "Raster has no CRS; cannot calculate area reliably."
        )

    pycrs = PyprojCRS.from_user_input(crs)

    if not pycrs.is_projected:
        raise ValueError(
            "Raster uses a geographic CRS. Reproject to a suitable "
            "metric CRS before calculating pixel area."
        )

    # Use pyproj.CRS.axis_info, not rasterio.crs.CRS.axis_info.
    axes = pycrs.axis_info
    unit_factor = (
        float(axes[0].unit_conversion_factor)
        if axes
        else 1.0
    )

    transform = profile["transform"]
    pixel_area = abs(
        float(
            transform.a * transform.e
            - transform.b * transform.d
        )
    )

    return pixel_area * unit_factor * unit_factor


def run_landuse_transition_analysis(
    *,
    aoi_geojson: dict,
    baseline_year: int,
    comparison_year: int,
    start_month: int,
    end_month: int,
    max_cloud_cover: int,
    vegetation_threshold: float = 0.35,
    builtup_ndbi_threshold: float = 0.05,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run two-date land-cover proxy classification and transition analysis."""

    if baseline_year >= comparison_year:
        raise ValueError(
            "Comparison year must be later than baseline year."
        )

    if comparison_year > date.today().year:
        raise ValueError("Comparison year cannot be in the future.")

    if not 1 <= start_month <= 12 or not 1 <= end_month <= 12:
        raise ValueError("Months must be between 1 and 12.")

    if start_month > end_month:
        raise ValueError(
            "Season start month must be on or before end month."
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = []
    paths = []

    # Retrieve one low-cloud scene for each year.
    for year in (baseline_year, comparison_year):
        _, meta = _find_best_scene(
            aoi_geojson,
            year,
            start_month,
            end_month,
            max_cloud_cover,
        )

        downloaded = Path(
            download_sentinel2_bands(
                aoi_geojson=aoi_geojson,
                acquisition_date=meta["date"],
                max_cloud_cover=max_cloud_cover,
            )
        )

        if not downloaded.exists():
            raise FileNotFoundError(
                f"Satellite retrieval did not create {downloaded}"
            )

        dest = out / f"landuse_sentinel2_{year}_{meta['date']}.tif"

        if downloaded.resolve() != dest.resolve():
            shutil.copy2(downloaded, dest)

        paths.append(dest)
        records.append(meta)

    # Read baseline and align comparison to its grid.
    baseline_bands, profile = _read_bands(paths[0])
    comparison_bands = _align_bands(paths[1], profile)

    baseline_class, baseline_indices = _classify(
        baseline_bands,
        vegetation_threshold,
        builtup_ndbi_threshold,
    )

    comparison_class, comparison_indices = _classify(
        comparison_bands,
        vegetation_threshold,
        builtup_ndbi_threshold,
    )

    # Compare only pixels valid in both dates.
    valid = (
        baseline_indices["valid"]
        & comparison_indices["valid"]
        & (baseline_class > 0)
        & (comparison_class > 0)
    )

    if not np.any(valid):
        raise RuntimeError(
            "No overlapping valid pixels remain after cloud/shadow "
            "masking. Try another seasonal window or cloud limit."
        )

    area_m2 = _pixel_area_m2(profile)
    class_area = {}
    transition_rows = []
    total_pixels = int(valid.sum())

    for code, name in CLASS_NAMES.items():
        if code == 0:
            continue

        baseline_pixels = int(
            np.count_nonzero(valid & (baseline_class == code))
        )
        comparison_pixels = int(
            np.count_nonzero(valid & (comparison_class == code))
        )

        class_area[name] = {
            "baseline_ha": baseline_pixels * area_m2 / 10000,
            "comparison_ha": comparison_pixels * area_m2 / 10000,
            "change_ha": (
                comparison_pixels - baseline_pixels
            ) * area_m2 / 10000,
        }

    for from_code, from_name in CLASS_NAMES.items():
        if from_code == 0:
            continue

        for to_code, to_name in CLASS_NAMES.items():
            if to_code == 0:
                continue

            count = int(
                np.count_nonzero(
                    valid
                    & (baseline_class == from_code)
                    & (comparison_class == to_code)
                )
            )

            if count:
                transition_rows.append({
                    "from_class": from_name,
                    "to_class": to_name,
                    "pixel_count": count,
                    "area_ha": count * area_m2 / 10000,
                    "transition": (
                        "No class change"
                        if from_code == to_code
                        else f"{from_name} → {to_name}"
                    ),
                })

    transitions = pd.DataFrame(transition_rows)

    if not transitions.empty:
        transitions = transitions.sort_values(
            "area_ha",
            ascending=False,
        )

    # Change codes:
    # 0 invalid, 1 unchanged, 2 vegetation->built-up,
    # 3 vegetation->bare soil, 4 vegetation->water, 5 other transition.
    change = np.zeros(
        baseline_class.shape,
        dtype="uint8",
    )

    change[valid & (baseline_class == comparison_class)] = 1

    change[
        valid & (baseline_class == 1) & (comparison_class == 3)
    ] = 2

    change[
        valid & (baseline_class == 1) & (comparison_class == 4)
    ] = 3

    change[
        valid & (baseline_class == 1) & (comparison_class == 2)
    ] = 4

    other_transition = (
        valid
        & (baseline_class != comparison_class)
        & ~(
            (baseline_class == 1)
            & np.isin(comparison_class, [2, 3, 4])
        )
    )
    change[other_transition] = 5

    class_profile = profile.copy()
    class_profile.update(
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
    )

    class_path = out / (
        f"landuse_transition_{baseline_year}_{comparison_year}.tif"
    )

    with rasterio.open(class_path, "w", **class_profile) as dst:
        dst.write(change, 1)
        dst.set_band_description(
            1,
            "1=same proxy class, 2=vegetation-to-built-up proxy, "
            "3=vegetation-to-bare proxy, 4=vegetation-to-water proxy, "
            "5=other transition, 0=invalid",
        )

    baseline_path = out / f"landcover_proxy_{baseline_year}.tif"
    comparison_path = out / f"landcover_proxy_{comparison_year}.tif"

    for path, data in (
        (baseline_path, baseline_class),
        (comparison_path, comparison_class),
    ):
        raster_profile = profile.copy()
        raster_profile.update(
            count=1,
            dtype="uint8",
            nodata=0,
            compress="deflate",
        )

        with rasterio.open(path, "w", **raster_profile) as dst:
            dst.write(data, 1)
            dst.set_band_description(
                1,
                "1=vegetation, 2=water proxy, "
                "3=built-up/impervious proxy, "
                "4=bare/sparse cover, 0=invalid",
            )

    csv_path = out / (
        f"landuse_transition_summary_{baseline_year}_{comparison_year}.csv"
    )
    transitions.to_csv(csv_path, index=False)

    stats_path = out / (
        f"landcover_class_areas_{baseline_year}_{comparison_year}.csv"
    )

    pd.DataFrame([
        {"class": name, **values}
        for name, values in class_area.items()
    ]).to_csv(stats_path, index=False)

    return {
        "baseline_year": baseline_year,
        "comparison_year": comparison_year,
        "records": records,
        "baseline_class": baseline_class,
        "comparison_class": comparison_class,
        "change_map": change,
        "valid": valid,
        "class_area": class_area,
        "transitions": transitions,
        "valid_area_ha": total_pixels * area_m2 / 10000,
        "class_path": str(class_path),
        "baseline_path": str(baseline_path),
        "comparison_path": str(comparison_path),
        "csv_path": str(csv_path),
        "stats_path": str(stats_path),
        "source": "Sentinel-2 five-band raster via project Copernicus service",
        "limitations": (
            "Threshold-based proxy classes only. Built-up versus bare soil "
            "is uncertain. One low-cloud scene per year can be affected by "
            "seasonality, residual cloud, crop cycles, and scene timing. "
            "Validate with reference data before reporting official "
            "land-cover change."
        ),
    }
