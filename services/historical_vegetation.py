"""Historical Sentinel-2 NDVI change detection for a confirmed AOI.

This first implementation selects the lowest-cloud catalogue scene in the same
calendar window for each comparison year, retrieves the imagery through the
project's existing Copernicus service, calculates cloud-masked NDVI, aligns the
rasters, and reports indicative vegetation greenness decline/gain.
"""
from __future__ import annotations

import calendar
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from pyproj import CRS as PyprojCRS

from services.copernicus import search_sentinel2, download_sentinel2_bands
from analysis.vegetation import compute_ndvi


def _scene_properties(scene: dict[str, Any]) -> tuple[str, float]:
    props = scene.get("properties") or {}
    acquisition = props.get("datetime") or props.get("start_datetime")
    if not acquisition:
        raise ValueError("A catalogue scene has no acquisition date.")
    scene_date = str(acquisition)[:10]
    cloud = props.get("eo:cloud_cover", props.get("cloudCover", 100))
    try:
        cloud = float(cloud) if cloud is not None else 100.0
    except (TypeError, ValueError):
        cloud = 100.0
    return scene_date, cloud


def _find_best_scene(aoi_geojson: dict, year: int, start_month: int,
                     end_month: int, max_cloud_cover: int) -> tuple[dict, dict]:
    if start_month > end_month:
        raise ValueError("Start month must be on or before end month.")
    start_date = date(year, start_month, 1)
    end_date = date(year, end_month, calendar.monthrange(year, end_month)[1])
    scenes = search_sentinel2(
        geometry=aoi_geojson,
        start_date=start_date,
        end_date=end_date,
        max_cloud_cover=max_cloud_cover,
        limit=100,
    )
    if isinstance(scenes, dict):
        scenes = scenes.get("features", [])
    scenes = list(scenes or [])
    if not scenes:
        raise RuntimeError(
            f"No Sentinel-2 scenes were found for {year}-{start_month:02d}-01 "
            f"to {year}-{end_month:02d}-{end_date.day:02d} with scene cloud "
            f"cover <= {max_cloud_cover}%. Try a wider month window or a higher limit."
        )
    ranked = []
    for scene in scenes:
        try:
            scene_date, cloud = _scene_properties(scene)
            ranked.append((cloud, scene_date, scene))
        except ValueError:
            continue
    if not ranked:
        raise RuntimeError(f"Catalogue results for {year} had no usable acquisition dates.")
    cloud, scene_date, scene = min(ranked, key=lambda item: item[0])
    return scene, {"year": year, "date": scene_date, "cloud_cover": cloud,
                   "scene_id": scene.get("id", "Unknown scene"),
                   "candidate_count": len(scenes)}


def _read_float_raster(path: str | Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        data = src.read(1).astype("float32")
        profile = src.profile.copy()
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        data[~np.isfinite(data)] = np.nan
        return data, profile


def _align_to_reference(source_path: str | Path, ref_profile: dict) -> np.ndarray:
    with rasterio.open(source_path) as src:
        source = src.read(1).astype("float32")
        src_nodata = src.nodata
        if src_nodata is not None:
            source[source == src_nodata] = np.nan
        source[~np.isfinite(source)] = np.nan
        destination = np.full(
            (ref_profile["height"], ref_profile["width"]), np.nan, dtype="float32"
        )
        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        return destination


def _pixel_area_m2(profile: dict) -> float:
    """Return pixel area in square metres for a projected raster.

    Rasterio's CRS wrapper does not expose ``axis_info`` in all supported
    versions. Convert it to a pyproj CRS before reading axis unit metadata.
    """
    crs = profile.get("crs")
    transform = profile["transform"]
    area_units = abs(transform.a * transform.e - transform.b * transform.d)

    if crs is None:
        raise ValueError("The NDVI raster has no CRS; cannot calculate pixel area reliably.")

    try:
        projected_crs = PyprojCRS.from_user_input(crs)
    except Exception as exc:
        raise ValueError(f"Could not interpret raster CRS {crs!s}: {exc}") from exc

    if projected_crs.is_projected:
        axes = projected_crs.axis_info
        factor = float(axes[0].unit_conversion_factor) if axes else 1.0
        return float(area_units * factor * factor)

    raise ValueError(
        "The NDVI raster uses a geographic CRS, whose pixel dimensions are in degrees. "
        "Area calculations require a projected raster. Reproject the NDVI raster to a suitable "
        "metric CRS (for example, the appropriate UTM zone for the study area) before running "
        "change detection."
    )


def run_historical_change_detection(
    *,
    aoi_geojson: dict,
    baseline_year: int,
    comparison_year: int,
    start_month: int,
    end_month: int,
    max_cloud_cover: int,
    decline_threshold: float,
    vegetation_threshold: float,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Retrieve two seasonal observations and calculate NDVI change statistics."""
    if comparison_year <= baseline_year:
        raise ValueError("Comparison year must be later than baseline year.")
    if comparison_year > date.today().year:
        raise ValueError("Comparison year cannot be in the future.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    ndvi_paths = []
    for year in (baseline_year, comparison_year):
        _, metadata = _find_best_scene(
            aoi_geojson, year, start_month, end_month, max_cloud_cover
        )
        records.append(metadata)
        downloaded = Path(download_sentinel2_bands(
            aoi_geojson=aoi_geojson,
            acquisition_date=metadata["date"],
            max_cloud_cover=max_cloud_cover,
        ))
        if not downloaded.exists():
            raise FileNotFoundError(f"Copernicus retrieval did not create {downloaded}")

        # Preserve each year before the next retrieval can overwrite the service output.
        raw_path = out_dir / f"historical_sentinel2_{year}_{metadata['date']}.tif"
        if downloaded.resolve() != raw_path.resolve():
            shutil.copy2(downloaded, raw_path)
        else:
            raw_path = downloaded

        ndvi_path = out_dir / f"historical_ndvi_{year}_{metadata['date']}.tif"
        compute_result = compute_ndvi(input_path=str(raw_path), output_path=str(ndvi_path))
        returned_path = compute_result.get("output_path", str(ndvi_path)) if isinstance(compute_result, dict) else str(ndvi_path)
        ndvi_path = Path(returned_path)
        if not ndvi_path.exists():
            raise FileNotFoundError(f"NDVI calculation did not create {ndvi_path}")
        ndvi_paths.append(ndvi_path)

    baseline, profile = _read_float_raster(ndvi_paths[0])
    comparison = _align_to_reference(ndvi_paths[1], profile)
    valid = np.isfinite(baseline) & np.isfinite(comparison)
    if not np.any(valid):
        raise RuntimeError(
            "The two NDVI rasters have no overlapping valid pixels. This can happen when "
            "the selected scenes cover different areas or cloud masking removed the overlap."
        )

    delta = np.full(baseline.shape, np.nan, dtype="float32")
    delta[valid] = comparison[valid] - baseline[valid]
    baseline_vegetation = valid & (baseline >= vegetation_threshold)
    decline = baseline_vegetation & (delta <= -decline_threshold)
    gain = valid & (comparison >= vegetation_threshold) & (delta >= decline_threshold)
    stable = valid & ~decline & ~gain

    pixel_area = _pixel_area_m2(profile)
    valid_area_ha = float(np.count_nonzero(valid) * pixel_area / 10000)
    decline_area_ha = float(np.count_nonzero(decline) * pixel_area / 10000)
    gain_area_ha = float(np.count_nonzero(gain) * pixel_area / 10000)
    stable_area_ha = float(np.count_nonzero(stable) * pixel_area / 10000)

    class_map = np.zeros(baseline.shape, dtype="uint8")
    class_map[stable] = 1
    class_map[decline] = 2
    class_map[gain] = 3
    class_profile = profile.copy()
    class_profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
    class_path = out_dir / f"vegetation_change_{baseline_year}_{comparison_year}.tif"
    with rasterio.open(class_path, "w", **class_profile) as dst:
        dst.write(class_map, 1)
        dst.set_band_description(1, "1=stable/minor change, 2=decline, 3=gain; 0=invalid")

    delta_profile = profile.copy()
    delta_profile.update(count=1, dtype="float32", nodata=-9999.0, compress="deflate")
    delta_path = out_dir / f"ndvi_delta_{baseline_year}_{comparison_year}.tif"
    delta_to_write = np.where(np.isfinite(delta), delta, -9999.0).astype("float32")
    with rasterio.open(delta_path, "w", **delta_profile) as dst:
        dst.write(delta_to_write, 1)
        dst.set_band_description(1, "NDVI change (comparison minus baseline)")

    csv_path = out_dir / f"vegetation_change_summary_{baseline_year}_{comparison_year}.csv"
    summary_rows = [
        {"metric": "Baseline year", "value": baseline_year, "unit": "year"},
        {"metric": "Comparison year", "value": comparison_year, "unit": "year"},
        {"metric": "Baseline acquisition date", "value": records[0]["date"], "unit": "date"},
        {"metric": "Comparison acquisition date", "value": records[1]["date"], "unit": "date"},
        {"metric": "Baseline scene cloud cover", "value": records[0]["cloud_cover"], "unit": "%"},
        {"metric": "Comparison scene cloud cover", "value": records[1]["cloud_cover"], "unit": "%"},
        {"metric": "Mean baseline NDVI", "value": float(np.nanmean(np.where(valid, baseline, np.nan))), "unit": "NDVI"},
        {"metric": "Mean comparison NDVI", "value": float(np.nanmean(np.where(valid, comparison, np.nan))), "unit": "NDVI"},
        {"metric": "Mean NDVI change", "value": float(np.nanmean(delta)), "unit": "NDVI"},
        {"metric": "Vegetation decline area", "value": decline_area_ha, "unit": "ha"},
        {"metric": "Vegetation gain area", "value": gain_area_ha, "unit": "ha"},
        {"metric": "Stable/minor-change area", "value": stable_area_ha, "unit": "ha"},
        {"metric": "Valid comparison area", "value": valid_area_ha, "unit": "ha"},
        {"metric": "Valid pixels", "value": int(np.count_nonzero(valid)), "unit": "pixels"},
        {"metric": "Decline threshold", "value": decline_threshold, "unit": "NDVI change"},
        {"metric": "Vegetation threshold", "value": vegetation_threshold, "unit": "NDVI"},
    ]
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)

    return {
        "baseline_year": baseline_year,
        "comparison_year": comparison_year,
        "records": records,
        "baseline_ndvi": str(ndvi_paths[0]),
        "comparison_ndvi": str(ndvi_paths[1]),
        "class_path": str(class_path),
        "delta_path": str(delta_path),
        "csv_path": str(csv_path),
        "baseline": baseline,
        "comparison": comparison,
        "delta": delta,
        "valid": valid,
        "decline_area_ha": decline_area_ha,
        "gain_area_ha": gain_area_ha,
        "stable_area_ha": stable_area_ha,
        "valid_area_ha": valid_area_ha,
        "mean_baseline_ndvi": float(np.nanmean(np.where(valid, baseline, np.nan))),
        "mean_comparison_ndvi": float(np.nanmean(np.where(valid, comparison, np.nan))),
        "mean_delta": float(np.nanmean(delta)),
        "decline_pct": 100 * decline_area_ha / valid_area_ha if valid_area_ha else 0,
        "gain_pct": 100 * gain_area_ha / valid_area_ha if valid_area_ha else 0,
        "class_map": class_map,
        "profile": profile,
    }
