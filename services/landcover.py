"""Land-cover and mapped-feature screening for a selected area of interest.

Land-cover classes come from ESA WorldCover 2021 (10 m nominal resolution).
Roads, settlement places, rivers and named water features come from OpenStreetMap.
This module intentionally reports the dataset year and distinguishes mapped features
from satellite-derived land-cover classes.
"""
from __future__ import annotations

import math
import time
from typing import Any

import requests
import rasterio
from rasterio.mask import mask
from shapely.geometry import LineString, Point, Polygon, shape, box, mapping
from shapely.ops import transform as transform_geometry
from pyproj import Transformer

WORLDCOVER_BASE = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
    "v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
WORLD_COVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetlands",
    95: "Mangroves",
    100: "Moss and lichen",
}


def _tile_name(lat0: int, lon0: int) -> str:
    lat_prefix = "N" if lat0 >= 0 else "S"
    lon_prefix = "E" if lon0 >= 0 else "W"
    return f"{lat_prefix}{abs(lat0):02d}{lon_prefix}{abs(lon0):03d}"


def _tiles_for_bounds(bounds: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    minx, miny, maxx, maxy = bounds
    if maxx - minx > 2.5 or maxy - miny > 2.5:
        raise ValueError(
            "The selected AOI spans more than 2.5 degrees. Please select a smaller "
            "area for this land-cover screening."
        )
    tiles = []
    lat0 = math.floor(miny / 3) * 3
    lon0 = math.floor(minx / 3) * 3
    lat_end = math.floor(maxy / 3) * 3
    lon_end = math.floor(maxx / 3) * 3
    for lat in range(lat0, lat_end + 1, 3):
        for lon in range(lon0, lon_end + 1, 3):
            tiles.append((lat, lon))
    return tiles


def _worldcover_stats(aoi) -> dict[str, Any]:
    """Read only the AOI windows from public ESA WorldCover COG tiles."""
    from collections import Counter

    class_pixels: Counter[int] = Counter()
    total_valid_pixels = 0
    tile_sources: list[str] = []
    warnings: list[str] = []
    for lat0, lon0 in _tiles_for_bounds(aoi.bounds):
        tile = _tile_name(lat0, lon0)
        url = WORLDCOVER_BASE.format(tile=tile)
        tile_bbox = box(lon0, lat0, lon0 + 3, lat0 + 3)
        if not aoi.intersects(tile_bbox):
            continue
        try:
            # GDAL's /vsicurl reads the requested COG window without downloading the
            # complete global dataset. GDAL_HTTP_MULTIRANGE is handled by GDAL itself.
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
                with rasterio.open("/vsicurl/" + url) as src:
                    geom = aoi.intersection(tile_bbox)
                    if geom.is_empty:
                        continue
                    arr, _ = mask(src, [mapping(geom)], crop=True, filled=False)
                    values = arr[0].compressed()
                    if values.size:
                        vals, counts = __import__("numpy").unique(values, return_counts=True)
                        for val, count in zip(vals, counts):
                            if int(val) != 0:
                                class_pixels[int(val)] += int(count)
                                total_valid_pixels += int(count)
                    tile_sources.append(tile)
        except Exception as exc:
            warnings.append(f"ESA WorldCover tile {tile} could not be read: {exc}")

    if total_valid_pixels == 0:
        detail = " ".join(warnings[:2])
        raise RuntimeError(
            "No ESA WorldCover pixels could be retrieved for this AOI. Check internet "
            "access and try again. " + detail
        )

    # WorldCover is nominally 10 m. Use the documented nominal pixel area for a
    # transparent area estimate; exact pixel area varies slightly with latitude.
    areas_km2 = {code: count * 100.0 / 1_000_000 for code, count in class_pixels.items()}
    valid_area = sum(areas_km2.values())
    return {
        "class_pixels": dict(class_pixels),
        "class_areas_km2": areas_km2,
        "valid_pixels": total_valid_pixels,
        "valid_area_km2": valid_area,
        "tiles": tile_sources,
        "warnings": warnings,
        "source": "ESA WorldCover 2021 v200, nominal 10 m land-cover map",
        "attribution": "© ESA WorldCover project 2021; contains modified Copernicus Sentinel data processed by the ESA WorldCover consortium.",
    }


def _overpass_features(aoi) -> dict[str, Any]:
    minx, miny, maxx, maxy = aoi.bounds
    bbox_text = f"{miny},{minx},{maxy},{maxx}"
    query = f"""[out:json][timeout:45];
(
  way[highway]({bbox_text});
  way[natural=water]({bbox_text});
  way[water]({bbox_text});
  way[waterway~"^(river|stream)$"]({bbox_text});
  way[landuse=reservoir]({bbox_text});
  way[landuse=basin]({bbox_text});
  way[landuse=residential]({bbox_text});
  way[place]({bbox_text});
  node[place]({bbox_text});
);
out body geom;"""
    last_error = None
    response_data = None
    for endpoint in OVERPASS_URLS:
        try:
            response = requests.post(
                endpoint,
                data={"data": query},
                timeout=(10, 60),
                headers={"User-Agent": "EnvironmentalForensicsClimateIntelligence/1.0"},
            )
            response.raise_for_status()
            response_data = response.json()
            break
        except Exception as exc:
            last_error = exc
            time.sleep(0.5)
    if response_data is None:
        raise RuntimeError(f"OpenStreetMap Overpass request failed: {last_error}")

    roads_m: dict[str, float] = {}
    road_count: dict[str, int] = {}
    settlement_names: set[str] = set()
    settlement_count = 0
    river_length_m = 0.0
    water_area_m2 = 0.0
    reservoir_area_m2 = 0.0
    water_feature_count = 0
    to_metric = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform

    def clipped_metric_length(geom) -> float:
        clipped = geom.intersection(aoi)
        if clipped.is_empty:
            return 0.0
        return transform_geometry(to_metric, clipped).length

    def clipped_metric_area(geom) -> float:
        clipped = geom.intersection(aoi)
        if clipped.is_empty:
            return 0.0
        return transform_geometry(to_metric, clipped).area

    for element in response_data.get("elements", []):
        tags = element.get("tags", {})
        geom = None
        if element.get("type") == "node":
            geom = Point(element["lon"], element["lat"])
        elif element.get("geometry"):
            coords = [(p["lon"], p["lat"]) for p in element["geometry"] if "lon" in p and "lat" in p]
            if len(coords) >= 2:
                try:
                    geom = Polygon(coords) if coords[0] == coords[-1] and len(coords) >= 4 else LineString(coords)
                except Exception:
                    geom = None
        if geom is None or not geom.intersects(aoi):
            continue

        if tags.get("highway"):
            highway = tags["highway"]
            roads_m[highway] = roads_m.get(highway, 0.0) + clipped_metric_length(geom)
            road_count[highway] = road_count.get(highway, 0) + 1

        place = tags.get("place")
        if place:
            settlement_count += 1
            if tags.get("name"):
                settlement_names.add(tags["name"])

        is_water = (tags.get("natural") == "water" or "water" in tags or tags.get("landuse") in {"reservoir", "basin"} or tags.get("waterway") in {"river", "stream"})
        if is_water:
            water_feature_count += 1
            if isinstance(geom, Polygon):
                area = clipped_metric_area(geom)
                water_area_m2 += area
                if tags.get("landuse") == "reservoir" or tags.get("water") == "reservoir":
                    reservoir_area_m2 += area
            elif isinstance(geom, LineString):
                if tags.get("waterway") == "river" or tags.get("waterway") == "stream":
                    river_length_m += clipped_metric_length(geom)

    residential_area_km2 = 0.0
    # Residential ways are counted separately as a mapped settlement footprint.
    for element in response_data.get("elements", []):
        tags = element.get("tags", {})
        if tags.get("landuse") != "residential" or not element.get("geometry"):
            continue
        coords = [(p["lon"], p["lat"]) for p in element["geometry"] if "lon" in p and "lat" in p]
        if len(coords) >= 4 and coords[0] == coords[-1]:
            try:
                residential_area_km2 += clipped_metric_area(Polygon(coords)) / 1_000_000
            except Exception:
                pass

    return {
        "road_length_km_by_type": {k: v / 1000 for k, v in roads_m.items()},
        "road_count_by_type": road_count,
        "road_length_km_total": sum(roads_m.values()) / 1000,
        "settlement_feature_count": settlement_count,
        "settlement_names": sorted(settlement_names),
        "residential_footprint_km2": residential_area_km2,
        "river_stream_length_km": river_length_m / 1000,
        "mapped_water_area_km2": water_area_m2 / 1_000_000,
        "reservoir_area_km2": reservoir_area_m2 / 1_000_000,
        "water_feature_count": water_feature_count,
        "source": "OpenStreetMap contributors via Overpass API",
        "warning": "OSM completeness varies by location. Road lengths and mapped feature areas include only features mapped in OpenStreetMap; they are not guaranteed to represent all real-world features.",
    }


def analyze_landcover(geometry: dict[str, Any]) -> dict[str, Any]:
    """Analyze land-cover classes and OSM mapped infrastructure for GeoJSON AOI."""
    aoi = shape(geometry)
    if aoi.is_empty or not aoi.is_valid:
        raise ValueError("The AOI geometry is empty or invalid. Please redraw it.")
    if aoi.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("Please select a polygon area, not a point or line.")
    if aoi.area <= 0:
        raise ValueError("The AOI has no measurable area.")

    landcover = _worldcover_stats(aoi)
    try:
        mapped = _overpass_features(aoi)
    except Exception as exc:
        # Keep the satellite land-cover result useful even if the public OSM service
        # is temporarily unavailable or rate-limited.
        mapped = {
            "road_length_km_by_type": {},
            "road_count_by_type": {},
            "road_length_km_total": 0.0,
            "settlement_feature_count": 0,
            "settlement_names": [],
            "residential_footprint_km2": 0.0,
            "river_stream_length_km": 0.0,
            "mapped_water_area_km2": 0.0,
            "reservoir_area_km2": 0.0,
            "water_feature_count": 0,
            "source": "OpenStreetMap contributors via Overpass API (unavailable for this run)",
            "warning": f"OpenStreetMap feature query failed, so roads, settlements and water-feature metrics are unavailable in this run: {exc}",
        }
    class_areas = landcover["class_areas_km2"]
    crop = class_areas.get(40, 0.0)
    tree = class_areas.get(10, 0.0)
    built = class_areas.get(50, 0.0)
    grass = class_areas.get(30, 0.0)
    shrub = class_areas.get(20, 0.0)
    water = class_areas.get(80, 0.0) + class_areas.get(90, 0.0)
    return {
        "landcover": landcover,
        "mapped_features": mapped,
        "summary": {
            "cropland_km2": crop,
            "tree_cover_km2": tree,
            "built_up_km2": built,
            "grassland_km2": grass,
            "shrubland_km2": shrub,
            "worldcover_water_wetlands_km2": water,
            "other_landcover_km2": max(0.0, landcover["valid_area_km2"] - crop - tree - built - grass - shrub - water),
        },
        "caveats": [
            "ESA WorldCover 2021 is a land-cover snapshot, not a current 2026 map. It classifies annual cropland; orchards and plantations may appear under tree cover rather than cropland.",
            "Area values are estimates based on nominal 10 m pixels and may differ slightly from geodesic/projected measurements.",
            mapped["warning"],
        ],
    }
