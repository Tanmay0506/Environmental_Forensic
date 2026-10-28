
import math
import os
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import requests
import rasterio
from dotenv import load_dotenv
from pyproj import CRS, Transformer
from rasterio.io import MemoryFile
from rasterio.mask import mask
from shapely.geometry import mapping, shape
from shapely.ops import transform as transform_geometry


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

STAC_SEARCH_URL = "https://stac.dataspace.copernicus.eu/v1/search"

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/"
    "CDSE/protocol/openid-connect/token"
)

# Restore the endpoint used by your earlier working retrieval.
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

COLLECTION = "sentinel-2-l2a"

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_TIMEOUT = 180
MAX_PIXELS_PER_SIDE = 2500

BAND_NAMES = [
    "B04_Red",
    "B08_NIR",
    "B03_Green",
    "B11_SWIR",
    "SCL_Classification",
]


# ============================================================
# DATE HELPERS
# ============================================================

def normalize_date(value):
    """Accept date, datetime, or YYYY-MM-DD string."""

    if isinstance(value, datetime):
        return value.date().isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip()[:10]).isoformat()
        except ValueError as exc:
            raise ValueError(
                f"Invalid date {value!r}. Use YYYY-MM-DD."
            ) from exc

    raise TypeError(
        f"Unsupported date type: {type(value).__name__}"
    )


def date_range_utc(start_value, end_value):
    """Create a UTC time interval covering both calendar dates."""

    start_day = date.fromisoformat(normalize_date(start_value))
    end_day = date.fromisoformat(normalize_date(end_value))

    if start_day > end_day:
        raise ValueError("Start date must not be after end date.")

    start_dt = datetime.combine(
        start_day, time.min, tzinfo=timezone.utc
    )
    end_dt = datetime.combine(
        end_day + timedelta(days=1),
        time.min,
        tzinfo=timezone.utc,
    )

    return (
        start_dt.isoformat().replace("+00:00", "Z"),
        end_dt.isoformat().replace("+00:00", "Z"),
    )


# ============================================================
# AOI VALIDATION
# ============================================================

def validate_aoi(geometry):
    """Validate a GeoJSON Polygon or MultiPolygon."""

    if not isinstance(geometry, dict):
        raise TypeError("AOI must be a GeoJSON geometry dictionary.")

    if "type" not in geometry or "coordinates" not in geometry:
        raise ValueError(
            "AOI GeoJSON must contain type and coordinates."
        )

    try:
        geom = shape(geometry)
    except Exception as exc:
        raise ValueError(f"Invalid AOI GeoJSON: {exc}") from exc

    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError("AOI must be a Polygon or MultiPolygon.")

    if geom.is_empty:
        raise ValueError("AOI geometry is empty.")

    if not geom.is_valid:
        geom = geom.buffer(0)

    if geom.is_empty or not geom.is_valid:
        raise ValueError("AOI geometry is invalid.")

    minx, miny, maxx, maxy = geom.bounds

    if (
        minx < -180 or maxx > 180
        or miny < -90 or maxy > 90
    ):
        raise ValueError("AOI coordinates are outside geographic bounds.")

    if maxx - minx > 10 or maxy - miny > 10:
        raise ValueError(
            "AOI is too large for this regional processing workflow."
        )

    return geom


# ============================================================
# STEP 3: COPERNICUS CATALOGUE SEARCH
# ============================================================

def search_sentinel2(
    geometry,
    start_date,
    end_date,
    max_cloud_cover=20,
    limit=100,
):
    """Search the Sentinel-2 L2A STAC catalogue."""

    aoi = validate_aoi(geometry)
    start_utc, end_utc = date_range_utc(start_date, end_date)

    try:
        cloud_limit = float(max_cloud_cover)
    except (TypeError, ValueError) as exc:
        raise ValueError("Cloud cover must be numeric.") from exc

    if not 0 <= cloud_limit <= 100:
        raise ValueError("Cloud cover must be between 0 and 100.")

    if not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("Limit must be an integer from 1 to 100.")

    payload = {
        "collections": [COLLECTION],
        "intersects": mapping(aoi),
        "datetime": f"{start_utc}/{end_utc}",
        "limit": limit,
        "query": {
            "eo:cloud_cover": {"lte": cloud_limit}
        },
        "sortby": [
            {
                "field": "properties.datetime",
                "direction": "desc",
            }
        ],
    }

    try:
        response = requests.post(
            STAC_SEARCH_URL,
            json=payload,
            headers={"Accept": "application/geo+json"},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not connect to Copernicus catalogue: {exc}"
        ) from exc

    if not response.ok:
        raise RuntimeError(
            f"Catalogue HTTP {response.status_code}: "
            f"{response.text[:1500]}"
        )

    try:
        items = response.json().get("features", [])
    except ValueError as exc:
        raise RuntimeError("Catalogue returned invalid JSON.") from exc

    results = []

    for item in items:
        properties = item.get("properties") or {}
        cloud = properties.get("eo:cloud_cover")

        if cloud is None:
            results.append(item)
        else:
            try:
                if float(cloud) <= cloud_limit:
                    results.append(item)
            except (TypeError, ValueError):
                continue

    return results


# ============================================================
# STEP 4: OAUTH AUTHENTICATION
# ============================================================

def get_access_token():
    """Obtain a Copernicus Data Space access token."""

    client_id = os.getenv("COPERNICUS_CLIENT_ID")
    client_secret = os.getenv("COPERNICUS_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing COPERNICUS_CLIENT_ID or "
            "COPERNICUS_CLIENT_SECRET in the project .env file."
        )

    try:
        response = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not connect to Copernicus authentication: {exc}"
        ) from exc

    if not response.ok:
        raise RuntimeError(
            f"Authentication HTTP {response.status_code}: "
            f"{response.text[:1000]}"
        )

    try:
        return response.json()["access_token"]
    except (ValueError, KeyError) as exc:
        raise RuntimeError(
            "Authentication response did not contain an access token."
        ) from exc


# ============================================================
# PROJECTION
# ============================================================

def get_utm_crs(geometry):
    """Choose a UTM projection based on the AOI centroid."""

    centroid = geometry.centroid
    lon, lat = centroid.x, centroid.y

    if not -80 <= lat <= 84:
        raise ValueError(
            "This workflow supports latitudes from 80°S to 84°N."
        )

    zone = max(1, min(60, int(math.floor((lon + 180) / 6)) + 1))
    epsg = 32600 + zone if lat >= 0 else 32700 + zone

    return CRS.from_epsg(epsg)


# ============================================================
# FIVE-BAND EVALSCRIPT
# ============================================================

def build_evalscript():
    """
    Return B04, B08, B03, B11 and SCL in one five-band output.

    SCL is categorical data; do not use it as a reflectance band.
    """

    return """
    //VERSION=3

    function setup() {
      return {
        input: ["B04", "B08", "B03", "B11", "SCL"],
        output: {
          bands: 5,
          sampleType: "FLOAT32"
        }
      };
    }

    function evaluatePixel(sample) {
      return [
        sample.B04,
        sample.B08,
        sample.B03,
        sample.B11,
        sample.SCL
      ];
    }
    """


# ============================================================
# STEP 4: RETRIEVE FIVE-BAND RASTER
# ============================================================

def download_sentinel2_bands(
    aoi_geojson,
    acquisition_date,
    max_cloud_cover=20,
):
    """
    Retrieve a Sentinel-2 L2A raster and crop it to the AOI.

    Band order:
      1: B04 red
      2: B08 near-infrared
      3: B03 green
      4: B11 short-wave infrared
      5: SCL classification

    Returns the output GeoTIFF path as a string.
    """

    aoi = validate_aoi(aoi_geojson)
    day = normalize_date(acquisition_date)

    try:
        cloud_limit = float(max_cloud_cover)
    except (TypeError, ValueError) as exc:
        raise ValueError("Cloud cover must be numeric.") from exc

    if not 0 <= cloud_limit <= 100:
        raise ValueError("Cloud cover must be between 0 and 100.")

    utm_crs = get_utm_crs(aoi)

    transformer = Transformer.from_crs(
        "EPSG:4326",
        utm_crs,
        always_xy=True,
    )

    projected_aoi = transform_geometry(transformer.transform, aoi)

    minx, miny, maxx, maxy = projected_aoi.bounds
    extent_x = maxx - minx
    extent_y = maxy - miny

    if extent_x <= 0 or extent_y <= 0:
        raise ValueError("AOI has zero width or height.")

    resolution = max(
        10.0,
        extent_x / MAX_PIXELS_PER_SIDE,
        extent_y / MAX_PIXELS_PER_SIDE,
    )

    width = max(
        1,
        min(MAX_PIXELS_PER_SIDE, math.ceil(extent_x / resolution)),
    )
    height = max(
        1,
        min(MAX_PIXELS_PER_SIDE, math.ceil(extent_y / resolution)),
    )

    start_utc, end_utc = date_range_utc(day, day)
    token = get_access_token()

    payload = {
        "input": {
            "bounds": {
                "bbox": [minx, miny, maxx, maxy],
                "properties": {
                    "crs": (
                        "http://www.opengis.net/def/crs/EPSG/0/"
                        f"{utm_crs.to_epsg()}"
                    )
                },
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": start_utc,
                            "to": end_utc,
                        },
                        "maxCloudCoverage": cloud_limit,
                        "mosaickingOrder": "leastCC",
                    },
                }
            ],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [
                {
                    "identifier": "default",
                    "format": {"type": "image/tiff"},
                }
            ],
        },
        "evalscript": build_evalscript(),
    }

    try:
        response = requests.post(
            PROCESS_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "image/tiff",
            },
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Satellite Process API request failed: {exc}"
        ) from exc

    if not response.ok:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text[:1500]

        raise RuntimeError(
            f"Satellite Process API HTTP {response.status_code}: {detail}"
        )

    # Check the response before saving.
    try:
        with MemoryFile(response.content) as memory_file:
            with memory_file.open() as src:
                if src.count != 5:
                    raise RuntimeError(
                        f"Expected five bands but the API returned "
                        f"{src.count}. SCL was not returned as expected."
                    )

                image = src.read().astype("float32")
                profile = src.profile.copy()
                source_crs = src.crs
                source_transform = src.transform

    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "The API response could not be read as a GeoTIFF. "
            f"Content type: {response.headers.get('Content-Type', 'unknown')}. "
            f"Details: {exc}"
        ) from exc

    if source_crs is None:
        raise RuntimeError("Returned raster has no coordinate reference system.")

    profile.update(
        driver="GTiff",
        width=image.shape[2],
        height=image.shape[1],
        count=5,
        dtype="float32",
        crs=source_crs,
        transform=source_transform,
        nodata=-9999.0,
    )

    # Crop the rectangular response to the selected polygon.
    with MemoryFile() as memory_file:
        with memory_file.open(**profile) as src:
            src.write(image)

            cropped, cropped_transform = mask(
                src,
                [mapping(projected_aoi)],
                crop=True,
                filled=True,
                nodata=-9999.0,
            )

            output_profile = src.profile.copy()
            output_profile.update(
                driver="GTiff",
                width=cropped.shape[2],
                height=cropped.shape[1],
                count=5,
                dtype="float32",
                transform=cropped_transform,
                nodata=-9999.0,
                compress="deflate",
            )

    # New filename prevents reuse of the previous four-band raster.
    output_path = (
        RAW_DATA_DIR
        / f"sentinel2_{day}_utm{utm_crs.to_epsg()}_5band.tif"
    )

    with rasterio.open(output_path, "w", **output_profile) as dst:
        dst.write(cropped.astype("float32"))

        for index, name in enumerate(BAND_NAMES, start=1):
            dst.set_band_description(index, name)

    # Validate the saved file.
    with rasterio.open(output_path) as check:
        if check.count != 5:
            output_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Saved raster contains {check.count} bands, not five."
            )

    return str(output_path)
