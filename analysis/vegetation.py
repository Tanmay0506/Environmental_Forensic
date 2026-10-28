
from pathlib import Path

import numpy as np
import rasterio


# Sentinel-2 SCL classes excluded from vegetation analysis.
INVALID_SCL_CLASSES = {
    0,   # No data
    1,   # Saturated or defective
    3,   # Cloud shadow
    7,   # Unclassified
    8,   # Cloud, medium probability
    9,   # Cloud, high probability
    10,  # Thin cirrus
    11,  # Snow or ice
}


def compute_ndvi(
    input_path: str | Path,
    output_path: str | Path,
) -> dict:
    """
    Calculate cloud-masked NDVI.

    Required input band order:
      1 = B04 (red)
      2 = B08 (near-infrared)
      3 = B03 (green)
      4 = B11 (short-wave infrared)
      5 = SCL (Scene Classification Layer)

    Output invalid pixels are written as -9999.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Satellite raster not found: {input_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        if src.count < 5:
            raise ValueError(
                "Cloud masking requires five bands: B04, B08, "
                "B03, B11, and SCL. Retrieve the satellite data again "
                "using the updated services/copernicus.py."
            )

        red = src.read(1, masked=True).astype("float32")
        nir = src.read(2, masked=True).astype("float32")
        scl = src.read(5, masked=True)

        red_data = np.asarray(red.filled(np.nan))
        nir_data = np.asarray(nir.filled(np.nan))
        scl_data = np.asarray(scl.filled(-9999))

        denominator = red_data + nir_data

        valid = (
            np.isfinite(red_data)
            & np.isfinite(nir_data)
            & (scl_data >= 0)
            & (denominator != 0)
        )

        for class_value in INVALID_SCL_CLASSES:
            valid &= scl_data != class_value

        ndvi = np.full(
            red_data.shape,
            np.nan,
            dtype=np.float32,
        )

        ndvi[valid] = (
            (nir_data[valid] - red_data[valid])
            / denominator[valid]
        )

        ndvi[valid] = np.clip(ndvi[valid], -1.0, 1.0)

        valid_values = ndvi[np.isfinite(ndvi)]

        total_pixels = int(ndvi.size)
        valid_pixels = int(valid_values.size)
        masked_pixels = total_pixels - valid_pixels

        if valid_pixels == 0:
            raise ValueError(
                "No valid pixels remain after SCL masking. "
                "Try another acquisition date or a different AOI."
            )

        output_profile = src.profile.copy()
        output_profile.update(
            count=1,
            dtype="float32",
            nodata=-9999.0,
            compress="deflate",
        )

        output_array = np.where(
            np.isfinite(ndvi),
            ndvi,
            -9999.0,
        ).astype("float32")

        with rasterio.open(
            output_path, "w", **output_profile
        ) as dst:
            dst.write(output_array, 1)
            dst.set_band_description(1, "Cloud-masked NDVI")

    return {
        "output_path": str(output_path),
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "masked_pixels": masked_pixels,
        "masked_percent": (
            100.0 * masked_pixels / total_pixels
            if total_pixels
            else 0.0
        ),
        "minimum": float(valid_values.min()),
        "maximum": float(valid_values.max()),
        "mean": float(valid_values.mean()),
        "median": float(np.median(valid_values)),
    }


def load_ndvi_for_display(ndvi_path):
    """Load an NDVI raster for visualization."""

    with rasterio.open(ndvi_path) as src:
        ndvi = src.read(1, masked=True).astype("float32")
        bounds = src.bounds
        crs = src.crs

    values = np.asarray(ndvi.filled(np.nan))
    values[~np.isfinite(values)] = np.nan

    return values, bounds, crs
