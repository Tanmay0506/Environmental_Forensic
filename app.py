
from datetime import date, timedelta
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from folium.plugins import Draw
from geopy.geocoders import Nominatim
from pyproj import Geod
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from streamlit_folium import st_folium

from services.copernicus import (
    search_sentinel2,
    download_sentinel2_bands,
)
from services.landcover import analyze_landcover
from services.historical_vegetation import run_historical_change_detection
from services.climate_stress import fetch_climate_series, add_climate_anomalies
from services.landuse_forensics import run_landuse_transition_analysis
from services.cause_attribution import build_cause_attribution
from services.change_interpretation import build_change_interpretation
from services.forensic_report import build_forensic_pdf
from services.gemini_decision_support import generate_gemini_next_steps
from analysis.vegetation import (
    compute_ndvi,
    load_ndvi_for_display,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

NDVI_OUTPUT = PROCESSED_DIR / "ndvi_latest.tif"

st.set_page_config(
    page_title="AI-Powered Environmental Forensics & Climate Intelligence",
    page_icon="🌍",
    layout="wide",
)

st.markdown("""
<style>
/* Environmental Forensics — glassmorphism workspace */
:root { --ef-glass: rgba(255,255,255,.075); --ef-line: rgba(255,255,255,.16); }
.stApp {
  background:
    radial-gradient(ellipse at 12% 4%, rgba(31, 151, 137, .22), transparent 36%),
    radial-gradient(ellipse at 88% 12%, rgba(74, 109, 210, .20), transparent 34%),
    linear-gradient(135deg, #07131d 0%, #0b1824 48%, #101b2a 100%);
  background-attachment: fixed;
}
[data-testid="stHeader"] { background: rgba(5, 14, 22, .25); backdrop-filter: blur(18px); }
[data-testid="stToolbar"] { right: 1rem; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }
h1, h2, h3 { letter-spacing: -.035em; }
h1 { font-weight: 760; background: linear-gradient(100deg,#f4fbff 10%,#a9eee1 55%,#a8c5ff 100%); -webkit-background-clip:text; background-clip:text; color:transparent; }
h2, h3 { color: #e7f4ff; }
p, label, [data-testid="stCaptionContainer"] { line-height: 1.55; }
/* Translucent panels */
div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stExpander"],
div[data-testid="stForm"],
div[data-testid="stDataFrame"],
div[data-testid="stTable"],
div[data-testid="stMetric"],
div[data-testid="stAlert"] {
  border: 1px solid var(--ef-line) !important;
  border-radius: 18px !important;
  background: var(--ef-glass) !important;
  backdrop-filter: blur(18px) saturate(135%);
  -webkit-backdrop-filter: blur(18px) saturate(135%);
  box-shadow: 0 10px 34px rgba(0,0,0,.13), inset 0 1px 0 rgba(255,255,255,.055);
}
div[data-testid="stMetric"] { padding: 14px 16px; transition: transform .22s ease, border-color .22s ease, box-shadow .22s ease; }
div[data-testid="stMetric"]:hover { transform: translateY(-3px); border-color: rgba(127,220,211,.48) !important; box-shadow: 0 15px 35px rgba(0,0,0,.2); }
div[data-testid="stMetricValue"] { font-weight: 720; }
/* Inputs */
.stTextInput input, .stNumberInput input, .stDateInput input, .stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"], .stTextArea textarea {
  border-radius: 12px !important;
  background: rgba(8, 19, 30, .62) !important;
  border-color: rgba(177,208,232,.2) !important;
}
/* Buttons */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
  border-radius: 12px !important;
  border: 1px solid rgba(152,226,218,.32) !important;
  background: linear-gradient(120deg, rgba(28,132,124,.9), rgba(54,105,171,.9)) !important;
  color: #f6ffff !important; font-weight: 650 !important;
  box-shadow: 0 7px 20px rgba(0,0,0,.18);
  transition: transform .18s ease, filter .18s ease, box-shadow .18s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-2px); filter: brightness(1.12); box-shadow: 0 12px 26px rgba(0,0,0,.24);
}
.stButton > button:active, .stFormSubmitButton > button:active { transform: translateY(0) scale(.99); }
/* Stage navigation */
div[role="radiogroup"] { gap: .45rem; padding: .55rem; border-radius: 16px; background: rgba(255,255,255,.045); border: 1px solid rgba(255,255,255,.10); }
div[role="radiogroup"] label { padding: .5rem .75rem; border-radius: 11px; transition: background .2s ease; }
/* Tabs, expanders and tables */
.stTabs [data-baseweb="tab-list"] { gap: .4rem; }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; }
div[data-testid="stExpander"] { overflow: hidden; }
div[data-testid="stDataFrame"] { padding: .25rem; }
hr { border-color: rgba(200,225,245,.13) !important; }
/* Smooth reveal without hiding content */
@keyframes efFadeUp { from { opacity: 0; transform: translateY(7px); } to { opacity: 1; transform: translateY(0); } }
[data-testid="stVerticalBlock"] > div { animation: efFadeUp .32s ease both; }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; scroll-behavior: auto !important; } }
</style>
""", unsafe_allow_html=True)

st.title("🌍 Environmental Forensics & Climate Intelligence")
st.caption(
    "Analyze an area, compare satellite and climate indicators, and investigate possible environmental causes."
)

st.caption("Select an area, inspect vegetation change, assess climate and land-use signals, then review the evidence.")

# ============================================================
# 2. SESSION STATE
# ============================================================

DEFAULTS = {
    "map_center": [18.366, 74.028],
    "map_zoom": 11,
    "map_version": 0,
    "drawings": [],
    "confirmed_aoi": None,
    "confirmed_area_km2": None,
    "copernicus_scenes": [],
    "copernicus_search_done": False,
    "selected_scene_id": None,
    "latest_satellite_raster": None,
    "latest_ndvi_raster": None,
    "location_message": None,
    "search_message": None,
    "download_message": None,
    "ndvi_message": None,
    "landcover_result": None,
    "landcover_message": None,
    "historical_change_result": None,
    "historical_change_message": None,
    "climate_stress_result": None,
    "climate_stress_message": None,
    "landuse_transition_result": None,
    "landuse_transition_message": None,
    "cause_attribution_result": None,
    "cause_attribution_message": None,
    "change_interpretation_result": None,
    "change_interpretation_message": None,
    "gemini_next_steps": None,
}

for key, default_value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def calculate_aoi(features):
    """Validate drawn polygon features and calculate geodesic area."""

    if not features:
        raise ValueError("Draw a polygon or rectangle on the map.")

    geometries = []

    for feature in features:
        if not isinstance(feature, dict):
            continue

        geometry_data = feature.get("geometry")

        if not geometry_data:
            continue

        try:
            geom = shape(geometry_data)
        except Exception:
            continue

        if geom.geom_type not in ("Polygon", "MultiPolygon"):
            continue

        if geom.is_empty:
            continue

        if not geom.is_valid:
            geom = geom.buffer(0)

        if geom.is_valid and not geom.is_empty:
            geometries.append(geom)

    if not geometries:
        raise ValueError(
            "No valid polygon found. Draw a polygon and try again."
        )

    combined = unary_union(geometries)

    if combined.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError("The selected geometries did not form a polygon.")

    if combined.is_empty or not combined.is_valid:
        raise ValueError("The resulting AOI geometry is invalid.")

    minx, miny, maxx, maxy = combined.bounds

    if (
        minx < -180
        or maxx > 180
        or miny < -90
        or maxy > 90
    ):
        raise ValueError("AOI coordinates are outside valid geographic bounds.")

    geod = Geod(ellps="WGS84")
    area_m2, _ = geod.geometry_area_perimeter(combined)
    area_km2 = abs(area_m2) / 1_000_000

    if area_km2 <= 0:
        raise ValueError("The selected AOI has zero area.")

    if area_km2 > 10_000:
        raise ValueError(
            "AOI exceeds 10,000 km². Select a smaller area."
        )

    return mapping(combined), area_km2


def get_scene_properties(scene):
    """Extract common fields from a STAC catalogue item."""

    properties = scene.get("properties") or {}

    acquisition = (
        properties.get("datetime")
        or properties.get("start_datetime")
        or "Unknown date"
    )

    cloud_cover = properties.get("eo:cloud_cover")

    if cloud_cover is None:
        cloud_cover = properties.get("cloudCover")

    return {
        "id": scene.get("id", "Unknown scene"),
        "date": str(acquisition)[:10],
        "cloud_cover": cloud_cover,
        "satellite": (
            properties.get("platform")
            or properties.get("constellation")
            or "Sentinel-2"
        ),
        "collection": scene.get("collection", "Unknown collection"),
    }


def get_scene_date(scene):
    """Get the acquisition calendar date from a STAC item."""

    properties = scene.get("properties") or {}

    acquisition = (
        properties.get("datetime")
        or properties.get("start_datetime")
    )

    if not acquisition:
        raise ValueError(
            "The selected scene has no usable acquisition date."
        )

    return str(acquisition)[:10]


def show_raster_info(raster_path):
    """Return basic raster metadata for display."""

    raster_path = Path(raster_path)

    with rasterio.open(raster_path) as src:
        return {
            "Bands": src.count,
            "Width": src.width,
            "Height": src.height,
            "CRS": str(src.crs),
            "Pixel size": str(src.res),
            "File size": (
                f"{raster_path.stat().st_size / (1024 * 1024):.2f} MB"
            ),
        }


def invalidate_analysis():
    """Clear cached analysis results when the AOI or scene changes."""

    st.session_state.latest_satellite_raster = None
    st.session_state.latest_ndvi_raster = None
    st.session_state.download_message = None
    st.session_state.ndvi_message = None
    st.session_state.historical_change_result = None
    st.session_state.historical_change_message = None
    st.session_state.cause_attribution_result = None
    st.session_state.cause_attribution_message = None


# ============================================================
# SIMPLE FOUR-STAGE WORKSPACE
# ============================================================
st.markdown("### Environmental Investigation")
st.caption("Choose one stage at a time. Your selected area and completed results remain available as you move between stages.")
STAGES = [
    "1 · Select area & map context",
    "2 · Vegetation maps & change",
    "3 · Climate & land-use analysis",
    "4 · Causes & final interpretation",
]
active_stage = st.radio("Investigation stage", STAGES, horizontal=True, label_visibility="collapsed", key="active_investigation_stage")
st.divider()

if active_stage == STAGES[0]:
    # ============================================================
    # AREA OF INTEREST
    # ============================================================

    st.header("Select Area & Map Context")

    st.caption("Search for a place, draw a polygon or rectangle, then confirm the boundary before running analyses.")

    with st.form("location_search_form"):
        location_query = st.text_input(
            "Find a location",
            placeholder="Example: Purandar, Maharashtra, India",
        )

        find_location = st.form_submit_button("Find location")

    if find_location:
        if not location_query.strip():
            st.warning("Enter a location name first.")
        else:
            try:
                geocoder = Nominatim(
                    user_agent="environmental_forensics_climate_intelligence"
                )

                result = geocoder.geocode(
                    location_query.strip(),
                    timeout=15,
                )

                if result:
                    st.session_state.map_center = [
                        result.latitude,
                        result.longitude,
                    ]
                    st.session_state.map_zoom = 11
                    st.session_state.map_version += 1
                    st.session_state.location_message = (
                        f"Location found: {result.address}"
                    )
                    st.rerun()
                else:
                    st.warning(
                        "Location not found. Try a more specific place name."
                    )

            except Exception as exc:
                st.error(f"Location search failed: {exc}")

    if st.session_state.location_message:
        st.success(st.session_state.location_message)
        st.session_state.location_message = None


    map_object = folium.Map(
        location=st.session_state.map_center,
        zoom_start=st.session_state.map_zoom,
        control_scale=True,
        tiles="OpenStreetMap",
    )

    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "rectangle": True,
            "polygon": True,
            "circle": False,
            "circlemarker": False,
            "marker": False,
        },
        edit_options={
            "edit": True,
            "remove": True,
        },
    ).add_to(map_object)

    map_result = st_folium(
        map_object,
        key=f"aoi_map_{st.session_state.map_version}",
        height=520,
        returned_objects=["all_drawings"],
    )

    if map_result and map_result.get("all_drawings") is not None:
        st.session_state.drawings = map_result["all_drawings"]


    confirm_col, clear_col = st.columns(2)

    with confirm_col:
        confirm_aoi = st.button(
            "Confirm selected AOI",
            type="primary",
            use_container_width=True,
        )

    with clear_col:
        clear_aoi = st.button(
            "Clear AOI",
            use_container_width=True,
        )


    if clear_aoi:
        st.session_state.drawings = []
        st.session_state.confirmed_aoi = None
        st.session_state.confirmed_area_km2 = None
        st.session_state.terrain_result = None
        st.session_state.terrain_message = None
        st.session_state.landcover_result = None
        st.session_state.landcover_message = None
        st.session_state.copernicus_scenes = []
        st.session_state.copernicus_search_done = False
        st.session_state.selected_scene_id = None
        invalidate_analysis()
        st.session_state.map_version += 1
        st.rerun()


    if confirm_aoi:
        try:
            geojson, area_km2 = calculate_aoi(
                st.session_state.drawings
            )

            st.session_state.confirmed_aoi = geojson
            st.session_state.confirmed_area_km2 = area_km2
            st.session_state.terrain_result = None
            st.session_state.terrain_message = None
            st.session_state.landcover_result = None
            st.session_state.landcover_message = None

            st.session_state.copernicus_scenes = []
            st.session_state.copernicus_search_done = False
            st.session_state.selected_scene_id = None

            invalidate_analysis()

            st.success("AOI confirmed successfully.")

        except Exception as exc:
            st.error(f"Could not confirm AOI: {exc}")


    if st.session_state.confirmed_aoi:
        st.subheader("Confirmed Area of Interest")

        area_km2 = st.session_state.confirmed_area_km2

        col1, col2, col3 = st.columns(3)
        col1.metric("Area (km²)", f"{area_km2:,.3f}")
        col2.metric("Area (hectares)", f"{area_km2 * 100:,.2f}")
        col3.metric("Area (m²)", f"{area_km2 * 1_000_000:,.0f}")

        with st.expander("View AOI GeoJSON"):
            st.json(st.session_state.confirmed_aoi)

    else:
        st.info("Draw and confirm an AOI to continue.")



    # ============================================================
    # 7. STEP 4 — LAND-COVER AND HUMAN/NATURAL FEATURES
    # ============================================================

    st.divider()
    st.subheader("Land Cover & Mapped Features")
    st.write(
        "Estimate agricultural and tree-cover areas from ESA WorldCover, then inspect "
        "mapped roads, settlements, rivers, lakes and reservoirs from OpenStreetMap."
    )

    if not st.session_state.confirmed_aoi:
        st.info("Draw and confirm an area above to enable land-cover analysis.")
    else:
        st.caption(
            f"Selected AOI: {st.session_state.confirmed_area_km2:,.3f} km². "
            "The land-cover classification is ESA WorldCover 2021 (10 m nominal resolution)."
        )
        run_landcover = st.button(
            "Analyze Land Cover & Infrastructure",
            type="primary",
            use_container_width=False,
            key="run_landcover_analysis",
        )
        if run_landcover:
            try:
                with st.spinner(
                    "Reading ESA WorldCover satellite classification and querying mapped features..."
                ):
                    result = analyze_landcover(st.session_state.confirmed_aoi)
                st.session_state.landcover_result = result
                st.session_state.landcover_message = None
            except Exception as exc:
                st.session_state.landcover_result = None
                st.session_state.landcover_message = str(exc)

        if st.session_state.landcover_message:
            st.error(f"Land-cover analysis failed: {st.session_state.landcover_message}")

        lc_result = st.session_state.landcover_result
        if lc_result:
            summary = lc_result["summary"]
            mapped = lc_result["mapped_features"]
            st.subheader("Land-Cover Area Estimates")
            m1, m2, m3 = st.columns(3)
            m1.metric("Agricultural / cropland", f"{summary['cropland_km2']:,.2f} km²", f"{summary['cropland_km2'] * 100:,.1f} ha")
            m2.metric("Forest / tree cover", f"{summary['tree_cover_km2']:,.2f} km²", f"{summary['tree_cover_km2'] * 100:,.1f} ha")
            m3.metric("Built-up land", f"{summary['built_up_km2']:,.2f} km²", f"{summary['built_up_km2'] * 100:,.1f} ha")
            m4, m5, m6 = st.columns(3)
            m4.metric("Grassland", f"{summary['grassland_km2']:,.2f} km²")
            m5.metric("Shrubland", f"{summary['shrubland_km2']:,.2f} km²")
            m6.metric("Water bodies + wetlands", f"{summary['worldcover_water_wetlands_km2']:,.2f} km²")

            lc_areas = lc_result["landcover"]["class_areas_km2"]
            lc_rows = [
                {"Land-cover class": name, "Area (km²)": area, "Area (hectares)": area * 100}
                for code, name in __import__("services.landcover", fromlist=["WORLD_COVER_CLASSES"]).WORLD_COVER_CLASSES.items()
                if (area := lc_areas.get(code, 0.0)) > 0
            ]
            if lc_rows:
                lc_df = pd.DataFrame(lc_rows).sort_values("Area (km²)", ascending=False)
                st.bar_chart(lc_df.set_index("Land-cover class")["Area (km²)"])
                st.dataframe(lc_df, use_container_width=True, hide_index=True)

            st.subheader("Roads, Settlements and Water Features")
            r1, r2, r3 = st.columns(3)
            r1.metric("Mapped road length", f"{mapped['road_length_km_total']:,.1f} km")
            r2.metric("Settlement/place features", f"{mapped['settlement_feature_count']:,}")
            r3.metric("Mapped water features", f"{mapped['water_feature_count']:,}")
            w1, w2, w3 = st.columns(3)
            w1.metric("Mapped rivers/streams", f"{mapped['river_stream_length_km']:,.1f} km")
            w2.metric("Mapped lake/water polygons", f"{mapped['mapped_water_area_km2']:,.2f} km²")
            w3.metric("Mapped reservoir area", f"{mapped['reservoir_area_km2']:,.2f} km²")
            if mapped["road_length_km_by_type"]:
                roads_df = pd.DataFrame([
                    {"OSM road class": road_type, "Length (km)": length,
                     "Mapped ways": mapped["road_count_by_type"].get(road_type, 0)}
                    for road_type, length in sorted(mapped["road_length_km_by_type"].items(), key=lambda x: x[1], reverse=True)
                ])
                with st.expander("Road network breakdown"):
                    st.dataframe(roads_df, use_container_width=True, hide_index=True)
            if mapped["settlement_names"]:
                st.write("**Named mapped places:** " + ", ".join(mapped["settlement_names"][:50]))
            st.metric("Mapped residential footprint", f"{mapped['residential_footprint_km2']:,.2f} km²")

            export_rows = [{"Metric": name, "Value": value, "Unit": unit} for name, value, unit in [
                ("Agricultural / cropland area", summary["cropland_km2"], "km²"),
                ("Tree-cover area", summary["tree_cover_km2"], "km²"),
                ("Built-up area", summary["built_up_km2"], "km²"),
                ("Grassland area", summary["grassland_km2"], "km²"),
                ("Shrubland area", summary["shrubland_km2"], "km²"),
                ("WorldCover water and wetlands", summary["worldcover_water_wetlands_km2"], "km²"),
                ("Mapped total road length", mapped["road_length_km_total"], "km"),
                ("Mapped river/stream length", mapped["river_stream_length_km"], "km"),
                ("Mapped lake/water polygon area", mapped["mapped_water_area_km2"], "km²"),
                ("Mapped reservoir area", mapped["reservoir_area_km2"], "km²"),
                ("Mapped residential footprint", mapped["residential_footprint_km2"], "km²"),
            ]]
            st.download_button(
                "Download land-cover and feature summary (CSV)",
                data=pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8"),
                file_name="landcover_infrastructure_summary.csv",
                mime="text/csv",
                key="download_landcover_summary",
            )
            for caveat in lc_result["caveats"]:
                st.warning(caveat)
            st.caption("Land-cover source: " + lc_result["landcover"]["source"])
            st.caption("Mapped feature source: " + mapped["source"])
            st.caption(lc_result["landcover"]["attribution"])
            if lc_result["landcover"]["warnings"]:
                with st.expander("Data access notes"):
                    for warning in lc_result["landcover"]["warnings"]:
                        st.write("- " + warning)




elif active_stage == STAGES[1]:
    # ============================================================
    # 7. STEP 4 — COPERNICUS SCENE SEARCH
    # ============================================================

    st.divider()
    st.subheader("Satellite Data")

    if not st.session_state.confirmed_aoi:
        st.info("Confirm your AOI before searching the catalogue.")

    else:
        today = date.today()

        date_col1, date_col2, cloud_col = st.columns(3)

        with date_col1:
            start_date = st.date_input(
                "Start date",
                value=today - timedelta(days=90),
                key="search_start_date",
            )

        with date_col2:
            end_date = st.date_input(
                "End date",
                value=today,
                key="search_end_date",
            )

        with cloud_col:
            max_cloud_cover = st.slider(
                "Maximum scene cloud cover (%)",
                min_value=0,
                max_value=100,
                value=20,
                key="satellite_cloud_cover",
            )

        if start_date > end_date:
            st.error("Start date must be on or before end date.")

        elif st.button(
            "Search Copernicus scenes",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner("Searching the Copernicus catalogue..."):
                    scenes = search_sentinel2(
                        geometry=st.session_state.confirmed_aoi,
                        start_date=start_date,
                        end_date=end_date,
                        max_cloud_cover=max_cloud_cover,
                        limit=100,
                    )

                if isinstance(scenes, dict):
                    scenes = scenes.get("features", [])

                st.session_state.copernicus_scenes = list(scenes or [])
                st.session_state.copernicus_search_done = True
                st.session_state.selected_scene_id = None

                invalidate_analysis()

                st.session_state.search_message = None

            except Exception as exc:
                st.session_state.copernicus_scenes = []
                st.session_state.copernicus_search_done = False
                st.session_state.search_message = str(exc)

        if st.session_state.search_message:
            st.error(
                f"Copernicus search failed: "
                f"{st.session_state.search_message}"
            )

        if st.session_state.copernicus_search_done:
            scenes = st.session_state.copernicus_scenes

            if not scenes:
                st.warning(
                    "No scenes found. Try expanding the date range "
                    "or increasing the cloud-cover limit."
                )

            else:
                st.success(f"Found {len(scenes)} catalogue item(s).")

                scene_rows = [
                    get_scene_properties(scene)
                    for scene in scenes
                ]

                scene_table = pd.DataFrame(scene_rows)

                st.dataframe(
                    scene_table,
                    use_container_width=True,
                    hide_index=True,
                )

                scene_ids = [row["id"] for row in scene_rows]

                selected_scene_id = st.selectbox(
                    "Choose a scene",
                    options=scene_ids,
                    format_func=lambda scene_id: next(
                        (
                            f"{row['date']} | {row['id']} | "
                            f"Cloud: {row['cloud_cover']}%"
                            for row in scene_rows
                            if row["id"] == scene_id
                        ),
                        scene_id,
                    ),
                    key="scene_selection",
                )

                if (
                    st.session_state.selected_scene_id
                    != selected_scene_id
                ):
                    st.session_state.selected_scene_id = selected_scene_id
                    invalidate_analysis()

                selected_scene = next(
                    (
                        scene for scene in scenes
                        if scene.get("id") == selected_scene_id
                    ),
                    None,
                )

                if selected_scene:
                    with st.expander("Selected scene metadata"):
                        st.json(selected_scene)


    # ============================================================
    # 8. STEP 5 — SATELLITE PIXEL RETRIEVAL
    # ============================================================

    st.divider()
    st.subheader("Retrieve Imagery")

    selected_scene = None

    if (
        st.session_state.confirmed_aoi
        and st.session_state.selected_scene_id
        and st.session_state.copernicus_scenes
    ):
        selected_scene = next(
            (
                scene
                for scene in st.session_state.copernicus_scenes
                if scene.get("id")
                == st.session_state.selected_scene_id
            ),
            None,
        )

    if selected_scene:
        scene_info = get_scene_properties(selected_scene)

        st.write(f"**Selected scene:** `{scene_info['id']}`")
        st.write(f"**Acquisition date:** {scene_info['date']}")

        st.warning(
            "The retrieval request uses the selected date and AOI. "
            "It may return a suitable mosaic rather than the exact "
            "catalogue item ID."
        )

        if st.button(
            "Retrieve satellite pixels",
            type="primary",
            use_container_width=True,
        ):
            try:
                acquisition_date = get_scene_date(selected_scene)

                with st.spinner(
                    "Requesting satellite data and creating the raster..."
                ):
                    raster_path = download_sentinel2_bands(
                        aoi_geojson=st.session_state.confirmed_aoi,
                        acquisition_date=acquisition_date,
                        max_cloud_cover=st.session_state.get(
                            "satellite_cloud_cover", 20
                        ),
                    )

                raster_path = Path(raster_path)

                if not raster_path.exists():
                    raise FileNotFoundError(
                        f"Expected raster was not created: {raster_path}"
                    )

                with rasterio.open(raster_path) as src:
                    if src.count != 5:
                        raise ValueError(
                            f"Expected five bands (B04, B08, B03, B11, SCL), "
                            f"but the raster contains {src.count}. "
                            "Check services/copernicus.py and retrieve again."
                        )

                st.session_state.latest_satellite_raster = str(raster_path)
                st.session_state.latest_ndvi_raster = None
                st.session_state.download_message = None
                st.session_state.ndvi_message = None

            except Exception as exc:
                st.session_state.download_message = str(exc)

        if st.session_state.download_message:
            st.error(
                f"Satellite retrieval failed: "
                f"{st.session_state.download_message}"
            )

    else:
        st.info(
            "Search for satellite scenes above and select a scene to enable retrieval."
        )


    # Display retrieved raster information.
    raster_path = st.session_state.latest_satellite_raster

    if raster_path and Path(raster_path).exists():
        st.subheader("Retrieved Raster Details")

        try:
            raster_info = show_raster_info(raster_path)

            info_cols = st.columns(3)

            for index, (label, value) in enumerate(raster_info.items()):
                info_cols[index % 3].metric(label, value)

            with rasterio.open(raster_path) as src:
                band_names = [
                    src.descriptions[index] or f"Band {index + 1}"
                    for index in range(src.count)
                ]

            st.write("**Band order:** " + " → ".join(band_names))
            st.write(f"**File:** `{raster_path}`")

            with open(raster_path, "rb") as file:
                st.download_button(
                    "Download satellite GeoTIFF",
                    data=file,
                    file_name=Path(raster_path).name,
                    mime="image/tiff",
                    key="download_satellite_geotiff",
                )

        except Exception as exc:
            st.error(f"Could not read raster information: {exc}")


    # ============================================================
    # CLOUD-MASKED NDVI
    # ============================================================

    st.divider()
    st.subheader("Calculate Vegetation Greenness (NDVI)")

    st.markdown(
        """
        **NDVI = (B08 − B04) / (B08 + B04)**

        The analysis excludes selected Scene Classification Layer (SCL)
        classes associated with clouds, cloud shadows, cirrus, snow/ice,
        unclassified pixels, and invalid data.
        """
    )

    raster_path = st.session_state.latest_satellite_raster

    if raster_path and Path(raster_path).exists():
        if st.button(
            "Calculate cloud-masked NDVI",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    "Applying the SCL mask and calculating NDVI..."
                ):
                    result = compute_ndvi(
                        input_path=raster_path,
                        output_path=NDVI_OUTPUT,
                    )

                st.session_state.latest_ndvi_raster = result["output_path"]
                st.session_state.ndvi_message = None

            except Exception as exc:
                st.session_state.ndvi_message = str(exc)

        if st.session_state.ndvi_message:
            st.error(
                f"NDVI calculation failed: "
                f"{st.session_state.ndvi_message}"
            )

    else:
        st.info("Retrieve a five-band satellite raster above before calculating cloud-masked NDVI.")


    ndvi_path = st.session_state.latest_ndvi_raster

    if ndvi_path and Path(ndvi_path).exists():
        try:
            with rasterio.open(ndvi_path) as src:
                ndvi_data = src.read(1, masked=True).astype("float32")
                ndvi_values = np.asarray(ndvi_data.filled(np.nan))

            valid_values = ndvi_values[
                np.isfinite(ndvi_values) & (ndvi_values != -9999)
            ]

            total_pixels = ndvi_values.size
            valid_pixels = valid_values.size
            masked_pixels = total_pixels - valid_pixels

            if valid_pixels == 0:
                st.warning("The NDVI raster contains no valid pixels.")

            else:
                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Mean NDVI",
                    f"{float(np.mean(valid_values)):.3f}",
                )

                col2.metric(
                    "Minimum NDVI",
                    f"{float(np.min(valid_values)):.3f}",
                )

                col3.metric(
                    "Maximum NDVI",
                    f"{float(np.max(valid_values)):.3f}",
                )

                mask_col1, mask_col2, mask_col3 = st.columns(3)

                mask_col1.metric(
                    "Total pixels",
                    f"{total_pixels:,}",
                )

                mask_col2.metric(
                    "Valid pixels",
                    f"{valid_pixels:,}",
                )

                mask_col3.metric(
                    "Masked pixels",
                    f"{masked_pixels:,}",
                    help=(
                        "Pixels excluded from the output because of "
                        "invalid data or the selected SCL classes."
                    ),
                )

                masked_percentage = (
                    100 * masked_pixels / total_pixels
                    if total_pixels
                    else 0
                )

                st.progress(
                    min(1.0, max(0.0, masked_percentage / 100)),
                    text=f"Excluded pixels: {masked_percentage:.1f}%",
                )

                st.caption(
                    "This percentage includes all NoData pixels in the output, "
                    "not only clouds. It is not a direct measurement of cloud "
                    "cover."
                )

                with open(ndvi_path, "rb") as file:
                    st.download_button(
                        "Download cloud-masked NDVI GeoTIFF",
                        data=file,
                        file_name=Path(ndvi_path).name,
                        mime="image/tiff",
                        key="download_masked_ndvi",
                    )

        except Exception as exc:
            st.error(f"Could not read NDVI output: {exc}")


    # ============================================================
    # 10. STEP 7 — NDVI MAP VISUALIZATION
    # ============================================================

    st.divider()
    st.subheader("Vegetation Map")

    ndvi_path = st.session_state.latest_ndvi_raster

    if ndvi_path and Path(ndvi_path).exists():
        try:
            ndvi_values, bounds, raster_crs = load_ndvi_for_display(
                ndvi_path
            )

            valid_values = ndvi_values[np.isfinite(ndvi_values)]

            if valid_values.size == 0:
                st.warning("No valid NDVI pixels are available to display.")

            else:
                fig, ax = plt.subplots(figsize=(11, 7))

                image = ax.imshow(
                    ndvi_values,
                    cmap="RdYlGn",
                    vmin=-1,
                    vmax=1,
                    interpolation="nearest",
                    extent=[
                        bounds.left,
                        bounds.right,
                        bounds.bottom,
                        bounds.top,
                    ],
                    origin="upper",
                )

                colorbar = fig.colorbar(
                    image,
                    ax=ax,
                    shrink=0.82,
                )
                colorbar.set_label("NDVI value")

                ax.set_title("Cloud-Masked Sentinel-2 NDVI")
                ax.set_xlabel(f"Easting / X ({raster_crs})")
                ax.set_ylabel(f"Northing / Y ({raster_crs})")

                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.caption(
                    f"Raster CRS: {raster_crs} | "
                    f"NDVI range: {valid_values.min():.3f} to "
                    f"{valid_values.max():.3f}"
                )

                st.markdown(
                    """
                    **Interpretation guide**

                    - **Negative values:** often water, clouds, or other
                      non-vegetated surfaces.
                    - **Values near zero:** often bare soil or sparse vegetation.
                    - **Higher positive values:** generally indicate greener
                      vegetation.

                    These ranges are indicative, not universal land-cover
                    thresholds. Interpret results in their seasonal and
                    geographic context.
                    """
                )

        except Exception as exc:
            st.error(f"NDVI visualization failed: {exc}")

    else:
        st.info(
            "Calculate cloud-masked NDVI above to display the map here."
        )


    # ============================================================
    # 11. STEP 9 — HISTORICAL VEGETATION CHANGE DETECTION
    # ============================================================

    st.divider()
    st.subheader("Historical Vegetation Change")
    st.markdown(
        "Compare cloud-masked Sentinel-2 NDVI for the same seasonal window in two "
        "different years. The first version selects the lowest-cloud catalogue "
        "scene in each period, then estimates vegetation greenness decline, gain, "
        "and stable/minor-change areas."
    )

    if not st.session_state.confirmed_aoi:
        st.info("Confirm an area of interest above to enable historical change detection.")
    else:
        current_year = date.today().year
        years = list(range(2015, current_year + 1))
        default_baseline = 2021 if 2021 in years else years[0]
        default_comparison = current_year - 1 if current_year - 1 > default_baseline else current_year
        default_comparison = min(default_comparison, current_year)

        hcol1, hcol2, hcol3 = st.columns(3)
        with hcol1:
            baseline_year = st.selectbox(
                "Baseline year", options=years,
                index=years.index(default_baseline), key="historical_baseline_year"
            )
        with hcol2:
            comparison_options = [y for y in years if y > baseline_year]
            if not comparison_options:
                comparison_options = [current_year]
            comparison_default = current_year - 1 if current_year - 1 in comparison_options else comparison_options[-1]
            comparison_year = st.selectbox(
                "Comparison year", options=comparison_options,
                index=comparison_options.index(comparison_default),
                key="historical_comparison_year"
            )
        with hcol3:
            max_historical_cloud = st.slider(
                "Maximum scene cloud cover (%)", 0, 100, 30,
                key="historical_cloud_limit"
            )

        hcol4, hcol5, hcol6 = st.columns(3)
        with hcol4:
            start_month = st.selectbox(
                "Season start month", options=list(range(1, 13)), index=9,
                format_func=lambda m: date(2000, m, 1).strftime("%B"),
                key="historical_start_month"
            )
        with hcol5:
            end_month = st.selectbox(
                "Season end month", options=list(range(1, 13)), index=10,
                format_func=lambda m: date(2000, m, 1).strftime("%B"),
                key="historical_end_month"
            )
        with hcol6:
            decline_threshold = st.slider(
                "NDVI decline threshold", min_value=0.05, max_value=0.40,
                value=0.15, step=0.05, key="historical_decline_threshold",
                help="A pixel is marked as decline when NDVI falls by at least this amount and baseline NDVI meets the vegetation threshold."
            )

        vegetation_threshold = st.slider(
            "Baseline vegetation NDVI threshold", min_value=0.10, max_value=0.70,
            value=0.30, step=0.05, key="historical_vegetation_threshold",
            help="Used to focus the decline estimate on pixels that had detectable vegetation in the baseline image."
        )

        st.caption("For a fair comparison, use the same months in both years. Choose a seasonal window appropriate to the study area, and test alternatives if crop cycles or cloud cover may affect the results.")

        if start_month > end_month:
            st.warning("Season start month must be on or before season end month.")
        elif baseline_year >= comparison_year:
            st.warning("Choose a comparison year later than the baseline year.")
        elif st.button(
            "Run historical vegetation change analysis",
            type="primary", use_container_width=True,
            key="run_historical_change_analysis"
        ):
            try:
                st.session_state.historical_change_result = None
                st.session_state.cause_attribution_result = None
                st.session_state.cause_attribution_message = None
                with st.spinner(
                    "Searching historical Sentinel-2 scenes, retrieving both observations, "
                    "calculating cloud-masked NDVI, and comparing rasters. This can take several minutes..."
                ):
                    result = run_historical_change_detection(
                        aoi_geojson=st.session_state.confirmed_aoi,
                        baseline_year=int(baseline_year),
                        comparison_year=int(comparison_year),
                        start_month=int(start_month),
                        end_month=int(end_month),
                        max_cloud_cover=int(max_historical_cloud),
                        decline_threshold=float(decline_threshold),
                        vegetation_threshold=float(vegetation_threshold),
                        output_dir=PROCESSED_DIR / "historical_change",
                    )
                st.session_state.historical_change_result = result
                st.session_state.historical_change_message = None
            except Exception as exc:
                st.session_state.historical_change_result = None
                st.session_state.historical_change_message = str(exc)

        if st.session_state.historical_change_message:
            st.error("Historical change analysis failed: " + st.session_state.historical_change_message)

        change_result = st.session_state.historical_change_result
        if change_result:
            recs = change_result["records"]
            st.subheader("Selected historical observations")
            scene_df = pd.DataFrame([
                {"Year": r["year"], "Acquisition date": r["date"],
                 "Scene cloud cover (%)": round(r["cloud_cover"], 2),
                 "Catalogue candidates": r["candidate_count"],
                 "Scene ID": r["scene_id"]}
                for r in recs
            ])
            st.dataframe(scene_df, use_container_width=True, hide_index=True)

            metric_cols = st.columns(4)
            metric_cols[0].metric("Vegetation decline", f"{change_result['decline_area_ha']:,.2f} ha")
            metric_cols[1].metric("Vegetation gain", f"{change_result['gain_area_ha']:,.2f} ha")
            metric_cols[2].metric("Stable / minor change", f"{change_result['stable_area_ha']:,.2f} ha")
            metric_cols[3].metric("Mean NDVI change", f"{change_result['mean_delta']:+.3f}")
            st.caption(
                f"Decline covers {change_result['decline_pct']:.1f}% of valid comparison area "
                f"({change_result['valid_area_ha']:,.2f} ha). This is a satellite greenness-change "
                "indicator, not proof of deforestation or its cause."
            )

            map_col1, map_col2 = st.columns(2)
            with map_col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                im = ax.imshow(change_result["baseline"], cmap="RdYlGn", vmin=-1, vmax=1)
                ax.set_title(f"Baseline NDVI — {change_result['baseline_year']}")
                ax.set_axis_off()
                fig.colorbar(im, ax=ax, shrink=0.8, label="NDVI")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            with map_col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                im = ax.imshow(change_result["comparison"], cmap="RdYlGn", vmin=-1, vmax=1)
                ax.set_title(f"Comparison NDVI — {change_result['comparison_year']}")
                ax.set_axis_off()
                fig.colorbar(im, ax=ax, shrink=0.8, label="NDVI")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 6))
            delta_view = np.ma.masked_invalid(change_result["delta"])
            im = ax.imshow(delta_view, cmap="RdYlGn", vmin=-0.5, vmax=0.5)
            ax.set_title(f"NDVI change ({change_result['comparison_year']} − {change_result['baseline_year']})")
            ax.set_axis_off()
            fig.colorbar(im, ax=ax, shrink=0.8, label="NDVI difference")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            with open(change_result["class_path"], "rb") as file:
                st.download_button(
                    "Download vegetation change classification GeoTIFF",
                    data=file,
                    file_name=Path(change_result["class_path"]).name,
                    mime="image/tiff", key="download_vegetation_change_tif"
                )
            with open(change_result["delta_path"], "rb") as file:
                st.download_button(
                    "Download NDVI difference GeoTIFF",
                    data=file,
                    file_name=Path(change_result["delta_path"]).name,
                    mime="image/tiff", key="download_ndvi_delta_tif"
                )
            with open(change_result["csv_path"], "rb") as file:
                st.download_button(
                    "Download change statistics CSV",
                    data=file,
                    file_name=Path(change_result["csv_path"]).name,
                    mime="text/csv", key="download_vegetation_change_csv"
                )
            st.warning(
                "Interpretation limit: this MVP compares one selected low-cloud scene per year, "
                "not a multi-scene seasonal composite. Differences can reflect crop calendars, "
                "harvest, soil background, residual haze/clouds, or scene timing. Review the maps "
                "and scene dates before attributing environmental damage."
            )



elif active_stage == STAGES[2]:
    # ============================================================
    # 12. STEP 10 — CLIMATE STRESS ANALYSIS
    # ============================================================

    st.divider()
    st.header("Climate & Land-Use Analysis")
    st.markdown(
        "Investigate whether rainfall deficits, unusually warm conditions, or low "
        "modelled soil moisture coincide with vegetation changes. Data are retrieved "
        "from the Open-Meteo Historical Weather API using ERA5-Land reanalysis."
    )

    if not st.session_state.confirmed_aoi:
        st.info("Confirm an area of interest above to enable climate stress analysis.")
    else:
        climate_now_year = date.today().year
        cc1, cc2 = st.columns(2)
        with cc1:
            climate_start_year = st.selectbox(
                "Climate analysis start year", options=list(range(1950, climate_now_year + 1)),
                index=max(0, list(range(1950, climate_now_year + 1)).index(max(2015, 1950))),
                key="climate_start_year",
            )
        with cc2:
            climate_end_year_options = list(range(climate_start_year, climate_now_year + 1))
            climate_end_year = st.selectbox(
                "Climate analysis end year", options=climate_end_year_options,
                index=len(climate_end_year_options) - 1, key="climate_end_year",
            )

        baseline_options = list(range(1950, climate_end_year + 1))
        default_base_start = 1991 if 1991 in baseline_options else baseline_options[0]
        default_base_end = 2020 if 2020 in baseline_options else min(climate_end_year, default_base_start + 29)
        bc1, bc2 = st.columns(2)
        with bc1:
            climate_baseline_start = st.selectbox(
                "Baseline start year", options=baseline_options,
                index=baseline_options.index(default_base_start), key="climate_baseline_start",
            )
        with bc2:
            climate_baseline_end_options = [y for y in baseline_options if y >= climate_baseline_start]
            if not climate_baseline_end_options:
                climate_baseline_end_options = [climate_baseline_start]
            climate_baseline_end = st.selectbox(
                "Baseline end year", options=climate_baseline_end_options,
                index=climate_baseline_end_options.index(
                    default_base_end if default_base_end in climate_baseline_end_options
                    else climate_baseline_end_options[-1]
                ), key="climate_baseline_end",
            )

        st.caption(
            "The selected baseline is used to calculate rainfall and temperature anomalies. "
            "For climate interpretation, prefer a multi-decade baseline when sufficient data are available."
        )

        if st.button(
            "Run Climate Stress Analysis", type="primary", use_container_width=True,
            key="run_climate_stress_analysis",
        ):
            try:
                st.session_state.climate_stress_result = None
                st.session_state.climate_stress_message = None
                st.session_state.cause_attribution_result = None
                st.session_state.cause_attribution_message = None
                start_date = date(min(int(climate_start_year), int(climate_baseline_start)), 1, 1)
                end_date = min(date(int(climate_end_year), 12, 31), date.today())
                with st.spinner("Retrieving historical rainfall, temperature and soil-moisture data..."):
                    climate_result = fetch_climate_series(
                        aoi_geojson=st.session_state.confirmed_aoi,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    climate_result["annual_anomalies"] = add_climate_anomalies(
                        climate_result["annual"],
                        int(climate_baseline_start), int(climate_baseline_end),
                    )
                    climate_result["baseline_start_year"] = int(climate_baseline_start)
                    climate_result["baseline_end_year"] = int(climate_baseline_end)
                    st.session_state.climate_stress_result = climate_result
            except Exception as exc:
                st.session_state.climate_stress_result = None
                st.session_state.climate_stress_message = str(exc)

        if st.session_state.climate_stress_message:
            st.error("Climate stress analysis failed: " + st.session_state.climate_stress_message)

        climate_result = st.session_state.climate_stress_result
        if climate_result:
            annual_climate = climate_result["annual_anomalies"]
            monthly_climate = climate_result["monthly"]
            baseline_rows = annual_climate[
                annual_climate["year"].between(
                    climate_result["baseline_start_year"], climate_result["baseline_end_year"]
                )
            ]
            latest_row = annual_climate.iloc[-1]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Latest annual rainfall", f"{latest_row['rainfall_mm']:,.0f} mm")
            m2.metric("Rainfall anomaly", f"{latest_row['rainfall_anomaly_pct']:+.1f}%")
            m3.metric("Mean temperature anomaly", f"{latest_row['temperature_anomaly_c']:+.2f} °C")
            soil_value = latest_row.get("soil_moisture_0_7cm", float("nan"))
            m4.metric("Mean topsoil moisture", f"{soil_value:.3f} m³/m³" if pd.notna(soil_value) else "Unavailable")

            st.subheader("Annual anomaly table")
            display_cols = [
                "year", "rainfall_mm", "rainfall_anomaly_mm", "rainfall_anomaly_pct",
                "temperature_mean_c", "temperature_anomaly_c", "soil_moisture_0_7cm",
                "topsoil_moisture_anomaly", "days_observed",
            ]
            st.dataframe(
                annual_climate[display_cols].rename(columns={
                    "year": "Year", "rainfall_mm": "Annual rainfall (mm)",
                    "rainfall_anomaly_mm": "Rainfall anomaly (mm)",
                    "rainfall_anomaly_pct": "Rainfall anomaly (%)",
                    "temperature_mean_c": "Mean temperature (°C)",
                    "temperature_anomaly_c": "Temperature anomaly (°C)",
                    "soil_moisture_0_7cm": "Topsoil moisture (m³/m³)",
                    "topsoil_moisture_anomaly": "Topsoil moisture anomaly",
                    "days_observed": "Days observed",
                }).round(3), use_container_width=True, hide_index=True,
            )

            csv_data = annual_climate.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download annual climate indicators (CSV)", data=csv_data,
                file_name="climate_stress_annual.csv", mime="text/csv",
                key="download_climate_stress_csv",
            )
            monthly_csv = monthly_climate.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download monthly climate indicators (CSV)", data=monthly_csv,
                file_name="climate_stress_monthly.csv", mime="text/csv",
                key="download_climate_stress_monthly_csv",
            )
            st.caption(
                f"Source: {climate_result['source']}. Coordinates sampled: "
                f"{climate_result['latitude']:.4f}, {climate_result['longitude']:.4f}. "
                f"{climate_result['resolution_note']}"
            )
            st.warning(
                "Climate indicators can support investigation but do not establish that climate caused "
                "vegetation loss. Compare with seasonally matched satellite observations and land-use evidence. "
                "Recent reanalysis years may be incomplete or revised."
            )


    # ============================================================
    # 13. STEP 11 — LAND-USE CHANGE FORENSICS
    # ============================================================

    st.divider()
    st.subheader("Land-Use Change")
    st.markdown(
        "Compare threshold-based land-cover proxies from two seasonally matched Sentinel-2 "
        "observations. The module highlights vegetation-to-built-up, vegetation-to-bare-soil, "
        "and other transitions for further investigation. These are screening proxies, not a "
        "validated land-cover classification."
    )

    if not st.session_state.confirmed_aoi:
        st.info("Confirm an area of interest above to enable land-use change forensics.")
    else:
        lu_year_now = date.today().year
        lu_years = list(range(2015, lu_year_now + 1))
        lu1, lu2, lu3 = st.columns(3)
        with lu1:
            lu_baseline = st.selectbox("Land-use baseline year", lu_years,
                index=lu_years.index(2021) if 2021 in lu_years else 0, key="lu_baseline_year")
        with lu2:
            lu_comp_options = [y for y in lu_years if y > lu_baseline] or [lu_year_now]
            lu_default_comp = lu_year_now - 1 if lu_year_now - 1 in lu_comp_options else lu_comp_options[-1]
            lu_comparison = st.selectbox("Land-use comparison year", lu_comp_options,
                index=lu_comp_options.index(lu_default_comp), key="lu_comparison_year")
        with lu3:
            lu_cloud = st.slider("Maximum scene cloud cover (%)", 0, 100, 30, key="lu_cloud_limit")
        lu4, lu5, lu6 = st.columns(3)
        with lu4:
            lu_start_month = st.selectbox("Season start month", list(range(1, 13)), index=9,
                format_func=lambda m: date(2000, m, 1).strftime("%B"), key="lu_start_month")
        with lu5:
            lu_end_month = st.selectbox("Season end month", list(range(1, 13)), index=10,
                format_func=lambda m: date(2000, m, 1).strftime("%B"), key="lu_end_month")
        with lu6:
            lu_vegetation_threshold = st.slider("Vegetation NDVI threshold", 0.15, 0.70, 0.35, 0.05,
                key="lu_vegetation_threshold", help="Higher values are more conservative for identifying green vegetation.")
        lu_builtup_threshold = st.slider("Built-up proxy NDBI threshold", -0.10, 0.30, 0.05, 0.05,
            key="lu_builtup_threshold", help="This is a spectral proxy; bare soil can also have positive NDBI.")
        st.caption("Use the same months in both years. Choose a season appropriate to your study area; crop cycles and acquisition dates can affect class labels.")
        if lu_start_month > lu_end_month:
            st.warning("Season start month must be on or before season end month.")
        elif lu_baseline >= lu_comparison:
            st.warning("Choose a comparison year later than the baseline year.")
        elif st.button("Run Land-Use Change Forensics", type="primary", use_container_width=True, key="run_landuse_forensics"):
            try:
                st.session_state.landuse_transition_result = None
                st.session_state.landuse_transition_message = None
                st.session_state.cause_attribution_result = None
                st.session_state.cause_attribution_message = None
                with st.spinner("Retrieving seasonally matched Sentinel-2 observations and classifying land-cover proxies..."):
                    lu_result = run_landuse_transition_analysis(
                        aoi_geojson=st.session_state.confirmed_aoi,
                        baseline_year=int(lu_baseline), comparison_year=int(lu_comparison),
                        start_month=int(lu_start_month), end_month=int(lu_end_month),
                        max_cloud_cover=int(lu_cloud), vegetation_threshold=float(lu_vegetation_threshold),
                        builtup_ndbi_threshold=float(lu_builtup_threshold),
                        output_dir=PROCESSED_DIR / "landuse_forensics",
                    )
                st.session_state.landuse_transition_result = lu_result
            except Exception as exc:
                st.session_state.landuse_transition_result = None
                st.session_state.landuse_transition_message = str(exc)
        if st.session_state.landuse_transition_message:
            st.error("Land-use change analysis failed: " + st.session_state.landuse_transition_message)
        lu_result = st.session_state.landuse_transition_result
        if lu_result:
            st.subheader("Selected satellite observations")
            st.dataframe(pd.DataFrame([{"Year": r["year"], "Acquisition date": r["date"],
                "Scene cloud cover (%)": round(r["cloud_cover"], 2), "Catalogue candidates": r["candidate_count"],
                "Scene ID": r["scene_id"]} for r in lu_result["records"]]), use_container_width=True, hide_index=True)
            st.metric("Valid comparison area", f"{lu_result['valid_area_ha']:,.2f} ha")
            st.subheader("Land-cover proxy area by year")
            area_df = pd.DataFrame([{"Class": name, **vals} for name, vals in lu_result["class_area"].items()])
            st.dataframe(area_df.round(2), use_container_width=True, hide_index=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            area_plot = area_df.set_index("Class")[["baseline_ha", "comparison_ha"]]
            area_plot.plot(kind="bar", ax=ax)
            ax.set_ylabel("Area (ha)"); ax.set_xlabel(""); ax.set_title("Land-cover proxy area comparison")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            map_cols = st.columns(3)
            for col, array, title in zip(map_cols,
                [lu_result["baseline_class"], lu_result["comparison_class"], lu_result["change_map"]],
                [f"Baseline proxy classes — {lu_result['baseline_year']}", f"Comparison proxy classes — {lu_result['comparison_year']}", "Transition map"]):
                with col:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    im = ax.imshow(np.ma.masked_where(array == 0, array), interpolation="nearest")
                    ax.set_title(title); ax.set_axis_off(); fig.colorbar(im, ax=ax, shrink=0.75)
                    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            st.subheader("Largest mapped transitions")
            st.dataframe(lu_result["transitions"].head(20).round(2), use_container_width=True, hide_index=True)
            for key, label, filename in [("class_path", "Download transition GeoTIFF", None),
                                         ("csv_path", "Download transition summary CSV", None),
                                         ("stats_path", "Download class-area CSV", None)]:
                file_path = Path(lu_result[key])
                with open(file_path, "rb") as fh:
                    st.download_button(label, data=fh, file_name=file_path.name,
                        mime="image/tiff" if file_path.suffix.lower() == ".tif" else "text/csv",
                        key=f"download_lu_{key}")
            st.warning(lu_result["limitations"])
            st.caption("Source: " + lu_result["source"] + ". The maps use raster grid coordinates and are intended for screening; validate proxy transitions before treating them as evidence of confirmed land conversion.")


else:
    # ============================================================
    # 12. UNIFIED ENVIRONMENTAL CAUSE ATTRIBUTION
    # ============================================================

    st.divider()
    st.subheader("Possible Causes (Evidence Ranking)")
    st.markdown(
        "Combine the completed outputs from historical vegetation change, climate stress, "
        "and land-use transition analysis. Scores rank the available evidence; they are "
        "not probabilities and do not establish causation."
    )

    ready_hist = st.session_state.get("historical_change_result")
    ready_climate = st.session_state.get("climate_stress_result")
    ready_landuse = st.session_state.get("landuse_transition_result")

    status_cols = st.columns(3)
    status_cols[0].write("**Module 1 — Vegetation**")
    (status_cols[0].success if ready_hist else status_cols[0].info)("Results available" if ready_hist else "Not run yet")
    status_cols[1].write("**Module 2 — Climate**")
    (status_cols[1].success if ready_climate else status_cols[1].info)("Results available" if ready_climate else "Not run yet")
    status_cols[2].write("**Module 3 — Land use**")
    (status_cols[2].success if ready_landuse else status_cols[2].info)("Results available" if ready_landuse else "Not run yet")

    if not st.session_state.confirmed_aoi:
        st.info("Confirm an area of interest before running cause attribution.")
    elif not any([ready_hist, ready_climate, ready_landuse]):
        st.info("For the strongest assessment, run vegetation, climate, and land-use comparisons for the same area and comparable periods.")
    else:
        attribution_year_options = sorted({
            int(r["comparison_year"]) for r in [ready_hist, ready_landuse]
            if r and r.get("comparison_year") is not None
        })
        if ready_climate and isinstance(ready_climate.get("annual_anomalies"), pd.DataFrame):
            attribution_year_options = sorted(set(attribution_year_options).union(
                int(y) for y in ready_climate["annual_anomalies"]["year"].tolist()
            ))
        if not attribution_year_options:
            attribution_year_options = [date.today().year - 1]
        default_attribution_year = (
            int(ready_hist["comparison_year"]) if ready_hist else
            int(ready_landuse["comparison_year"]) if ready_landuse else
            attribution_year_options[-1]
        )
        if default_attribution_year not in attribution_year_options:
            default_attribution_year = attribution_year_options[-1]
        attribution_year = st.selectbox(
            "Year to investigate", attribution_year_options,
            index=attribution_year_options.index(default_attribution_year),
            key="cause_attribution_year",
        )
        if st.button("Run Environmental Cause Attribution", type="primary", use_container_width=True, key="run_cause_attribution"):
            try:
                with st.spinner("Aligning evidence and evaluating environmental hypotheses..."):
                    st.session_state.cause_attribution_result = build_cause_attribution(
                        historical=ready_hist, climate=ready_climate, landuse=ready_landuse,
                        target_year=int(attribution_year),
                    )
                    st.session_state.cause_attribution_message = None
            except Exception as exc:
                st.session_state.cause_attribution_result = None
                st.session_state.cause_attribution_message = str(exc)

        if st.session_state.get("cause_attribution_message"):
            st.error("Cause attribution failed: " + st.session_state.cause_attribution_message)

        attribution = st.session_state.get("cause_attribution_result")
        if attribution:
            st.subheader("Investigation summary")
            v = attribution["vegetation"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Vegetation decline", f"{v['decline_area_ha']:,.2f} ha" if v.get("decline_area_ha") is not None else "Unavailable")
            m2.metric("Mean NDVI change", f"{v['mean_delta']:+.3f}" if v.get("mean_delta") is not None else "Unavailable")
            climate_ev = attribution["climate"]
            m3.metric("Rainfall anomaly", f"{climate_ev['rainfall_anomaly_pct']:+.1f}%" if climate_ev.get("rainfall_anomaly_pct") is not None else "Unavailable")

            st.subheader("Ranked hypotheses")
            hypothesis_table = attribution["hypotheses"]
            st.dataframe(hypothesis_table, use_container_width=True, hide_index=True)
            st.caption("Evidence score is a transparent screening score, not a probability that a cause is true. Missing evidence is shown as unavailable rather than scored as zero.")

            if climate_ev.get("available"):
                st.subheader(f"Climate evidence for {climate_ev['year_used']}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Rainfall anomaly", f"{climate_ev['rainfall_anomaly_pct']:+.1f}%" if climate_ev.get("rainfall_anomaly_pct") is not None else "Unavailable")
                c2.metric("Temperature anomaly", f"{climate_ev['temperature_anomaly_c']:+.2f} °C" if climate_ev.get("temperature_anomaly_c") is not None else "Unavailable")
                c3.metric("Topsoil moisture anomaly", f"{climate_ev['soil_moisture_anomaly']:+.3f}" if climate_ev.get("soil_moisture_anomaly") is not None else "Unavailable")
                st.caption(climate_ev["message"])
            else:
                st.warning("Climate evidence is unavailable for the selected investigation year. Run Module 2 with a date range that includes this year.")

            overlap = attribution["landuse_overlap"]
            st.subheader("Spatial overlap evidence")
            if overlap.get("available"):
                o1, o2, o3 = st.columns(3)
                o1.metric("Decline overlapping built-up proxy", f"{overlap['decline_to_builtup_ha']:.2f} ha" if overlap.get("decline_to_builtup_ha") is not None else "Unavailable")
                o2.metric("Decline overlapping bare-soil proxy", f"{overlap['decline_to_bare_ha']:.2f} ha" if overlap.get("decline_to_bare_ha") is not None else "Unavailable")
                o3.metric("Decline overlapping water proxy", f"{overlap['decline_to_water_ha']:.2f} ha" if overlap.get("decline_to_water_ha") is not None else "Unavailable")
            st.caption(overlap.get("message", "Spatial overlap not available."))

            report = {
                "investigation_year": attribution["target_year"],
                "vegetation_summary": v,
                "climate_evidence": climate_ev,
                "landuse_overlap": overlap,
                "hypotheses": hypothesis_table.to_dict(orient="records"),
                "limitations": attribution["limitations"],
            }
            import json
            report_json = json.dumps(report, indent=2, default=str).encode("utf-8")
            st.download_button(
                "Download forensic attribution report (JSON)", data=report_json,
                file_name=f"environmental_forensic_attribution_{attribution['target_year']}.json",
                mime="application/json", key="download_cause_attribution_json",
            )
            st.download_button(
                "Download ranked hypotheses (CSV)",
                data=hypothesis_table.to_csv(index=False).encode("utf-8"),
                file_name=f"environmental_forensic_hypotheses_{attribution['target_year']}.csv",
                mime="text/csv", key="download_cause_attribution_csv",
            )



    # ============================================================
    # 13. COMPREHENSIVE ENVIRONMENTAL CHANGE INTERPRETATION
    # ============================================================

    st.divider()
    st.header("Final Environmental Interpretation")
    st.caption("Summarizes observed changes, plausible explanations, and supporting or conflicting evidence. Potential causes remain hypotheses until verified.")

    interpret_hist = st.session_state.get("historical_change_result")
    interpret_climate = st.session_state.get("climate_stress_result")
    interpret_landuse = st.session_state.get("landuse_transition_result")
    if not any([interpret_hist, interpret_climate, interpret_landuse]):
        st.info("Run at least one change-analysis module first; using all three gives a more complete interpretation.")
    else:
        year_candidates = sorted({
            int(result[key]) for result, key in [(interpret_hist, "comparison_year"), (interpret_landuse, "comparison_year")]
            if result and result.get(key) is not None
        })
        if interpret_climate and isinstance(interpret_climate.get("annual_anomalies"), pd.DataFrame):
            year_candidates = sorted(set(year_candidates).union(
                int(y) for y in interpret_climate["annual_anomalies"]["year"].tolist()
            ))
        if not year_candidates:
            year_candidates = [date.today().year - 1]
        default_year = int(interpret_hist["comparison_year"]) if interpret_hist else (int(interpret_landuse["comparison_year"]) if interpret_landuse else year_candidates[-1])
        if default_year not in year_candidates:
            default_year = year_candidates[-1]
        interpretation_year = st.selectbox(
            "Interpretation year", year_candidates, index=year_candidates.index(default_year),
            key="change_interpretation_year"
        )
        if st.button("Explain Environmental Changes", type="primary", use_container_width=True, key="run_change_interpretation"):
            try:
                st.session_state.change_interpretation_result = build_change_interpretation(
                    historical=interpret_hist, climate=interpret_climate, landuse=interpret_landuse,
                    target_year=int(interpretation_year)
                )
                st.session_state.change_interpretation_message = None
            except Exception as exc:
                st.session_state.change_interpretation_result = None
                st.session_state.change_interpretation_message = str(exc)
        if st.session_state.get("change_interpretation_message"):
            st.error("Environmental interpretation failed: " + st.session_state.change_interpretation_message)
        interpretation = st.session_state.get("change_interpretation_result")
        if interpretation:
            st.subheader("Overall interpretation")
            st.write(interpretation["summary"])
            st.caption("Investigation year: " + str(interpretation["investigation_year"]) + " | Baselines are shown below; interpretation quality depends on comparable dates and data coverage.")
            st.subheader("Indicator-by-indicator explanations")
            st.dataframe(interpretation["indicators"], use_container_width=True, hide_index=True)
            st.subheader("Cross-indicator evidence")
            for evidence_line in interpretation["cross_indicator_evidence"]:
                st.write("• " + evidence_line)
            st.subheader("Baseline and comparison periods")
            st.dataframe(pd.DataFrame([{"Item": k, "Value": v} for k, v in interpretation["baseline_context"].items()]), use_container_width=True, hide_index=True)
            import json
            interpretation_report = {
                "investigation_year": interpretation["investigation_year"],
                "baseline_context": interpretation["baseline_context"],
                "summary": interpretation["summary"],
                "indicators": interpretation["indicators"].to_dict(orient="records"),
                "cross_indicator_evidence": interpretation["cross_indicator_evidence"],
                "limitations": interpretation["limitations"],
            }
            st.download_button(
                "Download environmental interpretation report (JSON)",
                data=json.dumps(interpretation_report, indent=2, default=str).encode("utf-8"),
                file_name=f"environmental_change_interpretation_{interpretation['investigation_year']}.json",
                mime="application/json", key="download_change_interpretation_json"
            )
            st.download_button(
                "Download indicator explanations (CSV)",
                data=interpretation["indicators"].to_csv(index=False).encode("utf-8"),
                file_name=f"environmental_change_interpretation_{interpretation['investigation_year']}.csv",
                mime="text/csv", key="download_change_interpretation_csv"
            )



    # ============================================================
    # 14. AUTOMATED FORENSIC REPORT
    # ============================================================
    st.divider()
    st.header("Automated Forensic Report")
    st.caption(
        "Generate a consolidated PDF from completed analyses, including available maps, measured changes, evidence, and recommended actions."
    )

    st.subheader("Gemini Environmental Prevention & Action Plan")
    st.write(
        "Gemini interprets detected environmental changes—such as vegetation decline, warming, rainfall or soil-moisture stress, and land-use transitions—and recommends practical measures to prevent or reduce further damage. Recommendations are tied to the evidence available in this session."
    )
    gemini_inputs_ready = any(
        st.session_state.get(key) is not None for key in [
            "historical_change_result", "climate_stress_result", "landuse_transition_result",
            "cause_attribution_result", "change_interpretation_result"
        ]
    )
    if st.button(
        "Generate Environmental Prevention Plan with Gemini",
        type="secondary", use_container_width=True,
        key="generate_gemini_next_steps",
        disabled=not gemini_inputs_ready
    ):
        try:
            with st.spinner("Gemini is reviewing the available evidence..."):
                st.session_state["gemini_next_steps"] = generate_gemini_next_steps(
                    location=st.session_state.get("location_query", "Selected area"),
                    historical=st.session_state.get("historical_change_result"),
                    climate=st.session_state.get("climate_stress_result"),
                    landuse=st.session_state.get("landuse_transition_result"),
                    attribution=st.session_state.get("cause_attribution_result"),
                    interpretation=st.session_state.get("change_interpretation_result"),
                    st_module=st,
                )
            st.success("Environmental prevention plan generated.")
        except Exception as exc:
            st.error(str(exc))

    if st.session_state.get("gemini_next_steps"):
        st.markdown(st.session_state["gemini_next_steps"])
        st.caption("Use this as a screening-level action plan. Confirm affected locations and causes with field evidence and local environmental/agricultural experts before implementation.")

    report_hist = st.session_state.get("historical_change_result")
    report_climate = st.session_state.get("climate_stress_result")
    report_landuse = st.session_state.get("landuse_transition_result")
    report_attribution = st.session_state.get("cause_attribution_result")
    report_interpretation = st.session_state.get("change_interpretation_result")
    report_ready = any([report_hist, report_climate, report_landuse, report_attribution, report_interpretation])

    if not st.session_state.get("confirmed_aoi"):
        st.info("Select and confirm an area before generating the report.")
    elif not report_ready:
        st.info("Run at least one analysis module first. For a complete report, run vegetation, climate, land-use, cause attribution, and interpretation.")
    else:
        report_title = st.text_input(
            "Report title", value="Environmental Forensic Assessment",
            key="forensic_report_title"
        )
        report_location = st.text_input(
            "Study area / location name", value=st.session_state.get("location_query", "Selected area"),
            key="forensic_report_location"
        )
        st.caption("Only results already generated in this Streamlit session are included. Missing modules are explicitly marked as unavailable.")
        if st.button("Generate Professional PDF Report", type="primary", use_container_width=True, key="generate_forensic_pdf"):
            try:
                with st.spinner("Assembling maps, charts, evidence, and recommendations..."):
                    pdf_bytes = build_forensic_pdf(
                        title=report_title.strip() or "Environmental Forensic Assessment",
                        location=report_location.strip() or "Selected area",
                        aoi_geojson=st.session_state.get("confirmed_aoi"),
                        area_km2=st.session_state.get("confirmed_area_km2"),
                        historical=report_hist,
                        climate=report_climate,
                        landuse=report_landuse,
                        attribution=report_attribution,
                        interpretation=report_interpretation,
                        gemini_guidance=st.session_state.get("gemini_next_steps"),
                    )
                    st.session_state["forensic_pdf_bytes"] = pdf_bytes
                    st.session_state["forensic_pdf_name"] = (
                        f"environmental_forensic_report_{date.today().isoformat()}.pdf"
                    )
                    st.success("PDF report generated.")
            except Exception as exc:
                st.error("Report generation failed: " + str(exc))

        if st.session_state.get("forensic_pdf_bytes"):
            st.download_button(
                "Download Environmental Forensic Report (PDF)",
                data=st.session_state["forensic_pdf_bytes"],
                file_name=st.session_state.get("forensic_pdf_name", "environmental_forensic_report.pdf"),
                mime="application/pdf",
                key="download_full_forensic_pdf",
                use_container_width=True,
            )


st.divider()
st.caption("AI-Powered Environmental Forensics & Climate Intelligence · Research prototype · Sentinel-2 data via Copernicus")
