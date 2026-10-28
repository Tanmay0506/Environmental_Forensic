"""
Gemini-based environmental prevention and intervention planning.
Produces evidence-grounded mitigation actions from analysis outputs supplied by the app.
Requires GEMINI_API_KEY (environment variable or Streamlit secrets).
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd


def _compact(value: Any, depth: int = 0):
    """Convert analysis results into a bounded JSON-safe summary; omit large raster arrays."""
    if depth > 4:
        return "[nested data omitted]"
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return {
            "type": "table",
            "columns": [str(c) for c in value.columns[:12]],
            "row_count": int(len(value)),
            "sample_rows": value.head(12).replace({np.nan: None}).to_dict(orient="records"),
        }
    if isinstance(value, pd.Series):
        return value.head(12).to_dict()
    if isinstance(value, np.ndarray):
        if value.size > 400:
            finite = value[np.isfinite(value)] if np.issubdtype(value.dtype, np.number) else np.array([])
            return {
                "type": "raster_array",
                "shape": list(value.shape),
                "valid_pixel_count": int(finite.size),
                "min": float(np.min(finite)) if finite.size else None,
                "max": float(np.max(finite)) if finite.size else None,
                "mean": float(np.mean(finite)) if finite.size else None,
                "note": "Raster summarized; full array omitted.",
            }
        return value.tolist()
    if isinstance(value, dict):
        result = {}
        for k, v in list(value.items())[:60]:
            # Avoid binary/large image/raster payloads and raw AOI geometries.
            if any(token in str(k).lower() for token in ("image_bytes", "binary", "raw_raster", "geometry", "geojson", "aoi")):
                continue
            result[str(k)] = _compact(v, depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_compact(v, depth + 1) for v in value[:30]]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)):
        return value
    try:
        return str(value)[:500]
    except Exception:
        return repr(value)[:500]


def _get_api_key(st=None):
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key and st is not None:
        try:
            key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            key = None
    return key


def generate_gemini_next_steps(
    *, location: str, historical=None, climate=None, landuse=None,
    attribution=None, interpretation=None, st_module=None
) -> str:
    """Generate an evidence-grounded prevention plan; raises actionable errors for UI handling."""
    api_key = _get_api_key(st_module)
    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. Set GEMINI_API_KEY in your environment or Streamlit secrets."
        )
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "Gemini SDK is missing. Install it with: pip install google-genai"
        ) from exc

    payload = {
        "study_area": location,
        "historical_vegetation": _compact(historical),
        "climate_stress": _compact(climate),
        "land_use_change": _compact(landuse),
        "cause_attribution": _compact(attribution),
        "integrated_interpretation": _compact(interpretation),
    }
    if all(payload[k] is None for k in payload if k != "study_area"):
        raise RuntimeError("No analysis results found. Run at least one analysis before asking Gemini for next steps.")

    prompt = f"""
You are the Environmental Prevention and Intervention Engine for an environmental
forensics platform. Your job is NOT merely to summarize results or list more analyses.
Use the supplied measured findings to recommend practical actions that can prevent,
reduce, or help reverse further environmental damage.

The study area and analysis results are the only evidence you may treat as observed.
Do not invent measurements, named locations, causes, affected communities, budgets,
or completed interventions. If a value or module is absent, say it is unavailable.

Decision logic:
1. Identify each meaningful observed problem in the evidence, such as vegetation/NDVI
   decline, increasing temperature, rainfall deficit, soil-moisture decline, or land-use
   transition. Quote actual values, comparison period, units, and area only when supplied.
   If there is no evidence for a problem, do not claim it exists.
2. Separate observations from hypotheses. Do not claim that one variable caused another
   based only on correlation. Explain what evidence would confirm or weaken each hypothesis.
3. Recommend prevention, mitigation, restoration, or adaptation measures matched to the
   observed problem and local context. For vegetation decline, consider protecting existing
   vegetation, locally appropriate native restoration, agroforestry, soil cover, and grazing,
   fire, or harvest checks where relevant. For heat stress, consider tree/shade cover,
   heat-resilient land management, and heat monitoring. For rainfall/soil-moisture stress,
   consider rainwater harvesting, soil-moisture retention, mulching, contour measures, and
   efficient irrigation where terrain, water balance, and local practice support them. For
   land conversion, consider verifying transitions, protecting sensitive areas, and planning
   or regulatory review. These are candidate actions, not assumptions that all are suitable.
4. Rank actions by urgency, expected environmental benefit, feasibility, and confidence in
   the evidence. Never invent a numeric impact estimate or cost.
5. Make the plan implementable: give a suggested location/target area only if the data supports
   it; timing (immediate / next season / longer term); suggested responsible stakeholder type;
   evidence or approvals needed before implementation; and a measurable success indicator.
6. Include low-regret actions that can begin now, plus targeted investigations needed before
   higher-cost or irreversible actions.
7. Explain what to monitor after intervention and when to reassess. Use comparable seasons
   for satellite/NDVI comparisons.
8. Explicitly note limitations: NDVI is a greenness indicator, not direct proof of vegetation
   area; land-cover outputs may be classification proxies; gridded reanalysis climate data are
   not local station measurements. Field verification may be required.

Output format:
# Environmental Condition Summary
Summarize only the most important measured findings and their periods.

# Priority Problems
For each evidence-supported problem: observed evidence; likely explanations (clearly labelled
as hypotheses); confidence and missing evidence.

# Prevention and Mitigation Plan
Provide 3–7 ranked actions. For each action include:
- Priority (High/Medium/Low) and timing
- Problem addressed and evidence supporting the action
- Specific action and how it may prevent/reduce damage
- Suggested target/area (only if spatial evidence supports it)
- Suggested responsible stakeholder type
- What must be checked/approved before acting
- Success indicator and follow-up monitoring schedule

# First Three Actions
List the three most useful practical actions to start with, distinguishing actions that can
start now from actions requiring field validation or official approval.

# What Could Change This Recommendation
State the key uncertainties and what additional evidence would change the plan.

# Monitoring and Success Criteria
Specify measurable indicators, a reasonable monitoring cadence, and how to compare results.

Be concise but concrete, around 600–900 words maximum. Do not present recommendations as
proof of causation, emergency advice, or a substitute for local expert/authority review.

Analysis summary JSON:
{json.dumps(payload, ensure_ascii=False, default=str)[:24000]}
"""
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model, contents=prompt)
        text = getattr(response, "text", None)
        if not text or not str(text).strip():
            raise RuntimeError("Gemini returned an empty response. Try again or check the configured model.")
        return str(text).strip()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Gemini request failed: {exc}") from exc
