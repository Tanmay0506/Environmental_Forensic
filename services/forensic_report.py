"""
Professional PDF report generator for the Environmental Forensics project.
Consumes existing analysis results; it does not fabricate missing data.
"""
from __future__ import annotations

from io import BytesIO
from datetime import date
import math
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    Image as RLImage, KeepTogether
)
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics


def _safe(value, default="Not available"):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return default
        return f"{float(value):,.3f}"
    return str(value)


def _table(rows, widths=None, header=True):
    data = [[Paragraph(str(cell).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), _styles["table"]) for cell in row] for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#D7DEE8")),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#EAF0F6")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#16324F")),
        ]
    t.setStyle(TableStyle(commands))
    return t


def _figure_image(fig, story, width=165*mm):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = RLImage(buf, width=width, height=width * 0.56)
    story.append(img)
    story.append(Spacer(1, 4*mm))


def _array_map(array, title, story):
    if array is None:
        return
    try:
        arr = np.asarray(array)
        if arr.ndim != 2 or arr.size == 0:
            return
        fig, ax = plt.subplots(figsize=(8, 4.4))
        masked = np.ma.masked_invalid(arr.astype(float))
        ax.imshow(masked, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()
        fig.tight_layout()
        _figure_image(fig, story, 155*mm)
    except Exception:
        return


def build_forensic_pdf(
    *, title, location, aoi_geojson=None, area_km2=None,
    historical=None, climate=None, landuse=None,
    attribution=None, interpretation=None
):
    """Return complete PDF bytes. Any missing module is transparently labelled."""
    global _styles
    styles = getSampleStyleSheet()
    _styles = {
        "table": ParagraphStyle("ReportTable", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.5, leading=9),
    }
    title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold",
                                 fontSize=22, leading=26, textColor=colors.HexColor("#16324F"),
                                 alignment=TA_CENTER, spaceAfter=8*mm)
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=10,
                                   textColor=colors.HexColor("#526477"), alignment=TA_CENTER)
    h1 = ParagraphStyle("SectionHeading", parent=styles["Heading1"], fontSize=15, leading=18,
                        textColor=colors.HexColor("#16324F"), spaceBefore=5*mm, spaceAfter=3*mm)
    h2 = ParagraphStyle("SubHeading", parent=styles["Heading2"], fontSize=11, leading=14,
                        textColor=colors.HexColor("#235789"), spaceBefore=3*mm, spaceAfter=2*mm)
    body = ParagraphStyle("ReportBody", parent=styles["BodyText"], fontSize=9, leading=13, spaceAfter=2.5*mm)
    small = ParagraphStyle("Small", parent=body, fontSize=7.5, leading=10, textColor=colors.HexColor("#536273"))
    bullet = ParagraphStyle("Bullet", parent=body, leftIndent=10, firstLineIndent=-7)

    story = []
    story += [
        Spacer(1, 22*mm),
        Paragraph(title or "Environmental Forensic Assessment", title_style),
        Paragraph("AI-Powered Environmental Forensics &amp; Climate Intelligence", subtitle_style),
        Spacer(1, 12*mm),
        _table([
            ["Report field", "Value"],
            ["Study area", location or "Selected area"],
            ["Report generated", date.today().isoformat()],
            ["Area of interest", f"{_safe(area_km2)} km²" if area_km2 is not None else "Not available"],
            ["Analyses included", ", ".join(name for name, result in [
                ("Historical vegetation", historical), ("Climate stress", climate),
                ("Land-use change", landuse), ("Cause attribution", attribution),
                ("Change interpretation", interpretation)] if result) or "None"],
        ], [45*mm, 115*mm]),
        Spacer(1, 8*mm),
        Paragraph("<b>Purpose.</b> Consolidate available geospatial and climate analyses into a screening-level environmental assessment. The report distinguishes observed measurements from possible explanations and does not treat correlations or heuristic scores as proof of causation.", body),
        Paragraph("<b>Evidence coverage.</b> This PDF uses results already calculated in the current application session. Missing modules are identified as unavailable; no missing values are inferred.", body),
        PageBreak(),
    ]

    # Executive summary
    story.append(Paragraph("1. Executive Summary", h1))
    if interpretation:
        story.append(Paragraph(str(interpretation.get("summary", "Interpretation summary unavailable.")), body))
    elif attribution:
        story.append(Paragraph("Cause-attribution results are available, but the full change-interpretation module has not been run. See the evidence and hypothesis sections below.", body))
    else:
        story.append(Paragraph("A consolidated interpretation was not available when this report was generated. Findings below summarize the available individual analysis modules.", body))

    summary_rows = [["Indicator", "Result", "Interpretation note"]]
    if historical:
        summary_rows += [
            ["Vegetation period", f"{historical.get('baseline_year')} → {historical.get('comparison_year')}", "Historical NDVI comparison"],
            ["Mean NDVI change", _safe(historical.get("mean_delta")), "Greenness indicator; not direct vegetation area"],
            ["Decline area", f"{_safe(historical.get('decline_area_ha'))} ha", "Threshold-based mapped decline"],
            ["Gain area", f"{_safe(historical.get('gain_area_ha'))} ha", "Threshold-based mapped gain"],
        ]
    else:
        summary_rows.append(["Vegetation", "Unavailable", "Run historical vegetation analysis"])
    if climate:
        summary_rows += [
            ["Climate period", f"{climate.get('start_date')} → {climate.get('end_date')}", "Open-Meteo ERA5-Land reanalysis"],
            ["Climate data source", climate.get("source", "Not available"), climate.get("resolution_note", "")],
        ]
    else:
        summary_rows.append(["Climate", "Unavailable", "Run climate stress analysis"])
    if landuse:
        summary_rows.append(["Land-use period", f"{landuse.get('baseline_year')} → {landuse.get('comparison_year')}", "Land-cover proxy transition analysis"])
    else:
        summary_rows.append(["Land use", "Unavailable", "Run land-use change analysis"])
    story.append(_table(summary_rows, [38*mm, 43*mm, 79*mm]))

    # AOI context
    story.append(Paragraph("2. Study Area &amp; Data Context", h1))
    story.append(Paragraph(f"Study area label: <b>{location}</b>. AOI area: <b>{_safe(area_km2)} km²</b>.", body))
    if climate:
        story.append(Paragraph(f"Climate source: {climate.get('source', 'Not available')}. {climate.get('resolution_note', '')}", body))
        story.append(Paragraph(f"Period note: {climate.get('date_range_note', '')}", small))
    story.append(Paragraph("Satellite-derived maps should be interpreted with awareness of spatial resolution, cloud contamination, seasonal comparability, georeferencing, and classification uncertainty.", body))

    # Vegetation module
    story.append(Paragraph("3. Module 1 — Historical Vegetation Change", h1))
    if not historical:
        story.append(Paragraph("Status: Not available in this session.", body))
    else:
        story.append(_table([
            ["Metric", "Value"],
            ["Baseline / comparison", f"{historical.get('baseline_year')} / {historical.get('comparison_year')}"],
            ["Valid area", f"{_safe(historical.get('valid_area_ha'))} ha"],
            ["Mean baseline NDVI", _safe(historical.get("mean_baseline_ndvi"))],
            ["Mean comparison NDVI", _safe(historical.get("mean_comparison_ndvi"))],
            ["Mean NDVI delta", _safe(historical.get("mean_delta"))],
            ["Decline / gain / stable", f"{_safe(historical.get('decline_area_ha'))} / {_safe(historical.get('gain_area_ha'))} / {_safe(historical.get('stable_area_ha'))} ha"],
        ], [60*mm, 100*mm]))
        _array_map(historical.get("delta"), "NDVI change (comparison minus baseline)", story)
        _array_map(historical.get("class_map"), "Vegetation change classes", story)
        story.append(Paragraph("Interpretation: NDVI represents vegetation greenness and canopy condition. A decrease may be associated with seasonal timing, rainfall deficit, harvest, grazing, fire, land conversion, or sensor/processing differences. NDVI alone cannot identify the cause.", body))

    # Climate
    story.append(Paragraph("4. Module 2 — Climate Stress Evidence", h1))
    if not climate:
        story.append(Paragraph("Status: Not available in this session.", body))
    else:
        story.append(Paragraph(f"Source: {climate.get('source', 'Not available')}. {_safe(climate.get('resolution_note'), '')}", body))
        annual = climate.get("annual_anomalies")
        if not isinstance(annual, pd.DataFrame):
            annual = climate.get("annual")
        if isinstance(annual, pd.DataFrame) and not annual.empty:
            story.append(Paragraph("Annual climate series", h2))
            df = annual.copy()
            # Plot common columns, if present.
            numeric_cols = [c for c in df.columns if c != "year" and pd.api.types.is_numeric_dtype(df[c])]
            if "year" in df.columns and numeric_cols:
                col = next((c for c in numeric_cols if "precip" in c.lower() or "rain" in c.lower()), numeric_cols[0])
                fig, ax = plt.subplots(figsize=(8, 3.6))
                ax.plot(df["year"], df[col], marker="o", linewidth=1.3)
                ax.set_title(f"Annual climate indicator: {col}")
                ax.set_xlabel("Year"); ax.set_ylabel(col); ax.grid(True, alpha=0.25)
                _figure_image(fig, story, 155*mm)
            table_cols = [c for c in ["year"] + numeric_cols[:4] if c in df.columns]
            if table_cols:
                tail = df[table_cols].tail(12)
                rows = [table_cols] + [[_safe(v) for v in row] for row in tail.itertuples(index=False, name=None)]
                story.append(_table(rows, [160*mm/len(table_cols)]*len(table_cols)))
        else:
            story.append(Paragraph("Annual climate table was not available.", body))
        story.append(Paragraph("Climate values represent a reanalysis grid cell near the AOI centroid, not field-level measurements. Rainfall and temperature anomalies require an explicitly selected baseline; short or incomplete periods can bias comparisons.", body))

    # Land-use
    story.append(Paragraph("5. Module 3 — Land-Use Change Forensics", h1))
    if not landuse:
        story.append(Paragraph("Status: Not available in this session.", body))
    else:
        story.append(_table([
            ["Metric", "Value"],
            ["Baseline / comparison", f"{landuse.get('baseline_year')} / {landuse.get('comparison_year')}"],
            ["Valid mapped area", f"{_safe(landuse.get('valid_area_ha'))} ha"],
            ["Analysis note", landuse.get("limitations", "Land-cover outputs are screening proxies.")],
        ], [45*mm, 115*mm]))
        _array_map(landuse.get("change_map"), "Land-use transition map (class codes)", story)
        transitions = landuse.get("transitions")
        if isinstance(transitions, pd.DataFrame) and not transitions.empty:
            story.append(Paragraph("Largest mapped transitions", h2))
            cols = list(transitions.columns)[:5]
            rows = [cols] + [[_safe(v) for v in row] for row in transitions[cols].head(12).itertuples(index=False, name=None)]
            story.append(_table(rows, [160*mm/len(cols)]*len(cols)))
        story.append(Paragraph("Mapped classes are proxy classifications. Bare soil, built-up surfaces, water, and vegetation can be confused by spectral similarity, seasonality, or image quality. Validate important transitions before treating them as confirmed land conversion.", body))

    # Causes
    story.append(Paragraph("6. Possible Causes &amp; Evidence Ranking", h1))
    if not attribution:
        story.append(Paragraph("Cause attribution was not run or is unavailable. Possible causes are therefore not ranked in this report.", body))
    else:
        hypotheses = attribution.get("hypotheses")
        if isinstance(hypotheses, pd.DataFrame) and not hypotheses.empty:
            cols = list(hypotheses.columns)[:6]
            story.append(_table([cols] + [[_safe(v) for v in row] for row in hypotheses[cols].itertuples(index=False, name=None)], [160*mm/len(cols)]*len(cols)))
        for limitation in attribution.get("limitations", []):
            story.append(Paragraph("• " + str(limitation), bullet))
        story.append(Paragraph("Rankings are heuristic evidence scores, not probabilities. Spatial overlap and correlation support hypotheses but do not establish causal attribution.", body))

    # Interpretation
    story.append(Paragraph("7. Integrated Interpretation", h1))
    if not interpretation:
        story.append(Paragraph("Integrated interpretation was not run or is unavailable.", body))
    else:
        indicators = interpretation.get("indicators")
        if isinstance(indicators, pd.DataFrame) and not indicators.empty:
            cols = list(indicators.columns)[:6]
            story.append(_table([cols] + [[_safe(v) for v in row] for row in indicators[cols].itertuples(index=False, name=None)], [160*mm/len(cols)]*len(cols)))
        story.append(Paragraph("Cross-indicator evidence", h2))
        for line in interpretation.get("cross_indicator_evidence", []):
            story.append(Paragraph("• " + str(line), bullet))
        story.append(Paragraph("Baseline context", h2))
        context = interpretation.get("baseline_context", {})
        if context:
            story.append(_table([["Field", "Value"]] + [[str(k), _safe(v)] for k, v in context.items()], [70*mm, 90*mm]))
        for limitation in interpretation.get("limitations", []):
            story.append(Paragraph("• " + str(limitation), bullet))

    # Confidence limitations and recommendations
    story.append(Paragraph("8. Confidence, Limitations &amp; Recommended Actions", h1))
    limitations = [
        "The report is only as complete as the analyses run in the current session; missing modules are not inferred.",
        "Confirm that baseline and comparison imagery use comparable seasons, spatial grids, and processing settings.",
        "Check cloud/shadow masks, valid-pixel coverage, georeferencing, and missing data before interpreting spatial patterns.",
        "NDVI change is a greenness signal, not a direct measurement of vegetation area or proof of a specific cause.",
        "ERA5-Land climate data are gridded reanalysis estimates and may not represent local microclimates.",
        "Land-use classes and transitions are proxies and require independent validation.",
        "Possible causes and evidence scores are screening hypotheses, not causal proof or statistical probabilities.",
    ]
    for item in limitations:
        story.append(Paragraph("• " + item, bullet))
    story.append(Paragraph("Recommended follow-up", h2))
    actions = [
        "Field-check the highest-change and highest-risk locations; collect geotagged photos and observations.",
        "Compare multiple dates from the same season and inspect satellite quality/cloud metadata.",
        "Verify suspected land conversion against higher-resolution imagery, local records, or field surveys.",
        "Cross-check climate anomalies against local rain gauges and soil-moisture observations where available.",
        "Prioritize areas where independent indicators agree, while retaining alternative explanations.",
        "Document thresholds, data sources, dates, coordinate reference systems, and processing versions for reproducibility.",
    ]
    for item in actions:
        story.append(Paragraph("• " + item, bullet))

    story.append(Spacer(1, 5*mm))
    story.append(Paragraph("End of report · Generated from the current analysis session · Research prototype", small))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D7DEE8"))
        canvas.line(18*mm, 14*mm, 192*mm, 14*mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#536273"))
        canvas.drawString(18*mm, 9*mm, "Environmental Forensics & Climate Intelligence")
        canvas.drawRightString(192*mm, 9*mm, f"Page {doc.page}")
        canvas.restoreState()

    output = BytesIO()
    doc = SimpleDocTemplate(output, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                            topMargin=17*mm, bottomMargin=20*mm,
                            title=title or "Environmental Forensic Assessment",
                            author="Environmental Forensics & Climate Intelligence")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output.getvalue()
