"""
generate_tolerability_report.py
───────────────────────────────
Reads scored data from BigQuery `tolerability_table` and generates one
professional PDF report **per molecule** using Gemini for narrative generation.

Only the LATEST row per molecule (by created_at) is used.

The report pulls ALL available fields from BigQuery (discontinuation rates,
AE profile, SoC comparison, justification, scoring breakdown). If those
fields are empty or too short, the script automatically calls Gemini with
Google Search to enrich the data before generating the final narrative.

Report structure (single-molecule, business-facing — max 2 pages):
  - Executive Summary
  - Tolerability Profile Overview
  - Scoring Breakdown & Rationale
  - Strategic Implications
  - Scoring reference table (end of document)

Usage:
    # All molecules — one PDF each
    python generate_tolerability_report.py

    # One specific molecule
    python generate_tolerability_report.py --molecule Semaglutide

    # Two molecules — two separate PDFs
    python generate_tolerability_report.py --molecule "Semaglutide,Tirzepatide"

    # Custom output directory
    python generate_tolerability_report.py --outdir ./reports
"""

import os
import re
import json
import argparse
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)

from google import genai as genai_client
from google.genai import types

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("GEMINI_API_KEY", "")
CREDENTIALS_PATH = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
MODEL = "gemini-2.5-flash"

BQ_PROJECT_ID = os.environ.get("PROJECT_ID", "cognito-prod-394707")
BQ_DATASET_ID = os.environ.get("BQ_DATASET_ID", "cognito_prod_datamart")
BQ_LOCATION = "asia-south1"
BQ_TABLE = "tolerability_table"

GCS_BUCKET = "cognito-gcs"
GCS_BASE_PATH = "Cognito_new/reports"
GCS_FILE_NAME = "Tolerability_Analysis.pdf"

REPORT_TITLE = "Patient Tolerability & Burden Analysis"

# Minimum character count to consider justification field "sufficient"
MIN_JUSTIFICATION_LENGTH = 80

# ── Colors ────────────────────────────────────────────────────────────────────
DARK_BLUE = colors.HexColor("#1F3864")
LIGHT_BLUE_BG = colors.HexColor("#E8EDF3")
ACCENT_BLUE = colors.HexColor("#2E5FA3")
WHITE = colors.white
LIGHT_GRAY = colors.HexColor("#666666")
DIVIDER_COLOR = colors.HexColor("#D0D7E3")

SCORE_COLORS = {
    5: colors.HexColor("#008000"),
    4: colors.HexColor("#4CAF50"),
    3: colors.HexColor("#CC9900"),
    2: colors.HexColor("#E65100"),
    1: colors.HexColor("#CC0000"),
}

SCORE_LABEL = {
    5: "Excellent",
    4: "Good",
    3: "Moderate",
    2: "Poor",
    1: "Very Poor",
}

GUARDRAIL_PASS_COLOR = colors.HexColor("#008000")
GUARDRAIL_FAIL_COLOR = colors.HexColor("#CC0000")


# ── Gemini helpers ────────────────────────────────────────────────────────────

def call_gemini(prompt: str, use_search: bool = False) -> str:
    """Call Gemini with optional Google Search grounding."""
    client = genai_client.Client(api_key=API_KEY)
    config_kwargs = {"temperature": 0.3}
    if use_search:
        config_kwargs["tools"] = [types.Tool(googleSearch=types.GoogleSearch())]
    config = types.GenerateContentConfig(**config_kwargs)
    response = client.models.generate_content(model=MODEL, contents=prompt, config=config)
    return response.text.strip() if response.text else ""


def _extract_json(text: str):
    text = re.sub(r"^```(?:json)?", "", text.strip()).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ── Data loading ──────────────────────────────────────────────────────────────

def _get_credentials():
    if CREDENTIALS_PATH and os.path.exists(CREDENTIALS_PATH):
        return service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)
    return None


def _bq_client():
    credentials = _get_credentials()
    return bigquery.Client(project=BQ_PROJECT_ID, credentials=credentials, location=BQ_LOCATION)


def load_from_bigquery(molecules: list[str] | None = None) -> pd.DataFrame:
    """
    Load the LATEST row per molecule from tolerability_table.
    If `molecules` is provided, only those molecules are fetched.
    """
    client = _bq_client()

    molecule_filter = ""
    if molecules:
        escaped = ", ".join(f"'{m.strip()}'" for m in molecules)
        molecule_filter = f"AND molecule_name IN ({escaped})"

    query = f"""
    WITH ranked AS (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY molecule_name ORDER BY created_at DESC) AS _rn
        FROM `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE}`
        WHERE molecule_name IS NOT NULL {molecule_filter}
    )
    SELECT * EXCEPT(_rn) FROM ranked WHERE _rn = 1
    ORDER BY molecule_name
    """
    print(f"Loading latest tolerability data from {BQ_TABLE}...")
    df = client.query(query).to_dataframe()
    print(f"  Loaded {len(df)} molecule(s).")
    return df


# ── Single-molecule statistics ────────────────────────────────────────────────

def extract_molecule_stats(row: pd.Series) -> dict:
    """Extract ALL relevant fields for a single molecule row."""
    def safe(val, fallback="N/A"):
        return str(val).strip() if pd.notna(val) and str(val).strip() else fallback

    score_raw = row.get("score_numeric")
    score_int = None
    try:
        score_int = int(float(score_raw))
    except (ValueError, TypeError):
        pass

    base_score_raw = row.get("base_score")
    base_score_int = None
    try:
        base_score_int = int(float(base_score_raw))
    except (ValueError, TypeError):
        pass

    soc_adj_raw = row.get("soc_adjustment")
    soc_adj_int = None
    try:
        soc_adj_int = int(float(soc_adj_raw))
    except (ValueError, TypeError):
        pass

    burden_adj_raw = row.get("burden_adjustment")
    burden_adj_int = None
    try:
        burden_adj_int = int(float(burden_adj_raw))
    except (ValueError, TypeError):
        pass

    trials_used_raw = row.get("trials_used")
    trials_used_int = None
    try:
        trials_used_int = int(float(trials_used_raw))
    except (ValueError, TypeError):
        pass

    total_trials_raw = row.get("total_trials_extracted")
    total_trials_int = None
    try:
        total_trials_int = int(float(total_trials_raw))
    except (ValueError, TypeError):
        pass

    return {
        "molecule_name": safe(row.get("molecule_name")),
        "tolerability_score": safe(row.get("tolerability_score")),
        "score_int": score_int,
        "score_label": SCORE_LABEL.get(score_int, "N/A") if score_int else "N/A",
        # Supporting data
        "discontinuation_rate_drug": safe(row.get("discontinuation_rate_drug")),
        "discontinuation_rate_soc": safe(row.get("discontinuation_rate_soc")),
        "soc_source": safe(row.get("soc_source")),
        "difference": safe(row.get("difference")),
        # Side effect profile
        "key_aes": safe(row.get("key_aes")),
        "ae_frequency": safe(row.get("ae_frequency")),
        "ae_severity": safe(row.get("ae_severity")),
        # Comparisons
        "vs_placebo": safe(row.get("vs_placebo")),
        "vs_soc": safe(row.get("vs_soc")),
        # Guardrail
        "guardrail": safe(row.get("guardrail")),
        "guardrail_reason": safe(row.get("guardrail_reason"), ""),
        # Justification (full text)
        "justification": safe(row.get("justification"), ""),
        # Scoring breakdown
        "base_score": base_score_int,
        "soc_adjustment": soc_adj_int,
        "burden_adjustment": burden_adj_int,
        "trials_used": trials_used_int,
        "total_trials_extracted": total_trials_int,
    }


# ── Data enrichment (fetch more info if BQ data is thin) ─────────────────────

def _is_data_sufficient(stats: dict) -> bool:
    """Check whether the BQ justification and AE fields have enough substance."""
    justification_ok = len(stats.get("justification", "")) >= MIN_JUSTIFICATION_LENGTH
    has_aes = len(stats.get("key_aes", "")) > 5 and stats.get("key_aes") != "N/A"
    has_disc_rate = stats.get("discontinuation_rate_drug") not in ("N/A", "")
    return justification_ok and has_aes and has_disc_rate


def enrich_molecule_data(stats: dict) -> dict:
    """If BQ data is insufficient, call Gemini with Google Search to fetch
    additional tolerability information for the molecule."""
    if _is_data_sufficient(stats):
        print(f"  BQ data is sufficient for {stats['molecule_name']} — skipping enrichment.")
        return stats

    print(f"  BQ data is thin for {stats['molecule_name']} — enriching via Gemini + Google Search...")

    prompt = f"""You are a pharmaceutical research analyst. Research the tolerability and safety profile
of the following drug thoroughly using the latest clinical trial data, FDA labels, and medical literature.

Drug: {stats['molecule_name']}
Known Discontinuation Rate (Drug): {stats['discontinuation_rate_drug']}
Known Discontinuation Rate (SoC): {stats['discontinuation_rate_soc']}
Known Key AEs: {stats['key_aes']}

Provide a comprehensive tolerability analysis. Return ONLY a valid JSON object:
{{
    "discontinuation_rate_drug": "X.X% (from which source/trial)",
    "discontinuation_rate_soc": "X.X% (SoC drug name)",
    "key_aes": "Top 5 adverse events with frequencies, e.g. Nausea (44%), Diarrhea (30%)",
    "ae_severity": "Mild/Moderate/Severe/Mixed — overall severity classification",
    "ae_persistence": "Transient/Persistent/Mixed — do AEs diminish over time?",
    "vs_placebo_detail": "2-3 sentences comparing drug tolerability vs placebo from pivotal trials",
    "vs_soc_detail": "2-3 sentences comparing drug tolerability vs standard of care",
    "management_burden": "What additional interventions patients need (dose titration, antiemetics, monitoring)",
    "clinical_trial_sources": "Key trial names/IDs used (e.g. SUSTAIN, PIONEER, SURPASS)",
    "tolerability_narrative": "4-6 sentence comprehensive tolerability summary covering discontinuation, AEs, persistence, and patient burden"
}}"""

    try:
        text = call_gemini(prompt, use_search=True)
        enrichment = _extract_json(text)
        if enrichment:
            enriched = dict(stats)
            # Only fill fields that are currently empty/thin
            if stats.get("discontinuation_rate_drug") in ("N/A", "") and enrichment.get("discontinuation_rate_drug"):
                enriched["discontinuation_rate_drug"] = enrichment["discontinuation_rate_drug"]
            if stats.get("discontinuation_rate_soc") in ("N/A", "") and enrichment.get("discontinuation_rate_soc"):
                enriched["discontinuation_rate_soc"] = enrichment["discontinuation_rate_soc"]
            if (stats.get("key_aes") in ("N/A", "") or len(stats.get("key_aes", "")) < 10) and enrichment.get("key_aes"):
                enriched["key_aes"] = enrichment["key_aes"]
            if stats.get("ae_severity") in ("N/A", "") and enrichment.get("ae_severity"):
                enriched["ae_severity"] = enrichment["ae_severity"]
            if len(stats.get("justification", "")) < MIN_JUSTIFICATION_LENGTH and enrichment.get("tolerability_narrative"):
                enriched["justification"] = enrichment["tolerability_narrative"]

            # Store enrichment extras for the narrative prompt
            enriched["_enrichment"] = enrichment
            print(f"  Enrichment complete for {stats['molecule_name']}")
            return enriched
    except Exception as e:
        print(f"  [WARN] Enrichment failed for {stats['molecule_name']}: {e}")

    return stats


# ── LLM narrative (single molecule) ──────────────────────────────────────────

def generate_tolerability_narrative(stats: dict) -> dict:
    """
    Generate a business-focused tolerability report narrative for Medical Affairs.

    Returns a dict with: key_findings, insights_implications, profile_summary.
    """
    # Build comprehensive data block from all available fields
    data_parts = []
    data_parts.append(f"Molecule: {stats['molecule_name']}")
    data_parts.append(f"Tolerability Score: {stats['tolerability_score']} ({stats['score_label']})")
    data_parts.append(f"Guardrail: {stats['guardrail']}")
    if stats.get("guardrail_reason"):
        data_parts.append(f"Guardrail Reason: {stats['guardrail_reason']}")
    data_parts.append(f"Discontinuation Rate (Drug): {stats['discontinuation_rate_drug']}")
    data_parts.append(f"Discontinuation Rate (SoC): {stats['discontinuation_rate_soc']}")
    data_parts.append(f"Difference vs SoC: {stats['difference']}")
    data_parts.append(f"SoC Source: {stats['soc_source']}")
    data_parts.append(f"Key Adverse Events: {stats['key_aes']}")
    data_parts.append(f"AE Frequency: {stats['ae_frequency']}")
    data_parts.append(f"AE Severity: {stats['ae_severity']}")
    data_parts.append(f"vs Placebo: {stats['vs_placebo']}")
    data_parts.append(f"vs SoC: {stats['vs_soc']}")

    scoring_parts = []
    if stats.get("base_score") is not None:
        scoring_parts.append(f"Base Score: {stats['base_score']}/5")
    if stats.get("soc_adjustment") is not None:
        scoring_parts.append(f"SoC Adjustment: {stats['soc_adjustment']:+d}")
    if stats.get("burden_adjustment") is not None:
        scoring_parts.append(f"Burden Adjustment: {stats['burden_adjustment']:+d}")
    if stats.get("trials_used") is not None:
        scoring_parts.append(f"Trials Used for Scoring: {stats['trials_used']}")
    if stats.get("total_trials_extracted") is not None:
        scoring_parts.append(f"Total Trials Extracted: {stats['total_trials_extracted']}")

    justification_block = stats.get("justification", "")

    # Include enrichment data if available
    enrichment = stats.get("_enrichment", {})
    enrichment_block = ""
    if enrichment:
        enrich_parts = []
        for k in ["vs_placebo_detail", "vs_soc_detail", "management_burden", "clinical_trial_sources", "tolerability_narrative", "ae_persistence"]:
            val = enrichment.get(k, "")
            if val:
                enrich_parts.append(f"{k.replace('_', ' ').title()}: {val}")
        if enrich_parts:
            enrichment_block = "\n".join(enrich_parts)

    prompt = f"""You are a business-focused medical insights analyst.

Goal: Create a concise, 2-page tolerability report for {stats['molecule_name']} highlighting
key findings and insights derived from the provided data outputs. The report is intended for
a Medical Affairs business audience.

Context: The data comes from structured outputs containing tolerability-related information
(discontinuation rates, adverse events, severity, patient burden, comparison with standard
of care, etc.). The audience is non-technical and not familiar with internal analytical
frameworks, scoring methodologies, or internal jargon.

Source: Use only the provided data outputs as the source of truth. Focus specifically on
tolerability-related data points. Do not introduce external assumptions unless clearly
derived from the data.

=== PROVIDED DATA ===
{chr(10).join(data_parts)}

SCORING CONTEXT (for your reference only — do NOT expose methodology):
{chr(10).join(scoring_parts) if scoring_parts else 'Not available'}

DETAILED ANALYSIS:
{justification_block if justification_block else 'Not available'}

ADDITIONAL RESEARCH DATA:
{enrichment_block if enrichment_block else 'Not available'}

=== INSTRUCTIONS ===

1. Start with a section: "Key Tolerability Findings for {stats['molecule_name']}"
   - Summarize the most important observations related to tolerability
   - Include the overall tolerability score (e.g., "scored X out of 5") in simple terms
     early in the findings — but do NOT explain how the score was derived
   - Highlight:
     - Discontinuation rates in comparison to placebo and standard of care
     - Key adverse events observed
     - Frequency and severity of side effects
     - Any notable differences versus comparator treatments
   - Focus on what the data shows, not how it was calculated

2. Follow with: "Insights and Implications"
   - Translate tolerability findings into business-relevant insights
   - Highlight:
     - Overall patient experience and burden
     - Key drivers of discontinuation (if identifiable)
     - Strengths or concerns compared to standard treatments
     - Potential impact on adoption, adherence, or positioning
   - Keep insights simple, clear, and actionable

3. Include: "Tolerability Profile Summary"
   - Provide a high-level summary of how well the molecule is tolerated overall
   - Comment on consistency of tolerability across studies if available
   - Briefly mention the overall tolerability score in simple terms without explaining
     scoring methodology

Language and Style:
- Do NOT use internal jargon, scoring framework names, or technical modeling terms
- Avoid methodological explanations (e.g., how score was derived, base scores, adjustments)
- Use clear, simple, business-friendly language
- Translate clinical findings into plain-language impact (e.g., what side effects mean
  for patients and treatment continuation)

Tone: Professional, objective, and insight-driven. Focus on clarity, relevance, and
business impact.

Respond ONLY with a valid JSON object (no markdown fences, no extra text):
{{
  "key_findings": {{
    "summary_bullets": [
      "Key finding 1 — MUST state the tolerability score as X out of 5 (use the exact score from the data) and what it indicates for patient tolerability in plain terms",
      "Key finding 2 about discontinuation rates vs placebo and standard of care",
      "Key finding 3 about severity of side effects and patient experience",
      "Key finding 4 about notable differences vs comparator treatments",
      "Key finding 5 about any other important tolerability observations"
    ],
    "discontinuation_detail": "2-3 sentences in plain language about what the discontinuation rates mean — how many patients stopped treatment due to side effects compared to placebo and standard of care",
    "adverse_event_detail": "2-3 sentences describing the most common side effects, how often they occur, and whether they are mild/moderate/severe",
    "comparator_detail": "2-3 sentences on how this drug's tolerability compares to existing treatments in plain business terms"
  }},
  "insights_implications": {{
    "patient_experience": "2-3 sentences on overall patient experience — what does the side effect profile mean for patients day-to-day",
    "discontinuation_drivers": "2-3 sentences on what appears to drive patients to stop treatment, if identifiable from the data",
    "strengths_concerns": "2-3 sentences on key tolerability strengths or red flags compared to standard treatments",
    "adoption_impact": "2-3 sentences on how this tolerability profile could affect real-world adoption, patient adherence, and market positioning"
  }},
  "profile_summary": {{
    "overall_assessment": "3-4 sentences providing a high-level summary of how well tolerated this molecule is overall, written for a business audience",
    "cross_study_consistency": "1-2 sentences commenting on whether tolerability findings are consistent across available studies",
    "score_context": "1-2 sentences stating the tolerability score explicitly (e.g., 'The molecule scored 3 out of 5 on tolerability, indicating a moderate tolerability profile'). ALWAYS include the numeric score (X out of 5). Do NOT explain how the score was derived."
  }}
}}"""

    text = call_gemini(prompt)
    result = _extract_json(text)
    if result:
        return result

    # Fallback
    return {
        "key_findings": {
            "summary_bullets": [
                f"{stats['molecule_name']} received a tolerability score of {stats['tolerability_score']} ({stats['score_label']}), reflecting its overall side-effect profile.",
                f"Discontinuation rate (drug): {stats['discontinuation_rate_drug']}.",
                f"Key adverse events reported: {stats['key_aes']}.",
                f"AE severity: {stats['ae_severity']}. vs Placebo: {stats['vs_placebo']}.",
                f"Guardrail status: {stats['guardrail']}.",
            ],
            "discontinuation_detail": stats.get("justification") or "See detailed analysis.",
            "adverse_event_detail": f"Key AEs: {stats['key_aes']}. Severity: {stats['ae_severity']}.",
            "comparator_detail": f"vs SoC: {stats['vs_soc']}. vs Placebo: {stats['vs_placebo']}.",
        },
        "insights_implications": {
            "patient_experience": "See detailed analysis.",
            "discontinuation_drivers": "See detailed analysis.",
            "strengths_concerns": "See detailed analysis.",
            "adoption_impact": "See detailed analysis.",
        },
        "profile_summary": {
            "overall_assessment": f"{stats['molecule_name']} received a tolerability score of {stats['tolerability_score']} ({stats['score_label']}).",
            "cross_study_consistency": "Data consistency is based on available clinical trial evidence.",
            "score_context": f"The molecule scored {stats['tolerability_score']} on tolerability, indicating a {stats['score_label'].lower()} tolerability profile.",
        },
    }


# ── Style helpers ─────────────────────────────────────────────────────────────

def build_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Normal"],
            fontSize=20, leading=26, textColor=DARK_BLUE,
            alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=2,
        ),
        "molecule_name_title": ParagraphStyle(
            "MoleculeNameTitle", parent=base["Normal"],
            fontSize=14, leading=18, textColor=ACCENT_BLUE,
            alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle", parent=base["Normal"],
            fontSize=9, leading=12, textColor=LIGHT_GRAY,
            alignment=TA_CENTER, fontName="Helvetica", spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Normal"],
            fontSize=13, leading=16, textColor=DARK_BLUE,
            fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "H3", parent=base["Normal"],
            fontSize=11, leading=14, textColor=DARK_BLUE,
            fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["Normal"],
            fontSize=10, leading=14, textColor=colors.HexColor("#333333"),
            fontName="Helvetica", spaceAfter=4, alignment=TA_JUSTIFY,
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=base["Normal"],
            fontSize=9, leading=13, textColor=colors.HexColor("#333333"),
            fontName="Helvetica", spaceAfter=3, leftIndent=18,
            bulletIndent=6,
        ),
        "methodology_note": ParagraphStyle(
            "MethodologyNote", parent=base["Normal"],
            fontSize=8, leading=12, textColor=colors.HexColor("#333333"),
            fontName="Helvetica", spaceAfter=4, alignment=TA_JUSTIFY,
        ),
        "footer": ParagraphStyle(
            "Footer", parent=base["Normal"],
            fontSize=7, leading=10, textColor=colors.HexColor("#999999"),
            fontName="Helvetica", alignment=TA_CENTER, spaceBefore=10,
        ),
        "cell": ParagraphStyle(
            "Cell", parent=base["Normal"],
            fontSize=8, leading=11, textColor=colors.HexColor("#333333"),
            fontName="Helvetica", alignment=TA_CENTER,
        ),
        "cell_header": ParagraphStyle(
            "CellHeader", parent=base["Normal"],
            fontSize=8, leading=11, textColor=WHITE,
            fontName="Helvetica-Bold", alignment=TA_CENTER,
        ),
        "section_label": ParagraphStyle(
            "SectionLabel", parent=base["Normal"],
            fontSize=10, leading=13, textColor=DARK_BLUE,
            fontName="Helvetica-Bold", spaceAfter=2, leftIndent=6,
        ),
    }


def _score_color(score_val):
    try:
        return SCORE_COLORS.get(int(float(score_val)), colors.black)
    except (ValueError, TypeError):
        return colors.black


def _render_bullets(items: list, styles: dict, story: list):
    """Render a list of strings as bullet points."""
    for item in items:
        if item and str(item).strip():
            story.append(Paragraph(
                f"&#8226; {item}",
                styles["bullet"],
            ))


def _scoring_framework_table(styles: dict) -> Table:
    """Render the Tolerability scoring reference table."""
    framework = [
        ("5", "Excellent", "Discontinuation rate at or below placebo; mild, transient AEs; no management needed"),
        ("4", "Good", "Discontinuation rate above placebo but below 5%; better than SoC"),
        ("3", "Moderate", "Discontinuation rate 5-10%; similar to SoC; manageable side effects"),
        ("2", "Poor", "Discontinuation rate 10-20%; worse than SoC; persistent or moderate AEs"),
        ("1", "Very Poor", "Discontinuation rate 20%+; severe or management-requiring AEs"),
    ]
    header = [
        Paragraph("Score", styles["cell_header"]),
        Paragraph("Label", styles["cell_header"]),
        Paragraph("Description", styles["cell_header"]),
    ]
    rows = [header]
    for sc, lbl, desc in framework:
        rows.append([
            Paragraph(sc,   ParagraphStyle("FWScore", parent=styles["cell"], fontName="Helvetica")),
            Paragraph(lbl,  ParagraphStyle("FWLabel", parent=styles["cell"], fontName="Helvetica")),
            Paragraph(desc, ParagraphStyle("FWDesc",  parent=styles["cell"], alignment=TA_LEFT)),
        ])

    tbl = Table(rows, colWidths=[0.5 * inch, 1.2 * inch, 5.0 * inch])
    row_bgs = [
        ("BACKGROUND", (0, i), (-1, i), LIGHT_BLUE_BG if i % 2 == 0 else WHITE)
        for i in range(1, len(rows))
    ]
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",      (0, 0), (1, -1), "CENTER"),
        ("ALIGN",      (2, 1), (2, -1), "LEFT"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        *row_bgs,
    ]))
    return tbl


# ── Single-molecule report builder ────────────────────────────────────────────

def build_single_molecule_report(stats: dict, narrative: dict, output_path: str):
    """Build and save a business-focused PDF report for one molecule."""
    styles = build_styles()
    story = []

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        title=f"{REPORT_TITLE} — {stats['molecule_name']}",
        author="Tolerability Scorer",
    )

    # ── Title block ───────────────────────────────────────────────────────────
    story.append(Paragraph(REPORT_TITLE, styles["title"]))
    story.append(Paragraph(stats["molecule_name"], styles["molecule_name_title"]))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%B %d, %Y')}  •  Tolerability Score: {stats['tolerability_score']} ({stats['score_label']})",
        styles["subtitle"],
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=DARK_BLUE, spaceAfter=12))

    # ══════════════════════════════════════════════════════════════════════
    # Key Tolerability Findings
    # ══════════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
    story.append(Paragraph(
        f"Key Tolerability Findings for {stats['molecule_name']}", styles["h2"]
    ))

    key_findings = narrative.get("key_findings", {})
    if isinstance(key_findings, dict):
        # Bullet summary
        bullets = key_findings.get("summary_bullets", [])
        if isinstance(bullets, list):
            _render_bullets(bullets, styles, story)
        story.append(Spacer(1, 6))

        disc_detail = key_findings.get("discontinuation_detail", "")
        if disc_detail:
            story.append(Paragraph("<b>Discontinuation Rates:</b>", styles["section_label"]))
            story.append(Paragraph(disc_detail, styles["body"]))
            story.append(Spacer(1, 4))

        ae_detail = key_findings.get("adverse_event_detail", "")
        if ae_detail:
            story.append(Paragraph("<b>Adverse Events:</b>", styles["section_label"]))
            story.append(Paragraph(ae_detail, styles["body"]))
            story.append(Spacer(1, 4))

        comparator_detail = key_findings.get("comparator_detail", "")
        if comparator_detail:
            story.append(Paragraph("<b>Comparator Analysis:</b>", styles["section_label"]))
            story.append(Paragraph(comparator_detail, styles["body"]))
    elif isinstance(key_findings, str):
        story.append(Paragraph(key_findings, styles["body"]))
    story.append(Spacer(1, 6))

    # ══════════════════════════════════════════════════════════════════════
    # Insights and Implications
    # ══════════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
    story.append(Paragraph("Insights and Implications", styles["h2"]))

    insights = narrative.get("insights_implications", {})
    if isinstance(insights, dict):
        patient_exp = insights.get("patient_experience", "")
        if patient_exp:
            story.append(Paragraph(f"<b>Patient Experience:</b> {patient_exp}", styles["body"]))
            story.append(Spacer(1, 4))

        disc_drivers = insights.get("discontinuation_drivers", "")
        if disc_drivers:
            story.append(Paragraph(f"<b>Drivers of Discontinuation:</b> {disc_drivers}", styles["body"]))
            story.append(Spacer(1, 4))

        strengths = insights.get("strengths_concerns", "")
        if strengths:
            story.append(Paragraph(f"<b>Strengths &amp; Concerns:</b> {strengths}", styles["body"]))
            story.append(Spacer(1, 4))

        adoption = insights.get("adoption_impact", "")
        if adoption:
            story.append(Paragraph(f"<b>Impact on Adoption &amp; Positioning:</b> {adoption}", styles["body"]))
    elif isinstance(insights, str):
        story.append(Paragraph(insights, styles["body"]))
    story.append(Spacer(1, 6))

    # ══════════════════════════════════════════════════════════════════════
    # Tolerability Profile Summary
    # ══════════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
    story.append(Paragraph("Tolerability Profile Summary", styles["h2"]))

    profile_summary = narrative.get("profile_summary", {})
    if isinstance(profile_summary, dict):
        overall = profile_summary.get("overall_assessment", "")
        if overall:
            story.append(Paragraph(overall, styles["body"]))
            story.append(Spacer(1, 4))

        consistency = profile_summary.get("cross_study_consistency", "")
        if consistency:
            story.append(Paragraph(consistency, styles["body"]))
            story.append(Spacer(1, 4))

        score_ctx = profile_summary.get("score_context", "")
        if score_ctx:
            story.append(Paragraph(score_ctx, styles["body"]))
    elif isinstance(profile_summary, str):
        story.append(Paragraph(profile_summary, styles["body"]))
    story.append(Spacer(1, 8))

    # ── Scoring reference table ───────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
    story.append(Paragraph("Tolerability Scoring Reference", styles["h2"]))
    story.append(_scoring_framework_table(styles))
    story.append(Spacer(1, 8))

    legend_text = "  |  ".join(f"{k} = {v}" for k, v in SCORE_LABEL.items())
    story.append(Paragraph(
        f"<b>Score Legend:</b>  {legend_text}",
        ParagraphStyle("Legend", parent=styles["body"], fontSize=8, textColor=LIGHT_GRAY),
    ))

    # ── Footer ────────────────────────────────────────────────────────────
    story.append(HRFlowable(
        width="100%", thickness=0.5,
        color=colors.HexColor("#CCCCCC"), spaceBefore=14,
    ))
    story.append(Paragraph(
        "This report was auto-generated from Tolerability analysis output. "
        "For internal use only.",
        styles["footer"],
    ))

    doc.build(story)
    print(f"  Report saved -> {output_path}")


# ── GCS Upload ────────────────────────────────────────────────────────────────

def upload_to_gcs(local_path: str, molecule_name: str) -> str:
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError("google-cloud-storage required. pip install google-cloud-storage")

    credentials = _get_credentials()
    client = storage.Client(project=BQ_PROJECT_ID, credentials=credentials)
    bucket = client.bucket(GCS_BUCKET)

    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(molecule_name))
    blob_name = f"{GCS_BASE_PATH}/{safe_name}/{GCS_FILE_NAME}"
    gcs_uri = f"gs://{GCS_BUCKET}/{blob_name}"

    print(f"  Uploading to GCS: {gcs_uri}")
    bucket.blob(blob_name).upload_from_filename(local_path, content_type="application/pdf")
    return gcs_uri


# ── Public entry point ────────────────────────────────────────────────────────

def generate_tolerability_report(
    molecules: list[str] | None = None,
    outdir: str | None = None,
) -> list[str]:
    """
    Generate one PDF report per molecule.

    Args:
        molecules: List of molecule names to report on. None = all in table.
        outdir:    Directory to write PDFs into. Defaults to current directory.

    Returns:
        List of output PDF paths that were successfully created.
    """
    if not API_KEY:
        print("[Tolerability Report] GEMINI_API_KEY not set — skipping report generation.")
        return []

    out_root = Path(outdir) if outdir else Path(".")
    out_root.mkdir(parents=True, exist_ok=True)

    df = load_from_bigquery(molecules)
    if df.empty:
        print("[Tolerability Report] No data found — skipping.")
        return []

    output_paths = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for _, row in df.iterrows():
        stats = extract_molecule_stats(row)
        molecule_name = stats["molecule_name"]

        print(f"\nProcessing: {molecule_name}")

        # Enrich data if BQ fields are insufficient
        stats = enrich_molecule_data(stats)

        print("  Generating narrative with Gemini...")
        narrative = generate_tolerability_narrative(stats)

        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", molecule_name)
        output_path = str(out_root / f"tolerability_report_{safe_name}_{ts}.pdf")

        build_single_molecule_report(stats, narrative, output_path)
        output_paths.append(output_path)

        try:
            gcs_uri = upload_to_gcs(output_path, molecule_name)
            print(f"  GCS: {gcs_uri}")
        except Exception as e:
            print(f"  [WARN] GCS upload failed for '{molecule_name}': {e}")

    print(f"\nDone. {len(output_paths)} report(s) generated.")
    return output_paths


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate one Tolerability PDF per molecule from BigQuery data."
    )
    parser.add_argument(
        "--molecule", "-m",
        default=None,
        help=(
            "Comma-separated molecule name(s) to report on. "
            "E.g. --molecule Semaglutide  or  --molecule 'Semaglutide,Tirzepatide'. "
            "Omit to process all molecules in the table."
        ),
    )
    parser.add_argument(
        "--outdir", "-o",
        default=None,
        help="Output directory for generated PDFs (default: current directory).",
    )
    args = parser.parse_args()

    molecules = None
    if args.molecule:
        molecules = [m.strip() for m in args.molecule.split(",") if m.strip()]

    generate_tolerability_report(molecules=molecules, outdir=args.outdir)


if __name__ == "__main__":
    main()
