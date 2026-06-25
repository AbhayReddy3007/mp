"""
generate_tolerability_report.py
───────────────────────────────
Reads scored data from BigQuery `tolerability_table` and generates one
professional PDF report **per molecule** using Gemini for narrative generation.

Only the LATEST row per molecule (by created_at) is used.

The report pulls ALL available fields from BigQuery — discontinuation rates,
AE profiles, scoring breakdown, justification, etc.  If those fields are
empty or too short, the script calls Gemini with Google Search to enrich
the data before generating the final narrative.

Report structure (single-molecule, business-facing — max 2 pages):
  - Executive Summary
  - Tolerability Profile Overview
  - Categorization & Rationale
  - Scoring & Implications
  - Scoring reference table

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

# Minimum character count to consider the justification field "sufficient"
MIN_JUSTIFICATION_LENGTH = 100

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
    Returns one row per molecule, sorted by molecule_name.
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

    base_score = None
    try:
        base_score = int(float(row.get("base_score", 0)))
    except (ValueError, TypeError):
        pass

    soc_adj = None
    try:
        soc_adj = int(float(row.get("soc_adjustment", 0)))
    except (ValueError, TypeError):
        pass

    burden_adj = None
    try:
        burden_adj = int(float(row.get("burden_adjustment", 0)))
    except (ValueError, TypeError):
        pass

    trials_used = None
    try:
        trials_used = int(float(row.get("trials_used", 0)))
    except (ValueError, TypeError):
        pass

    total_trials = None
    try:
        total_trials = int(float(row.get("total_trials_extracted", 0)))
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
        # Side effects
        "key_aes": safe(row.get("key_aes")),
        "ae_frequency": safe(row.get("ae_frequency")),
        "ae_severity": safe(row.get("ae_severity")),
        # Comparisons
        "vs_placebo": safe(row.get("vs_placebo")),
        "vs_soc": safe(row.get("vs_soc")),
        # Guardrail
        "guardrail": safe(row.get("guardrail")),
        "guardrail_reason": safe(row.get("guardrail_reason"), ""),
        # Justification
        "justification": safe(row.get("justification"), ""),
        # Scoring breakdown
        "base_score": base_score,
        "soc_adjustment": soc_adj,
        "burden_adjustment": burden_adj,
        "trials_used": trials_used,
        "total_trials_extracted": total_trials,
    }


# ── Data enrichment ───────────────────────────────────────────────────────────

def _is_data_sufficient(stats: dict) -> bool:
    """Check whether the BQ justification has enough substance."""
    justification = stats.get("justification", "")
    has_justification = len(justification) >= MIN_JUSTIFICATION_LENGTH
    has_disc_rate = stats.get("discontinuation_rate_drug", "N/A") != "N/A"
    has_aes = stats.get("key_aes", "N/A") != "N/A"
    return has_justification and has_disc_rate and has_aes


def enrich_molecule_data(stats: dict) -> dict:
    """If BQ data is insufficient, call Gemini with Google Search to fetch
    additional tolerability info. Returns an enriched copy."""
    if _is_data_sufficient(stats):
        print(f"  BQ data is sufficient for {stats['molecule_name']} — skipping enrichment.")
        return stats

    print(f"  BQ data is thin for {stats['molecule_name']} — enriching via Gemini + Google Search...")

    prompt = f"""You are a pharmaceutical clinical safety analyst. Research the tolerability
and safety profile of the following drug using the latest clinical trial data,
FDA labels, and published literature.

Drug: {stats['molecule_name']}

Provide a comprehensive tolerability analysis. Return ONLY a valid JSON object:
{{
    "discontinuation_rate_drug": "X.X% (from Phase 3 trials)",
    "discontinuation_rate_placebo": "X.X%",
    "soc_drug": "Name of the standard of care comparator drug in the same class",
    "discontinuation_rate_soc": "X.X% (standard of care drug)",
    "key_adverse_events": "Nausea (X%), Vomiting (X%), Diarrhea (X%), etc.",
    "ae_severity": "Mild/Moderate/Severe/Mixed",
    "ae_persistence": "Transient/Persistent/Mixed",
    "management_required": "None/Occasional/Regular",
    "tolerability_summary": "4-6 sentence summary of overall tolerability profile, including how AEs typically evolve over time and impact on patient compliance",
    "comparison_vs_class": "3-4 sentences comparing tolerability to other drugs in the same class",
    "clinical_significance": "2-3 sentences on clinical significance of the tolerability profile for prescribing decisions",
    "sources": ["List of key clinical trials and publications used"]
}}"""

    try:
        text = call_gemini(prompt, use_search=True)
        enrichment = _extract_json(text)
        if enrichment:
            enriched = dict(stats)
            # Fill empty fields
            if enriched.get("discontinuation_rate_drug", "N/A") == "N/A":
                enriched["discontinuation_rate_drug"] = enrichment.get("discontinuation_rate_drug", "N/A")
            if enriched.get("discontinuation_rate_soc", "N/A") == "N/A":
                soc_name = enrichment.get("soc_drug", "")
                soc_rate = enrichment.get("discontinuation_rate_soc", "")
                if soc_name and soc_rate:
                    enriched["discontinuation_rate_soc"] = f"{soc_rate} ({soc_name})"
            if enriched.get("key_aes", "N/A") == "N/A":
                enriched["key_aes"] = enrichment.get("key_adverse_events", "N/A")
            if enriched.get("ae_frequency", "N/A") == "N/A":
                enriched["ae_frequency"] = enrichment.get("key_adverse_events", "N/A")
            if enriched.get("ae_severity", "N/A") == "N/A":
                enriched["ae_severity"] = enrichment.get("ae_severity", "N/A")

            # Build/supplement justification
            existing_just = enriched.get("justification", "")
            new_parts = []
            for key in ["tolerability_summary", "comparison_vs_class", "clinical_significance"]:
                val = enrichment.get(key, "")
                if val and val not in existing_just:
                    new_parts.append(val)
            if new_parts:
                enriched["justification"] = (existing_just + "\n\n" + "\n\n".join(new_parts)).strip()

            enriched["_enrichment_sources"] = enrichment.get("sources", [])
            print(f"  ✅ Enrichment complete for {stats['molecule_name']}")
            return enriched
    except Exception as e:
        print(f"  [WARN] Enrichment failed for {stats['molecule_name']}: {e}")

    return stats


# ── LLM narrative ─────────────────────────────────────────────────────────────

def generate_tolerability_narrative(stats: dict) -> dict:
    """
    Generate a structured, detailed report narrative for a single molecule's
    tolerability profile.

    Returns a dict with: executive_summary, tolerability_overview,
    categorization_rationale, scoring_implications, score_methodology_note.
    """
    # Build scoring breakdown description
    scoring_breakdown = ""
    if stats.get("base_score") is not None:
        scoring_breakdown = f"""
Scoring Breakdown:
- Base Score: {stats['base_score']}/5 (from discontinuation rate vs placebo)
- SoC Adjustment: {stats['soc_adjustment']:+d} (comparison vs standard of care)
- Burden Adjustment: {stats['burden_adjustment']:+d} (patient burden from AE severity/persistence)
- Final Score: {stats['base_score']} + ({stats['soc_adjustment']}) + ({stats['burden_adjustment']}) = {stats['score_int']}/5"""

    prompt = f"""You are an expert pharmaceutical analyst specializing in patient safety and
tolerability assessment. Your task is to generate a concise, structured report
(maximum 2 pages) evaluating the Patient Tolerability & Burden of a given product.

INPUT:
- Product Name: {stats['molecule_name']}
- Tolerability Score: {stats['tolerability_score']} ({stats['score_label']})
- Discontinuation Rate (Drug): {stats['discontinuation_rate_drug']}
- Discontinuation Rate (SoC): {stats['discontinuation_rate_soc']}
- Difference vs SoC: {stats['difference']}
- Key Adverse Events: {stats['key_aes']}
- AE Frequency: {stats['ae_frequency']}
- AE Severity: {stats['ae_severity']}
- Comparison vs Placebo: {stats['vs_placebo']}
- Comparison vs SoC: {stats['vs_soc']}
- Guardrail: {stats['guardrail']}
- Guardrail Reason: {stats.get('guardrail_reason', 'N/A')}
- Trials Used: {stats.get('trials_used', 'N/A')}
- Total Trials Extracted: {stats.get('total_trials_extracted', 'N/A')}
{scoring_breakdown}

DETAILED JUSTIFICATION FROM ANALYSIS:
{stats.get('justification', 'Limited information available.')}

OUTPUT REQUIREMENTS — Generate a structured report with these exact sections:

EXECUTIVE SUMMARY
- Provide a crisp overview (6-8 bullet points) covering:
  - The tolerability profile of the product
  - The assigned score and what it means
  - Key adverse events and their clinical significance
  - How the drug compares to standard of care
  - Guardrail status and any flags
- Keep this section concise and decision-oriented

TOLERABILITY PROFILE OVERVIEW
- Describe the overall tolerability and side-effect profile
- Name the key adverse events with their frequency and severity
- Explain whether AEs are transient or persistent
- Describe patient burden (need for dose titration, antiemetics, monitoring)
- Reference the discontinuation rate and what drives it

CATEGORIZATION & RATIONALE
- Classify the tolerability: Excellent / Good / Moderate / Poor / Very Poor
- Justify the classification with specific data
- Compare against standard of care and placebo
- Reference the scoring breakdown (base score, adjustments)

SCORING & IMPLICATIONS
- Provide the tolerability score and explain why it was assigned
- Explain what the score implies for:
  - Patient compliance and adherence
  - Prescriber confidence
  - Competitive positioning vs other drugs in class
  - Risk of label warnings or REMS requirements
- If guardrail failed, explain implications clearly

FORMATTING INSTRUCTIONS:
- Use bullet points wherever possible
- Keep paragraphs short (2-4 lines max)
- Avoid repetition
- Use precise, evidence-based reasoning
- Professional tone suitable for leadership review

Respond ONLY with a valid JSON object (no markdown fences, no extra text):
{{
  "executive_summary": [
    "Bullet point 1 about the drug's overall tolerability profile",
    "Bullet point 2 about the score and classification",
    "Bullet point 3 about key adverse events",
    "Bullet point 4 about discontinuation rates and comparison to SoC",
    "Bullet point 5 about guardrail status",
    "Bullet point 6 about clinical/strategic implications"
  ],
  "tolerability_overview": {{
    "profile_description": "3-5 sentence description of overall tolerability and side-effect profile",
    "key_adverse_events": "2-3 sentences detailing the most common AEs with frequency, severity, and persistence",
    "patient_burden": "2-3 sentences describing patient burden — dose titration needs, supportive medication, monitoring requirements"
  }},
  "categorization_rationale": {{
    "category": "Excellent/Good/Moderate/Poor/Very Poor",
    "rationale_points": [
      "Justification point 1 with data reference",
      "Justification point 2 comparing to SoC",
      "Justification point 3 about scoring breakdown"
    ]
  }},
  "scoring_implications": {{
    "score_explanation": "2-3 sentences explaining why this score was assigned, referencing the base score and adjustments",
    "patient_compliance": "2-3 sentences on impact on patient adherence and real-world persistence",
    "competitive_positioning": "2-3 sentences on competitive tolerability positioning vs class",
    "risk_considerations": "1-2 sentences on any safety/regulatory risk flags (especially if guardrail failed)"
  }},
  "score_methodology_note": "2-3 plain-language sentences explaining how the tolerability score is determined"
}}"""

    text = call_gemini(prompt)
    result = _extract_json(text)
    if result:
        return result

    # Fallback
    return {
        "executive_summary": [
            f"{stats['molecule_name']} scored {stats['tolerability_score']} ({stats['score_label']}) on tolerability.",
            f"Discontinuation rate (drug arm): {stats['discontinuation_rate_drug']}.",
            f"Key adverse events: {stats['key_aes']}.",
            f"Guardrail: {stats['guardrail']}.",
        ],
        "tolerability_overview": {
            "profile_description": stats.get("justification", "See detailed analysis."),
            "key_adverse_events": f"Key AEs: {stats['key_aes']}. Severity: {stats['ae_severity']}.",
            "patient_burden": "Refer to detailed analysis.",
        },
        "categorization_rationale": {
            "category": stats["score_label"],
            "rationale_points": [
                f"Discontinuation (drug): {stats['discontinuation_rate_drug']} vs SoC: {stats['discontinuation_rate_soc']}.",
            ],
        },
        "scoring_implications": {
            "score_explanation": f"Score: {stats['tolerability_score']}.",
            "patient_compliance": "Refer to detailed analysis.",
            "competitive_positioning": f"vs SoC: {stats['vs_soc']}.",
            "risk_considerations": stats.get("guardrail_reason") or "No guardrail issues.",
        },
        "score_methodology_note": (
            "Tolerability is scored 1-5 based on discontinuation rate vs placebo, "
            "adjusted for SoC comparison and patient burden."
        ),
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
    for item in items:
        if item and str(item).strip():
            story.append(Paragraph(f"&#8226; {item}", styles["bullet"]))


def _scoring_framework_table(styles: dict) -> Table:
    """Render the tolerability scoring reference table."""
    framework = [
        ("5", "Excellent", "Discontinuation ≤ placebo; mild, transient AEs; no management needed"),
        ("4", "Good",      "Discontinuation < 5% and > placebo; well-tolerated overall"),
        ("3", "Moderate",  "Discontinuation 5-10%; manageable AEs; some dose titration needed"),
        ("2", "Poor",      "Discontinuation 10-20%; persistent or moderate AEs requiring management"),
        ("1", "Very Poor", "Discontinuation ≥ 20%; severe AE burden; regular intervention needed"),
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


# ── Single-molecule report builder ───────────────────────────────────────────

def build_single_molecule_report(stats: dict, narrative: dict, output_path: str):
    """Build and save a detailed PDF report for one molecule."""
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

    # ── Title block ──────────────────────────────────────────────────────
    story.append(Paragraph(REPORT_TITLE, styles["title"]))
    story.append(Paragraph(stats["molecule_name"], styles["molecule_name_title"]))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%B %d, %Y')}",
        styles["subtitle"],
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=DARK_BLUE, spaceAfter=12))

    # ══════════════════════════════════════════════════════════════════════
    # Executive Summary
    # ══════════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
    story.append(Paragraph("Executive Summary", styles["h2"]))

    exec_summary = narrative.get("executive_summary", [])
    if isinstance(exec_summary, list):
        _render_bullets(exec_summary, styles, story)
    elif isinstance(exec_summary, str):
        story.append(Paragraph(exec_summary, styles["body"]))
    story.append(Spacer(1, 6))

    # ══════════════════════════════════════════════════════════════════════
    # Tolerability Profile Overview
    # ══════════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
    story.append(Paragraph("Tolerability Profile Overview", styles["h2"]))

    tol_overview = narrative.get("tolerability_overview", {})
    if isinstance(tol_overview, dict):
        profile_desc = tol_overview.get("profile_description", "")
        if profile_desc:
            story.append(Paragraph(profile_desc, styles["body"]))
            story.append(Spacer(1, 4))

        key_aes = tol_overview.get("key_adverse_events", "")
        if key_aes:
            story.append(Paragraph("<b>Key Adverse Events:</b>", styles["section_label"]))
            story.append(Paragraph(key_aes, styles["body"]))
            story.append(Spacer(1, 4))

        patient_burden = tol_overview.get("patient_burden", "")
        if patient_burden:
            story.append(Paragraph("<b>Patient Burden:</b>", styles["section_label"]))
            story.append(Paragraph(patient_burden, styles["body"]))
    elif isinstance(tol_overview, str):
        story.append(Paragraph(tol_overview, styles["body"]))
    story.append(Spacer(1, 6))

    # ══════════════════════════════════════════════════════════════════════
    # Categorization & Rationale
    # ══════════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
    story.append(Paragraph("Categorization &amp; Rationale", styles["h2"]))

    cat_rationale = narrative.get("categorization_rationale", {})
    if isinstance(cat_rationale, dict):
        category = cat_rationale.get("category", stats["score_label"])
        score_color = _score_color(stats.get("score_int"))
        cat_style = ParagraphStyle(
            "CatValue", parent=styles["body"],
            textColor=score_color, fontName="Helvetica-Bold", fontSize=11,
        )
        story.append(Paragraph(
            f"Classification: {category} — Score {stats['tolerability_score']}",
            cat_style,
        ))
        story.append(Spacer(1, 4))

        rationale_points = cat_rationale.get("rationale_points", [])
        if rationale_points:
            _render_bullets(rationale_points, styles, story)
    elif isinstance(cat_rationale, str):
        story.append(Paragraph(cat_rationale, styles["body"]))
    story.append(Spacer(1, 6))

    # ══════════════════════════════════════════════════════════════════════
    # Scoring & Implications
    # ══════════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
    story.append(Paragraph("Scoring &amp; Implications", styles["h2"]))

    scoring = narrative.get("scoring_implications", {})
    if isinstance(scoring, dict):
        score_exp = scoring.get("score_explanation", "")
        if score_exp:
            story.append(Paragraph(f"<b>Score Rationale:</b> {score_exp}", styles["body"]))
            story.append(Spacer(1, 4))

        compliance = scoring.get("patient_compliance", "")
        if compliance:
            story.append(Paragraph(f"<b>Patient Compliance:</b> {compliance}", styles["body"]))
            story.append(Spacer(1, 4))

        comp_pos = scoring.get("competitive_positioning", "")
        if comp_pos:
            story.append(Paragraph(f"<b>Competitive Positioning:</b> {comp_pos}", styles["body"]))
            story.append(Spacer(1, 4))

        risk = scoring.get("risk_considerations", "")
        if risk:
            story.append(Paragraph(f"<b>Risk Considerations:</b> {risk}", styles["body"]))
    elif isinstance(scoring, str):
        story.append(Paragraph(scoring, styles["body"]))
    story.append(Spacer(1, 8))

    # ── Methodology note ──────────────────────────────────────────────────
    methodology_note = narrative.get("score_methodology_note", "")
    if methodology_note:
        story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
        story.append(Paragraph("About the Tolerability Score", styles["h2"]))
        story.append(Paragraph(methodology_note, styles["methodology_note"]))

    # ── Scoring reference table ───────────────────────────────────────────
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
        "This report was auto-generated from Tolerability Scorer output "
        "using Gemini for narrative analysis. For internal use only.",
        styles["footer"],
    ))

    doc.build(story)
    print(f"  ✅ Report saved → {output_path}")


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
            print(f"  ✅ GCS: {gcs_uri}")
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
