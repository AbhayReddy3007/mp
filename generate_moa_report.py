"""
generate_moa_report.py
──────────────────────
Reads scored data from BigQuery `moa_innovation_table` and generates one
professional PDF report **per drug** using Gemini for narrative generation.

Only the LATEST row per drug (by created_at) is used.

The report pulls ALL available justification fields from BigQuery. If those
fields are empty or too short, the script automatically calls Gemini with
Google Search to enrich the data before generating the final narrative.

Report structure (single-drug, business-facing — max 2 pages):
  A. Executive Summary
  B. Mechanism of Action Overview
  C. Categorization & Rationale
  D. Scoring & Implications
  - Scoring reference table (end of document)

Usage:
    # All drugs — one PDF each
    python generate_moa_report.py

    # One specific drug
    python generate_moa_report.py --molecule Semaglutide

    # Two drugs — two separate PDFs
    python generate_moa_report.py --molecule "Semaglutide,Tirzepatide"

    # Custom output directory
    python generate_moa_report.py --outdir ./reports
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
BQ_TABLE = "moa_innovation_table"

GCS_BUCKET = "cognito-gcs"
GCS_BASE_PATH = "Cognito_new/reports"
GCS_FILE_NAME = "MoA_Innovation_Analysis.pdf"

REPORT_TITLE = "MoA Innovation Analysis"

# Minimum character count to consider a justification field "sufficient"
MIN_JUSTIFICATION_LENGTH = 50

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
    5: "Exceptional",
    4: "Strong",
    3: "Moderate",
    2: "Weak",
    1: "Poor",
}

CLASSIFICATION_COLORS = {
    "first-in-class": colors.HexColor("#008000"),
    "best-in-class": colors.HexColor("#4CAF50"),
    "me-too": colors.HexColor("#CC9900"),
    "weak/outdated": colors.HexColor("#E65100"),
    "poor/invalid": colors.HexColor("#CC0000"),
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
    Load the LATEST row per drug from moa_innovation_table.
    If `molecules` is provided, only those drugs are fetched.
    Returns one row per drug, sorted by drug_name.
    """
    client = _bq_client()

    molecule_filter = ""
    if molecules:
        escaped = ", ".join(f"'{m.strip()}'" for m in molecules)
        molecule_filter = f"AND drug_name IN ({escaped})"

    query = f"""
    WITH ranked AS (
        SELECT *,
            ROW_NUMBER() OVER (PARTITION BY drug_name ORDER BY created_at DESC) AS _rn
        FROM `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE}`
        WHERE drug_name IS NOT NULL {molecule_filter}
    )
    SELECT * EXCEPT(_rn) FROM ranked WHERE _rn = 1
    ORDER BY drug_name
    """
    print(f"Loading latest MoA data from {BQ_TABLE}...")
    df = client.query(query).to_dataframe()
    print(f"  Loaded {len(df)} drug(s).")
    return df


# ── Single-drug statistics ────────────────────────────────────────────────────

def extract_drug_stats(row: pd.Series) -> dict:
    """Extract ALL relevant fields for a single drug row, including
    justification sub-fields, narrative rationale, and sources."""
    def safe(val, fallback="N/A"):
        return str(val).strip() if pd.notna(val) and str(val).strip() else fallback

    score_raw = row.get("score")
    score_int = None
    try:
        score_int = int(float(score_raw))
    except (ValueError, TypeError):
        pass

    # Parse source fields (stored as JSON strings in BQ)
    def _parse_sources(val):
        raw = safe(val, "")
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else [str(parsed)]
        except (json.JSONDecodeError, TypeError):
            return [s.strip() for s in raw.split(",") if s.strip()]

    return {
        "drug_name": safe(row.get("drug_name")),
        "indication": safe(row.get("indication")),
        "mechanism": safe(row.get("mechanism_statement", "")),
        "classification": safe(row.get("moa_classification")),
        "score": score_raw,
        "score_int": score_int,
        "score_label": SCORE_LABEL.get(score_int, "N/A") if score_int else "N/A",
        "guardrail": safe(row.get("guardrail")),
        "confidence_tier": safe(row.get("confidence_tier")),
        "confidence_score": row.get("confidence_score"),
        "analysis_date": safe(row.get("analysis_date")),
        # ── Justification sub-fields ──
        "justification_mechanism_summary": safe(row.get("justification_mechanism_summary"), ""),
        "justification_novelty_vs_soc": safe(row.get("justification_novelty_vs_soc"), ""),
        "justification_competitor_comparison": safe(row.get("justification_competitor_comparison"), ""),
        "justification_biological_rationale": safe(row.get("justification_biological_rationale"), ""),
        "justification_prior_validation_failure": safe(row.get("justification_prior_validation_failure"), ""),
        "justification_why_not_higher_score": safe(row.get("justification_why_not_higher_score"), ""),
        # ── Narrative rationale ──
        "narrative_rationale": safe(row.get("narrative_rationale"), ""),
        # ── Sources ──
        "sources_primary": _parse_sources(row.get("sources_primary")),
        "sources_secondary": _parse_sources(row.get("sources_secondary")),
        "sources_tertiary": _parse_sources(row.get("sources_tertiary")),
    }


# ── Data enrichment (fetch more info if BQ data is thin) ─────────────────────

def _is_data_sufficient(drug_stats: dict) -> bool:
    """Check whether the BQ justification fields have enough substance
    to produce a detailed report without additional research."""
    key_fields = [
        "justification_mechanism_summary",
        "justification_novelty_vs_soc",
        "justification_competitor_comparison",
        "justification_biological_rationale",
        "justification_prior_validation_failure",
    ]
    filled = sum(
        1 for f in key_fields
        if len(drug_stats.get(f, "")) >= MIN_JUSTIFICATION_LENGTH
    )
    # Consider sufficient if at least 3 out of 5 key fields are filled
    return filled >= 3


def enrich_drug_data(drug_stats: dict) -> dict:
    """If BQ data is insufficient, call Gemini with Google Search to fetch
    additional MoA information for the drug. Returns an enriched copy."""
    if _is_data_sufficient(drug_stats):
        print(f"  BQ data is sufficient for {drug_stats['drug_name']} — skipping enrichment.")
        return drug_stats

    print(f"  BQ data is thin for {drug_stats['drug_name']} — enriching via Gemini + Google Search...")

    prompt = f"""You are a pharmaceutical research analyst. Research the following drug thoroughly
using the latest available scientific literature, FDA data, and clinical trial registries.

Drug: {drug_stats['drug_name']}
Indication: {drug_stats['indication']}
Known Mechanism: {drug_stats['mechanism']}
Current Classification: {drug_stats['classification']}

Provide a comprehensive analysis. Return ONLY a valid JSON object:
{{
    "mechanism_summary": "Detailed explanation (4-6 sentences) of what the drug targets, how the mechanism works at the receptor/pathway level, and the physiological effects. Include specific receptor names, pathways, and downstream effects.",
    "novelty_vs_soc": "Detailed analysis (4-6 sentences) of whether this mechanism is novel to the indication. Compare to current standard of care treatments. Name specific competing drugs and their mechanisms.",
    "competitor_comparison": "Detailed comparison (4-6 sentences) with other approved drugs and pipeline candidates targeting similar pathways. Discuss mechanistic differentiation (dual/triple agonism, selectivity, etc.).",
    "biological_rationale": "Detailed explanation (4-6 sentences) of why targeting this pathway makes biological sense for this indication. Reference key publications, preclinical evidence, and translational data.",
    "prior_validation_failure": "Detailed analysis (3-5 sentences) of whether this mechanism has been clinically validated or has had failures. Distinguish mechanism-level vs molecule-level failures.",
    "why_not_higher_score": "Explanation (2-4 sentences) of factors that limit the drug's MoA innovation score.",
    "sources": ["List of key sources: journal names, FDA references, trial IDs"]
}}"""

    try:
        text = call_gemini(prompt, use_search=True)
        enrichment = _extract_json(text)
        if enrichment:
            enriched = dict(drug_stats)
            # Only fill fields that are currently empty/thin
            field_map = {
                "mechanism_summary": "justification_mechanism_summary",
                "novelty_vs_soc": "justification_novelty_vs_soc",
                "competitor_comparison": "justification_competitor_comparison",
                "biological_rationale": "justification_biological_rationale",
                "prior_validation_failure": "justification_prior_validation_failure",
                "why_not_higher_score": "justification_why_not_higher_score",
            }
            for src_key, dst_key in field_map.items():
                existing = enriched.get(dst_key, "")
                new_val = enrichment.get(src_key, "")
                if len(new_val) > len(existing):
                    enriched[dst_key] = new_val

            # Merge sources
            new_sources = enrichment.get("sources", [])
            if new_sources and not enriched.get("sources_primary"):
                enriched["sources_primary"] = new_sources

            print(f"  ✅ Enrichment complete for {drug_stats['drug_name']}")
            return enriched
    except Exception as e:
        print(f"  [WARN] Enrichment failed for {drug_stats['drug_name']}: {e}")

    return drug_stats


# ── LLM narrative (single drug) ───────────────────────────────────────────────

def generate_drug_narrative(drug_stats: dict) -> dict:
    """
    Generate a structured, detailed report narrative for a single drug
    using the prescribed report template.

    Returns a dict with: executive_summary, moa_overview, categorization_rationale,
    scoring_implications, score_methodology_note.
    """
    # Build comprehensive MoA details block from all available data
    moa_details_parts = []
    if drug_stats.get("justification_mechanism_summary"):
        moa_details_parts.append(f"Mechanism Summary: {drug_stats['justification_mechanism_summary']}")
    if drug_stats.get("justification_novelty_vs_soc"):
        moa_details_parts.append(f"Novelty vs Standard of Care: {drug_stats['justification_novelty_vs_soc']}")
    if drug_stats.get("justification_competitor_comparison"):
        moa_details_parts.append(f"Competitor Comparison: {drug_stats['justification_competitor_comparison']}")
    if drug_stats.get("justification_biological_rationale"):
        moa_details_parts.append(f"Biological Rationale: {drug_stats['justification_biological_rationale']}")
    if drug_stats.get("justification_prior_validation_failure"):
        moa_details_parts.append(f"Prior Validation/Failure: {drug_stats['justification_prior_validation_failure']}")
    if drug_stats.get("justification_why_not_higher_score"):
        moa_details_parts.append(f"Score Limitation Factors: {drug_stats['justification_why_not_higher_score']}")
    if drug_stats.get("narrative_rationale"):
        moa_details_parts.append(f"Full Narrative Rationale: {drug_stats['narrative_rationale']}")

    moa_details_block = "\n\n".join(moa_details_parts) if moa_details_parts else "Limited information available."

    sources_block = ""
    if drug_stats.get("sources_primary"):
        sources_block += f"Primary Sources: {', '.join(drug_stats['sources_primary'])}\n"
    if drug_stats.get("sources_secondary"):
        sources_block += f"Secondary Sources: {', '.join(drug_stats['sources_secondary'])}\n"
    if drug_stats.get("sources_tertiary"):
        sources_block += f"Tertiary Sources: {', '.join(drug_stats['sources_tertiary'])}\n"

    prompt = f"""You are an expert pharmaceutical analyst specializing in product evaluation
and mechanism of action (MoA) assessment. Your task is to generate a concise, structured
report (maximum 2 pages) evaluating the Mechanism of Action (MoA) of a given product.

INPUT:
- Product Name: {drug_stats['drug_name']}
- Indication: {drug_stats['indication']}
- Mechanism of Action: {drug_stats['mechanism']}
- MoA Classification: {drug_stats['classification']}
- MoA Score: {drug_stats['score']} / 5  ({drug_stats['score_label']})
- Guardrail: {drug_stats['guardrail']}
- Confidence Tier: {drug_stats['confidence_tier']}
- Confidence Score: {drug_stats.get('confidence_score', 'N/A')}

DETAILED MOA ANALYSIS DATA (use this extensively):
{moa_details_block}

SOURCES:
{sources_block if sources_block else 'Not specified'}

OUTPUT REQUIREMENTS — Generate a structured report with these exact sections:

A. EXECUTIVE SUMMARY
- Provide a crisp overview (6-8 bullet points) covering:
  - The mechanism of action of the product
  - The categorization bucket (as per the results generated)
  - The assigned MoA score (on a scale of 1-5)
- Keep this section concise and decision-oriented

B. MECHANISM OF ACTION OVERVIEW
- Explain what the mechanism of action is in detail
- Describe the biological pathway or target involved
- Name specific receptors, enzymes, or pathways
- Use simple, clear, scientifically accurate language
- Avoid unnecessary jargon

C. CATEGORIZATION & RATIONALE
- Clearly state the MoA category (First-in-Class, Best-in-Class, Me-too, etc.)
- Justify the classification with detailed reasoning
- Reference competitor drugs and landscape context
- Explain how this drug compares to the current standard of care

D. SCORING & IMPLICATIONS
- Provide the MoA score (1-5 scale) and explain why this score was assigned
- Explain what the score implies for:
  - Competitive positioning
  - Strategic value in portfolio selection
- If relevant, note what would be needed to achieve a higher score

FORMATTING & HYGIENE INSTRUCTIONS:
- Use bullet points wherever possible to improve readability
- Keep paragraphs short (2-4 lines max)
- Avoid repetition and verbose explanations
- Ensure logical flow across sections
- Use precise, evidence-based reasoning (do not speculate)
- Maintain a professional, analytical tone suitable for leadership review
- Prioritize clarity, brevity, and structured thinking

Respond ONLY with a valid JSON object (no markdown fences, no extra text):
{{
  "executive_summary": [
    "Bullet point 1 about the drug's MoA and its significance",
    "Bullet point 2 about the classification and what it means",
    "Bullet point 3 about the score and confidence level",
    "Bullet point 4 about key differentiators or limitations",
    "Bullet point 5 about competitive landscape positioning",
    "Bullet point 6 about strategic implications"
  ],
  "moa_overview": {{
    "mechanism_description": "Detailed 3-5 sentence description of how the mechanism works at the molecular/pathway level",
    "biological_pathway": "2-3 sentence description of the biological pathway or target involved, including specific receptors and downstream effects",
    "clinical_relevance": "2-3 sentences explaining why this mechanism matters clinically for the indication"
  }},
  "categorization_rationale": {{
    "category": "The MoA classification (e.g. First-in-Class, Best-in-Class, Me-too, etc.)",
    "rationale_points": [
      "Detailed justification point 1 for the classification",
      "Detailed justification point 2 referencing competitors",
      "Detailed justification point 3 referencing SOC comparison"
    ]
  }},
  "scoring_implications": {{
    "score_explanation": "2-3 sentences explaining why this specific score was assigned based on the scoring criteria",
    "competitive_positioning": "2-3 sentences on what this score means for competitive positioning in the market",
    "strategic_value": "2-3 sentences on the strategic value and portfolio implications",
    "path_to_higher_score": "1-2 sentences on what would be needed to achieve a higher score (if applicable)"
  }},
  "score_methodology_note": "2-3 plain-language sentences explaining how the MoA score is determined at a high level"
}}"""

    text = call_gemini(prompt)
    result = _extract_json(text)
    if result:
        return result

    # Fallback
    return {
        "executive_summary": [
            f"{drug_stats['drug_name']} targets {drug_stats['indication']} via {drug_stats['mechanism'][:150]}.",
            f"Classified as {drug_stats['classification']} with a MoA score of {drug_stats['score']}/5 ({drug_stats['score_label']}).",
            f"Guardrail status: {drug_stats['guardrail']}. Confidence: {drug_stats['confidence_tier']}.",
        ],
        "moa_overview": {
            "mechanism_description": drug_stats.get("justification_mechanism_summary") or f"MoA: {drug_stats['mechanism'][:300]}",
            "biological_pathway": drug_stats.get("justification_biological_rationale") or "See detailed analysis.",
            "clinical_relevance": drug_stats.get("justification_novelty_vs_soc") or "See detailed analysis.",
        },
        "categorization_rationale": {
            "category": drug_stats["classification"],
            "rationale_points": [
                drug_stats.get("justification_competitor_comparison") or "See score summary for details.",
            ],
        },
        "scoring_implications": {
            "score_explanation": f"MoA score: {drug_stats['score']} / 5 ({drug_stats['score_label']}).",
            "competitive_positioning": "Refer to detailed analysis.",
            "strategic_value": "Refer to detailed analysis.",
            "path_to_higher_score": drug_stats.get("justification_why_not_higher_score") or "N/A",
        },
        "score_methodology_note": "Scores reflect the novelty and clinical validation of the drug's mechanism of action on a 1-5 scale.",
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
        "drug_name_title": ParagraphStyle(
            "DrugNameTitle", parent=base["Normal"],
            fontSize=14, leading=18, textColor=ACCENT_BLUE,
            alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle", parent=base["Normal"],
            fontSize=9, leading=12, textColor=LIGHT_GRAY,
            alignment=TA_CENTER, fontName="Helvetica", spaceAfter=12,
        ),
        "headline_box": ParagraphStyle(
            "HeadlineBox", parent=base["Normal"],
            fontSize=11, leading=16, textColor=WHITE,
            fontName="Helvetica-Bold", alignment=TA_LEFT,
            spaceAfter=0, leftIndent=0,
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
        "insight_label": ParagraphStyle(
            "InsightLabel", parent=base["Normal"],
            fontSize=10, leading=14, textColor=DARK_BLUE,
            fontName="Helvetica", spaceAfter=1, leftIndent=12,
        ),
        "insight_justification": ParagraphStyle(
            "InsightJustification", parent=base["Normal"],
            fontSize=9, leading=13, textColor=colors.HexColor("#444444"),
            fontName="Helvetica", spaceAfter=6, leftIndent=24,
        ),
        "score_ref": ParagraphStyle(
            "ScoreRef", parent=base["Normal"],
            fontSize=9, leading=12, textColor=LIGHT_GRAY,
            fontName="Helvetica", spaceAfter=4,
        ),
        "bottom_line": ParagraphStyle(
            "BottomLine", parent=base["Normal"],
            fontSize=10, leading=14, textColor=colors.HexColor("#1A1A1A"),
            fontName="Helvetica", spaceAfter=4, alignment=TA_JUSTIFY,
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
        "cell_label": ParagraphStyle(
            "CellLabel", parent=base["Normal"],
            fontSize=9, leading=12, textColor=colors.HexColor("#333333"),
            fontName="Helvetica-Bold",
        ),
        "cell_value": ParagraphStyle(
            "CellValue", parent=base["Normal"],
            fontSize=9, leading=12, textColor=colors.HexColor("#333333"),
            fontName="Helvetica",
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


def _classification_color(cls_str):
    cls_lower = str(cls_str).lower()
    for key, col in CLASSIFICATION_COLORS.items():
        if key in cls_lower:
            return col
    return colors.black


def _headline_box(headline_text: str, styles: dict, story: list):
    """Render a dark-blue shaded headline banner."""
    tbl = Table(
        [[Paragraph(f"&#9654; {headline_text}", styles["headline_box"])]],
        colWidths=[6.8 * inch],
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARK_BLUE),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 8))


def _render_bullets(items: list, styles: dict, story: list):
    """Render a list of strings as bullet points."""
    for item in items:
        if item and str(item).strip():
            story.append(Paragraph(
                f"&#8226; {item}",
                styles["bullet"],
            ))


def _score_summary_table(drug_stats: dict, styles: dict) -> Table:
    """Render a compact single-drug score summary table."""
    guardrail_val = drug_stats["guardrail"]
    guardrail_color = (
        GUARDRAIL_PASS_COLOR if guardrail_val == "PASS"
        else GUARDRAIL_FAIL_COLOR if guardrail_val == "FAIL"
        else colors.black
    )
    score_color = _score_color(drug_stats["score"])
    cls_color = _classification_color(drug_stats["classification"])

    def labeled_cell(label, value, value_color=None):
        val_style = ParagraphStyle(
            f"VS_{label}", parent=styles["cell_value"],
            textColor=value_color or colors.HexColor("#333333"),
            fontName="Helvetica",
        )
        return [
            Paragraph(label, styles["cell_label"]),
            Paragraph(value, val_style),
        ]

    conf_score = drug_stats.get("confidence_score")
    conf_display = f"{drug_stats['confidence_tier']}"
    if conf_score is not None:
        try:
            conf_display += f" ({float(conf_score)*100:.0f}%)"
        except (ValueError, TypeError):
            pass

    data = [
        labeled_cell("Drug", drug_stats["drug_name"]),
        labeled_cell("Indication", drug_stats["indication"]),
        labeled_cell("Classification", drug_stats["classification"], cls_color),
        labeled_cell("MoA Score", f"{drug_stats['score']} / 5  ({drug_stats['score_label']})", score_color),
        labeled_cell("Guardrail", guardrail_val, guardrail_color),
        labeled_cell("Confidence", conf_display),
        labeled_cell("Analysis Date", drug_stats["analysis_date"]),
    ]

    tbl = Table(data, colWidths=[2.0 * inch, 4.7 * inch])
    tbl.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE_BG),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return tbl


def _scoring_framework_table(styles: dict) -> Table:
    """Render the MoA scoring reference table."""
    framework = [
        ("5", "Exceptional", "True First-in-Class: novel mechanism, strong rationale, class-creating potential"),
        ("4", "Strong",      "Validated class with clearly superior mechanism differentiation (Best-in-Class)"),
        ("3", "Moderate",    "Validated class with limited innovation (Me-too / Fast Follower)"),
        ("2", "Weak",        "Older or strategically outdated mechanism"),
        ("1", "Poor",        "Weak biological rationale or clinically invalidated mechanism"),
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


# ── Single-drug report builder ────────────────────────────────────────────────

def build_single_drug_report(drug_stats: dict, narrative: dict, output_path: str):
    """Build and save a detailed PDF report for one drug."""
    styles = build_styles()
    story = []

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        title=f"{REPORT_TITLE} — {drug_stats['drug_name']}",
        author="MoA Innovation Scorer",
    )

    # ── Title block ───────────────────────────────────────────────────────────
    story.append(Paragraph(REPORT_TITLE, styles["title"]))
    story.append(Paragraph(drug_stats["drug_name"], styles["drug_name_title"]))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%B %d, %Y')}  •  {drug_stats['indication']}",
        styles["subtitle"],
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=DARK_BLUE, spaceAfter=12))

    # ── Score summary table ───────────────────────────────────────────────────
    story.append(Paragraph("Score Summary", styles["h2"]))
    story.append(_score_summary_table(drug_stats, styles))
    story.append(Spacer(1, 10))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION A: Executive Summary
    # ══════════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
    story.append(Paragraph("A. Executive Summary", styles["h2"]))

    exec_summary = narrative.get("executive_summary", [])
    if isinstance(exec_summary, list):
        _render_bullets(exec_summary, styles, story)
    elif isinstance(exec_summary, str):
        story.append(Paragraph(exec_summary, styles["body"]))
    story.append(Spacer(1, 6))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION B: Mechanism of Action Overview
    # ══════════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
    story.append(Paragraph("B. Mechanism of Action Overview", styles["h2"]))

    moa_overview = narrative.get("moa_overview", {})
    if isinstance(moa_overview, dict):
        mechanism_desc = moa_overview.get("mechanism_description", "")
        if mechanism_desc:
            story.append(Paragraph(mechanism_desc, styles["body"]))
            story.append(Spacer(1, 4))

        bio_pathway = moa_overview.get("biological_pathway", "")
        if bio_pathway:
            story.append(Paragraph("<b>Biological Pathway / Target:</b>", styles["section_label"]))
            story.append(Paragraph(bio_pathway, styles["body"]))
            story.append(Spacer(1, 4))

        clinical_rel = moa_overview.get("clinical_relevance", "")
        if clinical_rel:
            story.append(Paragraph("<b>Clinical Relevance:</b>", styles["section_label"]))
            story.append(Paragraph(clinical_rel, styles["body"]))
    elif isinstance(moa_overview, str):
        story.append(Paragraph(moa_overview, styles["body"]))
    story.append(Spacer(1, 6))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION C: Categorization & Rationale
    # ══════════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
    story.append(Paragraph("C. Categorization &amp; Rationale", styles["h2"]))

    cat_rationale = narrative.get("categorization_rationale", {})
    if isinstance(cat_rationale, dict):
        category = cat_rationale.get("category", drug_stats["classification"])
        cls_color = _classification_color(category)
        cat_style = ParagraphStyle(
            "CatValue", parent=styles["body"],
            textColor=cls_color, fontName="Helvetica-Bold", fontSize=11,
        )
        story.append(Paragraph(f"Classification: {category}", cat_style))
        story.append(Spacer(1, 4))

        rationale_points = cat_rationale.get("rationale_points", [])
        if rationale_points:
            _render_bullets(rationale_points, styles, story)
    elif isinstance(cat_rationale, str):
        story.append(Paragraph(cat_rationale, styles["body"]))
    story.append(Spacer(1, 6))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION D: Scoring & Implications
    # ══════════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
    story.append(Paragraph("D. Scoring &amp; Implications", styles["h2"]))

    scoring = narrative.get("scoring_implications", {})
    if isinstance(scoring, dict):
        score_exp = scoring.get("score_explanation", "")
        if score_exp:
            story.append(Paragraph(f"<b>Score Rationale:</b> {score_exp}", styles["body"]))
            story.append(Spacer(1, 4))

        comp_pos = scoring.get("competitive_positioning", "")
        if comp_pos:
            story.append(Paragraph(f"<b>Competitive Positioning:</b> {comp_pos}", styles["body"]))
            story.append(Spacer(1, 4))

        strat_val = scoring.get("strategic_value", "")
        if strat_val:
            story.append(Paragraph(f"<b>Strategic Value:</b> {strat_val}", styles["body"]))
            story.append(Spacer(1, 4))

        path_higher = scoring.get("path_to_higher_score", "")
        if path_higher:
            story.append(Paragraph(f"<b>Path to Higher Score:</b> {path_higher}", styles["body"]))
    elif isinstance(scoring, str):
        story.append(Paragraph(scoring, styles["body"]))
    story.append(Spacer(1, 8))

    # ── Sources ───────────────────────────────────────────────────────────
    has_sources = any([
        drug_stats.get("sources_primary"),
        drug_stats.get("sources_secondary"),
        drug_stats.get("sources_tertiary"),
    ])
    if has_sources:
        story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
        story.append(Paragraph("Sources", styles["h2"]))
        if drug_stats.get("sources_primary"):
            story.append(Paragraph(
                f"<b>Primary:</b> {', '.join(drug_stats['sources_primary'][:5])}",
                styles["methodology_note"],
            ))
        if drug_stats.get("sources_secondary"):
            story.append(Paragraph(
                f"<b>Secondary:</b> {', '.join(drug_stats['sources_secondary'][:5])}",
                styles["methodology_note"],
            ))
        if drug_stats.get("sources_tertiary"):
            story.append(Paragraph(
                f"<b>Tertiary:</b> {', '.join(drug_stats['sources_tertiary'][:5])}",
                styles["methodology_note"],
            ))
        story.append(Spacer(1, 6))

    # ── Methodology note ──────────────────────────────────────────────────
    methodology_note = narrative.get("score_methodology_note", "")
    if methodology_note:
        story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
        story.append(Paragraph("About the MoA Innovation Score", styles["h2"]))
        story.append(Paragraph(methodology_note, styles["methodology_note"]))

    # ── Scoring reference table ───────────────────────────────────────────
    story.append(Paragraph("MoA Innovation Scoring Reference", styles["h2"]))
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
        "This report was auto-generated from MoA Innovation Scorer output "
        "using Gemini for narrative analysis. For internal use only.",
        styles["footer"],
    ))

    doc.build(story)
    print(f"  ✅ Report saved → {output_path}")


# ── GCS Upload ────────────────────────────────────────────────────────────────

def upload_to_gcs(local_path: str, drug_name: str) -> str:
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError("google-cloud-storage required. pip install google-cloud-storage")

    credentials = _get_credentials()
    client = storage.Client(project=BQ_PROJECT_ID, credentials=credentials)
    bucket = client.bucket(GCS_BUCKET)

    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(drug_name))
    blob_name = f"{GCS_BASE_PATH}/{safe_name}/{GCS_FILE_NAME}"
    gcs_uri = f"gs://{GCS_BUCKET}/{blob_name}"

    print(f"  Uploading to GCS: {gcs_uri}")
    bucket.blob(blob_name).upload_from_filename(local_path, content_type="application/pdf")
    return gcs_uri


# ── Public entry point ────────────────────────────────────────────────────────

def generate_moa_report(
    molecules: list[str] | None = None,
    outdir: str | None = None,
) -> list[str]:
    """
    Generate one PDF report per drug.

    Args:
        molecules: List of drug names to report on. None = all drugs in table.
        outdir:    Directory to write PDFs into. Defaults to current directory.

    Returns:
        List of output PDF paths that were successfully created.
    """
    if not API_KEY:
        print("[MoA Report] GEMINI_API_KEY not set — skipping report generation.")
        return []

    out_root = Path(outdir) if outdir else Path(".")
    out_root.mkdir(parents=True, exist_ok=True)

    df = load_from_bigquery(molecules)
    if df.empty:
        print("[MoA Report] No data found — skipping.")
        return []

    output_paths = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for _, row in df.iterrows():
        drug_stats = extract_drug_stats(row)
        drug_name = drug_stats["drug_name"]

        print(f"\nProcessing: {drug_name}")

        # Enrich data if BQ fields are insufficient
        drug_stats = enrich_drug_data(drug_stats)

        print("  Generating narrative with Gemini...")
        narrative = generate_drug_narrative(drug_stats)

        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", drug_name)
        output_path = str(out_root / f"moa_report_{safe_name}_{ts}.pdf")

        build_single_drug_report(drug_stats, narrative, output_path)
        output_paths.append(output_path)

        try:
            gcs_uri = upload_to_gcs(output_path, drug_name)
            print(f"  ✅ GCS: {gcs_uri}")
        except Exception as e:
            print(f"  [WARN] GCS upload failed for '{drug_name}': {e}")

    print(f"\nDone. {len(output_paths)} report(s) generated.")
    return output_paths


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate one MoA Innovation PDF per drug from BigQuery data."
    )
    parser.add_argument(
        "--molecule", "-m",
        default=None,
        help=(
            "Comma-separated drug name(s) to report on. "
            "E.g. --molecule Semaglutide  or  --molecule 'Semaglutide,Tirzepatide'. "
            "Omit to process all drugs in the table."
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

    generate_moa_report(molecules=molecules, outdir=args.outdir)


if __name__ == "__main__":
    main()
