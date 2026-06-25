"""
generate_efficacy_report.py
───────────────────────────
Reads clinical trial data from BigQuery `clinical_efficacy` table and generates
one professional PDF report **per molecule** using Gemini for narrative generation.

The report aggregates per-trial data into endpoint-level summaries
(Weight Loss, HbA1c, MASH Resolution, ALT Reduction), uses the stored
efficacy score and rationale, and enriches via Gemini+Search if data is thin.

Report structure (single-molecule, business-facing — max 2 pages):
  - Executive Summary
  - Endpoint Performance Overview
  - Scoring Methodology & Rationale
  - Strategic Implications
  - Scoring reference table (end of document)

Usage:
    python generate_efficacy_report.py
    python generate_efficacy_report.py --molecule Semaglutide
    python generate_efficacy_report.py --molecule "Semaglutide,Tirzepatide"
    python generate_efficacy_report.py --outdir ./reports
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
BQ_TABLE = "clinical_efficacy"

GCS_BUCKET = "cognito-gcs"
GCS_BASE_PATH = "Cognito_new/reports"
GCS_FILE_NAME = "Clinical_Efficacy_Analysis.pdf"

REPORT_TITLE = "Clinical Efficacy Analysis"

MIN_RATIONALE_LENGTH = 80

# Scoring constants (from clinical_efficacy_scorer.py)
SCORE_TABLE = [(22.0, 5), (16.0, 4), (10.0, 3), (5.0, 2), (0.0, 1)]
ENDPOINT_WEIGHTS = {"weight_loss": 0.40, "hba1c": 0.40, "mash": 0.10, "alt": 0.10}
ENDPOINT_LABELS = {
    "weight_loss": "Weight Loss",
    "hba1c": "HbA1c Reduction",
    "mash": "MASH Resolution",
    "alt": "ALT Reduction",
}
PHASE_PENALTY = {3: 1.00, 2: 0.85, 1: 0.65}

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


# ── Gemini helpers ────────────────────────────────────────────────────────────

def call_gemini(prompt: str, use_search: bool = False) -> str:
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


def load_from_bigquery(molecules: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """
    Load ALL trial rows per molecule from the clinical_efficacy table.
    Returns a dict: {molecule_name: DataFrame of trials}.
    """
    client = _bq_client()

    molecule_filter = ""
    if molecules:
        escaped = ", ".join(f"'{m.strip()}'" for m in molecules)
        molecule_filter = f"AND molecule_name IN ({escaped})"

    query = f"""
    SELECT *
    FROM `{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE}`
    WHERE molecule_name IS NOT NULL {molecule_filter}
    ORDER BY molecule_name, phase DESC, created_at DESC
    """
    print(f"Loading clinical efficacy data from {BQ_TABLE}...")
    df = client.query(query).to_dataframe()
    if df.empty:
        print("  No data found.")
        return {}

    grouped = {name: group for name, group in df.groupby("molecule_name")}
    print(f"  Loaded data for {len(grouped)} molecule(s), {len(df)} total trial rows.")
    return grouped


# ── Endpoint computation from trial data ─────────────────────────────────────

def _parse_float(raw) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if s.lower() in ("n/a", "", "0", "none", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_phase(raw) -> int | None:
    if raw is None:
        return None
    s = str(raw).strip().upper().replace("PHASE", "").strip()
    if s.startswith("3"):
        return 3
    if s.startswith("2"):
        return 2
    if s.startswith("1"):
        return 1
    try:
        v = float(s)
        return 3 if v >= 3 else (2 if v >= 2 else 1)
    except ValueError:
        return None


def _pct_to_score(pct: float) -> int:
    for threshold, score in SCORE_TABLE:
        if pct >= threshold:
            return score
    return 1


def _compute_endpoint(trials_df: pd.DataFrame, value_col: str, duration_col: str) -> dict:
    """Compute best value for a single endpoint using phase-anchored selection."""
    best = {"raw_value": None, "adj_value": None, "phase_used": None,
            "penalty": 1.0, "score": None, "trial_id": "N/A",
            "dosage": "N/A", "duration": "N/A", "reason": "No valid data"}

    candidates = []
    for _, row in trials_df.iterrows():
        phase = _parse_phase(row.get("phase"))
        value = _parse_float(row.get(value_col))
        size = _parse_float(row.get("trial_size"))
        if phase is None or value is None or (size is not None and size <= 0):
            continue
        candidates.append({
            "phase": phase, "value": value,
            "trial_id": str(row.get("trial_id", "N/A")),
            "dosage": str(row.get("dosage", "N/A")),
            "duration": str(row.get(duration_col, "N/A")),
        })

    if not candidates:
        return best

    for target_phase in (3, 2, 1):
        phase_cands = [c for c in candidates if c["phase"] == target_phase]
        if not phase_cands:
            continue
        top = max(phase_cands, key=lambda c: c["value"])
        pen = PHASE_PENALTY[target_phase]
        adj = top["value"] * pen
        return {
            "raw_value": round(top["value"], 2),
            "adj_value": round(adj, 2),
            "phase_used": target_phase,
            "penalty": pen,
            "score": _pct_to_score(adj),
            "trial_id": top["trial_id"],
            "dosage": top["dosage"],
            "duration": top["duration"],
            "reason": f"Phase {target_phase}" + (f" (x{pen} penalty)" if pen < 1 else ""),
        }
    return best


# ── Extract molecule stats ────────────────────────────────────────────────────

def extract_molecule_stats(molecule_name: str, trials_df: pd.DataFrame) -> dict:
    """Aggregate trial data into a structured stats dict for report generation."""
    # Get stored score/rationale from first row (same across rows)
    first = trials_df.iloc[0]
    efficacy_score = _parse_float(first.get("efficacy_score"))
    data_coverage = str(first.get("data_coverage", "N/A")) if pd.notna(first.get("data_coverage")) else "N/A"
    rationale = str(first.get("rationale", "")) if pd.notna(first.get("rationale")) else ""

    score_int = None
    if efficacy_score is not None:
        score_int = max(1, min(5, round(efficacy_score)))

    # Compute per-endpoint breakdowns
    endpoints = {
        "weight_loss": _compute_endpoint(trials_df, "weight_change_pct", "weight_duration"),
        "hba1c": _compute_endpoint(trials_df, "hba1c_change_pct", "hba1c_duration"),
        "mash": _compute_endpoint(trials_df, "mash_resolution_pct", "mash_duration"),
        "alt": _compute_endpoint(trials_df, "alt_reduction_pct", "alt_duration"),
    }

    # Recompute weighted score from endpoints
    weighted_score = 0.0
    scored_count = 0
    for ep, result in endpoints.items():
        w = ENDPOINT_WEIGHTS[ep]
        if result["score"] is not None:
            weighted_score += result["score"] * w
            scored_count += 1

    # Prefer stored score, fall back to recomputed
    display_score = efficacy_score if efficacy_score is not None else round(weighted_score, 3)

    total_trials = len(trials_df)
    phases = trials_df["phase"].dropna().apply(lambda x: str(x).strip()).value_counts().to_dict()

    # Company name
    company = "N/A"
    for _, row in trials_df.iterrows():
        c = row.get("company_name")
        if pd.notna(c) and str(c).strip():
            company = str(c).strip()
            break

    return {
        "molecule_name": molecule_name,
        "efficacy_score": display_score,
        "score_int": score_int,
        "score_label": SCORE_LABEL.get(score_int, "N/A") if score_int else "N/A",
        "data_coverage": data_coverage,
        "rationale": rationale,
        "total_trials": total_trials,
        "phase_distribution": phases,
        "company_name": company,
        "endpoints": endpoints,
        "scored_endpoints": scored_count,
    }


# ── Data enrichment ───────────────────────────────────────────────────────────

def _is_data_sufficient(stats: dict) -> bool:
    rationale_ok = len(stats.get("rationale", "")) >= MIN_RATIONALE_LENGTH
    has_any_endpoint = stats.get("scored_endpoints", 0) >= 1
    return rationale_ok and has_any_endpoint


def enrich_molecule_data(stats: dict) -> dict:
    """If BQ data is insufficient, call Gemini with Google Search to fetch
    additional clinical efficacy information."""
    if _is_data_sufficient(stats):
        print(f"  BQ data is sufficient for {stats['molecule_name']} — skipping enrichment.")
        return stats

    print(f"  BQ data is thin for {stats['molecule_name']} — enriching via Gemini + Google Search...")

    prompt = f"""You are a pharmaceutical research analyst. Research the clinical efficacy data
for the following drug using the latest clinical trial publications, FDA labels, and registries.

Drug: {stats['molecule_name']}
Company: {stats['company_name']}
Current Score: {stats['efficacy_score']}/5

Find the best available data for these endpoints from Phase 2/3 trials:
1. Weight Loss (% body weight reduction)
2. HbA1c Reduction (% point reduction)
3. MASH/NASH Resolution (% of patients achieving resolution)
4. ALT Reduction (% reduction in ALT levels)

Return ONLY a valid JSON object:
{{
    "weight_loss_pct": "Best % weight loss from pivotal trials (e.g. 15.2%)",
    "weight_loss_trial": "Trial name/ID and dosage (e.g. STEP 1, 2.4mg, 68 weeks)",
    "hba1c_pct": "Best % HbA1c reduction (e.g. 1.8%)",
    "hba1c_trial": "Trial name/ID and dosage",
    "mash_pct": "Best MASH resolution % (or N/A if not studied)",
    "mash_trial": "Trial name/ID if available",
    "alt_pct": "Best ALT reduction % (or N/A)",
    "alt_trial": "Trial name/ID if available",
    "efficacy_narrative": "5-6 sentence comprehensive clinical efficacy summary covering all endpoints, key trial results, dosages, durations, and clinical significance",
    "key_trials": ["List of key trial names/IDs: STEP 1, SURPASS-1, etc."]
}}"""

    try:
        text = call_gemini(prompt, use_search=True)
        enrichment = _extract_json(text)
        if enrichment:
            enriched = dict(stats)
            if len(stats.get("rationale", "")) < MIN_RATIONALE_LENGTH and enrichment.get("efficacy_narrative"):
                enriched["rationale"] = enrichment["efficacy_narrative"]
            enriched["_enrichment"] = enrichment
            print(f"  Enrichment complete for {stats['molecule_name']}")
            return enriched
    except Exception as e:
        print(f"  [WARN] Enrichment failed for {stats['molecule_name']}: {e}")

    return stats


# ── LLM narrative (single molecule) ──────────────────────────────────────────

def generate_efficacy_narrative(stats: dict) -> dict:
    """Generate a structured, detailed clinical efficacy report narrative."""

    # Build endpoint summary block
    ep_lines = []
    for ep_key, ep_label in ENDPOINT_LABELS.items():
        ep = stats["endpoints"].get(ep_key, {})
        weight_pct = int(ENDPOINT_WEIGHTS[ep_key] * 100)
        if ep.get("score") is not None:
            ep_lines.append(
                f"- {ep_label} ({weight_pct}% weight): Raw {ep['raw_value']}%, "
                f"Adjusted {ep['adj_value']}%, Score {ep['score']}/5, "
                f"Phase {ep['phase_used']}, Trial {ep['trial_id']}, "
                f"Dosage {ep['dosage']}, Duration {ep['duration']}, "
                f"{ep['reason']}"
            )
        else:
            ep_lines.append(f"- {ep_label} ({weight_pct}% weight): No valid data available")

    enrichment = stats.get("_enrichment", {})
    enrichment_block = ""
    if enrichment:
        enrich_parts = []
        for k in ["weight_loss_pct", "weight_loss_trial", "hba1c_pct", "hba1c_trial",
                   "mash_pct", "mash_trial", "alt_pct", "alt_trial", "efficacy_narrative"]:
            val = enrichment.get(k, "")
            if val and val != "N/A":
                enrich_parts.append(f"{k.replace('_', ' ').title()}: {val}")
        if enrich_parts:
            enrichment_block = "\n".join(enrich_parts)

    prompt = f"""You are an expert pharmaceutical analyst specializing in clinical efficacy
assessment. Your task is to generate a concise, structured report (maximum 2 pages)
evaluating the Clinical Efficacy of a given product.

INPUT:
- Product Name: {stats['molecule_name']}
- Company: {stats['company_name']}
- Clinical Efficacy Score: {stats['efficacy_score']} / 5 ({stats['score_label']})
- Data Coverage: {stats['data_coverage']}
- Total Trials Analyzed: {stats['total_trials']}
- Phase Distribution: {json.dumps(stats['phase_distribution'])}
- Endpoints Scored: {stats['scored_endpoints']}/4

ENDPOINT PERFORMANCE:
{chr(10).join(ep_lines)}

STORED RATIONALE FROM ANALYSIS:
{stats.get('rationale', 'Not available')}

ADDITIONAL RESEARCH DATA:
{enrichment_block if enrichment_block else 'Not available'}

OUTPUT REQUIREMENTS — Generate a structured report with these exact sections:

EXECUTIVE SUMMARY
- Provide a crisp overview (6-8 bullet points) covering:
  - The clinical efficacy score and what it means
  - Performance across the four endpoints (Weight Loss, HbA1c, MASH, ALT)
  - Data quality: number of trials, phase distribution, coverage
  - Key differentiators or limitations
- Keep this section concise and decision-oriented

ENDPOINT PERFORMANCE OVERVIEW
- For each of the 4 endpoints, provide a focused subsection:
  - Weight Loss: best result achieved, dosage, duration, trial phase, clinical significance
  - HbA1c Reduction: best result, dosage, duration, trial, clinical benchmarks
  - MASH Resolution: result if available, or explain why data is limited
  - ALT Reduction: result if available, or explain limited data
- Reference specific trial IDs, dosages, and durations
- Compare against clinical significance thresholds (e.g., >15% weight loss = exceptional)

SCORING METHODOLOGY & RATIONALE
- Explain the scoring methodology (threshold table, phase penalties, endpoint weights)
- Show how the weighted score was calculated
- Justify why this score was assigned
- Note any penalties applied (Phase 2 data = x0.85, Phase 1 = x0.65)

STRATEGIC IMPLICATIONS
- Explain what the efficacy score means for competitive positioning
- Discuss strategic value for portfolio selection
- Note implications for regulatory pathway and commercial potential
- Compare against class benchmarks where relevant

FORMATTING & HYGIENE INSTRUCTIONS:
- Use bullet points wherever possible
- Keep paragraphs short (2-4 lines max)
- Avoid repetition
- Use precise, evidence-based reasoning
- Professional, analytical tone suitable for leadership review

Respond ONLY with a valid JSON object (no markdown fences, no extra text):
{{
  "executive_summary": [
    "Bullet 1 about the efficacy score and overall performance",
    "Bullet 2 about weight loss performance",
    "Bullet 3 about HbA1c performance",
    "Bullet 4 about MASH/ALT or data coverage",
    "Bullet 5 about data quality and trial distribution",
    "Bullet 6 about strategic implications"
  ],
  "endpoint_overview": {{
    "weight_loss": "3-5 sentences on weight loss efficacy with specific numbers, dosage, duration, trial ID, and clinical significance benchmarks",
    "hba1c": "3-5 sentences on HbA1c reduction with specific numbers, dosage, duration, trial ID, and clinical benchmarks",
    "mash": "2-3 sentences on MASH resolution data or explanation of why it is limited/not available",
    "alt": "2-3 sentences on ALT reduction data or explanation of why it is limited/not available"
  }},
  "scoring_rationale": {{
    "methodology": "2-3 sentences explaining the scoring methodology (thresholds, phase penalties, weights)",
    "calculation": "2-3 sentences showing how the weighted score was computed from endpoint scores",
    "justification": "2-3 sentences on why this score accurately reflects the drug's clinical efficacy"
  }},
  "strategic_implications": {{
    "competitive_positioning": "2-3 sentences on how this efficacy profile positions the drug competitively",
    "regulatory_commercial": "2-3 sentences on regulatory pathway and commercial potential implications",
    "portfolio_value": "2-3 sentences on strategic value for portfolio decisions"
  }},
  "score_methodology_note": "2-3 plain-language sentences explaining how the clinical efficacy score is determined at a high level"
}}"""

    text = call_gemini(prompt)
    result = _extract_json(text)
    if result:
        return result

    # Fallback
    return {
        "executive_summary": [
            f"{stats['molecule_name']} achieves a clinical efficacy score of {stats['efficacy_score']}/5 ({stats['score_label']}).",
            f"Data coverage: {stats['data_coverage']} across {stats['total_trials']} trials.",
            f"Endpoints scored: {stats['scored_endpoints']}/4.",
        ],
        "endpoint_overview": {
            "weight_loss": stats.get("rationale", "See detailed analysis."),
            "hba1c": "See detailed analysis.",
            "mash": "See detailed analysis.",
            "alt": "See detailed analysis.",
        },
        "scoring_rationale": {
            "methodology": "Scored using phase-anchored threshold table with endpoint weighting.",
            "calculation": f"Weighted score: {stats['efficacy_score']}/5.",
            "justification": "Based on best available clinical trial data.",
        },
        "strategic_implications": {
            "competitive_positioning": "Refer to detailed analysis.",
            "regulatory_commercial": "Refer to detailed analysis.",
            "portfolio_value": "Refer to detailed analysis.",
        },
        "score_methodology_note": (
            "The clinical efficacy score is a weighted average of four endpoints: "
            "Weight Loss (40%), HbA1c Reduction (40%), MASH Resolution (10%), and ALT Reduction (10%). "
            "Phase penalties apply for Phase 2 (x0.85) and Phase 1 (x0.65) data."
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
            fontName="Helvetica", spaceAfter=3, leftIndent=18, bulletIndent=6,
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
    """Render the clinical efficacy scoring reference table."""
    framework = [
        ("5", "Exceptional", "Adjusted efficacy value >= 22% (class-leading performance)"),
        ("4", "Strong",      "Adjusted efficacy value 16-21.9% (above-average efficacy)"),
        ("3", "Moderate",    "Adjusted efficacy value 10-15.9% (clinically meaningful)"),
        ("2", "Weak",        "Adjusted efficacy value 5-9.9% (below expectations)"),
        ("1", "Poor",        "Adjusted efficacy value < 5% (minimal clinical benefit)"),
    ]
    header = [
        Paragraph("Score", styles["cell_header"]),
        Paragraph("Label", styles["cell_header"]),
        Paragraph("Description", styles["cell_header"]),
    ]
    rows = [header]
    for sc, lbl, desc in framework:
        rows.append([
            Paragraph(sc, ParagraphStyle("FWS", parent=styles["cell"], fontName="Helvetica")),
            Paragraph(lbl, ParagraphStyle("FWL", parent=styles["cell"], fontName="Helvetica")),
            Paragraph(desc, ParagraphStyle("FWD", parent=styles["cell"], alignment=TA_LEFT)),
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


def _endpoint_summary_table(stats: dict, styles: dict) -> Table:
    """Render the 4-endpoint performance summary table."""
    header = [
        Paragraph("Endpoint", styles["cell_header"]),
        Paragraph("Weight", styles["cell_header"]),
        Paragraph("Raw Value", styles["cell_header"]),
        Paragraph("Adjusted", styles["cell_header"]),
        Paragraph("Score", styles["cell_header"]),
        Paragraph("Phase", styles["cell_header"]),
        Paragraph("Trial / Dosage", styles["cell_header"]),
    ]
    rows = [header]
    for ep_key, ep_label in ENDPOINT_LABELS.items():
        ep = stats["endpoints"].get(ep_key, {})
        weight_pct = f"{int(ENDPOINT_WEIGHTS[ep_key] * 100)}%"
        if ep.get("score") is not None:
            sc = ep["score"]
            sc_color = _score_color(sc)
            score_style = ParagraphStyle("EPS", parent=styles["cell"], textColor=sc_color, fontName="Helvetica-Bold")
            rows.append([
                Paragraph(ep_label, ParagraphStyle("EPL", parent=styles["cell"], alignment=TA_LEFT)),
                Paragraph(weight_pct, styles["cell"]),
                Paragraph(f"{ep['raw_value']}%", styles["cell"]),
                Paragraph(f"{ep['adj_value']}%", styles["cell"]),
                Paragraph(f"{sc}/5", score_style),
                Paragraph(f"P{ep['phase_used']}", styles["cell"]),
                Paragraph(f"{ep['trial_id']} / {ep['dosage']}", ParagraphStyle("EPT", parent=styles["cell"], alignment=TA_LEFT, fontSize=7)),
            ])
        else:
            rows.append([
                Paragraph(ep_label, ParagraphStyle("EPL2", parent=styles["cell"], alignment=TA_LEFT)),
                Paragraph(weight_pct, styles["cell"]),
                Paragraph("N/A", styles["cell"]),
                Paragraph("N/A", styles["cell"]),
                Paragraph("N/A", styles["cell"]),
                Paragraph("—", styles["cell"]),
                Paragraph("No valid data", ParagraphStyle("EPT2", parent=styles["cell"], alignment=TA_LEFT, fontSize=7)),
            ])

    tbl = Table(rows, colWidths=[1.1*inch, 0.5*inch, 0.7*inch, 0.7*inch, 0.5*inch, 0.5*inch, 2.7*inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        *[("BACKGROUND", (0, i), (-1, i), LIGHT_BLUE_BG if i % 2 == 0 else WHITE)
          for i in range(1, len(rows))],
    ]))
    return tbl


# ── Single-molecule report builder ────────────────────────────────────────────

def build_single_molecule_report(stats: dict, narrative: dict, output_path: str):
    """Build and save a detailed PDF report for one molecule."""
    styles = build_styles()
    story = []

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
        title=f"{REPORT_TITLE} — {stats['molecule_name']}",
        author="Clinical Efficacy Scorer",
    )

    # ── Title block ───────────────────────────────────────────────────────
    story.append(Paragraph(REPORT_TITLE, styles["title"]))
    story.append(Paragraph(stats["molecule_name"], styles["molecule_name_title"]))
    score_display = f"{stats['efficacy_score']}/5" if stats['efficacy_score'] is not None else "N/A"
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%B %d, %Y')}  •  Score: {score_display} ({stats['score_label']})  •  {stats['total_trials']} trials",
        styles["subtitle"],
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=DARK_BLUE, spaceAfter=12))

    # ── Executive Summary ─────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
    story.append(Paragraph("Executive Summary", styles["h2"]))
    exec_summary = narrative.get("executive_summary", [])
    if isinstance(exec_summary, list):
        _render_bullets(exec_summary, styles, story)
    elif isinstance(exec_summary, str):
        story.append(Paragraph(exec_summary, styles["body"]))
    story.append(Spacer(1, 6))

    # ── Endpoint Performance Overview ─────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
    story.append(Paragraph("Endpoint Performance Overview", styles["h2"]))

    # Endpoint summary table
    story.append(_endpoint_summary_table(stats, styles))
    story.append(Spacer(1, 8))

    ep_overview = narrative.get("endpoint_overview", {})
    if isinstance(ep_overview, dict):
        for ep_key, ep_label in ENDPOINT_LABELS.items():
            content = ep_overview.get(ep_key, "")
            if content:
                story.append(Paragraph(f"<b>{ep_label}:</b>", styles["section_label"]))
                story.append(Paragraph(content, styles["body"]))
                story.append(Spacer(1, 3))
    elif isinstance(ep_overview, str):
        story.append(Paragraph(ep_overview, styles["body"]))
    story.append(Spacer(1, 6))

    # ── Scoring Methodology & Rationale ───────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
    story.append(Paragraph("Scoring Methodology &amp; Rationale", styles["h2"]))

    scoring = narrative.get("scoring_rationale", {})
    if isinstance(scoring, dict):
        methodology = scoring.get("methodology", "")
        if methodology:
            story.append(Paragraph(f"<b>Methodology:</b> {methodology}", styles["body"]))
            story.append(Spacer(1, 4))
        calculation = scoring.get("calculation", "")
        if calculation:
            story.append(Paragraph(f"<b>Calculation:</b> {calculation}", styles["body"]))
            story.append(Spacer(1, 4))
        justification = scoring.get("justification", "")
        if justification:
            story.append(Paragraph(f"<b>Justification:</b> {justification}", styles["body"]))
    elif isinstance(scoring, str):
        story.append(Paragraph(scoring, styles["body"]))
    story.append(Spacer(1, 6))

    # ── Strategic Implications ────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
    story.append(Paragraph("Strategic Implications", styles["h2"]))

    strategic = narrative.get("strategic_implications", {})
    if isinstance(strategic, dict):
        comp_pos = strategic.get("competitive_positioning", "")
        if comp_pos:
            story.append(Paragraph(f"<b>Competitive Positioning:</b> {comp_pos}", styles["body"]))
            story.append(Spacer(1, 4))
        reg_comm = strategic.get("regulatory_commercial", "")
        if reg_comm:
            story.append(Paragraph(f"<b>Regulatory &amp; Commercial:</b> {reg_comm}", styles["body"]))
            story.append(Spacer(1, 4))
        portfolio = strategic.get("portfolio_value", "")
        if portfolio:
            story.append(Paragraph(f"<b>Portfolio Value:</b> {portfolio}", styles["body"]))
    elif isinstance(strategic, str):
        story.append(Paragraph(strategic, styles["body"]))
    story.append(Spacer(1, 8))

    # ── Methodology note ──────────────────────────────────────────────────
    methodology_note = narrative.get("score_methodology_note", "")
    if methodology_note:
        story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
        story.append(Paragraph("About the Clinical Efficacy Score", styles["h2"]))
        story.append(Paragraph(methodology_note, styles["methodology_note"]))

    # ── Scoring reference table ───────────────────────────────────────────
    story.append(Paragraph("Clinical Efficacy Scoring Reference", styles["h2"]))
    story.append(_scoring_framework_table(styles))
    story.append(Spacer(1, 8))

    legend_text = "  |  ".join(f"{k} = {v}" for k, v in SCORE_LABEL.items())
    story.append(Paragraph(
        f"<b>Score Legend:</b>  {legend_text}",
        ParagraphStyle("Legend", parent=styles["body"], fontSize=8, textColor=LIGHT_GRAY),
    ))

    # ── Footer ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC"), spaceBefore=14))
    story.append(Paragraph(
        "This report was auto-generated from Clinical Efficacy Scorer output "
        "using Gemini for narrative analysis. For internal use only.",
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

def generate_efficacy_report(
    molecules: list[str] | None = None,
    outdir: str | None = None,
) -> list[str]:
    if not API_KEY:
        print("[Efficacy Report] GEMINI_API_KEY not set — skipping report generation.")
        return []

    out_root = Path(outdir) if outdir else Path(".")
    out_root.mkdir(parents=True, exist_ok=True)

    molecule_data = load_from_bigquery(molecules)
    if not molecule_data:
        print("[Efficacy Report] No data found — skipping.")
        return []

    output_paths = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for molecule_name, trials_df in molecule_data.items():
        print(f"\nProcessing: {molecule_name}")

        stats = extract_molecule_stats(molecule_name, trials_df)
        stats = enrich_molecule_data(stats)

        print("  Generating narrative with Gemini...")
        narrative = generate_efficacy_narrative(stats)

        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", molecule_name)
        output_path = str(out_root / f"efficacy_report_{safe_name}_{ts}.pdf")

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
        description="Generate one Clinical Efficacy PDF per molecule from BigQuery data."
    )
    parser.add_argument("--molecule", "-m", default=None,
        help="Comma-separated molecule name(s). Omit for all.")
    parser.add_argument("--outdir", "-o", default=None,
        help="Output directory for PDFs (default: current directory).")
    args = parser.parse_args()

    mols = None
    if args.molecule:
        mols = [m.strip() for m in args.molecule.split(",") if m.strip()]

    generate_efficacy_report(molecules=mols, outdir=args.outdir)


if __name__ == "__main__":
    main()
