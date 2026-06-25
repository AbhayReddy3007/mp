"""
generate_moa_report.py
──────────────────────
Reads scored data from BigQuery `moa_innovation_table` and generates one
professional PDF report **per drug** using Gemini for narrative generation.

Only the LATEST row per drug (by created_at) is used.

Report structure (single-drug, business-facing):
  - Drug Name (top)
  - MoA Innovation Analysis (subtitle)
  - Executive Summary
  - Headline
  - Key Insights with justification
  - Bottom-line implication
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

# ── Colors ────────────────────────────────────────────────────────────────────
DARK_BLUE = colors.HexColor("#1F3864")
LIGHT_BLUE_BG = colors.HexColor("#E8EDF3")
ACCENT_BLUE = colors.HexColor("#2E5FA3")
WHITE = colors.white
LIGHT_GRAY = colors.HexColor("#666666")
DIVIDER_COLOR = colors.HexColor("#D0D7E3")
BODY_TEXT_COLOR = colors.HexColor("#333333")

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


# ── Gemini helper ─────────────────────────────────────────────────────────────

def call_gemini(prompt: str) -> str:
    client = genai_client.Client(api_key=API_KEY)
    config = types.GenerateContentConfig(temperature=0.3)
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
    print("[Auth] No service account file found — using Application Default Credentials.")
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
    """Extract all relevant fields for a single drug row."""
    def safe(val, fallback="N/A"):
        return str(val) if pd.notna(val) else fallback

    score_raw = row.get("score")
    score_int = None
    try:
        score_int = int(float(score_raw))
    except (ValueError, TypeError):
        pass

    return {
        "drug_name": safe(row.get("drug_name")),
        "indication": safe(row.get("indication")),
        "mechanism": safe(row.get("mechanism_statement", ""))[:300],
        "classification": safe(row.get("moa_classification")),
        "score": score_raw,
        "score_int": score_int,
        "score_label": SCORE_LABEL.get(score_int, "N/A") if score_int else "N/A",
        "guardrail": safe(row.get("guardrail")),
        "confidence_tier": safe(row.get("confidence_tier")),
        "confidence_score": row.get("confidence_score"),
        "analysis_date": safe(row.get("analysis_date")),
    }


# ── LLM narrative (single drug) ───────────────────────────────────────────────

def generate_drug_narrative(drug_stats: dict) -> dict:
    """
    Generate a business-facing analytical report narrative for a single drug.
    Returns a dict with: executive_summary, headline, key_insights,
    score_reference, bottom_line, score_methodology_note.
    """
    prompt = f"""You are a senior pharmaceutical business analyst preparing an executive report
on the Mechanism of Action (MoA) Innovation dimension for a single drug.

DRUG DATA:
- Drug name: {drug_stats['drug_name']}
- Indication: {drug_stats['indication']}
- Mechanism summary: {drug_stats['mechanism']}
- MoA classification: {drug_stats['classification']}
- MoA score: {drug_stats['score']} / 5  ({drug_stats['score_label']})
- Guardrail outcome: {drug_stats['guardrail']}
- Confidence tier: {drug_stats['confidence_tier']}
- Analysis date: {drug_stats['analysis_date']}

Your task is to generate a business-facing analytical report section that helps senior
decision-makers quickly understand what this drug's MoA innovation means for the business.
Follow these rules strictly:

1. EXECUTIVE SUMMARY (appears first in the report)
   - Write a 3–5 sentence paragraph that synthesises ALL available data about this drug.
   - Cover the drug's mechanism, indication, classification, score, guardrail outcome,
     and confidence tier in plain language.
   - This should give a busy executive a complete picture before reading any further.

2. FOCUS ON CRITICAL INSIGHTS ONLY
   - Identify the 3 most important findings that materially affect this drug's attractiveness,
     competitive position, or risk profile.
   - Do NOT list all data points. Only surface what truly matters for decision-making.

3. STRONG JUSTIFICATION
   - For every insight, explain WHY it matters using simple cause → impact reasoning.
   - Link each observation to its business implication: revenue potential, competitive risk,
     regulatory complexity, or execution feasibility.

4. SCORE REFERENCE (minimal)
   - Mention the score only once, briefly, as a reference point
     (e.g., "This drug's MoA innovation is rated Strong at 4/5").
   - Do NOT explain how the score was calculated — save that for the very end.

5. NO TECHNICAL JARGON
   - Avoid internal model terms, scoring logic details, or evaluation framework language.
   - Write in clear, natural business language a non-scientist executive would understand.

6. EXECUTIVE-FRIENDLY FORMAT
   - Keep it crisp, confident, and insight-driven.
   - Every statement must add insight or implication — no generic filler.

Respond ONLY with a valid JSON object (no markdown fences, no extra text):
{{
  "executive_summary": "<3–5 sentence paragraph synthesising all available data on this drug, covering mechanism, indication, classification, score, guardrail, and confidence — in plain business language>",
  "headline": "<One crisp sentence capturing the single most important business implication of this drug's MoA innovation>",
  "key_insights": [
    {{
      "insight": "<Concise insight statement about this drug's MoA>",
      "justification": "<Why this matters: cause → business impact, 2–3 sentences>"
    }},
    {{
      "insight": "<Second most important insight>",
      "justification": "<Why this matters: cause → business impact, 2–3 sentences>"
    }},
    {{
      "insight": "<Third most important insight>",
      "justification": "<Why this matters: cause → business impact, 2–3 sentences>"
    }}
  ],
  "score_reference": "<One sentence referencing this drug's MoA score level without methodology details>",
  "bottom_line": "<2–3 sentences telling the decision-maker what to infer and what action to consider>",
  "score_methodology_note": "<2–3 plain-language sentences at the end explaining in general terms how the MoA score is determined — high-level, no jargon>"
}}
"""
    text = call_gemini(prompt)
    result = _extract_json(text)
    if result:
        return result

    # Fallback
    return {
        "executive_summary": (
            f"{drug_stats['drug_name']} is being evaluated for {drug_stats['indication']}. "
            f"It is classified as {drug_stats['classification']} with an MoA score of "
            f"{drug_stats['score']}/5 ({drug_stats['score_label']}). "
            f"The guardrail outcome is {drug_stats['guardrail']} at a "
            f"{drug_stats['confidence_tier']} confidence level."
        ),
        "headline": f"MoA innovation analysis for {drug_stats['drug_name']} — see details below.",
        "key_insights": [
            {"insight": "See the scoring reference table for detailed data.", "justification": ""},
        ],
        "score_reference": f"MoA score: {drug_stats['score']} / 5 ({drug_stats['score_label']}).",
        "bottom_line": "Refer to the scoring reference for individual assessment details.",
        "score_methodology_note": "Scores reflect the novelty and clinical validation of the drug's mechanism of action.",
    }


# ── Style helpers ─────────────────────────────────────────────────────────────

def build_styles():
    base = getSampleStyleSheet()

    # Shared body style — used consistently across ALL narrative paragraph sections
    body = ParagraphStyle(
        "Body", parent=base["Normal"],
        fontSize=10, leading=15,
        textColor=BODY_TEXT_COLOR,
        fontName="Helvetica",
        spaceAfter=6,
        alignment=TA_JUSTIFY,
    )

    return {
        "drug_name_title": ParagraphStyle(
            "DrugNameTitle", parent=base["Normal"],
            fontSize=20, leading=26, textColor=DARK_BLUE,
            alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=4,
        ),
        "report_subtitle": ParagraphStyle(
            "ReportSubtitle", parent=base["Normal"],
            fontSize=13, leading=18, textColor=ACCENT_BLUE,
            alignment=TA_CENTER, fontName="Helvetica-Bold", spaceAfter=14,
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
        # All body-level paragraph styles below share the same look
        "body": body,
        "executive_summary": body,
        "insight_justification": ParagraphStyle(
            "InsightJustification", parent=body,
            spaceAfter=8, leftIndent=16,
        ),
        "score_ref": ParagraphStyle(
            "ScoreRef", parent=body,
            textColor=LIGHT_GRAY, fontSize=9, leading=13,
        ),
        "bottom_line": body,
        "methodology_note": ParagraphStyle(
            "MethodologyNote", parent=body,
            fontSize=8, leading=12, textColor=LIGHT_GRAY,
        ),
        "footer": ParagraphStyle(
            "Footer", parent=base["Normal"],
            fontSize=7, leading=10, textColor=colors.HexColor("#999999"),
            fontName="Helvetica", alignment=TA_CENTER, spaceBefore=10,
        ),
        "cell_header": ParagraphStyle(
            "CellHeader", parent=base["Normal"],
            fontSize=8, leading=11, textColor=WHITE,
            fontName="Helvetica-Bold", alignment=TA_CENTER,
        ),
        "cell": ParagraphStyle(
            "Cell", parent=base["Normal"],
            fontSize=8, leading=11, textColor=BODY_TEXT_COLOR,
            fontName="Helvetica", alignment=TA_CENTER,
        ),
    }


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


def _render_key_insights(insights: list, styles: dict, story: list):
    """
    Render key insights as bullet points — no 'Insight N:' labels.
    Each insight is a bold bullet; justification follows indented below it.
    """
    for item in insights:
        insight_text = item.get("insight", "")
        justification = item.get("justification", "")

        # Bold bullet for the insight statement — explicit font color to stay #333333
        story.append(Paragraph(
            f'<font color="#333333">&#8226; <b>{insight_text}</b></font>',
            styles["body"],
        ))
        if justification:
            story.append(Paragraph(
                f"&#8594; {justification}",
                styles["insight_justification"],
            ))


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
            Paragraph(sc,   styles["cell"]),
            Paragraph(lbl,  styles["cell"]),
            Paragraph(desc, ParagraphStyle(
                f"FWDesc_{sc}", parent=styles["cell"], alignment=TA_LEFT,
            )),
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
    """Build and save a PDF report for one drug."""
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

    # ── Title block: Drug Name first, then report title ───────────────────────
    story.append(Paragraph(drug_stats["drug_name"], styles["drug_name_title"]))
    story.append(Paragraph(REPORT_TITLE, styles["report_subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=DARK_BLUE, spaceAfter=12))

    # ── Executive Summary ─────────────────────────────────────────────────────
    executive_summary = narrative.get("executive_summary", "")
    if executive_summary:
        story.append(Paragraph("Executive Summary", styles["h2"]))
        story.append(Paragraph(executive_summary, styles["executive_summary"]))
        story.append(Spacer(1, 8))

    # ── Headline banner ───────────────────────────────────────────────────────
    headline = narrative.get("headline", "")
    if headline:
        _headline_box(headline, styles, story)

    story.append(Spacer(1, 4))

    # ── Key Insights ──────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
    story.append(Paragraph("Key Insights", styles["h2"]))
    _render_key_insights(narrative.get("key_insights", []), styles, story)
    story.append(Spacer(1, 6))

    # ── Score reference ───────────────────────────────────────────────────────
    score_ref = narrative.get("score_reference", "")
    if score_ref:
        story.append(Paragraph(score_ref, styles["score_ref"]))
        story.append(Spacer(1, 4))

    # ── Bottom-line implication ───────────────────────────────────────────────
    bottom_line = narrative.get("bottom_line", "")
    if bottom_line:
        story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
        story.append(Paragraph("Bottom-Line Implication", styles["h2"]))
        story.append(Paragraph(bottom_line, styles["bottom_line"]))
        story.append(Spacer(1, 10))

    # ── Methodology note (end of document) ───────────────────────────────────
    methodology_note = narrative.get("score_methodology_note", "")
    if methodology_note:
        story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=6))
        story.append(Paragraph("About the MoA Innovation Score", styles["h2"]))
        story.append(Paragraph(methodology_note, styles["methodology_note"]))

    # ── Scoring reference table ───────────────────────────────────────────────
    story.append(Paragraph("MoA Innovation Scoring Reference", styles["h2"]))
    story.append(_scoring_framework_table(styles))
    story.append(Spacer(1, 8))

    legend_text = "  |  ".join(f"{k} = {v}" for k, v in SCORE_LABEL.items())
    story.append(Paragraph(
        f"<b>Score Legend:</b>  {legend_text}",
        ParagraphStyle(
            "Legend", parent=styles["body"],
            fontSize=8, textColor=LIGHT_GRAY,
        ),
    ))

    # ── Footer ────────────────────────────────────────────────────────────────
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
