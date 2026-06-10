"""
generate_moa_report.py
──────────────────────
Reads scored data from BigQuery `moa_innovation_table` and generates a
professional PDF report using Gemini for narrative generation.

Only the LATEST row per drug (by created_at) is used.

Usage:
    python market_potential/generate_moa_report.py
    python market_potential/generate_moa_report.py --molecule Semaglutide
    python market_potential/generate_moa_report.py --output moa_report.pdf
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
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.platypus.flowables import HRFlowable

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

REPORT_TITLE = "MoA Innovation Scoring Analysis"

# ── Colors ────────────────────────────────────────────────────────────────────
DARK_BLUE = colors.HexColor("#1F3864")
LIGHT_BLUE_BG = colors.HexColor("#E8EDF3")
WHITE = colors.white
LIGHT_GRAY = colors.HexColor("#666666")
VERY_LIGHT_GRAY = colors.HexColor("#999999")

SCORE_COLORS = {
    5: colors.HexColor("#008000"),  # Green  – Exceptional
    4: colors.HexColor("#4CAF50"),  # Light green – Strong
    3: colors.HexColor("#CC9900"),  # Amber – Moderate
    2: colors.HexColor("#E65100"),  # Orange-red – Weak
    1: colors.HexColor("#CC0000"),  # Red – Poor
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
    return None


def _bq_client():
    credentials = _get_credentials()
    return bigquery.Client(project=BQ_PROJECT_ID, credentials=credentials, location=BQ_LOCATION)


def load_from_bigquery(molecule: str = None) -> pd.DataFrame:
    """Load LATEST row per drug from moa_innovation_table."""
    client = _bq_client()

    molecule_filter = ""
    if molecule:
        molecule_filter = f"AND drug_name = '{molecule}'"

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


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_statistics(df: pd.DataFrame) -> dict:
    stats = {
        "total_drugs": len(df),
        "drugs": [],
        "avg_score": None,
        "score_distribution": {},
        "classification_distribution": {},
        "guardrail_pass": 0,
        "guardrail_fail": 0,
        "per_drug_stats": {},
    }
    if df.empty:
        return stats

    stats["drugs"] = df["drug_name"].dropna().unique().tolist()

    if "score" in df.columns:
        numeric = pd.to_numeric(df["score"], errors="coerce").dropna()
        if len(numeric):
            stats["avg_score"] = float(round(numeric.mean(), 2))
            for s in range(1, 6):
                stats["score_distribution"][s] = int((numeric == s).sum())

    if "moa_classification" in df.columns:
        for cls_val, cnt in df["moa_classification"].value_counts().items():
            stats["classification_distribution"][str(cls_val)] = int(cnt)

    if "guardrail" in df.columns:
        stats["guardrail_pass"] = int((df["guardrail"] == "PASS").sum())
        stats["guardrail_fail"] = int((df["guardrail"] == "FAIL").sum())

    for _, row in df.iterrows():
        drug = str(row.get("drug_name", ""))
        stats["per_drug_stats"][drug] = {
            "score": row.get("score"),
            "classification": str(row.get("moa_classification", "N/A")),
            "guardrail": str(row.get("guardrail", "N/A")),
            "confidence_tier": str(row.get("confidence_tier", "N/A")),
            "confidence_score": row.get("confidence_score"),
            "indication": str(row.get("indication", "N/A")),
        }

    return stats


# ── LLM narrative ─────────────────────────────────────────────────────────────

def generate_executive_summary(stats: dict, df: pd.DataFrame) -> dict:
    drug_rows = []
    for _, r in df.iterrows():
        drug_rows.append({
            "drug": str(r.get("drug_name", "")),
            "indication": str(r.get("indication", "")),
            "mechanism": str(r.get("mechanism_statement", ""))[:300],
            "classification": str(r.get("moa_classification", "")),
            "score": r.get("score"),
            "guardrail": str(r.get("guardrail", "")),
            "confidence": r.get("confidence_score"),
            "confidence_tier": str(r.get("confidence_tier", "")),
        })

    prompt = f"""You are a senior pharmaceutical analyst writing a MoA Innovation report.
Based on the data below, produce a thorough analytical report. Be specific — reference
drug names, classifications, scores, and indications.

PORTFOLIO STATISTICS:
- Total drugs assessed: {stats['total_drugs']}
- Drugs: {', '.join(stats['drugs'])}
- Average MoA score: {stats['avg_score']} (1=Poor, 5=Exceptional)
- Score distribution: {json.dumps(stats['score_distribution'])}
- Classification distribution: {json.dumps(stats['classification_distribution'])}
- Guardrail: {stats['guardrail_pass']} PASS, {stats['guardrail_fail']} FAIL

DRUG-LEVEL DATA (JSON):
{json.dumps(drug_rows, indent=1)}

Respond ONLY with a valid JSON object (no markdown fences):
{{
  "executive_summary": "<5-8 sentences overview of the MoA innovation landscape across the assessed drugs>",
  "key_findings": ["<finding 1 with drug names and scores>", "<finding 2>", "<finding 3>", "<finding 4>"],
  "classification_analysis": "<4-6 sentences analysing the classification distribution — which drugs are FIC/BIC/Me-too and why>",
  "risk_highlights": "<3-5 sentences on drugs with low scores or FAIL guardrails>",
  "strength_highlights": "<3-5 sentences on drugs with high scores and strong innovation>",
  "per_drug_narratives": {{
    "<drug_name>": "<3-5 sentence analysis of this drug's MoA innovation, classification rationale, and strategic implications>"
  }}
}}
"""
    text = call_gemini(prompt)
    result = _extract_json(text)
    if result:
        return result
    return {
        "executive_summary": "Analysis complete. See tables below for details.",
        "key_findings": ["See detailed tables."],
        "classification_analysis": "See classification table.",
        "risk_highlights": "Refer to score tables.",
        "strength_highlights": "Refer to score tables.",
        "per_drug_narratives": {},
    }


# ── Style helpers ─────────────────────────────────────────────────────────────

def build_styles():
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Normal"],
            fontSize=20,
            leading=24,
            textColor=DARK_BLUE,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
            textColor=LIGHT_GRAY,
            alignment=TA_CENTER,
            fontName="Helvetica",
            spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Normal"],
            fontSize=13,
            leading=16,
            textColor=DARK_BLUE,
            fontName="Helvetica-Bold",
            spaceBefore=14,
            spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=base["Normal"],
            fontSize=11,
            leading=14,
            textColor=DARK_BLUE,
            fontName="Helvetica-Bold",
            spaceBefore=10,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#333333"),
            fontName="Helvetica",
            spaceAfter=4,
            alignment=TA_JUSTIFY,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#333333"),
            fontName="Helvetica",
            spaceAfter=3,
            leftIndent=16,
            bulletIndent=4,
        ),
        "stat": ParagraphStyle(
            "Stat",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
            textColor=LIGHT_GRAY,
            fontName="Helvetica-Oblique",
            spaceAfter=4,
        ),
        "footer": ParagraphStyle(
            "Footer",
            parent=base["Normal"],
            fontSize=7,
            leading=10,
            textColor=colors.HexColor("#999999"),
            fontName="Helvetica-Oblique",
            alignment=TA_CENTER,
            spaceBefore=10,
        ),
        "cell": ParagraphStyle(
            "Cell",
            parent=base["Normal"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#333333"),
            fontName="Helvetica",
            alignment=TA_CENTER,
        ),
        "cell_header": ParagraphStyle(
            "CellHeader",
            parent=base["Normal"],
            fontSize=8,
            leading=11,
            textColor=WHITE,
            fontName="Helvetica-Bold",
            alignment=TA_CENTER,
        ),
        "cell_label": ParagraphStyle(
            "CellLabel",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#333333"),
            fontName="Helvetica-Bold",
        ),
        "cell_value": ParagraphStyle(
            "CellValue",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#333333"),
            fontName="Helvetica",
        ),
    }
    return styles


def _score_color(score_val):
    try:
        s = int(float(score_val))
        return SCORE_COLORS.get(s, colors.black)
    except (ValueError, TypeError):
        return colors.black


def _classification_color(cls_str):
    cls_lower = str(cls_str).lower()
    for key, color in CLASSIFICATION_COLORS.items():
        if key in cls_lower:
            return color
    return colors.black


# ── Report builder ────────────────────────────────────────────────────────────

def build_report(df: pd.DataFrame, output_path: str):
    if df.empty:
        print("ERROR: No MoA data to report.")
        return

    stats = compute_statistics(df)

    print("Generating narrative with Gemini...")
    narrative = generate_executive_summary(stats, df)

    styles = build_styles()
    story = []

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        topMargin=0.8 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        title=REPORT_TITLE,
        author="MoA Innovation Scorer",
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    story.append(Paragraph(REPORT_TITLE, styles["title"]))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%B %d, %Y')}  •  "
        f"{stats['total_drugs']} Drug(s) Assessed",
        styles["subtitle"],
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=DARK_BLUE, spaceAfter=12))

    # ── Executive Summary ─────────────────────────────────────────────────────
    story.append(Paragraph("Executive Summary", styles["h2"]))
    story.append(Paragraph(narrative.get("executive_summary", ""), styles["body"]))
    story.append(Spacer(1, 6))

    # ── Portfolio Overview ────────────────────────────────────────────────────
    story.append(Paragraph("Portfolio Overview", styles["h2"]))

    dist_parts = [
        f"{SCORE_LABEL[s]}: {stats['score_distribution'].get(s, 0)}"
        for s in [5, 4, 3, 2, 1]
        if stats["score_distribution"].get(s, 0) > 0
    ]
    cls_parts = [
        f"{cls}: {cnt}"
        for cls, cnt in stats["classification_distribution"].items()
    ]

    overview_data = [
        [Paragraph("Total Drugs Assessed", styles["cell_label"]),
         Paragraph(str(stats["total_drugs"]), styles["cell_value"])],
        [Paragraph("Drugs Covered", styles["cell_label"]),
         Paragraph(", ".join(stats["drugs"]), styles["cell_value"])],
        [Paragraph("Average MoA Score", styles["cell_label"]),
         Paragraph(f"{stats['avg_score']} / 5" if stats["avg_score"] else "N/A", styles["cell_value"])],
        [Paragraph("Guardrail Summary", styles["cell_label"]),
         Paragraph(f"{stats['guardrail_pass']} PASS  |  {stats['guardrail_fail']} FAIL", styles["cell_value"])],
    ]
    if dist_parts:
        overview_data.append([
            Paragraph("Score Distribution", styles["cell_label"]),
            Paragraph(";  ".join(dist_parts), styles["cell_value"]),
        ])
    if cls_parts:
        overview_data.append([
            Paragraph("Classification Distribution", styles["cell_label"]),
            Paragraph(";  ".join(cls_parts), styles["cell_value"]),
        ])

    ov_table = Table(overview_data, colWidths=[2.5 * inch, 4.2 * inch])
    ov_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE_BG),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(ov_table)
    story.append(Spacer(1, 12))

    # ── Key Findings ──────────────────────────────────────────────────────────
    story.append(Paragraph("Key Findings", styles["h2"]))
    for finding in narrative.get("key_findings", []):
        story.append(Paragraph(f"• {finding}", styles["bullet"]))
    story.append(Spacer(1, 8))

    # ── Drug Score Summary Table ──────────────────────────────────────────────
    story.append(Paragraph("MoA Innovation Score Summary", styles["h2"]))

    display_cols = ["Drug", "Indication", "Classification", "Score", "Guardrail", "Confidence", "Date"]
    data_cols = ["drug_name", "indication", "moa_classification", "score", "guardrail",
                 "confidence_tier", "analysis_date"]

    header_row = [Paragraph(c, styles["cell_header"]) for c in display_cols]
    table_data = [header_row]

    col_widths = [1.1 * inch, 1.2 * inch, 1.15 * inch, 0.5 * inch, 0.7 * inch, 0.85 * inch, 0.9 * inch]

    score_cell_style_overrides = []  # (row_idx, col_idx, color)

    for row_i, (_, row) in enumerate(df.iterrows(), start=1):
        cells = []
        for col_i, col in enumerate(data_cols):
            val = row.get(col, "")
            val_str = str(val) if pd.notna(val) else "N/A"

            if col == "score":
                color = _score_color(val)
                cell_style = ParagraphStyle(
                    f"ScoreCell_{row_i}",
                    parent=styles["cell"],
                    textColor=color,
                    fontName="Helvetica-Bold",
                )
                cells.append(Paragraph(val_str, cell_style))
            elif col == "moa_classification":
                color = _classification_color(val_str)
                cell_style = ParagraphStyle(
                    f"ClsCell_{row_i}",
                    parent=styles["cell"],
                    textColor=color,
                    fontName="Helvetica-Bold",
                )
                cells.append(Paragraph(val_str, cell_style))
            elif col == "guardrail":
                if val_str == "PASS":
                    color = GUARDRAIL_PASS_COLOR
                elif val_str == "FAIL":
                    color = GUARDRAIL_FAIL_COLOR
                else:
                    color = colors.black
                cell_style = ParagraphStyle(
                    f"GRCell_{row_i}",
                    parent=styles["cell"],
                    textColor=color,
                    fontName="Helvetica-Bold",
                )
                cells.append(Paragraph(val_str, cell_style))
            else:
                cells.append(Paragraph(val_str, styles["cell"]))
        table_data.append(cells)

    score_table = Table(table_data, colWidths=col_widths, repeatRows=1)

    row_bg_commands = []
    for i in range(1, len(table_data)):
        bg = LIGHT_BLUE_BG if i % 2 == 0 else WHITE
        row_bg_commands.append(("BACKGROUND", (0, i), (-1, i), bg))

    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        *row_bg_commands,
    ]))
    story.append(score_table)
    story.append(Spacer(1, 12))

    # ── Classification Analysis ───────────────────────────────────────────────
    cls_text = narrative.get("classification_analysis", "")
    if cls_text:
        story.append(Paragraph("Classification Analysis", styles["h2"]))
        story.append(Paragraph(cls_text, styles["body"]))
        story.append(Spacer(1, 8))

    # ── Risk & Strength Highlights ────────────────────────────────────────────
    story.append(Paragraph("Risk &amp; Strength Analysis", styles["h2"]))

    story.append(Paragraph(
        '<font color="#CC0000"><b>Weak / At-Risk Drugs</b></font>',
        styles["h3"],
    ))
    story.append(Paragraph(narrative.get("risk_highlights", "N/A"), styles["body"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        '<font color="#008000"><b>Strongest Innovators</b></font>',
        styles["h3"],
    ))
    story.append(Paragraph(narrative.get("strength_highlights", "N/A"), styles["body"]))
    story.append(Spacer(1, 12))

    # ── Per-Drug Breakdown ────────────────────────────────────────────────────
    per_drug = narrative.get("per_drug_narratives", {})
    if per_drug:
        story.append(Paragraph("Drug-Level Analysis", styles["h2"]))

        for drug_name, drug_narrative in per_drug.items():
            drug_stat = stats.get("per_drug_stats", {}).get(drug_name, {})

            block = []
            block.append(Paragraph(drug_name, styles["h3"]))

            if drug_stat:
                block.append(Paragraph(
                    f"Score: {drug_stat.get('score', 'N/A')}/5  |  "
                    f"Classification: {drug_stat.get('classification', 'N/A')}  |  "
                    f"Guardrail: {drug_stat.get('guardrail', 'N/A')}  |  "
                    f"Confidence: {drug_stat.get('confidence_tier', 'N/A')}",
                    styles["stat"],
                ))

            block.append(Paragraph(drug_narrative, styles["body"]))
            story.append(KeepTogether(block))
            story.append(Spacer(1, 6))

    # ── Scoring Framework ─────────────────────────────────────────────────────
    story.append(Paragraph("MoA Innovation Scoring Framework", styles["h2"]))

    framework = [
        ("5", "Exceptional", "True First-in-Class: novel mechanism, strong rationale, class-creating potential"),
        ("4", "Strong", "Validated class with clearly superior mechanism differentiation (Best-in-Class)"),
        ("3", "Moderate", "Validated class with limited innovation (Me-too / Fast Follower)"),
        ("2", "Weak", "Older or strategically outdated mechanism"),
        ("1", "Poor", "Weak biological rationale or clinically invalidated mechanism"),
    ]

    fw_header = [
        Paragraph("Score", styles["cell_header"]),
        Paragraph("Label", styles["cell_header"]),
        Paragraph("Description", styles["cell_header"]),
    ]
    fw_data = [fw_header]
    for sc, lbl, desc in framework:
        fw_data.append([
            Paragraph(sc, ParagraphStyle("FWScore", parent=styles["cell"], fontName="Helvetica-Bold")),
            Paragraph(lbl, ParagraphStyle("FWLabel", parent=styles["cell"], fontName="Helvetica-Bold")),
            Paragraph(desc, ParagraphStyle("FWDesc", parent=styles["cell"], alignment=TA_LEFT)),
        ])

    fw_table = Table(fw_data, colWidths=[0.5 * inch, 1.2 * inch, 4.6 * inch])
    fw_row_bgs = []
    for i in range(1, len(fw_data)):
        bg = LIGHT_BLUE_BG if i % 2 == 0 else WHITE
        fw_row_bgs.append(("BACKGROUND", (0, i), (-1, i), bg))

    fw_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (1, -1), "CENTER"),
        ("ALIGN", (2, 1), (2, -1), "LEFT"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        *fw_row_bgs,
    ]))
    story.append(fw_table)
    story.append(Spacer(1, 10))

    # ── Score Legend ──────────────────────────────────────────────────────────
    legend_text = "  |  ".join(f"{k} = {v}" for k, v in SCORE_LABEL.items())
    story.append(Paragraph(
        f"<b>Score Legend:</b>  {legend_text}",
        ParagraphStyle("Legend", parent=styles["body"], fontSize=8, textColor=LIGHT_GRAY),
    ))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC"), spaceBefore=14))
    story.append(Paragraph(
        "This report was auto-generated from MoA Innovation Scorer output "
        "using Gemini for narrative analysis.",
        styles["footer"],
    ))

    doc.build(story)
    print(f"\n✅ MoA report saved → {output_path}")


# ── GCS Upload ────────────────────────────────────────────────────────────────

def upload_to_gcs(local_path: str, drug_names: list) -> list:
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError("google-cloud-storage required. pip install google-cloud-storage")

    credentials = _get_credentials()
    client = storage.Client(project=BQ_PROJECT_ID, credentials=credentials)
    bucket = client.bucket(GCS_BUCKET)
    gcs_uris = []

    for drug_name in drug_names:
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(drug_name))
        blob_name = f"{GCS_BASE_PATH}/{safe_name}/{GCS_FILE_NAME}"
        gcs_uri = f"gs://{GCS_BUCKET}/{blob_name}"
        print(f"  Uploading to GCS: {gcs_uri}")
        try:
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(
                local_path,
                content_type="application/pdf",
            )
            gcs_uris.append(gcs_uri)
        except Exception as e:
            print(f"  [ERROR] GCS upload failed for '{drug_name}': {e}")
            raise

    return gcs_uris


# ── Public entry point (for run_all.py) ───────────────────────────────────────

def generate_moa_report(molecule: str = None, output_path: str = None) -> str:
    """Generate MoA report. Returns the output file path."""
    if not API_KEY:
        print("[MoA Report] GEMINI_API_KEY not set — skipping report generation.")
        return ""

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"moa_innovation_report_{ts}.pdf"

    df = load_from_bigquery(molecule)
    if df.empty:
        print("[MoA Report] No data found — skipping.")
        return ""

    build_report(df, output_path)

    drug_names = df["drug_name"].dropna().unique().tolist()
    if drug_names:
        try:
            print(f"\nUploading MoA report to GCS for {len(drug_names)} drug(s)...")
            gcs_uris = upload_to_gcs(output_path, drug_names)
            for uri in gcs_uris:
                print(f"  ✅ {uri}")
        except Exception as e:
            print(f"  [WARN] GCS upload failed: {e}")

    return output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate MoA Innovation PDF report from BigQuery")
    parser.add_argument("--output", "-o", default=None, help="Output .pdf path")
    parser.add_argument("--molecule", default=None, help="Filter to a specific molecule")
    args = parser.parse_args()

    generate_moa_report(molecule=args.molecule, output_path=args.output)


if __name__ == "__main__":
    main()
