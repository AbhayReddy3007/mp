"""
generate_moa_report.py
──────────────────────
Reads scored data from BigQuery `moa_innovation_table` and generates a
professional PDF report using Gemini for narrative generation.

Only the LATEST row per drug (by created_at) is used.

Report structure follows a business-facing analytical format:
  - Headline (one-line dimension summary)
  - Key Insights with justification
  - Score (brief mention only)
  - Bottom-line implication

Usage:
    python generate_moa_report.py
    python generate_moa_report.py --molecule Semaglutide
    python generate_moa_report.py --output moa_report.pdf
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

REPORT_TITLE = "MoA Innovation Analysis"

# ── Colors ────────────────────────────────────────────────────────────────────
DARK_BLUE = colors.HexColor("#1F3864")
LIGHT_BLUE_BG = colors.HexColor("#E8EDF3")
ACCENT_BLUE = colors.HexColor("#2E5FA3")
WHITE = colors.white
LIGHT_GRAY = colors.HexColor("#666666")
VERY_LIGHT_GRAY = colors.HexColor("#999999")
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
    """
    Generate a business-facing analytical narrative using the structured
    insight report format:
      - Headline
      - Key Insights (with justification)
      - Score reference
      - Bottom-line implication
      - Per-drug breakdowns following the same format
    """
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

    score_label_map = {5: "Exceptional", 4: "Strong", 3: "Moderate", 2: "Weak", 1: "Poor"}

    prompt = f"""You are a senior pharmaceutical business analyst preparing an executive report
on the Mechanism of Action (MoA) Innovation dimension for a pharmaceutical portfolio.

Your task is to generate a business-facing analytical report that helps senior stakeholders
make informed decisions. Follow these rules strictly:

1. FOCUS ON CRITICAL INSIGHTS ONLY
   - Surface the 3 most important findings that materially affect product attractiveness,
     competitive risk, or market opportunity.
   - Do NOT list all data points. Only include what truly matters.

2. STRONG JUSTIFICATION
   - For every insight, explain WHY it matters using simple cause → impact reasoning.
   - Link each observation to its business implication (revenue potential, competitive risk,
     regulatory complexity, or execution feasibility).

3. SCORE REFERENCE (minimal)
   - Mention the overall portfolio score only briefly as a reference point
     (e.g., "This dimension is rated Moderate overall").
   - Do NOT explain how the score was calculated. Save that for the very end.

4. NO TECHNICAL JARGON
   - Avoid internal model terms, scoring logic details, or evaluation framework terminology.
   - Use clear, natural business language a non-scientist executive would understand.

5. EXECUTIVE-FRIENDLY FORMAT
   - Keep it crisp, confident, and insight-driven.
   - Every statement must add insight or implication — no generic filler.

6. STRICT LENGTH: The entire portfolio-level report (headline + insights + score +
   bottom-line) must fit within 2 printed pages. Per-drug narratives should each be
   3–5 sentences maximum.

PORTFOLIO DATA:
- Total drugs assessed: {stats['total_drugs']}
- Drugs: {', '.join(stats['drugs'])}
- Average MoA score: {stats['avg_score']} / 5  ({score_label_map.get(round(stats['avg_score']) if stats['avg_score'] else 0, 'N/A')} overall)
- Score distribution: {json.dumps({score_label_map.get(k, k): v for k, v in stats['score_distribution'].items() if v > 0})}
- Classification breakdown: {json.dumps(stats['classification_distribution'])}
- Guardrail outcomes: {stats['guardrail_pass']} PASS, {stats['guardrail_fail']} FAIL

DRUG-LEVEL DATA (JSON):
{json.dumps(drug_rows, indent=1)}

Respond ONLY with a valid JSON object (no markdown fences, no extra text):
{{
  "headline": "<One crisp sentence summarising the single most important implication of MoA innovation across this portfolio>",
  "key_insights": [
    {{
      "insight": "<Concise insight statement referencing specific drugs where relevant>",
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
  "score_reference": "<One sentence referencing the overall portfolio score level (e.g. Moderate at X.X/5) without methodology details>",
  "bottom_line": "<2–3 sentences telling the decision-maker what to infer and what action to consider based on this dimension>",
  "score_methodology_note": "<Brief, plain-language note (2–3 sentences max) at the end explaining in general terms how the overall MoA score was determined — keep it high-level and jargon-free>",
  "per_drug_narratives": {{
    "<drug_name>": {{
      "headline": "<One-line implication for this drug's MoA innovation>",
      "key_insights": [
        {{"insight": "<Key insight>", "justification": "<Why it matters, 1–2 sentences>"}},
        {{"insight": "<Second key insight>", "justification": "<Why it matters, 1–2 sentences>"}}
      ],
      "score_reference": "<Brief one-line score reference for this drug>",
      "bottom_line": "<1–2 sentences: what should the decision-maker infer about this drug?>"
    }}
  }}
}}
"""
    text = call_gemini(prompt)
    result = _extract_json(text)
    if result:
        return result

    # Fallback if JSON extraction fails
    return {
        "headline": "MoA innovation analysis complete — see details below.",
        "key_insights": [
            {"insight": "See tables below for detailed drug-level data.", "justification": ""},
        ],
        "score_reference": f"Portfolio average score: {stats['avg_score']} / 5.",
        "bottom_line": "Refer to the drug-level tables for individual assessment details.",
        "score_methodology_note": "Scores reflect the novelty and clinical validation of each drug's mechanism of action.",
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
            leading=26,
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
        "headline_box": ParagraphStyle(
            "HeadlineBox",
            parent=base["Normal"],
            fontSize=11,
            leading=16,
            textColor=WHITE,
            fontName="Helvetica-Bold",
            alignment=TA_LEFT,
            spaceAfter=0,
            leftIndent=0,
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
        "h3_drug": ParagraphStyle(
            "H3Drug",
            parent=base["Normal"],
            fontSize=11,
            leading=14,
            textColor=ACCENT_BLUE,
            fontName="Helvetica-Bold",
            spaceBefore=10,
            spaceAfter=2,
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
        "insight_label": ParagraphStyle(
            "InsightLabel",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            textColor=DARK_BLUE,
            fontName="Helvetica-Bold",
            spaceAfter=1,
            leftIndent=12,
        ),
        "insight_justification": ParagraphStyle(
            "InsightJustification",
            parent=base["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#444444"),
            fontName="Helvetica-Oblique",
            spaceAfter=6,
            leftIndent=24,
        ),
        "score_ref": ParagraphStyle(
            "ScoreRef",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
            textColor=LIGHT_GRAY,
            fontName="Helvetica-Oblique",
            spaceAfter=4,
        ),
        "bottom_line": ParagraphStyle(
            "BottomLine",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#1A1A1A"),
            fontName="Helvetica-Bold",
            spaceAfter=4,
            alignment=TA_JUSTIFY,
        ),
        "methodology_note": ParagraphStyle(
            "MethodologyNote",
            parent=base["Normal"],
            fontSize=8,
            leading=12,
            textColor=LIGHT_GRAY,
            fontName="Helvetica-Oblique",
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


def _headline_box(headline_text: str, styles: dict, story: list):
    """Render a dark-blue shaded headline banner."""
    headline_table = Table(
        [[Paragraph(f"&#9654; {headline_text}", styles["headline_box"])]],
        colWidths=[6.8 * inch],
    )
    headline_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARK_BLUE),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(headline_table)
    story.append(Spacer(1, 8))


def _render_key_insights(insights: list, styles: dict, story: list):
    """Render numbered key insights with justification."""
    for i, item in enumerate(insights, start=1):
        insight_text = item.get("insight", "")
        justification_text = item.get("justification", "")
        story.append(Paragraph(
            f"<b>Insight {i}:</b> {insight_text}",
            styles["insight_label"],
        ))
        if justification_text:
            story.append(Paragraph(
                f"→ {justification_text}",
                styles["insight_justification"],
            ))


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
        topMargin=0.75 * inch,
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

    # ── Portfolio Headline ────────────────────────────────────────────────────
    portfolio_headline = narrative.get("headline", "")
    if portfolio_headline:
        _headline_box(portfolio_headline, styles, story)

    # ── Portfolio Overview Table ───────────────────────────────────────────────
    story.append(Paragraph("Portfolio Snapshot", styles["h2"]))

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
        [Paragraph("Drugs Assessed", styles["cell_label"]),
         Paragraph(", ".join(stats["drugs"]), styles["cell_value"])],
        [Paragraph("Average MoA Score", styles["cell_label"]),
         Paragraph(f"{stats['avg_score']} / 5" if stats["avg_score"] else "N/A", styles["cell_value"])],
        [Paragraph("Guardrail Outcomes", styles["cell_label"]),
         Paragraph(f"{stats['guardrail_pass']} PASS  |  {stats['guardrail_fail']} FAIL", styles["cell_value"])],
    ]
    if dist_parts:
        overview_data.append([
            Paragraph("Score Distribution", styles["cell_label"]),
            Paragraph(";  ".join(dist_parts), styles["cell_value"]),
        ])
    if cls_parts:
        overview_data.append([
            Paragraph("Classification Breakdown", styles["cell_label"]),
            Paragraph(";  ".join(cls_parts), styles["cell_value"]),
        ])

    ov_table = Table(overview_data, colWidths=[2.2 * inch, 4.5 * inch])
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
    story.append(Spacer(1, 10))

    # ── Key Insights ──────────────────────────────────────────────────────────
    story.append(Paragraph("Key Insights", styles["h2"]))
    _render_key_insights(narrative.get("key_insights", []), styles, story)
    story.append(Spacer(1, 8))

    # ── Score Reference ───────────────────────────────────────────────────────
    score_ref = narrative.get("score_reference", "")
    if score_ref:
        story.append(Paragraph(f"Score: {score_ref}", styles["score_ref"]))
        story.append(Spacer(1, 4))

    # ── Bottom-Line Implication ───────────────────────────────────────────────
    bottom_line = narrative.get("bottom_line", "")
    if bottom_line:
        story.append(Paragraph("Bottom-Line Implication", styles["h2"]))
        story.append(Paragraph(bottom_line, styles["bottom_line"]))
        story.append(Spacer(1, 10))

    # ── Drug Score Summary Table ──────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER_COLOR, spaceAfter=10))
    story.append(Paragraph("MoA Innovation Score Summary", styles["h2"]))

    display_cols = ["Drug", "Indication", "Classification", "Score", "Guardrail", "Confidence", "Date"]
    data_cols = ["drug_name", "indication", "moa_classification", "score", "guardrail",
                 "confidence_tier", "analysis_date"]

    header_row = [Paragraph(c, styles["cell_header"]) for c in display_cols]
    table_data = [header_row]

    col_widths = [1.1 * inch, 1.2 * inch, 1.15 * inch, 0.5 * inch, 0.7 * inch, 0.85 * inch, 0.9 * inch]

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

    # ── Per-Drug Breakdown ────────────────────────────────────────────────────
    per_drug = narrative.get("per_drug_narratives", {})
    if per_drug:
        story.append(PageBreak())
        story.append(Paragraph("Drug-Level Analysis", styles["h2"]))

        for drug_name, drug_narrative in per_drug.items():
            drug_stat = stats.get("per_drug_stats", {}).get(drug_name, {})

            block = []

            # Drug name header
            block.append(Paragraph(drug_name, styles["h3_drug"]))

            # Stat line
            if drug_stat:
                block.append(Paragraph(
                    f"Score: {drug_stat.get('score', 'N/A')}/5  |  "
                    f"Classification: {drug_stat.get('classification', 'N/A')}  |  "
                    f"Guardrail: {drug_stat.get('guardrail', 'N/A')}  |  "
                    f"Confidence: {drug_stat.get('confidence_tier', 'N/A')}",
                    styles["stat"],
                ))

            # Drug headline
            drug_headline = drug_narrative.get("headline", "")
            if drug_headline:
                _headline_box(drug_headline, styles, block)

            # Drug key insights
            drug_insights = drug_narrative.get("key_insights", [])
            if drug_insights:
                block.append(Paragraph("Key Insights", styles["h3"]))
                _render_key_insights(drug_insights, styles, block)

            # Drug score reference
            drug_score_ref = drug_narrative.get("score_reference", "")
            if drug_score_ref:
                block.append(Paragraph(f"Score: {drug_score_ref}", styles["score_ref"]))

            # Drug bottom line
            drug_bottom_line = drug_narrative.get("bottom_line", "")
            if drug_bottom_line:
                block.append(Paragraph("Bottom-Line", styles["h3"]))
                block.append(Paragraph(drug_bottom_line, styles["bottom_line"]))

            block.append(HRFlowable(
                width="100%", thickness=0.5, color=DIVIDER_COLOR,
                spaceBefore=8, spaceAfter=4,
            ))

            story.append(KeepTogether(block))
            story.append(Spacer(1, 4))

    # ── Score Reference / Methodology Note ────────────────────────────────────
    methodology_note = narrative.get("score_methodology_note", "")
    if methodology_note:
        story.append(Spacer(1, 10))
        story.append(Paragraph("About the MoA Innovation Score", styles["h2"]))
        story.append(Paragraph(methodology_note, styles["methodology_note"]))

    # ── MoA Scoring Framework Reference Table ─────────────────────────────────
    story.append(Paragraph("MoA Innovation Scoring Reference", styles["h2"]))

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

    # Score legend
    legend_text = "  |  ".join(f"{k} = {v}" for k, v in SCORE_LABEL.items())
    story.append(Paragraph(
        f"<b>Score Legend:</b>  {legend_text}",
        ParagraphStyle("Legend", parent=styles["body"], fontSize=8, textColor=LIGHT_GRAY),
    ))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CCCCCC"), spaceBefore=14))
    story.append(Paragraph(
        "This report was auto-generated from MoA Innovation Scorer output "
        "using Gemini for narrative analysis. For internal use only.",
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
            blob.upload_from_filename(local_path, content_type="application/pdf")
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
    parser = argparse.ArgumentParser(
        description="Generate MoA Innovation business-facing PDF report from BigQuery"
    )
    parser.add_argument("--output", "-o", default=None, help="Output .pdf path")
    parser.add_argument("--molecule", default=None, help="Filter to a specific molecule")
    args = parser.parse_args()

    generate_moa_report(molecule=args.molecule, output_path=args.output)


if __name__ == "__main__":
    main()
