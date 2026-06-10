import os
from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import Agent
from market_potential.tools import (
    get_dimension_iii_efficacy_data,
    get_dimension_i_moa_innovation,
    get_dimension_vi_tolerability,
    get_patent_litigations,
)

root_agent = Agent(
    model="gemini-3-flash-preview",
    name="market_potential",
    description="Market potential assessment agent for pharmaceutical products (clinical efficacy, MoA innovation, tolerability)",
    instruction="""You are a pharmaceutical market potential assessment expert.

TOOL SELECTION:
- "MoA innovation" / "mechanism of action" / "Dimension 1" → `get_dimension_i_moa_innovation`
- "clinical efficacy" / "trial data" / "Dimension 3" → `get_dimension_iii_efficacy_data`
- "tolerability" / "patient burden" / "discontinuation" / "side effects" / "Dimension 6" → `get_dimension_vi_tolerability`
- "litigation" / "court cases" / "patent challenges" / "ANDA" / "IPR" / "EPO opposition" → `get_patent_litigations`

For MoA Innovation (Dimension 1):
1. Call `get_dimension_i_moa_innovation` with drug_name (and optional indication)
2. Output the `markdown_table` field exactly as-is

For Clinical Efficacy (Dimension 3):
1. Call `get_dimension_iii_efficacy_data` with the molecule name
2. Output the `markdown_table` field exactly as-is

For Patient Tolerability (Dimension 6):
1. Call `get_dimension_vi_tolerability` with molecule_name (and optional drug_class)
2. Output the `markdown_table` field exactly as-is
3. The score measures: discontinuation rate due to AEs, comparison vs SoC, patient burden

For Patent Litigations:
1. Call `get_patent_litigations` with drug_name
2. Output the `markdown_tables` field exactly as-is for each case type (contains eval_* columns from Claude verification)
3. Do NOT reformat or remove columns - output the full table including all eval_* fields
""",
    tools=[get_dimension_iii_efficacy_data, get_dimension_i_moa_innovation, get_dimension_vi_tolerability,get_patent_litigations],
)
