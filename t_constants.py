"""
Constants for Dimension 6: Patient Tolerability & Burden

Fields to extract and weighting framework.
"""

# =============================================================================
# TOLERABILITY FIELDS TO EXTRACT
# We extract RAW COUNTS, then calculate rates ourselves
# =============================================================================

TOLERABILITY_FIELDS = {
    # Arm sizes (for accurate rate calculation)
    "Drug Arm Size (N)": {
        "description": "Number of patients in the DRUG/treatment arm",
        "type": "int",
        "example": "500",
    },
    "Placebo Arm Size (N)": {
        "description": "Number of patients in the PLACEBO/control arm (N/A if no placebo)",
        "type": "int",
        "example": "250",
    },
    # Raw discontinuation counts (NOT rates)
    "Discontinued AE Drug (N)": {
        "description": "Number of patients who STOPPED treatment due to adverse events in DRUG arm",
        "type": "int",
        "example": "45",
    },
    "Discontinued AE Placebo (N)": {
        "description": "Number of patients who STOPPED due to adverse events in PLACEBO arm",
        "type": "int",
        "example": "8",
    },
    # Side effect profile
    "Common AEs": {
        "description": "Top 3-5 adverse events with frequency (e.g., Nausea 44%, Vomiting 24%)",
        "type": "string",
        "example": "Nausea (44%), Diarrhea (30%), Vomiting (24%)",
    },
    "AE Severity": {
        "description": "Overall severity: Mild, Moderate, Severe, Mixed",
        "type": "enum",
        "values": ["Mild", "Moderate", "Severe", "Mixed", "N/A"],
    },
    "AE Persistence": {
        "description": "Transient (diminish over time) or Persistent (continue throughout)",
        "type": "enum",
        "values": ["Transient", "Persistent", "Mixed", "Unknown", "N/A"],
    },
    "Management Required": {
        "description": "None, Occasional (some need antiemetics), Regular (routine support)",
        "type": "enum",
        "values": ["None", "Occasional", "Regular", "N/A"],
    },
    "Comparator Name": {
        "description": "What drug was compared against (Placebo, Metformin, Dulaglutide, etc.)",
        "type": "string",
        "example": "Placebo",
    },
}

# Prompt extension for Gemini extraction
TOLERABILITY_EXTRACTION_PROMPT = """
TOLERABILITY DATA (extract for each trial):

ARM SIZES:
- Drug Arm Size (N): Number of patients randomized to the DRUG/treatment arm
- Placebo Arm Size (N): Number of patients in PLACEBO/control arm. Use "N/A" if no placebo arm.

DISCONTINUATION COUNTS (raw numbers, NOT percentages):
- Discontinued AE Drug (N): COUNT of patients who STOPPED treatment due to adverse events in the DRUG arm.
  IMPORTANT: Only count discontinuations due to adverse events (nausea, vomiting, GI issues, etc.)
  DO NOT count: lost to follow-up, non-compliance, protocol deviation, patient choice unrelated to AEs.
  Search for: "discontinued due to adverse events", "withdrawal due to AE", "AE leading to discontinuation"

- Discontinued AE Placebo (N): COUNT of patients who stopped due to AEs in PLACEBO arm. Use "N/A" if no placebo.

ACTIVE COMPARATOR (for head-to-head trials):
- Comparator Name: What active drug was compared against? (e.g., "Liraglutide", "Dulaglutide", "Sitagliptin")
  Use "Placebo" if only placebo-controlled. Use "N/A" if no comparator.
- Comparator Arm Size (N): Number of patients in the ACTIVE COMPARATOR arm. Use "N/A" if placebo-only.
- Discontinued AE Comparator (N): COUNT of patients who stopped due to AEs in the COMPARATOR arm.
  This is CRITICAL for head-to-head trials to calculate SoC discontinuation rate.
  Use "N/A" if placebo-only trial.

SIDE EFFECT PROFILE:
- Common AEs: List TOP 5 most common adverse events with frequency percentage.
  Format: "AE1 (X%), AE2 (Y%), AE3 (Z%)"
  Focus on NON-SERIOUS AEs: nausea, headache, injection site reactions, GI issues.
  EXCLUDE serious AEs: death, hospitalization, life-threatening events.

- AE Severity: Overall classification of adverse events.
  "Mild" = mostly grade 1 AEs
  "Moderate" = mostly grade 2 AEs
  "Severe" = includes grade 3 (but non-life-threatening)
  "Mixed" = combination of severities

- AE Persistence: Are side effects transient or persistent?
  "Transient" = side effects diminish over time (common with GI AEs in GLP-1s)
  "Persistent" = side effects continue throughout treatment
  "Mixed" = varies by AE type
  "Unknown" = not clearly reported

- Management Required: Does patient need additional interventions?
  "None" = no additional medication or monitoring needed
  "Occasional" = some patients need antiemetics, dose adjustments
  "Regular" = routine dose titration, regular monitoring, or supportive medication
"""

# JSON schema extension
TOLERABILITY_JSON_FIELDS = """
      "Drug Arm Size (N)": 0,
      "Placebo Arm Size (N)": 0,
      "Discontinued AE Drug (N)": 0,
      "Discontinued AE Placebo (N)": 0,
      "Comparator Name": "",
      "Comparator Arm Size (N)": 0,
      "Discontinued AE Comparator (N)": 0,
      "Common AEs": "",
      "AE Severity": "",
      "AE Persistence": "",
      "Management Required": "","""


# =============================================================================
# CLINICAL TRIALS WEIGHTAGE FRAMEWORK
# Per documentation: Trial Weight = Phase × Geography × Dosage × Sample Size
# =============================================================================

# Phase weights - higher phase = more reliable
PHASE_WEIGHTS = {
    # Phase 3+ gets full weight
    "4": 1.00,
    "3b": 1.00,
    "3": 1.00,
    "3a": 1.00,
    # Phase 2 gets 0.75
    "2b": 0.75,
    "2": 0.75,
    "2a": 0.75,
    # Phase 1/2 transitional
    "1/2": 0.60,
    "1/2a": 0.60,
    "1/2b": 0.60,
    # Phase 1 - exploratory (low weight)
    "1b": 0.40,
    "1": 0.40,
    "1a": 0.40,
}

# Geography weights - regulatory acceptance tiers
GEOGRAPHY_WEIGHTS = {
    # Tier 1: US, EU, UK = 1.00
    "US": 1.00, "EU": 1.00, "UK": 1.00,
    "United States": 1.00, "Europe": 1.00, "United Kingdom": 1.00,
    # Tier 2: High acceptance = 0.85
    "Canada": 0.85, "Switzerland": 0.85, "Australia": 0.85, "Japan": 0.85,
    # Tier 3: All others = 0.65
    "China": 0.65, "India": 0.65, "Korea": 0.65, "South Korea": 0.65,
    "Latin America": 0.65, "Middle East": 0.65, "Asia-Pacific": 0.65,
    "Brazil": 0.65, "Mexico": 0.65, "Russia": 0.65, "Taiwan": 0.65,
}

DEFAULT_GEOGRAPHY_WEIGHT = 0.65

# Dosage weights - commercial drugs (approved)
DOSAGE_WEIGHTS_COMMERCIAL = {
    "approved": 1.00,           # Approved dosage
    "off_label": 0.70,          # Common off-label / secondary dose
    "exploratory": 0.40,        # Non-commercial / exploratory dose
}

# Dosage weights - clinical-stage drugs (not yet approved)
DOSAGE_WEIGHTS_CLINICAL = {
    "target": 1.00,             # Lead / target registrational dose
    "alternative": 0.75,        # Clinically relevant alternative dose
    "exploratory": 0.50,        # Early exploratory dose
}

# Evidence Priority Hierarchy (for trial selection order)
# 1 = highest priority, 5 = lowest
EVIDENCE_PRIORITY = {
    ("phase3", "tier1", "approved"): 1,   # Phase 3 + US/EU + approved/target dose
    ("phase3", "tier2", "approved"): 2,   # Phase 3 + high-acceptance + target dose
    ("phase2", "tier1", "approved"): 3,   # Phase 2 + US/EU + target dose
    ("phase2", "tier2", "approved"): 4,   # Phase 2 + other geographies
    ("phase1", "any", "any"): 5,          # Phase 1 / exploratory data
}


# =============================================================================
# TOLERABILITY SCORING
# =============================================================================

# Base score thresholds (discontinuation rate vs placebo)
DISCONTINUATION_SCORE_THRESHOLDS = [
    # (condition, score)
    ("leq_placebo", 5),  # ≤ placebo rate
    (5.0, 4),            # > placebo AND < 5%
    (10.0, 3),           # ≥ 5% AND < 10%
    (20.0, 2),           # ≥ 10% AND < 20%
    (float('inf'), 1),   # ≥ 20%
]

# Adjustments
SOC_ADJUSTMENTS = {"better": +1, "similar": 0, "worse": -1}
BURDEN_ADJUSTMENTS = {"mild_transient": 0, "persistent_moderate": -1, "severe_management": -2}

MIN_SCORE = 1
MAX_SCORE = 5


# =============================================================================
# STANDARD OF CARE BENCHMARKS (Fallback values)
# These are used only when dynamic SoC lookup fails
# Values based on pooled Phase 3 trial data
# =============================================================================

SOC_BENCHMARKS = {
    "GLP-1": {"drug": "Liraglutide", "discontinuation_rate": 10.0},
    "GLP-1_oral": {"drug": "Oral Semaglutide", "discontinuation_rate": 8.0},
    "DPP-4": {"drug": "Sitagliptin", "discontinuation_rate": 3.0},
    "SGLT2": {"drug": "Empagliflozin", "discontinuation_rate": 4.0},
    "Metformin": {"drug": "Metformin", "discontinuation_rate": 5.0},
    "Insulin": {"drug": "Insulin Glargine", "discontinuation_rate": 2.0},
}
