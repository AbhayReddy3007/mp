"""
Patient Tolerability & Burden - Dimension 6 Tools

This module provides tolerability extraction and scoring for clinical trials.
It extends the clinical efficacy extraction with tolerability-specific fields.
"""

import os
import re
import json
from typing import Optional
from dotenv import load_dotenv
from google import genai
from json_repair import repair_json

load_dotenv()

# Initialize Gemini client
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

from market_potential.gemini_extractor import extract_clinical_trial_data
from google.genai import types as genai_types

from market_potential.tolerability_constants import (
    TOLERABILITY_EXTRACTION_PROMPT,
    TOLERABILITY_JSON_FIELDS,
    PHASE_WEIGHTS,
    GEOGRAPHY_WEIGHTS,
    DEFAULT_GEOGRAPHY_WEIGHT,
    DOSAGE_WEIGHTS_COMMERCIAL,
    DOSAGE_WEIGHTS_CLINICAL,
    EVIDENCE_PRIORITY,
    DISCONTINUATION_SCORE_THRESHOLDS,
    SOC_ADJUSTMENTS,
    BURDEN_ADJUSTMENTS,
    MIN_SCORE,
    MAX_SCORE,
    SOC_BENCHMARKS,
)


# =============================================================================
# DYNAMIC SOC LOOKUP
# =============================================================================

async def get_soc_benchmark_dynamic(drug_class: str, indication: str = None,
                                    trial_context: list[dict] = None) -> dict:
    """Dynamically fetch Standard of Care benchmark using web search.

    Searches for:
    1. Current standard of care drug for the class/indication
    2. Its discontinuation rate from clinical trials

    Args:
        drug_class: Drug class (e.g., "GLP-1", "SGLT2")
        indication: Optional indication (e.g., "type 2 diabetes", "obesity")
        trial_context: Optional list of trial dicts with Comparator Name, Phase, etc.

    Returns:
        dict: {"drug": str, "discontinuation_rate": float, "source": str}
    """
    indication_str = f" for {indication}" if indication else ""

    # Extract relevant context from trials
    comparators = set()
    phases = set()
    head_to_head_trials = []  # Trials with active comparators (not just placebo)

    if trial_context:
        for t in trial_context[:50]:  # Check more trials for head-to-head
            comp = t.get("Comparator Name", "")
            trial_title = t.get("Trial Title", "")
            trial_id = t.get("Trial ID", "")

            if comp and comp not in ["N/A", "Placebo", "placebo", ""]:
                comparators.add(comp)
                # This is a head-to-head trial - save it
                if trial_id:
                    head_to_head_trials.append({
                        "id": trial_id,
                        "title": trial_title[:100] if trial_title else "",
                        "comparator": comp,
                        "phase": t.get("Phase", "")
                    })

            phase = t.get("Phase", "")
            if phase:
                phases.add(str(phase))

    comparator_str = ""
    head_to_head_str = ""

    if comparators:
        comp_list = list(comparators)[:10]
        print(f"  📊 Trial context: {len(trial_context)} trials, comparators: {', '.join(comp_list)}")
        comparator_str = f"\n\nComparators seen in extracted trials: {', '.join(comp_list)}"
        comparator_str += "\nThese are active comparators used in clinical trials - one may be the current SoC."
    else:
        print(f"  📊 Trial context: {len(trial_context) if trial_context else 0} trials, no active comparators found")

    # Add head-to-head trial info for more accurate SoC lookup
    if head_to_head_trials:
        h2h_list = head_to_head_trials[:10]  # Top 10 head-to-head trials
        print(f"  🔬 Found {len(head_to_head_trials)} head-to-head trials")
        head_to_head_str = "\n\n=== HEAD-TO-HEAD TRIALS (search these for SoC discontinuation rates) ===\n"
        for h2h in h2h_list:
            head_to_head_str += f"- {h2h['id']}: vs {h2h['comparator']} (Phase {h2h['phase']})\n"
        head_to_head_str += "\nIMPORTANT: Search for the COMPARATOR's discontinuation rate FROM these specific head-to-head trials."

    prompt = f"""Identify the Standard of Care (SoC) drug for {drug_class} class{indication_str} and find its discontinuation rate.
{comparator_str}
{head_to_head_str}

=== DEFINITIONS ===

**Standard of Care (SoC) Drug**: The first-line, most established, or most widely prescribed drug in this class that serves as the clinical benchmark. This is typically:
- The first drug approved in the class (has longest track record)
- The drug most commonly used as an active comparator in clinical trials
- The drug recommended in treatment guidelines as first-line therapy

**Discontinuation Rate Due to Adverse Events**: The percentage of patients who STOPPED taking the drug because of side effects/adverse events during clinical trials. This is NOT:
- Overall dropout rate (which includes other reasons like lost to follow-up)
- Adverse event incidence rate (patients who experienced AEs but continued)
Look for phrases like: "discontinued due to adverse events", "withdrawal due to AEs", "treatment discontinuation for safety"

=== WHERE TO FIND THIS DATA (PRIORITY ORDER) ===

1. **HEAD-TO-HEAD TRIALS (HIGHEST PRIORITY)**: If NCT IDs are provided above, search for "[NCT ID] discontinuation adverse events" to find the COMPARATOR's discontinuation rate from that specific trial
2. **Recent Phase 3 Trials**: Search for "{drug_class} Phase 3 trial discontinuation adverse events"
3. **Meta-analyses/Pooled Analyses**: Search for "{drug_class} meta-analysis safety discontinuation"
4. **Real-World Evidence**: Post-marketing studies often show higher discontinuation than controlled trials
5. **FDA Drug Labels**: The "Adverse Reactions" section (use as fallback - may be outdated)

=== IMPORTANT GUIDANCE ===

- PREFER recent data (last 5 years) over older trials
- PREFER head-to-head comparisons over single-arm or placebo-controlled data
- Real-world discontinuation rates are typically HIGHER than controlled trial rates
- FDA label data may underestimate real-world discontinuation
- If multiple sources conflict, use the more recent or larger study

=== SEARCH TASK ===

Find:
1. The SoC drug for {drug_class} class{indication_str}
2. Its discontinuation rate due to adverse events from the BEST available source (prefer head-to-head or recent trials)

Return JSON:
```json
{{
  "soc_drug": "generic drug name",
  "discontinuation_rate": X.X,
  "source": "specific trial name or publication",
  "notes": "why this is SoC and how you found the rate"
}}
```

IMPORTANT: You MUST provide a numeric discontinuation_rate. Prefer recent head-to-head or real-world data over older FDA label data.
"""

    try:
        print(f"  🔍 Calling Gemini for dynamic SoC lookup...")
        # Note: Cannot use response_mime_type with Google Search tool
        response = await gemini_client.aio.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
            ),
        )
        # print("soc responseeeee",response)
        result_text = response.text.strip() if response.text else ""
        # print(f"  📝 SoC API response received ({len(result_text)} chars)")

        # Debug: show first 300 chars of response
        if result_text:
            preview = result_text
            # print(f"  📝 Response preview: {preview}...")
        else:
            print(f"  ⚠️ SoC API returned empty response")

        # Parse JSON from response using json_repair first
        if result_text:
            # Clean up response - extract JSON block
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            else:
                # Try to find JSON object in text
                json_start = result_text.find('{')
                json_end = result_text.rfind('}')
                if json_start != -1 and json_end != -1:
                    result_text = result_text[json_start:json_end+1]

            # Use json_repair library first
            try:
                data = repair_json(result_text, return_objects=True)
            except Exception as parse_err:
                print(f"  ⚠️ json_repair failed: {parse_err}, trying json.loads")
                data = json.loads(result_text)

            # print(f"  📝 Parsed data type: {type(data).__name__}, value: {data}")

            soc_drug = data.get("soc_drug", "Unknown")
            disc_rate = data.get("discontinuation_rate")
            source = data.get("source", "Web search")

            if disc_rate is not None:
                print(f"  ✓ Dynamic SoC: {soc_drug} ({disc_rate}% discontinuation) - {source}")
                return {
                    "drug": soc_drug,
                    "discontinuation_rate": float(disc_rate),
                    "source": source,
                    "dynamic": True,
                }
            else:
                print(f"  ⚠️ SoC API returned null discontinuation rate for {soc_drug}")

    except Exception as e:
        print(f"  ⚠️ Dynamic SoC lookup failed: {e}")

    # Fallback to hardcoded benchmarks
    fallback = SOC_BENCHMARKS.get(drug_class, {"drug": "Unknown", "discontinuation_rate": 5.0})
    print(f"  ⚠️ Using fallback SoC: {fallback['drug']} ({fallback['discontinuation_rate']}%)")
    return {
        "drug": fallback["drug"],
        "discontinuation_rate": fallback["discontinuation_rate"],
        "source": "Hardcoded benchmark (fallback)",
        "dynamic": False,
    }


# =============================================================================
# HELPERS (defined first as they're used in weighting/extraction)
# =============================================================================

def _parse_int(value) -> Optional[int]:
    """Parse integer from various formats."""
    if value is None or str(value).upper() in ["N/A", "NONE", ""]:
        return None
    try:
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        cleaned = re.sub(r'[^\d]', '', str(value))
        return int(cleaned) if cleaned else None
    except (ValueError, TypeError):
        return None


def _parse_float(value) -> Optional[float]:
    """Parse float from various formats."""
    if value is None or str(value).upper() in ["N/A", "NONE", ""]:
        return None
    try:
        if isinstance(value, (int, float)):
            return float(value)
        cleaned = re.sub(r'[^\d.]', '', str(value))
        return float(cleaned) if cleaned else None
    except (ValueError, TypeError):
        return None


# =============================================================================
# EXTRACTION
# =============================================================================

async def extract_tolerability_data(molecule_name: str, max_trials: int = None) -> dict:
    """Extract clinical trial data WITH tolerability fields.

    This calls the standard clinical efficacy extractor with additional
    tolerability fields (discontinuation counts, AE profile, etc.)

    Args:
        molecule_name: Name of the molecule to search for
        max_trials: Optional limit on number of trials

    Returns:
        dict with:
            - trials: list of trial data (with tolerability fields)
            - total_trials: total number found
            - completeness: percentage of filled fields
    """
    result = await extract_clinical_trial_data(
        molecule_name=molecule_name,
        max_trials=max_trials,
        extra_fields_prompt=TOLERABILITY_EXTRACTION_PROMPT,
        extra_fields_json=TOLERABILITY_JSON_FIELDS,
    )

    # Calculate discontinuation rates from raw counts
    trials = result.get("trials", [])
    for trial in trials:
        _calculate_discontinuation_rates(trial)

    return result


def _calculate_discontinuation_rates(trial: dict) -> None:
    """Calculate discontinuation rates from raw counts.

    Formula: Discontinuation Rate = (Discontinued / Arm Size) × 100

    Modifies trial dict in place, adding:
        - Discontinuation Rate Drug (%)
        - Discontinuation Rate Placebo (%)
        - Discontinuation Rate Comparator (%)  [for head-to-head trials]
    """
    # Drug arm
    drug_disc = _parse_int(trial.get("Discontinued AE Drug (N)"))
    drug_size = _parse_int(trial.get("Drug Arm Size (N)"))

    if drug_disc is not None and drug_size and drug_size > 0:
        trial["Discontinuation Rate Drug (%)"] = round((drug_disc / drug_size) * 100, 2)
    else:
        # Fallback: use total Size if Drug Arm Size not available
        total_size = _parse_int(trial.get("Size"))
        if drug_disc is not None and total_size and total_size > 0:
            # Assume ~60-70% of total in drug arm for typical RCT
            estimated_drug_size = int(total_size * 0.65)
            trial["Discontinuation Rate Drug (%)"] = round((drug_disc / estimated_drug_size) * 100, 2)
        else:
            trial["Discontinuation Rate Drug (%)"] = None

    # Placebo arm
    placebo_disc = _parse_int(trial.get("Discontinued AE Placebo (N)"))
    placebo_size = _parse_int(trial.get("Placebo Arm Size (N)"))

    if placebo_disc is not None and placebo_size and placebo_size > 0:
        trial["Discontinuation Rate Placebo (%)"] = round((placebo_disc / placebo_size) * 100, 2)
    else:
        trial["Discontinuation Rate Placebo (%)"] = None

    # Comparator arm (for head-to-head trials)
    comp_disc = _parse_int(trial.get("Discontinued AE Comparator (N)"))
    comp_size = _parse_int(trial.get("Comparator Arm Size (N)"))

    if comp_disc is not None and comp_size and comp_size > 0:
        trial["Discontinuation Rate Comparator (%)"] = round((comp_disc / comp_size) * 100, 2)
    else:
        trial["Discontinuation Rate Comparator (%)"] = None


# =============================================================================
# TRIAL WEIGHTING & PRIORITIZATION
# Per documentation: Trial Weight = Phase × Geography × Dosage × Sample Size
# =============================================================================

def normalize_phase(phase: str) -> str:
    """Normalize phase string to standard format."""
    if not phase or str(phase).upper() in ["N/A", "NONE", ""]:
        return ""

    phase = str(phase).strip().upper().replace("PHASE", "").strip()

    # Convert Roman numerals
    roman = {"IV": "4", "III": "3", "II": "2", "I": "1"}
    if phase in roman:
        return roman[phase]

    return phase.lower()


def get_phase_tier(phase: str) -> str:
    """Get phase tier for evidence priority."""
    normalized = normalize_phase(phase)
    if normalized in ["3", "3a", "3b", "4"]:
        return "phase3"
    elif normalized in ["2", "2a", "2b"]:
        return "phase2"
    else:
        return "phase1"


def get_phase_weight(phase: str) -> float:
    """Get weight for a clinical trial phase."""
    normalized = normalize_phase(phase)
    if not normalized:
        return 0.40  # Conservative default for missing phase
    return PHASE_WEIGHTS.get(normalized, 0.40)


def get_geography_tier(region: str) -> str:
    """Get geography tier for evidence priority."""
    if not region or str(region).upper() in ["N/A", "NONE", ""]:
        return "tier3"

    region_lower = region.lower()
    tier1_keywords = ["us", "usa", "united states", "eu", "europe", "uk", "united kingdom"]
    tier2_keywords = ["canada", "switzerland", "australia", "japan"]

    for kw in tier1_keywords:
        if kw in region_lower:
            return "tier1"
    for kw in tier2_keywords:
        if kw in region_lower:
            return "tier2"
    return "tier3"


def get_geography_weight(region: str) -> float:
    """Get weight for geographic region(s)."""
    if not region or str(region).upper() in ["N/A", "NONE", ""]:
        return DEFAULT_GEOGRAPHY_WEIGHT

    # Map countries to regions
    country_map = {
        "united states": "US", "usa": "US", "us": "US",
        "germany": "EU", "france": "EU", "spain": "EU", "italy": "EU",
        "netherlands": "EU", "belgium": "EU", "poland": "EU",
        "united kingdom": "UK", "uk": "UK",
        "canada": "Canada", "australia": "Australia", "japan": "Japan",
        "china": "China", "india": "India", "korea": "Korea",
        "brazil": "Latin America", "mexico": "Latin America",
    }

    # Split by / and ,
    parts = re.split(r'[/,]', region.lower())
    weights = []

    for part in parts:
        part = part.strip()
        if part in country_map:
            r = country_map[part]
            weights.append(GEOGRAPHY_WEIGHTS.get(r, DEFAULT_GEOGRAPHY_WEIGHT))
        elif part.upper() in GEOGRAPHY_WEIGHTS:
            weights.append(GEOGRAPHY_WEIGHTS[part.upper()])

    return max(weights) if weights else DEFAULT_GEOGRAPHY_WEIGHT


def get_dosage_weight(dosage: str, is_approved: bool = False, dosage_type: str = None) -> float:
    """Get weight for dosage.

    Args:
        dosage: Dosage string
        is_approved: Whether drug is commercially approved
        dosage_type: Override type ("approved", "off_label", "target", "alternative", "exploratory")
    """
    if not dosage or str(dosage).upper() in ["N/A", "NONE", ""]:
        # Conservative weight for missing dosage
        return DOSAGE_WEIGHTS_CLINICAL["exploratory"] if not is_approved else DOSAGE_WEIGHTS_COMMERCIAL["exploratory"]

    # Use explicit dosage type if provided
    if dosage_type:
        if is_approved:
            return DOSAGE_WEIGHTS_COMMERCIAL.get(dosage_type, 0.70)
        else:
            return DOSAGE_WEIGHTS_CLINICAL.get(dosage_type, 0.75)

    # Default: approved drug = approved dose, clinical = target dose
    if is_approved:
        return DOSAGE_WEIGHTS_COMMERCIAL["approved"]
    else:
        return DOSAGE_WEIGHTS_CLINICAL["target"]


def get_evidence_priority(trial: dict) -> int:
    """Get evidence priority score (1=highest, 5=lowest) per hierarchy.

    Priority Order:
    1. Phase 3 + US/EU + approved/target dose
    2. Phase 3 + high-acceptance geographies + target dose
    3. Phase 2 + US/EU + target dose
    4. Phase 2 + other geographies
    5. Phase 1 / exploratory data
    """
    phase_tier = get_phase_tier(trial.get("Phase", ""))
    geo_tier = get_geography_tier(trial.get("Primary Region", ""))

    if phase_tier == "phase3":
        if geo_tier == "tier1":
            return 1
        elif geo_tier == "tier2":
            return 2
        else:
            return 3
    elif phase_tier == "phase2":
        if geo_tier in ["tier1", "tier2"]:
            return 3
        else:
            return 4
    else:
        return 5


def calculate_trial_weight(trial: dict, is_approved: bool = False) -> float:
    """Calculate overall weight for a trial.

    Formula (per documentation):
    Trial Weight = Phase Weight × Geography Weight × Dosage Weight × Sample Size

    Note: Sample size is multiplied directly (not sqrt) per documentation.
    """
    phase_w = get_phase_weight(trial.get("Phase", ""))
    geo_w = get_geography_weight(trial.get("Primary Region", ""))
    dosage_w = get_dosage_weight(trial.get("Dosage", ""), is_approved)

    # Sample size multiplied directly per documentation
    size = _parse_int(trial.get("Size")) or 1

    return phase_w * geo_w * dosage_w * size


def prioritize_trials(trials: list[dict], is_approved: bool = False) -> list[dict]:
    """Prioritize and filter trials according to evidence hierarchy.

    Steps (per documentation):
    1. Filter by dosage relevance (approved/target doses first)
    2. Apply phase weighting
    3. Handle edge case: same phase → pick dose with best efficacy + lowest discontinuation

    Args:
        trials: List of trial dicts
        is_approved: Whether drug is commercially approved

    Returns:
        Prioritized trials sorted by evidence hierarchy
    """
    if not trials:
        return []

    # Step 1: Calculate priority and weight for each trial
    for trial in trials:
        trial["_priority"] = get_evidence_priority(trial)
        trial["_weight"] = calculate_trial_weight(trial, is_approved)
        trial["_phase_weight"] = get_phase_weight(trial.get("Phase", ""))
        trial["_geo_weight"] = get_geography_weight(trial.get("Primary Region", ""))
        trial["_dosage_weight"] = get_dosage_weight(trial.get("Dosage", ""), is_approved)
        trial["_phase_tier"] = get_phase_tier(trial.get("Phase", ""))
        trial["_geo_tier"] = get_geography_tier(trial.get("Primary Region", ""))

    # Step 2: Handle edge case - multiple doses in same phase
    # Group by phase and pick best dose per phase
    phase_groups = {}
    for trial in trials:
        phase_tier = trial["_phase_tier"]
        if phase_tier not in phase_groups:
            phase_groups[phase_tier] = []
        phase_groups[phase_tier].append(trial)

    # For each phase group, rank by: lowest discontinuation + highest efficacy
    prioritized = []
    for phase_tier in ["phase3", "phase2", "phase1"]:
        if phase_tier not in phase_groups:
            continue

        group = phase_groups[phase_tier]

        # Sort within group by: (1) priority, (2) discontinuation (low=good), (3) weight (high=good)
        def sort_key(t):
            disc_rate = t.get("Discontinuation Rate Drug (%)")
            if disc_rate is None:
                disc_rate = 100  # Penalize missing data
            return (t["_priority"], disc_rate, -t["_weight"])

        group.sort(key=sort_key)
        prioritized.extend(group)

    return prioritized


def apply_trial_weights(trials: list[dict], is_approved: bool = False) -> list[dict]:
    """Apply weightage framework and return prioritized trials.

    This function:
    1. Prioritizes trials per evidence hierarchy (Phase 3 > Phase 2 > Phase 1)
    2. Calculates weights: Phase × Geography × Dosage × Sample Size
    3. Sorts by priority then weight

    Args:
        trials: List of trial dicts
        is_approved: Whether drug is commercially approved

    Returns:
        Trials with weights added, sorted by priority then weight
    """
    return prioritize_trials(trials, is_approved)


def calculate_weighted_discontinuation_rate(trials: list[dict], is_approved: bool = False) -> dict:
    """Calculate weighted average discontinuation rate across trials.

    Formula: Σ(Trial_Weight × Rate) / Σ(Trial_Weight)

    Returns:
        dict with:
            - drug_rate: weighted discontinuation rate for drug arm
            - placebo_rate: weighted discontinuation rate for placebo arm
            - trials_used: number of trials with valid data
    """
    drug_weighted_sum = 0.0
    drug_total_weight = 0.0
    placebo_weighted_sum = 0.0
    placebo_total_weight = 0.0
    trials_used = 0

    for trial in trials:
        weight = calculate_trial_weight(trial, is_approved)

        # Drug arm
        drug_rate = trial.get("Discontinuation Rate Drug (%)")
        if drug_rate is not None and isinstance(drug_rate, (int, float)):
            drug_weighted_sum += weight * drug_rate
            drug_total_weight += weight
            trials_used += 1

        # Placebo arm
        placebo_rate = trial.get("Discontinuation Rate Placebo (%)")
        if placebo_rate is not None and isinstance(placebo_rate, (int, float)):
            placebo_weighted_sum += weight * placebo_rate
            placebo_total_weight += weight

    return {
        "drug_rate": round(drug_weighted_sum / drug_total_weight, 2) if drug_total_weight > 0 else None,
        "placebo_rate": round(placebo_weighted_sum / placebo_total_weight, 2) if placebo_total_weight > 0 else None,
        "trials_used": trials_used,
    }


# =============================================================================
# SCORING
# =============================================================================

async def compute_tolerability_score(trials: list[dict], molecule: str,
                                      drug_class: str = "GLP-1",
                                      indication: str = None,
                                      is_approved: bool = False) -> dict:
    """Compute Dimension 6 Tolerability Score.

    Scoring Logic:
    1. Base Score: From discontinuation rate vs placebo
    2. Adjustment: SoC comparison (+1/-1)
    3. Adjustment: Patient burden (-1/-2)

    Args:
        trials: List of trial dicts with tolerability fields
        molecule: Name of the molecule (for rationale generation)
        drug_class: Drug class for SoC benchmark lookup
        indication: Optional indication for more specific SoC lookup
        is_approved: Whether drug is commercially approved

    Returns:
        dict with score, breakdown, and justification
    """
    # Apply weights and calculate weighted rates
    weighted_trials = apply_trial_weights(trials, is_approved)
    rates = calculate_weighted_discontinuation_rate(weighted_trials, is_approved)

    drug_rate = rates["drug_rate"]
    placebo_rate = rates["placebo_rate"]

    # Get SoC benchmark dynamically (with fallback to hardcoded)
    print(f"\n  Fetching Standard of Care benchmark for {drug_class}...")
    soc = await get_soc_benchmark_dynamic(drug_class, indication, trial_context=trials)
    soc_rate = soc.get("discontinuation_rate", 5.0)
    soc_source = soc.get("source", "Unknown")

    # ── STEP 1: Base Score ──
    base_score = 1
    base_reason = ""

    if drug_rate is None:
        base_score = 3  # Default if no data
        base_reason = "No discontinuation data available - using default score"
    elif placebo_rate is not None and drug_rate <= placebo_rate:
        base_score = 5
        base_reason = f"Drug ({drug_rate}%) ≤ placebo ({placebo_rate}%)"
    elif drug_rate < 5:
        base_score = 4
        base_reason = f"Drug ({drug_rate}%) > placebo but < 5%"
    elif drug_rate < 10:
        base_score = 3
        base_reason = f"Drug ({drug_rate}%) between 5-10%"
    elif drug_rate < 20:
        base_score = 2
        base_reason = f"Drug ({drug_rate}%) between 10-20%"
    else:
        base_score = 1
        base_reason = f"Drug ({drug_rate}%) ≥ 20%"

    # ── STEP 2: SoC Adjustment ──
    soc_adjustment = 0
    soc_comparison = "unknown"

    if drug_rate is not None:
        if drug_rate < soc_rate - 2:  # >2% better
            soc_adjustment = +1
            soc_comparison = "better"
        elif drug_rate > soc_rate + 2:  # >2% worse
            soc_adjustment = -1
            soc_comparison = "worse"
        else:
            soc_comparison = "similar"

    # ── STEP 3: Patient Burden Adjustment (per doc Section 6 Step 3) ──
    burden_adjustment = 0
    burden_level = "mild_transient"

    # Aggregate AE severity and persistence from trials
    severities = [t.get("AE Severity", "").lower() for t in weighted_trials[:5] if t.get("AE Severity")]
    persistences = [t.get("AE Persistence", "").lower() for t in weighted_trials[:5] if t.get("AE Persistence")]
    managements = [t.get("Management Required", "").lower() for t in weighted_trials[:5] if t.get("Management Required")]

    has_severe = any("severe" in s for s in severities)
    has_persistent = any("persistent" in p for p in persistences)
    has_regular_mgmt = any("regular" in m for m in managements)
    has_moderate = any("moderate" in s for s in severities)

    # Per doc Section 6 Step 3:
    # - Frequent and severe (non-serious) OR requires regular management → -2
    # - Persistent OR moderate → -1
    # - Mild and transient → 0
    if has_severe or has_regular_mgmt:
        burden_adjustment = -2
        burden_level = "severe_management"
    elif has_persistent or has_moderate:
        burden_adjustment = -1
        burden_level = "persistent_moderate"
    else:
        burden_level = "mild_transient"

    # ── FINAL SCORE ──
    final_score = base_score + soc_adjustment + burden_adjustment
    final_score = max(MIN_SCORE, min(MAX_SCORE, final_score))

    # ── GUARDRAIL CHECK ──
    guardrail_fail = False
    if drug_rate is not None and drug_rate >= soc_rate:
        if has_persistent or has_regular_mgmt:
            guardrail_fail = True

    # ── EXTRACT SIDE EFFECT SUMMARY FROM TOP TRIALS ──
    all_ae_names = set()
    all_ae_with_freq = []
    ae_severities = set()

    for t in weighted_trials[:5]:
        # Key AEs with frequency (e.g., "Nausea (44%), Diarrhea (30%)")
        common_ae = t.get("Common AEs", "")
        if common_ae and common_ae != "N/A":
            # Extract AE name + frequency pairs
            ae_pairs = re.findall(r'([A-Za-z][A-Za-z\s]+?)\s*\((\d+(?:\.\d+)?%?)\)', common_ae)
            if ae_pairs:
                for ae_name, freq in ae_pairs:
                    ae_name = ae_name.strip()
                    all_ae_names.add(ae_name)
                    freq_val = freq.replace('%', '')
                    all_ae_with_freq.append((ae_name, float(freq_val) if freq_val else 0))
            else:
                # Fallback: just extract AE names without percentages
                ae_names = re.split(r'[,;]', common_ae)
                for ae in ae_names:
                    ae_clean = re.sub(r'\([^)]*\)', '', ae).strip()
                    if ae_clean:
                        all_ae_names.add(ae_clean)

        # Severity (Mild/Moderate/Severe per doc Section 5 Step 3)
        sev = t.get("AE Severity", "")
        if sev and sev != "N/A":
            ae_severities.add(sev)

    # Format Key AEs - just the names
    key_aes_str = ", ".join(sorted(all_ae_names)[:5]) if all_ae_names else "Not reported"

    # Format Frequency - show actual percentages (per doc Section 5 Step 3)
    if all_ae_with_freq:
        # Sort by frequency descending and take top 5
        sorted_aes = sorted(all_ae_with_freq, key=lambda x: x[1], reverse=True)
        unique_aes = []
        seen = set()
        for ae, freq in sorted_aes:
            if ae not in seen and len(unique_aes) < 5:
                unique_aes.append(f"{ae} {freq:.0f}%")
                seen.add(ae)
        frequency_str = ", ".join(unique_aes) if unique_aes else "Not reported"
    else:
        frequency_str = "Not reported"

    # Format Severity - Mild/Moderate/Severe classification (per doc Section 3.3)
    severity_str = ", ".join(ae_severities) if ae_severities else "Not reported"

    # Calculate difference vs SoC
    diff_vs_soc = round(drug_rate - soc_rate, 2) if drug_rate is not None else None

    # Comparison descriptions
    if placebo_rate is not None and drug_rate is not None:
        if drug_rate <= placebo_rate:
            vs_placebo = f"Better (Drug {drug_rate}% vs Placebo {placebo_rate}%)"
        elif drug_rate < placebo_rate + 5:
            vs_placebo = f"Similar (Drug {drug_rate}% vs Placebo {placebo_rate}%)"
        else:
            vs_placebo = f"Worse (Drug {drug_rate}% vs Placebo {placebo_rate}%)"
    else:
        vs_placebo = "No placebo comparison available"

    if soc_comparison == "better":
        vs_soc = f"Better than SoC (Drug {drug_rate}% vs {soc.get('drug', 'SoC')} {soc_rate}%)"
    elif soc_comparison == "worse":
        vs_soc = f"Worse than SoC (Drug {drug_rate}% vs {soc.get('drug', 'SoC')} {soc_rate}%)"
    else:
        vs_soc = f"Similar to SoC (Drug {drug_rate}% vs {soc.get('drug', 'SoC')} {soc_rate}%)"

    # ── BUILD RESULT (per documentation Section 8) ──
    result = {
        # Tolerability Score
        "tolerability_score": f"{final_score}/5",
        "score_numeric": final_score,

        # Supporting Data
        "supporting_data": {
            "discontinuation_rate_drug": f"{drug_rate}%" if drug_rate is not None else "N/A",
            "discontinuation_rate_soc": f"{soc_rate}% ({soc.get('drug', 'Unknown')})",
            "soc_source": soc.get("source", "Unknown"),
            "difference": f"{diff_vs_soc}%" if diff_vs_soc is not None else "N/A",
        },

        # Side Effect Summary
        "side_effect_summary": {
            "key_aes": key_aes_str,
            "frequency": frequency_str,
            "severity": severity_str,
        },

        # Comparison
        "comparison": {
            "vs_placebo": vs_placebo,
            "vs_soc": vs_soc,
        },

        # Guardrail
        "guardrail": "FAIL" if guardrail_fail else "PASS",
        "guardrail_reason": "Drug is not competitive on tolerability (discontinuation ≥ SoC with persistent/managed AEs)" if guardrail_fail else None,

        # Justification (single comprehensive field)
        "justification": await _generate_tolerability_rationale(
            molecule, final_score, base_score, base_reason, soc_adjustment, soc_comparison,
            burden_adjustment, burden_level, drug_rate, placebo_rate, soc_rate,
            soc.get("drug", "Unknown"), rates["trials_used"], guardrail_fail,
            weighted_trials[:5]
        ),

        # Internal data (for debugging/audit)
        "_scoring_breakdown": {
            "base_score": base_score,
            "base_reason": base_reason,
            "soc_adjustment": soc_adjustment,
            "burden_adjustment": burden_adjustment,
            "trials_used": rates["trials_used"],
        },
    }

    # Generate markdown table
    result["markdown_table"] = _generate_tolerability_markdown_table(result, molecule)

    return result


def _generate_tolerability_markdown_table(output: dict, molecule: str) -> str:
    """Generate a markdown table from tolerability assessment output."""

    def _v(val):
        """Return value or N/A if empty/None."""
        sv = str(val).strip() if val is not None else ""
        return sv if sv and sv != "None" else "N/A"

    supporting = output.get("supporting_data", {})
    side_effects = output.get("side_effect_summary", {})
    comparison = output.get("comparison", {})
    breakdown = output.get("_scoring_breakdown", {})

    # Convert rationale newlines to <br> for table cell
    rationale = output.get("justification", "")
    rationale_for_table = rationale.replace("\n\n", "<br><br>").replace("\n", "<br>")

    # Build markdown table
    table_lines = [
        f"## Tolerability Assessment: {molecule}\n",
        "| Molecule | Tolerability Score | Discontinuation Rate (Drug) | Discontinuation Rate (SoC) | Difference | Key AEs | Frequency | Severity | vs Placebo | vs SoC | Guardrail | Rationale |",
        "|----------|-------------------|----------------------------|---------------------------|------------|---------|-----------|----------|------------|--------|-----------|-----------|",
    ]

    # Add data row
    table_lines.append(
        f"| {_v(molecule)} "
        f"| {_v(output.get('tolerability_score'))} "
        f"| {_v(supporting.get('discontinuation_rate_drug'))} "
        f"| {_v(supporting.get('discontinuation_rate_soc'))} "
        f"| {_v(supporting.get('difference'))} "
        f"| {_v(side_effects.get('key_aes'))} "
        f"| {_v(side_effects.get('frequency'))} "
        f"| {_v(side_effects.get('severity'))} "
        f"| {_v(comparison.get('vs_placebo'))} "
        f"| {_v(comparison.get('vs_soc'))} "
        f"| {_v(output.get('guardrail'))} "
        f"| {rationale_for_table} |"
    )

    # Add scoring breakdown
    table_lines.append(f"\n*Scoring: Base {breakdown.get('base_score', 'N/A')} + SoC adj ({breakdown.get('soc_adjustment', 0):+d}) + Burden adj ({breakdown.get('burden_adjustment', 0):+d}) = {output.get('score_numeric', 'N/A')}/5*")
    table_lines.append(f"*Trials used: {breakdown.get('trials_used', 'N/A')}*")

    # Add detailed rationale section
    table_lines.append(f"\n\n### Detailed Rationale\n\n{rationale}")

    return "\n".join(table_lines)


async def _generate_tolerability_rationale(
    molecule: str,
    final_score: int,
    base_score: int,
    base_reason: str,
    soc_adj: int,
    soc_comp: str,
    burden_adj: int,
    burden_level: str,
    drug_rate: float,
    placebo_rate: float,
    soc_rate: float,
    soc_drug: str,
    trials_used: int,
    guardrail_fail: bool,
    top_trials: list[dict],
) -> str:
    """Generate tolerability rationale using Gemini.

    Includes all required elements per documentation:
    - Data sources
    - Extracted values
    - Calculations
    - Adjustments applied
    - Final reasoning
    """
    # Collect data from top trials
    common_aes = []
    severities = []
    persistences = []
    trial_ids = []

    for t in top_trials[:5]:
        ae = t.get("Common AEs", "")
        if ae and ae != "N/A":
            common_aes.append(ae)
        sev = t.get("AE Severity", "")
        if sev and sev != "N/A":
            severities.append(sev)
        pers = t.get("AE Persistence", "")
        if pers and pers != "N/A":
            persistences.append(pers)
        tid = t.get("Trial ID", "")
        if tid and tid != "N/A":
            trial_ids.append(tid)

    ae_summary = "; ".join(common_aes[:3]) if common_aes else "Not reported"
    severity_summary = ", ".join(set(severities)) if severities else "Not reported"
    persistence_summary = ", ".join(set(persistences)) if persistences else "Not reported"
    sources = ", ".join(trial_ids[:5]) if trial_ids else f"{trials_used} clinical trials"

    # Calculate difference vs SoC
    diff_vs_soc = round(drug_rate - soc_rate, 2) if drug_rate is not None else "N/A"
    diff_vs_placebo = round(drug_rate - placebo_rate, 2) if drug_rate is not None and placebo_rate is not None else "N/A"

    prompt = f"""Generate a comprehensive tolerability justification for {molecule}.

=== EXTRACTED DATA ===
Data Sources: {sources}
Number of Trials: {trials_used}

Discontinuation Rates:
- Drug Arm: {drug_rate}%
- Placebo Arm: {placebo_rate}%
- Standard of Care ({soc_drug}): {soc_rate}%
- Difference vs Placebo: {diff_vs_placebo}%
- Difference vs SoC: {diff_vs_soc}%

Side Effect Profile:
- Key Adverse Events: {ae_summary}
- Severity: {severity_summary}
- Persistence: {persistence_summary}
- Patient Burden Level: {burden_level.replace('_', ' ')}

=== SCORING CALCULATION ===
Step 1 - Base Score: {base_score}/5
  Reason: {base_reason}

Step 2 - SoC Adjustment: {soc_adj:+d}
  Comparison vs {soc_drug}: {soc_comp}

Step 3 - Burden Adjustment: {burden_adj:+d}
  Burden Level: {burden_level.replace('_', ' ')}

Step 4 - Final Calculation:
  {base_score} (base) + ({soc_adj}) (SoC adj) + ({burden_adj}) (burden adj) = {final_score}

=== GUARDRAIL CHECK ===
Status: {"FAIL" if guardrail_fail else "PASS"}
{"Reason: Drug discontinuation rate ≥ SoC AND side effects are persistent/require management" if guardrail_fail else "Drug meets tolerability threshold"}

=== INSTRUCTIONS ===
Write a structured justification that includes ALL of the following sections:

the data sources are List the clinical trials used
the discontinuation rates are State the discontinuation rates (drug, placebo, SoC) and key AEs
the calculations are Show the weighted average calculation and scoring steps
the adjustments are Explain each adjustment applied (SoC comparison, patient burden)
the final reasoning is Conclude with overall tolerability assessment and final score

IMPORTANT: Do NOT use numbered headings (like "1.", "2.", etc.). Start each section with the heading name directly (e.g., "the data sources are" not "1. DATA SOURCES:").
Use clear paragraph format. Include specific numbers and percentages. Do not skip any section.
"""

    try:
        print(f"  📝 Calling Gemini for tolerability rationale generation...")
        response = await gemini_client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        
        rationale = response.text.strip() if response.text else ""
        print(f"  ✓ Rationale generated ({len(rationale)} chars)")

        # Add guardrail warning if failed
        if guardrail_fail:
            rationale += "\n\n⚠️ GUARDRAIL FAIL: Drug is not competitive on tolerability (discontinuation ≥ SoC with persistent/managed AEs)."

        return rationale

    except Exception as e:
        print(f"  ⚠️ Rationale generation error: {e}")
        # Fallback to structured format
        return _build_simple_justification(
            final_score, base_score, base_reason, soc_adj, soc_comp,
            burden_adj, burden_level, drug_rate, placebo_rate, soc_rate,
            soc_drug, trials_used, guardrail_fail, top_trials
        )


def _build_simple_justification(final_score, base_score, base_reason, soc_adj, soc_comp,
                                burden_adj, burden_level, drug_rate, placebo_rate, soc_rate,
                                soc_drug, trials_used, guardrail_fail, top_trials=None) -> str:
    """Fallback structured justification if Gemini fails."""
    # Collect trial IDs if available
    trial_ids = []
    common_aes = []
    if top_trials:
        for t in top_trials[:5]:
            tid = t.get("Trial ID", "")
            if tid and tid != "N/A":
                trial_ids.append(tid)
            ae = t.get("Common AEs", "")
            if ae and ae != "N/A":
                common_aes.append(ae)

    sources = ", ".join(trial_ids) if trial_ids else f"{trials_used} clinical trials"
    ae_summary = "; ".join(common_aes[:2]) if common_aes else "Not reported"

    diff_vs_soc = round(drug_rate - soc_rate, 2) if drug_rate is not None else "N/A"

    lines = [
        "**DATA SOURCES**",
        f"Based on {sources}.",
        "",
        "**EXTRACTED VALUES**",
        f"Discontinuation Rate (Drug): {drug_rate}%" if drug_rate else "Discontinuation Rate (Drug): N/A",
        f"Discontinuation Rate (Placebo): {placebo_rate}%" if placebo_rate else "Discontinuation Rate (Placebo): N/A",
        f"Discontinuation Rate (SoC - {soc_drug}): {soc_rate}%",
        f"Difference vs SoC: {diff_vs_soc}%",
        f"Key Adverse Events: {ae_summary}",
        "",
        "**CALCULATIONS**",
        f"Weighted average discontinuation rate calculated across {trials_used} trials.",
        f"Base Score: {base_score}/5 ({base_reason})",
        "",
        "**ADJUSTMENTS**",
        f"SoC Adjustment: {soc_adj:+d} (drug is {soc_comp} than {soc_drug})",
        f"Burden Adjustment: {burden_adj:+d} (patient burden: {burden_level.replace('_', ' ')})",
        "",
        "**FINAL REASONING**",
        f"Final Score = {base_score} + ({soc_adj}) + ({burden_adj}) = {final_score}/5",
    ]

    if guardrail_fail:
        lines.append("")
        lines.append("⚠️ **GUARDRAIL FAIL**: Drug is not competitive on tolerability (discontinuation ≥ SoC with persistent/managed AEs).")

    return "\n".join(lines)
