"""Dimension III - Clinical Efficacy Scorer

Phase-Anchored Efficacy Scoring for clinical trial data with rationale generation.

Scoring rules (from documentation):
  - Phase 3 data -> no penalty
  - Phase 2 data -> x0.85 penalty
  - Phase 1 data -> x0.65 penalty
  - Score table (applied to all 4 endpoints):
      >=22% -> 5 | 16-21.9% -> 4 | 10-15.9% -> 3 | 5-9.9% -> 2 | <5% -> 1
  - Weighted final score:
      Weight Loss 0.40 | HbA1c 0.40 | MASH 0.10 | ALT 0.10
"""
from __future__ import annotations

import os
import json
from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

# Initialize Gemini for rationale generation
genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# -- Constants (all from documentation) ----------------------------------------

SCORE_TABLE = [
    (22.0, 5),   # >= 22%      -> 5
    (16.0, 4),   # 16-21.9%   -> 4
    (10.0, 3),   # 10-15.9%   -> 3
    (5.0,  2),   # 5-9.9%     -> 2
    (0.0,  1),   # < 5%       -> 1
]

ENDPOINT_WEIGHTS = {
    "weight_loss": 0.40,
    "hba1c":       0.40,
    "mash":        0.10,
    "alt":         0.10,
}

# Maps endpoint key -> field name in the trial dict
FIELD_MAP = {
    "weight_loss": "weight_change_pct",
    "hba1c":       "hba1c_change_pct",
    "mash":        "mash_resolution_pct",
    "alt":         "alt_reduction_pct",
}

PHASE_PENALTY = {3: 1.00, 2: 0.85, 1: 0.65}


# -- Private helpers -----------------------------------------------------------

def _parse_phase(raw) -> int | None:
    """Normalise phase string/number to int 1/2/3. Returns None if unparseable."""
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
        if v >= 3:
            return 3
        if v >= 2:
            return 2
        return 1
    except ValueError:
        return None


def _parse_float(raw) -> float | None:
    """Return float or None for missing/N/A values."""
    if raw is None:
        return None
    s = str(raw).strip()
    if s.lower() in ("n/a", "", "0", "none"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _pct_to_score(pct: float) -> int:
    """Convert adjusted % to 1-5 score using the documented threshold table."""
    for threshold, score in SCORE_TABLE:
        if pct >= threshold:
            return score
    return 1


# -- Endpoint scoring ----------------------------------------------------------

def score_endpoint(trials: list[dict], value_field: str) -> dict:
    """
    Score a single endpoint across all trials for a molecule.

    Algorithm:
      1. Validate and parse all trials
      2. Phase-anchored selection: prefer Phase 3, fall back to Phase 2 (x0.85),
         then Phase 1 (x0.65)
      3. Within selected phase: GROUP BY trial_id and select highest dosage per trial
      4. Take highest value across all deduplicated trials in that phase
      5. Apply phase penalty -> convert to 1-5 score

    Returns:
        {
          "best_value": float | None,  # penalty-adjusted % used for scoring
          "raw_value":  float | None,  # before phase penalty
          "phase_used": int | None,    # 1, 2, or 3
          "penalty":    float,         # 1.0 / 0.85 / 0.65
          "score":      int | None,    # 1-5, or None if no valid data
          "trial_details": dict,       # Info about selected trial (trial_id, dosage, durations)
          "reason":     str
        }
    """
    # -- Step 1: Validate and parse all trials ---------------------------------
    valid = []
    for t in trials:
        phase = _parse_phase(t.get("phase"))
        value = _parse_float(t.get(value_field))
        n     = _parse_float(t.get("trial_size")) or 0
        trial_id = t.get("trial_id") or t.get("Trial ID")

        if phase is None or value is None or n <= 0:
            continue

        # Assign unique ID if missing (treats as separate trial)
        if not trial_id:
            trial_id = f"__unknown_{id(t)}"

        valid.append({
            "phase": phase,
            "value": value,
            "n": n,
            "trial_id": trial_id,
            "full_trial": t  # Preserve full trial data
        })

    if not valid:
        return {
            "best_value": None, "raw_value": None,
            "phase_used": None, "penalty": 1.0,
            "score": None, "reason": "No valid data for this endpoint",
        }

    # -- Step 2-5: Phase-anchored selection with multi-dosage handling ---------
    for target_phase in (3, 2, 1):
        # Get all trials in target phase
        phase_trials = [r for r in valid if r["phase"] == target_phase]
        if not phase_trials:
            continue

        # -- Step 3: Deduplicate multi-dosage trials within this phase ---------
        trial_groups = {}
        for t in phase_trials:
            tid = t["trial_id"]
            if tid not in trial_groups:
                trial_groups[tid] = []
            trial_groups[tid].append(t)

        # Select highest-performing dosage per trial
        deduplicated = []
        for tid, dosage_arms in trial_groups.items():
            best_dosage = max(dosage_arms, key=lambda x: x["value"])
            deduplicated.append(best_dosage)

        # -- Step 4: Take highest value across all deduplicated trials ---------
        best = max(deduplicated, key=lambda r: r["value"])
        raw  = best["value"]
        pen  = PHASE_PENALTY[target_phase]
        adj  = raw * pen

        # Extract trial details for rationale
        full_trial = best.get("full_trial", {})
        trial_details = {
            "trial_id": best["trial_id"],
            "dosage": full_trial.get("dosage", "N/A"),
            "weight_duration": full_trial.get("weight_duration", "N/A"),
            "hba1c_duration": full_trial.get("hba1c_duration", "N/A"),
            "mash_duration": full_trial.get("mash_duration", "N/A"),
            "alt_duration": full_trial.get("alt_duration", "N/A"),
        }

        return {
            "best_value": round(adj, 4),
            "raw_value":  round(raw, 4),
            "phase_used": target_phase,
            "penalty":    pen,
            "score":      _pct_to_score(adj),
            "trial_details": trial_details,
            "reason": (
                f"Phase {target_phase} data used"
                + (f" (x{pen} penalty applied)" if pen < 1 else "")
            ),
        }

    return {
        "best_value": None, "raw_value": None,
        "phase_used": None, "penalty": 1.0,
        "score": None, "reason": "Unexpected state",
    }


# -- Main scoring function -----------------------------------------------------

def compute_clinical_efficacy_score(data: dict) -> dict:
    """
    Compute the clinical efficacy score (1-5 scale) for a molecule.

    Args:
        data: dict with keys:
            - molecule / molecule_name  (str)
            - trials                    (list of trial dicts from pipeline)
            - total_trials              (int, optional)
            - completeness              (float, optional)

    Returns:
        {
          "molecule":        str,
          "total_trials":    int,
          "endpoints": {
            "weight_loss": { best_value, raw_value, phase_used, penalty, score, reason },
            "hba1c":       { ... },
            "mash":        { ... },
            "alt":         { ... },
          },
          "weighted_score":  float,   # weighted average (1-5 scale, rounded to 3 decimals)
          "score_breakdown": str,     # human-readable table
          "data_coverage":   str,     # e.g. "3/4 endpoints scored (missing: mash)"
        }
    """
    molecule = data.get("molecule") or data.get("molecule_name", "Unknown")
    trials   = data.get("trials", [])
    total    = data.get("total_trials", len(trials))

    # Score each endpoint
    endpoint_results = {
        ep: score_endpoint(trials, field)
        for ep, field in FIELD_MAP.items()
    }

    # Weighted average - missing endpoints contribute 0 (per documentation)
    # "Weighted average of all the endpoints score" - no re-normalization
    score_sum   = 0.0
    scored_eps  = []
    missing_eps = []

    for ep, result in endpoint_results.items():
        w = ENDPOINT_WEIGHTS[ep]
        if result["score"] is not None:
            score_sum  += result["score"] * w
            scored_eps.append(ep)
        else:
            # Missing endpoint contributes 0 to weighted average
            score_sum += 0 * w
            missing_eps.append(ep)

    # Total weights sum to 1.0, so score_sum is already the weighted average
    weighted_score = score_sum

    # Human-readable breakdown
    lines = []
    for ep, result in endpoint_results.items():
        w_pct = int(ENDPOINT_WEIGHTS[ep] * 100)
        if result["score"] is not None:
            lines.append(
                f"  {ep:12} | adj={result['best_value']:.2f}%  "
                f"score={result['score']}  weight={w_pct}%  ({result['reason']})"
            )
        else:
            lines.append(
                f"  {ep:12} | N/A  weight={w_pct}%  ({result['reason']})"
            )

    coverage = (
        f"{len(scored_eps)}/4 endpoints scored"
        + (f" (missing: {', '.join(missing_eps)})" if missing_eps else "")
    )

    return {
        "molecule":        molecule,
        "total_trials":    total,
        "endpoints":       endpoint_results,
        "weighted_score":  round(weighted_score, 3),
        "score_breakdown": "\n".join(lines),
        "data_coverage":   coverage,
    }


# -- Rationale Generation ------------------------------------------------------

def generate_rationale(molecule: str, clinical_data: dict, score_result: dict) -> str:
    """Generate comprehensive narrative rationale for the clinical efficacy score using Gemini.

    Args:
        molecule: The drug molecule name
        clinical_data: Dict containing trials list
        score_result: Output from compute_clinical_efficacy_score()

    Returns:
        Formatted narrative rationale explaining the score
    """
    trials = clinical_data.get("trials", [])

    # Prepare trial summary data for Gemini (limit to top 20 for context)
    trial_summaries = []
    for trial in trials[:20]:
        summary = {
            "id": trial.get("trial_id", "N/A"),
            "phase": trial.get("phase", "N/A"),
            "size": trial.get("trial_size", "N/A"),
            "dosage": trial.get("dosage", "N/A"),
            "weight_loss": trial.get("weight_change_pct", "N/A"),
            "weight_duration": trial.get("weight_duration", "N/A"),
            "hba1c": trial.get("hba1c_change_pct", "N/A"),
            "hba1c_duration": trial.get("hba1c_duration", "N/A"),
            "mash": trial.get("mash_resolution_pct", "N/A"),
            "mash_duration": trial.get("mash_duration", "N/A"),
            "alt": trial.get("alt_reduction_pct", "N/A"),
            "alt_duration": trial.get("alt_duration", "N/A"),
        }
        trial_summaries.append(summary)

    # Extract endpoint scores and selected trial details
    endpoints = score_result.get("endpoints", {})

    prompt = f"""You are a clinical pharmacology expert. Generate a comprehensive, evidence-based rationale explaining the clinical efficacy score for {molecule}.

SCORING RESULTS:
- Clinical Efficacy Score: {score_result['weighted_score']} / 5
- Coverage: {score_result['data_coverage']}

ENDPOINT PERFORMANCE (These are the EXACT trials used for scoring):
{json.dumps({
    "Weight Loss (40% weight)": {
        "score": endpoints.get("weight_loss", {}).get("score"),
        "best_value": endpoints.get("weight_loss", {}).get("best_value"),
        "raw_value": endpoints.get("weight_loss", {}).get("raw_value"),
        "phase_used": endpoints.get("weight_loss", {}).get("phase_used"),
        "trial_used": endpoints.get("weight_loss", {}).get("trial_details", {}).get("trial_id"),
        "dosage": endpoints.get("weight_loss", {}).get("trial_details", {}).get("dosage"),
        "duration": endpoints.get("weight_loss", {}).get("trial_details", {}).get("weight_duration"),
    },
    "HbA1c Reduction (40% weight)": {
        "score": endpoints.get("hba1c", {}).get("score"),
        "best_value": endpoints.get("hba1c", {}).get("best_value"),
        "raw_value": endpoints.get("hba1c", {}).get("raw_value"),
        "phase_used": endpoints.get("hba1c", {}).get("phase_used"),
        "trial_used": endpoints.get("hba1c", {}).get("trial_details", {}).get("trial_id"),
        "dosage": endpoints.get("hba1c", {}).get("trial_details", {}).get("dosage"),
        "duration": endpoints.get("hba1c", {}).get("trial_details", {}).get("hba1c_duration"),
    },
    "MASH Resolution (10% weight)": {
        "score": endpoints.get("mash", {}).get("score"),
        "best_value": endpoints.get("mash", {}).get("best_value"),
        "raw_value": endpoints.get("mash", {}).get("raw_value"),
        "phase_used": endpoints.get("mash", {}).get("phase_used"),
        "trial_used": endpoints.get("mash", {}).get("trial_details", {}).get("trial_id"),
        "dosage": endpoints.get("mash", {}).get("trial_details", {}).get("dosage"),
        "duration": endpoints.get("mash", {}).get("trial_details", {}).get("mash_duration"),
    },
    "ALT Reduction (10% weight)": {
        "score": endpoints.get("alt", {}).get("score"),
        "best_value": endpoints.get("alt", {}).get("best_value"),
        "raw_value": endpoints.get("alt", {}).get("raw_value"),
        "phase_used": endpoints.get("alt", {}).get("phase_used"),
        "trial_used": endpoints.get("alt", {}).get("trial_details", {}).get("trial_id"),
        "dosage": endpoints.get("alt", {}).get("trial_details", {}).get("dosage"),
        "duration": endpoints.get("alt", {}).get("trial_details", {}).get("alt_duration"),
    }
}, indent=2)}

TOTAL TRIALS ANALYZED: {score_result['total_trials']}

IMPORTANT: Use the trial_id, dosage, and duration from the "ENDPOINT PERFORMANCE" section above. Those are the EXACT trials selected for each endpoint's score. Do not guess or infer from other trials.

SCORING METHODOLOGY:
- Score ranges: 5 = >=22%, 4 = 16-21.9%, 3 = 10-15.9%, 2 = 5-9.9%, 1 = <5%
- Phase penalties: Phase 3 = no penalty, Phase 2 = x0.85, Phase 1 = x0.65
- Weighted average: Weight Loss (40%) + HbA1c (40%) + MASH (10%) + ALT (10%)

YOUR TASK:
Write a comprehensive 4-5 paragraph clinical rationale that:

1. **Opening paragraph**: State the clinical efficacy score (e.g., "4.85 - Exceptional") and provide a high-level summary of {molecule}'s clinical performance.

2. **Weight Loss paragraph**: Discuss weight loss efficacy. MUST reference:
   - Specific percentage achieved and duration (e.g., "15.2% at 68 weeks")
   - Dosage used (e.g., "2.4 mg once weekly")
   - Phase of data and key trial IDs if notable
   - Clinical benchmarks comparison (e.g., >15% is excellent, 10-15% is good)

3. **HbA1c Reduction paragraph**: Discuss glycemic control efficacy. MUST reference:
   - Specific percentage achieved and duration (e.g., "1.8% at 26 weeks")
   - Dosage used (e.g., "1.0 mg once weekly")
   - Phase of data and key trial IDs
   - Clinical significance comparison (e.g., >2% is excellent for diabetes management)

4. **MASH/ALT paragraph** (if data available): Discuss liver health endpoints with duration context. If N/A, briefly mention why (not the primary indication or limited data).

5. **Concluding paragraph**: Summarize the overall magnitude score rationale, mention data quality (phase distribution, sample sizes), and state why this score represents the molecule's true clinical efficacy profile.

WRITING GUIDELINES:
- MUST include dosage and duration details for each endpoint discussed (e.g., "achieved 15% weight loss at 2.4 mg over 68 weeks")
- Use specific numbers from the data (percentages, trial IDs, phase info, dosages, durations)
- Be objective and evidence-based
- Use medical terminology appropriately
- Reference clinical significance thresholds
- Keep paragraphs concise (3-5 sentences each)
- Format as plain text with paragraph breaks (no markdown, no headers, no bullets)
- Do not use phrases like "The rationale is..." - write the rationale directly

IMPORTANT: Write in a clinical, professional tone as if this is documentation for regulatory or pharma stakeholders. Focus on data and clinical significance.

Generate the rationale now:"""

    try:
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]
        config = types.GenerateContentConfig(
            temperature=0.3,  # Lower temperature for factual output
            response_mime_type="text/plain"
        )

        response_text = ""
        for chunk in genai_client.models.generate_content_stream(
            model="gemini-3-flash-preview",
            contents=contents,
            config=config,
        ):
            if chunk.text:
                response_text += chunk.text

        rationale = response_text.strip()
        rationale = rationale.replace("\n\n\n", "\n\n")  # Clean up triple breaks

        return rationale

    except Exception as e:
        print(f"  [Rationale] Failed to generate: {e}")
        # Fallback
        return f"{molecule} received a clinical efficacy score of {score_result['weighted_score']}/5 based on analysis of {score_result['total_trials']} trials. {score_result['data_coverage']}."
