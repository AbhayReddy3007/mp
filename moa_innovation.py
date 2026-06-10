"""MoA Innovation Assessment Tool (Dimension 1)

This module evaluates how innovative and strategically meaningful a molecule's
mechanism of action is for a given indication using Gemini with Google Search.
"""

import os
import json
import asyncio
import time
from datetime import datetime
from dotenv import load_dotenv
from json_repair import repair_json

load_dotenv()

from google import genai
from google.genai import types

from market_potential.bq_client import run_query
from market_potential.constants import MOA_INNOVATION_QUERY

# Initialize Gemini client
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Configuration
MODEL = "gemini-2.5-flash"
MAX_RETRIES = 3
INITIAL_BACKOFF = 2.0

# Use query from constants
MOA_QUERY = MOA_INNOVATION_QUERY

# Source registries for comprehensive search
SOURCES = {
    "clinical_registries": [
        {"name": "ClinicalTrials.gov", "url": "https://clinicaltrials.gov/", "description": "US clinical trials registry"},
        {"name": "EU Clinical Trials Register", "url": "https://www.clinicaltrialsregister.eu/", "description": "European clinical trials"},
        {"name": "WHO ICTRP", "url": "https://trialsearch.who.int/", "description": "WHO International Clinical Trials Registry Platform"},
    ],
    "regulatory": [
        {"name": "FDA", "url": "https://www.fda.gov/", "description": "US Food and Drug Administration"},
        {"name": "EMA", "url": "https://www.ema.europa.eu/", "description": "European Medicines Agency"},
        {"name": "FDA Orange Book", "url": "https://www.accessdata.fda.gov/scripts/cder/ob/", "description": "FDA approved drugs"},
    ],
    "journals": [
        {"name": "PubMed", "url": "https://pubmed.ncbi.nlm.nih.gov/", "description": "Biomedical literature database"},
        {"name": "Nature Medicine", "url": "https://www.nature.com/nm/", "description": "Peer-reviewed journal"},
        {"name": "NEJM", "url": "https://www.nejm.org/", "description": "New England Journal of Medicine"},
        {"name": "The Lancet", "url": "https://www.thelancet.com/", "description": "Peer-reviewed medical journal"},
    ],
    "patents": [
        {"name": "Google Patents", "url": "https://patents.google.com/", "description": "Patent search"},
        {"name": "USPTO", "url": "https://www.uspto.gov/", "description": "US Patent and Trademark Office"},
        {"name": "WIPO", "url": "https://www.wipo.int/", "description": "World Intellectual Property Organization"},
    ],
}


def _extract_json_from_response(text: str) -> str:
    """Extract JSON object/array from Gemini response text."""
    if not text or not text.strip():
        return ""

    text = text.strip()

    # Remove markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # Find first JSON structure
    obj_start = text.find('{')
    arr_start = text.find('[')

    if obj_start == -1 and arr_start == -1:
        return ""
    elif obj_start == -1:
        start = arr_start
        opening, closing = '[', ']'
    elif arr_start == -1:
        start = obj_start
        opening, closing = '{', '}'
    else:
        start = min(obj_start, arr_start)
        opening, closing = ('{', '}') if start == obj_start else ('[', ']')

    # Track depth to find matching close
    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start:i+1]

    return text[start:] if depth > 0 else ""


async def _gemini_search_call(prompt: str, temperature: float = 0) -> dict:
    """Make a Gemini API call with Google Search enabled and retry logic."""

    tools = [types.Tool(googleSearch=types.GoogleSearch())]
    config = types.GenerateContentConfig(tools=tools, temperature=temperature)

    retry_count = 0
    backoff_delay = INITIAL_BACKOFF

    while retry_count <= MAX_RETRIES:
        try:
            response = await gemini_client.aio.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=config
            )

            if not response.text:
                return {}

            json_str = _extract_json_from_response(response.text.strip())
            if json_str:
                data = repair_json(json_str)
                if isinstance(data, str):
                    data = json.loads(data)
                # Gemini sometimes wraps the object in an array — unwrap it
                if isinstance(data, list):
                    data = data[0] if (data and isinstance(data[0], dict)) else {"raw_response": response.text.strip(), "_parsed_list": data}
                return data

            # Return raw text if no JSON found
            return {"raw_response": response.text.strip()}

        except Exception as e:
            error_str = str(e).lower()

            if any(err in error_str for err in ["429", "rate limit", "quota", "resource exhausted"]):
                retry_count += 1
                if retry_count > MAX_RETRIES:
                    print(f"  [MoA] Max retries exceeded: {e}")
                    raise

                print(f"  [MoA] Rate limit - waiting {backoff_delay:.1f}s (retry {retry_count}/{MAX_RETRIES})...")
                await asyncio.sleep(backoff_delay)
                backoff_delay *= 2
            else:
                print(f"  [MoA] API error: {e}")
                raise

    return {}


async def _step1_get_moa_from_bq(drug_name: str) -> dict:
    """Step 1: Get MoA and Indication from BigQuery.

    If multiple indications exist, uses LLM to identify the PRIMARY indication.
    """
    print(f"[MoA Step 1] Querying BigQuery for {drug_name} MoA data...")

    try:
        rows = run_query(MOA_QUERY)

        # Collect ALL matching rows for this drug
        matching_rows = []
        for row in rows:
            name = row.get("cleaned_generic_name", "").lower()
            if name == drug_name.lower() or drug_name.lower() in name:
                matching_rows.append({
                    "drug_name": row.get("cleaned_generic_name"),
                    "mechanism_of_action": row.get("Mechanism_of_Action"),
                    "indication": row.get("Cleaned_Indication"),
                })

        if not matching_rows:
            print(f"  Drug not found in internal database, will use Google Search")
            return {"drug_name": drug_name, "mechanism_of_action": None, "indication": None}

        # Get unique indications
        indications = list(set(r["indication"] for r in matching_rows if r.get("indication")))
        moa = matching_rows[0].get("mechanism_of_action")  # MoA should be same across rows

        print(f"  Found {len(indications)} indication(s): {indications}")

        # If only one indication, use it directly
        if len(indications) == 1:
            result = {
                "drug_name": matching_rows[0]["drug_name"],
                "mechanism_of_action": moa,
                "indication": indications[0],
                "all_indications": indications,
                "source": "BigQuery internal database"
            }
            print(f"  Using: MoA={moa}, Indication={indications[0]}")
            return result

        # Multiple indications - ask LLM to identify the PRIMARY one
        print(f"  Multiple indications found, identifying primary indication...")
        primary_indication = await _identify_primary_indication(drug_name, moa, indications)

        result = {
            "drug_name": matching_rows[0]["drug_name"],
            "mechanism_of_action": moa,
            "indication": primary_indication,
            "all_indications": indications,
            "source": "BigQuery internal database + LLM primary selection"
        }
        print(f"  Primary indication selected: {primary_indication}")
        return result

    except Exception as e:
        print(f"  BigQuery error: {e}, will use Google Search")
        return {"drug_name": drug_name, "mechanism_of_action": None, "indication": None}


async def _identify_primary_indication(drug_name: str, moa: str, indications: list) -> str:
    """Use LLM to identify the PRIMARY indication for MoA assessment."""

    indications_str = "\n".join(f"- {ind}" for ind in indications)

    prompt = f"""For the drug {drug_name} with mechanism "{moa}", identify the PRIMARY indication.

Available indications from database:
{indications_str}

The PRIMARY indication is the one where:
1. The drug was FIRST approved or is MOST established
2. The mechanism of action is MOST validated and proven
3. The drug has the STRONGEST clinical evidence

Return JSON:
{{
    "primary_indication": "the selected primary indication exactly as listed above"
}}"""

    result = await _gemini_search_call(prompt)

    primary = result.get("primary_indication", "")

    # Validate that returned indication is in the list
    if primary and primary in indications:
        print(f"  LLM selected: {primary}")
        return primary

    # If LLM returned something not in list, try to match
    for ind in indications:
        if primary.lower() in ind.lower() or ind.lower() in primary.lower():
            print(f"  LLM selected (matched): {ind}")
            return ind

    # Fallback to first indication
    print(f"  LLM selection failed, using first: {indications[0]}")
    return indications[0]


async def _step2_build_mechanism_landscape(drug_name: str, moa: str, indication: str) -> dict:
    """Step 2: Build current mechanism landscape for the indication."""
    print(f"[MoA Step 2] Building mechanism landscape for {indication}...")

    prompt = f"""Analyze the current treatment landscape for {indication}.

Drug being assessed: {drug_name}
Drug's mechanism: {moa}

Search comprehensively for:
1. ALL currently approved drugs for {indication} - list each with its mechanism of action
2. Late-stage (Phase 2/3) pipeline drugs in {indication} - list each with its mechanism
3. Current standard of care (SOC) treatments and their mechanisms
4. Role of each mechanism in treatment (first-line, second-line, emerging)

Sources to search:
- FDA Orange Book approved drugs
- ClinicalTrials.gov Phase 2/3 trials
- Treatment guidelines and SOC documents
- EMA approved drugs in Europe

Return JSON:
{{
    "indication": "{indication}",
    "approved_drugs": [
        {{"drug_name": "name", "mechanism": "mechanism of action", "role": "first-line/second-line/etc", "approval_year": "YYYY"}}
    ],
    "pipeline_drugs_phase2_3": [
        {{"drug_name": "name", "mechanism": "mechanism of action", "phase": "Phase 2/3", "status": "Active/Completed"}}
    ],
    "standard_of_care": {{
        "first_line": ["mechanism/drug class list"],
        "second_line": ["mechanism/drug class list"],
        "emerging": ["mechanism/drug class list"]
    }},
    "dominant_mechanism_classes": ["list of major mechanism classes in this indication"],
    "landscape_conclusion": "Summary of whether this indication is dominated by certain mechanisms, crowded, or has room for new mechanisms",
    "sources": ["list of sources consulted"]
}}"""

    result = await _gemini_search_call(prompt)

    if result and not result.get("raw_response"):
        print(f"  Found {len(result.get('approved_drugs', []))} approved drugs, {len(result.get('pipeline_drugs_phase2_3', []))} pipeline drugs")

    return result


async def _step3_classify_moa_position(drug_name: str, moa: str, indication: str, landscape: dict) -> dict:
    """Step 3: Classify whether MoA is FIC, BIC, Me-too, Outdated, or Poor."""
    print(f"[MoA Step 3] Classifying MoA position...")

    landscape_summary = json.dumps(landscape, indent=2) if landscape else "Not available"

    prompt = f"""Based on the mechanism landscape, classify the MoA position for:

Drug: {drug_name}
Mechanism: {moa}
Indication: {indication}

Current landscape:
{landscape_summary}

CLASSIFICATION CRITERIA (from business documentation):

First-in-Class (FIC):
- The first meaningful drug targeting a novel biological mechanism in the indication
- REQUIRES: target/pathway is new in this disease setting
- REQUIRES: no prior approved or clearly established class exists
- NOTE: Biological rationale strength will be assessed separately in Step 4

Best-in-Class (BIC):
- The mechanism is already known, but the asset uses it in a clearly BETTER way
- Examples: dual agonism instead of single target, higher selectivity, better downstream biology
- Requires meaningful mechanistic differentiation (not just dosing/formulation)

Me-too / Fast Follower:
- The drug uses an already validated mechanism
- Offers little real mechanism-level innovation
- Competes mainly on non-MoA factors

Weak / Outdated:
- The mechanism belongs to an older class
- No longer strategically strong because newer mechanisms have replaced it
- Market/guidelines have shifted toward better alternatives

Poor / Invalid:
- The mechanism has weak rationale
- OR has already failed clinically in the same indication
- Without convincing reason this asset is different

QUESTIONS TO ANSWER:
1. Is this the FIRST meaningful mechanism of this type in this indication?
2. If not first, does it still use the mechanism in a clearly BETTER way?
3. If not clearly better, is it just another drug in the same mechanism class?
4. Has the field already moved beyond this mechanism to newer approaches?
5. Is the biology itself questionable or has it failed repeatedly?

Return JSON:
{{
    "moa_classification": "First-in-Class|Best-in-Class|Me-too|Weak/Outdated|Poor/Invalid",
    "is_first_in_class": true/false,
    "is_mechanism_new_to_indication": true/false,
    "no_prior_approved_class": true/false,
    "has_clear_mechanistic_improvement": true/false,
    "improvement_details": "If BIC, explain the mechanistic improvement (dual agonism, selectivity, etc.)",
    "is_same_class_as_soc": true/false,
    "mechanism_differentiation": "How this mechanism differs from existing mechanisms",
    "classification_rationale": "Detailed explanation referencing the criteria above",
    "sources": ["sources consulted"]
}}"""

    result = await _gemini_search_call(prompt)

    if isinstance(result, list):
        result = result[0] if (result and isinstance(result[0], dict)) else {}

    if result and not result.get("raw_response"):
        print(f"  Classification: {result.get('moa_classification', 'Unknown')}")

    return result


async def _step4_check_biological_rationale(drug_name: str, moa: str, indication: str) -> dict:
    """Step 4: Check whether the biological rationale is strong.

    Per documentation: A mechanism should not get a high innovation score only because
    it sounds new. It must also make biological sense.
    """
    print(f"[MoA Step 4] Checking biological rationale...")

    prompt = f"""Evaluate the biological rationale for:

Drug: {drug_name}
Mechanism: {moa}
Indication: {indication}

Search peer-reviewed scientific literature (PubMed, Nature, NEJM, Lancet) to determine:

1. Is the target/pathway clearly linked to disease biology/pathophysiology?
2. Is there scientific evidence that modulating this target affects the disease?
3. Is there strong scientific consensus supporting this mechanism?
4. Are there seminal papers or definitive reviews supporting the target relevance?

IMPORTANT FOR MULTI-MECHANISM DRUGS (dual/triple/quad agonists):
If the drug targets MULTIPLE pathways, evaluate EACH component separately:
- Assess each individual target/pathway's biological rationale
- A combination is considered to have STRONG rationale if the MAJORITY of its components have validated biological rationale
- Novel combinations of individually-validated mechanisms should NOT be penalized

RATIONALE ASSESSMENT CRITERIA (from documentation):

STRONG rationale looks like:
- Clear role of target/pathway in disease pathophysiology
- Strong disease biology support
- Translational or mechanistic literature supporting target relevance
- Expert or review-level discussion that this pathway matters
- For multi-agonists: each component individually has strong/proven rationale

WEAK rationale looks like:
- Target has only loose association with disease
- Little evidence that changing the pathway produces meaningful disease benefit
- Mostly theoretical or early hypothesis-level support
- Repeated uncertainty in published literature

Return JSON:
{{
    "target_pathways": ["List each target/pathway in the mechanism"],
    "is_multi_mechanism": true/false,
    "component_analysis": [
        {{
            "component": "pathway/receptor name",
            "is_validated": true/false,
            "validation_evidence": "approved drugs or positive trials using this component",
            "disease_link": "how this component relates to disease pathophysiology"
        }}
    ],
    "disease_biology_link": {{
        "is_clearly_linked": true/false,
        "explanation": "How the target(s) relate to disease pathophysiology"
    }},
    "key_publications": [
        {{"title": "paper title", "journal": "journal name", "finding": "key finding"}}
    ],
    "is_rationale_strong": true/false,
    "is_rationale_weak": true/false,
    "rationale_explanation": "Detailed explanation - for multi-agonists, explain rationale for each component",
    "sources": ["sources consulted"]
}}"""

    result = await _gemini_search_call(prompt)

    if result and not result.get("raw_response"):
        is_strong = result.get('is_rationale_strong', False)
        is_weak = result.get('is_rationale_weak', False)
        status = "Strong" if is_strong else ("Weak" if is_weak else "Adequate")
        print(f"  Rationale: {status}")

    return result


async def _step5_check_clinical_validation(drug_name: str, moa: str, indication: str) -> dict:
    """Step 5: Check whether the mechanism is already clinically validated."""
    print(f"[MoA Step 5] Checking clinical validation...")

    prompt = f"""Determine if this mechanism is clinically validated:

Drug: {drug_name}
Mechanism: {moa}
Indication: {indication}

Search FDA, ClinicalTrials.gov, and peer-reviewed publications to find:

1. Are there FDA/EMA approved drugs with THIS SAME mechanism in THIS indication?
2. Have there been successful Phase 2/3 clinical trials with this mechanism?
3. Is there published evidence of efficacy in humans?

Return JSON:
{{
    "is_clinically_validated": true/false,
    "validation_strength": "Strong|Moderate|Limited|None",
    "approved_drugs_same_mechanism": [
        {{"drug_name": "name", "approval_agency": "FDA/EMA", "approval_year": "YYYY", "indication": "specific indication"}}
    ],
    "successful_trials": [
        {{"trial_id": "NCT...", "phase": "Phase 2/3", "outcome": "Positive/Met primary endpoint", "publication": "if published"}}
    ],
    "human_efficacy_evidence": {{
        "exists": true/false,
        "summary": "Summary of human efficacy data"
    }},
    "validation_conclusion": "Whether mechanism is validated or novel/unproven",
    "sources": ["FDA, ClinicalTrials.gov, publications"]
}}"""

    result = await _gemini_search_call(prompt)

    if result and not result.get("raw_response"):
        validated = result.get('is_clinically_validated', False)
        print(f"  Clinically validated: {validated}")

    return result


async def _step6_check_mechanism_failures(drug_name: str, moa: str, indication: str) -> dict:
    """Step 6: Check if mechanism has already failed in this indication."""
    print(f"[MoA Step 6] Checking for mechanism failures...")

    prompt = f"""Search for clinical failures of this mechanism:

Drug: {drug_name}
Mechanism: {moa}
Indication: {indication}

Search ClinicalTrials.gov for Completed/Terminated/Withdrawn trials and publications to find:

1. Have other drugs using the SAME mechanism failed in THIS indication?
2. What was the reason for failure - was it the MECHANISM itself or molecule-specific issues?
   - Mechanism failure: Biology doesn't work in humans
   - Molecule failure: PK issues, toxicity, dosing, trial design

IMPORTANT: Distinguish between:
- MECHANISM-LEVEL failure (the biological approach doesn't work)
- MOLECULE-LEVEL failure (the specific drug had issues but mechanism may still be valid)

Return JSON:
{{
    "has_mechanism_failures": true/false,
    "failed_drugs_same_mechanism": [
        {{
            "drug_name": "name",
            "trial_id": "NCT...",
            "failure_reason": "efficacy|safety|trial design|PK|other",
            "failure_type": "mechanism-level|molecule-level",
            "details": "explanation of what went wrong"
        }}
    ],
    "is_mechanism_invalidated": true/false,
    "mechanism_invalidation_evidence": "If mechanism itself is considered invalid, explain why",
    "can_this_drug_overcome_failures": {{
        "assessment": true/false,
        "rationale": "If failures exist, why might this drug succeed where others failed?"
    }},
    "failure_analysis_conclusion": "Summary of failure analysis",
    "sources": ["sources consulted"]
}}"""

    result = await _gemini_search_call(prompt)

    if result and not result.get("raw_response"):
        failures = result.get('has_mechanism_failures', False)
        invalidated = result.get('is_mechanism_invalidated', False)
        print(f"  Has failures: {failures}, Mechanism invalidated: {invalidated}")

    return result


async def _step7_check_mechanistic_improvement(drug_name: str, moa: str, indication: str, classification: dict) -> dict:
    """Step 7: If not FIC, check if meaningfully better than existing class."""
    print(f"[MoA Step 7] Checking for mechanistic improvements...")

    moa_class = classification.get('moa_classification', '') if classification else ''

    if moa_class == 'First-in-Class':
        print("  Skipping - drug is First-in-Class")
        return {"skipped": True, "reason": "Drug is First-in-Class, no comparison needed"}

    prompt = f"""Evaluate if this drug has meaningful mechanistic improvements over existing class:

Drug: {drug_name}
Mechanism: {moa}
Indication: {indication}
Current Classification: {moa_class}

Search for specific mechanistic differentiators (NOT product-level advantages):

VALID mechanistic improvements:
- Dual/multi-agonism instead of single receptor (e.g., GLP-1/GIP dual agonist)
- Better target selectivity
- Biased signaling
- More relevant downstream biology
- Mechanism designed to overcome known class limitations

NOT mechanistic improvements (don't count these):
- Better dosing/formulation
- Device advantages
- Lower price
- Marketing positioning

Return JSON:
{{
    "has_mechanistic_improvement": true/false,
    "improvement_type": "dual-agonism|selectivity|biased-signaling|novel-pathway|none",
    "improvement_details": {{
        "description": "Detailed description of the mechanistic improvement",
        "compared_to": "What existing mechanism/drug class is this compared to",
        "scientific_basis": "Scientific explanation of why this is better"
    }},
    "overcomes_class_limitations": {{
        "yes_no": true/false,
        "limitations_addressed": ["list of class limitations this addresses"]
    }},
    "improvement_assessment": "Significant|Moderate|Minimal|None",
    "sources": ["sources consulted"]
}}"""

    result = await _gemini_search_call(prompt)

    if result and not result.get("raw_response"):
        has_improvement = result.get('has_mechanistic_improvement', False)
        print(f"  Has mechanistic improvement: {has_improvement}")

    return result


async def _step8_check_mechanism_currency(drug_name: str, moa: str, indication: str) -> dict:
    """Step 8: Check if mechanism is still strategically current or outdated."""
    print(f"[MoA Step 8] Checking mechanism currency...")

    prompt = f"""Assess if this mechanism is still strategically relevant:

Drug: {drug_name}
Mechanism: {moa}
Indication: {indication}

Determine:
1. Is this mechanism still part of current standard of care?
2. Has the field shifted to newer, better mechanisms?
3. Is this mechanism still considered clinically important?
4. Are treatment guidelines still recommending this mechanism class?

Return JSON:
{{
    "is_mechanism_current": true/false,
    "mechanism_status": "Current SOC|Emerging|Declining|Outdated",
    "guideline_status": {{
        "still_recommended": true/false,
        "recommendation_level": "First-line|Second-line|Alternative|Not recommended",
        "guidelines_consulted": ["list of guidelines"]
    }},
    "competitive_position": {{
        "newer_mechanisms_exist": true/false,
        "newer_mechanisms": ["list of newer mechanism approaches"],
        "this_mechanism_advantages": ["what this mechanism still offers"],
        "this_mechanism_disadvantages": ["limitations compared to newer approaches"]
    }},
    "strategic_relevance": "High|Moderate|Low|Minimal",
    "currency_conclusion": "Assessment of whether mechanism is still strategically valuable",
    "sources": ["sources consulted"]
}}"""

    result = await _gemini_search_call(prompt)

    if result and not result.get("raw_response"):
        status = result.get('mechanism_status', 'Unknown')
        print(f"  Mechanism status: {status}")

    return result


async def _step9_final_scoring(drug_name: str, moa: str, indication: str, all_data: dict) -> dict:
    """Step 9: Convert full analysis into final classification and score."""
    print(f"[MoA Step 9] Computing final score...")

    # Prepare data summary for scoring
    data_summary = json.dumps({
        "moa_data": all_data.get("step1_moa_data", {}),
        "landscape": all_data.get("step2_landscape", {}),
        "classification": all_data.get("step3_classification", {}),
        "biological_rationale": all_data.get("step4_rationale", {}),
        "clinical_validation": all_data.get("step5_validation", {}),
        "mechanism_failures": all_data.get("step6_failures", {}),
        "mechanistic_improvement": all_data.get("step7_improvement", {}),
        "mechanism_currency": all_data.get("step8_currency", {}),
    }, indent=2, default=str)

    prompt = f"""Based on the complete MoA Innovation analysis, determine the final score:

Drug: {drug_name}
Mechanism: {moa}
Indication: {indication}

Complete Analysis Data:
{data_summary}

SCORING RULES (from documentation - follow EXACTLY):

Score 5 = Exceptional (First-in-Class):
- True FIC mechanism (target/pathway is NEW to this disease setting)
- No prior approved or clearly established class exists
- Strong biological rationale
- Meaningful potential to create a new treatment class
- Do NOT give 5 for novelty alone - biology must be convincing

Score 4 = Strong (Best-in-Class):
- Known VALIDATED mechanism (approved drugs exist using similar mechanism)
- Clearly SUPERIOR implementation with real mechanistic differentiation
- Examples: dual/triple/quad agonism vs single target, better selectivity, better pathway modulation
- Overcomes known mechanistic limitations of the class
- This is the typical BIC score

Score 3 = Moderate (Me-too / Fast Follower):
- Known validated mechanism
- LIMITED mechanism-level differentiation
- Competes mainly on non-MoA factors (dosing, formulation, convenience)
- Standard fast follower in established class

Score 2 = Weak (Outdated):
- Mechanism is OLDER or strategically OUTDATED
- Class has been REPLACED by better mechanisms
- Market and guidelines have shifted toward better alternatives
- Even if mechanism still works, it is no longer a strong source of innovation

Score 1 = Poor (Invalid):
- WEAK or SPECULATIVE biology (loose association with disease, mostly theoretical)
- OR repeated clinical FAILURES for same mechanism in same indication
- OR mechanism has been clinically INVALIDATED
- No convincing rescue hypothesis

CLASSIFICATION DECISION TREE:
1. Is the mechanism NEW to this indication (no prior approved class)?
   → If YES + strong biology = Score 5 (FIC)
   → If YES + weak biology = Score 1 (Poor - speculative)

2. Is the mechanism VALIDATED but with CLEAR SUPERIOR implementation?
   → If YES (dual/triple/quad agonism, better selectivity, etc.) = Score 4 (BIC)

3. Is the mechanism VALIDATED but with LIMITED differentiation?
   → If YES = Score 3 (Me-too)

4. Is the mechanism OLDER/OUTDATED (replaced by better mechanisms)?
   → If YES = Score 2 (Weak/Outdated)

5. Is the biology WEAK/SPECULATIVE or has mechanism been INVALIDATED?
   → If YES = Score 1 (Poor)

IMPORTANT FOR MULTI-AGONISTS (dual/triple/quad):
- A quad agonist that improves on triple/dual agonism IS mechanistic differentiation
- If each component pathway is individually validated, the combination has STRONG rationale
- Multi-agonism is EXACTLY what BIC (Score 4) describes: "dual agonism instead of single target"

GUARDRAIL CHECK:
Mark FAIL if: The same mechanism has already been clinically INVALIDATED in the same indication AND there is no compelling scientific reason this drug should succeed.
Note: One failed molecule does NOT invalidate a mechanism - must be mechanism-level failure.

CONFIDENCE TIERS:
- Tier 1: Conclusion based on strong peer-reviewed or authoritative evidence
- Tier 2: Conclusion depends partly on secondary materials or moderate inference
- Tier 3: Evidence is limited and conclusion is inference-heavy

CONFIDENCE SCORE (0.0 to 1.0):
- 0.9-1.0: Very high - strong peer-reviewed evidence
- 0.7-0.89: High - good evidence with minor gaps
- 0.5-0.69: Moderate - mixed evidence, some inference required
- 0.3-0.49: Low - limited evidence, significant inference
- 0.0-0.29: Very low - mostly speculation

MANDATORY: You MUST explain why this score and not a higher one.

Return JSON:
{{
    "final_score": 1-5,
    "final_classification": "First-in-Class|Best-in-Class|Me-too|Weak/Outdated|Poor/Invalid",
    "guardrail_status": "PASS|FAIL",
    "guardrail_reason": "If FAIL, explain why. If PASS, state 'No mechanism invalidation detected'",
    "confidence_score": 0.0-1.0,
    "confidence_tier": "Tier 1|Tier 2|Tier 3",
    "confidence_reason": "Tier 1=strong peer-reviewed evidence, Tier 2=secondary sources/inference, Tier 3=limited evidence",
    "justification": {{
        "mechanism_summary": "DETAILED paragraph (3-5 sentences): Explain what the molecule targets, how the mechanism works at the receptor/pathway level, and the physiological effects",
        "novelty_vs_soc": "DETAILED paragraph (3-5 sentences): Explain whether this mechanism is novel to the indication, how it compares to current standard of care treatments",
        "competitor_comparison": "DETAILED paragraph (4-6 sentences): Compare this mechanism to other approved drugs and pipeline candidates. Discuss dual/triple/quad agonists and mechanistic differentiation",
        "biological_rationale": "DETAILED paragraph (3-5 sentences): Explain why targeting this pathway makes biological sense. For multi-agonists, explain each component's rationale",
        "prior_validation_failure": "DETAILED paragraph (3-5 sentences): Discuss whether this mechanism has been clinically validated. If failures exist, were they mechanism-level or molecule-specific?",
        "why_not_higher_score": "DETAILED paragraph (2-4 sentences): MANDATORY - Explain specifically why a higher score was not given. Reference the scoring criteria above"
    }},
    "sources_used": {{
        "primary": ["peer-reviewed journals, seminal papers"],
        "secondary": ["company disclosures, investor presentations"],
        "tertiary": ["patent databases"]
    }}
}}"""

    result = await _gemini_search_call(prompt)

    if result and not result.get("raw_response"):
        score = result.get('final_score', 'N/A')
        classification = result.get('final_classification', 'Unknown')
        guardrail = result.get('guardrail_status', 'Unknown')
        confidence = result.get('confidence_score', 'N/A')
        print(f"  Final Score: {score}, Classification: {classification}, Guardrail: {guardrail}, Confidence: {confidence}")

    return result


def _format_output(drug_name: str, moa: str, indication: str, final_result: dict, all_data: dict) -> dict:
    """Format the final output according to the required schema."""

    justification = final_result.get("justification", {})
    sources = final_result.get("sources_used", {})

    # Build the formatted output
    output = {
        "dimension": "MoA Innovation",
        "drug_name": drug_name,
        "mechanism_statement": moa,
        "indication": indication,
        "moa_classification": final_result.get("final_classification", "Unknown"),
        "score": final_result.get("final_score", "N/A"),
        "guardrail": final_result.get("guardrail_status", "Unknown"),
        "confidence_score": final_result.get("confidence_score", 0.0),
        "confidence_tier": final_result.get("confidence_tier", "Tier 3"),
        "justification": {
            "mechanism_summary": justification.get("mechanism_summary", ""),
            "novelty_vs_soc": justification.get("novelty_vs_soc", ""),
            "competitor_comparison": justification.get("competitor_comparison", ""),
            "biological_rationale": justification.get("biological_rationale", ""),
            "prior_validation_failure": justification.get("prior_validation_failure", ""),
            "why_not_higher_score": justification.get("why_not_higher_score", "")
        },
        "sources_used": {
            "primary": sources.get("primary", []),
            "secondary": sources.get("secondary", []),
            "tertiary": sources.get("tertiary", [])
        },
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "raw_data": all_data  # Include all intermediate data for transparency
    }

    return output


def _generate_narrative_rationale(output: dict) -> str:
    """Generate a flowing narrative rationale in paragraph format.

    Preserves ALL original justification content from the LLM, just formats
    it as flowing paragraphs instead of bullet points with headers.
    """
    j = output.get("justification", {})

    drug_name = output.get("drug_name", "The molecule")
    classification = output.get("moa_classification", "Unknown")
    score = output.get("score", "N/A")
    guardrail = output.get("guardrail", "Unknown")
    conf_score = output.get("confidence_score", 0.0)
    conf_tier = output.get("confidence_tier", "Tier 3")

    # Extract ALL justification components - preserve full content
    mechanism_summary = j.get("mechanism_summary", "").strip()
    novelty_vs_soc = j.get("novelty_vs_soc", "").strip()
    competitor_comparison = j.get("competitor_comparison", "").strip()
    biological_rationale = j.get("biological_rationale", "").strip()
    prior_validation = j.get("prior_validation_failure", "").strip()
    why_not_higher = j.get("why_not_higher_score", "").strip()

    # Build confidence description for intro
    if isinstance(conf_score, (int, float)):
        conf_pct = f"{conf_score * 100:.0f}%"
        if conf_score >= 0.9:
            conf_desc = "very high confidence supported by strong peer-reviewed evidence"
        elif conf_score >= 0.7:
            conf_desc = "high confidence with good evidence and minor gaps"
        elif conf_score >= 0.5:
            conf_desc = "moderate confidence with some inference required"
        elif conf_score >= 0.3:
            conf_desc = "limited confidence due to evidence gaps"
        else:
            conf_desc = "low confidence based on limited available evidence"
    else:
        conf_pct = "N/A"
        conf_desc = "assessed confidence"

    # Build classification description for intro
    class_desc = {
        "First-in-Class": "representing a pioneering therapeutic approach with a novel mechanism",
        "Best-in-Class": "demonstrating superior mechanistic differentiation within an established class",
        "Me-too": "operating within a validated mechanism class with limited differentiation",
        "Weak/Outdated": "utilizing a mechanism that has been superseded by newer approaches",
        "Poor/Invalid": "based on a mechanism with weak biological rationale or prior clinical invalidation"
    }.get(classification, f"classified as {classification}")

    guardrail_desc = "passing all safety guardrails" if guardrail == "PASS" else "flagging potential concerns requiring further evaluation"

    paragraphs = []

    # Paragraph 1: Brief intro + FULL mechanism summary (no truncation)
    intro = (
        f"{drug_name} achieves a MoA Innovation score of {score} out of 5, {class_desc}. "
        f"This assessment reflects {conf_desc} ({conf_tier}, {conf_pct} confidence), "
        f"with the analysis {guardrail_desc}."
    )
    if mechanism_summary:
        # Add mechanism summary as its own paragraph after intro
        paragraphs.append(intro)
        paragraphs.append(mechanism_summary)
    else:
        paragraphs.append(intro)

    # Paragraph: Novelty vs SOC - use FULL original content
    if novelty_vs_soc:
        paragraphs.append(novelty_vs_soc)

    # Paragraph: Competitor Comparison - use FULL original content
    if competitor_comparison:
        paragraphs.append(competitor_comparison)

    # Paragraph: Biological Rationale - use FULL original content
    if biological_rationale:
        paragraphs.append(biological_rationale)

    # Paragraph: Prior Validation/Failure - use FULL original content
    if prior_validation:
        paragraphs.append(prior_validation)

    # Paragraph: Why Not Higher Score - use FULL original content
    if why_not_higher:
        paragraphs.append(why_not_higher)

    # Join paragraphs with double newlines for separation
    return "\n\n".join(paragraphs)


def _generate_markdown_table(output: dict) -> str:
    """Generate a horizontal markdown table from the output with narrative rationale."""

    j = output.get("justification", {})
    s = output.get("sources_used", {})

    conf_score = output.get('confidence_score', 0.0)
    conf_pct = f"{conf_score * 100:.0f}%" if isinstance(conf_score, (int, float)) else "N/A"

    # Sources as strings
    primary = ', '.join(s.get('primary', [])) if s.get('primary') else 'N/A'
    secondary = ', '.join(s.get('secondary', [])) if s.get('secondary') else 'N/A'
    tertiary = ', '.join(s.get('tertiary', [])) if s.get('tertiary') else 'N/A'

    def _v(val):
        """Return value or N/A if empty/None."""
        sv = str(val).strip() if val is not None else ""
        return sv if sv and sv != "None" else "N/A"

    # Generate flowing narrative rationale instead of bullet points
    narrative_rationale = _generate_narrative_rationale(output)

    # Convert newlines to <br><br> for markdown table cell
    rationale_for_table = narrative_rationale.replace("\n\n", "<br><br>")

    # Build pre-formatted horizontal markdown table
    table_lines = [
        f"## MoA Innovation Assessment: {_v(output.get('drug_name'))}\n",
        "| Drug Name | Indication | Mechanism of Action | MoA Classification | Score | Guardrail | Confidence Score | Confidence Tier | Rationale | Primary Sources | Secondary Sources | Tertiary Sources |",
        "|-----------|------------|---------------------|-------------------|-------|-----------|------------------|-----------------|-----------|-----------------|-------------------|------------------|",
    ]

    # Add the data row
    table_lines.append(
        f"| {_v(output.get('drug_name'))} "
        f"| {_v(output.get('indication'))} "
        f"| {_v(output.get('mechanism_statement'))} "
        f"| {_v(output.get('moa_classification'))} "
        f"| {_v(output.get('score'))}/5 "
        f"| {_v(output.get('guardrail'))} "
        f"| {conf_score} ({conf_pct}) "
        f"| {_v(output.get('confidence_tier'))} "
        f"| {rationale_for_table} "
        f"| {primary} "
        f"| {secondary} "
        f"| {tertiary} |"
    )

    table_lines.append(f"\n*Analysis Date: {_v(output.get('analysis_date'))}*")

    # Also add the narrative as a separate section for better readability
    table_lines.append(f"\n\n### Detailed Rationale\n\n{narrative_rationale}")

    return "\n".join(table_lines)


async def get_moa_innovation_assessment(drug_name: str, indication: str = None) -> dict:
    """
    Assess the Mechanism of Action (MoA) Innovation for a drug molecule.

    This tool evaluates how innovative and strategically meaningful a molecule's
    mechanism of action is for the given indication.

    Classification types:
    - First-in-Class (FIC): First meaningful drug targeting a novel mechanism
    - Best-in-Class (BIC): Known mechanism used in a clearly better way
    - Me-too / Fast Follower: Already validated mechanism, limited innovation
    - Weak / Outdated: Older mechanism no longer strategically strong
    - Poor / Invalid: Weak rationale or clinically invalidated mechanism

    Scoring:
    - 5 = Exceptional (true FIC, strong rationale, class-creating potential)
    - 4 = Strong (validated class, clearly superior mechanism differentiation)
    - 3 = Moderate (validated class, limited innovation)
    - 2 = Weak (older or strategically outdated mechanism)
    - 1 = Poor (weak biology or clinically invalidated mechanism)

    Args:
        drug_name: Name of the drug molecule (e.g., "semaglutide", "tirzepatide")
        indication: Optional indication to assess. If not provided, uses indication from database.

    Returns:
        dict: Complete MoA Innovation assessment with score, classification, justification, and sources.
    """
    t0 = time.time()

    print("=" * 80)
    print(f"MOA INNOVATION ASSESSMENT")
    print(f"Drug: {drug_name}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    all_data = {}

    try:
        # Step 1: Get MoA from BigQuery
        step1_result = await _step1_get_moa_from_bq(drug_name)
        all_data["step1_moa_data"] = step1_result

        moa = step1_result.get("mechanism_of_action") or ""
        ind = indication or step1_result.get("indication") or ""

        # If MoA not found in BQ, search for it
        if not moa:
            print(f"[MoA] MoA not found in database, searching...")
            search_prompt = f"""Find the mechanism of action for drug: {drug_name}
Return JSON: {{"mechanism_of_action": "the mechanism", "indication": "primary indication"}}"""
            moa_search = await _gemini_search_call(search_prompt)
            moa = moa_search.get("mechanism_of_action", f"{drug_name} mechanism")
            ind = ind or moa_search.get("indication", "")
            all_data["step1_moa_data"]["mechanism_of_action"] = moa
            all_data["step1_moa_data"]["indication"] = ind
            all_data["step1_moa_data"]["source"] = "Google Search"

        if not ind:
            ind = "Type 2 Diabetes / Obesity"  # Default for GLP-1 drugs

        print(f"\n[MoA] Using: MoA='{moa}', Indication='{ind}'\n")

        # ══════════════════════════════════════════════════════════════════
        # SEQUENTIAL FLOW (per documentation)
        # Step 2 → Step 3 → Step 4 → Step 5 → Step 6 → Step 7 → Step 8 → Step 9
        # ══════════════════════════════════════════════════════════════════

        # Step 2: Build mechanism landscape (needed for Step 3)
        step2_result = await _step2_build_mechanism_landscape(drug_name, moa, ind)
        all_data["step2_landscape"] = step2_result

        # Step 3: Classify MoA position - "Only after building the mechanism landscape"
        step3_result = await _step3_classify_moa_position(drug_name, moa, ind, step2_result)
        all_data["step3_classification"] = step3_result

        # ══════════════════════════════════════════════════════════════════
        # PARALLEL BATCH: Steps 4, 5, 6 (independent research after classification)
        # These check rationale, validation, and failures - can run in parallel
        # ══════════════════════════════════════════════════════════════════
        print("[MoA] Running Steps 4, 5, 6 in parallel...")
        parallel_results = await asyncio.gather(
            _step4_check_biological_rationale(drug_name, moa, ind),     # Step 4
            _step5_check_clinical_validation(drug_name, moa, ind),      # Step 5
            _step6_check_mechanism_failures(drug_name, moa, ind),       # Step 6
        )
        step4_result, step5_result, step6_result = parallel_results
        all_data["step4_rationale"] = step4_result
        all_data["step5_validation"] = step5_result
        all_data["step6_failures"] = step6_result

        # Step 7: Check mechanistic improvement - needs Step 3 classification
        step7_result = await _step7_check_mechanistic_improvement(drug_name, moa, ind, step3_result)
        all_data["step7_improvement"] = step7_result

        # Step 8: Check mechanism currency
        step8_result = await _step8_check_mechanism_currency(drug_name, moa, ind)
        all_data["step8_currency"] = step8_result

        # ══════════════════════════════════════════════════════════════════
        # FINAL: Step 9 (needs all previous steps)
        # ══════════════════════════════════════════════════════════════════
        step9_result = await _step9_final_scoring(drug_name, moa, ind, all_data)
        all_data["step9_final"] = step9_result

        # Format output
        output = _format_output(drug_name, moa, ind, step9_result, all_data)

        # Generate narrative rationale (flowing paragraphs)
        narrative_rationale = _generate_narrative_rationale(output)
        output["narrative_rationale"] = narrative_rationale

        # Generate markdown table (includes narrative)
        markdown_table = _generate_markdown_table(output)
        output["markdown_table"] = markdown_table

        elapsed = time.time() - t0
        output["processing_time_seconds"] = round(elapsed, 1)

        print("\n" + "=" * 80)
        print("ASSESSMENT COMPLETE")
        print("=" * 80)
        print(f"Score: {output.get('score', 'N/A')}/5")
        print(f"Classification: {output.get('moa_classification', 'Unknown')}")
        print(f"Guardrail: {output.get('guardrail', 'Unknown')}")
        conf_score = output.get('confidence_score', 0.0)
        print(f"Confidence Score: {conf_score} ({conf_score*100:.0f}%)" if isinstance(conf_score, (int, float)) else f"Confidence Score: {conf_score}")
        print(f"Confidence Tier: {output.get('confidence_tier', 'Unknown')}")
        print(f"Time: {elapsed:.1f}s")
        print("=" * 80)

        return output

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n[MoA] ERROR after {elapsed:.1f}s: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

        return {
            "dimension": "MoA Innovation",
            "drug_name": drug_name,
            "indication": indication,
            "error": str(e),
            "processing_time_seconds": round(elapsed, 1)
        }


# CLI for standalone testing
if __name__ == "__main__":
    import sys

    drug = sys.argv[1] if len(sys.argv) > 1 else "semaglutide"
    indication = sys.argv[2] if len(sys.argv) > 2 else None

    result = asyncio.run(get_moa_innovation_assessment(drug, indication))

    # Print markdown table
    if "markdown_table" in result:
        print("\n" + result["markdown_table"])

    # Save JSON
    output_file = f"moa_innovation_{drug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nJSON saved to: {output_file}")
