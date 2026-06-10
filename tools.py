"""Market Potential Agent Tools

This module provides:
- Clinical efficacy extraction tool (Dimension III) using the two-step Gemini approach
- MoA Innovation assessment tool (Dimension I) using multi-step Gemini with Google Search
- Patient Tolerability assessment tool (Dimension VI) using extended clinical trial extraction
"""
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

from google.cloud import bigquery
from google.oauth2 import service_account

from market_potential.bq_client import run_query
from market_potential.constants import GLP1_PRODUCT_LIST_QUERY
from market_potential.gemini_extractor import extract_clinical_trial_data
from market_potential.moa_innovation import get_moa_innovation_assessment
from market_potential.tolerability_tools import extract_tolerability_data, compute_tolerability_score
from market_potential.clinical_efficacy_scorer import compute_clinical_efficacy_score, generate_rationale

# BigQuery configuration
PROJECT_ID = os.getenv("PROJECT_ID")
DATASET_ID = os.getenv("BQ_DATASET_ID")
CLINICAL_EFFICACY_TABLE = "clinical_efficacy"


def _get_bq_client() -> bigquery.Client:
    """Get BigQuery client with credentials."""
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path and os.path.exists(credentials_path):
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        return bigquery.Client(project=PROJECT_ID, credentials=credentials)
    return bigquery.Client(project=PROJECT_ID)


def _ensure_clinical_efficacy_table():
    """Create clinical_efficacy table if it doesn't exist."""
    client = _get_bq_client()
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{CLINICAL_EFFICACY_TABLE}"

    schema = [
        bigquery.SchemaField("molecule_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("dosage", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("trial_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("phase", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("trial_size", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("trial_location", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("trial_start_date", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("trial_completion_date", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("phase_status", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("secondary_locations", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("hba1c_change_pct", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("hba1c_duration", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("weight_change_pct", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("weight_duration", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("alt_reduction_pct", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("alt_duration", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("mash_resolution_pct", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("mash_duration", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("company_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("source_url", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("efficacy_score", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("data_coverage", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("rationale", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("updated_at", "TIMESTAMP", mode="NULLABLE"),
    ]

    table = bigquery.Table(table_id, schema=schema)
    try:
        client.create_table(table)
        print(f"[BQ] Created table: {table_id}")
    except Exception as e:
        if "Already Exists" in str(e):
            pass  # Table exists, OK
        else:
            print(f"[BQ] Table check: {e}")


def _get_existing_trials(client, molecule_name: str) -> dict:
    """Fetch existing trials for a molecule from BigQuery.

    Returns:
        dict: {trial_id: {"phase_status": str, "phase": str}}
    """
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{CLINICAL_EFFICACY_TABLE}"
    query = f"""
    SELECT trial_id, phase_status, phase
    FROM `{table_id}`
    WHERE molecule_name = @molecule_name
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("molecule_name", "STRING", molecule_name)
        ]
    )
    try:
        results = client.query(query, job_config=job_config).result()
        return {
            row.trial_id: {
                "phase_status": row.phase_status,
                "phase": row.phase
            }
            for row in results if row.trial_id
        }
    except Exception as e:
        if "Not found" in str(e):
            return {}
        return {}


def save_clinical_efficacy_to_bq(molecule_name: str, trials: list, score_result: dict = None, rationale: str = None):
    """Save clinical efficacy data to BigQuery (incremental - only new trials, update changed phase_status)."""
    client = _get_bq_client()
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{CLINICAL_EFFICACY_TABLE}"
    now = datetime.now(timezone.utc)
    created_at = now.isoformat()

    # Ensure table exists
    _ensure_clinical_efficacy_table()

    if not trials:
        print(f"[BQ] No trials to save for {molecule_name}")
        return

    # Get existing trials for this molecule
    existing_trials = _get_existing_trials(client, molecule_name)
    if existing_trials:
        print(f"[BQ] Found {len(existing_trials)} existing trials for {molecule_name}")

    rows = []  # New trials to insert
    updates = []  # Trials needing phase_status update
    skipped = 0

    for trial in trials:
        trial_id = str(trial.get("trial_id") or "")
        if not trial_id:
            continue  # Skip trials without ID

        new_phase_status = str(trial.get("phase_status") or "")
        new_phase = str(trial.get("phase") or "")

        # Check if trial exists
        if trial_id in existing_trials:
            existing = existing_trials[trial_id]
            old_phase_status = existing.get("phase_status") or ""
            old_phase = existing.get("phase") or ""

            # Check if phase_status or phase changed
            phase_status_changed = new_phase_status != old_phase_status
            phase_changed = new_phase != old_phase

            if phase_status_changed or phase_changed:
                updates.append({
                    "trial_id": trial_id,
                    "phase_status": new_phase_status,
                    "phase": new_phase,
                    "updated_at": created_at,
                })
            else:
                skipped += 1
            continue

        # New trial - build row for insert
        row = {
            "molecule_name": str(trial.get("molecule_name") or molecule_name or ""),
            "dosage": str(trial.get("dosage") or ""),
            "trial_id": trial_id,
            "phase": new_phase,
            "trial_size": str(trial.get("trial_size") or ""),
            "trial_location": str(trial.get("trial_location") or ""),
            "trial_start_date": str(trial.get("trial_start_date") or ""),
            "trial_completion_date": str(trial.get("trial_completion_date") or ""),
            "phase_status": new_phase_status,
            "secondary_locations": str(trial.get("secondary_locations") or ""),
            "hba1c_change_pct": str(trial.get("hba1c_change_pct") or ""),
            "hba1c_duration": str(trial.get("hba1c_duration") or ""),
            "weight_change_pct": str(trial.get("weight_change_pct") or ""),
            "weight_duration": str(trial.get("weight_duration") or ""),
            "alt_reduction_pct": str(trial.get("alt_reduction_pct") or ""),
            "alt_duration": str(trial.get("alt_duration") or ""),
            "mash_resolution_pct": str(trial.get("mash_resolution_pct") or ""),
            "mash_duration": str(trial.get("mash_duration") or ""),
            "company_name": str(trial.get("company_name") or ""),
            "source_url": str(trial.get("source_url") or ""),
            "efficacy_score": float(score_result.get("weighted_score", 0)) if score_result else None,
            "data_coverage": score_result.get("data_coverage", "") if score_result else "",
            "rationale": rationale or "",
            "created_at": created_at,
        }
        rows.append(row)

    # Execute updates for trials with changed phase_status
    updated_count = 0
    if updates:
        print(f"[BQ] Updating {len(updates)} trials with changed phase/status...")
        for upd in updates:
            update_query = f"""
            UPDATE `{table_id}`
            SET phase_status = @phase_status,
                phase = @phase,
                updated_at = @updated_at
            WHERE molecule_name = @molecule_name
              AND trial_id = @trial_id
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("phase_status", "STRING", upd["phase_status"]),
                    bigquery.ScalarQueryParameter("phase", "STRING", upd["phase"]),
                    bigquery.ScalarQueryParameter("updated_at", "TIMESTAMP", upd["updated_at"]),
                    bigquery.ScalarQueryParameter("molecule_name", "STRING", molecule_name),
                    bigquery.ScalarQueryParameter("trial_id", "STRING", upd["trial_id"]),
                ]
            )
            try:
                client.query(update_query, job_config=job_config).result()
                updated_count += 1
            except Exception as e:
                print(f"[BQ] Update error for {upd['trial_id']}: {e}")

    if not rows and not updates:
        if skipped > 0:
            print(f"[BQ] No changes for {molecule_name} ({skipped} trials unchanged)")
        else:
            print(f"[BQ] No trials to save for {molecule_name}")
        return

    if not rows:
        # Only updates, no new inserts
        print(f"[BQ] Updated {updated_count} trials for {molecule_name} (no new trials)")
        return

    # Insert only new rows
    errors = client.insert_rows_json(table_id, rows)
    if errors:
        print(f"[BQ] Insert errors for {molecule_name}: {errors[:3]}")
    else:
        print(f"[BQ] Saved {len(rows)} NEW trials for {molecule_name} (skipped {skipped} unchanged, updated {updated_count})")


async def get_dimension_iii_efficacy_data(molecule_name: str,
                                          extra_fields_prompt: str = None,
                                          extra_fields_json: str = None) -> dict:
    """Extract clinical efficacy data using two-step Gemini approach with multi-source search.

    This function uses the optimized gemini_extractor module which:
    - Step 1: Searches multiple international registries for ALL trial IDs
    - Step 2: Extracts detailed efficacy data in parallel batches

    When called WITHOUT extra_fields parameters (standard efficacy call), this function
    also computes the clinical efficacy score and generates a rationale, including them
    in the markdown output.

    When called WITH extra_fields parameters (e.g., for tolerability extraction), only
    data extraction is performed without scoring/rationale.

    Args:
        molecule_name: Name of the molecule to search for (e.g., "Semaglutide")
        extra_fields_prompt: Optional additional prompt text for extracting extra fields
                            (e.g., tolerability fields for Dimension 6)
        extra_fields_json: Optional additional JSON schema fields to include in response

    Returns:
        dict: {
            "molecule": str,
            "total_trials": int,
            "trials": list of trial dicts,
            "markdown_table": str (includes score/rationale when extra_fields not provided),
            "scoring_result": dict (only when extra_fields not provided),
            "rationale": str (only when extra_fields not provided),
        }
    """
    print(f"\n[Market Potential Agent] Starting extraction for: {molecule_name}")

    # Determine if this is a standard efficacy call (no extra fields = include scoring)
    include_scoring = (extra_fields_prompt is None and extra_fields_json is None)

    # Call the gemini extractor with extra fields if provided
    result = await extract_clinical_trial_data(
        molecule_name,
        extra_fields_prompt=extra_fields_prompt,
        extra_fields_json=extra_fields_json
    )

    trials = result.get("trials", [])

    if not trials:
        print(f"\n[Market Potential Agent] No trials found for {molecule_name}")
        return {"efficacy_data": []}

    # Format trials for display - map field names to match agent expectations
    display_rows = []
    for trial in trials:
        display_rows.append({
            "molecule_name": trial.get("Molecule"),
            "dosage": trial.get("Dosage"),
            "trial_id": trial.get("Trial ID"),
            "phase": trial.get("Phase"),
            "trial_size": trial.get("Size"),
            "trial_location": trial.get("Primary Region"),
            "trial_start_date": trial.get("Start Date"),
            "trial_completion_date": trial.get("Completion Date"),
            "phase_status": trial.get("Status"),
            "secondary_locations": trial.get("Secondary Countries"),
            "mash_resolution_pct": trial.get("MASH Outcome (%)"),
            "mash_duration": trial.get("MASH Duration"),
            "hba1c_change_pct": trial.get("HbA1c Change (%)"),
            "hba1c_duration": trial.get("HbA1c Duration"),
            "weight_change_pct": trial.get("Weight Loss (%)"),
            "weight_duration": trial.get("Weight Duration"),
            "alt_reduction_pct": trial.get("ALT Reduction (%)"),
            "alt_duration": trial.get("ALT Duration"),
            "company_name": trial.get("Company"),
            "source_url": trial.get("Source URL"),
        })

    print(f"\n[Market Potential Agent] ✓ Extracted {len(display_rows)} trials")

    # Build pre-formatted markdown table
    table_lines = [
        f"## Clinical Efficacy Data: {molecule_name}\n",
        "| Molecule | Dosage | Trial ID | Phase | Size | Location | Start Date | Completion Date | Status | HbA1c Change (%) | HbA1c Duration | Weight Loss (%) | Weight Duration | ALT Reduction (%) | ALT Duration | MASH Outcome (%) | MASH Duration | Company | Source |",
        "|----------|--------|----------|-------|------|----------|------------|-----------------|--------|------------------|----------------|-----------------|-----------------|-------------------|--------------|------------------|---------------|---------|--------|",
    ]

    def _v(val):
        """Return value or N/A if empty/None."""
        s = str(val).strip() if val is not None else ""
        return s if s and s != "None" else "N/A"

    for row in display_rows:
        url = row.get("source_url") or ""
        trial_id = _v(row.get("trial_id"))
        source = f"[{trial_id}]({url})" if url and url != "N/A" else trial_id

        table_lines.append(
            f"| {_v(row.get('molecule_name'))} "
            f"| {_v(row.get('dosage'))} "
            f"| {trial_id} "
            f"| {_v(row.get('phase'))} "
            f"| {_v(row.get('trial_size'))} "
            f"| {_v(row.get('trial_location'))} "
            f"| {_v(row.get('trial_start_date'))} "
            f"| {_v(row.get('trial_completion_date'))} "
            f"| {_v(row.get('phase_status'))} "
            f"| {_v(row.get('hba1c_change_pct'))} "
            f"| {_v(row.get('hba1c_duration'))} "
            f"| {_v(row.get('weight_change_pct'))} "
            f"| {_v(row.get('weight_duration'))} "
            f"| {_v(row.get('alt_reduction_pct'))} "
            f"| {_v(row.get('alt_duration'))} "
            f"| {_v(row.get('mash_resolution_pct'))} "
            f"| {_v(row.get('mash_duration'))} "
            f"| {_v(row.get('company_name'))} "
            f"| {source} |"
        )

    table_lines.append(f"\n*Found {len(display_rows)} trials*")

    # Initialize result
    output = {
        "molecule": molecule_name,
        "total_trials": len(display_rows),
        "trials": display_rows,
    }

    # When no extra_fields provided: compute score and generate rationale
    if include_scoring:
        print(f"\n[Market Potential Agent] Computing clinical efficacy score...")

        # Prepare data for scorer (uses internal field names)
        scorer_data = {
            "molecule": molecule_name,
            "trials": display_rows,
            "total_trials": len(display_rows),
        }
        score_result = compute_clinical_efficacy_score(scorer_data)

        print(f"[Market Potential Agent] Score: {score_result['weighted_score']}/5")
        print(f"[Market Potential Agent] Coverage: {score_result['data_coverage']}")

        # Generate rationale
        print(f"\n[Market Potential Agent] Generating rationale...")
        rationale = generate_rationale(molecule_name, scorer_data, score_result)
        print(f"[Market Potential Agent] Rationale generated ({len(rationale)} chars)")

        # Add scoring section to markdown
        table_lines.append("\n---\n")
        table_lines.append(f"## Clinical Efficacy Score: {score_result['weighted_score']}/5\n")
        table_lines.append(f"**Coverage:** {score_result['data_coverage']}\n")
        table_lines.append("\n### Score Breakdown\n")
        table_lines.append("```")
        table_lines.append(score_result['score_breakdown'])
        table_lines.append("```\n")
        table_lines.append("\n### Clinical Rationale\n")
        table_lines.append(rationale)

        # Add to output
        output["scoring_result"] = score_result
        output["rationale"] = rationale

        # Save to BigQuery
        save_clinical_efficacy_to_bq(
            molecule_name=molecule_name,
            trials=display_rows,
            score_result=score_result,
            rationale=rationale,
        )

    markdown_table = "\n".join(table_lines)
    output["markdown_table"] = markdown_table

    return output


async def get_glp1_product_list() -> list[str]:
    """Get list of GLP-1 products from BigQuery."""
    rows = run_query(GLP1_PRODUCT_LIST_QUERY)
    return [row["cleaned_generic_name"] for row in rows if row.get("cleaned_generic_name")]


async def get_dimension_i_moa_innovation(drug_name: str, indication: str = None) -> dict:
    """Assess Mechanism of Action (MoA) Innovation (Dimension 1).

    Evaluates how innovative and strategically meaningful a molecule's
    mechanism of action is for the given indication.

    This tool uses multi-step Gemini with Google Search to:
    - Step 1: Get MoA from internal database (BigQuery)
    - Step 2: Build mechanism landscape (approved drugs, pipeline, SOC)
    - Step 3: Classify MoA position (FIC, BIC, Me-too, Outdated, Poor)
    - Step 4: Check biological rationale (peer-reviewed journals)
    - Step 5: Check clinical validation (FDA, ClinicalTrials.gov)
    - Step 6: Check for mechanism failures
    - Step 7: Check for mechanistic improvements (if not FIC)
    - Step 8: Check mechanism currency (still relevant or outdated)
    - Step 9: Final scoring and classification

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
        indication: Optional indication. If not provided, uses indication from database.

    Returns:
        dict: Complete MoA Innovation assessment with:
            - dimension: "MoA Innovation"
            - mechanism_statement: The MoA description
            - indication: The indication assessed
            - moa_classification: FIC/BIC/Me-too/Weak-Outdated/Poor-Invalid
            - score: 1-5
            - guardrail: PASS/FAIL
            - confidence_tier: Tier 1/2/3
            - justification: Detailed breakdown
            - sources_used: Primary/Secondary/Tertiary sources
            - markdown_report: Formatted report
    """
    return await get_moa_innovation_assessment(drug_name, indication)


async def get_dimension_vi_tolerability(molecule_name: str, drug_class: str = "GLP-1", indication: str = None) -> dict:
    """Assess Patient Tolerability & Burden (Dimension 6).

    Evaluates how patient-friendly a drug is by measuring:
    - Discontinuation rate due to adverse events (primary metric)
    - Side effect profile (common AEs, severity, persistence)
    - Patient burden (need for additional management)

    Process:
    1. Extract clinical trial data with tolerability fields
    2. Apply Clinical Trials Weightage Framework (Phase × Geography × Dosage × N)
    3. Calculate weighted average discontinuation rate
    4. Dynamically fetch Standard of Care benchmark via web search
    5. Apply dimension-specific scoring logic
    6. Check guardrails
    7. Generate justification

    Scoring Logic:
    - Base score from discontinuation rate vs placebo (1-5)
    - SoC adjustment: Better (+1), Similar (0), Worse (-1)
    - Burden adjustment: Mild/transient (0), Persistent/moderate (-1), Severe/managed (-2)

    Guardrail FAIL if: Discontinuation ≥ SoC AND side effects are persistent/require management

    Args:
        molecule_name: Name of the drug molecule (e.g., "semaglutide")
        drug_class: Drug class for SoC benchmark (default: "GLP-1")
        indication: Optional indication for more specific SoC lookup (e.g., "type 2 diabetes", "obesity")

    Returns:
        dict with structure per documentation:
            - tolerability_score: "X/5"
            - supporting_data: discontinuation rates, difference, SoC source
            - side_effect_summary: key AEs, severity, persistence, management
            - comparison: vs placebo, vs SoC
            - guardrail: "PASS" or "FAIL"
            - justification: comprehensive rationale
    """
    # Step 1: Extract trial data with tolerability fields
    result = await extract_tolerability_data(molecule_name)
    trials = result.get("trials", [])

    if not trials:
        return {
            "molecule": molecule_name,
            "tolerability_score": "N/A",
            "error": "No clinical trials found with tolerability data",
            "guardrail": "FAIL",
            "justification": f"Unable to assess tolerability for {molecule_name} - no clinical trial data with discontinuation rates found.",
        }

    # Step 2: Apply weightage framework and compute score (includes dynamic SoC lookup)
    score_result = await compute_tolerability_score(
        trials, molecule=molecule_name, drug_class=drug_class, indication=indication
    )

    # Add molecule name and trial count to result
    score_result["molecule"] = molecule_name
    score_result["_scoring_breakdown"]["total_trials_extracted"] = len(trials)

    return score_result


async def get_patent_litigations(drug_name: str, include_evaluation: bool = True) -> dict:
    """Find all patent litigations and court cases for a drug with Claude evaluation.

    Searches for patent challenges and litigation across multiple jurisdictions:
    - ANDA Paragraph IV patent challenges (US generic entry)
    - Inter Partes Review (IPR) at PTAB
    - European Patent Office (EPO) oppositions
    - India patent litigation (Delhi High Court)
    - Compounding pharmacy lawsuits

    Then evaluates each litigation using Claude to verify:
    - Patent number validity
    - Case existence and details
    - Case type classification accuracy
    - Status consistency
    - Challenger validity
    - Rationale legitimacy
    - Relevance to the drug

    Args:
        drug_name: Name of the drug molecule (e.g., "semaglutide", "tirzepatide")
        include_evaluation: Whether to run Claude evaluation (default: True)

    Returns:
        dict: {
            "drug_name": str,
            "brand_names": list of brand names,
            "innovator": str (patent holder),
            "litigations": list of litigation dicts with:
                - patent_number: str
                - case_name: str (Plaintiff v. Defendant)
                - case_number: str
                - court: str
                - case_type: str (ANDA, IPR, EPO, India, Compounding)
                - status: str (Pending, Settled, Decided, Invalidated, Upheld)
                - outcome: str
                - challenger: str
                - rationale: str (why challenged)
                - eval_* fields (if evaluation enabled):
                    - eval_patent_number_valid: bool
                    - eval_case_exists: bool
                    - eval_case_type_correct: bool
                    - eval_status_consistent: bool
                    - eval_challenger_valid: bool
                    - eval_rationale_valid: bool
                    - eval_is_relevant: bool
                    - eval_confidence_score: float (0.0-1.0)
                    - eval_overall_assessment: str
            "litigations_by_type": dict grouped by case type,
            "markdown_tables": dict of markdown tables by case type,
            "summary": str,
            "analysis_date": str,
            "search_time_seconds": float,
            "statistics": {
                "total_cases": int,
                "by_type": dict,
                "unique_challengers": int
            },
            "evaluation_metrics": dict (if evaluation enabled)
        }
    """
    from patent_discovery.discovery import list_all_litigations
    from patent_discovery.evaluation import evaluate_litigations

    print(f"\n[Market Potential Agent] Finding patent litigations for: {drug_name}")
    result = await list_all_litigations(drug_name)

    total_cases = result.get("statistics", {}).get("total_cases", 0)
    print(f"[Market Potential Agent] Found {total_cases} litigation cases")

    # Run evaluation if enabled and there are litigations
    if include_evaluation and result.get("litigations"):
        print(f"\n[Market Potential Agent] Running Claude evaluation on {total_cases} litigations...")
        eval_result = evaluate_litigations(
            litigations=result["litigations"],
            drug_name=drug_name,
            innovator=result.get("innovator")
        )

        # Replace litigations with evaluated versions
        result["litigations"] = eval_result["evaluated_litigations"]
        result["evaluation_metrics"] = eval_result["metrics"]
        result["evaluation_time_seconds"] = eval_result["evaluation_time_seconds"]

        # Update litigations_by_type with evaluated data
        cases_by_type = {}
        for lit in result["litigations"]:
            case_type = lit.get("case_type", "Other")
            if case_type not in cases_by_type:
                cases_by_type[case_type] = []
            cases_by_type[case_type].append(lit)
        result["litigations_by_type"] = cases_by_type

        # Rebuild markdown tables with all evaluation columns
        markdown_tables = {}
        for case_type, cases in cases_by_type.items():
            if not cases:
                continue

            def _v(val):
                """Return value or N/A if empty/None."""
                if val is None:
                    return "N/A"
                if isinstance(val, bool):
                    return "True" if val else "False"
                if isinstance(val, float):
                    return f"{val:.2f}"
                s = str(val).strip()
                return s if s and s != "None" else "N/A"

            table_lines = [
                f"\n### {case_type} ({len(cases)} cases)\n",
                "| patent_number | case_number | case_name | challenger | court | case_type | status | outcome | rationale | eval_patent_number_valid | eval_patent_number_note | eval_case_exists | eval_case_exists_note | eval_case_type_correct | eval_case_type_expected | eval_case_type_note | eval_status_consistent | eval_status_note | eval_challenger_valid | eval_challenger_note | eval_rationale_valid | eval_rationale_note | eval_is_relevant | eval_relevance_note | eval_confidence_score | eval_confidence_reason | eval_issues | eval_overall_assessment |",
                "|---------------|-------------|-----------|------------|-------|-----------|--------|---------|-----------|--------------------------|-------------------------|------------------|----------------------|------------------------|-------------------------|---------------------|------------------------|------------------|----------------------|----------------------|----------------------|---------------------|------------------|---------------------|----------------------|------------------------|-------------|-------------------------|",
            ]
            for c in cases:
                table_lines.append(
                    f"| {_v(c.get('patent_number'))} "
                    f"| {_v(c.get('case_number'))} "
                    f"| {_v(c.get('case_name'))} "
                    f"| {_v(c.get('challenger'))} "
                    f"| {_v(c.get('court'))} "
                    f"| {_v(c.get('case_type'))} "
                    f"| {_v(c.get('status'))} "
                    f"| {_v(c.get('outcome'))} "
                    f"| {_v(c.get('rationale'))} "
                    f"| {_v(c.get('eval_patent_number_valid'))} "
                    f"| {_v(c.get('eval_patent_number_note'))} "
                    f"| {_v(c.get('eval_case_exists'))} "
                    f"| {_v(c.get('eval_case_exists_note'))} "
                    f"| {_v(c.get('eval_case_type_correct'))} "
                    f"| {_v(c.get('eval_case_type_expected'))} "
                    f"| {_v(c.get('eval_case_type_note'))} "
                    f"| {_v(c.get('eval_status_consistent'))} "
                    f"| {_v(c.get('eval_status_note'))} "
                    f"| {_v(c.get('eval_challenger_valid'))} "
                    f"| {_v(c.get('eval_challenger_note'))} "
                    f"| {_v(c.get('eval_rationale_valid'))} "
                    f"| {_v(c.get('eval_rationale_note'))} "
                    f"| {_v(c.get('eval_is_relevant'))} "
                    f"| {_v(c.get('eval_relevance_note'))} "
                    f"| {_v(c.get('eval_confidence_score'))} "
                    f"| {_v(c.get('eval_confidence_reason'))} "
                    f"| {_v(c.get('eval_issues'))} "
                    f"| {_v(c.get('eval_overall_assessment'))} |"
                )
            markdown_tables[case_type] = "\n".join(table_lines)
        result["markdown_tables"] = markdown_tables

        # Update summary with evaluation info
        metrics = eval_result["metrics"]
        result["summary"] = (
            f"Found {total_cases} litigation cases across {len(cases_by_type)} categories. "
            f"Evaluation: {metrics.get('fully_verified_count', 0)} fully verified, "
            f"{metrics.get('high_confidence_count', 0)} high confidence (≥0.7), "
            f"avg confidence: {metrics.get('avg_confidence_score', 0):.2f}. "
            f"Key challengers: {', '.join(set(lit.get('challenger', '') for lit in result['litigations'] if lit.get('challenger')))[:150]}"
        )

        print(f"[Market Potential Agent] Evaluation complete - {metrics.get('fully_verified_count', 0)} fully verified")

    return result
