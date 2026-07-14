#!/usr/bin/env python3
"""
Orchestrator for all Market Potential dimensions.
Replaces agent.py (ADK) — runs dimensions directly without the ADK framework.

Dimensions:
    1  - MoA Innovation
    3  - Clinical Efficacy
    6  - Tolerability
    P  - Patent Litigations

Usage:
    # Run ALL dimensions for ALL GLP-1 molecules
    python market_potential/run_all.py

    # Run all dimensions for a single molecule
    python market_potential/run_all.py --molecule Semaglutide

    # Run all dimensions for multiple specific molecules
    python market_potential/run_all.py --molecules "Semaglutide,Tirzepatide,Liraglutide"

    # Run specific dimensions only (comma-separated: 1, 3, 6, P)
    python market_potential/run_all.py --molecule Semaglutide --dimensions 1,3,6

    # Run with concurrency (parallel molecules, sequential dimensions per molecule)
    python market_potential/run_all.py --concurrency 3

    # Cloud Run parallel execution
    python market_potential/run_all.py --task-index 0 --task-count 5

    # Skip patent evaluation for faster runs
    python market_potential/run_all.py --molecule Semaglutide --dimensions P --no-patent-eval

    # Custom drug class / indication for tolerability
    python market_potential/run_all.py --molecule Semaglutide --dimensions 6 --drug-class GLP-1 --indication obesity
"""

import os
import sys
import json
import asyncio
import argparse
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from google.cloud import bigquery
from google.oauth2 import service_account

from market_potential.tools import (
    get_dimension_iii_efficacy_data,
    get_dimension_i_moa_innovation,
    get_dimension_vi_tolerability,
    get_patent_litigations,
)

# ── Configuration ────────────────────────────────────────────────────────────

PROJECT_ID = os.getenv("PROJECT_ID")
DATASET_ID = os.getenv("BQ_DATASET_ID")

VALID_DIMENSIONS = {"1", "3", "6", "P"}

# Same deterministic query used by run_clinical_efficacy.py
GLP1_PRODUCT_LIST_QUERY = f"""
SELECT DISTINCT cleaned_generic_name
FROM `{PROJECT_ID}.{DATASET_ID}.vw_drug_details_full`
WHERE
(
    UPPER(cleaned_Target) LIKE '%GLUCAGON LIKE PEPTIDE 1%'
    OR UPPER(cleaned_Target) LIKE '%GLP-1%'
    OR UPPER(cleaned_Target) LIKE '%GLUCAGON LIKE PEPTIDE-1%'
    OR (
        data_source = 'IPD'
        AND Mechanism_of_Action = 'Glucagon-like peptide-1 (GLP-1) agonist'
    )
)
AND Mechanism_of_Action IS NOT NULL
AND LOWER(Mechanism_of_Action) NOT LIKE '%antagonist%'
ORDER BY cleaned_generic_name ASC
"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_bq_client() -> bigquery.Client:
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path and os.path.exists(credentials_path):
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        return bigquery.Client(project=PROJECT_ID, credentials=credentials)
    return bigquery.Client(project=PROJECT_ID)


def get_glp1_molecules() -> list[str]:
    """Fetch all GLP-1 molecules from BigQuery (deterministic order)."""
    client = _get_bq_client()
    print("Fetching GLP-1 molecules from BigQuery...")
    results = client.query(GLP1_PRODUCT_LIST_QUERY).result()
    molecules = [row.cleaned_generic_name for row in results if row.cleaned_generic_name]
    print(f"Found {len(molecules)} GLP-1 molecules")
    return molecules


def _parse_molecules_list(raw: str) -> list[str]:
    """Parse a comma-separated molecules string into a clean, de-duplicated list.

    - Trims whitespace around each entry
    - Drops empty entries (e.g. trailing commas)
    - Preserves original order, removes exact duplicates
    """
    seen = set()
    molecules = []
    for m in raw.split(","):
        name = m.strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        molecules.append(name)
    return molecules


# ── Per-dimension runners ────────────────────────────────────────────────────

async def _run_dimension_1(molecule: str, indication: str = None) -> dict:
    """Dimension 1 — MoA Innovation."""
    print(f"\n{'─'*60}")
    print(f"[Dim 1] MoA Innovation → {molecule}")
    print(f"{'─'*60}")
    t0 = time.time()
    try:
        result = await get_dimension_i_moa_innovation(molecule, indication)
        elapsed = time.time() - t0
        score = result.get("score", "N/A")
        classification = result.get("moa_classification", "N/A")
        print(f"[Dim 1] {molecule} → Score: {score}/5, Class: {classification} ({elapsed:.1f}s)")
        if result.get("markdown_table"):
            print(result["markdown_table"])
        return {"dimension": "1_moa_innovation", "status": "success", "score": score, "time": elapsed, "result": result}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[Dim 1] {molecule} → ERROR: {e} ({elapsed:.1f}s)")
        return {"dimension": "1_moa_innovation", "status": "error", "error": str(e), "time": elapsed}


async def _run_dimension_3(molecule: str) -> dict:
    """Dimension 3 — Clinical Efficacy (also saves to BQ)."""
    print(f"\n{'─'*60}")
    print(f"[Dim 3] Clinical Efficacy → {molecule}")
    print(f"{'─'*60}")
    t0 = time.time()
    try:
        result = await get_dimension_iii_efficacy_data(molecule)
        elapsed = time.time() - t0
        n_trials = result.get("total_trials", 0)
        score = result.get("scoring_result", {}).get("weighted_score", "N/A")
        print(f"[Dim 3] {molecule} → Score: {score}/5, Trials: {n_trials} ({elapsed:.1f}s)")
        if result.get("markdown_table"):
            print(result["markdown_table"])
        return {"dimension": "3_clinical_efficacy", "status": "success", "score": score, "trials": n_trials, "time": elapsed, "result": result}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[Dim 3] {molecule} → ERROR: {e} ({elapsed:.1f}s)")
        return {"dimension": "3_clinical_efficacy", "status": "error", "error": str(e), "time": elapsed}


async def _run_dimension_6(molecule: str, drug_class: str = "GLP-1", indication: str = None) -> dict:
    """Dimension 6 — Tolerability."""
    print(f"\n{'─'*60}")
    print(f"[Dim 6] Tolerability → {molecule}")
    print(f"{'─'*60}")
    t0 = time.time()
    try:
        result = await get_dimension_vi_tolerability(molecule, drug_class, indication)
        elapsed = time.time() - t0
        score = result.get("tolerability_score", "N/A")
        guardrail = result.get("guardrail", "N/A")
        print(f"[Dim 6] {molecule} → Score: {score}, Guardrail: {guardrail} ({elapsed:.1f}s)")
        if result.get("markdown_table"):
            print(result["markdown_table"])
        return {"dimension": "6_tolerability", "status": "success", "score": score, "guardrail": guardrail, "time": elapsed, "result": result}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[Dim 6] {molecule} → ERROR: {e} ({elapsed:.1f}s)")
        return {"dimension": "6_tolerability", "status": "error", "error": str(e), "time": elapsed}


async def _run_patent_litigations(molecule: str, include_evaluation: bool = True) -> dict:
    """Patent Litigations."""
    print(f"\n{'─'*60}")
    print(f"[Patent] Litigations → {molecule}")
    print(f"{'─'*60}")
    t0 = time.time()
    try:
        result = await get_patent_litigations(molecule, include_evaluation)
        elapsed = time.time() - t0
        total = result.get("statistics", {}).get("total_cases", 0)
        print(f"[Patent] {molecule} → {total} cases found ({elapsed:.1f}s)")
        if result.get("markdown_tables"):
            for case_type, table in result["markdown_tables"].items():
                print(table)
        return {"dimension": "P_patent_litigations", "status": "success", "total_cases": total, "time": elapsed, "result": result}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[Patent] {molecule} → ERROR: {e} ({elapsed:.1f}s)")
        return {"dimension": "P_patent_litigations", "status": "error", "error": str(e), "time": elapsed}


# ── Molecule-level orchestrator ──────────────────────────────────────────────

async def process_molecule(molecule: str,
                           dimensions: set[str],
                           semaphore: asyncio.Semaphore,
                           drug_class: str = "GLP-1",
                           indication: str = None,
                           include_patent_eval: bool = True) -> dict:
    """Run requested dimensions for a single molecule, bounded by semaphore."""
    async with semaphore:
        mol_t0 = time.time()
        print(f"\n{'='*70}")
        print(f"  MOLECULE: {molecule}")
        print(f"  Dimensions: {', '.join(sorted(dimensions))}")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")

        dim_results = {}

        # Run dimensions sequentially per molecule to avoid Gemini rate-limit issues
        if "1" in dimensions:
            dim_results["1"] = await _run_dimension_1(molecule, indication)

        if "3" in dimensions:
            dim_results["3"] = await _run_dimension_3(molecule)

        if "6" in dimensions:
            dim_results["6"] = await _run_dimension_6(molecule, drug_class, indication)

        if "P" in dimensions:
            dim_results["P"] = await _run_patent_litigations(molecule, include_patent_eval)

        mol_elapsed = time.time() - mol_t0

        successes = sum(1 for d in dim_results.values() if d.get("status") == "success")
        failures = sum(1 for d in dim_results.values() if d.get("status") == "error")

        print(f"\n[{molecule}] All dimensions complete in {mol_elapsed:.1f}s — {successes} OK, {failures} failed")

        return {
            "molecule": molecule,
            "dimensions": dim_results,
            "status": "success" if failures == 0 else ("partial" if successes > 0 else "error"),
            "time": mol_elapsed,
        }


# ── Batch runner ─────────────────────────────────────────────────────────────

async def run_all(molecules: list[str],
                  dimensions: set[str],
                  max_concurrency: int,
                  drug_class: str = "GLP-1",
                  indication: str = None,
                  include_patent_eval: bool = True) -> list[dict]:
    """Process all molecules with bounded concurrency."""
    semaphore = asyncio.Semaphore(max_concurrency)
    print(f"Execution pool: max_concurrency={max_concurrency}, molecules={len(molecules)}, dimensions={sorted(dimensions)}")

    tasks = [
        process_molecule(m, dimensions, semaphore, drug_class, indication, include_patent_eval)
        for m in molecules
    ]
    results = await asyncio.gather(*tasks)
    return list(results)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run Market Potential dimensions for GLP-1 molecules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dimension codes:
  1  MoA Innovation
  3  Clinical Efficacy (saves to BQ)
  6  Tolerability
  P  Patent Litigations

Examples:
  python run_all.py --molecule Semaglutide
  python run_all.py --molecules "Semaglutide,Tirzepatide,Liraglutide" --dimensions 1,6
  python run_all.py --molecule Semaglutide --dimensions 1,3
  python run_all.py --concurrency 3
  python run_all.py --task-index 0 --task-count 5
""",
    )
    parser.add_argument("--molecule", help="Process a specific molecule only")
    parser.add_argument("--molecules", help="Comma-separated list of specific molecules to process (e.g. \"Semaglutide,Tirzepatide,Liraglutide\")")
    parser.add_argument("--dimensions", default="1,3,6,P",
                        help="Comma-separated dimension codes to run (default: 1,3,6,P)")
    parser.add_argument("--drug-class", default="GLP-1",
                        help="Drug class for tolerability SoC benchmark (default: GLP-1)")
    parser.add_argument("--indication", default=None,
                        help="Indication override for MoA and tolerability")
    parser.add_argument("--no-patent-eval", action="store_true",
                        help="Skip Claude evaluation of patent litigations")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Max concurrent molecules (default: 1)")
    parser.add_argument("--task-index", type=int, help="Cloud Run task index")
    parser.add_argument("--task-count", type=int, help="Cloud Run task count")
    parser.add_argument("--output-json", default=None,
                        help="Save full results to JSON file")
    args = parser.parse_args()

    # Parse dimensions
    dims = {d.strip().upper() for d in args.dimensions.split(",")}
    invalid = dims - VALID_DIMENSIONS
    if invalid:
        print(f"ERROR: Invalid dimension codes: {invalid}. Valid: {VALID_DIMENSIONS}")
        sys.exit(1)

    # --molecule and --molecules are mutually exclusive
    if args.molecule and args.molecules:
        print("ERROR: Use either --molecule or --molecules, not both.")
        sys.exit(1)

    # Cloud Run env fallbacks
    task_index = args.task_index if args.task_index is not None else int(os.getenv("CLOUD_RUN_TASK_INDEX", -1))
    task_count = args.task_count if args.task_count is not None else int(os.getenv("CLOUD_RUN_TASK_COUNT", -1))

    print("=" * 70)
    print("MARKET POTENTIAL — ALL DIMENSIONS PIPELINE")
    print(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Dimensions: {sorted(dims)}")
    print("=" * 70)

    # Validate parallel config
    if (task_index >= 0 and task_count <= 0) or (task_index < 0 and task_count > 0):
        print(f"ERROR: Invalid parallel config — task-index={task_index}, task-count={task_count}")
        sys.exit(1)

    # Determine molecules
    if args.molecules:
        molecules = _parse_molecules_list(args.molecules)
        if not molecules:
            print("ERROR: --molecules was provided but no valid molecule names were parsed.")
            sys.exit(1)
        print(f"Mode: Custom molecule list ({len(molecules)} molecules)")
    elif args.molecule:
        molecules = [args.molecule]
        print(f"Mode: Single molecule ({args.molecule})")
    elif task_index >= 0 and task_count > 0:
        all_molecules = get_glp1_molecules()
        if task_index >= task_count:
            print(f"ERROR: task-index ({task_index}) >= task-count ({task_count})")
            sys.exit(1)

        total = len(all_molecules)
        per_task = total // task_count
        remainder = total % task_count
        start = (task_index * per_task) + min(task_index, remainder)
        end = start + per_task + (1 if task_index < remainder else 0)
        molecules = all_molecules[start:end]

        print(f"Mode: Cloud Run parallel (task {task_index + 1}/{task_count})")
        print(f"Processing indices {start}..{end - 1} of {total}")
    else:
        molecules = get_glp1_molecules()
        print(f"Mode: All molecules ({len(molecules)} total)")

    print(f"Molecules to process: {len(molecules)}")
    print("-" * 70)

    if not molecules:
        print("No molecules to process.")
        return

    t0 = time.time()

    results = asyncio.run(
        run_all(
            molecules=molecules,
            dimensions=dims,
            max_concurrency=args.concurrency,
            drug_class=args.drug_class,
            indication=args.indication,
            include_patent_eval=not args.no_patent_eval,
        )
    )

    total_time = time.time() - t0

    # ── Summary ──────────────────────────────────────────────────────────
    successful = [r for r in results if r["status"] == "success"]
    partial = [r for r in results if r["status"] == "partial"]
    failed = [r for r in results if r["status"] == "error"]

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total molecules : {len(results)}")
    print(f"Fully succeeded : {len(successful)}")
    print(f"Partial success : {len(partial)}")
    print(f"Fully failed    : {len(failed)}")
    print(f"Total time      : {total_time:.1f}s ({total_time / 60:.1f} min)")

    # Per-dimension summary
    for dim_code in sorted(dims):
        dim_ok = sum(
            1 for r in results
            if r["dimensions"].get(dim_code, {}).get("status") == "success"
        )
        dim_err = sum(
            1 for r in results
            if r["dimensions"].get(dim_code, {}).get("status") == "error"
        )
        label = {"1": "MoA Innovation", "3": "Clinical Efficacy", "6": "Tolerability", "P": "Patent Litigations"}[dim_code]
        print(f"  Dim {dim_code} ({label}): {dim_ok} OK / {dim_err} failed")

    if failed or partial:
        print("\nIssues:")
        for r in failed + partial:
            for dc, dr in r["dimensions"].items():
                if dr.get("status") == "error":
                    print(f"  [{r['molecule']}] Dim {dc}: {dr.get('error', 'Unknown')}")

    # Save JSON if requested
    if args.output_json:
        # Strip large nested result objects for the summary file
        summary = []
        for r in results:
            entry = {"molecule": r["molecule"], "status": r["status"], "time": r["time"], "dimensions": {}}
            for dc, dr in r["dimensions"].items():
                entry["dimensions"][dc] = {k: v for k, v in dr.items() if k != "result"}
            summary.append(entry)

        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSummary saved to: {args.output_json}")

    print(f"\nDone. ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')})")


if __name__ == "__main__":
    main()
