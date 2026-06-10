#!/usr/bin/env python3
"""
Run Clinical Efficacy extraction for all GLP-1 molecules.
Saves each molecule's results to BigQuery as it processes.

Usage:
    # Run all GLP-1 molecules sequentially
    python market_potential/run_clinical_efficacy.py

    # Run with 3 concurrent workers inside this container instance
    python market_potential/run_clinical_efficacy.py --concurrency 3

    # Run specific molecule
    python market_potential/run_clinical_efficacy.py --molecule Semaglutide

    # Run with Cloud Run parallel execution split safely
    python market_potential/run_clinical_efficacy.py --task-index 0 --task-count 5
"""

import os
import sys
import asyncio
import argparse
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from google.cloud import bigquery
from google.oauth2 import service_account

from market_potential.tools import get_dimension_iii_efficacy_data

# Configuration
PROJECT_ID = os.getenv("PROJECT_ID")
DATASET_ID = os.getenv("BQ_DATASET_ID")

# Enforced ORDER BY to ensure parallel Cloud Run containers share the exact same list index blueprint
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


def get_bq_client() -> bigquery.Client:
    """Get BigQuery client."""
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path and os.path.exists(credentials_path):
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        return bigquery.Client(project=PROJECT_ID, credentials=credentials)
    return bigquery.Client(project=PROJECT_ID)


def get_glp1_molecules() -> list[str]:
    """Fetch all GLP-1 molecules from BigQuery."""
    client = get_bq_client()
    print(f"Fetching GLP-1 molecules from BigQuery...")

    query_job = client.query(GLP1_PRODUCT_LIST_QUERY)
    results = query_job.result()

    molecules = [row.cleaned_generic_name for row in results if row.cleaned_generic_name]
    print(f"Found {len(molecules)} GLP-1 molecules")
    return molecules


async def process_molecule(molecule_name: str, semaphore: asyncio.Semaphore) -> dict:
    """Process a single molecule safely bounded by a concurrency semaphore."""
    async with semaphore:
        t0 = time.time()

        print(f"\n{'='*70}")
        print(f"Processing: {molecule_name}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")

        try:
            # Run clinical efficacy extraction (saves to BQ automatically)
            result = await get_dimension_iii_efficacy_data(molecule_name)

            trials = result.get("trials", [])
            score = result.get("scoring_result", {}).get("weighted_score", "N/A")

            elapsed = time.time() - t0
            print(f"\n[{molecule_name}] Complete in {elapsed:.1f}s")
            print(f"[{molecule_name}] Trials: {len(trials)}, Score: {score}")

            return {
                "molecule": molecule_name,
                "status": "success",
                "trials": len(trials),
                "score": score,
                "time": elapsed,
            }

        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n[{molecule_name}] ERROR: {e}")
            return {
                "molecule": molecule_name,
                "status": "error",
                "error": str(e),
                "time": elapsed,
            }


async def run_all_molecules(molecules: list[str], max_concurrency: int) -> list[dict]:
    """Process all molecules concurrently up to the max_concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrency)
    print(f"Running execution pool with max_concurrency={max_concurrency}")
    
    tasks = [process_molecule(m, semaphore) for m in molecules]
    results = await asyncio.gather(*tasks)
    return list(results)


def main():
    parser = argparse.ArgumentParser(description="Run Clinical Efficacy extraction for GLP-1 molecules")
    parser.add_argument("--molecule", help="Process a specific molecule only")
    parser.add_argument("--task-index", type=int, help="Task index for parallel execution")
    parser.add_argument("--task-count", type=int, help="Total task count for parallel execution")
    parser.add_argument("--concurrency", type=int, default=1, help="Simultaneous network execution limit inside container")
    args = parser.parse_args()

    # Consolidate Cloud Run environment variables vs CLI fallbacks cleanly
    task_index = args.task_index if args.task_index is not None else int(os.getenv("CLOUD_RUN_TASK_INDEX", -1))
    task_count = args.task_count if args.task_count is not None else int(os.getenv("CLOUD_RUN_TASK_COUNT", -1))

    print("="*70)
    print("CLINICAL EFFICACY EXTRACTION PIPELINE")
    print(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("="*70)

    # Validate parallel configuration rules before proceeding
    if (task_index >= 0 and task_count <= 0) or (task_index < 0 and task_count > 0):
        print(f"ERROR: Invalid state. Check combination of task-index ({task_index}) and task-count ({task_count}).")
        sys.exit(1)

    # Determine which molecules to process
    if args.molecule:
        molecules = [args.molecule]
        print(f"Mode: Single molecule ({args.molecule})")
    elif task_index >= 0 and task_count > 0:
        # Cloud Run parallel mode (Safe remainder distribution formula)
        all_molecules = get_glp1_molecules()
        
        if task_index >= task_count:
            print(f"ERROR: task-index ({task_index}) cannot be greater than or equal to task-count ({task_count})")
            sys.exit(1)

        total_mols = len(all_molecules)
        per_task = total_mols // task_count
        remainder = total_mols % task_count

        start_idx = (task_index * per_task) + min(task_index, remainder)
        end_idx = start_idx + per_task + (1 if task_index < remainder else 0)

        molecules = all_molecules[start_idx:end_idx]

        print(f"Mode: Parallel Cloud Run (task {task_index + 1}/{task_count})")
        print(f"Processing indices: {start_idx} to {end_idx - 1} out of {total_mols}")
    else:
        molecules = get_glp1_molecules()
        print(f"Mode: All molecules ({len(molecules)} total)")

    print(f"Molecules to process: {len(molecules)}")
    print("-"*70)

    if not molecules:
        print("No molecules assigned to this container execution block.")
        return

    t0 = time.time()

    # Run asynchronous processing wrapper
    results = asyncio.run(run_all_molecules(molecules, args.concurrency))

    # Final summary calculations
    total_time = time.time() - t0
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]

    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Total molecules: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

    if successful:
        total_trials = sum(r.get("trials", 0) for r in successful)
        print(f"Total trials extracted: {total_trials}")

    if failed:
        print("\nFailed molecules:")
        for r in failed:
            print(f"  - {r['molecule']}: {r.get('error', 'Unknown error')}")

    print(f"\nResults saved to BigQuery: {PROJECT_ID}.{DATASET_ID}.clinical_efficacy")


if __name__ == "__main__":
    main()
