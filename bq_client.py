import os
from google.cloud import bigquery
from google.oauth2 import service_account
from market_potential.constants import PROJECT_ID

def get_bq_client():
    """Initialize and return a BigQuery client.

    Uses Google Application Default Credentials (ADC):
    - Local: Uses GOOGLE_APPLICATION_CREDENTIALS env var if set
    - Cloud Run: Automatically uses attached service account
    """
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path and os.path.exists(credentials_path):
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        return bigquery.Client(project=PROJECT_ID, credentials=credentials)

    return bigquery.Client(project=PROJECT_ID)

def run_query(query: str):
    """Run a BigQuery query and return the results as a list of dicts."""
    client = get_bq_client()
    query_job = client.query(query)
    results = query_job.result()
    return [dict(row) for row in results]
