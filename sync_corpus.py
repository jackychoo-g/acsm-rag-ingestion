"""Instructor-only idempotent RAG ingestion: local policy files -> GCS -> BigQuery.

Every run leaves the same end state: `raw/` in the bucket matches manifest.csv
(stale objects are deleted), policy_chunks is reloaded with WRITE_TRUNCATE and
the audit table is recreated.

Usage:
  uv run python sync_corpus.py --project <project-id> [--corpus-dir ./corpus]
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
from pathlib import Path

from google import genai
from google.cloud import bigquery, storage
from google.genai import types

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
RAW_PREFIX = "raw/"

MIME_MAP = {
    ".pdf": "application/pdf",
    ".html": "text/html; charset=utf-8",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".csv": "text/plain; charset=utf-8",
    ".eml": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
}


def sync_bucket(
    storage_client: storage.Client,
    bucket_name: str,
    region: str,
    corpus_dir: Path,
) -> int:
    """Upload every manifest file to raw/ and delete raw/ objects no longer in the manifest."""
    bucket = storage_client.bucket(bucket_name)
    if not bucket.exists():
        bucket = storage_client.create_bucket(bucket_name, location=region)
        bucket.iam_configuration.uniform_bucket_level_access_enabled = True
        bucket.patch()

    manifest_rows = list(csv.DictReader((corpus_dir / "manifest.csv").open("r", encoding="utf-8")))
    desired: set[str] = set()
    for row in manifest_rows:
        rel_path = row["file_path"]
        local_file = corpus_dir / rel_path
        ext = local_file.suffix.lower()
        content_type = MIME_MAP.get(ext) or mimetypes.guess_type(local_file.name)[0] or "application/octet-stream"

        blob_name = f"{RAW_PREFIX}{rel_path}"
        raw_blob = bucket.blob(blob_name)
        # Inline disposition so #page=N citation links open in the browser.
        raw_blob.content_disposition = "inline"
        raw_blob.upload_from_filename(str(local_file), content_type=content_type)
        desired.add(blob_name)

    for blob in storage_client.list_blobs(bucket_name, prefix=RAW_PREFIX):
        if blob.name not in desired:
            blob.delete()
    return len(desired)


def sync_bigquery(project: str, region: str, dataset: str, corpus_dir: Path) -> None:
    chunks = [
        json.loads(line)
        for line in (corpus_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ai = genai.Client(vertexai=True, project=project, location=region)
    for i in range(0, len(chunks), 50):
        batch = chunks[i : i + 50]
        resp = ai.models.embed_content(
            model=EMBED_MODEL,
            contents=[c["chunk_text"] for c in batch],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=EMBED_DIM),
        )
        for c, emb in zip(batch, resp.embeddings):
            c["embedding"] = list(emb.values)

    bq = bigquery.Client(project=project, location=region)
    ds_ref = bigquery.Dataset(f"{project}.{dataset}")
    ds_ref.location = region
    bq.create_dataset(ds_ref, exists_ok=True)

    schema = [
        bigquery.SchemaField("chunk_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("doc_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("family", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("title", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("version", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("effective_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("language", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("access", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("format", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("moment", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("clause_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("heading", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chunk_text", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("file_path", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
    ]
    bq.load_table_from_json(
        chunks,
        f"{project}.{dataset}.policy_chunks",
        job_config=bigquery.LoadJobConfig(schema=schema, write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE),
    ).result()

    ddl_audit = f"""
    CREATE OR REPLACE TABLE `{project}.{dataset}.collections_internal_audit` AS
    SELECT * FROM UNNEST([
      STRUCT('AUD-2026-001' AS audit_id, 'Johor Bahru' AS branch, 'MHP-I' AS product,
             'Approved 3-point DSR exception under Clause 2.5 with 24m clean CCRIS' AS finding,
             DATE '2026-06-15' AS audit_date),
      STRUCT('AUD-2026-002', 'Penang', 'CARD-GOLD',
             'Referred to income verification under Clause 3.3 due to bank statement mismatch',
             DATE '2026-06-18'),
      STRUCT('AUD-2026-003', 'Kuala Lumpur HQ', 'PF-I',
             'Escalated to Credit Committee under Clause 7.2 for dual policy exception',
             DATE '2026-06-20')
    ]);
    """
    bq.query(ddl_audit).result()


def _default_project() -> str:
    for name in ("ACSM_PROJECT", "PROJECT", "GOOGLE_CLOUD_PROJECT"):
        val = os.environ.get(name, "").strip()
        if val and not val.isdigit():
            return val
    try:
        import subprocess
        val = subprocess.check_output(
            ["gcloud", "config", "get-value", "project"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        if val and val != "(unset)" and not val.isdigit():
            return val
    except Exception:
        pass
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=_default_project())
    parser.add_argument("--region", default=os.environ.get("ACSM_REGION", "asia-southeast1"))
    parser.add_argument("--dataset", default=os.environ.get("ACSM_RAG_DATASET", "acsm_rag"))
    parser.add_argument("--corpus-dir", default="corpus")
    args = parser.parse_args()
    if not args.project:
        raise SystemExit("Pass --project <project-id> or run `gcloud config set project <project-id>`.")

    corpus_dir = Path(args.corpus_dir).resolve()
    bucket_name = f"{args.project}-acsm-rag-corpus"
    storage_client = storage.Client(project=args.project)
    file_count = sync_bucket(storage_client, bucket_name, args.region, corpus_dir)
    sync_bigquery(args.project, args.region, args.dataset, corpus_dir)
    print(f"Synced {file_count} files to gs://{bucket_name}/{RAW_PREFIX}")
    print(f"RAG_BUCKET={bucket_name}")
    print(f"RAG_TABLE={args.project}.{args.dataset}.policy_chunks")


if __name__ == "__main__":
    main()
