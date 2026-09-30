"""Instructor-only idempotent RAG ingestion and corpus-vs-bucket delete sync.

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
import vertexai
from vertexai.preview import rag

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
CORPUS_DISPLAY_NAME = "acsm-credit-policy-corpus"
CORPUS_DESCRIPTION = (
    "AEON Credit Service Malaysia (ACSM) Credit Policy, Product Disclosure "
    "Sheets, BNM Compliance, Collections SOPs, and Circulars."
)

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
) -> dict[str, dict]:
    bucket = storage_client.bucket(bucket_name)
    if not bucket.exists():
        bucket = storage_client.create_bucket(bucket_name, location=region)
        bucket.iam_configuration.uniform_bucket_level_access_enabled = True
        bucket.patch()

    manifest_rows = list(csv.DictReader((corpus_dir / "manifest.csv").open("r", encoding="utf-8")))
    chunks_by_doc: dict[str, list[dict]] = {}
    for line in (corpus_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            chunks_by_doc.setdefault(row["doc_id"], []).append(row)

    doc_map: dict[str, dict] = {}
    for row in manifest_rows:
        doc_id = row["doc_id"]
        rel_path = row["file_path"]
        local_file = corpus_dir / rel_path
        ext = local_file.suffix.lower()
        content_type = MIME_MAP.get(ext) or mimetypes.guess_type(local_file.name)[0] or "application/octet-stream"

        raw_blob = bucket.blob(f"raw/{rel_path}")
        raw_blob.content_disposition = "inline"
        raw_blob.upload_from_filename(str(local_file), content_type=content_type)

        if row["format"] in ("pdf", "docx", "html"):
            rag_rel_path = rel_path
            bucket.blob(f"rag-engine/{rag_rel_path}").upload_from_filename(str(local_file), content_type=content_type)
        else:
            rag_rel_path = str(Path(rel_path).with_suffix(".md"))
            md_lines = [
                f"# {row['title']} ({doc_id})",
                f"- **Document ID:** {doc_id}",
                f"- **Version:** {row['version']}",
                f"- **Effective Date:** {row['effective_date']}",
                f"- **Access Level:** {row['access']}",
                "",
            ]
            for ch in chunks_by_doc.get(doc_id, []):
                md_lines.extend([f"## Clause {ch['clause_id']} — {ch['heading']}", ch["chunk_text"], ""])
            bucket.blob(f"rag-engine/{rag_rel_path}").upload_from_string(
                "\n".join(md_lines), content_type="text/markdown; charset=utf-8"
            )

        doc_map[doc_id] = {
            "doc_id": doc_id,
            "raw_file_path": rel_path,
            "rag_file_path": rag_rel_path,
            "gcs_rag_uri": f"gs://{bucket_name}/rag-engine/{rag_rel_path}",
        }
    return doc_map


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


def sync_rag_engine(project: str, region: str, doc_map: dict[str, dict]) -> str:
    vertexai.init(project=project, location=region)
    existing = [c for c in rag.list_corpora() if c.display_name == CORPUS_DISPLAY_NAME]
    if existing:
        corpus = existing[0]
    else:
        embedding_cfg = rag.RagEmbeddingModelConfig(
            vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                publisher_model=f"publishers/google/models/{EMBED_MODEL}"
            )
        )
        corpus = rag.create_corpus(
            display_name=CORPUS_DISPLAY_NAME,
            description=CORPUS_DESCRIPTION,
            backend_config=rag.RagVectorDbConfig(rag_embedding_model_config=embedding_cfg),
        )

    desired_uris = sorted({info["gcs_rag_uri"] for info in doc_map.values()})
    desired_names = {Path(u).name for u in desired_uris}

    # Delete any RagFiles whose source object was removed from the desired set.
    for rf in rag.list_files(corpus_name=corpus.name):
        if rf.display_name not in desired_names:
            rag.delete_file(name=rf.name, corpus_name=corpus.name)

    # Import in batches of 20 (RAG Engine skips unchanged files and updates changed files in place).
    for i in range(0, len(desired_uris), 20):
        rag.import_files(
            corpus_name=corpus.name,
            paths=desired_uris[i : i + 20],
            transformation_config=rag.TransformationConfig(
                chunking_config=rag.ChunkingConfig(chunk_size=512, chunk_overlap=100)
            ),
            max_embedding_requests_per_min=900,
        )
    return corpus.name


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
    doc_map = sync_bucket(storage_client, bucket_name, args.region, corpus_dir)
    sync_bigquery(args.project, args.region, args.dataset, corpus_dir)
    corpus_name = sync_rag_engine(args.project, args.region, doc_map)
    print(f"RAG_BUCKET={bucket_name}")
    print(f"RAG_CORPUS={corpus_name}")


if __name__ == "__main__":
    main()
