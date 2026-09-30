# ACSM RAG Ingestion (Instructor-Only)

Idempotent ingestion job that populates the shared workshop RAG data in `asia-southeast1`:

1. Uploads the 47 policy files from `corpus/` to `gs://<project>-acsm-rag-corpus/raw/` (with `Content-Disposition: inline` so `#page=N` links open in the browser) and deletes any object under `raw/` that is no longer listed in `corpus/manifest.csv`.
2. Embeds all 440 chunks in `corpus/chunks.jsonl` with `gemini-embedding-001` (768 dimensions) and loads `<project>.acsm_rag.policy_chunks` plus the restricted `<project>.acsm_rag.collections_internal_audit` table.

Re-running the job is safe. `policy_chunks` is loaded with `WRITE_TRUNCATE` and the audit table is recreated, so each run ends in the same state.

## Run Locally

```bash
git clone https://github.com/jackychoo-g/acsm-rag-ingestion.git
cd acsm-rag-ingestion
uv run python sync_corpus.py --project <workshop-project-id>
```

The script prints `RAG_BUCKET` and `RAG_TABLE` at the end. Run `instructor/setup-shared-project.sh` from the lab repo afterwards to grant participants access and write the `acsm-lab-config` secret.

## Run as a Cloud Run Job

```bash
gcloud run jobs deploy acsm-rag-ingestion \
  --source . \
  --region asia-southeast1 \
  --project <workshop-project-id> \
  --set-env-vars ACSM_PROJECT=<workshop-project-id>,ACSM_REGION=asia-southeast1 \
  --task-timeout 1800s \
  --execute-now
```
