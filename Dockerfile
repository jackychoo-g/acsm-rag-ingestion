FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml sync_corpus.py ./
COPY corpus ./corpus
RUN pip install --no-cache-dir google-genai google-cloud-bigquery google-cloud-storage google-cloud-aiplatform
ENTRYPOINT ["python", "sync_corpus.py"]
