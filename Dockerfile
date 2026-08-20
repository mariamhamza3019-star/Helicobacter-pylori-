# Cloud Run needs a container; this builds one for the FastAPI search app.
FROM python:3.11-slim

WORKDIR /app

# System deps some ML wheels need at build/runtime (e.g. pymupdf, faiss)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects $PORT at runtime and expects the container to listen on it
ENV PORT=8080
EXPOSE 8080

CMD exec uvicorn app:app --host 0.0.0.0 --port $PORT
