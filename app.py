"""
Minimal API layer for the H. pylori RAG project.
Loads the FAISS index + metadata that ingestion/embedding already produced
(outputs/h_pylori_faiss.index, outputs/h_pylori_metadata.json) and exposes
a /search endpoint. Does not touch or re-run the pipeline.
"""

import os
import json

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer

# --- Config -----------------------------------------------------------
INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "outputs/h_pylori_faiss.index")
METADATA_PATH = os.getenv("FAISS_METADATA_PATH", "outputs/h_pylori_metadata.json")
EMBED_MODEL_NAME = os.getenv(
    "EMBED_MODEL_NAME", "pritamdeka/S-BioBert-snli-multinli-stsb"
)

app = FastAPI(title="H. pylori RAG Search API")

_index = None
_metadata = None
_model = None


def _load_resources():
    """Lazy-load heavy resources once, on first request or startup."""
    global _index, _metadata, _model

    if _index is None:
        if not os.path.exists(INDEX_PATH):
            raise RuntimeError(
                f"FAISS index not found at {INDEX_PATH}. "
                "Make sure vectorDB/h_pylori_faiss.index is committed to the repo "
                "(or generated as part of the build)."
            )
        _index = faiss.read_index(INDEX_PATH)

    if _metadata is None:
        if not os.path.exists(METADATA_PATH):
            raise RuntimeError(f"Metadata file not found at {METADATA_PATH}.")
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _metadata = json.load(f)
        # normalize to a list we can index by position
        if isinstance(_metadata, dict):
            # common pattern: {"chunks": [...]}
            _metadata = _metadata.get("chunks", list(_metadata.values()))

    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)


@app.on_event("startup")
def startup_event():
    # Fail fast with a clear log line instead of a silent crash loop,
    # but don't block startup if files aren't ready yet (Railway healthcheck
    # just needs the port to open).
    try:
        _load_resources()
    except Exception as e:  # noqa: BLE001
        print(f"[startup] Resources not loaded yet: {e}")


@app.get("/")
def root():
    return {"status": "ok", "service": "h-pylori-rag-search"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/search")
def search(q: str, k: int = 5):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required.")

    try:
        _load_resources()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(e))

    query_vec = _model.encode([q], normalize_embeddings=True)
    query_vec = np.asarray(query_vec, dtype="float32")

    scores, indices = _index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_metadata):
            continue
        entry = _metadata[idx]
        results.append({"score": float(score), "chunk": entry})

    return {"query": q, "results": results}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
