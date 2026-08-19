"""
FastAPI backend for H. pylori Clinical RAG Pipeline.
 
Endpoints:
  GET  /health
  POST /api/query
  GET  /api/gold-questions
"""
from __future__ import annotations
 
import json
import mimetypes
import os
import sys
import time
 
# Ensure Windows doesn't serve .js files as text/plain
mimetypes.init()
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("text/html", ".html")
 
# Redirect HF model cache to project directory where symlinks work and disk
# space is available (avoids the broken Windows AppData LocalCache path).
_HF_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache")
os.environ.setdefault("HF_HOME", _HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", _HF_CACHE)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List, Optional
 
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
 
# Ensure repo root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
 
from generate import (
    DEFAULT_RELEVANCE_THRESHOLD,
    generate_answer,
)
from hybrid_search import (
    BM25Retriever,
    MedCPTReranker,
    MinMaxRetriever,
    RRFRerankRetriever,
    RRFRetriever,
    SemanticIndex,
)
from paths import CHUNKS_JSON, FAISS_INDEX, GOLD_QUESTIONS
from rag_pipeline import (
    DenseRerankRetriever,
    DenseRetriever,
    run_clinical_rag,
)
 
# ---------------------------------------------------------------------------
# Global state / singleton cache
# ---------------------------------------------------------------------------
state: dict[str, Any] = {
    "chunks": [],
    "base": None,
    "reranker": None,
    "retrievers": {},
    "gold_questions": [],
}
 
 
def get_base_and_chunks() -> tuple[SemanticIndex, list[dict]]:
    if state["base"] is None:
        if not CHUNKS_JSON.exists() or not FAISS_INDEX.exists():
            raise RuntimeError(
                f"Missing chunks ({CHUNKS_JSON}) or FAISS index ({FAISS_INDEX}). "
                "Ensure ingestion and indexing have run."
            )
        with open(CHUNKS_JSON, encoding="utf-8") as f:
            chunks = json.load(f)
        state["chunks"] = chunks
        state["base"] = SemanticIndex(chunks)
    return state["base"], state["chunks"]
 
 
def get_reranker() -> MedCPTReranker:
    if state["reranker"] is None:
        state["reranker"] = MedCPTReranker()
    return state["reranker"]
 
 
def get_retriever(key: str = "rrf_rerank"):
    if key in state["retrievers"]:
        return state["retrievers"][key]
 
    base, _ = get_base_and_chunks()
    if key == "bm25":
        retriever = BM25Retriever(base)
    elif key == "minmax":
        retriever = MinMaxRetriever(base)
    elif key == "rrf":
        retriever = RRFRetriever(base)
    elif key == "dense":
        retriever = DenseRetriever(base)
    elif key == "dense_rerank":
        retriever = DenseRerankRetriever(base, get_reranker())
    elif key == "rrf_rerank":
        retriever = RRFRerankRetriever(base, get_reranker())
    else:
        raise ValueError(f"Unknown pipeline: {key}")
 
    state["retrievers"][key] = retriever
    return retriever
 
 
def load_gold_questions() -> list[dict]:
    if not state["gold_questions"]:
        if GOLD_QUESTIONS.exists():
            with open(GOLD_QUESTIONS, encoding="utf-8") as f:
                data = json.load(f)
                state["gold_questions"] = data.get("questions", [])
    return state["gold_questions"]
 
 
# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing H. pylori RAG models and index...")
    try:
        get_base_and_chunks()
        get_reranker()
        get_retriever("rrf_rerank")
        load_gold_questions()
        print(f"RAG initialization complete: {len(state['chunks'])} chunks loaded.")
    except Exception as exc:
        print(f"Warning: model initialization deferred or failed: {exc}")
    yield
 
 
app = FastAPI(
    title="H. pylori Clinical Guideline RAG API",
    description="FastAPI backend for ACG 2024 H. pylori clinical decision support and retrieval evaluation.",
    version="1.0.0",
    lifespan=lifespan,
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
 
# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class ChatTurn(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Turn text")
 
 
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Clinical question to ask")
    top_k: int = Field(5, ge=1, le=25, description="Number of evidence chunks to retrieve")
    pipeline: str = Field("rrf_rerank", description="Retrieval pipeline to use (rrf_rerank, rrf, dense_rerank, minmax, bm25)")
    relevance_threshold: float = Field(DEFAULT_RELEVANCE_THRESHOLD, ge=0.0, le=1.0, description="Minimum reranker score threshold")
    use_llm: bool = Field(True, description="Whether to invoke LLM generation if API key is set")
    history: List[ChatTurn] = Field(default_factory=list, description="Prior conversation turns, oldest first")
 
 
class CitationItem(BaseModel):
    document: str
    document_id: str
    section: str
    subsection: Optional[str] = None
    chunk_id: str
    page: Optional[int] = None
    excerpt: str
 
 
class RerankedDocumentItem(BaseModel):
    rank: int
    chunk_id: str
    document: str
    document_id: str
    section: str
    subsection: Optional[str] = None
    page: Optional[int] = None
    excerpt: str
    text: str
    score: Optional[float] = None
    relevance: Optional[float] = None
    raw_score: Optional[float] = None
    bm25_score: Optional[float] = None
    semantic_score: Optional[float] = None
    rrf_score: Optional[float] = None
    content_type: Optional[str] = "prose"
 
 
class QueryResponse(BaseModel):
    model_config = {"populate_by_name": True}
 
    recommendation: str
    evidence: List[str]
    citations: List[CitationItem]
    reranked_documents: List[RerankedDocumentItem]
    confidence: str
    answer_status: str
    refusal_reason: Optional[str] = None
    latency_ms: float
    pipeline_used: str
    meta: dict = Field(default_factory=dict, alias="_meta")
 
 
class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    num_chunks: int
    has_api_key: bool
    pipeline: str
 
 
# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint confirming index, chunk count, and model readiness."""
    has_key = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("GENERATION_API_KEY"))
    return {
        "status": "ok",
        "index_loaded": state["base"] is not None,
        "num_chunks": len(state["chunks"]),
        "has_api_key": has_key,
        "pipeline": "rrf_rerank",
    }
 
 
@app.post("/api/query", response_model=QueryResponse)
def query_guidelines(req: QueryRequest):
    """
    Run end-to-end clinical RAG pipeline:
      Question → BM25 + Semantic → Hybrid RRF → MedCPT Reranking → Evidence & LLM Answer.
    """
    query_str = req.query.strip()
    if not query_str:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
 
    t0 = time.perf_counter()
 
    try:
        base, chunks = get_base_and_chunks()
        retriever = get_retriever(req.pipeline)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load retrieval pipeline: {exc}")
 
    has_api_key = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("GENERATION_API_KEY"))
    should_call_llm = req.use_llm and has_api_key
 
    result = run_clinical_rag(
        query_str,
        chunks,
        retriever,
        k=req.top_k,
        relevance_threshold=req.relevance_threshold,
        use_llm=should_call_llm,
        generate_fn=generate_answer if should_call_llm else None,
        history=[turn.model_dump() for turn in req.history],
    )
 
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
 
    return {
        "recommendation": result.get("recommendation", ""),
        "evidence": result.get("evidence") or result.get("excerpt") or [],
        "citations": result.get("citations") or result.get("citation") or [],
        "reranked_documents": result.get("reranked_documents") or [],
        "confidence": result.get("confidence", "high"),
        "answer_status": result.get("answer_status", "answered"),
        "refusal_reason": result.get("refusal_reason"),
        "latency_ms": round(elapsed_ms, 2),
        "pipeline_used": req.pipeline,
        "_meta": result.get("_meta", {}),
    }
 
 
@app.get("/api/gold-questions")
def get_gold_questions():
    """Return gold question benchmark items for testing and evaluation."""
    questions = load_gold_questions()
    return {"total": len(questions), "questions": questions}
 
 
# ---------------------------------------------------------------------------
# Static frontend mounting (if built)
# ---------------------------------------------------------------------------
DIST_DIR = ROOT_DIR / "frontend" / "dist"
if DIST_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets", html=False), name="assets")
 
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = DIST_DIR / full_path
        if file_path.is_file():
            media_type, _ = mimetypes.guess_type(str(file_path))
            return FileResponse(file_path, media_type=media_type)
        return FileResponse(DIST_DIR / "index.html", media_type="text/html")
 
 
if __name__ == "__main__":
    import uvicorn
 
    uvicorn.run(app, host="127.0.0.1", port=8000)
 
