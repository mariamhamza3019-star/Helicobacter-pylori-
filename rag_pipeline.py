"""
Clinical RAG pipeline: BM25 + dense FAISS retrieval → RRF hybrid → MedCPT cross-encoder rerank → structured response.

Importable by FastAPI backend, Streamlit, and retrieval-evaluation scripts.

Reranker choice — MedCPT Cross-Encoder (ncbi/MedCPT-Cross-Encoder):
  + Trained on PubMed query–passage pairs; strong on biomedical/clinical text.
  + Already pulled in via ``transformers`` / ``torch`` (same stack as BioBERT embeddings).
  + Meaningfully reorders candidate pool (not just rescaled FAISS scores).
  + Actual reranker scores are computed and preserved with source metadata.
"""
from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np

from generate import (
    DEFAULT_RELEVANCE_THRESHOLD,
    build_refusal_response,
    generate_answer,
    should_refuse_low_relevance,
)
from hybrid_search import (
    MedCPTReranker,
    RRFRerankRetriever,
    SemanticIndex,
    chunk_passage,
    minmax_norm,
    section_multiplier,
)

DENSE_POOL_SIZE = 25
CHUNK_SCHEMA_KEYS = (
    "chunk_id",
    "document_id",
    "text",
    "page",
    "section",
    "source",
    "topic",
)


class DenseRetriever:
    """FAISS + BioBERT dense retrieval only (baseline, no reranker)."""

    name = "dense (FAISS + BioBERT)"

    def __init__(self, base: SemanticIndex):
        self.base = base
        self.chunks = base.chunks

    def search(self, query: str, k: int = 10) -> list[int]:
        sem = self.base.semantic_scores(query)
        return list(np.argsort(-sem)[:k])


class DenseRerankRetriever:
    """Top dense pool → MedCPT cross-encoder rerank; preserves full chunk metadata."""

    name = "dense + MedCPT rerank"

    def __init__(self, base: SemanticIndex, reranker: MedCPTReranker):
        self.dense = DenseRetriever(base)
        self.reranker = reranker
        self.chunks = base.chunks

    def _ranked_with_scores(self, query: str, k: int) -> list[tuple[int, float]]:
        pool_idx = self.dense.search(query, k=DENSE_POOL_SIZE)
        if not pool_idx:
            return []
        passages = [chunk_passage(self.chunks[i]) for i in pool_idx]
        scores = self.reranker.score_pairs(query, passages)
        for j, i in enumerate(pool_idx):
            scores[j] *= section_multiplier(self.chunks[i])
        order = np.argsort(-scores)
        top = order[:k]
        normed = minmax_norm(scores[top])
        return [(pool_idx[i], float(normed[j])) for j, i in enumerate(top)]

    def search(self, query: str, k: int = 10) -> list[int]:
        return [i for i, _ in self._ranked_with_scores(query, k)]

    def search_with_scores(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        return self._ranked_with_scores(query, k)

    def search_with_metadata(self, query: str, k: int = 10) -> list[dict]:
        pool_idx = self.dense.search(query, k=DENSE_POOL_SIZE)
        if not pool_idx:
            return []
        sem_all = self.dense.base.semantic_scores(query)
        passages = [chunk_passage(self.chunks[i]) for i in pool_idx]
        raw_scores = self.reranker.score_pairs(query, passages)
        scores = raw_scores.copy()
        for j, i in enumerate(pool_idx):
            scores[j] *= section_multiplier(self.chunks[i])
        order = np.argsort(-scores)
        top = order[:k]
        normed = minmax_norm(scores[top])

        results = []
        for j, i in enumerate(top):
            idx = pool_idx[i]
            results.append({
                "chunk_index": idx,
                "score": float(normed[j]),
                "raw_score": float(raw_scores[i]),
                "semantic_score": float(sem_all[idx]),
            })
        return results


def chunk_from_store(c: dict, meta: dict | None = None, score: float | None = None) -> dict:
    """Map stored chunk JSON to pipeline schema; all core fields & scores preserved."""
    doc_title = (
        c.get("source")
        or c.get("document_id")
        or "ACG Clinical Guideline 2024: Treatment of Helicobacter pylori Infection"
    )
    out = {
        "chunk_id": c["chunk_id"],
        "document_id": c.get("document_id", "ACG_2024"),
        "document": doc_title,
        "text": c["text"],
        "page": c.get("page_start") or c.get("page"),
        "section": c.get("section", ""),
        "subsection": c.get("subsection", ""),
        "source": doc_title,
        "topic": c.get("topic", ""),
        "content_type": c.get("content_type", "prose"),
    }
    if meta is not None:
        for k, v in meta.items():
            if k != "chunk_index":
                out[k] = v
    elif score is not None:
        out["score"] = score
    return out


def retrieve_ranked(
    retriever: Any,
    query: str,
    k: int,
    all_chunks: list[dict],
) -> list[dict]:
    """Return ranked chunks with schema fields + full rerank & channel scores."""
    if hasattr(retriever, "search_with_metadata"):
        ranked_meta = retriever.search_with_metadata(query, k=k)
        return [chunk_from_store(all_chunks[m["chunk_index"]], meta=m) for m in ranked_meta]
    if hasattr(retriever, "search_with_scores"):
        ranked = retriever.search_with_scores(query, k=k)
        return [chunk_from_store(all_chunks[i], score=s) for i, s in ranked]
    indices = retriever.search(query, k=k)
    return [chunk_from_store(all_chunks[i]) for i in indices]


def _first_sentence(text: str, max_len: int = 320) -> str:
    text = text.strip()
    if not text:
        return ""
    match = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    sentence = match[0] if match else text
    if len(sentence) > max_len:
        return sentence[: max_len - 3].rstrip() + "..."
    return sentence


def _top_score(chunks: list[dict]) -> float | None:
    scores = [c["score"] for c in chunks if c.get("score") is not None]
    return max(scores) if scores else None


def _confidence_label(chunks: list[dict], threshold: float) -> str:
    if should_refuse_low_relevance(chunks, threshold):
        return "low"
    return "high"


def _format_reranked_documents(chunks: list[dict]) -> list[dict]:
    """Format candidate chunks as clean reranked documents with actual reranker scores."""
    reranked = []
    for idx, c in enumerate(chunks, 1):
        doc_name = (
            c.get("document")
            or c.get("source")
            or c.get("document_id")
            or "ACG Clinical Guideline 2024: Treatment of Helicobacter pylori Infection"
        )
        score_val = c.get("score")
        if score_val is not None:
            relevance = round(float(score_val), 4)
        else:
            relevance = None

        doc_item = {
            "rank": idx,
            "chunk_id": c.get("chunk_id", ""),
            "document": doc_name,
            "document_id": c.get("document_id", "ACG_2024"),
            "section": c.get("section", ""),
            "subsection": c.get("subsection", ""),
            "page": c.get("page"),
            "excerpt": c.get("text", ""),
            "text": c.get("text", ""),
            "score": relevance,
            "relevance": relevance,
            "raw_score": round(float(c["raw_score"]), 4) if c.get("raw_score") is not None else None,
            "bm25_score": round(float(c["bm25_score"]), 4) if c.get("bm25_score") is not None else None,
            "semantic_score": round(float(c["semantic_score"]), 4) if c.get("semantic_score") is not None else None,
            "rrf_score": round(float(c["rrf_score"]), 6) if c.get("rrf_score") is not None else None,
            "content_type": c.get("content_type", "prose"),
        }
        reranked.append(doc_item)
    return reranked


def generation_to_structured(generated: dict, chunks: list[dict]) -> dict:
    """
    Convert internal GenerationResponse dict to public shape:
    recommendation, evidence/excerpt[], citations[], reranked_documents, confidence.
    """
    lookup = {c["chunk_id"]: c for c in chunks}
    excerpts: list[str] = []
    citations: list[dict] = []

    for cite in generated.get("citations") or []:
        excerpt_text = cite.get("excerpt", "").strip()
        if excerpt_text:
            excerpts.append(excerpt_text)
        chunk = lookup.get(cite.get("chunk_id", ""), {})
        doc_name = (
            chunk.get("document")
            or chunk.get("source")
            or cite.get("document_name")
            or "ACG Clinical Guideline 2024: Treatment of Helicobacter pylori Infection"
        )
        citations.append(
            {
                "document": doc_name,
                "document_id": chunk.get("document_id") or cite.get("document_name", "ACG_2024"),
                "section": cite.get("section") or chunk.get("section", ""),
                "subsection": chunk.get("subsection", ""),
                "chunk_id": cite.get("chunk_id") or chunk.get("chunk_id", ""),
                "page": cite.get("page") if cite.get("page") is not None else chunk.get("page"),
                "excerpt": excerpt_text,
            }
        )

    confidence = "high" if generated.get("answer_status") == "answered" else "low"
    return {
        "recommendation": generated.get("recommendation", ""),
        "excerpt": excerpts,
        "evidence": excerpts,
        "citation": citations,
        "citations": citations,
        "reranked_documents": _format_reranked_documents(chunks),
        "confidence": confidence,
        "answer_status": generated.get("answer_status", "answered"),
        "refusal_reason": generated.get("refusal_reason"),
        "suggested_followups": generated.get("suggested_followups", []),
        "_meta": generated.get("_meta", {}),
    }


def extractive_structured_response(
    chunks: list[dict],
    *,
    low_confidence: bool,
    refusal_reason: str | None = None,
    max_excerpts: int = 3,
    excerpt_chars: int = 600,
) -> dict:
    """Offline fallback: verbatim excerpts from top reranked chunks, no LLM."""
    if not chunks:
        return {
            "recommendation": (
                "No relevant guideline excerpts were retrieved. "
                "Consult the source guideline directly."
            ),
            "excerpt": [],
            "evidence": [],
            "citation": [],
            "citations": [],
            "reranked_documents": [],
            "confidence": "low",
            "answer_status": "insufficient_context",
            "refusal_reason": refusal_reason or "No chunks retrieved.",
            "_meta": {"llm_called": False},
        }

    if low_confidence:
        recommendation = (
            "The retrieved guideline excerpts do not provide enough relevant "
            "information to answer this question safely. "
            "Consult the ACG/WGO guideline directly before clinical action."
        )
        status = "insufficient_context"
    else:
        recommendation = (
            "Based on the top retrieved guideline excerpt: "
            + _first_sentence(chunks[0]["text"])
        )
        status = "answered"

    top = chunks[:max_excerpts] if not low_confidence else []
    excerpts = [c["text"][:excerpt_chars].strip() for c in top]
    citations = [
        {
            "document": c.get("document") or c.get("source", "ACG Clinical Guideline 2024"),
            "document_id": c.get("document_id", "ACG_2024"),
            "section": c.get("section", ""),
            "subsection": c.get("subsection", ""),
            "chunk_id": c.get("chunk_id", ""),
            "page": c.get("page"),
            "excerpt": c["text"][:excerpt_chars].strip(),
        }
        for c in top
    ]

    return {
        "recommendation": recommendation,
        "excerpt": excerpts,
        "evidence": excerpts,
        "citation": citations,
        "citations": citations,
        "reranked_documents": _format_reranked_documents(chunks),
        "confidence": "low" if low_confidence else "high",
        "answer_status": status,
        "refusal_reason": refusal_reason,
        "_meta": {"llm_called": False, "top_score": _top_score(chunks)},
    }


def _build_retrieval_query(
    query: str,
    history: list[dict] | None,
    max_context_turns: int = 2,
) -> str:
    """
    Cheap, zero-latency-cost context expansion for retrieval ONLY (never shown
    to the user, never sent to generation as "the question"). Concatenates the
    most recent user turn(s) with the current query so BM25/dense retrieval
    has enough keywords to find the right chunks for short follow-ups like
    "what about in children?" — without an extra LLM call to rewrite it.
    """
    if not history:
        return query
    recent_user_turns = [
        (turn.get("content") or "").strip()
        for turn in history
        if turn.get("role") == "user" and (turn.get("content") or "").strip()
    ][-max_context_turns:]
    if not recent_user_turns:
        return query
    return " ".join(recent_user_turns) + " " + query


def run_clinical_rag(
    query: str,
    all_chunks: list[dict],
    retriever: Any,
    *,
    k: int = 5,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    use_llm: bool = True,
    generate_fn: Callable[..., dict] | None = None,
    history: list[dict] | None = None,
) -> dict:
    """
    End-to-end: BM25 + dense FAISS → hybrid RRF pool → MedCPT rerank → structured response.

    Returns JSON-serializable dict with keys:
      recommendation, evidence/excerpt, citations/citation, reranked_documents,
      confidence, answer_status, refusal_reason, _meta.
    """
    retrieval_query = _build_retrieval_query(query, history)
    ranked = retrieve_ranked(retriever, retrieval_query, k, all_chunks)
    low = _confidence_label(ranked, relevance_threshold)

    if low == "low":
        top = _top_score(ranked)
        if not ranked:
            reason = "No chunks were retrieved for this query."
        elif top is None:
            reason = "Retrieved chunks lack reranker scores for confidence gating."
        else:
            reason = (
                f"Top rerank score {top:.3f} is below threshold {relevance_threshold:.3f}."
            )
        if use_llm and generate_fn is not None:
            generated = build_refusal_response(reason)
            generated["_meta"] = {"llm_called": False, "top_score": top}
            return generation_to_structured(generated, ranked)
        return extractive_structured_response(ranked, low_confidence=True, refusal_reason=reason)

    if use_llm:
        try:
            generated = (generate_fn or generate_answer)(query, ranked, history=history)
            return generation_to_structured(generated, ranked)
        except RuntimeError:
            pass

    return extractive_structured_response(ranked, low_confidence=False)


def ordering_changed(dense_indices: list[int], rerank_indices: list[int]) -> bool:
    """True when reranking changed rank order (not just relabeled scores)."""
    return dense_indices != rerank_indices
