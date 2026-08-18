"""
Streamlit tester for the ACG H. pylori RAG retrieval stack.

Drop this file in the repo ROOT (same folder as paths.py / hybrid_search.py)
and run:

    streamlit run streamlit_app.py

Requires the ingestion + embedding steps to already be done, i.e.
outputs/h_pylori_faiss.index and ingestion/data/processed/acg_chunks.json
must exist (run `python run_pipeline.py --embed` first if not).

Tab 3 runs the full retrieve → generate → safety-check pipeline.
"""

import json
import os
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate import generate_answer  # noqa: E402
from paths import CHUNKS_JSON, FAISS_INDEX, GOLD_QUESTIONS  # noqa: E402
from rag_pipeline import (  # noqa: E402
    CHUNK_SCHEMA_KEYS,
    DenseRerankRetriever,
    run_clinical_rag,
)
from hybrid_search import (  # noqa: E402
    SemanticIndex,
    MinMaxRetriever,
    BM25Retriever,
    RRFRetriever,
    RRFRerankRetriever,
    MedCPTReranker,
)

SHIPPING_PIPELINE = "rrf_rerank"

st.set_page_config(page_title="H. pylori RAG — Tester", layout="wide", page_icon="🔬")

PIPELINES = {
    "RRF + MedCPT rerank (shipping stack)": "rrf_rerank",
    "RRF + section downrank": "rrf",
    "Hybrid minmax (70/30, old)": "minmax",
    "BM25 only": "bm25",
}

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1200px;}
    .hit-card {
        border: 1px solid rgba(120,120,120,0.25);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
        background: rgba(120,120,120,0.04);
    }
    .hit-rank {
        display: inline-block;
        background: #2563eb;
        color: white;
        font-weight: 600;
        font-size: 0.75rem;
        border-radius: 6px;
        padding: 1px 8px;
        margin-right: 8px;
    }
    .hit-meta {
        font-size: 0.82rem;
        opacity: 0.75;
        margin-bottom: 4px;
    }
    .section-tag {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.02em;
        opacity: 0.6;
    }
    .expect-hit {border-left: 4px solid #16a34a;}
    .placeholder-box {
        border: 2px dashed rgba(120,120,120,0.35);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        opacity: 0.8;
    }
    .answer-card {
        border: 1px solid rgba(37,99,235,0.35);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        background: rgba(37,99,235,0.06);
    }
    .cite-card {
        border-left: 3px solid #2563eb;
        padding: 0.5rem 0 0.5rem 0.9rem;
        margin-bottom: 0.5rem;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading FAISS index + embedder (first run only)...")
def load_base():
    if not CHUNKS_JSON.exists() or not FAISS_INDEX.exists():
        return None, None
    chunks = json.load(open(CHUNKS_JSON, encoding="utf-8"))
    base = SemanticIndex(chunks)
    return base, chunks


@st.cache_resource(show_spinner="Loading MedCPT reranker (first run only)...")
def load_reranker():
    return MedCPTReranker()


@st.cache_resource(show_spinner="Loading dense→rerank pipeline...")
def load_clinical_retriever(_base):
    return DenseRerankRetriever(_base, load_reranker())


def get_retriever(key: str, base):
    if key == "bm25":
        return BM25Retriever(base)
    if key == "minmax":
        return MinMaxRetriever(base)
    if key == "rrf":
        return RRFRetriever(base)
    if key == "rrf_rerank":
        return RRFRerankRetriever(base, load_reranker())
    raise ValueError(key)


def load_gold():
    if not GOLD_QUESTIONS.exists():
        return []
    return json.load(open(GOLD_QUESTIONS, encoding="utf-8"))["questions"]


def chunk_for_generation(c: dict, score: float | None = None) -> dict:
    """Map a stored chunk to the schema expected by generate_answer()."""
    out = {
        "chunk_id": c["chunk_id"],
        "document_id": c.get("document_id", ""),
        "text": c["text"],
        "page": c.get("page_start") or c.get("page"),
        "section": c.get("section", ""),
        "source": c.get("source", ""),
        "topic": c.get("topic", ""),
    }
    if score is not None:
        out["score"] = score
    return out


def search_with_optional_scores(
    retriever, query: str, k: int
) -> list[tuple[int, float | None]]:
    """Return (chunk_index, score) pairs; score is set when reranker scores are available."""
    if isinstance(retriever, RRFRerankRetriever):
        return retriever.search_with_scores(query, k=k)
    indices = retriever.search(query, k=k)
    return [(i, None) for i in indices]


def retrieve_for_generation(
    retriever, query: str, k: int, all_chunks: list[dict]
) -> tuple[list[dict], list[int]]:
    ranked = search_with_optional_scores(retriever, query, k)
    indices = [i for i, _ in ranked]
    gen_chunks = [chunk_for_generation(all_chunks[i], score=s) for i, s in ranked]
    return gen_chunks, indices


def render_structured_response(result: dict):
    """Recommendation + expandable excerpts + citations (public pipeline shape)."""
    confidence = result.get("confidence", "high")
    if confidence == "low" or result.get("answer_status") == "insufficient_context":
        st.warning("⚠️ Low confidence — consult the source guideline directly before clinical action.")

    st.markdown(
        f'<div class="answer-card"><strong>Recommendation</strong><p>{result.get("recommendation", "")}</p></div>',
        unsafe_allow_html=True,
    )

    if result.get("refusal_reason"):
        st.caption(f"Reason: {result['refusal_reason']}")

    excerpts = result.get("excerpt") or []
    citations = result.get("citation") or []
    if excerpts:
        st.markdown("#### Supporting excerpts")
        for idx, (excerpt, cite) in enumerate(zip(excerpts, citations), 1):
            doc = cite.get("document_id", "")
            section = cite.get("section", "")
            page = cite.get("page", "")
            label = f"Source {idx}: {doc} · {section} · p.{page}"
            with st.expander(label, expanded=(idx == 1)):
                st.markdown(f'<div class="cite-card">"{excerpt}"</div>', unsafe_allow_html=True)
                st.markdown(
                    f"**Citation:** `{doc}` · {section} · page {page}",
                )

    meta = result.get("_meta") or {}
    warnings = meta.get("citation_warnings") or []
    if warnings:
        with st.expander("Citation verification warnings"):
            for w in warnings:
                st.caption(w)


def render_generation_result(result: dict):
    """Legacy adapter — maps old citations[] shape to structured renderer."""
    if "excerpt" in result and "citation" in result:
        render_structured_response(result)
        return
    excerpts = [c.get("excerpt", "") for c in (result.get("citations") or [])]
    citations = [
        {
            "document_id": c.get("document_name", ""),
            "section": c.get("section", ""),
            "page": c.get("page"),
        }
        for c in (result.get("citations") or [])
    ]
    render_structured_response(
        {
            "recommendation": result.get("recommendation", ""),
            "excerpt": excerpts,
            "citation": citations,
            "confidence": "high" if result.get("answer_status") == "answered" else "low",
            "answer_status": result.get("answer_status"),
            "refusal_reason": result.get("refusal_reason"),
            "_meta": result.get("_meta", {}),
        }
    )


def render_hit(rank: int, c: dict, expect_sections=None, score: float | None = None):
    page = c.get("page_start") or c.get("page")
    is_expected = expect_sections and c.get("section", "").upper() in {
        s.upper() for s in expect_sections
    }
    card_class = "hit-card expect-hit" if is_expected else "hit-card"
    sub = f" / {c['subsection']}" if c.get("subsection") else ""
    score_note = f" · score {score:.3f}" if score is not None else ""
    st.markdown(
        f"""
        <div class="{card_class}">
            <span class="hit-rank">#{rank}</span>
            <span class="section-tag">{c.get('section','')}{sub}</span>
            <div class="hit-meta">{c.get('chunk_id','')} · p.{page} · {c.get('content_type','prose')}{score_note}
            {' · ✅ expected section' if is_expected else ''}</div>
            <div>{c['text']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header + index load
# ---------------------------------------------------------------------------
st.title("🔬 H. pylori RAG — Tester")
st.caption("ACG 2024 H. pylori guideline · retrieval evaluation workbench")

base, chunks = load_base()

if base is None:
    st.error(
        "Couldn't find the FAISS index or chunks file.\n\n"
        f"Expected:\n- `{CHUNKS_JSON}`\n- `{FAISS_INDEX}`\n\n"
        "Run the ingestion + embedding pipeline first:\n\n"
        "```\npython run_pipeline.py --embed\n"
        "# then build the FAISS index in notebooks/VectorDB_Retrieval.ipynb\n```"
    )
    st.stop()

st.sidebar.header("⚙️ Settings")
pipeline_label = st.sidebar.selectbox("Retrieval pipeline", list(PIPELINES.keys()))
pipeline_key = PIPELINES[pipeline_label]
st.sidebar.markdown("---")
st.sidebar.metric("Chunks loaded", len(chunks))
st.sidebar.caption("88 prose · 10 table · 5 table_summary")

tab_query, tab_gold, tab_pipeline = st.tabs(
    ["🔎 Single Query", "📋 Gold Questions", "🧬 Full RAG Pipeline"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Single query (dense → rerank → structured answer)
# ---------------------------------------------------------------------------
with tab_query:
    st.caption(
        "Dense FAISS retrieval → MedCPT cross-encoder rerank → structured clinical response."
    )
    left, right = st.columns([3, 1])
    with left:
        query = st.text_input(
            "Ask a clinical question",
            placeholder="e.g. What is first-line therapy for penicillin-allergic patients?",
        )
    with right:
        top_k = st.number_input("Top K", min_value=1, max_value=20, value=5)

    has_api_key = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("GENERATION_API_KEY"))

    if st.button("Search", type="primary", key="single_search"):
        if not query.strip():
            st.warning("Type a question first.")
        else:
            clinical = load_clinical_retriever(base)
            with st.spinner("Retrieving and reranking..."):
                t0 = time.time()
                result = run_clinical_rag(
                    query,
                    chunks,
                    clinical,
                    k=int(top_k),
                    use_llm=has_api_key,
                    generate_fn=generate_answer if has_api_key else None,
                )
                elapsed = time.time() - t0
            st.success(f"Pipeline finished in {elapsed:.2f}s — **{clinical.name}**")
            render_structured_response(result)

            with st.expander("Retrieved chunks (debug)"):
                ranked = clinical.search_with_scores(query, k=int(top_k))
                for rank, (i, score) in enumerate(ranked, 1):
                    render_hit(rank, chunks[i], score=score)
                st.caption(f"Chunk schema preserved: {', '.join(CHUNK_SCHEMA_KEYS)}")

# ---------------------------------------------------------------------------
# TAB 2 — Gold questions, pick-your-own via checklist
# ---------------------------------------------------------------------------
with tab_gold:
    gold = load_gold()
    if not gold:
        st.info(f"No gold questions found at `{GOLD_QUESTIONS}`.")
    else:
        st.write(
            f"Pick which of the **{len(gold)} gold questions** to run. "
            "Each runs top-10 retrieval against the selected pipeline; "
            "green-edged cards mark chunks in the question's expected section(s). "
            "For the full retrieve → answer flow, use **Tab 3**."
        )

        select_all = st.checkbox("Select all", value=False)
        cols = st.columns(3)
        selected_ids = []
        for n, g in enumerate(gold):
            col = cols[n % 3]
            label = f"{g['id']} — {g['q'][:45]}{'...' if len(g['q']) > 45 else ''}"
            checked = col.checkbox(label, value=select_all, key=f"gold_{g['id']}")
            if checked:
                selected_ids.append(g["id"])

        st.markdown("---")
        run_gold = st.button(
            f"Run selected ({len(selected_ids)})", type="primary", key="run_gold"
        )

        if run_gold:
            if not selected_ids:
                st.warning("Select at least one question above.")
            else:
                retriever = get_retriever(pipeline_key, base)
                selected = [g for g in gold if g["id"] in selected_ids]
                progress = st.progress(0.0)
                for n, g in enumerate(selected, 1):
                    with st.expander(f"{g['id']} — {g['q']}", expanded=True):
                        st.caption(f"Expected section(s): {', '.join(g['expect_sections'])}")
                        ranked = search_with_optional_scores(retriever, g["q"], k=10)
                        for rank, (i, score) in enumerate(ranked, 1):
                            render_hit(
                                rank, chunks[i], expect_sections=g["expect_sections"], score=score
                            )
                    progress.progress(n / len(selected))

# ---------------------------------------------------------------------------
# TAB 3 — Full RAG pipeline (retrieve → generate → safety)
# ---------------------------------------------------------------------------
with tab_pipeline:
    st.markdown(
        "End-to-end flow: **dense FAISS → MedCPT rerank** → grounded LLM answer "
        "with verbatim excerpts and citations → relevance gating."
    )

    st.markdown("#### Pipeline status")
    steps = [
        ("Layer 1 — Ingestion", "PDF → chunks + metadata", True),
        ("Layer 1 — Embeddings", "BioBERT vectors + FAISS index", True),
        ("Layer 2 — Retrieval", "Dense FAISS + MedCPT rerank", True),
        ("Layer 3 — Generation", "Grounded LLM answer + citations", True),
        ("Layer 4 — Safety", "Relevance gate + citation grounding / refusal", True),
    ]
    for name, desc, done in steps:
        icon = "✅" if done else "⬜"
        st.markdown(f"{icon} **{name}** — {desc}")

    has_api_key = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("GENERATION_API_KEY"))
    if not has_api_key:
        st.info(
            "Set `OPENAI_API_KEY` (or `GENERATION_API_KEY`) in your environment to "
            "call the generation model. Retrieval and offline refusals still work without it."
        )

    st.markdown("---")
    pipe_left, pipe_right = st.columns([3, 1])
    with pipe_left:
        pipeline_query = st.text_input(
            "Ask a question end-to-end",
            placeholder="e.g. What is first-line therapy for penicillin-allergic patients?",
            key="pipeline_query",
        )
    with pipe_right:
        pipeline_k = st.number_input("Top K", min_value=1, max_value=10, value=5, key="pipeline_k")

    run_pipeline = st.button("Run full pipeline", type="primary", key="run_pipeline")

    if run_pipeline:
        if not pipeline_query.strip():
            st.warning("Type a question first.")
        else:
            clinical = load_clinical_retriever(base)
            with st.spinner("Retrieving and reranking..."):
                t0 = time.time()
                result = run_clinical_rag(
                    pipeline_query,
                    chunks,
                    clinical,
                    k=int(pipeline_k),
                    use_llm=has_api_key,
                    generate_fn=generate_answer if has_api_key else None,
                )
                retrieve_elapsed = time.time() - t0

            st.caption(
                f"Pipeline finished in {retrieve_elapsed:.2f}s — **{clinical.name}**"
            )
            render_structured_response(result)

            with st.expander("Retrieved evidence (ranked chunks)"):
                ranked = clinical.search_with_scores(pipeline_query, k=int(pipeline_k))
                for rank, (i, score) in enumerate(ranked, 1):
                    render_hit(rank, chunks[i], score=score)
