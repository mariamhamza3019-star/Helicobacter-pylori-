"""
Streamlit tester for the ACG H. pylori RAG retrieval stack.

Drop this file in the repo ROOT (same folder as paths.py / hybrid_search.py)
and run:

    streamlit run streamlit_app.py

Requires the ingestion + embedding steps to already be done, i.e.
outputs/h_pylori_faiss.index and ingestion/data/processed/acg_chunks.json
must exist (run `python run_pipeline.py --embed` first if not).

This repo currently implements Layers 1-2 (ingestion + retrieval) only.
There's no generation/safety layer yet — Tab 3 is a placeholder that will
come alive once that's built.
"""

import json
import os
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import CHUNKS_JSON, FAISS_INDEX, GOLD_QUESTIONS  # noqa: E402
from hybrid_search import (  # noqa: E402
    SemanticIndex,
    MinMaxRetriever,
    BM25Retriever,
    RRFRetriever,
    RRFRerankRetriever,
    MedCPTReranker,
    eval_retriever,
)
from generation import generate_answer  # noqa: E402

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


def render_hit(rank: int, c: dict, expect_sections=None):
    page = c.get("page_start") or c.get("page")
    is_expected = expect_sections and c.get("section", "").upper() in {
        s.upper() for s in expect_sections
    }
    card_class = "hit-card expect-hit" if is_expected else "hit-card"
    sub = f" / {c['subsection']}" if c.get("subsection") else ""
    st.markdown(
        f"""
        <div class="{card_class}">
            <span class="hit-rank">#{rank}</span>
            <span class="section-tag">{c.get('section','')}{sub}</span>
            <div class="hit-meta">{c.get('chunk_id','')} · p.{page} · {c.get('content_type','prose')}
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
# TAB 1 — Single query
# ---------------------------------------------------------------------------
with tab_query:
    left, right = st.columns([3, 1])
    with left:
        query = st.text_input(
            "Ask a clinical question",
            placeholder="e.g. What is first-line therapy for penicillin-allergic patients?",
        )
    with right:
        top_k = st.number_input("Top K", min_value=1, max_value=20, value=5)

    if st.button("Search", type="primary", key="single_search"):
        if not query.strip():
            st.warning("Type a question first.")
        else:
            with st.spinner(f"Retrieving with: {pipeline_label}..."):
                retriever = get_retriever(pipeline_key, base)
                t0 = time.time()
                idx = retriever.search(query, k=top_k)
                elapsed = time.time() - t0
            st.success(f"{len(idx)} results in {elapsed:.2f}s — **{retriever.name}**")
            for rank, i in enumerate(idx, 1):
                render_hit(rank, chunks[i])

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
            "green-edged cards mark chunks in the question's expected section(s) "
            "(there's no generated answer yet — that's Tab 3, once generation is built)."
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
                        idx = retriever.search(g["q"], k=10)
                        for rank, i in enumerate(idx, 1):
                            render_hit(rank, chunks[i], expect_sections=g["expect_sections"])
                    progress.progress(n / len(selected))

# ---------------------------------------------------------------------------
# TAB 3 — Full RAG pipeline (generation + grounding)
# ---------------------------------------------------------------------------
with tab_pipeline:
    st.markdown("### 🧬 Full RAG Pipeline")

    # Check API key
    api_key_set = os.getenv("GROQ_API_KEY") is not None
    if not api_key_set:
        st.warning(
            "⚠️ **Set GROQ_API_KEY first**\n\n"
            "PowerShell: `$env:GROQ_API_KEY = 'gsk-...'`\n\n"
            "[Get key](https://console.groq.com/keys)"
        )

    # Query input
    question = st.text_input(
        "Ask a question",
        placeholder="e.g. What is first-line therapy?",
        key="rag_question",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        top_k = st.number_input("Top K", min_value=1, max_value=10, value=5, key="rag_k")
    with col2:
        run_btn = st.button("Run", type="primary", key="rag_run")

    if run_btn:
        if not question.strip():
            st.warning("Enter a question")
        elif not api_key_set:
            st.error("Set GROQ_API_KEY first")
        else:
            # Retrieve
            retriever = get_retriever(pipeline_key, base)
            idx = retriever.search(question, k=top_k)
            retrieved_chunks = [chunks[i] for i in idx]

            # Generate
            result = generate_answer(question, retrieved_chunks)

            # Display
            st.markdown("---")
            if result.get("refused"):
                st.warning(f"Refused: {result.get('refusal_reason')}")
            else:
                st.success("✅ Answer")
                st.markdown(f"**{result.get('recommendation')}**")
                if result.get("citations"):
                    st.markdown("**Sources:**")
                    for cite in result["citations"]:
                        st.text(f"{cite['chunk_id']} - {cite['section']} (p{cite['page']})")
            
            # Show errors
            if result.get("validation_errors"):
                st.error(f"Issues: {', '.join(result['validation_errors'])}")
            
            # Show chunks
            with st.expander("Retrieved evidence"):
                for i, chunk in enumerate(retrieved_chunks, 1):
                    render_hit(i, chunk)

    # Status
    st.markdown("---")
    st.markdown("**Pipeline Status:**")
    steps = [
        ("Layer 1 — Ingestion", True),
        ("Layer 2 — Retrieval", True),
        ("Layer 3 — Generation", True),
        ("Layer 4 — Safety", True),
    ]
    for name, done in steps:
        icon = "✅" if done else "⬜"
        st.markdown(f"{icon} {name}")