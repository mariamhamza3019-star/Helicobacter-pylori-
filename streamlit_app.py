"""
Streamlit tester for the ACG H. pylori RAG retrieval stack.

Drop this file in the repo ROOT (same folder as paths.py / hybrid_search.py)
and run:

    streamlit run streamlit_app.py

Requires the ingestion + embedding steps to already be done, i.e.
outputs/h_pylori_faiss.index and ingestion/data/processed/acg_chunks.json
must exist (run `python run_pipeline.py --embed` first if not).

Note: this repo currently only implements Layers 1-2 (ingestion + retrieval).
There's no generation/LLM layer yet, so this app tests RETRIEVAL QUALITY —
i.e. "given a question, are the right guideline chunks coming back" — not
a chat-with-your-docs experience. Once you add a generation step you can
wire its output in below the results table.
"""

import json
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
    chunk_passage,
)

st.set_page_config(page_title="H. pylori RAG — Retrieval Tester", layout="wide")

PIPELINES = {
    "RRF + MedCPT rerank (shipping stack)": "rrf_rerank",
    "RRF + section downrank": "rrf",
    "Hybrid minmax (70/30, old)": "minmax",
    "BM25 only": "bm25",
}


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
        reranker = load_reranker()
        return RRFRerankRetriever(base, reranker)
    raise ValueError(key)


def load_gold():
    if not GOLD_QUESTIONS.exists():
        return []
    return json.load(open(GOLD_QUESTIONS, encoding="utf-8"))["questions"]


st.title("🔬 H. pylori RAG — Retrieval Tester")
st.caption(
    "Query the ACG 2024 H. pylori guideline index directly and inspect what "
    "each retrieval pipeline pulls back."
)

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

st.sidebar.header("Settings")
pipeline_label = st.sidebar.selectbox("Retrieval pipeline", list(PIPELINES.keys()))
pipeline_key = PIPELINES[pipeline_label]
top_k = st.sidebar.slider("Top K results", min_value=1, max_value=20, value=5)
st.sidebar.markdown("---")
st.sidebar.caption(f"Corpus: **{len(chunks)}** chunks loaded")

tab_query, tab_eval = st.tabs(["🔎 Query", "📊 Gold-set evaluation"])

with tab_query:
    query = st.text_input(
        "Ask a clinical question about H. pylori treatment",
        placeholder="e.g. What is first-line therapy for penicillin-allergic patients?",
    )
    run = st.button("Search", type="primary")

    if run and query.strip():
        with st.spinner(f"Retrieving with: {pipeline_label}..."):
            retriever = get_retriever(pipeline_key, base)
            t0 = time.time()
            idx = retriever.search(query, k=top_k)
            elapsed = time.time() - t0

        st.success(f"Retrieved {len(idx)} chunks in {elapsed:.2f}s using **{retriever.name}**")

        for rank, i in enumerate(idx, 1):
            c = chunks[i]
            page = c.get("page_start") or c.get("page")
            header = f"#{rank} · {c.get('section', '')}"
            if c.get("subsection"):
                header += f" / {c['subsection']}"
            header += f" · p.{page} · {c.get('chunk_id', '')}"

            with st.expander(header, expanded=(rank <= 3)):
                st.write(c["text"])
                st.caption(f"content_type: {c.get('content_type', 'prose')}")
    elif run:
        st.warning("Type a question first.")

with tab_eval:
    gold = load_gold()
    if not gold:
        st.info(f"No gold questions found at `{GOLD_QUESTIONS}` — skipping eval.")
    else:
        st.write(f"Runs all pipelines against **{len(gold)} gold questions** and reports Recall@k / MRR.")
        if st.button("Run comparison"):
            results = {}
            progress = st.progress(0.0)
            keys = ["bm25", "minmax", "rrf", "rrf_rerank"]
            for n, key in enumerate(keys, 1):
                retriever = get_retriever(key, base)
                results[retriever.name] = eval_retriever(retriever, gold)
                progress.progress(n / len(keys))

            rows = []
            for name, m in results.items():
                r = m["recall"]
                rows.append(
                    {
                        "Method": name,
                        "R@1": f"{r[1]:.1%}",
                        "R@3": f"{r[3]:.1%}",
                        "R@5": f"{r[5]:.1%}",
                        "R@10": f"{r[10]:.1%}",
                        "MRR": f"{m['mrr']:.3f}",
                        "ABSTRACT@1 %": f"{m['abstract_top1_pct']:.1f}%",
                    }
                )
            st.table(rows)
            st.caption("ABSTRACT@1 = % of queries where ABSTRACT ranks #1 (lower is better — you want evidence sections, not the abstract).")
