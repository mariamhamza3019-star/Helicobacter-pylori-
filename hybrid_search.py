"""
Hybrid retrieval: 70% semantic (FAISS cosine) + 30% BM25.

Scores are min-max normalized per query, then fused:
    hybrid = 0.7 * semantic_norm + 0.3 * bm25_norm

Run from repo root:
    python hybrid_search.py
    python hybrid_search.py --query "What is first-line therapy for H. pylori?"
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from paths import CHUNKS_JSON, FAISS_INDEX, GOLD_QUESTIONS, OUTPUTS

SEMANTIC_WEIGHT = 0.70
BM25_WEIGHT = 0.30
INSPECT_KS = (3, 5, 10)
MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"
REPORT_PATH = OUTPUTS / "hybrid_search_report.txt"

STOP = set(
    """a an and are as at be by for from has have how in is it its of on or that the
    this to was were what when which who why with should can do does may will not""".split()
)


def tok(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 1]


def minmax_norm(scores: np.ndarray) -> np.ndarray:
    lo, hi = float(scores.min()), float(scores.max())
    if hi == lo:
        return np.ones_like(scores) if hi > 0 else np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tok(d) for d in docs]
        self.n = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(self.n, 1)
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for d in self.docs:
            for w in set(d):
                df[w] += 1
        self.idf = {
            w: math.log(1 + (self.n - n + 0.5) / (n + 0.5)) for w, n in df.items()
        }

    def score_all(self, query: str) -> np.ndarray:
        q = tok(query)
        out = np.zeros(self.n, dtype=np.float64)
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = 0.0
            for w in q:
                f = tf.get(w)
                if not f:
                    continue
                s += (
                    self.idf.get(w, 0.0)
                    * f
                    * (self.k1 + 1)
                    / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                )
            out[i] = s
        return out


class HybridRetriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        corpus = [
            f"{c.get('section', '')} {c.get('subsection', '')} {c['text']}" for c in chunks
        ]
        self.bm25 = BM25(corpus)

        print(f"Loading FAISS index: {FAISS_INDEX}")
        self.index = faiss.read_index(str(FAISS_INDEX))
        if self.index.ntotal != len(chunks):
            raise SystemExit(
                f"FAISS has {self.index.ntotal} vectors but chunks has {len(chunks)}"
            )

        print(f"Loading embedding model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)

    def semantic_scores(self, query: str) -> np.ndarray:
        q = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q)
        scores, _ = self.index.search(q, self.index.ntotal)
        return scores[0].astype(np.float64)

    def search(self, query: str, k: int = 10) -> list[dict]:
        sem = minmax_norm(self.semantic_scores(query))
        bm = minmax_norm(self.bm25.score_all(query))
        hybrid = SEMANTIC_WEIGHT * sem + BM25_WEIGHT * bm
        order = np.argsort(-hybrid)[:k]
        return [
            {
                "rank": r + 1,
                "chunk_id": self.chunks[i]["chunk_id"],
                "section": self.chunks[i].get("section", ""),
                "subsection": self.chunks[i].get("subsection", ""),
                "page": self.chunks[i].get("page_start") or self.chunks[i].get("page"),
                "hybrid_score": round(float(hybrid[i]), 4),
                "semantic_score": round(float(sem[i]), 4),
                "bm25_score": round(float(bm[i]), 4),
                "text_preview": self.chunks[i]["text"][:140] + "…",
            }
            for r, i in enumerate(order)
        ]


def format_hits(hits: list[dict]) -> str:
    lines = []
    for h in hits:
        lines.append(
            f"  #{h['rank']:>2}  {h['chunk_id']}  hybrid={h['hybrid_score']:.3f}  "
            f"sem={h['semantic_score']:.3f}  bm25={h['bm25_score']:.3f}  "
            f"p{h['page']}  {h['section'][:52]}"
        )
        if h.get("subsection"):
            lines.append(f"       / {h['subsection'][:70]}")
    return "\n".join(lines)


def inspect_query(retriever: HybridRetriever, query: str, label: str = "") -> str:
    blocks = []
    header = label or query
    blocks.append(f"\n{'=' * 72}\nQUERY: {header}\n{'=' * 72}")
    for k in INSPECT_KS:
        hits = retriever.search(query, k=k)
        blocks.append(f"\n--- TOP {k} (70% semantic + 30% BM25) ---")
        blocks.append(format_hits(hits))
    return "\n".join(blocks)


def eval_recall(retriever: HybridRetriever, gold: list[dict], k: int) -> tuple[int, int]:
    hits = 0
    for g in gold:
        want = {s.upper() for s in g["expect_sections"]}
        top = retriever.search(g["q"], k=k)
        if any(h["section"].upper() in want for h in top):
            hits += 1
    return hits, len(gold)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Hybrid semantic + BM25 search")
    parser.add_argument("--query", action="append", help="Ad-hoc query (repeatable)")
    parser.add_argument("--gold-only", action="store_true", help="Skip sample queries")
    args = parser.parse_args()

    if not CHUNKS_JSON.exists():
        sys.exit(f"{CHUNKS_JSON} not found — run ingestion first.")
    if not FAISS_INDEX.exists():
        sys.exit(f"{FAISS_INDEX} not found — build FAISS index first.")

    chunks = json.load(open(CHUNKS_JSON, encoding="utf-8"))
    retriever = HybridRetriever(chunks)

    out: list[str] = []
    out.append("HYBRID SEARCH — 70% semantic (FAISS/BioBERT) + 30% BM25")
    out.append(f"Corpus: {len(chunks)} chunks | inspect k = {INSPECT_KS}")
    out.append("Per-query scores are min-max normalized before fusion.\n")

    gold = json.load(open(GOLD_QUESTIONS, encoding="utf-8"))["questions"]
    out.append("--- SECTION RECALL (gold set) ---")
    for k in INSPECT_KS:
        hits, n = eval_recall(retriever, gold, k)
        out.append(f"  recall@{k:<3}: {hits}/{n}  ({100 * hits / n:.1f}%)")
    out.append("")

    # Full inspection for every gold question at k=3,5,10
    out.append("--- GOLD QUESTIONS — TOP 3 / 5 / 10 ---")
    for g in gold:
        out.append(inspect_query(retriever, g["q"], label=f"{g['id']} — {g['q']}"))

    if args.query:
        out.append("\n--- AD-HOC QUERIES ---")
        for q in args.query:
            out.append(inspect_query(retriever, q))
    elif not args.gold_only:
        samples = [
            "What is the preferred first-line regimen for treatment-naive patients?",
            "How long after therapy should test of cure be performed?",
            "Which patients should be tested for H. pylori?",
        ]
        out.append("\n--- SAMPLE QUERIES ---")
        for q in samples:
            out.append(inspect_query(retriever, q))

    text = "\n".join(out)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
