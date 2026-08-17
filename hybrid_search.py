"""
Hybrid retrieval with comparison mode.

Pipelines:
  1. minmax   — 70% semantic + 30% BM25, min-max normalized (previous)
  2. rrf      — weighted RRF (70/30) + section downrank
  3. rrf_rerank — RRF pool → MedCPT-Cross-Encoder rerank (shipping stack)

Run from repo root:
    python hybrid_search.py --compare          # metrics table: old vs new
    python hybrid_search.py --query "..."      # inspect with rrf_rerank
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from abc import ABC, abstractmethod
from collections import Counter

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from paths import CHUNKS_JSON, FAISS_INDEX, GOLD_QUESTIONS, HYBRID_REPORT, OUTPUTS

SEMANTIC_WEIGHT = 0.70
BM25_WEIGHT = 0.30
RRF_K = 60
INSPECT_KS = (1, 3, 5, 10)
EMBED_MODEL = "pritamdeka/S-BioBert-snli-multinli-stsb"
RERANK_MODEL = "ncbi/MedCPT-Cross-Encoder"

# Soft downrank via effective-rank inflation (divide rank by weight).
SECTION_WEIGHT = {
    "ABSTRACT": 0.45,
    "INTRODUCTION": 0.55,
    "METHODS": 0.70,
    "CONFLICTS OF INTEREST": 0.0,
    "REFERENCES": 0.0,
    "FRONT MATTER": 0.0,
}

POOL_PER_CHANNEL = 25
RERANK_POOL_MAX = 40

STOP = set(
    """a an and are as at be by for from has have how in is it its of on or that the
    this to was were what when which who why with should can do does may will not""".split()
)


def tok(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 1]


def chunk_passage(c: dict) -> str:
    parts = [c.get("section", ""), c.get("subsection", ""), c["text"]]
    return " ".join(p for p in parts if p).strip()


def section_multiplier(c: dict) -> float:
    w = SECTION_WEIGHT.get(c.get("section", "").upper(), 1.0)
    if w <= 0:
        return 0.0
    ct = c.get("content_type", "prose")
    if ct in ("table", "table_summary"):
        w *= 1.12
    sub = (c.get("subsection") or "").lower()
    if "recommendation" in sub or "table 2" in sub or "table 5" in sub or "table 6" in sub:
        w *= 1.10
    return w


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


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


class SemanticIndex:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        corpus = [chunk_passage(c) for c in chunks]
        self.bm25 = BM25(corpus)
        print(f"Loading FAISS: {FAISS_INDEX}")
        self.index = faiss.read_index(str(FAISS_INDEX))
        if self.index.ntotal != len(chunks):
            raise SystemExit(
                f"FAISS has {self.index.ntotal} vectors but chunks has {len(chunks)}"
            )
        print(f"Loading embedder: {EMBED_MODEL}")
        self.embedder = SentenceTransformer(EMBED_MODEL)

    def semantic_scores(self, query: str) -> np.ndarray:
        q = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q)
        scores, _ = self.index.search(q, self.index.ntotal)
        return scores[0].astype(np.float64)


class MedCPTReranker:
    def __init__(self):
        print(f"Loading reranker: {RERANK_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL)
        self.model.eval()

    def score_pairs(self, query: str, passages: list[str], batch_size: int = 8) -> np.ndarray:
        scores = []
        with torch.no_grad():
            for start in range(0, len(passages), batch_size):
                batch = passages[start:start + batch_size]
                pairs = [[query, p] for p in batch]
                encoded = self.tokenizer(
                    pairs,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=512,
                )
                logits = self.model(**encoded).logits.squeeze(-1)
                scores.append(logits.cpu().numpy())
        return np.concatenate(scores)


class MinMaxRetriever:
    name = "minmax (70/30, old)"

    def __init__(self, base: SemanticIndex):
        self.base = base
        self.chunks = base.chunks

    def search(self, query: str, k: int = 10) -> list[int]:
        sem = minmax_norm(self.base.semantic_scores(query))
        bm = minmax_norm(self.base.bm25.score_all(query))
        hybrid = SEMANTIC_WEIGHT * sem + BM25_WEIGHT * bm
        return list(np.argsort(-hybrid)[:k])


class BM25Retriever:
    name = "bm25 only"

    def __init__(self, base: SemanticIndex):
        self.base = base
        self.chunks = base.chunks

    def search(self, query: str, k: int = 10) -> list[int]:
        bm = self.base.bm25.score_all(query)
        return list(np.argsort(-bm)[:k])


class RRFRetriever:
    name = "rrf + section downrank"

    def __init__(self, base: SemanticIndex):
        self.base = base
        self.chunks = base.chunks

    def rrf_scores(self, query: str) -> np.ndarray:
        sem_r = ranks_from_scores(self.base.semantic_scores(query))
        bm_r = ranks_from_scores(self.base.bm25.score_all(query))
        n = len(self.chunks)
        rrf = np.zeros(n, dtype=np.float64)
        for i, c in enumerate(self.chunks):
            w = section_multiplier(c)
            if w <= 0:
                continue
            # Lower weight → inflated effective rank → lower RRF contribution
            eff_sem = sem_r[i] / w
            eff_bm = bm_r[i] / w
            rrf[i] = SEMANTIC_WEIGHT / (RRF_K + eff_sem) + BM25_WEIGHT / (RRF_K + eff_bm)
        return rrf

    def search(self, query: str, k: int = 10) -> list[int]:
        scores = self.rrf_scores(query)
        return list(np.argsort(-scores)[:k])

    def candidate_pool(self, query: str) -> list[int]:
        bm = list(np.argsort(-self.base.bm25.score_all(query))[:POOL_PER_CHANNEL])
        sem = list(np.argsort(-self.base.semantic_scores(query))[:POOL_PER_CHANNEL])
        rrf = self.search(query, k=POOL_PER_CHANNEL)
        seen, pool = set(), []
        for i in bm + sem + rrf:
            if i not in seen:
                seen.add(i)
                pool.append(i)
        return pool[:RERANK_POOL_MAX]


class RRFRerankRetriever:
    name = "rrf + downrank + MedCPT rerank"

    def __init__(self, base: SemanticIndex, reranker: MedCPTReranker):
        self.rrf = RRFRetriever(base)
        self.reranker = reranker
        self.chunks = base.chunks

    def search(self, query: str, k: int = 10) -> list[int]:
        pool_idx = self.rrf.candidate_pool(query)
        if not pool_idx:
            return []
        passages = [chunk_passage(self.chunks[i]) for i in pool_idx]
        scores = self.reranker.score_pairs(query, passages)
        # Section downrank on reranker scores (keeps evidence sections competitive)
        for j, i in enumerate(pool_idx):
            scores[j] *= section_multiplier(self.chunks[i])
        order = np.argsort(-scores)
        ranked = [pool_idx[i] for i in order]
        return ranked[:k]


def eval_retriever(retriever, gold: list[dict]) -> dict:
    n = len(gold)
    recall = {k: 0 for k in INSPECT_KS}
    rr = 0.0
    abstract_at1 = 0

    for g in gold:
        want = {s.upper() for s in g["expect_sections"]}
        top = retriever.search(g["q"], k=max(INSPECT_KS))
        sections = [retriever.chunks[i].get("section", "").upper() for i in top]
        if sections and sections[0] == "ABSTRACT":
            abstract_at1 += 1
        rank = next((r for r, s in enumerate(sections, 1) if s in want), None)
        rr += 1.0 / rank if rank else 0.0
        for k in INSPECT_KS:
            if rank and rank <= k:
                recall[k] += 1

    return {
        "recall": {k: recall[k] / n for k in INSPECT_KS},
        "mrr": rr / n,
        "abstract_top1_pct": 100 * abstract_at1 / n,
    }


def hit_row(c: dict, rank: int) -> dict:
    return {
        "rank": rank,
        "chunk_id": c["chunk_id"],
        "section": c.get("section", ""),
        "subsection": c.get("subsection", ""),
        "page": c.get("page_start") or c.get("page"),
    }


def format_hit_line(h: dict) -> str:
    line = f"  #{h['rank']:>2}  {h['chunk_id']}  p{h['page']}  {h['section'][:56]}"
    if h.get("subsection"):
        line += f"\n       / {h['subsection'][:70]}"
    return line


def compare_table(results: dict[str, dict]) -> str:
    lines = [
        "",
        "=" * 72,
        "RETRIEVAL COMPARISON (30 gold questions)",
        "=" * 72,
        f"{'Method':<32} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'ABST@1':>7}",
        "-" * 72,
    ]
    for name, m in results.items():
        r = m["recall"]
        lines.append(
            f"{name:<32} {r[1]:>5.1%} {r[3]:>5.1%} {r[5]:>5.1%} {r[10]:>5.1%} "
            f"{m['mrr']:>5.3f} {m['abstract_top1_pct']:>6.1f}%"
        )
    lines.append("")
    lines.append("ABST@1 = % of queries where ABSTRACT is rank #1 (lower is better for evidence)")
    return "\n".join(lines)


def inspect(retriever, query: str, label: str) -> str:
    blocks = [f"\n{'=' * 72}\nQUERY: {label}\n{'=' * 72}"]
    for k in (3, 5, 10):
        idx = retriever.search(query, k=k)
        blocks.append(f"\n--- TOP {k} ({retriever.name}) ---")
        for r, i in enumerate(idx, 1):
            blocks.append(format_hit_line(hit_row(retriever.chunks[i], r)))
    return "\n".join(blocks)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Run all pipelines and print table")
    parser.add_argument("--query", action="append")
    parser.add_argument("--gold-only", action="store_true")
    args = parser.parse_args()

    if not CHUNKS_JSON.exists() or not FAISS_INDEX.exists():
        sys.exit("Missing chunks or FAISS index — run ingestion + indexing first.")

    chunks = json.load(open(CHUNKS_JSON, encoding="utf-8"))
    gold = json.load(open(GOLD_QUESTIONS, encoding="utf-8"))["questions"]
    base = SemanticIndex(chunks)

    minmax = MinMaxRetriever(base)
    bm25 = BM25Retriever(base)
    rrf = RRFRetriever(base)
    reranker = MedCPTReranker()
    rrf_rerank = RRFRerankRetriever(base, reranker)

    out: list[str] = []

    if args.compare or not args.query:
        print("Evaluating pipelines…")
        results = {
            bm25.name: eval_retriever(bm25, gold),
            minmax.name: eval_retriever(minmax, gold),
            rrf.name: eval_retriever(rrf, gold),
            rrf_rerank.name: eval_retriever(rrf_rerank, gold),
        }
        table = compare_table(results)
        out.append(table)
        print(table)

        # Side-by-side on a few high-value queries
        samples = [
            ("Q01", gold[0]["q"]),
            ("Q02", gold[1]["q"]),
            ("Q11", gold[10]["q"]),
            ("Q20", gold[19]["q"]),
        ]
        out.append("\n--- SAMPLE QUERY COMPARISON (top 3) ---")
        for label, q in samples:
            out.append(f"\n{label}: {q[:70]}")
            for ret in (minmax, rrf_rerank):
                top3 = ret.search(q, k=3)
                out.append(f"  [{ret.name}]")
                for r, i in enumerate(top3, 1):
                    c = ret.chunks[i]
                    out.append(f"    #{r} {c['chunk_id']} {c['section'][:50]}")

    shipping = rrf_rerank
    if args.query:
        out.append("\n--- AD-HOC QUERIES ---")
        for q in args.query:
            out.append(inspect(shipping, q, q))

    if not args.compare and not args.query and not args.gold_only:
        out.append("\n--- GOLD SET (rrf + MedCPT) — TOP 3/5/10 ---")
        for g in gold:
            out.append(inspect(shipping, g["q"], f"{g['id']} — {g['q']}"))

    text = "\n".join(out).strip()
    if text:
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        HYBRID_REPORT.write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved: {HYBRID_REPORT}")


if __name__ == "__main__":
    main()
