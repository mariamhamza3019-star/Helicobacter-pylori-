"""
Compare dense-only vs dense+MedCPT rerank on gold clinical queries.

Prints a before/after metrics table and per-query ordering changes.
Run: python test_rerank_comparison.py
     python -m unittest test_rerank_comparison.py -v
"""
from __future__ import annotations

import json
import sys
import unittest

from paths import CHUNKS_JSON, FAISS_INDEX, GOLD_QUESTIONS
from hybrid_search import INSPECT_KS, SemanticIndex, MedCPTReranker
from rag_pipeline import DenseRetriever, DenseRerankRetriever, ordering_changed


def eval_section_recall(
    retriever: DenseRetriever | DenseRerankRetriever,
    gold: list[dict],
    *,
    dense_baseline: DenseRetriever | None = None,
) -> dict:
    """Recall@k and MRR using expected guideline sections."""
    n = len(gold)
    recall = {k: 0 for k in INSPECT_KS}
    rr = 0.0
    reorder_count = 0

    for g in gold:
        want = {s.upper() for s in g["expect_sections"]}
        top = retriever.search(g["q"], k=max(INSPECT_KS))
        sections = [retriever.chunks[i].get("section", "").upper() for i in top]
        rank = next((r for r, s in enumerate(sections, 1) if s in want), None)
        rr += 1.0 / rank if rank else 0.0
        for k in INSPECT_KS:
            if rank and rank <= k:
                recall[k] += 1

        if dense_baseline is not None:
            dense_top = dense_baseline.search(g["q"], k=len(top))
            if ordering_changed(dense_top, top):
                reorder_count += 1

    out: dict = {
        "recall": {k: recall[k] / n for k in INSPECT_KS},
        "mrr": rr / n,
    }
    if dense_baseline is not None:
        out["ordering_changed_pct"] = 100.0 * reorder_count / n
    return out


def _load_stack():
    if not CHUNKS_JSON.exists() or not FAISS_INDEX.exists():
        raise SystemExit("Missing chunks or FAISS index — run ingestion + indexing first.")
    with open(CHUNKS_JSON, encoding="utf-8") as f:
        chunks = json.load(f)
    with open(GOLD_QUESTIONS, encoding="utf-8") as f:
        gold = json.load(f)["questions"]
    base = SemanticIndex(chunks)
    dense = DenseRetriever(base)
    rerank = DenseRerankRetriever(base, MedCPTReranker())
    return dense, rerank, gold


def print_comparison(dense_m: dict, rerank_m: dict, n_questions: int) -> str:
    lines = [
        "",
        "=" * 72,
        f"DENSE vs RERANK — {n_questions} gold clinical queries",
        "=" * 72,
        f"{'Pipeline':<28} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6}",
        "-" * 72,
    ]
    for label, m in [("dense (FAISS + BioBERT)", dense_m), ("dense + MedCPT rerank", rerank_m)]:
        r = m["recall"]
        lines.append(
            f"{label:<28} "
            f"{r[1]:>5.1%} {r[3]:>5.1%} {r[5]:>5.1%} {r[10]:>5.1%} "
            f"{m['mrr']:>6.3f}"
        )
    lines.append("")
    delta_r5 = rerank_m["recall"][5] - dense_m["recall"][5]
    delta_mrr = rerank_m["mrr"] - dense_m["mrr"]
    lines.append(f"Delta Recall@5: {delta_r5:+.1%}   Delta MRR: {delta_mrr:+.3f}")
    if "ordering_changed_pct" in rerank_m:
        lines.append(
            f"Reranker reordered top-{max(INSPECT_KS)} vs dense on "
            f"{rerank_m['ordering_changed_pct']:.0f}% of queries "
            "(proves scores affect ordering, not just relabeling)."
        )
    return "\n".join(lines)


class TestRerankComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dense, cls.rerank, cls.gold = _load_stack()

    def test_rerank_improves_or_matches_dense_recall_at_5(self):
        dense_m = eval_section_recall(self.dense, self.gold)
        rerank_m = eval_section_recall(self.rerank, self.gold, dense_baseline=self.dense)
        report = print_comparison(dense_m, rerank_m, len(self.gold))
        print(report)
        self.assertGreaterEqual(
            rerank_m["recall"][5],
            dense_m["recall"][5],
            "Rerank Recall@5 should be >= dense baseline",
        )

    def test_rerank_changes_ordering_on_some_queries(self):
        rerank_m = eval_section_recall(self.rerank, self.gold, dense_baseline=self.dense)
        self.assertGreater(
            rerank_m.get("ordering_changed_pct", 0),
            0,
            "Reranker should change dense ordering on at least one query",
        )


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    dense, rerank, gold = _load_stack()
    dense_m = eval_section_recall(dense, gold)
    rerank_m = eval_section_recall(rerank, gold, dense_baseline=dense)
    print(print_comparison(dense_m, rerank_m, len(gold)))


if __name__ == "__main__":
    main()
