"""
STEP 3 — Retrieval evaluation against the gold question set.

Why this exists
---------------
Every chunking parameter so far (400 tokens, 60 overlap, section boundaries) is
justified by engineering reasoning, not by measurement on THIS document. That
is a real gap: without a score, the next person to change CHUNK_TOKENS has no
way to tell whether they helped or hurt.

This script gives a number today, with no embedding model and no vector
database, using BM25 — the standard lexical baseline. It is deliberately NOT
the same retriever the system will ship with. Its job is regression detection:
if a chunking change drops this score, the change made retrieval worse.

Metrics
-------
  section recall@k : did any of the top-k chunks come from a section that
                     actually contains the answer?
  MRR              : 1 / rank of the first correct chunk, averaged.

Run:  python 3_eval.py
"""
import json
import math
import os
import re
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CHUNKS = "data/processed/acg_chunks.json"
GOLD = "gold_questions.json"
REPORT = "data/processed/eval_report.txt"
KS = (1, 3, 5, 10)

STOP = set("""a an and are as at be by for from has have how in is it its of on or that the
this to was were what when which who why with should can do does may will not""".split())


def tok(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower())
            if w not in STOP and len(w) > 1]


class BM25:
    """Plain BM25 Okapi. ~40 lines, stdlib only, no model download."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [tok(d) for d in docs]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(self.N, 1)
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for d in self.docs:
            for w in set(d):
                df[w] += 1
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
                    for w, n in df.items()}

    def search(self, query, k=10):
        q = tok(query)
        scores = []
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = 0.0
            for w in q:
                f = tf.get(w)
                if not f:
                    continue
                s += (self.idf.get(w, 0.0) * f * (self.k1 + 1)
                      / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)))
            scores.append((s, i))
        scores.sort(reverse=True)
        return [i for s, i in scores[:k] if s > 0]


def main():
    if not os.path.exists(CHUNKS):
        sys.exit(f"{CHUNKS} not found — run 2_parse_chunk.py first.")
    chunks = json.load(open(CHUNKS, encoding="utf-8"))
    gold = json.load(open(GOLD, encoding="utf-8"))["questions"]

    # index the same text a retriever would see: section context + body
    corpus = [f"{c.get('section','')} {c.get('subsection','')} {c['text']}"
              for c in chunks]
    bm = BM25(corpus)

    maxk = max(KS)
    hits = {k: 0 for k in KS}
    rr_total = 0.0
    lines = []

    for g in gold:
        want = {s.upper() for s in g["expect_sections"]}
        top = bm.search(g["q"], k=maxk)
        got_sections = [chunks[i].get("section", "").upper() for i in top]

        rank = next((r for r, s in enumerate(got_sections, 1) if s in want), None)
        rr_total += 1.0 / rank if rank else 0.0
        for k in KS:
            if rank and rank <= k:
                hits[k] += 1

        status = "PASS" if rank and rank <= 5 else "FAIL"
        first = chunks[top[0]] if top else None
        lines.append(
            f"[{status}] {g['id']} rank={rank if rank else '-'}  {g['q'][:78]}\n"
            f"         top1: {(first.get('section','') if first else 'nothing retrieved')[:60]}"
            f" | p{first.get('page_start') if first else '-'}"
            f" | {first.get('chunk_id') if first else '-'}"
        )

    n = len(gold)
    out = []
    out.append("=" * 66)
    out.append("RETRIEVAL EVAL — BM25 lexical baseline (regression detector)")
    out.append("=" * 66)
    out.append(f"chunks indexed : {len(chunks)}")
    out.append(f"questions      : {n}")
    out.append("")
    for k in KS:
        out.append(f"  section recall@{k:<3}: {hits[k]}/{n}  ({100*hits[k]/n:.1f}%)")
    out.append(f"  MRR             : {rr_total/n:.3f}")
    out.append("")
    out.append("--- PER QUESTION ---")
    out.extend(lines)
    out.append("")
    out.append("Reading this:")
    out.append("  recall@5 below ~0.75 means chunks are landing in the wrong")
    out.append("  section or the text is too mangled to match. Look at the FAILs")
    out.append("  and open those chunks in acg_chunks.json.")
    out.append("")
    out.append("  This is BM25, not the shipping retriever. Treat it as a")
    out.append("  before/after number when changing chunk size, overlap or")
    out.append("  cleaning rules — not as a quality guarantee.")

    text = "\n".join(out)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w", encoding="utf-8").write(text)
    print(text)
    print(f"\nSaved: {REPORT}")


if __name__ == "__main__":
    main()
