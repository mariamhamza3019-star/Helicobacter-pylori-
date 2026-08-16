# ACG H. pylori — Person 2: Parsing + Cleaning + Chunking

## How to run

1. Put the ACG guideline PDF in `data/raw/` (any filename — it's auto-detected).
2. Double-click **`RUN_ME.bat`**.
3. It installs deps, runs both steps, and opens the log in Notepad.

That's it. No typing.

## Files

| File | Role |
|---|---|
| `1_inspect.py` | Diagnostic. Font-size histogram + every heading candidate. Run first. |
| `2_parse_chunk.py` | Parse → clean → detect sections → section-aware chunk. |
| `RUN_ME.bat` | Runs both, captures the log. |

## Output

| Path | Contents |
|---|---|
| `data/processed/FULL_OUTPUT.txt` | Full console log — paste this back for tuning |
| `data/processed/acg_sections.json` | Every detected section + page range + preview. **Verify this before embedding.** |
| `data/processed/acg_chunks.json` | The chunks, ready for embedding |

## Retrieval evaluation (measured, not asserted)

`3_eval.py` scores 30 clinical questions in `gold_questions.json` against the
produced chunks using BM25 — stdlib only, no model, no vector database.

```
section recall@1  : 23/30   76.7%
section recall@3  : 30/30  100.0%
section recall@5  : 30/30  100.0%
MRR               : 0.883
```

Measured progression, which is the point of having the harness at all:

| change | recall@1 | recall@5 | MRR |
|---|---:|---:|---:|
| prose only | 73.3% | 100.0% | 0.846 |
| + tables filed under a flat `TABLES` section | 70.0% | 96.7% | 0.829 |
| + tables filed under their real section | **76.7%** | **100.0%** | **0.883** |

The middle row is worth keeping. Extracting the regimen tables made the output
strictly better, but bucketing them under `TABLES` made the eval score them as
misses — the retriever was returning the dosing table for a dosing question and
being marked wrong. Without the harness that regression would have shipped
invisibly, because the tables themselves looked perfect.

Zero failures at k=5. Since a RAG pipeline normally feeds the top 5 to the LLM,
the answer is present in every case.

**What this number is for.** It is a regression detector, not a quality
guarantee. BM25 is not the retriever this system will ship with. Its value is
that changing `CHUNK_TOKENS`, `OVERLAP_TOKENS` or a cleaning rule and watching
this score move tells you whether the change helped. Before this existed, every
such change was a guess.

**Caveat worth knowing:** the `expect_sections` in `gold_questions.json` were
written from the guideline's structure and abstract, not from a page-by-page
read. If a question fails because the expectation is wrong, fix the JSON, not
the code.

## Measured results (verified run, 24-page ACG 2024 PDF)

```
Pages            : 24
Body font size   : 9.5
Section headings : 10.0 ALL CAPS  ->  ratio 1.053
Article title    : 20.9
Sections kept    : 34  (3 dropped: front matter, conflicts of interest, references)
Chunks           : 102
Chunk tokens     : min 59 | median 361 | max 400
Token counting   : exact, bge-small-en-v1.5
```

Top-level sections found — all 11, none missing, none spurious:

| Chunks | Section |
|---:|---|
| 3 | ABSTRACT |
| 3 | INTRODUCTION |
| 4 | METHODS |
| 7 | EPIDEMIOLOGY |
| 15 | INDICATIONS FOR H. PYLORI TESTING AND TREATMENT |
| 26 | ERADICATING H. PYLORI IN TREATMENT-NAIVE PATIENTS |
| 3 | POST-TREATMENT TESTING FOR CURE |
| 31 | ERADICATING H. PYLORI IN TREATMENT-EXPERIENCED PATIENTS |
| 6 | ANTIBIOTIC SUSCEPTIBILITY TESTING |
| 2 | PROBIOTICS AND H. PYLORI THERAPY |
| 2 | FUTURE RESEARCH PRIORITIES |

Subsections captured include `Recommendation`, `Key concept`,
`Summary of recommendations`, `Dosing frequency`, `Quality of evidence Criteria` —
the run-in bold headings that carry the actual clinical statements.

Dropped: `FRONT MATTER` (title + author affiliations), `CONFLICTS OF INTEREST`,
`REFERENCES` (27,533 characters that would otherwise have polluted retrieval).

## Chunk size and overlap — the answer for your task

| Parameter | Value | Why |
|---|---|---|
| Chunk size | **400 tokens** | BGE / E5 / OpenAI embedding models truncate at **512 tokens, silently**. No error is raised. 800 words ≈ 1,200 tokens, so two-thirds of every chunk would never reach the vector while the metadata still claimed it did. |
| Overlap | **60 tokens (15%)** | Recovers context lost at the cut point. Higher inflates the index and returns near-duplicate results. |
| Boundary | **Section = hard wall** | A chunk never spans two sections, and overlap never leaks across one. Mixing "first-line therapy" with "salvage therapy" in one vector is how a retriever returns the wrong regimen. |
| Counting | **Real tokenizer** | "clarithromycin", "levofloxacin", "esomeprazole" are 4–6 tokens each. Word-count chunking underestimates clinical text by roughly 45%. |
| Floor | 40 tokens | Fragments below this carry no retrievable meaning. |

## Why section detection kept failing before

Two bugs, both now fixed:

1. **Per-page detection with `break`** — the loop found the first heading on a page and stopped. Any page containing the end of one section and the start of another lost the second heading entirely. This is what reduced 9 sections to 5.
2. **Headings wrap across two lines** in a two-column journal, so exact string comparison against a hardcoded list could never match the long ones.

**The fix isn't a better list — it's no list at all.** Headings are now detected from the document's own typography (font size relative to body text, weight, capitalisation), then consecutive heading lines are merged. Nothing is hardcoded, so it also works on a second guideline.

## Also fixed

- Two-column reading order (`sort=True`) — default order can interleave columns into sentences that never existed.
- Gap-aware span joining — fixes `ofGastroenterology`, `bismuthquadruple` (superscript affiliation markers split spans and glued words together).
- Front matter removed structurally — authors, affiliations, download stamps no longer leak into chunk 1.
- `page_start` / `page_end` as separate ints — Chroma, Qdrant and pgvector all reject list-valued metadata, so the old int-or-list `page` field would have crashed at indexing time.
- Hyphen-aware line joining — `treatment-naive` no longer becomes `treatmentnaive`.

## Tuning

If `1_inspect.py` shows the section headings at a different size ratio than assumed, adjust two constants at the top of `2_parse_chunk.py`:

```python
H1_RATIO = 1.30   # size >= body * this -> section
H2_RATIO = 1.12   # size >= body * this -> subsection
```

## Tables

Extracted as tables, not flattened into prose. Rows are never split, the header
row is repeated in every part, and a table is never truncated to fit the token
budget — it warns instead, because cutting a dosing row is the exact harm this
path exists to prevent.

| Table | Page | Rows | Filed under |
|---|---:|---:|---|
| Table 1. GRADE criteria | 3 | 8 | EPIDEMIOLOGY *(should be METHODS)* |
| Table 5. Recommended regimens, treatment-naive | 8 | 31 | ERADICATING ... TREATMENT-NAIVE |
| Table 6. Recommended salvage regimens | 14 | 35 | ERADICATING ... TREATMENT-EXPERIENCED |

Detection runs behind a hard quality gate: >=3 rows, >=2 columns after phantom
columns are collapsed, >=25% of rows substantive, first row not page furniture,
area between 2% and 90% of the page. Anything that fails is left in the prose
and nothing is removed. That asymmetry is deliberate — a missed table costs a
little, a false table deletes real text. On the first attempt a false table on
page 1 (the running header) swallowed the ABSTRACT heading and the entire
abstract vanished into dropped front matter.

**Still missing: Tables 2, 3 and 4** (pages 4-6: guideline recommendations,
summary of key concepts, indications for testing). `find_tables()` does not
detect them at all, not even as rejected candidates. Their text still reaches
the index as prose, but labelled under whichever section spans that page —
which is why questions about dosing sometimes surface `EPIDEMIOLOGY` chunks from
page 3. If these three matter, hand-curate them into a JSON file rather than
tuning the detector further.

## Known limitation — read this

**Tables are not handled as tables.** The regimen and dosing tables are the most clinically important content in this guideline, and text extraction flattens them into prose. A scrambled dosage row reads like clean text — that is the worst failure mode available in a clinical RAG system.

Before this is production-ready, tables need a separate path via `page.find_tables()`, indexed as whole-table chunks. Flag this to whoever owns retrieval.

## Next stage (Person 3)

`acg_chunks.json` is ready for embedding. Suggested: `BAAI/bge-small-en-v1.5` (384-dim, local, free) into Chroma with cosine distance and `normalize_embeddings=True`. Prefix each chunk with `section > subsection` before embedding so an isolated chunk still carries its context.
