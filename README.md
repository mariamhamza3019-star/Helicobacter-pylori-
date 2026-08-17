# ACG H. pylori Clinical RAG — Hackathon Project

Evidence-based clinical decision support using Retrieval-Augmented Generation over the **ACG 2024 H. pylori treatment guideline**.

## Architecture (4 layers)

```
Layer 1 — Ingestion     ingestion/          PDF → chunks + metadata
Layer 1 — Embeddings    embedding_generator.py
Layer 2 — Retrieval     notebooks/          FAISS index + search
Layer 3 — Generation    (Day 2+)            Grounded LLM + citations
Layer 4 — Safety        (Day 3+)            Hallucination check / refusal
```

All paths are defined in `paths.py` — import from there, don't hardcode.

## Quick start

```bash
pip install -r requirements.txt

# Ingestion only (inspect → parse → chunk → eval)
python run_pipeline.py

# Ingestion + embeddings
python run_pipeline.py --embed

# Build FAISS index (open in Jupyter / Colab)
# notebooks/VectorDB_Retrieval.ipynb
```

Windows shortcut: double-click `ingestion/RUN_ME.bat` (or `AI project/RUN_ME.bat`).

## Key outputs

| File | What |
|---|---|
| `ingestion/data/processed/acg_chunks.json` | 103 chunks, ready for embedding |
| `ingestion/data/processed/acg_sections.json` | Section detection audit log |
| `ingestion/data/processed/eval_report.txt` | BM25 regression eval (30 questions) |
| `outputs/chunks_with_embeddings.json` | Chunks + BioBERT vectors |
| `outputs/h_pylori_faiss.index` | FAISS index |
| `outputs/h_pylori_metadata.json` | Chunk metadata for retrieval |

## Corpus stats

- **Guideline:** ACG Clinical Guideline 2024 — Treatment of H. pylori Infection (24 pages)
- **Chunks:** 103 (88 prose + 10 table + 5 table_summary)
- **Embedding model:** `pritamdeka/S-BioBert-snli-multinli-stsb` (768-dim)
- **Eval:** section recall@1 86.7%, recall@5 100%, MRR 0.933

## Folder layout

```
├── paths.py                  # shared path constants
├── run_pipeline.py           # one-command ingestion runner
├── embedding_generator.py    # BioBERT embedding step
├── requirements.txt
├── ingestion/                # Layer 1 scripts + data (was "AI project")
│   ├── 1_inspect.py
│   ├── 2_parse_chunk.py
│   ├── 3_eval.py
│   ├── gold_questions.json
│   └── data/raw/ + data/processed/
├── notebooks/                # Layer 2 Jupyter notebook + README
└── outputs/                  # embeddings, FAISS index, retrieval results
```

## Team notes

- Chunking details and tuning guide: `ingestion/README.md`
- Retrieval notebook README: `notebooks/README.md`
- When the `AI project` folder isn't locked, rename it to `ingestion` — `paths.py` already supports both names.
