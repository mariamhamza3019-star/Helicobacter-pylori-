# Layer 2: Vector DB + Retrieval

Build a FAISS index from embedded chunks and test retrieval against clinical queries.

## Prerequisites

Run ingestion + embeddings first (from repo root):

```bash
python run_pipeline.py --embed
```

This produces `outputs/chunks_with_embeddings.json`.

## How to run

Open `VectorDB_Retrieval.ipynb` in Jupyter or Colab.

**Run from the repo root** so paths resolve correctly:

```python
# First cell — set paths relative to repo root
from pathlib import Path
ROOT = Path.cwd()
ACG_FILE = ROOT / "outputs" / "chunks_with_embeddings.json"
INDEX_FILE = ROOT / "outputs" / "h_pylori_faiss.index"
METADATA_FILE = ROOT / "outputs" / "h_pylori_metadata.json"
RESULTS_FILE = ROOT / "outputs" / "retrieval_results.csv"
GOLD = ROOT / "ingestion" / "gold_questions.json"  # or "AI project" if not renamed
```

## Results

- **Embedding model**: `pritamdeka/S-BioBert-snli-multinli-stsb`
- **Vector dimension**: 768
- **Similarity**: cosine (normalized inner product via FAISS `IndexFlatIP`)
- **Recall@5**: 85.71% on 7 test queries

## Outputs (`outputs/`)

| File | Contents |
|---|---|
| `h_pylori_faiss.index` | Saved FAISS index |
| `h_pylori_metadata.json` | Per-chunk metadata (id, section, page, citation) |
| `retrieval_results.csv` | Query results with rank, score, relevance |
