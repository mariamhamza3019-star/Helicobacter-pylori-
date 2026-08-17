# Task 5: Vector DB + Indexing + Retrieval Testing

##  Description
Build a vector database from the chunks (with embeddings) prepared in Task 4, index them with FAISS, and run retrieval tests against a set of clinical queries to confirm the Top-K results are actually relevant.

## Steps
- Load the chunks along with their precomputed embeddings
- Validate that every chunk has an embedding and that chunk IDs are unique
- Build the embedding matrix and normalize it (for cosine similarity)
- Create a FAISS index (`IndexFlatIP` on normalized vectors)
- Save the index and metadata
- Implement a retrieval function that takes a text query and returns the closest chunks
- Test 7 clinical queries and evaluate relevance by comparing retrieved sections against `expect_sections` in `gold_questions.json`

## Results
- **Embedding model**: `pritamdeka/S-BioBert-snli-multinli-stsb`
- **Vector dimension**: 768
- **Similarity metric**: Cosine similarity (via normalized inner product)
- **Queries tested**: 7
- **Recall@5**: 85.71% (6/7 queries retrieved a chunk from the correct section within the top 5 results)


## note
The one query that failed was "duration of treatment" — retrieval returned chunks from the treatment-experienced section instead of treatment-naive, likely because one densely-worded chunk (`ACG_0072`) repeatedly surfaces across unrelated queries due to topical/vocabulary overlap rather than true relevance.

## 🚀 How to run
```bash
pip install -r requirements.txt
python build_and_retrieve.py
```

### Outputs (outputs/)
- `h_pylori_faiss.index` — the saved FAISS index
- `h_pylori_metadata.json` — metadata for each chunk (id, source, page, section, ...)
- `retrieval_results.csv` — retrieval results for all 7 queries (rank, score, relevant, query_passed)
