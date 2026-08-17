"""
Single source of truth for all file paths in this repo.
Import from here instead of hardcoding paths in each script.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Layer 1 — ingestion (rename "AI project" → ingestion when the folder isn't locked)
INGESTION = ROOT / "ingestion" if (ROOT / "ingestion").is_dir() else ROOT / "AI project"

PDF_DIR = INGESTION / "data" / "raw"
PROCESSED = INGESTION / "data" / "processed"
CHUNKS_JSON = PROCESSED / "acg_chunks.json"
SECTIONS_JSON = PROCESSED / "acg_sections.json"
EVAL_REPORT = PROCESSED / "eval_report.txt"
INSPECT_TXT = PROCESSED / "inspect.txt"
GOLD_QUESTIONS = INGESTION / "gold_questions.json"

# Layer 1 — embeddings & Layer 2 — retrieval
OUTPUTS = ROOT / "outputs"
EMBEDDINGS_JSON = OUTPUTS / "chunks_with_embeddings.json"
FAISS_INDEX = OUTPUTS / "h_pylori_faiss.index"
METADATA_JSON = OUTPUTS / "h_pylori_metadata.json"
RETRIEVAL_CSV = OUTPUTS / "retrieval_results.csv"
HYBRID_REPORT = OUTPUTS / "hybrid_search_report.txt"

NOTEBOOKS = ROOT / "notebooks"
