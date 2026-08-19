"""
Single source of truth for all file paths in this repo.
Import from here instead of hardcoding paths in each script.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Layer 1 — ingestion scripts
INGESTION = ROOT / "ingestion"

# Data — raw PDF + processed JSON artifacts
DATA = ROOT / "data"
PDF_DIR = DATA / "raw"
PROCESSED = DATA / "processed"
CHUNKS_JSON = PROCESSED / "acg_chunks.json"
SECTIONS_JSON = PROCESSED / "acg_sections.json"
GOLD_QUESTIONS = PROCESSED / "gold_questions.json"

# Vector DB — FAISS index + its metadata + the embeddings that feed it
# (binary/data artifacts, kept separate from human-readable reports)
VECTOR_DB = ROOT / "vectorDB"
FAISS_INDEX = VECTOR_DB / "h_pylori_faiss.index"
METADATA_JSON = VECTOR_DB / "h_pylori_metadata.json"
EMBEDDINGS_JSON = VECTOR_DB / "chunks_with_embeddings.json"

# Inspection — test scripts + human-readable logs/reports
INSPECTION = ROOT / "inspection"
EVAL_REPORT = INSPECTION / "eval_report.txt"
INSPECT_TXT = INSPECTION / "inspect.txt"
HYBRID_REPORT = INSPECTION / "hybrid_search_report.txt"
RETRIEVAL_CSV = INSPECTION / "retrieval_results.csv"
DAY4_EVAL_REPORT = INSPECTION / "day4_eval_report.txt"

# OUTPUTS no longer exists as its own folder — hybrid_search.py and
# day4_eval.py both still import OUTPUTS by name to decide where to write
# their reports. Aliasing it to INSPECTION keeps both working with zero
# changes to those files, and their output now correctly lands next to the
# other reports in inspection/.
OUTPUTS = INSPECTION

NOTEBOOKS = ROOT / "notebooks"