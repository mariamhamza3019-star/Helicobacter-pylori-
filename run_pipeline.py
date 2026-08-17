"""
Run the full Day-1 ingestion pipeline from the repo root.

  python run_pipeline.py           # inspect + parse + eval
  python run_pipeline.py --embed   # above + generate embeddings
"""
import argparse
import subprocess
import sys

from paths import EMBEDDINGS_JSON, EVAL_REPORT, CHUNKS_JSON, INGESTION, ROOT


def run(script: str, extra: list[str] | None = None) -> None:
    path = INGESTION / script
    cmd = [sys.executable, str(path), *(extra or [])]
    print(f"\n{'=' * 60}\n  {script}\n{'=' * 60}")
    subprocess.check_call(cmd, cwd=INGESTION)


def main() -> None:
    parser = argparse.ArgumentParser(description="ACG H. pylori RAG — ingestion pipeline")
    parser.add_argument("--embed", action="store_true",
                        help="Also run embedding_generator.py after chunking")
    args = parser.parse_args()

    print(f"Repo root: {ROOT}")
    run("1_inspect.py")
    run("2_parse_chunk.py")
    run("3_eval.py")

    if args.embed:
        embed = ROOT / "embedding_generator.py"
        print(f"\n{'=' * 60}\n  embedding_generator.py\n{'=' * 60}")
        subprocess.check_call([sys.executable, str(embed)], cwd=ROOT)

    print("\nDone.")
    print(f"  Chunks : {CHUNKS_JSON}")
    print(f"  Eval   : {EVAL_REPORT}")
    if args.embed:
        print(f"  Embeddings: {EMBEDDINGS_JSON}")


if __name__ == "__main__":
    main()
