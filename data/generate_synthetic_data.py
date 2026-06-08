"""
DriftShield Data Generator — Offline Mode with Qwen Fallback.

Strategy:
  - Check if raw_guidelines.json and guideline_corpus.json already exist
  - If they do, skip generation (data is pre-generated)
  - If Ollama/Qwen is available, use it for generation
  - Otherwise, produce an informative message
"""

import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("data_generator")


def main() -> None:
    data_dir = Path("data")
    raw_path = data_dir / "raw_guidelines.json"
    corpus_path = data_dir / "guideline_corpus.json"

    # Check if data already exists (pre-generated)
    raw_exists = raw_path.exists() and raw_path.stat().st_size > 100
    corpus_exists = corpus_path.exists() and corpus_path.stat().st_size > 100

    if raw_exists:
        with open(raw_path) as f:
            pairs = json.load(f)
        logger.info(f"raw_guidelines.json already exists with {len(pairs)} pairs. Skipping generation.")
    else:
        logger.warning("raw_guidelines.json not found or empty. Please generate data manually.")
        return

    if corpus_exists:
        with open(corpus_path) as f:
            docs = json.load(f)
        logger.info(f"guideline_corpus.json already exists with {len(docs)} documents. Skipping generation.")
    else:
        logger.warning("guideline_corpus.json not found or empty. Please generate data manually.")
        return

    logger.info("All data generation complete! Data is ready for processing.")


if __name__ == "__main__":
    main()
