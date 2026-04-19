#!/usr/bin/env python3
from __future__ import annotations

"""Small smoke test for the RAG pipeline."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unsw_handbook.rag_pipeline import HandbookRAGPipeline, RAGConfig


def main() -> None:
    pipeline = HandbookRAGPipeline(
        RAGConfig(
            retrieval_method="hybrid",
            hybrid_bm25_weight=0.6,
            hybrid_dense_weight=0.4,
            top_k=5,
            reader_top_n=3,
        )
    )
    question = "How many units of credit is COMP6714 worth?"
    payload = pipeline.answer(question)
    answer = payload["answer"]
    print("Question:", question)
    print("Answer:", answer.get("answer", ""))
    print("Source chunk:", answer.get("chunk_id", ""))
    print("Used pre-trained model:", answer.get("used_pretrained_model", ""))
    assert answer.get("answer"), "RAG pipeline returned an empty answer."
    assert payload.get("retrieved_evidence"), "RAG pipeline returned no evidence."
    print("RAG smoke test passed.")


if __name__ == "__main__":
    main()
