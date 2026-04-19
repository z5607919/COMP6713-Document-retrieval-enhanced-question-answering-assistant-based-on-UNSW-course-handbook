#!/usr/bin/env python3
from __future__ import annotations

"""Smoke test for the optional LangChain integration layer."""

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
            chunks_path="data/parsed/chunks.jsonl",
            bm25_index_path="data/index/bm25_index.json",
            dense_index_path="data/index/dense_index.npz",
            retrieval_method="hybrid",
            hybrid_bm25_weight=0.6,
            hybrid_dense_weight=0.4,
        )
    )
    runnable = pipeline.to_langchain_runnable()
    payload = runnable.invoke("How many units of credit is COMP6714 worth?")
    answer = payload.get("answer", {}).get("answer", "")
    source = payload.get("answer", {}).get("chunk_id", "")
    print("LangChain Runnable smoke test")
    print(f"Answer: {answer}")
    print(f"Source chunk: {source}")
    if not answer:
        raise SystemExit("LangChain smoke test failed: empty answer")
    print("LangChain smoke test passed.")


if __name__ == "__main__":
    main()
