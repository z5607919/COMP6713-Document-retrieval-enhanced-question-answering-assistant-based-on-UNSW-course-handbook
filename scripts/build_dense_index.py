#!/usr/bin/env python3
from __future__ import annotations

"""Build a dense retrieval index for UNSW Handbook chunks.

Example from the project root:

    $env:PYTHONPATH="src"
    python scripts/build_dense_index.py `
        --chunks data/parsed/chunks.jsonl `
        --out data/index/dense_index.npz `
        --model-name sentence-transformers/all-MiniLM-L6-v2

The first run may download the pre-trained model. The generated dense_index.npz can
be reused for evaluation and demo runs.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unsw_handbook.build_index import load_chunks
from unsw_handbook.dense_index import DEFAULT_DENSE_MODEL, build_dense_embeddings, save_dense_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a dense index over Handbook chunks.")
    parser.add_argument("--chunks", default="data/parsed/chunks.jsonl", help="Path to chunks.jsonl.")
    parser.add_argument("--out", default="data/index/dense_index.npz", help="Output dense index path.")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_DENSE_MODEL,
        help="SentenceTransformer model name or local model path.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size.")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    if not chunks:
        raise ValueError(f"No chunks loaded from {args.chunks}")

    print(f"Loaded {len(chunks)} chunks from {args.chunks}")
    print(f"Encoding chunks with model: {args.model_name}")
    embeddings = build_dense_embeddings(chunks, model_name=args.model_name, batch_size=args.batch_size)
    save_dense_index(args.out, chunks, embeddings, model_name=args.model_name)
    print(f"Saved dense index to {args.out}")
    print(f"Embedding matrix shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
