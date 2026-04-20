#!/usr/bin/env python3
from __future__ import annotations

"""Small smoke test for the generative reader path."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from answer_with_generative import main

if __name__ == "__main__":
    sys.argv = [
        "answer_with_generative.py",
        "--question", "Which terms is COMP6714 offered in?",
        "--method", "hybrid",
        "--chunks", "data/parsed/chunks.jsonl",
        "--index", "data/index/bm25_index.json",
        "--dense-index", "data/index/dense_index.npz",
        "--selector-model", "data/models/answer_selector.json",
        "--hybrid-bm25-weight", "0.6",
        "--hybrid-dense-weight", "0.4",
        "--top-k", "5",
        "--reader-top-n", "1",
    ]
    main()
