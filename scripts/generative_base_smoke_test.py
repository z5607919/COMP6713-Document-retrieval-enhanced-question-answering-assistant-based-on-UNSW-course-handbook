#!/usr/bin/env python3
from __future__ import annotations

print("Run this command from the project root:")
print(
    "python scripts/answer_with_generative.py "
    "--question \"Which terms is COMP6714 offered in?\" "
    "--method hybrid "
    "--chunks data/parsed/chunks.jsonl "
    "--index data/index/bm25_index.json "
    "--dense-index data/index/dense_index.npz "
    "--selector-model data/models/answer_selector.json "
    "--generative-model-name google/flan-t5-base "
    "--hybrid-bm25-weight 0.6 "
    "--hybrid-dense-weight 0.4 "
    "--top-k 5 "
    "--reader-top-n 1 "
    "--max-new-tokens 64 "
    "--num-beams 5"
)
