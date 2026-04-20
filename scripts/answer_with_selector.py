#!/usr/bin/env python3
from __future__ import annotations

"""Answer one question using retrieval -> supervised selector -> QA reader."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from answer_with_qa import load_retrieval_resources, retrieve_for_answer
from unsw_handbook.answer_selector import TrainedAnswerSelector
from unsw_handbook.build_index import load_chunks
from unsw_handbook.qa_reader import DEFAULT_QA_MODEL, ExtractiveQAReader


def build_arg_parser():
    p = argparse.ArgumentParser(description="Answer with supervised selector-RAG.")
    p.add_argument("--question", required=True)
    p.add_argument("--annotations", default="annotations.csv")
    p.add_argument("--chunks", default="data/parsed/chunks.jsonl")
    p.add_argument("--method", default="hybrid", choices=["bm25", "dense", "dense_code", "hybrid"])
    p.add_argument("--index", default="data/index/bm25_index.json")
    p.add_argument("--dense-index", default="data/index/dense_index.npz")
    p.add_argument("--dense-model-name", default="")
    p.add_argument("--selector-model", default="data/models/answer_selector.json")
    p.add_argument("--qa-model-name", default=DEFAULT_QA_MODEL)
    p.add_argument("--device", type=int, default=-1)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--reader-top-n", type=int, default=1)
    p.add_argument("--max-context-chars", type=int, default=3000)
    p.add_argument("--min-answer-score", type=float, default=0.0)
    p.add_argument("--hybrid-pool-size", type=int, default=50)
    p.add_argument("--hybrid-rrf-k", type=float, default=60.0)
    p.add_argument("--hybrid-bm25-weight", type=float, default=0.6)
    p.add_argument("--hybrid-dense-weight", type=float, default=0.4)
    p.add_argument("--no-strict-code-filter", dest="strict_code_filter", action="store_false")
    p.set_defaults(strict_code_filter=True)
    p.add_argument("--json", action="store_true")
    return p


def main():
    args = build_arg_parser().parse_args()
    chunks = load_chunks(args.chunks)
    bm25_index, dense_index = load_retrieval_resources(args, chunks)
    retrieved = retrieve_for_answer(args, chunks, bm25_index, dense_index)
    selector = TrainedAnswerSelector.load(args.selector_model)
    reranked = selector.rerank(args.question, retrieved)
    selected = reranked[:1]
    reader = ExtractiveQAReader(
        model_name=args.qa_model_name,
        device=args.device,
        max_context_chars=args.max_context_chars,
        min_answer_score=args.min_answer_score,
        fallback_to_evidence=True,
    )
    answer = reader.answer_from_chunks(args.question, selected, top_n_contexts=args.reader_top_n)
    payload = {
        "question": args.question,
        "retrieval_method": args.method,
        "selector_model": args.selector_model,
        "answer": answer.to_dict(),
        "reranked_evidence": [
            {
                "rank": i,
                "selector_score": float(score),
                "chunk_id": str(chunk.get("chunk_id", "")),
                "section": str(chunk.get("section", "")),
                "code": str(chunk.get("code", "")),
                "title": str(chunk.get("title", "")),
                "url": str(chunk.get("url", "")),
                "preview": str(chunk.get("chunk_text", "") or chunk.get("text", ""))[:700],
            }
            for i, (chunk, score) in enumerate(reranked, start=1)
        ],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("=" * 80)
        print("Selector-RAG answer")
        print("=" * 80)
        print(answer.answer)
        print("")
        print(f"Source chunk: {answer.chunk_id}")
        print(f"Source section: {answer.section}")
        print(f"QA model: {answer.model_name}")
        print(f"Answer strategy: {answer.answer_strategy}")
        print("")
        print("Top selector-ranked evidence")
        print("-" * 80)
        for item in payload["reranked_evidence"]:
            print(f"#{item['rank']} | selector_score={item['selector_score']:.4f} | {item['chunk_id']} | {item['section']}")
            print(item["preview"])
            print("")


if __name__ == "__main__":
    main()
