#!/usr/bin/env python3
from __future__ import annotations

"""Answer one question with retrieval -> optional selector -> generative reader."""

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
from unsw_handbook.generative_reader import DEFAULT_GENERATIVE_MODEL, GenerativeQAReader


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Answer with selector-RAG and a generative reader.")
    p.add_argument("--question", required=True)
    p.add_argument("--chunks", default="data/parsed/chunks.jsonl")
    p.add_argument("--method", default="hybrid", choices=["bm25", "dense", "dense_code", "hybrid"])
    p.add_argument("--index", default="data/index/bm25_index.json")
    p.add_argument("--dense-index", default="data/index/dense_index.npz")
    p.add_argument("--dense-model-name", default="")
    p.add_argument("--selector-model", default="data/models/answer_selector.json")
    p.add_argument("--no-selector", action="store_true", help="Skip the supervised selector and feed retrieved chunks directly to the reader.")
    p.add_argument("--generative-model-name", default=DEFAULT_GENERATIVE_MODEL)
    p.add_argument("--device", type=int, default=-1)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--reader-top-n", type=int, default=1)
    p.add_argument("--max-context-chars", type=int, default=1800)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--num-beams", type=int, default=4)
    p.add_argument("--hybrid-pool-size", type=int, default=50)
    p.add_argument("--hybrid-rrf-k", type=float, default=60.0)
    p.add_argument("--hybrid-bm25-weight", type=float, default=0.6)
    p.add_argument("--hybrid-dense-weight", type=float, default=0.4)
    p.add_argument("--no-strict-code-filter", dest="strict_code_filter", action="store_false")
    p.set_defaults(strict_code_filter=True)
    p.add_argument("--json", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    chunks = load_chunks(args.chunks)
    bm25_index, dense_index = load_retrieval_resources(args, chunks)
    retrieved = retrieve_for_answer(args, chunks, bm25_index, dense_index)

    selector_used = False
    reranked = retrieved
    if not args.no_selector:
        selector = TrainedAnswerSelector.load(args.selector_model)
        reranked = selector.rerank(args.question, retrieved)
        selector_used = True

    reader_input = reranked[: max(1, int(args.reader_top_n))]
    reader = GenerativeQAReader(
        model_name=args.generative_model_name,
        device=args.device,
        max_context_chars=args.max_context_chars,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        fallback_to_evidence=True,
    )
    answer = reader.answer_from_chunks(args.question, reader_input, top_n_contexts=args.reader_top_n)

    evidence_key = "selector_ranked_evidence" if selector_used else "retrieved_evidence"
    payload = {
        "question": args.question,
        "retrieval_method": args.method,
        "selector_used": int(selector_used),
        "selector_model": "" if args.no_selector else args.selector_model,
        "generative_model_name": args.generative_model_name,
        "answer": answer.to_dict(),
        evidence_key: [
            {
                "rank": i,
                "score": float(score),
                "chunk_id": str(chunk.get("chunk_id", "")),
                "page_id": str(chunk.get("page_id", "")),
                "section": str(chunk.get("section", "")),
                "code": str(chunk.get("code", "")),
                "title": str(chunk.get("title", "")),
                "url": str(chunk.get("url", "")),
                "preview": str(chunk.get("chunk_text", "") or chunk.get("text", ""))[:900],
            }
            for i, (chunk, score) in enumerate(reranked, start=1)
        ],
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("=" * 80)
    print("Selector-RAG generative answer" if selector_used else "Generative RAG answer")
    print("=" * 80)
    print(answer.answer)
    print("")
    print(f"Source chunk: {answer.chunk_id}")
    print(f"Source section: {answer.section}")
    print(f"Reader model: {answer.model_name}")
    print(f"Answer strategy: {answer.answer_strategy}")
    print(f"Fallback used: {int(answer.fallback_used)}")
    print("")
    print("Top evidence")
    print("-" * 80)
    for item in payload[evidence_key][:5]:
        print(f"#{item['rank']} | score={item['score']:.4f} | {item['chunk_id']} | {item['section']}")
        print(item["preview"])
        print("")


if __name__ == "__main__":
    main()
