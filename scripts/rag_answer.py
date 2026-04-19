#!/usr/bin/env python3
from __future__ import annotations

"""Run the reusable RAG pipeline from the command line."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from unsw_handbook.qa_reader import DEFAULT_QA_MODEL
    from unsw_handbook.rag_pipeline import HandbookRAGPipeline, RAGConfig, save_payload
except ImportError as exc:  # pragma: no cover - user environment dependent
    raise SystemExit(
        "Could not import RAG pipeline modules. Run from the project root and set PYTHONPATH=src. "
        f"Original error: {exc}"
    ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Answer a UNSW Handbook question with the RAG pipeline.")
    parser.add_argument("--question", required=True, help="Question to ask.")
    parser.add_argument("--chunks", default="data/parsed/chunks.jsonl")
    parser.add_argument("--index", default="data/index/bm25_index.json")
    parser.add_argument("--dense-index", default="data/index/dense_index.npz")
    parser.add_argument("--method", default="hybrid", choices=["bm25", "dense", "dense_code", "hybrid"])
    parser.add_argument("--dense-model-name", default="")
    parser.add_argument("--qa-model-name", default=DEFAULT_QA_MODEL)
    parser.add_argument("--device", type=int, default=-1, help="-1 for CPU, 0 for first CUDA GPU.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--reader-top-n", type=int, default=3)
    parser.add_argument("--max-context-chars", type=int, default=3000)
    parser.add_argument("--min-answer-score", type=float, default=0.0)
    parser.add_argument("--hybrid-pool-size", type=int, default=50)
    parser.add_argument("--hybrid-rrf-k", type=float, default=60.0)
    parser.add_argument("--hybrid-bm25-weight", type=float, default=0.6)
    parser.add_argument("--hybrid-dense-weight", type=float, default=0.4)
    parser.add_argument("--use-langchain", action="store_true", help="Run through the optional LangChain Runnable wrapper.")
    parser.add_argument("--json", action="store_true", help="Print full JSON payload instead of compact text.")
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    return parser


def make_config(args: argparse.Namespace) -> RAGConfig:
    return RAGConfig(
        chunks_path=args.chunks,
        bm25_index_path=args.index,
        dense_index_path=args.dense_index,
        retrieval_method=args.method,
        dense_model_name=args.dense_model_name,
        qa_model_name=args.qa_model_name,
        device=args.device,
        top_k=args.top_k,
        reader_top_n=args.reader_top_n,
        max_context_chars=args.max_context_chars,
        min_answer_score=args.min_answer_score,
        hybrid_pool_size=args.hybrid_pool_size,
        hybrid_rrf_k=args.hybrid_rrf_k,
        hybrid_bm25_weight=args.hybrid_bm25_weight,
        hybrid_dense_weight=args.hybrid_dense_weight,
    )


def format_payload(payload: Dict[str, Any]) -> str:
    answer = payload.get("answer", {})
    evidence = payload.get("retrieved_evidence", [])
    lines = []
    lines.append("=" * 80)
    lines.append("RAG answer")
    lines.append("=" * 80)
    lines.append(str(answer.get("answer", "")))
    lines.append("")
    lines.append(f"Retrieval method: {payload.get('retrieval_method', '')}")
    lines.append(f"QA model: {answer.get('model_name', payload.get('qa_model_name', ''))}")
    lines.append(f"Answer strategy: {answer.get('answer_strategy', '')}")
    lines.append(f"QA score: {float(answer.get('qa_score', 0.0)):.4f}")
    lines.append(f"Source chunk: {answer.get('chunk_id', '')}")
    lines.append(f"Source section: {answer.get('section', '')}")
    lines.append(f"Source URL: {answer.get('url', '')}")
    warning = str(answer.get("warning", ""))
    if warning:
        lines.append(f"Warning: {warning}")
    lines.append("")
    lines.append("Top retrieved evidence")
    lines.append("-" * 80)
    for row in evidence:
        lines.append(
            f"#{row.get('rank')} | score={float(row.get('score', 0.0)):.6f} | "
            f"chunk={row.get('chunk_id', '')} | section={row.get('section', '')} | "
            f"code={row.get('code', '')} | title={row.get('title', '')}"
        )
        lines.append(str(row.get("preview", "")))
        lines.append("")
    return "\n".join(lines).rstrip()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    pipeline = HandbookRAGPipeline(make_config(args))

    if args.use_langchain:
        runnable = pipeline.to_langchain_runnable()
        payload = runnable.invoke(args.question)
    else:
        payload = pipeline.answer(args.question)

    if args.out:
        save_payload(args.out, payload)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(format_payload(payload))


if __name__ == "__main__":
    main()
