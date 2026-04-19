#!/usr/bin/env python3
from __future__ import annotations

"""Ask one question using retrieval + an extractive QA reader.

This script is the end-to-end answer-generation entry point for the UNSW
Handbook RAG project. It supports BM25, dense, code-constrained dense, and hybrid
retrieval, then passes the top evidence chunks to a pre-trained QA model.

Two practical safeguards are included:
1. Direct AutoModelForQuestionAnswering loading, via qa_reader.py, avoids
   transformers.pipeline task-registry issues.
2. For interactive questions that explicitly mention a course/program code, the
   retriever can apply a strict code filter so that a question about COMP6713 is
   not answered using evidence from an unrelated program page.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from evaluate_retrieval import retrieve_with_method
    from unsw_handbook.build_index import build_bm25_index, load_bm25_index, load_chunks
    from unsw_handbook.dense_index import extract_code_candidates, retrieve_dense
    from unsw_handbook.qa_reader import DEFAULT_QA_MODEL, ExtractiveQAReader
except ImportError as exc:  # pragma: no cover - user environment dependent
    raise SystemExit(
        "Could not import project modules. Run from the project root and set PYTHONPATH=src. "
        f"Original error: {exc}"
    ) from exc


Chunk = Dict[str, Any]
Retrieved = List[Tuple[Chunk, float]]


def _chunk_matches_code(chunk: Chunk, codes: Sequence[str]) -> bool:
    """Return True if the chunk metadata belongs to one of the target codes."""
    if not codes:
        return False
    code_set = {str(code).upper() for code in codes if str(code).strip()}
    fields = [
        str(chunk.get("code", "")).upper(),
        str(chunk.get("page_id", "")).upper(),
        str(chunk.get("url", "")).upper(),
        str(chunk.get("title", "")).upper(),
    ]
    return any(code in field for code in code_set for field in fields)


def filter_chunks_by_question_code(question: str, chunks: Sequence[Chunk]) -> List[Chunk]:
    """Filter chunks when the question explicitly mentions a course/program code."""
    codes = extract_code_candidates(question)
    if not codes:
        return list(chunks)
    matched = [chunk for chunk in chunks if _chunk_matches_code(chunk, codes)]
    return matched or list(chunks)


def load_retrieval_resources(args: argparse.Namespace, chunks: List[Chunk]):
    """Load BM25 and/or dense resources needed by the selected method."""
    bm25_index = None
    dense_index = None
    if args.method in {"bm25", "hybrid"}:
        bm25_index = load_bm25_index(args.index) if args.index else build_bm25_index(chunks)
    if args.method in {"dense", "dense_code", "hybrid"}:
        from unsw_handbook.dense_index import load_dense_index

        dense_index = load_dense_index(args.dense_index)
    return bm25_index, dense_index


def _fuse_ranked_lists(
    bm25_results: Retrieved,
    dense_results: Retrieved,
    limit: int,
    rrf_k: float,
    bm25_weight: float,
    dense_weight: float,
) -> Retrieved:
    """Weighted reciprocal-rank fusion for answer-time retrieval."""
    fused: Dict[str, Dict[str, Any]] = {}

    def add(results: Retrieved, weight: float, source_name: str) -> None:
        for rank, (chunk, original_score) in enumerate(results, start=1):
            chunk_id = str(chunk.get("chunk_id", ""))
            if not chunk_id:
                continue
            item = fused.setdefault(
                chunk_id,
                {
                    "chunk": chunk,
                    "score": 0.0,
                    "bm25_rank": "",
                    "dense_rank": "",
                    "bm25_score": "",
                    "dense_score": "",
                },
            )
            item["score"] += float(weight) / (float(rrf_k) + float(rank))
            item[f"{source_name}_rank"] = rank
            item[f"{source_name}_score"] = float(original_score)

    add(bm25_results, bm25_weight, "bm25")
    add(dense_results, dense_weight, "dense")

    ranked = sorted(
        fused.values(),
        key=lambda item: (
            -float(item["score"]),
            int(item["bm25_rank"] or 10**9),
            int(item["dense_rank"] or 10**9),
        ),
    )
    return [(item["chunk"], float(item["score"])) for item in ranked[: max(1, int(limit))]]


def retrieve_for_answer(args: argparse.Namespace, chunks: List[Chunk], bm25_index: Any, dense_index: Any) -> Retrieved:
    """Retrieve evidence for QA, with an optional strict code filter.

    The retrieval evaluation intentionally measures the raw methods. For the
    interactive answer-generation script, however, it is safer to respect an
    explicit course/program code in the question as a hard page constraint.
    """
    limit = max(1, int(args.top_k))
    pool_size = max(int(args.hybrid_pool_size), limit)
    question = args.question

    strict_filter = bool(getattr(args, "strict_code_filter", True))
    code_filtered_chunks = filter_chunks_by_question_code(question, chunks) if strict_filter else list(chunks)
    has_code_filter = strict_filter and len(code_filtered_chunks) < len(chunks)

    if has_code_filter and args.method == "bm25":
        local_index = build_bm25_index(code_filtered_chunks)
        return retrieve_with_method(
            method="bm25",
            question=question,
            chunks=code_filtered_chunks,
            limit=limit,
            bm25_index=local_index,
        )

    if has_code_filter and args.method == "hybrid":
        local_index = build_bm25_index(code_filtered_chunks)
        bm25_results = retrieve_with_method(
            method="bm25",
            question=question,
            chunks=code_filtered_chunks,
            limit=pool_size,
            bm25_index=local_index,
        )
        dense_results = retrieve_dense(
            question=question,
            chunks=chunks,
            dense_index=dense_index,
            limit=pool_size,
            model_name=(args.dense_model_name or None),
            constrain_to_question_code=True,
        )
        # Keep dense results inside the same explicit code constraint as BM25.
        dense_results = [(c, s) for c, s in dense_results if _chunk_matches_code(c, extract_code_candidates(question))]
        return _fuse_ranked_lists(
            bm25_results=bm25_results,
            dense_results=dense_results,
            limit=limit,
            rrf_k=args.hybrid_rrf_k,
            bm25_weight=args.hybrid_bm25_weight,
            dense_weight=args.hybrid_dense_weight,
        )

    # For dense, automatically switch to code-constrained dense when strict mode
    # finds an explicit code. This avoids obviously wrong page answers.
    method = args.method
    if has_code_filter and method == "dense":
        method = "dense_code"

    return retrieve_with_method(
        method=method,
        question=question,
        chunks=chunks,
        limit=limit,
        bm25_index=bm25_index,
        dense_index=dense_index,
        dense_model_name=(args.dense_model_name or None),
        hybrid_pool_size=args.hybrid_pool_size,
        hybrid_rrf_k=args.hybrid_rrf_k,
        hybrid_bm25_weight=args.hybrid_bm25_weight,
        hybrid_dense_weight=args.hybrid_dense_weight,
    )


def format_text_output(result: Dict[str, Any], evidence_rows: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("Answer")
    lines.append("=" * 80)
    lines.append(result.get("answer", ""))
    lines.append("")
    lines.append(f"QA model: {result.get('model_name', '')}")
    lines.append(f"Answer strategy: {result.get('answer_strategy', '')}")
    lines.append(f"QA score: {float(result.get('qa_score', 0.0)):.4f}")
    lines.append(f"Source chunk: {result.get('chunk_id', '')}")
    lines.append(f"Source section: {result.get('section', '')}")
    lines.append(f"Source URL: {result.get('url', '')}")
    if result.get("warning"):
        lines.append(f"Warning: {result.get('warning')}")
    lines.append("")
    lines.append("Top retrieved evidence")
    lines.append("-" * 80)
    for row in evidence_rows:
        lines.append(
            f"#{row['rank']} | score={row['score']:.6f} | "
            f"chunk={row['chunk_id']} | section={row['section']} | code={row['code']} | title={row['title']}"
        )
        lines.append(row["preview"])
        lines.append("")
    return "\n".join(lines).rstrip()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Answer a Handbook question with retrieval + extractive QA.")
    parser.add_argument("--question", required=True, help="Question to ask the system.")
    parser.add_argument(
        "--method",
        default="hybrid",
        choices=["bm25", "dense", "dense_code", "hybrid"],
        help="Retrieval method used before the QA reader.",
    )
    parser.add_argument("--chunks", default="data/parsed/chunks.jsonl", help="Path to chunks.jsonl.")
    parser.add_argument("--index", default="data/index/bm25_index.json", help="Path to bm25_index.json.")
    parser.add_argument("--dense-index", default="data/index/dense_index.npz", help="Path to dense_index.npz.")
    parser.add_argument("--dense-model-name", default="", help="Optional SentenceTransformer model name/local path.")
    parser.add_argument("--qa-model-name", default=DEFAULT_QA_MODEL, help="Hugging Face extractive QA model.")
    parser.add_argument("--device", type=int, default=-1, help="Transformers device: -1 for CPU, 0 for first GPU.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks to show.")
    parser.add_argument("--reader-top-n", type=int, default=3, help="Number of chunks passed to the QA reader.")
    parser.add_argument("--max-context-chars", type=int, default=3000, help="Max characters per context.")
    parser.add_argument("--min-answer-score", type=float, default=0.0, help="Fallback if QA score is below this.")
    parser.add_argument("--hybrid-pool-size", type=int, default=50)
    parser.add_argument("--hybrid-rrf-k", type=float, default=60.0)
    parser.add_argument("--hybrid-bm25-weight", type=float, default=0.6)
    parser.add_argument("--hybrid-dense-weight", type=float, default=0.4)
    parser.add_argument(
        "--no-strict-code-filter",
        dest="strict_code_filter",
        action="store_false",
        help="Disable hard filtering by explicit course/program code for answer generation.",
    )
    parser.set_defaults(strict_code_filter=True)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a human-readable answer.")
    parser.add_argument("--out", default="", help="Optional path to save JSON output.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    bm25_index, dense_index = load_retrieval_resources(args, chunks)
    retrieved = retrieve_for_answer(args, chunks, bm25_index, dense_index)

    reader = ExtractiveQAReader(
        model_name=args.qa_model_name,
        device=args.device,
        max_context_chars=args.max_context_chars,
        min_answer_score=args.min_answer_score,
        fallback_to_evidence=True,
    )
    answer = reader.answer_from_chunks(args.question, retrieved, top_n_contexts=args.reader_top_n)

    evidence_rows = []
    for rank, (chunk, score) in enumerate(retrieved, start=1):
        text = str(chunk.get("chunk_text", "") or chunk.get("text", "")).replace("\n", " ").strip()
        evidence_rows.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": str(chunk.get("chunk_id", "")),
                "page_id": str(chunk.get("page_id", "")),
                "section": str(chunk.get("section", "")),
                "code": str(chunk.get("code", "")),
                "title": str(chunk.get("title", "")),
                "url": str(chunk.get("url", "")),
                "preview": text[:500],
            }
        )

    payload = {
        "question": args.question,
        "retrieval_method": args.method,
        "strict_code_filter": bool(args.strict_code_filter),
        "answer": answer.to_dict(),
        "retrieved_evidence": evidence_rows,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(format_text_output(answer.to_dict(), evidence_rows))


if __name__ == "__main__":
    main()
