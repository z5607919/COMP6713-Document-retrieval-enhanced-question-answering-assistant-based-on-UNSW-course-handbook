#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate retrieval + extractive QA answers against annotated short answers."""

import argparse
import csv
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from answer_with_qa import load_retrieval_resources, retrieve_for_answer
    from evaluate_retrieval import read_annotations
    from unsw_handbook.build_index import load_chunks
    from unsw_handbook.qa_reader import DEFAULT_QA_MODEL, ExtractiveQAReader
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import project modules. Run from the project root and set PYTHONPATH=src. "
        f"Original error: {exc}"
    ) from exc


def normalise_answer(text: str) -> str:
    """SQuAD-style answer normalisation for simple exact/F1 metrics."""
    text = str(text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalise_answer(prediction).split()
    gold_tokens = normalise_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def write_csv(path: str | Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate retrieval + extractive QA answers.")
    parser.add_argument("--annotations", default="annotations.csv")
    parser.add_argument("--chunks", default="data/parsed/chunks.jsonl")
    parser.add_argument(
        "--method",
        default="hybrid",
        choices=["bm25", "dense", "dense_code", "hybrid"],
        help="Retrieval method used before the QA reader.",
    )
    parser.add_argument("--index", default="data/index/bm25_index.json")
    parser.add_argument("--dense-index", default="data/index/dense_index.npz")
    parser.add_argument("--dense-model-name", default="")
    parser.add_argument("--qa-model-name", default=DEFAULT_QA_MODEL)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--reader-top-n", type=int, default=3)
    parser.add_argument("--max-context-chars", type=int, default=3000)
    parser.add_argument("--min-answer-score", type=float, default=0.0)
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
    parser.add_argument("--max-examples", type=int, default=0, help="Optional limit for quick smoke tests.")
    parser.add_argument("--out-dir", default="data/results/qa_hybrid")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotations = read_annotations(args.annotations)
    if args.max_examples and args.max_examples > 0:
        annotations = annotations[: args.max_examples]

    chunks = load_chunks(args.chunks)
    bm25_index, dense_index = load_retrieval_resources(args, chunks)
    reader = ExtractiveQAReader(
        model_name=args.qa_model_name,
        device=args.device,
        max_context_chars=args.max_context_chars,
        min_answer_score=args.min_answer_score,
        fallback_to_evidence=True,
    )

    rows: List[Dict[str, Any]] = []
    for i, ann in enumerate(annotations, start=1):
        question = ann.get("question", "")
        args.question = question
        retrieved = retrieve_for_answer(args, chunks, bm25_index, dense_index)
        answer = reader.answer_from_chunks(question, retrieved, top_n_contexts=args.reader_top_n)
        pred = answer.answer
        gold = ann.get("answer", "")
        norm_pred = normalise_answer(pred)
        norm_gold = normalise_answer(gold)
        exact_match = int(norm_pred == norm_gold)
        contains_gold = int(bool(norm_gold) and norm_gold in norm_pred)
        gold_contains_pred = int(bool(norm_pred) and norm_pred in norm_gold)
        f1 = token_f1(pred, gold)
        retrieved_ids = [str(chunk.get("chunk_id", "")) for chunk, _ in retrieved]
        row = {
            "id": ann.get("id", ""),
            "question": question,
            "gold_answer": gold,
            "predicted_answer": pred,
            "normalised_gold": norm_gold,
            "normalised_prediction": norm_pred,
            "exact_match": exact_match,
            "token_f1": f"{f1:.6f}",
            "contains_gold": contains_gold,
            "gold_contains_prediction": gold_contains_pred,
            "qa_score": f"{answer.qa_score:.6f}",
            "answer_strategy": answer.answer_strategy,
            "source_rank": answer.source_rank,
            "source_chunk_id": answer.chunk_id,
            "source_section": answer.section,
            "source_url": answer.url,
            "retrieved_chunk_ids": " | ".join(retrieved_ids),
            "gold_chunk_id": ann.get("gold_chunk_id", ""),
            "gold_in_retrieved_topk": int(ann.get("gold_chunk_id", "") in retrieved_ids),
            "used_pretrained_model": int(answer.used_pretrained_model),
            "fallback_used": int(answer.fallback_used),
            "warning": answer.warning,
        }
        rows.append(row)
        if i % 25 == 0:
            print(f"Processed {i}/{len(annotations)} examples...")

    n = max(1, len(rows))
    metrics = {
        "method": args.method,
        "qa_model_name": args.qa_model_name,
        "n_examples": len(rows),
        "exact_match": sum(int(r["exact_match"]) for r in rows) / n,
        "average_token_f1": sum(float(r["token_f1"]) for r in rows) / n,
        "contains_gold": sum(int(r["contains_gold"]) for r in rows) / n,
        "gold_contains_prediction": sum(int(r["gold_contains_prediction"]) for r in rows) / n,
        "gold_in_retrieved_topk": sum(int(r["gold_in_retrieved_topk"]) for r in rows) / n,
        "fallback_rate": sum(int(r["fallback_used"]) for r in rows) / n,
        "pretrained_model_usage_rate": sum(int(r["used_pretrained_model"]) for r in rows) / n,
        "strict_code_filter": int(bool(args.strict_code_filter)),
        "hybrid_bm25_weight": args.hybrid_bm25_weight if args.method == "hybrid" else "",
        "hybrid_dense_weight": args.hybrid_dense_weight if args.method == "hybrid" else "",
    }

    fields = [
        "id",
        "question",
        "gold_answer",
        "predicted_answer",
        "normalised_gold",
        "normalised_prediction",
        "exact_match",
        "token_f1",
        "contains_gold",
        "gold_contains_prediction",
        "qa_score",
        "answer_strategy",
        "source_rank",
        "source_chunk_id",
        "source_section",
        "source_url",
        "retrieved_chunk_ids",
        "gold_chunk_id",
        "gold_in_retrieved_topk",
        "used_pretrained_model",
        "fallback_used",
        "warning",
    ]
    write_csv(out_dir / "qa_predictions.csv", rows, fields)
    write_csv(out_dir / "qa_metrics.csv", [{"metric": k, "value": v} for k, v in metrics.items()], ["metric", "value"])
    (out_dir / "qa_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print(f"QA evaluation summary ({args.method} + extractive QA)")
    print("=" * 80)
    print(f"Examples: {metrics['n_examples']}")
    print(f"Exact Match: {metrics['exact_match']:.4f}")
    print(f"Average token F1: {metrics['average_token_f1']:.4f}")
    print(f"Contains gold answer: {metrics['contains_gold']:.4f}")
    print(f"Gold in retrieved top-{args.top_k}: {metrics['gold_in_retrieved_topk']:.4f}")
    print(f"Pretrained model usage rate: {metrics['pretrained_model_usage_rate']:.4f}")
    print(f"Fallback rate: {metrics['fallback_rate']:.4f}")
    print("=" * 80)
    print(f"Saved QA results to: {out_dir}")


if __name__ == "__main__":
    main()
