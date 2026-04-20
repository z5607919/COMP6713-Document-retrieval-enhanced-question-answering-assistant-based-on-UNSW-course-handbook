#!/usr/bin/env python3
from __future__ import annotations

"""Train a supervised answer selector from annotations.csv."""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from answer_with_qa import load_retrieval_resources, retrieve_for_answer
from evaluate_retrieval import read_annotations
from unsw_handbook.answer_selector import FeatureExtractor, LogisticSelectorModel, TrainedAnswerSelector, build_matrix, evaluate_selector
from unsw_handbook.build_index import load_chunks


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a supervised answer selector.")
    parser.add_argument("--annotations", default="annotations.csv")
    parser.add_argument("--chunks", default="data/parsed/chunks.jsonl")
    parser.add_argument("--method", default="hybrid", choices=["bm25", "dense", "dense_code", "hybrid"])
    parser.add_argument("--index", default="data/index/bm25_index.json")
    parser.add_argument("--dense-index", default="data/index/dense_index.npz")
    parser.add_argument("--dense-model-name", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--hybrid-pool-size", type=int, default=50)
    parser.add_argument("--hybrid-rrf-k", type=float, default=60.0)
    parser.add_argument("--hybrid-bm25-weight", type=float, default=0.6)
    parser.add_argument("--hybrid-dense-weight", type=float, default=0.4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--out", default="data/models/answer_selector.json")
    parser.add_argument("--out-dir", default="data/results/answer_selector")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--no-strict-code-filter", dest="strict_code_filter", action="store_false")
    parser.set_defaults(strict_code_filter=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotations = read_annotations(args.annotations)
    if args.max_examples and args.max_examples > 0:
        annotations = annotations[: args.max_examples]
    chunks = load_chunks(args.chunks)
    bm25_index, dense_index = load_retrieval_resources(args, chunks)
    extractor = FeatureExtractor.from_chunks(chunks)

    questions: List[str] = []
    gold_ids: List[str] = []
    retrieved_lists = []
    candidate_rows: List[Dict[str, Any]] = []

    for i, ann in enumerate(annotations, start=1):
        question = ann.get("question", "")
        args.question = question
        retrieved = retrieve_for_answer(args, chunks, bm25_index, dense_index)
        questions.append(question)
        gold_id = ann.get("gold_chunk_id", "")
        gold_ids.append(gold_id)
        retrieved_lists.append(retrieved)
        ids = [str(c.get("chunk_id", "")) for c, _ in retrieved]
        for rank, (chunk, score) in enumerate(retrieved, start=1):
            candidate_rows.append({
                "id": ann.get("id", ""),
                "question": question,
                "gold_chunk_id": gold_id,
                "candidate_rank": rank,
                "candidate_chunk_id": str(chunk.get("chunk_id", "")),
                "candidate_section": str(chunk.get("section", "")),
                "retrieval_score": f"{float(score):.8f}",
                "label": int(str(chunk.get("chunk_id", "")) == gold_id),
                "gold_in_top_k": int(gold_id in ids),
            })
        if i % 25 == 0:
            print(f"Built selector candidates for {i}/{len(annotations)} annotations...")

    indices = list(range(len(annotations)))
    random.Random(args.seed).shuffle(indices)
    test_n = max(1, int(round(len(indices) * args.test_size)))
    test_idx = sorted(indices[:test_n])
    train_idx = sorted(indices[test_n:])

    def take(seq, idx):
        return [seq[i] for i in idx]

    train_questions = take(questions, train_idx)
    train_retrieved = take(retrieved_lists, train_idx)
    train_gold = take(gold_ids, train_idx)
    test_questions = take(questions, test_idx)
    test_retrieved = take(retrieved_lists, test_idx)
    test_gold = take(gold_ids, test_idx)

    X_train, y_train = build_matrix(train_questions, train_retrieved, train_gold, extractor)
    model = LogisticSelectorModel(lr=args.learning_rate, epochs=args.epochs, l2=args.l2, seed=args.seed).fit(X_train, y_train)
    metadata = {
        "annotations": str(args.annotations),
        "retrieval_method": args.method,
        "top_k": int(args.top_k),
        "hybrid_bm25_weight": float(args.hybrid_bm25_weight),
        "hybrid_dense_weight": float(args.hybrid_dense_weight),
        "train_examples": len(train_idx),
        "test_examples": len(test_idx),
        "seed": int(args.seed),
        "test_size": float(args.test_size),
    }
    selector = TrainedAnswerSelector(extractor, model, metadata)
    metrics = {
        "train": evaluate_selector(selector, train_questions, train_retrieved, train_gold),
        "test": evaluate_selector(selector, test_questions, test_retrieved, test_gold),
        "all_for_diagnostics_only": evaluate_selector(selector, questions, retrieved_lists, gold_ids),
        "metadata": metadata,
    }
    # Do not store metrics inside selector.metadata; this avoids a circular JSON reference.
    selector.save(args.out)

    write_csv(out_dir / "selector_candidates.csv", candidate_rows, ["id", "question", "gold_chunk_id", "candidate_rank", "candidate_chunk_id", "candidate_section", "retrieval_score", "label", "gold_in_top_k"])
    (out_dir / "selector_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for split, vals in metrics.items():
        if isinstance(vals, dict) and split != "metadata":
            row = {"split": split}
            row.update(vals)
            rows.append(row)
    write_csv(out_dir / "selector_metrics.csv", rows, list(rows[0].keys()))

    test = metrics["test"]
    print("=" * 80)
    print("Answer selector training summary")
    print("=" * 80)
    print(f"Model saved to: {args.out}")
    print(f"Train examples: {len(train_idx)}")
    print(f"Test examples: {len(test_idx)}")
    print("-" * 80)
    print(f"Held-out gold in candidates: {test['gold_in_candidates']:.4f}")
    print(f"Held-out original top1:      {test['original_top1']:.4f}")
    print(f"Held-out selector top1:      {test['selector_top1']:.4f}")
    print(f"Held-out original MRR:       {test['original_mrr']:.4f}")
    print(f"Held-out selector MRR:       {test['selector_mrr']:.4f}")
    print("=" * 80)
    print(f"Saved selector outputs to: {out_dir}")


if __name__ == "__main__":
    main()
