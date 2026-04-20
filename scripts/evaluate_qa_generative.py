#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate retrieval -> optional selector -> generative QA reader."""

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

from answer_with_qa import load_retrieval_resources, retrieve_for_answer
from evaluate_retrieval import read_annotations
from unsw_handbook.answer_selector import TrainedAnswerSelector
from unsw_handbook.build_index import load_chunks
from unsw_handbook.generative_reader import DEFAULT_GENERATIVE_MODEL, GenerativeQAReader


def norm(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return re.sub(r"\s+", " ", text).strip()


def token_f1(pred: str, gold: str) -> float:
    p = norm(pred).split()
    g = norm(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(p)
    recall = same / len(g)
    return 2 * precision * recall / (precision + recall)


def write_csv(path: str | Path, rows: Iterable[Dict[str, Any]], fields: List[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate generative selector-RAG.")
    p.add_argument("--annotations", default="annotations.csv")
    p.add_argument("--chunks", default="data/parsed/chunks.jsonl")
    p.add_argument("--method", default="hybrid", choices=["bm25", "dense", "dense_code", "hybrid"])
    p.add_argument("--index", default="data/index/bm25_index.json")
    p.add_argument("--dense-index", default="data/index/dense_index.npz")
    p.add_argument("--dense-model-name", default="")
    p.add_argument("--selector-model", default="data/models/answer_selector.json")
    p.add_argument("--no-selector", action="store_true")
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
    p.add_argument("--max-examples", type=int, default=0)
    p.add_argument("--out-dir", default="data/results/qa_hybrid_selector_generative")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotations = read_annotations(args.annotations)
    if args.max_examples and args.max_examples > 0:
        annotations = annotations[: args.max_examples]

    chunks = load_chunks(args.chunks)
    bm25_index, dense_index = load_retrieval_resources(args, chunks)
    selector = None if args.no_selector else TrainedAnswerSelector.load(args.selector_model)
    reader = GenerativeQAReader(
        model_name=args.generative_model_name,
        device=args.device,
        max_context_chars=args.max_context_chars,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        fallback_to_evidence=True,
    )

    rows: List[Dict[str, Any]] = []
    for i, ann in enumerate(annotations, start=1):
        question = ann.get("question", "")
        args.question = question
        retrieved = retrieve_for_answer(args, chunks, bm25_index, dense_index)
        raw_ids = [str(c.get("chunk_id", "")) for c, _ in retrieved]
        reranked = selector.rerank(question, retrieved) if selector is not None else retrieved
        selected = reranked[: max(1, int(args.reader_top_n))]
        answer = reader.answer_from_chunks(question, selected, top_n_contexts=args.reader_top_n)
        pred = answer.answer
        gold = ann.get("answer", "")
        n_pred = norm(pred)
        n_gold = norm(gold)
        selected_id = str(selected[0][0].get("chunk_id", "")) if selected else ""
        rows.append({
            "id": ann.get("id", ""),
            "question": question,
            "gold_answer": gold,
            "predicted_answer": pred,
            "exact_match": int(n_pred == n_gold),
            "token_f1": f"{token_f1(pred, gold):.6f}",
            "contains_gold": int(bool(n_gold) and n_gold in n_pred),
            "gold_contains_prediction": int(bool(n_pred) and n_pred in n_gold),
            "gold_chunk_id": ann.get("gold_chunk_id", ""),
            "gold_in_retrieved_topk": int(ann.get("gold_chunk_id", "") in raw_ids),
            "selected_chunk_id": selected_id,
            "selector_correct_top1": int(selector is not None and selected_id == ann.get("gold_chunk_id", "")),
            "source_chunk_id": answer.chunk_id,
            "source_section": answer.section,
            "qa_score": f"{answer.qa_score:.6f}",
            "answer_strategy": answer.answer_strategy,
            "used_pretrained_model": int(answer.used_pretrained_model),
            "fallback_used": int(answer.fallback_used),
            "warning": answer.warning,
        })
        if i % 25 == 0:
            print(f"Processed {i}/{len(annotations)} examples...")

    n = max(1, len(rows))
    metrics = {
        "method": args.method,
        "selector_used": int(selector is not None),
        "selector_model": "" if selector is None else args.selector_model,
        "generative_model_name": args.generative_model_name,
        "n_examples": len(rows),
        "exact_match": sum(r["exact_match"] for r in rows) / n,
        "average_token_f1": sum(float(r["token_f1"]) for r in rows) / n,
        "contains_gold": sum(r["contains_gold"] for r in rows) / n,
        "gold_contains_prediction": sum(r["gold_contains_prediction"] for r in rows) / n,
        "gold_in_retrieved_topk": sum(r["gold_in_retrieved_topk"] for r in rows) / n,
        "selector_top1_accuracy": sum(r["selector_correct_top1"] for r in rows) / n if selector is not None else 0.0,
        "pretrained_model_usage_rate": sum(r["used_pretrained_model"] for r in rows) / n,
        "fallback_rate": sum(r["fallback_used"] for r in rows) / n,
    }

    fields = list(rows[0].keys()) if rows else []
    write_csv(out_dir / "qa_generative_predictions.csv", rows, fields)
    write_csv(out_dir / "qa_generative_metrics.csv", [{"metric": k, "value": v} for k, v in metrics.items()], ["metric", "value"])
    (out_dir / "qa_generative_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    label = "hybrid + selector + generative QA" if selector is not None else "hybrid + generative QA"
    print("=" * 80)
    print(f"QA evaluation summary ({label})")
    print("=" * 80)
    print(f"Examples: {metrics['n_examples']}")
    if selector is not None:
        print(f"Selector top1 accuracy: {metrics['selector_top1_accuracy']:.4f}")
    print(f"Exact Match: {metrics['exact_match']:.4f}")
    print(f"Average token F1: {metrics['average_token_f1']:.4f}")
    print(f"Contains gold answer: {metrics['contains_gold']:.4f}")
    print(f"Gold contains prediction: {metrics['gold_contains_prediction']:.4f}")
    print(f"Gold in retrieved top-{args.top_k}: {metrics['gold_in_retrieved_topk']:.4f}")
    print(f"Pretrained model usage rate: {metrics['pretrained_model_usage_rate']:.4f}")
    print(f"Fallback rate: {metrics['fallback_rate']:.4f}")
    print("=" * 80)
    print(f"Saved generative QA results to: {out_dir}")


if __name__ == "__main__":
    main()
