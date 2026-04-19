#!/usr/bin/env python3
"""Evaluate retrieval quality for the UNSW Handbook RAG project.

This script evaluates whether the retriever can return the annotated gold page/chunk
for each question in annotations.csv. It is designed for the current BM25 baseline,
and the structure is intentionally simple so that dense retrieval / QA evaluation can
be added later.

Example usage from the project root:

    # Windows PowerShell
    $env:PYTHONPATH="src"
    python scripts/evaluate_retrieval.py `
        --annotations ../MISC/annotations.csv `
        --chunks ../MISC/chunks.jsonl `
        --index ../MISC/bm25_index.json `
        --out-dir ../MISC/results/bm25 `
        --top-k 1 3 5

    # macOS / Linux
    PYTHONPATH=src python scripts/evaluate_retrieval.py \
        --annotations ../MISC/annotations.csv \
        --chunks ../MISC/chunks.jsonl \
        --index ../MISC/bm25_index.json \
        --out-dir ../MISC/results/bm25 \
        --top-k 1 3 5

If --index is not provided, the BM25 index is rebuilt from chunks.jsonl in memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

# Allow the script to be run as `python scripts/evaluate_retrieval.py` without
# requiring the user to manually install the package. This assumes the standard
# project layout: project_root/scripts/evaluate_retrieval.py and project_root/src/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from unsw_handbook.build_index import (
        bm25_score_query,
        build_bm25_index,
        load_bm25_index,
        load_chunks,
    )
except ImportError as exc:  # pragma: no cover - gives users a clearer error message
    raise SystemExit(
        "Could not import the project package. Run this script from the project root, "
        "or set PYTHONPATH=src before running it. Original error: " + str(exc)
    ) from exc


Annotation = Dict[str, str]
Chunk = Dict[str, Any]
Prediction = Dict[str, Any]


REQUIRED_ANNOTATION_COLUMNS = {
    "id",
    "question",
    "answer",
    "page_type",
    "target_code",
    "target_title",
    "source_url",
    "gold_page_id",
    "gold_chunk_id",
    "evidence_text",
    "annotator",
    "reviewer",
    "notes",
}


def read_annotations(path: str | Path) -> List[Annotation]:
    """Read annotations.csv and validate that the expected columns exist."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"annotations file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"annotations file has no header: {path}")

        actual_columns = set(reader.fieldnames)
        missing_columns = sorted(REQUIRED_ANNOTATION_COLUMNS - actual_columns)
        if missing_columns:
            raise ValueError(
                "annotations file is missing required columns: "
                + ", ".join(missing_columns)
            )

        rows: List[Annotation] = []
        for row in reader:
            clean_row = {k: (v or "").strip() for k, v in row.items()}
            if clean_row.get("id") or clean_row.get("question"):
                rows.append(clean_row)
    return rows


def ensure_out_dir(path: str | Path) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_csv(path: str | Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalise_top_k(values: Sequence[int]) -> List[int]:
    """Return sorted unique positive k values."""
    top_k = sorted({int(k) for k in values if int(k) > 0})
    if not top_k:
        raise ValueError("--top-k must contain at least one positive integer")
    return top_k


def build_chunk_lookup(chunks: Sequence[Chunk]) -> Dict[str, Chunk]:
    lookup: Dict[str, Chunk] = {}
    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id", ""))
        if chunk_id:
            lookup[chunk_id] = chunk
    return lookup


def retrieve_bm25(
    question: str,
    chunks: Sequence[Chunk],
    index: Dict[str, Any],
    limit: int,
) -> List[Tuple[Chunk, float]]:
    """Return top retrieved chunks and their BM25 scores."""
    ranked = bm25_score_query(question, list(chunks), index)[:limit]
    return [(chunks[i], float(score)) for i, score in ranked]


def classify_error(
    gold_page_id: str,
    gold_chunk_id: str,
    gold_section: str,
    gold_exists: bool,
    retrieved: Sequence[Tuple[Chunk, float]],
) -> str:
    """Classify the top-1 retrieval error for qualitative analysis."""
    if not gold_chunk_id:
        return "missing_gold_label"
    if not gold_exists:
        return "gold_chunk_not_found_in_corpus"
    if not retrieved:
        return "no_retrieval_result"

    top_chunk = retrieved[0][0]
    top_chunk_id = str(top_chunk.get("chunk_id", ""))
    top_page_id = str(top_chunk.get("page_id", ""))
    top_section = str(top_chunk.get("section", ""))

    if top_chunk_id == gold_chunk_id:
        return "correct_top1"
    if top_page_id != gold_page_id:
        return "wrong_page"
    if gold_section and top_section and top_section != gold_section:
        return "right_page_wrong_section"
    return "right_page_right_section_wrong_chunk"

def evaluate_one(
    ann: Annotation,
    retrieved: Sequence[Tuple[Chunk, float]],
    chunk_by_id: Dict[str, Chunk],
    top_k_values: Sequence[int],
) -> Prediction:
    """Evaluate one annotation row and return a rich prediction record."""
    gold_page_id = ann.get("gold_page_id", "")
    gold_chunk_id = ann.get("gold_chunk_id", "")
    gold_chunk = chunk_by_id.get(gold_chunk_id)
    gold_exists = gold_chunk is not None
    gold_section = str(gold_chunk.get("section", "")) if gold_chunk else ""

    retrieved_chunk_ids = [str(chunk.get("chunk_id", "")) for chunk, _ in retrieved]
    retrieved_page_ids = [str(chunk.get("page_id", "")) for chunk, _ in retrieved]
    retrieved_sections = [str(chunk.get("section", "")) for chunk, _ in retrieved]
    retrieved_scores = [float(score) for _, score in retrieved]

    gold_rank = 0
    if gold_chunk_id in retrieved_chunk_ids:
        gold_rank = retrieved_chunk_ids.index(gold_chunk_id) + 1

    gold_page_rank = 0
    if gold_page_id in retrieved_page_ids:
        gold_page_rank = retrieved_page_ids.index(gold_page_id) + 1

    top_chunk = retrieved[0][0] if retrieved else {}
    top_score = retrieved[0][1] if retrieved else 0.0

    record: Prediction = {
        "id": ann.get("id", ""),
        "question": ann.get("question", ""),
        "answer": ann.get("answer", ""),
        "page_type": ann.get("page_type", ""),
        "target_code": ann.get("target_code", ""),
        "target_title": ann.get("target_title", ""),
        "gold_page_id": gold_page_id,
        "gold_chunk_id": gold_chunk_id,
        "gold_section": gold_section,
        "gold_chunk_found_in_corpus": int(gold_exists),
        "gold_rank": gold_rank,
        "gold_page_rank": gold_page_rank,
        "mrr": (1.0 / gold_rank) if gold_rank else 0.0,
        "top1_chunk_id": str(top_chunk.get("chunk_id", "")),
        "top1_page_id": str(top_chunk.get("page_id", "")),
        "top1_section": str(top_chunk.get("section", "")),
        "top1_code": str(top_chunk.get("code", "")),
        "top1_title": str(top_chunk.get("title", "")),
        "top1_url": str(top_chunk.get("url", "")),
        "top1_score": f"{top_score:.6f}",
        "top1_text_preview": str(top_chunk.get("chunk_text", ""))[:500].replace("\n", " "),
        "retrieved_chunk_ids": " | ".join(retrieved_chunk_ids),
        "retrieved_page_ids": " | ".join(retrieved_page_ids),
        "retrieved_sections": " | ".join(retrieved_sections),
        "retrieved_scores": " | ".join(f"{score:.6f}" for score in retrieved_scores),
        "error_type": classify_error(gold_page_id, gold_chunk_id, gold_section, gold_exists, retrieved),
    }

    for k in top_k_values:
        top_chunk_ids_k = set(retrieved_chunk_ids[:k])
        top_page_ids_k = set(retrieved_page_ids[:k])
        record[f"chunk_hit@{k}"] = int(bool(gold_chunk_id) and gold_chunk_id in top_chunk_ids_k)
        record[f"page_hit@{k}"] = int(bool(gold_page_id) and gold_page_id in top_page_ids_k)

    # This is useful for diagnosing same-page section mistakes.
    record["section_hit@1"] = int(bool(gold_section) and record["top1_section"] == gold_section)
    return record


def aggregate_metrics(predictions: Sequence[Prediction], top_k_values: Sequence[int]) -> Dict[str, Any]:
    """Aggregate overall retrieval metrics."""
    n = len(predictions)
    if n == 0:
        raise ValueError("No predictions to aggregate")

    metrics: Dict[str, Any] = {
        "n_examples": n,
        "n_gold_chunks_found_in_corpus": sum(int(p["gold_chunk_found_in_corpus"]) for p in predictions),
        "mrr@max_k": sum(float(p["mrr"]) for p in predictions) / n,
        "section_hit@1": sum(int(p["section_hit@1"]) for p in predictions) / n,
    }
    for k in top_k_values:
        metrics[f"chunk_hit@{k}"] = sum(int(p[f"chunk_hit@{k}"]) for p in predictions) / n
        metrics[f"page_hit@{k}"] = sum(int(p[f"page_hit@{k}"]) for p in predictions) / n

    metrics["error_type_counts"] = dict(Counter(str(p["error_type"]) for p in predictions))
    return metrics


def aggregate_by_group(
    predictions: Sequence[Prediction],
    group_field: str,
    top_k_values: Sequence[int],
) -> List[Dict[str, Any]]:
    """Aggregate metrics grouped by page_type, gold_section, etc."""
    grouped: Dict[str, List[Prediction]] = defaultdict(list)
    for p in predictions:
        grouped[str(p.get(group_field, "") or "UNKNOWN")].append(p)

    rows: List[Dict[str, Any]] = []
    for group_value, rows_in_group in sorted(grouped.items(), key=lambda item: item[0]):
        n = len(rows_in_group)
        row: Dict[str, Any] = {
            group_field: group_value,
            "n_examples": n,
            "mrr@max_k": sum(float(p["mrr"]) for p in rows_in_group) / n,
            "section_hit@1": sum(int(p["section_hit@1"]) for p in rows_in_group) / n,
        }
        for k in top_k_values:
            row[f"chunk_hit@{k}"] = sum(int(p[f"chunk_hit@{k}"]) for p in rows_in_group) / n
            row[f"page_hit@{k}"] = sum(int(p[f"page_hit@{k}"]) for p in rows_in_group) / n
        rows.append(row)
    return rows


def format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_summary(metrics: Dict[str, Any], top_k_values: Sequence[int]) -> None:
    """Print a compact human-readable summary for the terminal."""
    print("=" * 80)
    print("Retrieval evaluation summary")
    print("=" * 80)
    print(f"Examples: {metrics['n_examples']}")
    print(f"Gold chunks found in corpus: {metrics['n_gold_chunks_found_in_corpus']}")
    print(f"MRR@{max(top_k_values)}: {metrics['mrr@max_k']:.4f}")
    print(f"Section Hit@1: {metrics['section_hit@1']:.4f}")
    print("-" * 80)
    for k in top_k_values:
        print(f"Chunk Hit@{k}: {metrics[f'chunk_hit@{k}']:.4f}")
    for k in top_k_values:
        print(f"Page  Hit@{k}: {metrics[f'page_hit@{k}']:.4f}")
    print("-" * 80)
    print("Error types:")
    for error_type, count in sorted(metrics["error_type_counts"].items(), key=lambda x: (-x[1], x[0])):
        print(f"  {error_type}: {count}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate BM25 retrieval against annotated gold Handbook evidence."
    )
    parser.add_argument(
        "--annotations",
        default="annotations.csv",
        help="Path to annotations.csv.",
    )
    parser.add_argument(
        "--chunks",
        default="data/parsed/chunks.jsonl",
        help="Path to chunks.jsonl.",
    )
    parser.add_argument(
        "--index",
        default="",
        help="Optional path to bm25_index.json. If omitted, the index is rebuilt from chunks.",
    )
    parser.add_argument(
        "--method",
        default="bm25",
        choices=["bm25"],
        help="Retrieval method to evaluate. Currently supports bm25; dense can be added later.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="One or more k values for Hit@k, e.g. --top-k 1 3 5.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/results/bm25",
        help="Directory where metrics and error files will be written.",
    )
    args = parser.parse_args()

    top_k_values = normalise_top_k(args.top_k)
    retrieve_limit = max(top_k_values)
    out_dir = ensure_out_dir(args.out_dir)

    annotations = read_annotations(args.annotations)
    chunks = load_chunks(args.chunks)
    chunk_by_id = build_chunk_lookup(chunks)

    if not chunks:
        raise ValueError(f"No chunks loaded from {args.chunks}")

    if args.index:
        index = load_bm25_index(args.index)
    else:
        index = build_bm25_index(chunks)

    predictions: List[Prediction] = []
    for ann in annotations:
        question = ann.get("question", "")
        retrieved = retrieve_bm25(question, chunks, index, retrieve_limit)
        predictions.append(evaluate_one(ann, retrieved, chunk_by_id, top_k_values))

    metrics = aggregate_metrics(predictions, top_k_values)
    by_section = aggregate_by_group(predictions, "gold_section", top_k_values)
    by_page_type = aggregate_by_group(predictions, "page_type", top_k_values)
    by_error_type = [
        {"error_type": error_type, "count": count}
        for error_type, count in sorted(
            Counter(str(p["error_type"]) for p in predictions).items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]

    prediction_fields = [
        "id",
        "question",
        "answer",
        "page_type",
        "target_code",
        "target_title",
        "gold_page_id",
        "gold_chunk_id",
        "gold_section",
        "gold_chunk_found_in_corpus",
        "gold_rank",
        "gold_page_rank",
        "mrr",
        "top1_chunk_id",
        "top1_page_id",
        "top1_section",
        "top1_code",
        "top1_title",
        "top1_url",
        "top1_score",
        "retrieved_chunk_ids",
        "retrieved_page_ids",
        "retrieved_sections",
        "retrieved_scores",
        "section_hit@1",
        *[f"chunk_hit@{k}" for k in top_k_values],
        *[f"page_hit@{k}" for k in top_k_values],
        "error_type",
        "top1_text_preview",
    ]

    metric_fields = ["metric", "value"]
    overall_metric_rows = []
    for key, value in metrics.items():
        if key == "error_type_counts":
            continue
        overall_metric_rows.append({"metric": key, "value": format_metric(value)})

    grouped_fields = [
        "n_examples",
        "mrr@max_k",
        "section_hit@1",
        *[f"chunk_hit@{k}" for k in top_k_values],
        *[f"page_hit@{k}" for k in top_k_values],
    ]

    error_cases = [p for p in predictions if p["error_type"] != "correct_top1"]

    write_json(out_dir / "overall_metrics.json", metrics)
    write_csv(out_dir / "overall_metrics.csv", overall_metric_rows, metric_fields)
    write_csv(out_dir / "predictions.csv", predictions, prediction_fields)
    write_csv(out_dir / "error_cases.csv", error_cases, prediction_fields)
    write_csv(out_dir / "metrics_by_gold_section.csv", by_section, ["gold_section", *grouped_fields])
    write_csv(out_dir / "metrics_by_page_type.csv", by_page_type, ["page_type", *grouped_fields])
    write_csv(out_dir / "error_type_counts.csv", by_error_type, ["error_type", "count"])

    print_summary(metrics, top_k_values)
    print(f"Saved results to: {out_dir}")
    print("Generated files:")
    for name in [
        "overall_metrics.json",
        "overall_metrics.csv",
        "predictions.csv",
        "error_cases.csv",
        "metrics_by_gold_section.csv",
        "metrics_by_page_type.csv",
        "error_type_counts.csv",
    ]:
        print(f"  - {out_dir / name}")


if __name__ == "__main__":
    main()
