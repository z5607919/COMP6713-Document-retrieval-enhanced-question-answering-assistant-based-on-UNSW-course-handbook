#!/usr/bin/env python3
from __future__ import annotations

"""Gradio demo for the UNSW Handbook RAG project.

Run from the project root:
    set PYTHONPATH=src   # PowerShell: $env:PYTHONPATH="src"
    python app.py

The demo reuses the same RAG pipeline used by the CLI scripts. It supports multiple
retrieval modes so assessors can compare the BM25 baseline, dense retrieval, hybrid
retrieval, and the final RAG answer pipeline from one web page.
"""

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - user environment dependent
    raise SystemExit(
        "Gradio is not installed. Install it with: pip install gradio\n"
        f"Original error: {exc}"
    ) from exc

try:
    from unsw_handbook.qa_reader import DEFAULT_QA_MODEL
    from unsw_handbook.rag_pipeline import HandbookRAGPipeline, RAGConfig
except ImportError as exc:  # pragma: no cover - user environment dependent
    raise SystemExit(
        "Could not import project modules. Run from the project root and set PYTHONPATH=src.\n"
        f"Original error: {exc}"
    ) from exc


METHODS: Dict[str, Dict[str, Any]] = {
    "Final RAG: Hybrid retrieval + QA reader (BM25 0.6 / Dense 0.4)": {
        "method": "hybrid",
        "bm25_weight": 0.6,
        "dense_weight": 0.4,
    },
    "BM25 baseline + QA reader": {
        "method": "bm25",
        "bm25_weight": 1.0,
        "dense_weight": 0.0,
    },
    "Dense retrieval + QA reader": {
        "method": "dense",
        "bm25_weight": 0.0,
        "dense_weight": 1.0,
    },
    "Code-constrained dense retrieval + QA reader": {
        "method": "dense_code",
        "bm25_weight": 0.0,
        "dense_weight": 1.0,
    },
    "Hybrid retrieval + QA reader (BM25 0.7 / Dense 0.3)": {
        "method": "hybrid",
        "bm25_weight": 0.7,
        "dense_weight": 0.3,
    },
}

EXAMPLES = [
    ["How many units of credit is COMP6714 worth?", "Final RAG: Hybrid retrieval + QA reader (BM25 0.6 / Dense 0.4)"],
    ["Which terms is COMP6714 offered in?", "Final RAG: Hybrid retrieval + QA reader (BM25 0.6 / Dense 0.4)"],
    ["What is COMP6714 about?", "Final RAG: Hybrid retrieval + QA reader (BM25 0.6 / Dense 0.4)"],
]


@lru_cache(maxsize=8)
def get_pipeline(
    method: str,
    bm25_weight: float,
    dense_weight: float,
    chunks_path: str,
    bm25_index_path: str,
    dense_index_path: str,
    top_k: int,
    reader_top_n: int,
) -> HandbookRAGPipeline:
    """Cache one pipeline per retrieval configuration to avoid reloading models."""
    config = RAGConfig(
        chunks_path=chunks_path,
        bm25_index_path=bm25_index_path,
        dense_index_path=dense_index_path,
        retrieval_method=method,
        qa_model_name=DEFAULT_QA_MODEL,
        device=-1,
        top_k=top_k,
        reader_top_n=reader_top_n,
        hybrid_bm25_weight=bm25_weight,
        hybrid_dense_weight=dense_weight,
    )
    return HandbookRAGPipeline(config)


def format_evidence(evidence: List[Dict[str, Any]]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for item in evidence:
        rows.append(
            [
                item.get("rank", ""),
                f"{float(item.get('score', 0.0)):.6f}",
                item.get("code", ""),
                item.get("title", ""),
                item.get("section", ""),
                item.get("chunk_id", ""),
                item.get("url", ""),
                item.get("preview", ""),
            ]
        )
    return rows


def answer_question(
    question: str,
    method_label: str,
    top_k: int,
    reader_top_n: int,
    chunks_path: str,
    bm25_index_path: str,
    dense_index_path: str,
) -> Tuple[str, str, List[List[Any]], str]:
    question = str(question or "").strip()
    if not question:
        return "Please enter a question.", "", [], "{}"

    method_cfg = METHODS.get(method_label, METHODS["Final RAG: Hybrid retrieval + QA reader (BM25 0.6 / Dense 0.4)"])
    top_k = int(top_k or 5)
    reader_top_n = int(reader_top_n or 3)

    try:
        pipeline = get_pipeline(
            method_cfg["method"],
            float(method_cfg["bm25_weight"]),
            float(method_cfg["dense_weight"]),
            chunks_path,
            bm25_index_path,
            dense_index_path,
            top_k,
            reader_top_n,
        )
        payload = pipeline.answer(question, top_k=top_k, reader_top_n=reader_top_n)
    except Exception as exc:  # pragma: no cover - interactive demo guardrail
        return f"Error: {exc}", "", [], json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2)

    answer = payload.get("answer", {})
    evidence = payload.get("retrieved_evidence", [])
    answer_text = str(answer.get("answer", ""))
    metadata = "\n".join(
        [
            f"Retrieval method: {payload.get('retrieval_method', '')}",
            f"QA model: {answer.get('model_name', payload.get('qa_model_name', ''))}",
            f"Answer strategy: {answer.get('answer_strategy', '')}",
            f"QA score: {float(answer.get('qa_score', 0.0)):.4f}",
            f"Source chunk: {answer.get('chunk_id', '')}",
            f"Source section: {answer.get('section', '')}",
            f"Source URL: {answer.get('url', '')}",
        ]
    )
    return answer_text, metadata, format_evidence(evidence), json.dumps(payload, ensure_ascii=False, indent=2)


with gr.Blocks(title="UNSW Handbook RAG Assistant") as demo:
    gr.Markdown(
        """
        # UNSW Handbook RAG Assistant

        This demo compares the BM25 baseline, dense retrieval, hybrid retrieval, and the final RAG pipeline.  
        The system retrieves evidence chunks from the UNSW Handbook corpus and uses a pre-trained extractive QA model to produce a short answer.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            question_box = gr.Textbox(
                label="Question",
                placeholder="e.g. How many units of credit is COMP6714 worth?",
                lines=3,
            )
            method_dropdown = gr.Dropdown(
                choices=list(METHODS.keys()),
                value="Final RAG: Hybrid retrieval + QA reader (BM25 0.6 / Dense 0.4)",
                label="Retrieval / RAG method",
            )
            with gr.Row():
                top_k_slider = gr.Slider(1, 10, value=5, step=1, label="Top-k retrieved chunks")
                reader_top_n_slider = gr.Slider(1, 5, value=3, step=1, label="QA reader contexts")
            submit_button = gr.Button("Ask", variant="primary")
        with gr.Column(scale=1):
            gr.Markdown("### Data paths")
            chunks_path_box = gr.Textbox(value="data/parsed/chunks.jsonl", label="Chunks path")
            bm25_index_path_box = gr.Textbox(value="data/index/bm25_index.json", label="BM25 index path")
            dense_index_path_box = gr.Textbox(value="data/index/dense_index.npz", label="Dense index path")

    answer_box = gr.Textbox(label="Answer", lines=3)
    metadata_box = gr.Textbox(label="Model and source metadata", lines=8)
    evidence_table = gr.Dataframe(
        headers=["rank", "score", "code", "title", "section", "chunk_id", "url", "preview"],
        label="Top retrieved evidence",
        wrap=True,
        interactive=False,
    )
    raw_json_box = gr.Code(label="Raw JSON payload", language="json", lines=18)

    gr.Examples(
        examples=EXAMPLES,
        inputs=[question_box, method_dropdown],
    )

    submit_button.click(
        answer_question,
        inputs=[
            question_box,
            method_dropdown,
            top_k_slider,
            reader_top_n_slider,
            chunks_path_box,
            bm25_index_path_box,
            dense_index_path_box,
        ],
        outputs=[answer_box, metadata_box, evidence_table, raw_json_box],
    )


if __name__ == "__main__":
    demo.launch()
