#!/usr/bin/env python3
from __future__ import annotations

"""Gradio demo for the UNSW Handbook RAG assistant.

This version keeps only the methods used in the final project narrative:
1. Original BM25 baseline
2. Hybrid retrieval
3. LangChain RAG wrapper
4. Supervised selector
5. FLAN-T5-base generative reader (experimental)

Run from the project root:
    $env:PYTHONPATH="src"
    python app.py
"""

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if SCRIPT_DIR.exists() and str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Gradio is not installed. Install it with: pip install gradio") from exc

try:
    from answer_with_qa import load_retrieval_resources, retrieve_for_answer
    from unsw_handbook.answer_selector import TrainedAnswerSelector
    from unsw_handbook.build_index import load_chunks
    from unsw_handbook.generative_reader import GenerativeQAReader
    from unsw_handbook.qa_reader import DEFAULT_QA_MODEL, ExtractiveQAReader
    from unsw_handbook.rag_pipeline import HandbookRAGPipeline, RAGConfig
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import project modules. Run from the project root and set PYTHONPATH=src.\n"
        f"Original error: {exc}"
    ) from exc


METHODS: Dict[str, Dict[str, Any]] = {
    "Original Baseline: BM25 + DistilBERT QA": {
        "kind": "pipeline",
        "method": "bm25",
        "bm25_weight": 1.0,
        "dense_weight": 0.0,
        "reader": "extractive",
    },
    "Hybrid retrieval: BM25 + Dense + DistilBERT QA": {
        "kind": "pipeline",
        "method": "hybrid",
        "bm25_weight": 0.6,
        "dense_weight": 0.4,
        "reader": "extractive",
    },
    "LangChain RAG wrapper: Hybrid + DistilBERT QA": {
        "kind": "langchain",
        "method": "hybrid",
        "bm25_weight": 0.6,
        "dense_weight": 0.4,
        "reader": "extractive",
    },
    "Supervised selector: Hybrid + Selector + DistilBERT QA": {
        "kind": "selector",
        "method": "hybrid",
        "bm25_weight": 0.6,
        "dense_weight": 0.4,
        "reader": "extractive",
    },
    "Experimental: Hybrid + Selector + FLAN-T5-base": {
        "kind": "selector",
        "method": "hybrid",
        "bm25_weight": 0.6,
        "dense_weight": 0.4,
        "reader": "generative",
    },
}

DEFAULT_METHOD = "Hybrid retrieval: BM25 + Dense + DistilBERT QA"

PRESET_QUESTIONS = [
    "What is COMP9021 about?",
    "How many units of credit is COMP9021 worth?",
    "Which terms is COMP9021 offered in?",
    "What is program 8543 about?",
    "How many units of credit are required for program 8543?",
    "What are the core requirements for program 8543?",
    "Where is program 7546 offered?",
]


class Args:
    """Small argparse-like object used by existing retrieval helper functions."""

    pass


@lru_cache(maxsize=12)
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
    """Cache RAG pipelines so models/indexes are not reloaded on every click."""
    config = RAGConfig(
        chunks_path=chunks_path,
        bm25_index_path=bm25_index_path,
        dense_index_path=dense_index_path,
        retrieval_method=method,
        qa_model_name=DEFAULT_QA_MODEL,
        device=-1,
        top_k=top_k,
        reader_top_n=reader_top_n,
        hybrid_bm25_weight=float(bm25_weight),
        hybrid_dense_weight=float(dense_weight),
    )
    return HandbookRAGPipeline(config)


@lru_cache(maxsize=4)
def load_selector_resources(
    chunks_path: str,
    bm25_index_path: str,
    dense_index_path: str,
    selector_model_path: str,
):
    """Load retrieval resources and the optional supervised selector."""
    chunks = load_chunks(chunks_path)
    args = Args()
    args.method = "hybrid"
    args.index = bm25_index_path
    args.dense_index = dense_index_path
    bm25_index, dense_index = load_retrieval_resources(args, chunks)
    selector = None
    if selector_model_path and Path(selector_model_path).exists():
        selector = TrainedAnswerSelector.load(selector_model_path)
    return chunks, bm25_index, dense_index, selector


@lru_cache(maxsize=2)
def get_extractive_reader() -> ExtractiveQAReader:
    return ExtractiveQAReader(model_name=DEFAULT_QA_MODEL, device=-1, fallback_to_evidence=True)


@lru_cache(maxsize=3)
def get_generative_reader(model_name: str) -> GenerativeQAReader:
    return GenerativeQAReader(
        model_name=model_name or "google/flan-t5-base",
        device=-1,
        fallback_to_evidence=True,
        max_new_tokens=64,
        num_beams=5,
    )


def _chunk_text(chunk: Dict[str, Any]) -> str:
    return str(chunk.get("chunk_text", "") or chunk.get("text", "") or chunk.get("content", ""))


def format_evidence_from_payload(evidence: List[Dict[str, Any]]) -> List[List[Any]]:
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


def format_evidence_from_retrieved(retrieved: List[Tuple[Dict[str, Any], float]]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for rank, (chunk, score) in enumerate(retrieved, start=1):
        text = " ".join(_chunk_text(chunk).split())
        rows.append(
            [
                rank,
                f"{float(score):.6f}",
                chunk.get("code", ""),
                chunk.get("title", ""),
                chunk.get("section", ""),
                chunk.get("chunk_id", ""),
                chunk.get("url", ""),
                text[:800],
            ]
        )
    return rows


def answer_with_pipeline(
    question: str,
    cfg: Dict[str, Any],
    top_k: int,
    reader_top_n: int,
    chunks_path: str,
    bm25_index_path: str,
    dense_index_path: str,
    use_langchain: bool = False,
) -> Dict[str, Any]:
    pipeline = get_pipeline(
        cfg["method"],
        float(cfg["bm25_weight"]),
        float(cfg["dense_weight"]),
        chunks_path,
        bm25_index_path,
        dense_index_path,
        top_k,
        reader_top_n,
    )
    if use_langchain:
        # This is the actual LangChain integration claimed in the scope. It wraps
        # the local retriever and QA reader through langchain-core's Runnable.
        runnable = pipeline.to_langchain_runnable()
        payload = runnable.invoke(question)
        payload["langchain_wrapper_used"] = 1
        return payload
    payload = pipeline.answer(question, top_k=top_k, reader_top_n=reader_top_n)
    payload["langchain_wrapper_used"] = 0
    return payload


def answer_with_selector(
    question: str,
    cfg: Dict[str, Any],
    top_k: int,
    reader_top_n: int,
    chunks_path: str,
    bm25_index_path: str,
    dense_index_path: str,
    selector_model_path: str,
    generative_model_name: str,
) -> Dict[str, Any]:
    chunks, bm25_index, dense_index, selector = load_selector_resources(
        chunks_path,
        bm25_index_path,
        dense_index_path,
        selector_model_path,
    )

    args = Args()
    args.question = question
    args.method = cfg["method"]
    args.index = bm25_index_path
    args.dense_index = dense_index_path
    args.dense_model_name = ""
    args.top_k = int(top_k)
    args.hybrid_pool_size = 50
    args.hybrid_rrf_k = 60.0
    args.hybrid_bm25_weight = float(cfg["bm25_weight"])
    args.hybrid_dense_weight = float(cfg["dense_weight"])
    args.strict_code_filter = True

    retrieved = retrieve_for_answer(args, chunks, bm25_index, dense_index)
    selector_used = selector is not None
    ranked = selector.rerank(question, retrieved) if selector_used else retrieved

    if cfg["reader"] == "generative":
        reader = get_generative_reader(generative_model_name or "google/flan-t5-base")
        # FLAN-T5 is experimental and tends to work best with a single selected evidence chunk.
        reader_input = ranked[:1]
        answer = reader.answer_from_chunks(question, reader_input, top_n_contexts=1)
        reader_type = "generative"
    else:
        reader = get_extractive_reader()
        reader_input = ranked[: max(1, int(reader_top_n))]
        answer = reader.answer_from_chunks(question, reader_input, top_n_contexts=max(1, int(reader_top_n)))
        reader_type = "extractive"

    evidence_rows = []
    for rank, (chunk, score) in enumerate(ranked, start=1):
        text = " ".join(_chunk_text(chunk).split())
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
                "preview": text[:800],
            }
        )

    return {
        "question": question,
        "retrieval_method": cfg["method"],
        "retrieval_top_k": int(top_k),
        "reader_top_n": int(reader_top_n),
        "hybrid_bm25_weight": float(cfg["bm25_weight"]),
        "hybrid_dense_weight": float(cfg["dense_weight"]),
        "selector_used": int(selector_used),
        "selector_model_path": selector_model_path,
        "reader_type": reader_type,
        "answer": answer.to_dict(),
        "retrieved_evidence": evidence_rows,
    }


def answer_question(
    question: str,
    method_label: str,
    top_k: int,
    reader_top_n: int,
    chunks_path: str,
    bm25_index_path: str,
    dense_index_path: str,
    selector_model_path: str,
    generative_model_name: str,
) -> Tuple[str, str, List[List[Any]], str]:
    question = str(question or "").strip()
    if not question:
        return "Please enter a question.", "", [], "{}"

    cfg = METHODS.get(method_label, METHODS[DEFAULT_METHOD])
    top_k = int(top_k or 5)
    reader_top_n = int(reader_top_n or 3)

    try:
        if cfg["kind"] == "selector":
            payload = answer_with_selector(
                question,
                cfg,
                top_k,
                reader_top_n,
                chunks_path,
                bm25_index_path,
                dense_index_path,
                selector_model_path,
                generative_model_name,
            )
        else:
            payload = answer_with_pipeline(
                question,
                cfg,
                top_k,
                reader_top_n,
                chunks_path,
                bm25_index_path,
                dense_index_path,
                use_langchain=(cfg["kind"] == "langchain"),
            )
    except Exception as exc:  # pragma: no cover - interactive demo guardrail
        err = {
            "error": str(exc),
            "hint": "Check data paths, model files, and optional dependencies such as langchain-core.",
        }
        return f"Error: {exc}", "", [], json.dumps(err, ensure_ascii=False, indent=2)

    answer = payload.get("answer", {})
    evidence = payload.get("retrieved_evidence", [])
    answer_text = str(answer.get("answer", ""))

    metadata = "\n".join(
        [
            f"Method: {method_label}",
            f"Retrieval method: {payload.get('retrieval_method', '')}",
            f"LangChain wrapper used: {payload.get('langchain_wrapper_used', 0)}",
            f"Selector used: {payload.get('selector_used', 0)}",
            f"Reader type: {payload.get('reader_type', 'extractive')}",
            f"QA / reader model: {answer.get('model_name', payload.get('qa_model_name', ''))}",
            f"Answer strategy: {answer.get('answer_strategy', '')}",
            f"QA score: {float(answer.get('qa_score', 0.0)):.4f}",
            f"Fallback used: {int(answer.get('fallback_used', 0))}",
            f"Source chunk: {answer.get('chunk_id', '')}",
            f"Source section: {answer.get('section', '')}",
            f"Source URL: {answer.get('url', '')}",
        ]
    )

    return answer_text, metadata, format_evidence_from_payload(evidence), json.dumps(payload, ensure_ascii=False, indent=2)


def set_question(q: str) -> str:
    return q


with gr.Blocks(title="UNSW Handbook RAG Assistant") as demo:
    gr.Markdown(
        """
        # UNSW Handbook RAG Assistant

        Use the preset questions below for quick testing. The demo compares the original BM25 baseline, hybrid retrieval, LangChain integration, the supervised selector, and the experimental FLAN-T5-base reader.
        """
    )

    question_box = gr.Textbox(
        label="Question",
        value="What is COMP9021 about?",
        placeholder="e.g. Which terms is COMP9021 offered in?",
        lines=3,
    )

    gr.Markdown("### Preset questions")
    with gr.Row():
        preset_buttons = [gr.Button(q, size="sm") for q in PRESET_QUESTIONS[:4]]
    with gr.Row():
        preset_buttons += [gr.Button(q, size="sm") for q in PRESET_QUESTIONS[4:]]
    for button, q in zip(preset_buttons, PRESET_QUESTIONS):
        button.click(lambda q=q: q, inputs=None, outputs=question_box)

    with gr.Row():
        with gr.Column(scale=2):
            method_dropdown = gr.Dropdown(
                choices=list(METHODS.keys()),
                value=DEFAULT_METHOD,
                label="Method",
            )
            with gr.Row():
                top_k_slider = gr.Slider(1, 10, value=5, step=1, label="Top-k retrieved chunks")
                reader_top_n_slider = gr.Slider(1, 5, value=3, step=1, label="Reader evidence chunks")
            submit_button = gr.Button("Ask", variant="primary")
        with gr.Column(scale=1):
            gr.Markdown("### Data and model paths")
            chunks_path_box = gr.Textbox(value="data/parsed/chunks.jsonl", label="Chunks path")
            bm25_index_path_box = gr.Textbox(value="data/index/bm25_index.json", label="BM25 index path")
            dense_index_path_box = gr.Textbox(value="data/index/dense_index.npz", label="Dense index path")
            selector_model_path_box = gr.Textbox(value="data/models/answer_selector.json", label="Selector model path")
            generative_model_name_box = gr.Textbox(value="google/flan-t5-base", label="Generative reader model")

    answer_box = gr.Textbox(label="Answer", lines=3)
    metadata_box = gr.Textbox(label="Model and source metadata", lines=10)
    evidence_table = gr.Dataframe(
        headers=["rank", "score", "code", "title", "section", "chunk_id", "url", "preview"],
        label="Top retrieved evidence",
        wrap=True,
        interactive=False,
    )
    raw_json_box = gr.Code(label="Raw JSON payload", language="json", lines=18)

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
            selector_model_path_box,
            generative_model_name_box,
        ],
        outputs=[answer_box, metadata_box, evidence_table, raw_json_box],
    )


if __name__ == "__main__":
    demo.launch()
