from __future__ import annotations

"""Reusable RAG pipeline for the UNSW Handbook QA project.

This module wraps the project components that were developed separately:

1. a retriever (BM25 / dense / hybrid),
2. an evidence reader based on a pre-trained extractive QA model, and
3. a small orchestration layer that returns the answer together with evidence.

The class below is intentionally lightweight. It keeps all model/index resources in
memory so that the CLI demo or Gradio demo can answer multiple questions without
reloading indexes and models each time.

Optional LangChain integration is provided through ``to_langchain_runnable`` and
``to_langchain_tool``. The default pipeline does not require LangChain, but these
methods make the system compatible with an external orchestration library when
``langchain-core`` is installed.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# Make this module work when the project is run from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if SCRIPTS_DIR.exists() and str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from evaluate_retrieval import retrieve_with_method
    from unsw_handbook.build_index import build_bm25_index, load_bm25_index, load_chunks
    from unsw_handbook.dense_index import DEFAULT_DENSE_MODEL, load_dense_index
    from unsw_handbook.qa_reader import DEFAULT_QA_MODEL, ExtractiveQAReader
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "Could not import project modules. Run from the project root and set PYTHONPATH=src. "
        f"Original error: {exc}"
    ) from exc


Chunk = Dict[str, Any]
Retrieved = List[Tuple[Chunk, float]]


@dataclass
class RAGConfig:
    """Configuration for the Handbook RAG pipeline."""

    chunks_path: str = "data/parsed/chunks.jsonl"
    bm25_index_path: str = "data/index/bm25_index.json"
    dense_index_path: str = "data/index/dense_index.npz"
    retrieval_method: str = "hybrid"
    dense_model_name: str = ""
    qa_model_name: str = DEFAULT_QA_MODEL
    device: int = -1
    top_k: int = 5
    reader_top_n: int = 3
    max_context_chars: int = 3000
    min_answer_score: float = 0.0
    hybrid_pool_size: int = 50
    hybrid_rrf_k: float = 60.0
    hybrid_bm25_weight: float = 0.6
    hybrid_dense_weight: float = 0.4


class HandbookRAGPipeline:
    """End-to-end retrieval-augmented question answering pipeline."""

    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig()
        self.chunks: List[Chunk] = load_chunks(self.config.chunks_path)
        if not self.chunks:
            raise ValueError(f"No chunks were loaded from {self.config.chunks_path}")

        method = self.config.retrieval_method
        if method not in {"bm25", "dense", "dense_code", "hybrid"}:
            raise ValueError(f"Unsupported retrieval method: {method}")

        self.bm25_index: Dict[str, Any] | None = None
        self.dense_index: Dict[str, Any] | None = None

        if method in {"bm25", "hybrid"}:
            if self.config.bm25_index_path:
                self.bm25_index = load_bm25_index(self.config.bm25_index_path)
            else:
                self.bm25_index = build_bm25_index(self.chunks)

        if method in {"dense", "dense_code", "hybrid"}:
            self.dense_index = load_dense_index(self.config.dense_index_path)

        self.reader = ExtractiveQAReader(
            model_name=self.config.qa_model_name,
            device=self.config.device,
            max_context_chars=self.config.max_context_chars,
            min_answer_score=self.config.min_answer_score,
            fallback_to_evidence=True,
        )

    def retrieve(self, question: str, top_k: int | None = None) -> Retrieved:
        """Retrieve evidence chunks for one user question."""
        limit = max(1, int(top_k or self.config.top_k))
        return retrieve_with_method(
            method=self.config.retrieval_method,
            question=question,
            chunks=self.chunks,
            limit=limit,
            bm25_index=self.bm25_index,
            dense_index=self.dense_index,
            dense_model_name=(self.config.dense_model_name or None),
            hybrid_pool_size=self.config.hybrid_pool_size,
            hybrid_rrf_k=self.config.hybrid_rrf_k,
            hybrid_bm25_weight=self.config.hybrid_bm25_weight,
            hybrid_dense_weight=self.config.hybrid_dense_weight,
        )

    def answer(self, question: str, top_k: int | None = None, reader_top_n: int | None = None) -> Dict[str, Any]:
        """Answer one question and return answer, model metadata and evidence."""
        question = str(question or "").strip()
        if not question:
            raise ValueError("Question is empty.")

        retrieved = self.retrieve(question, top_k=top_k)
        answer = self.reader.answer_from_chunks(
            question,
            retrieved,
            top_n_contexts=max(1, int(reader_top_n or self.config.reader_top_n)),
        )

        evidence_rows: List[Dict[str, Any]] = []
        for rank, (chunk, score) in enumerate(retrieved, start=1):
            text = str(chunk.get("chunk_text", "") or chunk.get("text", "") or chunk.get("content", ""))
            text = " ".join(text.split())
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
                    "preview": text[:700],
                }
            )

        return {
            "question": question,
            "retrieval_method": self.config.retrieval_method,
            "retrieval_top_k": int(top_k or self.config.top_k),
            "reader_top_n": int(reader_top_n or self.config.reader_top_n),
            "dense_model_name": self.config.dense_model_name or self._saved_dense_model_name(),
            "qa_model_name": self.config.qa_model_name,
            "hybrid_bm25_weight": self.config.hybrid_bm25_weight,
            "hybrid_dense_weight": self.config.hybrid_dense_weight,
            "answer": answer.to_dict(),
            "retrieved_evidence": evidence_rows,
        }

    def answer_text(self, question: str) -> str:
        """Return a compact answer string, useful for LangChain tool calls."""
        payload = self.answer(question)
        answer = payload["answer"]
        evidence = payload.get("retrieved_evidence", [])
        source = evidence[0] if evidence else {}
        lines = [
            str(answer.get("answer", "")),
            "",
            f"Source: {source.get('title', '')} {source.get('code', '')}".strip(),
            f"Section: {source.get('section', '')}",
            f"URL: {source.get('url', '')}",
        ]
        return "\n".join(line for line in lines if line is not None).strip()

    def to_langchain_runnable(self):
        """Return a LangChain Runnable that maps question text to a RAG payload.

        This is optional and only requires ``langchain-core`` when called. It is a
        thin integration layer around the local retrieval and QA tools.
        """
        try:
            from langchain_core.runnables import RunnableLambda
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "LangChain integration requires langchain-core. Install it with:\n"
                "    pip install langchain-core"
            ) from exc

        return RunnableLambda(lambda question: self.answer(str(question)))

    def to_langchain_tool(self):
        """Return a LangChain Tool for answering UNSW Handbook questions."""
        try:
            from langchain_core.tools import tool
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "LangChain tool integration requires langchain-core. Install it with:\n"
                "    pip install langchain-core"
            ) from exc

        pipeline = self

        @tool("unsw_handbook_rag")
        def unsw_handbook_rag(question: str) -> str:
            """Answer factual questions about UNSW Handbook course/program pages with evidence."""
            return pipeline.answer_text(question)

        return unsw_handbook_rag

    def _saved_dense_model_name(self) -> str:
        if isinstance(self.dense_index, dict):
            return str(self.dense_index.get("metadata", {}).get("model_name", ""))
        return DEFAULT_DENSE_MODEL


def save_payload(path: str | Path, payload: Dict[str, Any]) -> None:
    """Save a RAG output payload as UTF-8 JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
