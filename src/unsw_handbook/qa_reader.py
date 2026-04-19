from __future__ import annotations

"""Extractive QA reader for the UNSW Handbook RAG project.

This module uses a pre-trained Hugging Face extractive question-answering model
without relying on ``transformers.pipeline``. Some recent Transformers builds may
not expose the string task name ``question-answering`` through pipeline(), even
though the underlying AutoModelForQuestionAnswering class is available. Loading
the model directly makes the project more robust across environments.
"""

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Sequence, Tuple


DEFAULT_QA_MODEL = "distilbert-base-cased-distilled-squad"


@dataclass
class QAResult:
    """Structured output from the extractive QA reader."""

    question: str
    answer: str
    qa_score: float
    retrieval_score: float
    combined_score: float
    source_rank: int
    chunk_id: str
    page_id: str
    code: str
    title: str
    section: str
    url: str
    evidence_text: str
    model_name: str
    used_pretrained_model: bool
    fallback_used: bool
    warning: str = ""
    answer_strategy: str = "extractive_qa"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "qa_score": self.qa_score,
            "retrieval_score": self.retrieval_score,
            "combined_score": self.combined_score,
            "source_rank": self.source_rank,
            "chunk_id": self.chunk_id,
            "page_id": self.page_id,
            "code": self.code,
            "title": self.title,
            "section": self.section,
            "url": self.url,
            "evidence_text": self.evidence_text,
            "model_name": self.model_name,
            "used_pretrained_model": int(self.used_pretrained_model),
            "fallback_used": int(self.fallback_used),
            "warning": self.warning,
            "answer_strategy": self.answer_strategy,
        }


@lru_cache(maxsize=4)
def _load_qa_backend(model_name: str, device: int):
    """Load tokenizer/model once and reuse them.

    Direct AutoModel loading avoids Transformers pipeline task-registry issues.
    """
    try:
        import torch
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Extractive QA requires transformers and torch. Install them with:\n"
            "    pip install transformers torch\n"
            "or add them to requirements.txt."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.eval()

    if int(device) >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError("A GPU device was requested, but CUDA is not available.")
        model.to(f"cuda:{int(device)}")
    return tokenizer, model, torch


def clean_whitespace(text: str) -> str:
    """Collapse repeated whitespace while preserving readable text."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def get_chunk_text(chunk: Dict[str, Any]) -> str:
    """Return the best available evidence text from a chunk dictionary."""
    for key in ("chunk_text", "text", "content", "body"):
        value = clean_whitespace(str(chunk.get(key, "")))
        if value:
            return value
    return ""


def _truncate_context(text: str, max_chars: int) -> str:
    """Keep context short enough for a small extractive QA model."""
    text = clean_whitespace(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0]


def _fallback_answer(question: str, retrieved: Sequence[Tuple[Dict[str, Any], float]], warning: str) -> QAResult:
    """Return a safe evidence-based fallback answer when the QA model is unavailable."""
    top_chunk: Dict[str, Any] = retrieved[0][0] if retrieved else {}
    retrieval_score = float(retrieved[0][1]) if retrieved else 0.0
    evidence = get_chunk_text(top_chunk)
    first_sentence = re.split(r"(?<=[.!?])\s+", evidence)[0] if evidence else ""
    answer = first_sentence[:300].strip() or "No answer could be extracted from the retrieved evidence."
    return QAResult(
        question=question,
        answer=answer,
        qa_score=0.0,
        retrieval_score=retrieval_score,
        combined_score=retrieval_score,
        source_rank=1 if retrieved else 0,
        chunk_id=str(top_chunk.get("chunk_id", "")),
        page_id=str(top_chunk.get("page_id", "")),
        code=str(top_chunk.get("code", "")),
        title=str(top_chunk.get("title", "")),
        section=str(top_chunk.get("section", "")),
        url=str(top_chunk.get("url", "")),
        evidence_text=evidence,
        model_name="fallback-evidence-first-sentence",
        used_pretrained_model=False,
        fallback_used=True,
        warning=warning,
        answer_strategy="fallback_evidence_first_sentence",
    )


def _extract_simple_fact_answer(question: str, context: str) -> str:
    """Small Handbook-specific post-processing for structured facts.

    This is not used to replace the QA model. It only normalises very obvious
    key-value fields after the model has loaded, so the final answer is cleaner
    for handbook questions such as UOC/units-of-credit queries.
    """
    q = question.lower()
    context = clean_whitespace(context)

    if any(term in q for term in ["unit of credit", "units of credit", "uoc", "worth"]):
        patterns = [
            r"\bUoc:\s*(\d+)\s*Units?\s+of\s+Credit\b",
            r"\b(\d+)\s*Units?\s+of\s+Credit\b",
            r"\bUOC\s*[:=-]?\s*(\d+)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, context, flags=re.IGNORECASE)
            if match:
                number = match.group(1)
                return f"{number} Units of Credit"

    return ""


def _run_direct_qa_model(
    question: str,
    context: str,
    model_name: str,
    device: int,
    max_answer_tokens: int = 30,
) -> Tuple[str, float]:
    """Run an extractive QA model and return (answer, approximate_score)."""
    tokenizer, model, torch = _load_qa_backend(model_name, int(device))

    encoded = tokenizer(
        question,
        context,
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    offset_mapping = encoded.pop("offset_mapping")[0].tolist()
    sequence_ids = encoded.sequence_ids(0)

    model_device = next(model.parameters()).device
    encoded = {key: value.to(model_device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)

    start_logits = outputs.start_logits[0].detach().cpu()
    end_logits = outputs.end_logits[0].detach().cpu()

    # Only allow answer spans inside the context, not the question or special tokens.
    context_positions = [i for i, seq_id in enumerate(sequence_ids) if seq_id == 1]
    if not context_positions:
        return "", 0.0

    start_probs = torch.softmax(start_logits, dim=-1)
    end_probs = torch.softmax(end_logits, dim=-1)

    # Search top starts/ends instead of all tokens for efficiency and stability.
    top_starts = torch.topk(start_probs, k=min(20, len(start_probs))).indices.tolist()
    top_ends = torch.topk(end_probs, k=min(20, len(end_probs))).indices.tolist()
    context_set = set(context_positions)

    best = None
    for s in top_starts:
        if s not in context_set:
            continue
        for e in top_ends:
            if e not in context_set:
                continue
            if e < s:
                continue
            if e - s + 1 > max_answer_tokens:
                continue
            start_char = int(offset_mapping[s][0])
            end_char = int(offset_mapping[e][1])
            if end_char <= start_char:
                continue
            answer = clean_whitespace(context[start_char:end_char])
            if not answer:
                continue
            score = float(start_probs[s] * end_probs[e])
            if best is None or score > best[1]:
                best = (answer, score)

    if best is None:
        return "", 0.0
    return best


class ExtractiveQAReader:
    """Use a pre-trained extractive QA model over retrieved chunks."""

    def __init__(
        self,
        model_name: str = DEFAULT_QA_MODEL,
        device: int = -1,
        max_context_chars: int = 3000,
        min_answer_score: float = 0.0,
        fallback_to_evidence: bool = True,
        enable_rule_postprocess: bool = True,
    ) -> None:
        self.model_name = model_name or DEFAULT_QA_MODEL
        self.device = int(device)
        self.max_context_chars = int(max_context_chars)
        self.min_answer_score = float(min_answer_score)
        self.fallback_to_evidence = bool(fallback_to_evidence)
        self.enable_rule_postprocess = bool(enable_rule_postprocess)

    def answer_from_chunks(
        self,
        question: str,
        retrieved: Sequence[Tuple[Dict[str, Any], float]],
        top_n_contexts: int = 3,
    ) -> QAResult:
        """Extract the best short answer from the top retrieved chunks."""
        question = clean_whitespace(question)
        if not question:
            raise ValueError("Question is empty.")
        if not retrieved:
            return _fallback_answer(question, [], "No retrieved chunks were provided.")

        candidates = list(retrieved[: max(1, int(top_n_contexts))])
        best_result: QAResult | None = None

        for rank, (chunk, retrieval_score) in enumerate(candidates, start=1):
            context = _truncate_context(get_chunk_text(chunk), self.max_context_chars)
            if not context:
                continue

            try:
                model_answer, qa_score = _run_direct_qa_model(
                    question=question,
                    context=context,
                    model_name=self.model_name,
                    device=self.device,
                )
            except Exception as exc:  # pragma: no cover - environment/model dependent
                if self.fallback_to_evidence:
                    return _fallback_answer(question, retrieved, f"Could not run QA model: {exc}")
                raise

            if not model_answer:
                continue

            final_answer = model_answer
            strategy = "extractive_qa"
            if self.enable_rule_postprocess:
                simple_fact = _extract_simple_fact_answer(question, context)
                if simple_fact:
                    final_answer = simple_fact
                    strategy = "extractive_qa_with_rule_postprocess"

            combined_score = float(qa_score) + (0.03 / rank)
            result = QAResult(
                question=question,
                answer=clean_whitespace(final_answer),
                qa_score=float(qa_score),
                retrieval_score=float(retrieval_score),
                combined_score=combined_score,
                source_rank=rank,
                chunk_id=str(chunk.get("chunk_id", "")),
                page_id=str(chunk.get("page_id", "")),
                code=str(chunk.get("code", "")),
                title=str(chunk.get("title", "")),
                section=str(chunk.get("section", "")),
                url=str(chunk.get("url", "")),
                evidence_text=context,
                model_name=self.model_name,
                used_pretrained_model=True,
                fallback_used=False,
                warning="",
                answer_strategy=strategy,
            )
            if best_result is None or result.combined_score > best_result.combined_score:
                best_result = result

        if best_result is None or best_result.qa_score < self.min_answer_score:
            if self.fallback_to_evidence:
                return _fallback_answer(
                    question,
                    retrieved,
                    "QA model produced no confident answer; returned evidence fallback.",
                )
            return QAResult(
                question=question,
                answer="",
                qa_score=0.0,
                retrieval_score=float(retrieved[0][1]),
                combined_score=0.0,
                source_rank=0,
                chunk_id="",
                page_id="",
                code="",
                title="",
                section="",
                url="",
                evidence_text="",
                model_name=self.model_name,
                used_pretrained_model=True,
                fallback_used=False,
                warning="QA model produced no confident answer.",
                answer_strategy="empty_model_answer",
            )

        return best_result
