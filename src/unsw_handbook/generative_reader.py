from __future__ import annotations

"""Generative answer reader for the UNSW Handbook RAG project.

This reader uses an instruction-tuned sequence-to-sequence model, by default
``google/flan-t5-small``, to generate a short answer from a selected evidence
chunk.  It is designed to complement the DistilBERT extractive reader:

- retrieval/selector decides which evidence chunk is relevant;
- this reader receives only that evidence and generates a concise answer;
- the output is still grounded because the prompt explicitly restricts the
  answer to the provided evidence.

The class returns the same ``QAResult`` dataclass used by ``qa_reader.py`` so it
can be dropped into existing scripts and demos without changing output format.
"""

import re
from functools import lru_cache
from typing import Any, Dict, Sequence, Tuple

from unsw_handbook.qa_reader import QAResult, clean_whitespace, get_chunk_text

DEFAULT_GENERATIVE_MODEL = "google/flan-t5-small"


@lru_cache(maxsize=3)
def _load_seq2seq_backend(model_name: str, device: int):
    """Load a text-to-text generation model once and cache it."""
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Generative QA requires transformers and torch. Install them with:\n"
            "    pip install transformers torch\n"
            "or add them to requirements.txt."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()

    if int(device) >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError("A GPU device was requested, but CUDA is not available.")
        model.to(f"cuda:{int(device)}")

    return tokenizer, model, torch


def _fallback_result(question: str, retrieved: Sequence[Tuple[Dict[str, Any], float]], warning: str) -> QAResult:
    chunk: Dict[str, Any] = retrieved[0][0] if retrieved else {}
    score = float(retrieved[0][1]) if retrieved else 0.0
    evidence = get_chunk_text(chunk)
    first_sentence = re.split(r"(?<=[.!?])\s+", evidence)[0] if evidence else ""
    answer = first_sentence[:300].strip() or "No answer could be generated from the retrieved evidence."
    return QAResult(
        question=question,
        answer=answer,
        qa_score=0.0,
        retrieval_score=score,
        combined_score=score,
        source_rank=1 if retrieved else 0,
        chunk_id=str(chunk.get("chunk_id", "")),
        page_id=str(chunk.get("page_id", "")),
        code=str(chunk.get("code", "")),
        title=str(chunk.get("title", "")),
        section=str(chunk.get("section", "")),
        url=str(chunk.get("url", "")),
        evidence_text=evidence,
        model_name="fallback-evidence-first-sentence",
        used_pretrained_model=False,
        fallback_used=True,
        warning=warning,
        answer_strategy="generative_fallback_evidence_first_sentence",
    )


def _normalise_evidence_for_generation(evidence: str) -> str:
    """Make semi-structured Handbook text easier for a seq2seq model to read.

    This is deliberately *not* a keyword answer rule. It only cleans formatting
    so that labels such as ``Term Offerings: Offered in: Term 1`` remain visible
    to the language model instead of being buried in one very long line.
    """
    evidence = clean_whitespace(evidence)
    evidence = re.sub(r"\s+([A-Z][A-Za-z /&-]{2,40}:)", r"\n\1", evidence)
    evidence = re.sub(r"\s+(Offered in:)", r"\n\1", evidence, flags=re.IGNORECASE)
    evidence = re.sub(r"\s+(Uoc:)", r"\n\1", evidence, flags=re.IGNORECASE)
    evidence = re.sub(r"\s+(Fees snapshot:)", r"\n\1", evidence, flags=re.IGNORECASE)
    evidence = re.sub(r"\n+", "\n", evidence).strip()
    return evidence


def _build_prompt(question: str, evidence: str) -> str:
    """Build an evidence-grounded generation prompt.

    The earlier prompt over-emphasised ``Not found in evidence``. With
    FLAN-T5-small this sometimes caused false abstention even when the answer was
    plainly present in the selected evidence. This prompt asks for the short
    answer phrase first, and abstains only as a last resort.
    """
    q = clean_whitespace(question)
    ev = _normalise_evidence_for_generation(evidence)
    return (
        "You are answering questions about the UNSW Handbook.\n"
        "Use only the evidence. Return the shortest exact answer phrase that answers the question.\n"
        "For list answers, return only the list items separated by commas.\n"
        "Do not copy the course title or course code unless the question asks for them.\n"
        "Only answer 'Not found in evidence' if there is truly no answer in the evidence.\n\n"
        f"Question: {q}\n"
        f"Evidence:\n{ev}\n\n"
        "Short answer:"
    )


def _clean_generated_answer(text: str) -> str:
    text = clean_whitespace(text)
    text = re.sub(r"^(answer|short answer)\s*[:\-]\s*", "", text, flags=re.IGNORECASE).strip()
    text = text.strip(' \"\'')
    if len(text) > 1 and text.endswith("."):
        text = text[:-1].strip()
    return text or "Not found in evidence"



class GenerativeQAReader:
    """Generate an evidence-grounded answer with a pre-trained seq2seq model."""

    def __init__(
        self,
        model_name: str = DEFAULT_GENERATIVE_MODEL,
        device: int = -1,
        max_context_chars: int = 1800,
        max_new_tokens: int = 48,
        num_beams: int = 4,
        fallback_to_evidence: bool = True,
    ) -> None:
        self.model_name = model_name or DEFAULT_GENERATIVE_MODEL
        self.device = int(device)
        self.max_context_chars = int(max_context_chars)
        self.max_new_tokens = int(max_new_tokens)
        self.num_beams = int(num_beams)
        self.fallback_to_evidence = bool(fallback_to_evidence)

    def answer_from_chunks(
        self,
        question: str,
        retrieved: Sequence[Tuple[Dict[str, Any], float]],
        top_n_contexts: int = 1,
    ) -> QAResult:
        question = clean_whitespace(question)
        if not question:
            raise ValueError("Question is empty.")
        if not retrieved:
            return _fallback_result(question, [], "No retrieved chunks were provided.")

        # For generative evidence-grounded QA, using a single selected chunk is
        # usually safer than stuffing several semi-structured chunks together.
        # If top_n_contexts > 1, join only the first N chunks with separators.
        selected = list(retrieved[: max(1, int(top_n_contexts))])
        primary_chunk = selected[0][0]
        primary_score = float(selected[0][1])
        evidence_parts = []
        for rank, (chunk, _score) in enumerate(selected, start=1):
            text = clean_whitespace(get_chunk_text(chunk))
            if text:
                evidence_parts.append(f"[Evidence {rank}] {text}")
        evidence = "\n".join(evidence_parts)
        if self.max_context_chars > 0 and len(evidence) > self.max_context_chars:
            evidence = evidence[: self.max_context_chars].rsplit(" ", 1)[0]
        if not evidence:
            return _fallback_result(question, retrieved, "Selected evidence chunk has no text.")

        try:
            tokenizer, model, torch = _load_seq2seq_backend(self.model_name, self.device)
            prompt = _build_prompt(question, evidence)
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            model_device = next(model.parameters()).device
            encoded = {k: v.to(model_device) for k, v in encoded.items()}
            with torch.no_grad():
                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=max(1, self.num_beams),
                    do_sample=False,
                    early_stopping=True,
                )
            raw = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            answer = _clean_generated_answer(raw)
            # There is no calibrated probability from normal generate(); use a
            # neutral confidence so downstream CSV columns remain numeric.
            qa_score = 1.0 if answer and answer.lower() != "not found in evidence" else 0.0
        except Exception as exc:  # pragma: no cover - environment/model dependent
            if self.fallback_to_evidence:
                return _fallback_result(question, retrieved, f"Could not run generative QA model: {exc}")
            raise

        return QAResult(
            question=question,
            answer=answer,
            qa_score=float(qa_score),
            retrieval_score=primary_score,
            combined_score=float(qa_score) + primary_score,
            source_rank=1,
            chunk_id=str(primary_chunk.get("chunk_id", "")),
            page_id=str(primary_chunk.get("page_id", "")),
            code=str(primary_chunk.get("code", "")),
            title=str(primary_chunk.get("title", "")),
            section=str(primary_chunk.get("section", "")),
            url=str(primary_chunk.get("url", "")),
            evidence_text=evidence,
            model_name=self.model_name,
            used_pretrained_model=True,
            fallback_used=False,
            warning="",
            answer_strategy="generative_evidence_grounded_qa",
        )
