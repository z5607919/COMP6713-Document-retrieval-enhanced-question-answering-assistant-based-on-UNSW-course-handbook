from __future__ import annotations

"""Dense retrieval utilities for the UNSW Handbook RAG project.

This module uses a pre-trained SentenceTransformer model to encode handbook chunks
and user questions into dense vectors. Retrieval is performed with cosine
similarity. Embeddings are saved to an `.npz` file so that evaluation and the demo
can run without recomputing all chunk embeddings every time.

The module also supports code-constrained dense retrieval. Handbook questions in
this project usually mention a course code such as COMP6713 or a program code such
as 3778. For those questions, dense retrieval can first filter chunks to the same
course/program page and then rank only the evidence chunks inside that page. This
keeps the pre-trained embedding model as the ranking model while using structured
handbook metadata to avoid obvious wrong-page retrieval errors.
"""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str):
    """Load a SentenceTransformer model once and reuse it.

    Without caching, evaluation reloads the model for every question, which is
    extremely slow and repeatedly prints the same Hugging Face loading messages.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - user environment dependent
        raise ImportError(
            "Dense retrieval requires sentence-transformers. Install it with:\n"
            "    pip install sentence-transformers\n"
            "or add it to requirements.txt."
        ) from exc
    return SentenceTransformer(model_name)


def extract_code_candidates(text: str) -> List[str]:
    """Extract likely Handbook course/program codes from a question.

    Examples:
    - Course codes: COMP6713, MATH1131, COMM5005
    - Program codes: 3778, 8543, 7412

    The extraction is intentionally simple and transparent because it is part of
    the retrieval system rather than a learned model.
    """
    if not text:
        return []
    upper_text = text.upper()
    codes: List[str] = []

    for match in re.findall(r"\b[A-Z]{4}\d{4}\b", upper_text):
        if match not in codes:
            codes.append(match)

    for match in re.findall(r"\b\d{4}\b", upper_text):
        if match not in codes:
            codes.append(match)

    return codes


def _chunk_matches_any_code(chunk: Dict[str, Any], codes: Sequence[str]) -> bool:
    """Return True if a chunk belongs to one of the extracted codes."""
    if not codes:
        return False
    code_set = {str(c).upper() for c in codes if str(c).strip()}
    chunk_fields = [
        str(chunk.get("code", "")).upper(),
        str(chunk.get("page_id", "")).upper(),
        str(chunk.get("url", "")).upper(),
        str(chunk.get("title", "")).upper(),
    ]
    return any(code in field for code in code_set for field in chunk_fields)


def chunk_to_dense_text(chunk: Dict[str, Any]) -> str:
    """Build the text that will be embedded for one chunk.

    We include lightweight metadata because questions often mention course/program
    codes or section intent. This usually helps the embedding model connect the
    question to the right handbook chunk without changing the underlying data.
    """
    parts = [
        f"Code: {chunk.get('code', '')}",
        f"Title: {chunk.get('title', '')}",
        f"Type: {chunk.get('page_type', '')}",
        f"Section: {chunk.get('section', '')}",
        str(chunk.get("chunk_text", "")),
    ]
    return "\n".join(part for part in parts if str(part).strip())


def build_dense_embeddings(
    chunks: Sequence[Dict[str, Any]],
    model_name: str = DEFAULT_DENSE_MODEL,
    batch_size: int = 32,
) -> np.ndarray:
    """Encode all chunks into L2-normalised dense vectors."""
    if not chunks:
        raise ValueError("No chunks provided for dense indexing.")

    model = _load_sentence_transformer(model_name)
    texts = [chunk_to_dense_text(chunk) for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def save_dense_index(
    path: str | Path,
    chunks: Sequence[Dict[str, Any]],
    embeddings: np.ndarray,
    model_name: str = DEFAULT_DENSE_MODEL,
) -> None:
    """Save dense embeddings and minimal metadata to an `.npz` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(chunks) != int(embeddings.shape[0]):
        raise ValueError(
            f"Number of chunks ({len(chunks)}) does not match embeddings ({embeddings.shape[0]})."
        )

    chunk_ids = np.array([str(chunk.get("chunk_id", "")) for chunk in chunks], dtype="U256")
    metadata = {
        "model_name": model_name,
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "chunk_count": len(chunks),
        "normalised": True,
    }
    np.savez_compressed(
        path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        chunk_ids=chunk_ids,
        metadata=json.dumps(metadata, ensure_ascii=False),
    )


def load_dense_index(path: str | Path) -> Dict[str, Any]:
    """Load a saved dense index."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dense index not found: {path}")

    data = np.load(path, allow_pickle=False)
    metadata_raw = str(data["metadata"].item())
    return {
        "embeddings": np.asarray(data["embeddings"], dtype=np.float32),
        "chunk_ids": [str(x) for x in data["chunk_ids"].tolist()],
        "metadata": json.loads(metadata_raw),
    }


def retrieve_dense(
    question: str,
    chunks: Sequence[Dict[str, Any]],
    dense_index: Dict[str, Any],
    limit: int = 5,
    model_name: str | None = None,
    constrain_to_question_code: bool = False,
) -> List[Tuple[Dict[str, Any], float]]:
    """Retrieve top-k chunks using dense cosine similarity.

    If constrain_to_question_code=True and the question contains a course/program
    code, retrieval is restricted to chunks whose metadata contains that code.
    This is useful for Handbook QA because a code usually identifies the target
    page, while the dense model is better used for ranking evidence chunks within
    that page.
    """
    if not question.strip():
        return []

    embeddings = np.asarray(dense_index["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        return []
    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            "Dense index and chunks.jsonl have different lengths. Rebuild the dense index "
            "from the same chunks.jsonl file."
        )

    saved_model_name = str(dense_index.get("metadata", {}).get("model_name") or DEFAULT_DENSE_MODEL)
    selected_model_name = model_name or saved_model_name

    model = _load_sentence_transformer(selected_model_name)
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].astype(np.float32)

    scores = embeddings @ query_embedding

    if constrain_to_question_code:
        codes = extract_code_candidates(question)
        matching = [i for i, chunk in enumerate(chunks) if _chunk_matches_any_code(chunk, codes)]
        if matching:
            candidate_indices = np.asarray(matching, dtype=np.int64)
        else:
            candidate_indices = np.arange(len(chunks), dtype=np.int64)
    else:
        candidate_indices = np.arange(len(chunks), dtype=np.int64)

    if len(candidate_indices) == 0:
        return []

    candidate_scores = scores[candidate_indices]
    k = min(max(int(limit), 1), len(candidate_indices))
    if k == len(candidate_indices):
        local_top = np.argsort(-candidate_scores)[:k]
    else:
        local_candidates = np.argpartition(-candidate_scores, kth=k - 1)[:k]
        local_top = local_candidates[np.argsort(-candidate_scores[local_candidates])]

    top_indices = candidate_indices[local_top]
    return [(chunks[int(i)], float(scores[int(i)])) for i in top_indices]
