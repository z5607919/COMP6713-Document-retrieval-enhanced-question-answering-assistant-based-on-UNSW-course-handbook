from __future__ import annotations

"""Supervised answer selector for the UNSW Handbook RAG project.

The selector learns from annotations.csv to choose the best evidence chunk from
retrieved top-k candidates before the extractive QA reader is applied.  It is a
small NumPy logistic-regression reranker rather than a fine-tuned neural model.
"""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

Chunk = Dict[str, Any]
Retrieved = List[Tuple[Chunk, float]]

DEFAULT_SECTIONS = ["basic_info", "overview", "term_offerings", "requirements", "fees", "admission", "other"]
DEFAULT_PAGE_TYPES = ["course", "program", "other"]


def clean(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def tokens(text: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+", clean(text).lower())


def get_text(chunk: Chunk) -> str:
    for key in ("chunk_text", "text", "content", "body"):
        value = clean(chunk.get(key, ""))
        if value:
            return value
    return ""


def get_code(chunk: Chunk) -> str:
    for key in ("code", "target_code", "course_code", "program_code"):
        value = clean(chunk.get(key, ""))
        if value:
            return value.upper()
    return ""


@dataclass
class SelectorPrediction:
    chunk: Chunk
    retrieval_score: float
    selector_score: float
    original_rank: int


class FeatureExtractor:
    """Build transparent features for a question/chunk candidate pair."""

    def __init__(self, sections: Sequence[str] | None = None, page_types: Sequence[str] | None = None) -> None:
        self.sections = list(sections or DEFAULT_SECTIONS)
        self.page_types = list(page_types or DEFAULT_PAGE_TYPES)
        self.feature_names = self._feature_names()

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk]) -> "FeatureExtractor":
        sections = list(DEFAULT_SECTIONS)
        page_types = list(DEFAULT_PAGE_TYPES)
        for chunk in chunks:
            section = clean(chunk.get("section", "")).lower() or "other"
            page_type = clean(chunk.get("page_type", "") or chunk.get("type", "")).lower() or "other"
            if section not in sections:
                sections.append(section)
            if page_type not in page_types:
                page_types.append(page_type)
        return cls(sections, page_types)

    def _feature_names(self) -> List[str]:
        base = [
            "retrieval_score",
            "retrieval_score_norm",
            "rank_reciprocal",
            "is_rank1",
            "is_rank2",
            "is_rank3",
            "question_chunk_overlap_count",
            "question_chunk_overlap_ratio",
            "question_title_overlap_ratio",
            "question_section_overlap_ratio",
            "code_mentioned_in_question",
            "chunk_log_len",
            "question_log_len",
        ]
        return base + [f"section={s}" for s in self.sections] + [f"page_type={p}" for p in self.page_types]

    def to_dict(self) -> Dict[str, Any]:
        return {"sections": self.sections, "page_types": self.page_types, "feature_names": self.feature_names}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureExtractor":
        return cls(data.get("sections") or DEFAULT_SECTIONS, data.get("page_types") or DEFAULT_PAGE_TYPES)

    def extract(self, question: str, chunk: Chunk, retrieval_score: float, rank: int, all_scores: Sequence[float]) -> np.ndarray:
        q_toks = set(tokens(question))
        c_toks = set(tokens(get_text(chunk)))
        title_toks = set(tokens(chunk.get("title", "")))
        section = clean(chunk.get("section", "")).lower() or "other"
        section_toks = set(tokens(section.replace("_", " ")))
        page_type = clean(chunk.get("page_type", "") or chunk.get("type", "")).lower() or "other"
        code = get_code(chunk)
        q_upper = clean(question).upper()
        overlap = len(q_toks & c_toks)
        min_score = min(all_scores) if all_scores else float(retrieval_score)
        max_score = max(all_scores) if all_scores else float(retrieval_score)
        score_norm = 0.0 if max_score <= min_score else (float(retrieval_score) - min_score) / (max_score - min_score)
        values = [
            float(retrieval_score),
            float(score_norm),
            1.0 / max(1, rank),
            1.0 if rank == 1 else 0.0,
            1.0 if rank == 2 else 0.0,
            1.0 if rank == 3 else 0.0,
            float(overlap),
            overlap / max(1, len(q_toks)),
            len(q_toks & title_toks) / max(1, len(q_toks)),
            len(q_toks & section_toks) / max(1, len(section_toks)),
            1.0 if code and code in q_upper else 0.0,
            math.log1p(len(c_toks)),
            math.log1p(len(q_toks)),
        ]
        values.extend([1.0 if section == s else 0.0 for s in self.sections])
        values.extend([1.0 if page_type == p else 0.0 for p in self.page_types])
        return np.asarray(values, dtype=np.float32)


class LogisticSelectorModel:
    """Minimal logistic regression with class weighting, implemented in NumPy."""

    def __init__(self, lr: float = 0.05, epochs: int = 800, l2: float = 1e-3, seed: int = 42) -> None:
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.l2 = float(l2)
        self.seed = int(seed)
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticSelectorModel":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D matrix.")
        if len(set(y.astype(int).tolist())) < 2:
            raise ValueError("Training requires both positive and negative selector examples.")
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        Xs = (X - self.mean) / self.std
        rng = np.random.default_rng(self.seed)
        self.w = rng.normal(0, 0.01, size=Xs.shape[1]).astype(np.float32)
        self.b = 0.0
        pos = max(1.0, float(np.sum(y == 1)))
        neg = max(1.0, float(np.sum(y == 0)))
        weights = np.where(y == 1, neg / pos, 1.0).astype(np.float32)
        denom = float(np.sum(weights))
        for _ in range(self.epochs):
            z = Xs @ self.w + self.b
            p = 1 / (1 + np.exp(-np.clip(z, -35, 35)))
            err = (p - y) * weights
            grad_w = (Xs.T @ err) / denom + self.l2 * self.w
            grad_b = float(np.sum(err) / denom)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w is None or self.mean is None or self.std is None:
            raise ValueError("Selector model is not fitted or loaded.")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xs = (X - self.mean) / self.std
        z = Xs @ self.w + self.b
        return 1 / (1 + np.exp(-np.clip(z, -35, 35)))

    def to_dict(self) -> Dict[str, Any]:
        if self.w is None or self.mean is None or self.std is None:
            raise ValueError("Selector model is not fitted or loaded.")
        return {
            "lr": self.lr,
            "epochs": self.epochs,
            "l2": self.l2,
            "seed": self.seed,
            "weights": self.w.astype(float).tolist(),
            "bias": float(self.b),
            "mean": self.mean.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogisticSelectorModel":
        model = cls(float(data.get("lr", 0.05)), int(data.get("epochs", 800)), float(data.get("l2", 1e-3)), int(data.get("seed", 42)))
        model.w = np.asarray(data["weights"], dtype=np.float32)
        model.b = float(data["bias"])
        model.mean = np.asarray(data["mean"], dtype=np.float32)
        model.std = np.asarray(data["std"], dtype=np.float32)
        return model


class TrainedAnswerSelector:
    """Loaded supervised selector used to rerank retrieved chunks."""

    def __init__(self, extractor: FeatureExtractor, model: LogisticSelectorModel, metadata: Dict[str, Any] | None = None) -> None:
        self.extractor = extractor
        self.model = model
        self.metadata = metadata or {}

    def score(self, question: str, retrieved: Retrieved) -> List[SelectorPrediction]:
        scores = [float(s) for _, s in retrieved]
        rows = [self.extractor.extract(question, c, float(s), i, scores) for i, (c, s) in enumerate(retrieved, start=1)]
        if not rows:
            return []
        probs = self.model.predict_proba(np.vstack(rows)).reshape(-1)
        return [SelectorPrediction(c, float(s), float(p), i) for i, ((c, s), p) in enumerate(zip(retrieved, probs), start=1)]

    def rerank(self, question: str, retrieved: Retrieved, limit: int | None = None) -> Retrieved:
        preds = self.score(question, retrieved)
        preds.sort(key=lambda p: (-p.selector_score, p.original_rank))
        if limit is not None:
            preds = preds[: max(1, int(limit))]
        return [(p.chunk, p.selector_score) for p in preds]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def to_dict(self) -> Dict[str, Any]:
        # Keep the saved model JSON simple and serializable. Metadata is copied
        # via JSON to avoid accidental circular references from training scripts.
        try:
            safe_metadata = json.loads(json.dumps(self.metadata, ensure_ascii=False))
        except Exception:
            safe_metadata = {k: str(v) for k, v in dict(self.metadata).items()}
        return {"extractor": self.extractor.to_dict(), "model": self.model.to_dict(), "metadata": safe_metadata}

    @classmethod
    def load(cls, path: str | Path) -> "TrainedAnswerSelector":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(FeatureExtractor.from_dict(data["extractor"]), LogisticSelectorModel.from_dict(data["model"]), data.get("metadata", {}))


def build_matrix(questions: Sequence[str], retrieved_lists: Sequence[Retrieved], gold_ids: Sequence[str], extractor: FeatureExtractor):
    X, y = [], []
    for question, retrieved, gold_id in zip(questions, retrieved_lists, gold_ids):
        scores = [float(s) for _, s in retrieved]
        for rank, (chunk, score) in enumerate(retrieved, start=1):
            X.append(extractor.extract(question, chunk, float(score), rank, scores))
            y.append(1 if str(chunk.get("chunk_id", "")) == gold_id else 0)
    if not X:
        raise ValueError("No selector samples were built.")
    return np.vstack(X).astype(np.float32), np.asarray(y, dtype=np.float32)


def evaluate_selector(selector: TrainedAnswerSelector, questions: Sequence[str], retrieved_lists: Sequence[Retrieved], gold_ids: Sequence[str]) -> Dict[str, Any]:
    n = max(1, len(questions))
    gold_in, old_top1, new_top1, old_mrr, new_mrr = 0, 0, 0, 0.0, 0.0
    for question, retrieved, gold_id in zip(questions, retrieved_lists, gold_ids):
        old_ids = [str(c.get("chunk_id", "")) for c, _ in retrieved]
        if gold_id in old_ids:
            gold_in += 1
            old_mrr += 1.0 / (old_ids.index(gold_id) + 1)
        if old_ids and old_ids[0] == gold_id:
            old_top1 += 1
        reranked = selector.rerank(question, retrieved)
        new_ids = [str(c.get("chunk_id", "")) for c, _ in reranked]
        if new_ids and new_ids[0] == gold_id:
            new_top1 += 1
        if gold_id in new_ids:
            new_mrr += 1.0 / (new_ids.index(gold_id) + 1)
    return {
        "examples": len(questions),
        "gold_in_candidates": gold_in / n,
        "original_top1": old_top1 / n,
        "selector_top1": new_top1 / n,
        "original_mrr": old_mrr / n,
        "selector_mrr": new_mrr / n,
        "gold_in_candidates_count": gold_in,
        "original_top1_count": old_top1,
        "selector_top1_count": new_top1,
    }
