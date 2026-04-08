from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .utils import iter_jsonl

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
COURSE_CODE_RE = re.compile(r"\b[A-Z]{4}\d{4}\b", re.I)
PROGRAM_CODE_RE = re.compile(r"\b\d{4}\b")
STOPWORDS = {
    "what",
    "is",
    "are",
    "the",
    "for",
    "of",
    "to",
    "in",
    "a",
    "an",
    "and",
    "on",
    "do",
    "does",
    "about",
    "tell",
    "me",
    "please",
}


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOPWORDS]


def extract_explicit_codes(text: str) -> List[str]:
    course_codes = [m.group(0).upper() for m in COURSE_CODE_RE.finditer(text)]
    program_codes = [m.group(0) for m in PROGRAM_CODE_RE.finditer(text)]
    # preserve order, avoid duplicates
    seen = set()
    codes: List[str] = []
    for code in course_codes + program_codes:
        if code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def build_bm25_index(chunks: List[Dict], k1: float = 1.5, b: float = 0.75) -> Dict:
    tokenized_docs: List[List[str]] = [tokenize(chunk.get("chunk_text", "")) for chunk in chunks]
    doc_lens = [len(doc) for doc in tokenized_docs]
    avgdl = sum(doc_lens) / max(len(doc_lens), 1)

    term_doc_freq: Dict[str, int] = defaultdict(int)
    term_freqs: List[Counter] = []
    for doc in tokenized_docs:
        tf = Counter(doc)
        term_freqs.append(tf)
        for term in tf:
            term_doc_freq[term] += 1

    return {
        "k1": k1,
        "b": b,
        "avgdl": avgdl,
        "doc_lens": doc_lens,
        "term_doc_freq": dict(term_doc_freq),
        "term_freqs": [dict(tf) for tf in term_freqs],
        "chunk_count": len(chunks),
    }


def save_bm25_index(path: str | Path, index: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def load_bm25_index(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bm25_score_query(query: str, chunks: List[Dict], index: Dict) -> List[Tuple[int, float]]:
    terms = tokenize(query)
    if not terms:
        return []

    all_chunk_codes = {str(chunk.get("code", "") or "").upper() for chunk in chunks}
    query_codes = {code.upper() for code in extract_explicit_codes(query) if code.upper() in all_chunk_codes}
    wants_prereq = bool(re.search(r"prereq|pre[- ]?requisite", query, flags=re.I))
    wants_coreq = bool(re.search(r"coreq|co[- ]?requisite", query, flags=re.I))
    wants_uoc = bool(re.search(r"\buoc\b|units? of credit", query, flags=re.I))
    wants_overview = bool(re.search(r"overview|about|what is", query, flags=re.I))
    wants_requirements = bool(re.search(r"requirement|program structure", query, flags=re.I))

    k1 = float(index["k1"])
    b = float(index["b"])
    avgdl = float(index["avgdl"] or 1.0)
    N = int(index["chunk_count"])
    doc_lens = index["doc_lens"]
    term_doc_freq = index["term_doc_freq"]
    term_freqs = index["term_freqs"]

    scores: List[Tuple[int, float]] = []
    for i, chunk in enumerate(chunks):
        score = 0.0
        dl = max(float(doc_lens[i]), 1.0)
        tf_dict = term_freqs[i]
        for term in terms:
            if term not in tf_dict:
                continue
            tf = float(tf_dict[term])
            df = float(term_doc_freq.get(term, 0))
            idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1.0 - b + b * dl / avgdl)
            score += idf * ((tf * (k1 + 1.0)) / denom)

        chunk_code = str(chunk.get("code", "") or "").upper()
        section = str(chunk.get("section", "") or "").lower()

        if query_codes and chunk_code in query_codes:
            score += 8.0
        if wants_prereq and section == "prerequisites":
            score += 4.0
        if wants_coreq and section == "corequisites":
            score += 4.0
        if wants_uoc and section == "basic_info":
            score += 3.0
        if wants_overview and section == "overview":
            score += 2.0
        if wants_requirements and section == "requirements":
            score += 4.0

        if score > 0:
            scores.append((i, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def load_chunks(path: str | Path) -> List[Dict]:
    return list(iter_jsonl(path))
