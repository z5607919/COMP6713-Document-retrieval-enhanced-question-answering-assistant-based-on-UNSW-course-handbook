from __future__ import annotations

import argparse
import re
from typing import Dict, List

from .build_index import bm25_score_query, build_bm25_index, load_chunks

COURSE_CODE_RE = re.compile(r"\b[A-Z]{4}\d{4}\b", re.I)
PROGRAM_CODE_RE = re.compile(r"\b\d{4}\b")


def _extract_codes(text: str) -> List[str]:
    seen = set()
    codes: List[str] = []
    for regex in (COURSE_CODE_RE, PROGRAM_CODE_RE):
        for m in regex.finditer(text):
            code = m.group(0).upper()
            if code not in seen:
                seen.add(code)
                codes.append(code)
    return codes


def _simple_answer(question: str, ranked_chunks: List[Dict]) -> str:
    q = question.lower()
    q_codes = set(_extract_codes(question))

    if q_codes:
        same_code = [c for c in ranked_chunks if str(c.get("code", "")).upper() in q_codes]
        if same_code:
            ranked_chunks = same_code

    if re.search(r"prereq|pre[- ]?requisite", q):
        prereq_chunks = [c for c in ranked_chunks if c.get("section") == "prerequisites"]
        if prereq_chunks:
            text = prereq_chunks[0]["chunk_text"]
            m = re.search(r"Pre-?requisites?:\s*(.+)", text, flags=re.I | re.S)
            if m and m.group(1).strip():
                return re.sub(r"\s+", " ", m.group(1).strip())[:800]
            if q_codes:
                code = sorted(q_codes)[0]
                return f"No prerequisite information found for {code} in the retrieved handbook page."

    if re.search(r"coreq|co[- ]?requisite", q):
        coreq_chunks = [c for c in ranked_chunks if c.get("section") == "corequisites"]
        if coreq_chunks:
            text = coreq_chunks[0]["chunk_text"]
            m = re.search(r"Co-?requisites?:\s*(.+)", text, flags=re.I | re.S)
            if m and m.group(1).strip():
                return re.sub(r"\s+", " ", m.group(1).strip())[:800]
            if q_codes:
                code = sorted(q_codes)[0]
                return f"No corequisite information found for {code} in the retrieved handbook page."


    if re.search(r"requirement|program structure", q):
        req_chunks = [c for c in ranked_chunks if c.get("section") == "requirements"]
        if req_chunks and q_codes:
            code = sorted(q_codes)[0]
            text = req_chunks[0]["chunk_text"]
            m = re.search(r"Requirements?:\s*(.+)", text, flags=re.I | re.S)
            if m and m.group(1).strip():
                return re.sub(r"\s+", " ", m.group(1).strip())[:800]
            return f"No requirements information found for {code} in the retrieved handbook page."

    patterns = [
        (r"(units? of credit|\buoc\b)", [r"\b\d+\s+Units?\s+of\s+Credit\b", r"\b\d+\b"]),
        (r"requirement|program structure", [r"Requirements?:\s*(.+)"]),
        (r"overview|about|what is", [r"Overview:\s*(.+)"]),
        (r"fee|tuition", [r"Fees:\s*(.+)"]),
    ]
    for qpat, extraction_patterns in patterns:
        if not re.search(qpat, q):
            continue
        for chunk in ranked_chunks:
            text = chunk["chunk_text"]
            for epat in extraction_patterns:
                m = re.search(epat, text, flags=re.I | re.S)
                if m:
                    answer = m.group(1).strip() if m.lastindex else m.group(0).strip()
                    answer = re.sub(r"\s+", " ", answer)
                    return answer[:800]
    if ranked_chunks:
        return ranked_chunks[0]["chunk_text"][:800]
    return "No relevant evidence found."


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a BM25 baseline over UNSW Handbook chunks.")
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument("--chunks", required=True, help="Path to chunks.jsonl")
    parser.add_argument("--top-k", type=int, default=5, help="How many chunks to show")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    index = build_bm25_index(chunks)
    ranked = bm25_score_query(args.question, chunks, index)[: args.top_k]
    ranked_chunks = [{**chunks[i], "score": score} for i, score in ranked]

    answer = _simple_answer(args.question, ranked_chunks)
    print("=" * 80)
    print("Question:", args.question)
    print("-" * 80)
    print("Answer:")
    print(answer)
    print("-" * 80)
    print("Top evidence:")
    for idx, chunk in enumerate(ranked_chunks, start=1):
        print(f"[{idx}] score={chunk['score']:.4f} | code={chunk.get('code', '')} | section={chunk.get('section', '')}")
        print(f"URL: {chunk.get('url', '')}")
        print(chunk["chunk_text"][:1000])
        print("-" * 80)


if __name__ == "__main__":
    main()
