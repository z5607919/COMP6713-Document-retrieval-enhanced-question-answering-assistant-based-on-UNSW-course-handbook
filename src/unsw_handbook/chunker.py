from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .utils import normalize_ws

SECTION_ORDER = [
    ("basic_info", ["title", "code", "uoc"]),
    ("overview", ["overview"]),
    ("term_offerings", ["term_offerings"]),
    ("prerequisites", ["prerequisites"]),
    ("corequisites", ["corequisites"]),
    ("requirements", ["requirements"]),
    ("fees", ["fees"]),
    ("admission", ["admission"]),
    ("learning_outcomes", ["learning_outcomes"]),
]


def _build_prefix(page: Dict[str, Any]) -> List[str]:
    parts: List[str] = []
    if page.get("title"):
        parts.append(f"Title: {page['title']}")
    if page.get("code"):
        parts.append(f"Code: {page['code']}")
    if page.get("page_type"):
        parts.append(f"Type: {page['page_type']}")
    if page.get("career"):
        parts.append(f"Career: {page['career']}")
    if page.get("handbook_year"):
        parts.append(f"Year: {page['handbook_year']}")
    return parts


def _build_chunk_text(page: Dict[str, Any], section_name: str, fields: List[str]) -> str:
    content_parts: List[str] = []
    for field in fields:
        value = normalize_ws(str(page.get(field, "") or ""))
        if not value:
            continue
        label = field.replace("_", " ").title()
        content_parts.append(f"{label}: {value}")

    if not content_parts and section_name != "full_text_fallback":
        return ""

    parts = _build_prefix(page) + content_parts

    if section_name == "full_text_fallback" and page.get("full_text"):
        parts.append(page["full_text"])

    return normalize_ws("\n".join(parts))


def chunk_pages(pages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for page in pages:
        idx = 0
        page_chunk_count = 0
        for section_name, fields in SECTION_ORDER:
            text = _build_chunk_text(page, section_name, fields)
            if not text:
                continue
            idx += 1
            page_chunk_count += 1
            chunks.append(
                {
                    "chunk_id": f"{page['page_id']}_chunk_{idx:02d}",
                    "page_id": page["page_id"],
                    "page_type": page.get("page_type", ""),
                    "career": page.get("career", ""),
                    "handbook_year": page.get("handbook_year"),
                    "code": page.get("code", ""),
                    "title": page.get("title", ""),
                    "section": section_name,
                    "url": page.get("url", ""),
                    "chunk_text": text,
                }
            )

        if page_chunk_count == 0:
            fallback = _build_chunk_text(page, "full_text_fallback", [])
            if fallback:
                idx += 1
                chunks.append(
                    {
                        "chunk_id": f"{page['page_id']}_chunk_{idx:02d}",
                        "page_id": page["page_id"],
                        "page_type": page.get("page_type", ""),
                        "career": page.get("career", ""),
                        "handbook_year": page.get("handbook_year"),
                        "code": page.get("code", ""),
                        "title": page.get("title", ""),
                        "section": "full_text_fallback",
                        "url": page.get("url", ""),
                        "chunk_text": fallback,
                    }
                )
    return chunks
