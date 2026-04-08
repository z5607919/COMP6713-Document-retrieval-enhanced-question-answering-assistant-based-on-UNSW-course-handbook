from __future__ import annotations

import json
import re
from html import unescape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from bs4 import BeautifulSoup, Tag

from .utils import load_text, normalize_ws, repair_mojibake

SECTION_PATTERNS = {
    "overview": [r"overview", r"description", r"summary"],
    "learning_outcomes": [r"learning outcomes?", r"outcomes?"],
    "term_offerings": [r"term offerings?", r"offering", r"taught in", r"teaching period"],
    "prerequisites": [r"pre[- ]?requisites?", r"prerequisites?"],
    "corequisites": [r"co[- ]?requisites?", r"corequisites?"],
    "requirements": [r"requirements?", r"program requirements?", r"structure"],
    "fees": [r"indicative fees?", r"fees?"],
    "admission": [r"admission", r"entry requirements?"],
}

NOISE_LINE_PATTERNS = [
    r"^cl_id:\s*",
    r"^publish:\s*",
    r"^self_enrol:\s*",
    r"^academic_item:\s*",
    r"^key:\s*",
    r"^curriculum_structure_config$",
    r"^title:\s*curriculum_structure_config$",
    r"^value:\s*ugrd$",
    r"^value:\s*\{?$",
    r"^label:\s*(Undergraduate|Postgraduate)$",
    r"^wildcard_links_target:\s*",
]

POLICY_BOILERPLATE_PATTERNS = [
    r"For more information on university policy on progression requirements",
    r"Academic Progression",
    r"Higher Degree Research",
]


def _try_json_load(text: str) -> Optional[Any]:
    text = text.strip()
    if not text or text[0] not in "[{":
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _walk_json(obj: Any, out: List[Tuple[str, Any]], prefix: str = "") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_str = str(key)
            full_key = f"{prefix}.{key_str}" if prefix else key_str
            out.append((full_key, value))
            _walk_json(value, out, full_key)
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            full_key = f"{prefix}[{idx}]"
            out.append((full_key, value))
            _walk_json(value, out, full_key)


def _json_key_matches(key: str, *patterns: str) -> bool:
    return any(re.search(p, key, flags=re.I) for p in patterns)


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_ws(value)
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [_stringify_value(v) for v in value]
        return normalize_ws("\n".join([p for p in parts if p]))
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            sv = _stringify_value(v)
            if sv:
                parts.append(f"{k}: {sv}")
        return normalize_ws("\n".join(parts))
    return normalize_ws(str(value))


def _extract_json_field(json_pairs: List[Tuple[str, Any]], *patterns: str) -> str:
    candidates: List[str] = []
    for key, value in json_pairs:
        if _json_key_matches(key, *patterns):
            text = _stringify_value(value)
            if text:
                candidates.append(text)
    candidates = [c for c in candidates if len(c) < 6000]
    if not candidates:
        return ""
    candidates.sort(key=len, reverse=True)
    return candidates[0]


def _collect_script_json(soup: BeautifulSoup) -> List[Any]:
    json_objs: List[Any] = []
    for script in soup.find_all("script"):
        content = script.string or script.get_text("\n", strip=True)
        if not content:
            continue
        obj = _try_json_load(content)
        if obj is not None:
            json_objs.append(obj)
    return json_objs


def _meta_content(soup: BeautifulSoup, **attrs: str) -> str:
    tag = soup.find("meta", attrs=attrs)
    return tag.get("content", "").strip() if tag else ""


def _extract_title_code_uoc_from_text(text: str) -> Tuple[str, str, str]:
    title = ""
    code = ""
    uoc = ""

    title_match = re.search(r"(?:Print\s+)?(.+?)\s+page", text, flags=re.I)
    if title_match:
        title = normalize_ws(title_match.group(1))

    code_match = re.search(r"\b([A-Z]{4}\d{4}|\d{4})\b", text)
    if code_match:
        code = code_match.group(1)

    uoc_match = re.search(r"\b(\d+\s+Units?\s+of\s+Credit)\b", text, flags=re.I)
    if uoc_match:
        uoc = normalize_ws(uoc_match.group(1))

    return title, code, uoc


def _extract_heading_sections(soup: BeautifulSoup) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    headings = soup.find_all(re.compile(r"^h[1-6]$", re.I))
    for heading in headings:
        heading_text = normalize_ws(heading.get_text(" ", strip=True))
        if not heading_text:
            continue
        collected: List[str] = []
        for sib in heading.next_siblings:
            if isinstance(sib, Tag) and sib.name and re.fullmatch(r"h[1-6]", sib.name, flags=re.I):
                break
            if isinstance(sib, Tag):
                text = normalize_ws(sib.get_text(" ", strip=True))
                if text:
                    collected.append(text)
        joined = normalize_ws("\n".join(collected))
        if joined:
            sections[heading_text] = joined
    return sections


def _collect_visible_text(soup: BeautifulSoup) -> str:
    for bad in soup(["script", "style", "noscript"]):
        bad.extract()
    text = soup.get_text("\n", strip=True)
    return normalize_ws(text)


def _dedupe_lines(lines: List[str]) -> List[str]:
    deduped: List[str] = []
    seen: set[str] = set()
    for line in lines:
        normalized = normalize_ws(line)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _strip_html(text: str) -> str:
    if not text:
        return ""
    text = unescape(repair_mojibake(text))
    return BeautifulSoup(text, "html.parser").get_text("\n", strip=True)


def _summarise_term_offerings(text: str) -> str:
    terms = sorted({normalize_ws(t) for t in re.findall(r"(Term\s+\d+|Summer\s*Term|Hexamester\s+\d+)", text, flags=re.I)})
    locations = sorted({normalize_ws(t) for t in re.findall(r"location:\s*(.+)", text, flags=re.I)})
    attendance = sorted({normalize_ws(t) for t in re.findall(r"attendance[_ ]mode:\s*(.+)", text, flags=re.I)})
    fees = {m.group(1).replace("_", " ").title(): m.group(2) for m in re.finditer(r"(fees_[a-z_]+):\s*(\d+)", text, flags=re.I)}

    parts: List[str] = []
    if terms:
        parts.append("Offered in: " + ", ".join(terms))
    if locations:
        parts.append("Locations: " + ", ".join(locations[:5]))
    if attendance:
        parts.append("Attendance Modes: " + ", ".join(attendance[:5]))
    if fees:
        fee_text = ", ".join(f"{k} {v}" for k, v in fees.items())
        parts.append("Fees snapshot: " + fee_text)

    if parts:
        return normalize_ws("\n".join(parts))
    return ""


def _clean_section_text(section_name: str, text: str) -> str:
    if not text:
        return ""

    text = _strip_html(text)
    text = normalize_ws(text)
    lines = [normalize_ws(line) for line in text.splitlines() if normalize_ws(line)]
    filtered: List[str] = []
    for line in lines:
        if any(re.search(pat, line, flags=re.I) for pat in NOISE_LINE_PATTERNS):
            continue
        filtered.append(line)
    lines = _dedupe_lines(filtered)
    cleaned = normalize_ws("\n".join(lines))

    if section_name == "term_offerings":
        summary = _summarise_term_offerings(cleaned)
        if summary:
            return summary

    if section_name in {"prerequisites", "corequisites"}:
        if any(re.search(pat, cleaned, flags=re.I) for pat in POLICY_BOILERPLATE_PATTERNS):
            course_codes = re.findall(r"\b[A-Z]{4}\d{4}\b", cleaned)
            if not course_codes:
                return ""

    if section_name == "requirements" and "curriculum_structure_config" in cleaned and not re.search(r"\b[A-Z]{4}\d{4}\b", cleaned):
        return ""

    return cleaned




def _simplify_program_requirements(text: str) -> str:
    text = _strip_html(text)
    text = normalize_ws(text)
    if not text:
        return ""

    lines = [normalize_ws(line) for line in text.splitlines() if normalize_ws(line)]

    titles: List[str] = []
    groups: List[str] = []
    descriptions: List[str] = []
    notes: List[str] = []
    uoc_values: List[int] = []
    items: List[Dict[str, str]] = []
    current_item: Dict[str, str] = {}

    def flush_item() -> None:
        nonlocal current_item
        if current_item.get("code") or current_item.get("name"):
            items.append(current_item)
        current_item = {}

    for line in lines:
        if line.startswith("title:"):
            value = normalize_ws(line.split(":", 1)[1])
            if value and not re.search(r"curriculum structure", value, flags=re.I) and value.lower() != "curriculum_structure_config":
                titles.append(value)
            continue

        vg = re.match(r"vertical_grouping:\s*label:\s*(.+)", line, flags=re.I)
        if vg:
            value = normalize_ws(vg.group(1))
            if value:
                groups.append(value)
            continue

        m = re.match(r"credit_points_max:\s*(\d+)", line, flags=re.I)
        if m:
            uoc_values.append(int(m.group(1)))
            continue

        m = re.match(r"credit_points:\s*(\d+)", line, flags=re.I)
        if m:
            val = int(m.group(1))
            if val > 0:
                uoc_values.append(val)
            continue

        m = re.match(r"description:\s*(.+)", line, flags=re.I)
        if m:
            value = normalize_ws(m.group(1))
            if value:
                descriptions.append(value)
                note_match = re.search(r"(Note:\s*.+)$", value, flags=re.I)
                if note_match:
                    notes.append(normalize_ws(note_match.group(1)))
            continue

        if re.match(r"Note:\s*", line, flags=re.I):
            notes.append(line)
            continue

        m = re.match(r"academic_item_name:\s*(.+)", line, flags=re.I)
        if m:
            flush_item()
            current_item["name"] = normalize_ws(m.group(1))
            continue

        m = re.match(r"academic_item_credit_points:\s*(\d+)", line, flags=re.I)
        if m:
            current_item["uoc"] = m.group(1)
            continue

        m = re.match(r"academic_item_code:\s*([A-Z]{4}\d{4})", line, flags=re.I)
        if m:
            current_item["code"] = m.group(1).upper()
            flush_item()
            continue

    flush_item()

    titles = _dedupe_lines(titles)
    groups = _dedupe_lines(groups)
    descriptions = _dedupe_lines(descriptions)
    notes = _dedupe_lines(notes)

    summary_lines: List[str] = []

    heading_bits: List[str] = []
    if titles:
        heading_bits.append(titles[0])
    if groups and (not heading_bits or groups[0].lower() != heading_bits[0].lower()):
        heading_bits.append(groups[0])
    if uoc_values:
        heading_bits.append(f"{uoc_values[0]} UOC")
    if heading_bits:
        summary_lines.append("Requirement group: " + " — ".join(heading_bits))

    for desc in descriptions[:3]:
        summary_lines.append(desc)

    if items:
        item_texts = []
        for item in items[:12]:
            code = item.get("code", "").strip()
            name = item.get("name", "").strip()
            uoc = item.get("uoc", "").strip()
            part = " ".join([p for p in [code, name] if p])
            if uoc:
                part += f" ({uoc} UOC)"
            if part:
                item_texts.append(part)
        if item_texts:
            prefix = "Eligible courses include: "
            joined = "; ".join(item_texts)
            if len(items) > 12:
                joined += "; ..."
            summary_lines.append(prefix + joined)

    for note in notes[:3]:
        if note not in summary_lines:
            summary_lines.append(note)

    summary_lines = _dedupe_lines(summary_lines)
    return normalize_ws("\n".join(summary_lines))

def _infer_named_sections(raw_text: str, heading_sections: Dict[str, str], json_pairs: List[Tuple[str, Any]]) -> Dict[str, str]:
    named: Dict[str, str] = {}

    for canonical, patterns in SECTION_PATTERNS.items():
        for heading, content in heading_sections.items():
            if any(re.fullmatch(p, heading, flags=re.I) or re.search(p, heading, flags=re.I) for p in patterns):
                named[canonical] = content
                break

    json_lookup = {
        "overview": [r"overview", r"description", r"summary"],
        "term_offerings": [r"term.*off", r"offering", r"teachingPeriod", r"teaching.*period"],
        "prerequisites": [r"pre.*req"],
        "corequisites": [r"co.*req"],
        "requirements": [r"requirements?", r"program.*requirements?", r"structure"],
        "fees": [r"fees?", r"indicative.*fees?"],
        "learning_outcomes": [r"learning.*outcomes?", r"outcomes?"],
        "admission": [r"admission", r"entry.*requirements?"],
    }
    for canonical, patterns in json_lookup.items():
        if not named.get(canonical):
            named[canonical] = _extract_json_field(json_pairs, *patterns)

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    for canonical, patterns in SECTION_PATTERNS.items():
        if named.get(canonical):
            continue
        for i, line in enumerate(lines):
            if any(re.fullmatch(p, line, flags=re.I) or re.search(p, line, flags=re.I) for p in patterns):
                captured: List[str] = []
                for line2 in lines[i + 1 : i + 12]:
                    if any(
                        re.fullmatch(p2, line2, flags=re.I) or re.search(p2, line2, flags=re.I)
                        for plist in SECTION_PATTERNS.values()
                        for p2 in plist
                    ):
                        break
                    captured.append(line2)
                candidate = normalize_ws("\n".join(captured))
                if candidate:
                    named[canonical] = candidate
                    break

    cleaned_named = {}
    for key, value in named.items():
        cleaned = _clean_section_text(key, value)
        if cleaned:
            cleaned_named[key] = cleaned
    return cleaned_named


def parse_html_page(
    html_path: str | Path,
    page_type: str,
    career: str,
    year: int,
    code_hint: str,
    source_url: str,
    title_hint: str = "",
) -> Dict[str, Any]:
    html = load_text(html_path)
    soup = BeautifulSoup(html, "html.parser")

    json_objs = _collect_script_json(soup)
    json_pairs: List[Tuple[str, Any]] = []
    for obj in json_objs:
        _walk_json(obj, json_pairs)

    visible_text = _collect_visible_text(BeautifulSoup(html, "html.parser"))
    heading_sections = _extract_heading_sections(BeautifulSoup(html, "html.parser"))

    og_title = _meta_content(soup, property="og:title")
    title_from_text, code_from_text, uoc_from_text = _extract_title_code_uoc_from_text(visible_text)

    title_json = _extract_json_field(json_pairs, r"(^|\.)name$", r"title")
    title = next(
        (
            t
            for t in [title_hint, og_title, title_from_text, title_json]
            if t and len(t) < 300 and "Handbook" not in t
        ),
        title_hint or "",
    )
    title = _clean_section_text("basic_info", title)

    code = code_hint or code_from_text or _extract_json_field(json_pairs, r"courseCode", r"programCode", r"code")
    code = normalize_ws(code)

    uoc_json = _extract_json_field(
        json_pairs,
        r"uoc$",
        r"unit.*credit",
        r"credit.*points?",
        r"unitsOfCredit",
    )
    uoc = _clean_section_text("basic_info", uoc_from_text or uoc_json)

    named_sections = _infer_named_sections(visible_text, heading_sections, json_pairs)

    overview = named_sections.get("overview", "")
    term_offerings = named_sections.get("term_offerings", "")
    prerequisites = named_sections.get("prerequisites", "")
    corequisites = named_sections.get("corequisites", "")
    requirements = named_sections.get("requirements", "")
    if page_type == "program" and requirements:
        simplified_requirements = _simplify_program_requirements(requirements)
        if simplified_requirements:
            requirements = simplified_requirements
    fees = named_sections.get("fees", "")
    admission = named_sections.get("admission", "")
    learning_outcomes = named_sections.get("learning_outcomes", "")

    full_text_parts = [
        f"Title: {title}" if title else "",
        f"Code: {code}" if code else "",
        f"Units of Credit: {uoc}" if uoc else "",
        f"Overview: {overview}" if overview else "",
        f"Term Offerings: {term_offerings}" if term_offerings else "",
        f"Prerequisites: {prerequisites}" if prerequisites else "",
        f"Corequisites: {corequisites}" if corequisites else "",
        f"Requirements: {requirements}" if requirements else "",
        f"Fees: {fees}" if fees else "",
        f"Admission: {admission}" if admission else "",
        f"Learning Outcomes: {learning_outcomes}" if learning_outcomes else "",
        visible_text[:12000] if visible_text else "",
    ]
    full_text = normalize_ws("\n\n".join([p for p in full_text_parts if p]))

    return {
        "page_id": f"{page_type}_{career}_{year}_{code or code_hint}",
        "page_type": page_type,
        "career": career,
        "handbook_year": year,
        "code": code or code_hint,
        "title": title,
        "url": source_url,
        "uoc": uoc,
        "overview": overview,
        "term_offerings": term_offerings,
        "prerequisites": prerequisites,
        "corequisites": corequisites,
        "requirements": requirements,
        "fees": fees,
        "admission": admission,
        "learning_outcomes": learning_outcomes,
        "full_text": full_text,
        "raw_html_path": str(html_path),
        "parse_notes": {
            "heading_sections_found": list(heading_sections.keys())[:50],
            "json_objects_found": len(json_objs),
        },
    }


def parse_many(fetch_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in fetch_rows:
        if not row.get("ok") or not row.get("html_path"):
            continue
        out.append(
            parse_html_page(
                html_path=row["html_path"],
                page_type=row["page_type"],
                career=row["career"],
                year=int(row["year"]),
                code_hint=row["code"],
                source_url=row["final_url"] or row["url"],
                title_hint=row.get("title_hint", ""),
            )
        )
    return out