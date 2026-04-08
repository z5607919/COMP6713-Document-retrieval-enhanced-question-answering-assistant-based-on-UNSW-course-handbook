from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def iter_jsonl(path: str | Path) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_text(path: str | Path, text: str) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def load_text(path: str | Path) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def repair_mojibake(text: str) -> str:
    if not text:
        return ""

    suspicious = ("Â", "â", "Ã", "\ufffd")
    if any(marker in text for marker in suspicious):
        try:
            repaired = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
            if repaired:
                text = repaired
        except Exception:
            pass

    replacements = {
        "Â": "",
        "â": "“",
        "â": "”",
        "â": "‘",
        "â": "’",
        "â": "–",
        "â": "—",
        "â¦": "…",
        "â€": "”",
        "â€˜": "‘",
        "â€™": "’",
        "â€œ": "“",
        "â€“": "–",
        "â€”": "—",
        "â€¦": "…",
        "\ufeff": "",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def normalize_ws(text: str) -> str:
    text = repair_mojibake(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[\u200b-\u200d\u2060]", "", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_get(d: Dict, *keys: str, default: Optional[str] = None) -> Optional[str]:
    cur = d
    for key in keys:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def looks_like_course_code(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{4}\d{4}", text.strip()))


def looks_like_program_code(text: str) -> bool:
    return bool(re.fullmatch(r"\d{4}", text.strip()))


def env_truthy(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
