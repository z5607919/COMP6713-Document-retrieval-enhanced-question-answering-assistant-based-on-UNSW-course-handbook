from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .utils import read_csv_rows

BASE_URL = "https://www.handbook.unsw.edu.au"
VALID_CAREERS = ("undergraduate", "postgraduate")


@dataclass(frozen=True)
class CandidateURL:
    page_type: str
    career: str
    year: int
    code: str
    title_hint: str
    url: str


def build_course_url(career: str, year: int, course_code: str) -> str:
    return f"{BASE_URL}/{career}/courses/{year}/{course_code}"


def build_program_url(career: str, year: int, program_code: str) -> str:
    return f"{BASE_URL}/{career}/programs/{year}/{program_code}"


def load_course_candidates(seed_csv: str | Path, year: int, careers: Iterable[str]) -> List[CandidateURL]:
    rows = read_csv_rows(seed_csv)
    out: List[CandidateURL] = []
    for row in rows:
        code = row["course_code"].strip().upper()
        title_hint = row.get("title_hint", "").strip()
        for career in careers:
            out.append(
                CandidateURL(
                    page_type="course",
                    career=career,
                    year=year,
                    code=code,
                    title_hint=title_hint,
                    url=build_course_url(career, year, code),
                )
            )
    return out


def load_program_candidates(seed_csv: str | Path, year: int, careers: Iterable[str]) -> List[CandidateURL]:
    rows = read_csv_rows(seed_csv)
    out: List[CandidateURL] = []
    for row in rows:
        code = row["program_code"].strip()
        title_hint = row.get("title_hint", "").strip()
        for career in careers:
            out.append(
                CandidateURL(
                    page_type="program",
                    career=career,
                    year=year,
                    code=code,
                    title_hint=title_hint,
                    url=build_program_url(career, year, code),
                )
            )
    return out


def load_all_candidates(
    course_seed_csv: str | Path,
    program_seed_csv: str | Path,
    year: int,
    careers: Iterable[str],
) -> List[CandidateURL]:
    careers = [c.strip().lower() for c in careers if c.strip()]
    for c in careers:
        if c not in VALID_CAREERS:
            raise ValueError(f"Unsupported career: {c}")

    out: List[CandidateURL] = []
    out.extend(load_course_candidates(course_seed_csv, year, careers))
    out.extend(load_program_candidates(program_seed_csv, year, careers))
    return out
