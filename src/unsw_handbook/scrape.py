from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import requests
from tqdm import tqdm

from .url_builder import CandidateURL
from .utils import ensure_dir, save_text, slugify

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


@dataclass
class FetchResult:
    page_type: str
    career: str
    year: int
    code: str
    title_hint: str
    url: str
    ok: bool
    status_code: Optional[int]
    final_url: Optional[str]
    html_path: Optional[str]
    error: Optional[str] = None


def _requests_fetch(url: str, timeout: int = 30) -> tuple[str, int, str]:
    response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text, response.status_code, response.url


def _playwright_fetch(url: str, timeout_ms: int = 45000) -> tuple[str, int, str]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise RuntimeError(
            "Playwright is not installed. Run `pip install playwright` and "
            "`playwright install chromium` first."
        ) from e

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        response = page.goto(url, wait_until="networkidle", timeout=timeout_ms)
        html = page.content()
        final_url = page.url
        status_code = response.status if response else None
        browser.close()

    if status_code is None or status_code >= 400:
        raise RuntimeError(f"Playwright fetch failed for {url} with status {status_code}")

    return html, int(status_code), final_url


def fetch_candidates(
    candidates: Iterable[CandidateURL],
    out_dir: str | Path,
    delay: float = 1.5,
    use_playwright: bool = False,
    timeout: int = 30,
) -> List[FetchResult]:
    out_dir = ensure_dir(out_dir)
    results: List[FetchResult] = []

    for cand in tqdm(list(candidates), desc="Fetching pages"):
        folder = out_dir / cand.page_type / cand.career / str(cand.year)
        ensure_dir(folder)
        filename = f"{cand.code}.html"
        html_path = folder / filename

        try:
            if use_playwright:
                html, status_code, final_url = _playwright_fetch(cand.url)
            else:
                html, status_code, final_url = _requests_fetch(cand.url, timeout=timeout)

            save_text(html_path, html)
            results.append(
                FetchResult(
                    page_type=cand.page_type,
                    career=cand.career,
                    year=cand.year,
                    code=cand.code,
                    title_hint=cand.title_hint,
                    url=cand.url,
                    ok=True,
                    status_code=status_code,
                    final_url=final_url,
                    html_path=str(html_path),
                )
            )
        except Exception as e:  # noqa: BLE001 - keep broad for crawler logs
            results.append(
                FetchResult(
                    page_type=cand.page_type,
                    career=cand.career,
                    year=cand.year,
                    code=cand.code,
                    title_hint=cand.title_hint,
                    url=cand.url,
                    ok=False,
                    status_code=None,
                    final_url=None,
                    html_path=None,
                    error=str(e),
                )
            )
        time.sleep(delay)

    return results
