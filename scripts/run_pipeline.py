from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from unsw_handbook.build_index import build_bm25_index, save_bm25_index
from unsw_handbook.chunker import chunk_pages
from unsw_handbook.parser import parse_many
from unsw_handbook.scrape import fetch_candidates
from unsw_handbook.url_builder import load_all_candidates
from unsw_handbook.utils import ensure_dir, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the UNSW Handbook collection pipeline.")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument(
        "--careers",
        nargs="+",
        default=["undergraduate", "postgraduate"],
        help="Careers to try, e.g. undergraduate postgraduate",
    )
    parser.add_argument("--course-seeds", default="data/seeds/course_codes.csv")
    parser.add_argument("--program-seeds", default="data/seeds/program_codes.csv")
    parser.add_argument("--raw-dir", default="data/raw_html")
    parser.add_argument("--parsed-dir", default="data/parsed")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--use-playwright", action="store_true")
    parser.add_argument("--top-k", type=int, default=5, help="Unused now, kept for future compatibility")
    args = parser.parse_args()

    parsed_dir = ensure_dir(args.parsed_dir)
    index_dir = ensure_dir(args.index_dir)

    candidates = load_all_candidates(
        course_seed_csv=args.course_seeds,
        program_seed_csv=args.program_seeds,
        year=args.year,
        careers=args.careers,
    )
    print(f"Built {len(candidates)} candidate URLs")

    fetch_results = fetch_candidates(
        candidates=candidates,
        out_dir=args.raw_dir,
        delay=args.delay,
        use_playwright=args.use_playwright,
        timeout=args.timeout,
    )

    fetch_log_path = parsed_dir / "fetch_log.csv"
    with open(fetch_log_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(fetch_results[0]).keys()) if fetch_results else [])
        if fetch_results:
            writer.writeheader()
            for row in fetch_results:
                writer.writerow(asdict(row))
    print(f"Saved fetch log to {fetch_log_path}")

    pages = parse_many([asdict(r) for r in fetch_results])
    pages_path = parsed_dir / "pages.jsonl"
    write_jsonl(pages_path, pages)
    print(f"Saved {len(pages)} parsed pages to {pages_path}")

    chunks = chunk_pages(pages)
    chunks_path = parsed_dir / "chunks.jsonl"
    write_jsonl(chunks_path, chunks)
    print(f"Saved {len(chunks)} chunks to {chunks_path}")

    bm25_index = build_bm25_index(chunks)
    index_path = index_dir / "bm25_index.json"
    save_bm25_index(index_path, bm25_index)
    print(f"Saved BM25 index to {index_path}")


if __name__ == "__main__":
    main()
