# UNSW Handbook RAG Starter

A small, targeted data collection and retrieval starter for a COMP6713 group project based on the public UNSW Handbook.

This starter focuses on:

- collecting **course** and **program** pages from the UNSW Handbook
- parsing key fields such as title, code, UoC, overview, requirements, and prerequisites
- chunking the parsed pages for retrieval
- running a simple **BM25 baseline** with a CLI

## Project structure

```text
unsw_handbook_rag_starter/
├─ data/
│  ├─ seeds/
│  │  ├─ course_codes.csv
│  │  └─ program_codes.csv
│  ├─ raw_html/
│  ├─ parsed/
│  └─ index/
├─ scripts/
│  ├─ run_pipeline.py
│  └─ smoke_test.py
├─ src/
│  └─ unsw_handbook/
│     ├─ __init__.py
│     ├─ build_index.py
│     ├─ chunker.py
│     ├─ cli.py
│     ├─ parser.py
│     ├─ scrape.py
│     ├─ url_builder.py
│     └─ utils.py
├─ requirements.txt
└─ README.md
```

## Install

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Step 1: prepare seed codes

Edit:

- `data/seeds/course_codes.csv`
- `data/seeds/program_codes.csv`

Recommended first batch:

- 30 to 50 course codes
- 10 to 20 program codes

The provided seed files already contain a few examples.

## Step 2: run the collection + parsing pipeline

```bash
$env:PYTHONPATH="src"
python scripts/run_pipeline.py --year 2026 --careers undergraduate postgraduate --delay 1.5
```

This script will:

1. build candidate URLs from the seed CSV files
2. download HTML pages
3. parse page-level JSONL
4. build chunk-level JSONL
5. build a BM25 index file

Outputs:

- `data/parsed/pages.jsonl`
- `data/parsed/chunks.jsonl`
- `data/index/bm25_index.json`

## Optional: use Playwright fallback

If the site serves important content through client-side rendering in your environment, you can enable Playwright fallback:

```bash
pip install playwright
playwright install chromium
python scripts/run_pipeline.py --year 2026 --use-playwright
```

## Query the baseline

```bash
python -m unsw_handbook.cli --question "What are the prerequisites for COMP6713?" --chunks data/parsed/chunks.jsonl --top-k 5

python -m unsw_handbook.cli --question "What is the program structure for 3778?" --chunks data/parsed/chunks.jsonl --top-k 5
```

Example questions:

- `What is the UoC for COMP6713?`
- `What are the prerequisites for COMP6713?`
- `How many Units of Credit does program 3778 require?`
- `What does the Computer Science program require?`

## Important notes

1. This is a **starter pipeline**, not a production crawler.
2. The parser is intentionally **heuristic and fault-tolerant** because the Handbook page structure may vary.
3. Keep requests polite. Use delays and avoid unnecessarily large crawls.
4. For your project, start small and verify quality before scaling up.

## Suggested next steps for your group

- Add more seed codes and test the parser on 20 to 50 pages.
- Start creating a pilot QA set of 15 to 20 questions.
- Check whether the gold evidence aligns better with page-level or chunk-level labels.
- Compare this BM25 baseline with a dense retriever later.
