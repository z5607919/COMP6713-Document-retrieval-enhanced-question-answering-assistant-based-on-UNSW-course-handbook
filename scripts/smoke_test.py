from __future__ import annotations

import tempfile
from pathlib import Path

from unsw_handbook.build_index import bm25_score_query, build_bm25_index
from unsw_handbook.chunker import chunk_pages
from unsw_handbook.parser import parse_html_page
from unsw_handbook.utils import save_text


MOCK_COURSE_HTML = """
<html>
  <head>
    <title>Handbook - Natural Language Processing</title>
    <meta property="og:title" content="Natural Language Processing" />
  </head>
  <body>
    <h2>Natural Language Processing</h2>
    <h5>COMP6713</h5>
    <h5>6 Units of Credit</h5>
    <h3>Overview</h3>
    <p>This course introduces the fundamentals of NLP.</p>
    <h3>Prerequisites</h3>
    <p>COMP2521 and COMP3121</p>
    <h3>Term Offerings</h3>
    <p>Term 1</p>
  </body>
</html>
"""


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = Path(tmpdir) / "COMP6713.html"
        save_text(html_path, MOCK_COURSE_HTML)

        page = parse_html_page(
            html_path=html_path,
            page_type="course",
            career="undergraduate",
            year=2026,
            code_hint="COMP6713",
            source_url="https://example.org/COMP6713",
        )
        assert page["title"] == "Natural Language Processing"
        assert page["code"] == "COMP6713"
        assert "6 Units of Credit" in page["uoc"]
        assert "fundamentals of NLP" in page["overview"]
        assert "COMP2521" in page["prerequisites"]

        chunks = chunk_pages([page])
        assert chunks, "No chunks produced"

        index = build_bm25_index(chunks)
        ranked = bm25_score_query("What are the prerequisites for COMP6713?", chunks, index)
        assert ranked, "No retrieval result produced"

        print("Smoke test passed.")
        print(f"Parsed title: {page['title']}")
        print(f"Number of chunks: {len(chunks)}")
        print(f"Top chunk section: {chunks[ranked[0][0]]['section']}")


if __name__ == "__main__":
    main()
