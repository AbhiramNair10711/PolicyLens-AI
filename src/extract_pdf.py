"""Extract text from a PDF while preserving 1-indexed page numbers.

Single file:   python -m src.extract_pdf data/raw/policy.pdf --output data/processed/policy.json
Whole folder:  python -m src.batch_extract
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import pymupdf
except ImportError:  # PyMuPDF < 1.24.3
    import fitz as pymupdf


def extract_pdf(path: str | Path) -> list[dict]:
    with pymupdf.open(path) as doc:
        return [{"page": i, "text": page.get_text("text").strip()} for i, page in enumerate(doc, start=1)]


def save_extraction(pdf: str | Path, out: str | Path) -> list[dict]:
    pages = extract_pdf(pdf)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source_file": Path(pdf).name, "page_count": len(pages), "pages": pages},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    return pages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pages = save_extraction(args.pdf, args.output)
    empty = sum(not p["text"] for p in pages)
    print(f"Extracted {len(pages)} pages; {empty} pages have no selectable text (OCR may be needed).")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
