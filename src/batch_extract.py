"""Extract every PDF in data/raw/ to data/processed/<name>.json.

Run from anywhere:  python -m src.batch_extract
(Paths are resolved from the project root; they used to depend on the current folder.)
"""
from pathlib import Path

from src.extract_pdf import save_extraction

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"


def main():
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {RAW_DIR}")
        return
    for pdf_path in pdf_files:
        try:
            pages = save_extraction(pdf_path, OUTPUT_DIR / f"{pdf_path.stem}.json")
            empty = sum(not p["text"] for p in pages)
            print(f"Processed: {pdf_path.name} | {len(pages)} pages | {empty} pages without extractable text")
        except Exception as exc:
            print(f"Could not process {pdf_path.name}: {exc}")
    print(f"\nFinished. Check {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
