"""Turn extracted policy JSON files into a CSV of candidate clauses for labelling.

Run from anywhere:  python -m src.create_clause_csv
Paragraphs are split with the same function the app uses before classification, so the
labelled units match what the model sees at run time.
"""
import csv
import json
from pathlib import Path

from src.text_utils import paragraphs

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = ROOT / "data" / "annotations" / "clauses_to_label.csv"
COLUMNS = ["policy_id", "source_file", "page", "text", "label", "reviewed"]


def main():
    json_files = sorted(PROCESSED_DIR.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {PROCESSED_DIR}. Run `python -m src.batch_extract` first.")
        return

    rows = []
    for json_file in json_files:
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Could not read {json_file.name}: {exc}")
            continue
        for page_data in data.get("pages", []):
            for block in paragraphs(page_data.get("text", "")):
                rows.append({
                    "policy_id": json_file.stem,
                    "source_file": data.get("source_file", json_file.stem + ".pdf"),
                    "page": page_data.get("page"),
                    "text": block,
                    "label": "",
                    "reviewed": "no",
                })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Candidate clause blocks created: {len(rows)}")
    print(f"Output: {OUTPUT_FILE}")
    print("Labels are blank intentionally. Review clauses before training.")


if __name__ == "__main__":
    main()
