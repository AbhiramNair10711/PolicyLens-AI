"""Evaluate the BGE clause classifier on the fixed 300-row test set and write reports/evaluation.md.

Run from the project root:
    python -m src.evaluate_on_testset

It uses the same loading and prediction code as the app (src.clause_classifier), so the score
describes exactly what users see. It refuses to run if the BGE model did not load, instead of
silently scoring the keyword fallback.

Test labels were produced by an LLM (Claude) and are NOT human-verified, so the scores
below mean "agreement with LLM-made labels".
"""
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.clause_classifier import LABELS, METHOD_TRANSFORMER, model_status, predict_clauses

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "eval" / "test_set_labeled.csv"
REPORT = ROOT / "reports" / "evaluation.md"
LABEL_COLUMN = "claude_label"


def main():
    method, why = model_status()
    if method != METHOD_TRANSFORMER:
        raise SystemExit(f"BGE model not loaded ({why}). Restore models/clause-classifier/ first.")

    df = pd.read_csv(TEST)
    if LABEL_COLUMN not in df.columns:
        raise SystemExit(f"{TEST} has no '{LABEL_COLUMN}' column")
    df = df[df.text.notna() & df[LABEL_COLUMN].notna()]

    pred = [label for label, _, _ in predict_clauses(df.text.astype(str).tolist())]
    true = df[LABEL_COLUMN].astype(str).str.strip().str.lower().tolist()
    labels = sorted(set(LABELS) | set(true) | set(pred))

    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, labels=labels, average="macro", zero_division=0)
    rep = classification_report(true, pred, labels=labels, zero_division=0)
    cm = pd.DataFrame(confusion_matrix(true, pred, labels=labels), index=labels, columns=labels)

    text = (
        "# Clause classifier evaluation\n\n"
        f"- Model: {method} (`models/clause-classifier/`)\n"
        f"- Test rows: {len(df)} (random paragraphs from {df.policy_id.nunique()} policy documents, never used for training)\n"
        "- Test labels: made by an LLM (Claude), not human-verified; scores mean agreement with those labels\n"
        "- Caveat: in the Colab notebook this same test set was also used to choose BGE over Legal-BERT, "
        "so the score is slightly optimistic\n"
        f"- **Accuracy: {acc:.4f}**\n- **Macro-F1: {f1:.4f}**\n\n"
        f"## Per-class report\n```\n{rep}\n```\n\n"
        f"## Confusion matrix (rows = true label, columns = predicted)\n```\n{cm.to_string()}\n```\n"
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved {REPORT}")


if __name__ == "__main__":
    main()
