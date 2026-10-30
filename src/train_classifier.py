"""Train the TF-IDF + Logistic Regression fallback from human-reviewed clauses.

Run:  python -m src.train_classifier

- Uses only rows with reviewed == "yes" in data/annotations/clauses.csv.
- Splits by policy document (not random rows) so near-duplicate clauses from the same policy
  cannot sit in both train and test.
- Also scores the model on the fixed 300-row test set, the same one used for BGE.
The saved model is only used by the app if the BGE checkpoint cannot load.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from src.text_utils import normalise

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "annotations" / "clauses.csv"
TEST = ROOT / "data" / "eval" / "test_set_labeled.csv"
OUT = ROOT / "models" / "clause_classifier.joblib"


def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def main():
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    df = df[(df["reviewed"].astype(str).str.strip().str.lower() == "yes") & df["text"].notna() & df["label"].notna()].copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # Never train on a paragraph that is also in the fixed test set.
    test = pd.read_csv(TEST) if TEST.exists() else None
    if test is not None:
        df = df[~df["text"].map(normalise).isin(set(test["text"].map(normalise)))]

    if df["label"].nunique() < 2 or len(df) < 10:
        raise SystemExit("Add at least 10 human-reviewed clauses (reviewed=yes) across 2+ labels to "
                         "data/annotations/clauses.csv before training. The bundled rows are illustrative only.")

    groups = df["policy_id"].fillna("unknown").astype(str)
    if groups.nunique() >= 3:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42).split(df, groups=groups))
        model = build_model().fit(df.iloc[tr]["text"], df.iloc[tr]["label"])
        print("Held-out documents:\n", classification_report(df.iloc[te]["label"], model.predict(df.iloc[te]["text"]), zero_division=0))
    else:
        print("Fewer than 3 policy documents: skipping the document-level split.")

    model = build_model().fit(df["text"], df["label"])   # final model uses every reviewed row
    if test is not None:
        pred = model.predict(test["text"].astype(str))
        print(f"Fixed test set: accuracy={accuracy_score(test['claude_label'], pred):.4f} "
              f"macro-F1={f1_score(test['claude_label'], pred, average='macro', zero_division=0):.4f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
