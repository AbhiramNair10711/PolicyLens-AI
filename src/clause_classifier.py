"""Clause classifier: fine-tuned BGE transformer (preferred), sklearn model, or keyword fallback.

Labels are the nine classes used in the training data and in the Colab notebook:
claims, coverage, definition, exclusion, limit, other, premium, renewal, waiting_period.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "clause_classifier.joblib"
TRANSFORMER_PATH = ROOT / "models" / "clause-classifier"  # BGE checkpoint from the Colab notebook
MAX_LEN = 256          # same max_length the model was fine-tuned and evaluated with
BATCH_SIZE = 32
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")

LABELS = [
    "claims", "coverage", "definition", "exclusion", "limit",
    "other", "premium", "renewal", "waiting_period",
]

METHOD_TRANSFORMER = "fine-tuned BGE transformer"
METHOD_SKLEARN = "TF-IDF + Logistic Regression"
METHOD_KEYWORD = "keyword fallback (not a trained model)"

KEYWORDS = {
    "waiting_period": ["waiting period", "pre-existing disease", "initial waiting", "cooling off", "months from the commencement"],
    "exclusion": ["excluded", "exclusion", "excludes", "excluding", "not covered", "shall not cover",
                  "not payable", "not eligible", "will not be covered", "expenses shall be excluded"],
    "coverage": ["covered", "coverage", "benefit", "hospitalisation", "hospitalization", "medical expenses"],
    "claims": ["claim", "notify", "intimation", "documents required", "reimbursement", "cashless"],
    "limit": ["sub-limit", "sublimit", "sum insured", "deductible", "co-payment", "co-pay", "room rent limit"],
    "premium": ["premium", "premium payable", "premium amount", "premium payment", "instalment", "installment"],
    "renewal": ["renewal", "renewed", "policy renewal", "grace period", "portability", "cancellation", "free look"],
    "definition": [" means ", "defined as", "definition", "shall mean"],
}

_load_error: str | None = None


def _fix_weight_filename() -> None:
    """Browsers/uploads often prefix the file (e.g. "1793302240612_model.safetensors").
    from_pretrained only finds the exact name, so rename a single stray *.safetensors file."""
    if any((TRANSFORMER_PATH / w).exists() for w in WEIGHT_FILES):
        return
    stray = list(TRANSFORMER_PATH.glob("*.safetensors"))
    if len(stray) == 1:
        stray[0].rename(TRANSFORMER_PATH / "model.safetensors")


def _read_id2label(model) -> dict[int, str]:
    """labels.json is the source of truth; refuse to run if it disagrees with config.json."""
    cfg = {int(k): v for k, v in (model.config.id2label or {}).items()}
    f = TRANSFORMER_PATH / "labels.json"
    if f.exists():
        id2label = {int(v): k for k, v in json.loads(f.read_text(encoding="utf-8")).items()}
        if cfg and not all(str(v).startswith("LABEL_") for v in cfg.values()) and cfg != id2label:
            raise ValueError(f"labels.json {id2label} does not match config.json id2label {cfg}")
    elif cfg and not all(str(v).startswith("LABEL_") for v in cfg.values()):
        id2label = cfg
    else:
        id2label = dict(enumerate(LABELS))
    if len(id2label) != model.config.num_labels:
        raise ValueError(f"Model has {model.config.num_labels} outputs but {len(id2label)} labels were found")
    return id2label


@lru_cache(maxsize=1)
def _transformer_bundle():
    """Load (tokenizer, model, id->label) once. Records why loading failed instead of hiding it."""
    global _load_error
    if not (TRANSFORMER_PATH / "config.json").exists():
        _load_error = f"no config.json in {TRANSFORMER_PATH}"
        return None
    try:
        _fix_weight_filename()
    except OSError:
        pass
    if not any((TRANSFORMER_PATH / w).exists() for w in WEIGHT_FILES):
        _load_error = f"weights missing: put model.safetensors in {TRANSFORMER_PATH}"
        return None
    try:
        import torch  # noqa: F401  (fail early with a clear message if torch is missing)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(TRANSFORMER_PATH))
        model = AutoModelForSequenceClassification.from_pretrained(str(TRANSFORMER_PATH))
        model.eval()
        id2label = _read_id2label(model)
        return tok, model, id2label
    except Exception as exc:  # torch/transformers missing, corrupt weights, label mismatch ...
        _load_error = f"{type(exc).__name__}: {exc}"
        return None


@lru_cache(maxsize=1)
def _sklearn_model():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None


def reload_model() -> None:
    """Forget a failed (or successful) load so newly added weights are picked up without a restart."""
    global _load_error
    _load_error = None
    _transformer_bundle.cache_clear()
    _sklearn_model.cache_clear()


def model_status() -> tuple[str, str | None]:
    """(method that will be used, reason the BGE model is unavailable or None)."""
    if _transformer_bundle() is not None:
        return METHOD_TRANSFORMER, None
    if _sklearn_model() is not None:
        return METHOD_SKLEARN, _load_error
    return METHOD_KEYWORD, _load_error


def keyword_classify(text: str) -> tuple[str, float]:
    """Classify using keyword counts (last-resort fallback, not a trained model)."""
    t = " " + re.sub(r"\s+", " ", text.lower()) + " "
    scores = {label: sum(t.count(k) for k in terms) for label, terms in KEYWORDS.items()}
    label, score = max(scores.items(), key=lambda item: item[1])
    if score == 0:
        return "other", 0.0
    total = sum(scores.values())
    return label, min(0.95, score / max(total, 1))


def _transformer_predict(texts: list[str], bundle) -> list[tuple[str, float]]:
    import torch
    tok, model, id2label = bundle
    # Batch paragraphs of similar length together so little time is spent on padding.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out: list[tuple[str, float] | None] = [None] * len(texts)
    with torch.inference_mode():
        for s in range(0, len(order), BATCH_SIZE):
            idx = order[s:s + BATCH_SIZE]
            batch = tok([texts[i] for i in idx], truncation=True, max_length=MAX_LEN,
                        padding=True, return_tensors="pt")
            conf, pred = torch.softmax(model(**batch).logits.float(), dim=-1).max(dim=-1)
            for j, i in enumerate(idx):
                out[i] = (id2label[int(pred[j])], float(conf[j]))
    return out


def predict_clauses(texts: list[str]) -> list[tuple[str, float, str]]:
    """Classify many paragraphs at once (batched; much faster than one call per paragraph)."""
    texts = [str(t) for t in texts]
    if not texts:
        return []
    bundle = _transformer_bundle()
    if bundle is not None:
        try:
            return [(lab, conf, METHOD_TRANSFORMER) for lab, conf in _transformer_predict(texts, bundle)]
        except Exception as exc:
            global _load_error
            _load_error = f"inference failed: {type(exc).__name__}: {exc}"

    model = _sklearn_model()
    if model is not None:
        try:
            labels = model.predict(texts)
            confs = model.predict_proba(texts).max(axis=1) if hasattr(model, "predict_proba") else [0.0] * len(texts)
            return [(str(l), float(c), METHOD_SKLEARN) for l, c in zip(labels, confs)]
        except Exception:
            pass

    return [(*keyword_classify(t), METHOD_KEYWORD) for t in texts]


def predict_clause(text: str) -> tuple[str, float, str]:
    """Return (label, confidence, method) for one paragraph."""
    return predict_clauses([text])[0]
