"""Simple, evidence-grounded policy summaries for PolicyLens AI."""
from __future__ import annotations

import re

from src.text_utils import normalise, paragraphs, split_sentences

# Keys here are the exact keys the app reads (the app previously asked for "Limits" while this
# dict used "Limits and cost sharing", so limits never appeared in the summary).
TOPICS = {
    "Coverage": [
        "coverage", "covered", "benefit", "benefits",
        "hospitalisation", "hospitalization", "medical expenses",
    ],
    "Waiting periods": [
        "waiting period", "waiting periods", "pre-existing disease",
        "pre existing disease", "30-day waiting", "thirty day waiting",
    ],
    "Exclusions": [
        "exclusion", "exclusions", "excluded", "not covered",
        "shall not be covered",
    ],
    "Limits": [
        "sub-limit", "sublimit", "co-payment", "copayment",
        "deductible", "sum insured", "maximum limit", "room rent limit",
    ],
    "Claims": [
        "claim procedure", "claims procedure", "claim intimation",
        "reimbursement", "cashless", "pre-authorisation",
        "pre-authorization", "claim documents",
    ],
}

# Sentences that merely say something is NOT covered should not be listed as coverage.
_NEGATED = re.compile(r"\bnot (?:be )?(?:covered|payable|liable)|\bexclu|\bshall not\b|\bno cover")


def _sentences(text: str) -> list[str]:
    """Split extracted PDF text into sentence-like pieces."""
    out = []
    for block in paragraphs(text, min_len=0):           # keeps headings out of the next sentence
        if _is_heading(block):
            continue
        for piece in re.split(r"\s*•\s*", block):     # bullets are separate items
            out.extend(_LEADING_NUMBER.sub("", p).strip(" \t-") for p in split_sentences(piece, min_len=24))
    return out


_LEADING_NUMBER = re.compile(r"^\(?(?:\d{1,2}(?:\.\d{1,2})*|[ivxIVX]{1,4}|[a-zA-Z])[.)]\s+")


def _is_heading(block: str) -> bool:
    letters = [c for c in block if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.8


# Tie-break order when a sentence matches two topics equally: the more specific topic wins.
PRIORITY = ["Waiting periods", "Exclusions", "Claims", "Limits", "Coverage"]


def summarize_policy(pages: list[dict], max_per_topic: int = 3) -> dict:
    """Return a topic-organized summary based only on uploaded policy text.

    Each sentence goes to the single topic whose keywords it matches most (so a sentence is never
    listed twice). Each item includes its source page for verification.
    """
    buckets: dict[str, list[dict]] = {topic: [] for topic in TOPICS}
    seen = set()
    for page in pages:
        page_number = page.get("page", "?")
        for sentence in _sentences(page.get("text", "")):
            lower = sentence.lower()
            key = normalise(lower)[:250]
            if key in seen:
                continue
            scores = {t: sum(1 for w in TOPICS[t] if w in lower) for t in PRIORITY}
            if _NEGATED.search(lower):
                scores["Coverage"] = 0          # "X is not covered" is not coverage
            best = max(PRIORITY, key=lambda t: (scores[t], -PRIORITY.index(t)))
            if scores[best] == 0:
                continue
            seen.add(key)
            buckets[best].append({"text": sentence, "page": page_number, "score": scores[best]})

    result = {}
    for topic, items in buckets.items():
        items.sort(key=lambda item: item["score"], reverse=True)   # stable: earlier pages win ties
        result[topic] = [{"text": m["text"], "page": m["page"]} for m in items[:max_per_topic]]
    return result
