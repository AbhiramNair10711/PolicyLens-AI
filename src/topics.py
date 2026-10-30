"""Shared topic vocabulary for retrieval and question answering.

Previously the retriever and the QA module each kept their own (slightly different) copies of
these lists, and topic detection used plain substring checks, so questions such as
"Is there a co-payment or deductible?" or "What is not covered?" never activated any topic.
"""
from __future__ import annotations

import re

# Words in the user's question that switch a topic on (matched on word boundaries).
TRIGGERS: dict[str, list[str]] = {
    "waiting": ["waiting", "wait", "pre-existing", "pre existing", "ped", "cooling off"],
    "exclusion": ["exclusion", "exclusions", "excluded", "exclude", "excludes", "not covered",
                  "not cover", "not pay", "not payable"],
    "claim": ["claim", "claims", "reimbursement", "reimburse", "cashless", "pre-authorisation",
              "pre-authorization", "intimation"],
    "limit": ["limit", "limits", "sub-limit", "sublimit", "co-payment", "copayment", "co-pay",
              "copay", "deductible", "sum insured", "room rent", "cap"],
    "renewal": ["renewal", "renew", "renewed", "grace period", "portability", "port",
                "cancel", "cancellation", "free look", "free-look"],
    "premium": ["premium", "premiums", "instalment", "installment"],
}

# Policy phrases that indicate a passage/sentence is about that topic.
PHRASES: dict[str, list[str]] = {
    "waiting": ["waiting period", "waiting periods", "pre-existing disease", "pre existing disease",
                "specific disease", "specified illness", "initial waiting period",
                "maternity waiting period", "continuous coverage"],
    "exclusion": ["exclusion", "excluded", "not covered", "shall not be covered", "not payable",
                  "permanent exclusions", "specific exclusions"],
    "claim": ["claim procedure", "claims procedure", "claim documents", "documents required",
              "claim intimation", "reimbursement", "cashless", "claim settlement",
              "pre-authorisation", "pre-authorization"],
    "limit": ["sum insured", "sub-limit", "sublimit", "deductible", "co-payment", "copayment",
              "maximum limit", "room rent", "maximum payable"],
    "renewal": ["policy renewal", "renewal of the policy", "renewal conditions", "renewed",
                "grace period", "portability", "cancellation", "free look"],
    "premium": ["premium payable", "premium amount", "premium payment", "premium due date",
                "mode of premium payment", "instalment"],
}

_TRIGGER_RE = {
    topic: re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b")
    for topic, words in TRIGGERS.items()
}


def detect_topics(question: str) -> list[str]:
    """Return every topic the question mentions (not just the first one)."""
    q = re.sub(r"\s+", " ", question.lower())
    return [topic for topic, rx in _TRIGGER_RE.items() if rx.search(q)]


def topic_phrases(question: str) -> list[str]:
    """All policy phrases for the topics a question mentions, de-duplicated, in order."""
    out: list[str] = []
    for topic in detect_topics(question):
        out.extend(PHRASES[topic])
    return list(dict.fromkeys(out))
