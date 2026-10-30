"""Evidence-first question answering for PolicyLens AI.

Answers are built only from sentences found in the uploaded policy; nothing is generated.
(An unused RoBERTa QA loader was removed: it was never called, so the README's claim that the
app used `deepset/roberta-base-squad2` was not true.)
"""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.text_utils import normalise, split_sentences
from src.topics import detect_topics, topic_phrases

MIN_RETRIEVAL_SCORE = 0.025
BOILERPLATE = [
    "registered office address", "toll free number", "toll free no", "cin ",
    "irdai reg no", "mailing address",
]


def _sentences(text: str) -> list[str]:
    """Split extracted policy text into readable sentences (abbreviation-aware: "Rs. 5,000")."""
    return split_sentences(text, min_len=25)


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _relevant_sentences(question: str, retrieved, limit: int = 5):
    """Rank relevant sentences and remove duplicate wording."""
    question_words = {w for w in re.findall(r"[a-z0-9]+", question.lower())
                      if len(w) >= 3 and w not in ENGLISH_STOP_WORDS}
    phrases = topic_phrases(question)
    candidates = []
    seen = set()

    for passage, retrieval_score in retrieved:
        for sentence in _sentences(passage.text):
            norm = normalise(sentence)
            if not norm or norm in seen:
                continue
            if any(phrase in norm for phrase in BOILERPLATE):
                continue
            seen.add(norm)

            overlap = len(question_words & set(norm.split()))
            phrase_matches = sum(1 for phrase in phrases if phrase in sentence.lower())
            # Sentences must share something with the question, not just sit in a retrieved passage.
            if overlap == 0 and phrase_matches == 0:
                continue
            score = overlap * 0.15 + phrase_matches * 1.5 + min(float(retrieval_score), 1.0)
            candidates.append((score, sentence, int(passage.page), float(retrieval_score)))

    candidates.sort(key=lambda item: item[0], reverse=True)

    selected, selected_tokens = [], []
    for item in candidates:
        tokens = set(normalise(item[1]).split())
        if any(_jaccard(tokens, prev) >= 0.82 for prev in selected_tokens):
            continue
        selected.append(item)
        selected_tokens.append(tokens)
        if len(selected) >= limit:
            break
    return selected


def _evidence(retrieved, limit: int = 3) -> list[dict]:
    """Build concise evidence records and skip near-duplicate passages."""
    results, seen_tokens = [], []
    for passage, score in retrieved:
        text = re.sub(r"\s+", " ", passage.text).strip()
        tokens = set(normalise(text).split())
        if not tokens or any(_jaccard(tokens, prev) >= 0.75 for prev in seen_tokens):
            continue
        seen_tokens.append(tokens)
        results.append({"page": int(passage.page), "text": text[:1200], "score": round(float(score), 3)})
        if len(results) >= limit:
            break
    return results


EXPLANATIONS = {
    "waiting": ("A waiting period is a time during which certain conditions or treatments may not be "
                "covered. The exact duration and the conditions it applies to must be checked in the "
                "policy wording below."),
    "exclusion": ("An exclusion is a condition or expense that the policy may not cover. Check the cited "
                  "wording for exceptions and conditions."),
    "claim": ("This concerns how to make or receive payment for a claim. Check the cited wording for "
              "required documents, deadlines, and claim conditions."),
    "renewal": ("Renewal means continuing the policy for another period. Check the cited wording for the "
                "applicable conditions and deadlines."),
    "limit": ("These clauses explain limits on the insurer's payment or costs you may need to pay "
              "yourself. Check the cited wording for the exact amounts and conditions."),
    "premium": ("The premium is the amount you pay for the policy. Check the cited wording for how and "
                "when it is payable."),
}
DEFAULT_EXPLANATION = ("The passages below may help answer your question. Check the cited wording and any "
                       "related conditions or exceptions.")


def _plain_english_explanation(question: str) -> str:
    """Explain the topic without inventing policy-specific facts."""
    topics = detect_topics(question)
    return EXPLANATIONS[topics[0]] if topics else DEFAULT_EXPLANATION


def _safe_md(text: str) -> str:
    """Policy text may contain '*' (e.g. 'Sum Insured*'); stop it being read as markdown."""
    return text.replace("*", "∗")


def answer_question(question: str, retrieved: list[tuple[object, float]], max_evidence: int = 3) -> dict:
    """Answer using retrieved policy text, with a cautious fallback."""
    if not question or not question.strip():
        return {"answer": "Please enter a question about the uploaded policy.", "evidence": [],
                "method": "no question"}

    if not retrieved:
        return {"answer": ("I couldn't find relevant text in the uploaded policy. "
                           "Try different wording or inspect the policy manually."),
                "evidence": [], "method": "no evidence"}

    retrieved = sorted(retrieved, key=lambda item: float(item[1]), reverse=True)
    evidence = _evidence(retrieved, limit=max(1, max_evidence))

    if float(retrieved[0][1]) < MIN_RETRIEVAL_SCORE:
        return {"answer": ("I couldn't find strong evidence in the uploaded policy "
                           "to answer confidently. Try rephrasing your question."),
                "evidence": evidence, "method": "low-confidence abstention"}

    selected = _relevant_sentences(question, retrieved, limit=3)
    if not selected:
        return {"answer": ("I found potentially relevant passages, but couldn't identify "
                           "a sufficiently clear answer. Please inspect the evidence below."),
                "evidence": evidence, "method": "evidence-only fallback"}

    answer_lines = [f"- {_safe_md(sentence)} *(page {page})*" for _, sentence, page, _ in selected]
    answer = (
        "**Explanation in simple English:**\n\n"
        + _plain_english_explanation(question)
        + "\n\n**Relevant wording found in the uploaded policy:**\n\n"
        + "\n".join(answer_lines)
        + "\n\n**Please note:** These are extracts, not a definitive interpretation or claim-approval "
        "decision. Check the complete clauses, policy certificate, applicable section, and exceptions."
    )
    return {"answer": answer, "evidence": evidence, "method": "TF-IDF retrieval + sentence ranking"}
