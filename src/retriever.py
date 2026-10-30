"""Passage retrieval for PolicyLens AI (TF-IDF with topic expansion and topic boosting)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.topics import PHRASES, detect_topics

# sklearn's English stop-word list contains negations. Dropping "not" turns the question
# "What is not covered?" into "covered", which retrieves coverage clauses instead of exclusions.
KEEP_WORDS = {"not", "no", "nor", "without", "except", "cannot", "never", "none",
              "only", "before", "after", "within"}
STOP_WORDS = sorted(ENGLISH_STOP_WORDS - KEEP_WORDS)

TOPIC_BOOST_PER_MATCH = 0.08
TOPIC_BOOST_MAX = 0.24


@dataclass
class Passage:
    text: str
    page: int


def chunk_pages(pages: list[dict], chunk_words: int = 180, overlap: int = 35) -> list[Passage]:
    if overlap >= chunk_words:
        raise ValueError("overlap must be smaller than chunk_words")
    chunks = []
    for page in pages:
        words = re.findall(r"\S+", page.get("text", ""))
        start = 0
        while start < len(words):
            end = min(len(words), start + chunk_words)
            chunks.append(Passage(" ".join(words[start:end]), int(page["page"])))
            if end == len(words):
                break
            start = end - overlap
    return chunks


class Retriever:
    def __init__(self, passages: list[Passage]):
        self.passages = passages
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words=STOP_WORDS,
            max_features=50000,
            sublinear_tf=True,
        )
        self.matrix = None
        self._lower = [p.text.lower() for p in passages]
        if passages:
            try:
                self.matrix = self.vectorizer.fit_transform([p.text for p in passages])
            except ValueError:
                # "empty vocabulary": e.g. a PDF whose text is only numbers or symbols.
                self.matrix = None

    def search(self, query: str, top_k: int = 5):
        if not query.strip() or self.matrix is None:
            return []

        topics = detect_topics(query)
        terms = [query]
        for topic in topics:
            terms.extend(PHRASES[topic])
        expanded_query = " ".join(dict.fromkeys(terms))

        q = self.vectorizer.transform([expanded_query])
        scores = cosine_similarity(q, self.matrix)[0]

        # Boost passages that contain exact phrases for every topic in the question.
        if topics:
            phrases = [ph for t in topics for ph in PHRASES[t]]
            for i, text in enumerate(self._lower):
                matches = sum(1 for ph in phrases if ph in text)
                if matches:
                    scores[i] += min(matches * TOPIC_BOOST_PER_MATCH, TOPIC_BOOST_MAX)

        ids = scores.argsort()[::-1][:max(1, top_k)]
        return [(self.passages[i], float(scores[i])) for i in ids if scores[i] > 0]
