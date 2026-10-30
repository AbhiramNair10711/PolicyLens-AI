"""Text helpers shared by the app, QA, summary and the data-preparation scripts.

Two bugs fixed here:
- "Rs." / "No." / "Sec." were treated as sentence ends, so "Room rent is limited to Rs. 5,000 per
  day" was cut at "Rs." and both halves fell under the 45-character minimum (the clause vanished).
- A wrapped line starting with "2.5 lakh" was treated as a new numbered clause ("2.").
`create_clause_csv.py` also used to split on blank lines, which PyMuPDF rarely emits, so it
produced whole pages as single "clauses"; it now uses `paragraphs` like the app does.
"""
from __future__ import annotations

import re

ABBREV = (r"(?:Rs|Re|INR|No|Nos|Sec|Cl|Art|Dr|Mr|Mrs|Ms|Smt|Shri|St|approx|incl|excl|etc|vs|viz|"
          r"Ltd|Pvt|Co|Corp|Inc|Govt|Dept|Reg|Regd|Max|Min|Yr|Yrs|Mo|Hrs|i\.e|e\.g|p\.a)")
_ABBREV_END = re.compile(rf"(?:^|[\s(\[]){ABBREV}\.$", re.IGNORECASE)
_ABBREV_DOT = re.compile(rf"\b({ABBREV})\.", re.IGNORECASE)
_DOT = "⁣"  # invisible placeholder for protected dots

# New block: "2.1 ", "(a) ", "iv) ", ALL-CAPS heading, or a bullet. The number must be followed by
# whitespace, so a wrapped "2.5 lakh" line does not start a new clause.
STARTS_NEW = re.compile(
    r"^(?:\(?(?:\d{1,2}(?:\.\d{1,2})*|[ivxIVX]{1,4}|[a-zA-Z])[.)](?=\s)"
    r"|[A-Z][A-Z .&-]{8,}$|[•▪●◦]|-\s)"
)


def ends_sentence(text: str) -> bool:
    t = text.rstrip()
    if not t.endswith((".", "!", "?", ":", ";")):
        return False
    return not _ABBREV_END.search(t)          # "...limited to Rs." is NOT an ending


def split_sentences(text: str, min_len: int = 25) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    protected = _ABBREV_DOT.sub(lambda m: m.group(1) + _DOT, text)
    protected = re.sub(r"(\d)\.(\d)", lambda m: m.group(1) + _DOT + m.group(2), protected)  # 2.5, 1.2.3
    parts = re.split(r"(?<=[.!?])\s+", protected)
    out = [p.replace(_DOT, ".").strip() for p in parts]
    return [p for p in out if len(p) > min_len]


def paragraphs(text: str, min_len: int = 45, max_len: int = 700) -> list[str]:
    """PDF text arrives one visual line at a time. Join lines back into paragraph-sized blocks."""
    blocks, cur = [], ""
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            if cur:                     # a blank line is a paragraph break when the PDF has one
                blocks.append(cur)
                cur = ""
            continue
        if cur and (ends_sentence(cur) or STARTS_NEW.match(line) or len(cur) > max_len):
            blocks.append(cur)
            cur = ""
        cur = f"{cur} {line}".strip()
    if cur:
        blocks.append(cur)
    return [b for b in blocks if len(b) >= min_len]


def normalise(text: str) -> str:
    """Lower-case alphanumerics only; used for duplicate detection."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
