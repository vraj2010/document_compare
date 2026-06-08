"""
normalization/__init__.py + normalizer.py combined.

Normalization reduces noise before comparison so that trivial
differences (extra spaces, fancy quotes, unicode variants) don't
inflate the change count.

Stages applied (in order)
--------------------------
1. Unicode normalization (NFC)          — canonical form
2. Whitespace normalization             — collapse runs, strip edges
3. Punctuation normalization            — smart quotes → ASCII, dashes
4. Heading normalization                — consistent casing
5. Table normalization                  — strip cell whitespace, lowercase headers
"""

from __future__ import annotations

import re
import unicodedata

from models import Document, DocumentSection, Table


# ---------------------------------------------------------------------------
# String-level normalisers
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """NFC normalization collapses combining characters into precomposed forms."""
    return unicodedata.normalize("NFC", text)


_WHITESPACE_RE = re.compile(r"[ \t]+")
_NEWLINE_RE = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    text = _WHITESPACE_RE.sub(" ", text)
    text = _NEWLINE_RE.sub("\n\n", text)
    return text.strip()


_PUNCT_MAP = str.maketrans({
    "\u2018": "'",    # left single quotation mark
    "\u2019": "'",    # right single quotation mark
    "\u201c": '"',    # left double quotation mark
    "\u201d": '"',    # right double quotation mark
    "\u2013": "-",    # en dash
    "\u2014": "-",    # em dash
    "\u2026": "...",  # horizontal ellipsis (1 char → 3 chars, needs dict form)
    "\u00a0": " ",    # non-breaking space
})


def normalize_punctuation(text: str) -> str:
    """Replace typographic punctuation with ASCII equivalents."""
    return text.translate(_PUNCT_MAP)


def normalize_text(text: str) -> str:
    """Full pipeline for a single string."""
    text = normalize_unicode(text)
    text = normalize_punctuation(text)
    text = normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# Structural normalisers
# ---------------------------------------------------------------------------

def normalize_heading(heading: str) -> str:
    """Strip trailing punctuation, collapse whitespace, title-case."""
    h = normalize_text(heading)
    h = h.rstrip(":.!?")
    return h.strip()


def normalize_table(table: Table) -> Table:
    return Table(
        caption=normalize_text(table.caption),
        headers=[normalize_text(h).lower() for h in table.headers],
        rows=[[normalize_text(cell) for cell in row] for row in table.rows],
    )


def normalize_section(section: DocumentSection) -> DocumentSection:
    return DocumentSection(
        heading=normalize_heading(section.heading),
        heading_level=section.heading_level,
        paragraphs=[normalize_text(p) for p in section.paragraphs if normalize_text(p)],
        lists=[[normalize_text(item) for item in lst if normalize_text(item)]
               for lst in section.lists],
        tables=[normalize_table(t) for t in section.tables],
    )


class DocumentNormalizer:
    """Applies all normalization passes to a Document in place (returns new obj)."""

    def normalize(self, doc: Document) -> Document:
        normalized_sections = [normalize_section(s) for s in doc.sections]
        # Remove empty sections
        normalized_sections = [
            s for s in normalized_sections
            if s.heading or s.paragraphs or s.tables or s.lists
        ]
        return Document(
            metadata=doc.metadata,
            sections=normalized_sections,
        )