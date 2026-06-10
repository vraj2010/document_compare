"""Convert chunk-level matches (ChunkMatch) into section-level diffs for display."""
# Modified to trigger Streamlit file watcher

from __future__ import annotations

import difflib
from collections import defaultdict
from dataclasses import dataclass, field

from models import ChunkMatch, ChangeType, Document, DocumentSection


@dataclass
class SectionDiff:
    """One section-level difference for display."""

    status: str  # "MODIFIED" | "REMOVED" | "ADDED"
    heading: str  # section heading
    original_text: str = ""  # full original section text (for Removed / Modified)
    revised_text: str = ""  # full revised section text (for Added / Modified)
    highlighted_revised: str = ""  # revised text with <span> highlights on changed words
    highlighted_original: str = ""  # original text with <span> highlights on changed words
    section_index_a: int = -1
    section_index_b: int = -1


# ---------------------------------------------------------------------------
# Word-level highlighting
# ---------------------------------------------------------------------------

_RED_SPAN = '<span style="background-color:#ffcdd2;color:#b71c1c;">'
_GREEN_SPAN = '<span style="background-color:#c8e6c9;color:#1b5e20;">'
_SPAN_CLOSE = "</span>"


def highlight_revised_only(text_a: str, text_b: str) -> tuple[str, str]:
    """Return ``(highlighted_original, highlighted_revised)`` with word-level
    ``<span>`` highlights showing differences between *text_a* (original) and
    *text_b* (revised).

    * **equal** blocks – words copied as-is to both outputs.
    * **delete** blocks – words wrapped in a red span in the *original* output
      only; nothing added to the revised output.
    * **insert** blocks – words wrapped in a green span in the *revised* output
      only; nothing added to the original output.
    * **replace** blocks – old words in a red span for original, new words in a
      green span for revised.

    No strikethrough is applied anywhere.
    """
    if not text_a and not text_b:
        return ("", "")

    words_a = text_a.split()
    words_b = text_b.split()

    matcher = difflib.SequenceMatcher(None, words_a, words_b)

    original_parts: list[str] = []
    revised_parts: list[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            equal_chunk = " ".join(words_a[i1:i2])
            original_parts.append(equal_chunk)
            revised_parts.append(equal_chunk)

        elif tag == "delete":
            deleted = " ".join(words_a[i1:i2])
            original_parts.append(f"{_RED_SPAN}{deleted}{_SPAN_CLOSE}")
            # Nothing added to revised output.

        elif tag == "insert":
            inserted = " ".join(words_b[j1:j2])
            revised_parts.append(f"{_GREEN_SPAN}{inserted}{_SPAN_CLOSE}")
            # Nothing added to original output.

        elif tag == "replace":
            old = " ".join(words_a[i1:i2])
            new = " ".join(words_b[j1:j2])
            original_parts.append(f"{_RED_SPAN}{old}{_SPAN_CLOSE}")
            revised_parts.append(f"{_GREEN_SPAN}{new}{_SPAN_CLOSE}")

    highlighted_original = " ".join(original_parts)
    highlighted_revised = " ".join(revised_parts)
    return (highlighted_original, highlighted_revised)


# ---------------------------------------------------------------------------
# Section-diff builder
# ---------------------------------------------------------------------------

def build_section_diffs(
    doc_a: Document,
    doc_b: Document,
    matches: list[ChunkMatch],
) -> list[SectionDiff]:
    """Build a list of :class:`SectionDiff` from chunk-level *matches*.
    Uses a greedy 1-to-1 bipartite mapping between Doc A sections and Doc B sections
    to ensure we only compare related sections together. Unmapped sections are
    treated as completely ADDED or REMOVED.
    """
    scores: dict[tuple[int, int], float] = defaultdict(float)
    max_sim: dict[tuple[int, int], float] = defaultdict(float)

    for m in matches:
        if m.chunk_a is not None and m.chunk_b is not None:
            if m.chunk_a.section_index >= 0 and m.chunk_b.section_index >= 0:
                pair = (m.chunk_a.section_index, m.chunk_b.section_index)
                
                # Weight by match quality
                if m.change_type == ChangeType.EXACT:
                    w = 2.0
                elif m.change_type == ChangeType.NEAR_EXACT:
                    w = 1.5
                elif m.change_type == ChangeType.SEMANTIC:
                    w = 1.0
                elif m.change_type == ChangeType.MODIFIED:
                    w = 0.5
                else:
                    w = 0.1
                
                scores[pair] += w
                max_sim[pair] = max(max_sim[pair], m.similarity_score)

    # Add heading bonuses
    for i, sec_a in enumerate(doc_a.sections):
        for j, sec_b in enumerate(doc_b.sections):
            h_a = sec_a.heading.strip().lower()
            h_b = sec_b.heading.strip().lower()
            if h_a and h_b:
                if h_a == h_b:
                    scores[(i, j)] += 10.0
                    max_sim[(i, j)] = 1.0
                else:
                    fuzz_score = difflib.SequenceMatcher(None, h_a, h_b).ratio()
                    if fuzz_score > 0.8:
                        scores[(i, j)] += 5.0
                        max_sim[(i, j)] = max(max_sim[(i, j)], fuzz_score)

    # Sort pairs by score descending
    sorted_pairs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    mapped_a: set[int] = set()
    mapped_b: set[int] = set()
    b_to_a: dict[int, int] = {}
    
    # Greedily assign 1-to-1
    for (a_idx, b_idx), score in sorted_pairs:
        if a_idx in mapped_a or b_idx in mapped_b:
            continue
            
        # Threshold: must have > 0.5 score and at least one chunk with > 0.65 similarity
        # (or high heading similarity which boosts both)
        if score > 0.5 and max_sim[(a_idx, b_idx)] > 0.65:
            b_to_a[b_idx] = a_idx
            mapped_a.add(a_idx)
            mapped_b.add(b_idx)

    diffs: list[SectionDiff] = []

    # 1. REMOVED (in doc_a but not mapped)
    for a_idx, sec_a in enumerate(doc_a.sections):
        if a_idx not in mapped_a:
            orig_text = sec_a.to_text().strip()
            if not orig_text:
                continue
            diffs.append(
                SectionDiff(
                    status="REMOVED",
                    heading=sec_a.heading,
                    original_text=orig_text,
                    section_index_a=a_idx,
                )
            )

    # 2. ADDED (in doc_b but not mapped)
    for b_idx, sec_b in enumerate(doc_b.sections):
        if b_idx not in mapped_b:
            rev_text = sec_b.to_text().strip()
            if not rev_text:
                continue
            diffs.append(
                SectionDiff(
                    status="ADDED",
                    heading=sec_b.heading,
                    revised_text=rev_text,
                    section_index_b=b_idx,
                )
            )

    # 3. MODIFIED (mapped pairs)
    for b_idx, a_idx in b_to_a.items():
        sec_a = doc_a.sections[a_idx]
        sec_b = doc_b.sections[b_idx]
        
        orig_text = sec_a.to_text().strip()
        rev_text = sec_b.to_text().strip()
        
        if orig_text == rev_text:
            continue
            
        # Find matches specifically for this pair to check if we should skip
        pair_matches = [
            m for m in matches 
            if m.chunk_a and m.chunk_b 
            and m.chunk_a.section_index == a_idx 
            and m.chunk_b.section_index == b_idx
        ]
        
        # Skip if all chunks are highly similar (minor formatting/whitespace)
        if pair_matches and all(m.fuzzy_score >= 0.75 and m.semantic_score >= 0.82 for m in pair_matches):
            continue

        highlighted_original, highlighted_revised = highlight_revised_only(
            orig_text, rev_text
        )

        diffs.append(
            SectionDiff(
                status="MODIFIED",
                heading=sec_b.heading or sec_a.heading,
                original_text=orig_text,
                revised_text=rev_text,
                highlighted_revised=highlighted_revised,
                highlighted_original=highlighted_original,
                section_index_a=a_idx,
                section_index_b=b_idx,
            )
        )

    # Sort: MODIFIED -> REMOVED -> ADDED
    _order = {"MODIFIED": 0, "REMOVED": 1, "ADDED": 2}
    diffs.sort(key=lambda d: (_order.get(d.status, 3), d.section_index_b if d.section_index_b >= 0 else d.section_index_a))

    return diffs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_section(doc: Document, index: int) -> DocumentSection | None:
    """Safely retrieve a section by index, returning *None* if out of range."""
    if 0 <= index < len(doc.sections):
        return doc.sections[index]
    return None
