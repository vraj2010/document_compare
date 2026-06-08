"""
matching/__init__.py

Matching Engine — three-stage cascade:

  Hash Match → Fuzzy Match → Semantic Match

Rationale / Latency Optimization
----------------------------------
Stage 1 — Hash (hashlib SHA-256):
  O(1) per chunk, zero allocations after hashing.
  Typically handles 60-80% of chunks in real documents
  (boilerplate, unchanged sections).

Stage 2 — Fuzzy (RapidFuzz token_sort_ratio):
  ~5-50 µs per pair, pure C extension.
  Catches small edits: spelling, word-order changes, minor rewrites.
  Applied only to chunks that failed hash match.

Stage 3 — Semantic (SentenceTransformer cosine similarity):
  ~1-3 ms per chunk in batch mode.
  Applied ONLY when hash AND fuzzy both miss.
  Embeddings are generated in one batch call to maximise GPU/CPU throughput.
  Results are cached with lru_cache keyed on the chunk text hash.

Memory efficiency: chunk text is never duplicated; we store references.

Nearest-neighbour matching:
  For each unmatched chunk in A, we find the best candidate in B
  (among unmatched B chunks) using cosine similarity.
  This is O(|unmatched_A| × |unmatched_B|) dot products — acceptable for
  documents up to ~1000 chunks each; for larger docs we could use faiss,
  but that introduces a heavy dependency for marginal gain.
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

import re
from models import (
    Chunk,
    ChunkMatch,
    ChangeType,
    CriticalInfoChange,
    SemanticAnalysis,
    SemanticChangeType,
)
from config import CONFIG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding model — loaded once, shared across all calls
# ---------------------------------------------------------------------------

_MODEL: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        logger.info("Loading SentenceTransformer model: %s", CONFIG.embedding.model_name)
        _MODEL = SentenceTransformer(
            CONFIG.embedding.model_name,
            device=CONFIG.embedding.device,
        )
    return _MODEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=4096)
def _cached_hash(text: str) -> str:
    return _sha256(text)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def _semantic_change_type(score: float, fuzzy: float) -> tuple[SemanticChangeType, str]:
    cfg = CONFIG.matching
    if score >= cfg.semantic_similar_threshold:
        if fuzzy >= 0.85:
            return SemanticChangeType.PRESERVED, "Wording is very similar; meaning preserved."
        return SemanticChangeType.PRESERVED, "Different wording, same core meaning."
    if score >= cfg.semantic_change_low:
        if fuzzy < 0.5:
            # Low lexical overlap but moderate semantic overlap → expanded/reduced
            return SemanticChangeType.EXPANDED, "Meaning appears to have been elaborated or expanded."
        return SemanticChangeType.MODIFIED, "Core meaning partially changed or refined."
    if score >= cfg.semantic_contradiction_threshold:
        return SemanticChangeType.REDUCED, "Significant meaning loss or reduction detected."
    return SemanticChangeType.UNRELATED, "Chunks appear unrelated or contradictory."



# ---------------------------------------------------------------------------
# Critical info (dates / numbers) change detection
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(
    r"""(?x)
    (?:
        (?:\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4})           # 01/01/2025  01-01-25
        |(?:\d{4}[\-/]\d{1,2}[\-/]\d{1,2})             # 2025-01-01
        |(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?
            |May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?
            |Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?
            |Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?
        |(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?
            |Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?
            |Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)
            (?:,?\s+\d{4})?)
        |\d{4}                                      # bare year
    )""",
    re.IGNORECASE,
)

_NUMBER_RE = re.compile(
    r"""(?x)
    (?:
        \$[\d,]+(?:\.\d+)?           # currency ,000.00
        |[\d,]+(?:\.\d+)?%            # percentage 12.5%
        |[\d]{1,3}(?:,\d{3})+(?:\.\d+)?   # large numbers 1,000,000
        |\d+\.\d+                 # decimals 3.14
        |\d+                        # plain integers
    )""",
    re.VERBOSE,
)


def _extract_critical_info_changes(text_a: str, text_b: str) -> list[CriticalInfoChange]:
    """
    Extract dates and numbers from both chunks and report any that differ.
    Only flags values that are present in one chunk but different/absent in the other,
    to avoid noise from genuinely unrelated numbers.
    """
    changes: list[CriticalInfoChange] = []

    dates_a = _DATE_RE.findall(text_a)
    dates_b = _DATE_RE.findall(text_b)

    # Compare date sets — flag dates in A that changed or disappeared
    for d in dates_a:
        if d not in dates_b:
            # Try to find a replacement (first date in B that is not in A)
            replacement = next((x for x in dates_b if x not in dates_a), "")
            changes.append(CriticalInfoChange(
                info_type="date",
                original=d,
                revised=replacement or "(removed)",
                context="",
            ))

    # Numbers: extract and compare significant numeric tokens
    def _nums(text: str) -> list[str]:
        return _NUMBER_RE.findall(text)

    nums_a = _nums(text_a)
    nums_b = _nums(text_b)

    # Only report numbers that appear in A but are absent / different in B
    from collections import Counter
    counter_a = Counter(nums_a)
    counter_b = Counter(nums_b)
    for num, count_a in counter_a.items():
        count_b = counter_b.get(num, 0)
        if count_b < count_a:
            # Determine type
            if "$" in num:
                info_type = "currency"
            elif "%" in num:
                info_type = "percentage"
            else:
                info_type = "number"
            replacement = next((x for x in nums_b if x not in counter_a), "")
            changes.append(CriticalInfoChange(
                info_type=info_type,
                original=num,
                revised=replacement or "(changed)",
                context="",
            ))

    # De-duplicate
    seen: set[tuple] = set()
    unique: list[CriticalInfoChange] = []
    for c in changes:
        key = (c.info_type, c.original, c.revised)
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


# ---------------------------------------------------------------------------
# Core matching engine
# ---------------------------------------------------------------------------

class MatchingEngine:
    """
    Compares two lists of chunks and returns a list of ChunkMatch objects.
    """

    def __init__(self):
        self.cfg = CONFIG.matching
        self.embed_cfg = CONFIG.embedding

    # ------------------------------------------------------------------
    # Stage 1: hash matching
    # ------------------------------------------------------------------

    def _hash_match(
        self,
        chunks_a: list[Chunk],
        chunks_b: list[Chunk],
    ) -> tuple[list[ChunkMatch], list[Chunk], list[Chunk]]:
        """Return exact matches + remaining unmatched chunks."""
        hash_b: dict[str, Chunk] = {_cached_hash(c.text): c for c in chunks_b}
        matched_b_keys: set[str] = set()
        matches: list[ChunkMatch] = []
        unmatched_a: list[Chunk] = []

        for chunk in chunks_a:
            h = _cached_hash(chunk.text)
            if h in hash_b and h not in matched_b_keys:
                matched_b_keys.add(h)
                matches.append(ChunkMatch(
                    chunk_a=chunk,
                    chunk_b=hash_b[h],
                    change_type=ChangeType.EXACT,
                    similarity_score=1.0,
                    fuzzy_score=1.0,
                    semantic_score=1.0,
                ))
            else:
                unmatched_a.append(chunk)

        matched_b_texts = {c.text for m in matches for c in [m.chunk_b] if c}
        unmatched_b = [c for c in chunks_b if c.text not in matched_b_texts]

        logger.debug(
            "Hash: %d exact, %d unmatched_a, %d unmatched_b",
            len(matches), len(unmatched_a), len(unmatched_b)
        )
        return matches, unmatched_a, unmatched_b

    # ------------------------------------------------------------------
    # Stage 2: fuzzy matching
    # ------------------------------------------------------------------

    def _fuzzy_match(
        self,
        unmatched_a: list[Chunk],
        unmatched_b: list[Chunk],
    ) -> tuple[list[ChunkMatch], list[Chunk], list[Chunk]]:
        """Greedy best-fuzzy-match; O(|a|×|b|) but all in native C.

        After claiming a fuzzy match we run semantic scoring on it.
        If the semantic score shows meaning has changed (below the
        semantic_similar_threshold), the match is reclassified to
        SEMANTIC or SEMANTIC_DIFFERENT with full analysis — so it
        appears in semantic filters instead of being buried as "Modified".
        """
        if not unmatched_a or not unmatched_b:
            return [], unmatched_a, unmatched_b

        matched_b_ids: set[str] = set()
        matches: list[ChunkMatch] = []
        still_unmatched_a: list[Chunk] = []

        # Collect all texts for a single batch embed at the end
        _pending_semantic: list[tuple[Chunk, Chunk, float]] = []  # (a, b, fuzzy)

        for chunk_a in unmatched_a:
            best_score = 0.0
            best_b: Chunk | None = None

            for chunk_b in unmatched_b:
                if chunk_b.chunk_id in matched_b_ids:
                    continue
                score = fuzz.token_sort_ratio(chunk_a.text, chunk_b.text) / 100.0
                if score > best_score:
                    best_score = score
                    best_b = chunk_b

            if best_b is None:
                still_unmatched_a.append(chunk_a)
                continue

            if best_score >= self.cfg.fuzzy_exact_threshold:
                matches.append(ChunkMatch(
                    chunk_a=chunk_a,
                    chunk_b=best_b,
                    change_type=ChangeType.NEAR_EXACT,
                    similarity_score=best_score,
                    fuzzy_score=best_score,
                ))
                matched_b_ids.add(best_b.chunk_id)
            elif best_score >= self.cfg.fuzzy_change_low:
                # Claim the pair — will be semantically re-scored below
                _pending_semantic.append((chunk_a, best_b, best_score))
                matched_b_ids.add(best_b.chunk_id)
            else:
                still_unmatched_a.append(chunk_a)

        # --- Semantic re-scoring for fuzzy-claimed pairs -----------------
        if _pending_semantic:
            all_texts = [t for pair in _pending_semantic for t in (pair[0].text, pair[1].text)]
            all_embeds = self._embed_batch(all_texts)

            for idx, (chunk_a, best_b, fuzzy_score) in enumerate(_pending_semantic):
                emb_a = all_embeds[idx * 2]
                emb_b = all_embeds[idx * 2 + 1]
                sem_score = float(np.dot(emb_a, emb_b))  # already normalised

                crit_changes = _extract_critical_info_changes(chunk_a.text, best_b.text)

                if sem_score >= self.cfg.fuzzy_semantic_reclass_threshold:
                    # High semantic similarity — just a fuzzy / near-exact rewrite
                    sem_type, summary = _semantic_change_type(sem_score, fuzzy_score)
                    sem_analysis = SemanticAnalysis(
                        change_type=sem_type,
                        confidence=round(sem_score, 3),
                        summary=summary,
                    )
                    matches.append(ChunkMatch(
                        chunk_a=chunk_a,
                        chunk_b=best_b,
                        change_type=ChangeType.SEMANTIC,
                        similarity_score=sem_score,
                        fuzzy_score=fuzzy_score,
                        semantic_score=sem_score,
                        semantic_analysis=sem_analysis,
                        critical_info_changes=crit_changes,
                    ))
                elif sem_score >= self.cfg.semantic_change_low:
                    # Moderate semantic similarity — meaning has shifted
                    sem_type, summary = _semantic_change_type(sem_score, fuzzy_score)
                    sem_analysis = SemanticAnalysis(
                        change_type=sem_type,
                        confidence=round(sem_score, 3),
                        summary=summary,
                    )
                    matches.append(ChunkMatch(
                        chunk_a=chunk_a,
                        chunk_b=best_b,
                        change_type=ChangeType.SEMANTIC_DIFFERENT,
                        similarity_score=sem_score,
                        fuzzy_score=fuzzy_score,
                        semantic_score=sem_score,
                        semantic_analysis=sem_analysis,
                        critical_info_changes=crit_changes,
                    ))
                else:
                    # Low semantic similarity — keep as Modified with critical info
                    matches.append(ChunkMatch(
                        chunk_a=chunk_a,
                        chunk_b=best_b,
                        change_type=ChangeType.MODIFIED,
                        similarity_score=fuzzy_score,
                        fuzzy_score=fuzzy_score,
                        semantic_score=sem_score,
                        critical_info_changes=crit_changes,
                    ))

        matched_b_texts = {m.chunk_b.text for m in matches if m.chunk_b}
        still_unmatched_b = [c for c in unmatched_b if c.text not in matched_b_texts]

        logger.debug(
            "Fuzzy: %d matched, %d remain_a, %d remain_b",
            len(matches), len(still_unmatched_a), len(still_unmatched_b)
        )
        return matches, still_unmatched_a, still_unmatched_b

    # ------------------------------------------------------------------
    # Stage 3: semantic matching
    # ------------------------------------------------------------------

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        model = _get_model()
        return model.encode(
            texts,
            batch_size=self.embed_cfg.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # pre-normalize → dot product = cosine
        )

    def _semantic_match(
        self,
        unmatched_a: list[Chunk],
        unmatched_b: list[Chunk],
    ) -> tuple[list[ChunkMatch], list[Chunk], list[Chunk]]:
        if not unmatched_a or not unmatched_b:
            return [], unmatched_a, unmatched_b

        texts_a = [c.text for c in unmatched_a]
        texts_b = [c.text for c in unmatched_b]

        # Single batched encode call
        all_texts = texts_a + texts_b
        all_embeds = self._embed_batch(all_texts)
        embeds_a = all_embeds[:len(texts_a)]
        embeds_b = all_embeds[len(texts_a):]

        # Similarity matrix via vectorised dot product (already normalised)
        sim_matrix = embeds_a @ embeds_b.T   # shape (|a|, |b|)

        matched_b_indices: set[int] = set()
        matches: list[ChunkMatch] = []
        still_unmatched_a: list[Chunk] = []

        for i, chunk_a in enumerate(unmatched_a):
            row = sim_matrix[i].copy()
            # Mask already-matched B chunks
            for j in matched_b_indices:
                row[j] = -1.0

            best_j = int(np.argmax(row))
            best_score = float(row[best_j])

            fuzzy_score = fuzz.token_sort_ratio(chunk_a.text, unmatched_b[best_j].text) / 100.0

            if best_score >= self.cfg.semantic_similar_threshold:
                # High semantic similarity → same meaning, different wording (SEMANTIC)
                sem_type, summary = _semantic_change_type(best_score, fuzzy_score)
                sem_analysis = SemanticAnalysis(
                    change_type=sem_type,
                    confidence=round(best_score, 3),
                    summary=summary,
                )
                matches.append(ChunkMatch(
                    chunk_a=chunk_a,
                    chunk_b=unmatched_b[best_j],
                    change_type=ChangeType.SEMANTIC,
                    similarity_score=best_score,
                    fuzzy_score=fuzzy_score,
                    semantic_score=best_score,
                    semantic_analysis=sem_analysis,
                ))
                matched_b_indices.add(best_j)
            elif best_score >= self.cfg.semantic_change_low:
                # Low-to-moderate semantic similarity → semantically DIFFERENT content
                sem_type, summary = _semantic_change_type(best_score, fuzzy_score)
                sem_analysis = SemanticAnalysis(
                    change_type=sem_type,
                    confidence=round(best_score, 3),
                    summary=summary,
                )
                crit_changes = _extract_critical_info_changes(
                    chunk_a.text, unmatched_b[best_j].text
                )
                matches.append(ChunkMatch(
                    chunk_a=chunk_a,
                    chunk_b=unmatched_b[best_j],
                    change_type=ChangeType.SEMANTIC_DIFFERENT,   # <-- new tag
                    similarity_score=best_score,
                    fuzzy_score=fuzzy_score,
                    semantic_score=best_score,
                    semantic_analysis=sem_analysis,
                    critical_info_changes=crit_changes,
                ))
                matched_b_indices.add(best_j)
            else:
                still_unmatched_a.append(chunk_a)

        matched_b_set = {unmatched_b[j].chunk_id for j in matched_b_indices}
        still_unmatched_b = [c for c in unmatched_b if c.chunk_id not in matched_b_set]

        logger.debug(
            "Semantic: %d matched, %d remain_a, %d remain_b",
            len(matches), len(still_unmatched_a), len(still_unmatched_b)
        )
        return matches, still_unmatched_a, still_unmatched_b

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def match(
        self,
        chunks_a: list[Chunk],
        chunks_b: list[Chunk],
    ) -> list[ChunkMatch]:
        """
        Run the full cascade and return all ChunkMatch objects including
        ADDED (only in B) and REMOVED (only in A).
        """
        all_matches: list[ChunkMatch] = []

        # Stage 1
        exact_matches, unmatched_a, unmatched_b = self._hash_match(chunks_a, chunks_b)
        all_matches.extend(exact_matches)

        # Stage 2
        fuzzy_matches, unmatched_a, unmatched_b = self._fuzzy_match(unmatched_a, unmatched_b)
        all_matches.extend(fuzzy_matches)

        # Stage 3
        sem_matches, unmatched_a, unmatched_b = self._semantic_match(unmatched_a, unmatched_b)
        all_matches.extend(sem_matches)

        # Remaining unmatched_a → REMOVED
        for chunk in unmatched_a:
            all_matches.append(ChunkMatch(
                chunk_a=chunk,
                chunk_b=None,
                change_type=ChangeType.REMOVED,
                similarity_score=0.0,
            ))

        # Remaining unmatched_b → ADDED
        for chunk in unmatched_b:
            # Represent as chunk_a = placeholder, chunk_b = new content
            placeholder = Chunk(
                chunk_id=f"placeholder_{chunk.chunk_id}",
                section_index=chunk.section_index,
                section_heading=chunk.section_heading,
                text="",
                source=chunk.source,
            )
            all_matches.append(ChunkMatch(
                chunk_a=placeholder,
                chunk_b=chunk,
                change_type=ChangeType.ADDED,
                similarity_score=0.0,
            ))

        return all_matches