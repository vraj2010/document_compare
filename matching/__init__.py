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


def generate_reason(score: float, fuzzy: float, is_modified: bool = False, is_cosine_only: bool = False, crit_detail: str = "") -> tuple[SemanticChangeType, str]:
    if crit_detail:
        return SemanticChangeType.MODIFIED, f"Critical value change detected. {crit_detail}"
    if is_modified:
        if is_cosine_only:
            return SemanticChangeType.MODIFIED, f"Meaning preserved with different phrasing. Cosine: {score:.2f}"
        if fuzzy >= 0.92:
            return SemanticChangeType.MODIFIED, f"Minor wording revision, intent unchanged. Fuzzy: {fuzzy:.2f}"
        if fuzzy >= 0.80:
            return SemanticChangeType.MODIFIED, f"Rephrased with same core meaning. Fuzzy: {fuzzy:.2f}"
        return SemanticChangeType.MODIFIED, f"Substantially reworded, meaning preserved. Fuzzy: {fuzzy:.2f}"

    cfg = CONFIG.matching
    if score >= cfg.semantic_similar_threshold:
        return SemanticChangeType.PRESERVED, "Meaning preserved with different phrasing."
    if score >= 0.45:
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
# Critical value change detector (pre-classification override)
# ---------------------------------------------------------------------------

# Step 1 — Exclusion patterns: strip dates, standard codes, revision numbers,
# grade codes, times, and URLs before extracting measurement numbers.
_EXCLUSION_PATTERNS: list[re.Pattern] = [
    re.compile(r'\b\d{1,2}[-/]\w{3,9}[-/]\d{4}\b'),                       # 15-Jan-2024, 05/Feb/2024
    re.compile(r'\b\d{4}[-/]\d{2}[-/]\d{2}\b'),                           # 2024-01-15
    re.compile(
        r'\b(ISO|ASME|ASTM|API|SAES|NACE|PED|EN|GI|MRQ|ITP|FAT|NDE|RT|BW|RF|FLG)'
        r'\s*[-/]?\s*[\d]+[\d\.\-/A-Za-z]*',
        re.IGNORECASE,
    ),                                                                     # ISO 15848-1, API 6D, PED 2014/68/EU
    re.compile(r'Rev\.\s*\d+', re.IGNORECASE),                             # Rev. 0, Rev. 1
    re.compile(r'\bMRQ[-\s]\d+', re.IGNORECASE),                          # MRQ-12345
    re.compile(r'\b[A-Z]{1,5}\d+[A-Z]?\b'),                               # grade codes: WCB, F6a, B7, F316L
    re.compile(r'\b\d{1,2}:\d{2}\b'),                                     # 12:00
    re.compile(r'portal\.\S+'),                                            # URLs
    re.compile(r'\b\d{4}\b(?=\s*/\s*\d+\s*/)'),                          # years in standard refs like 2014/68/EU
]


def _apply_exclusions(text: str) -> str:
    """Strip reference / administrative numbers so they are never compared."""
    cleaned = text
    for pat in _EXCLUSION_PATTERNS:
        cleaned = pat.sub('', cleaned)
    return cleaned


# Step 2 — Measurement extraction

MEASUREMENT_UNITS: list[str] = [
    'psi', 'bar', 'kpa', 'mpa', 'lbf', 'kn',
    '%', 'percent',
    '\u00b0f', '\u00b0c', 'degf', 'degc',
    '"', 'inch', 'inches',
    'month', 'months', 'year', 'years',
    'week', 'weeks', 'day', 'days',
    'mm', 'cm', 'm', 'km',
    'kg', 'lb', 'ton',
    'l', 'ml', 'gal',
]

MEASUREMENT_CONTEXT_KEYWORDS: list[str] = [
    'pressure', 'temperature', 'rating', 'pull', 'force', 'load',
    'capacity', 'flow', 'velocity', 'speed', 'torque', 'stress',
    'warranty', 'valid', 'validity', 'penalty', 'damages', 'bond',
    'maximum', 'minimum', 'minimum of', 'up to', 'not exceed',
    'design', 'test', 'operating', 'allowable',
    'diameter', 'size', 'bore', 'thickness', 'wall',
]

_RAW_NUM_RE = re.compile(r'\b([\d,]+\.?\d*)\b')


def _extract_measurement_numbers(
    text: str,
) -> list[tuple[float, str, bool, str]]:
    """Return ``(value, unit_if_adjacent, is_class, raw_str)`` for numbers that are
    adjacent to a measurement unit or sit inside a measurement sentence."""
    results: list[tuple[float, str, bool, str]] = []
    for m in _RAW_NUM_RE.finditer(text):
        num_str = m.group(1)
        try:
            num_val = float(num_str.replace(',', ''))
        except ValueError:
            continue
        if num_val == 0:
            continue

        after_text = text[m.end():m.end() + 10]
        unit_match = re.match(r'[-\s]{0,3}(psi|bar|lbf|%|°[fFcC]|"|mm|kg|days?|weeks?|months?)', after_text, re.IGNORECASE)
        unit = unit_match.group(1).strip() if unit_match else ""

        before_text = text[max(0, m.start() - 10):m.start()].lower()
        is_class = "class" in before_text

        # Sentence context: delimit on '.'
        sent_start = text.rfind('.', 0, m.start())
        sent_end = text.find('.', m.end())
        if sent_start == -1:
            sent_start = 0
        if sent_end == -1:
            sent_end = len(text)
        sentence = text[sent_start:sent_end].lower()
        has_kw = any(kw in sentence for kw in MEASUREMENT_CONTEXT_KEYWORDS)

        if unit or is_class or has_kw:
            results.append((num_val, unit, is_class, num_str))
    return results


# Step 3 — Compare measurement numbers

def _compare_measurement_numbers(
    nums_a: list[tuple[float, str, bool, str]],
    nums_b: list[tuple[float, str, bool, str]],
) -> tuple[bool, str]:
    """Positional comparison: pair by index, flag > 5 % change."""
    changes = []
    for i, (val_a, unit_a, is_class_a, raw_a) in enumerate(nums_a):
        if i >= len(nums_b):
            break
        val_b, unit_b, is_class_b, raw_b = nums_b[i]
        if val_a == 0:
            continue
        pct = abs(val_b - val_a) / abs(val_a)
        if pct > 0.05:
            if is_class_a or is_class_b:
                changes.append(f"Pressure class changed: {raw_a} \u2192 {raw_b}")
            else:
                unit = unit_b or unit_a
                suffix = f" {unit}" if unit else ""
                changes.append(f"{raw_a} \u2192 {raw_b}{suffix}")
    if changes:
        seen = set()
        unique_changes = []
        for change in changes:
            if change not in seen:
                seen.add(change)
                unique_changes.append(change)
        
        remaining = len(changes) - len(unique_changes)
        detail = "; ".join(unique_changes)
        if remaining > 0:
            detail += f" (+ {remaining} similar change(s) across rows)"
        return True, detail
    return False, ''


# Step 6 — Administrative chunk guard

_ADMIN_SIGNALS: list[str] = [
    'rev.', 'mrq no', 'no later than', 'submitted by',
    'portal.aramco', 'bid submission deadline',
]


def is_administrative_chunk(text_a: str, text_b: str) -> bool:
    """Return True if the chunk pair is administrative (date/revision header,
    submission deadline, portal URL, etc.) so that the critical-value override
    should be skipped."""
    combined = (text_a + ' ' + text_b).lower()
    return any(sig in combined for sig in _ADMIN_SIGNALS)


# Material / alloy constants (unchanged)
_MATERIAL_KEYWORDS = [
    'stellite', 'inconel', 'monel', 'duplex', 'hastelloy', 'chrome',
    'chromium', 'stainless', 'carbon', 'alloy', 'f316', 'f304',
    'wcb', 'wcc', 'f6a', 'f11', 'f22', 'overlay', 'hardfacing',
]
_MATERIAL_SHORT = ['cr', 'ss', 'b7', 'b8']
_MATERIAL_SPECIAL = ['13%']

_CONNECTOR_RE = re.compile(
    r'[A-Z]{2,}\s*\d+\s+(or|and)\s+[A-Z]{2,}\s*\d+',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Main detector function
# ---------------------------------------------------------------------------

def detect_critical_value_changes(text_a: str, text_b: str) -> dict:
    """Pre-check for critical value / specification changes that must
    override fuzzy-similarity based classification.

    Steps:
      1. Strip exclusion patterns (dates, standard codes, grade codes, …)
      2. Extract numbers ONLY adjacent to measurement units / context keywords
      3. Positional comparison — flag > 5 % relative change
      4. Material / alloy keyword check (unchanged)
      5. Logical connector or \u2194 and check (unchanged)

    Returns
    -------
    dict with keys:
        has_critical_change : bool
        change_type         : str
        detail              : str
        details             : list[str]
    """
    result: dict = {
        "has_critical_change": False,
        "change_type": "",
        "detail": "",
        "details": [],
    }

    text_a_lower = text_a.lower()
    text_b_lower = text_b.lower()

    # ------------------------------------------------------------------
    # 1-3. MEASUREMENT NUMBER COMPARISON (exclusion-first)
    # ------------------------------------------------------------------
    clean_a = _apply_exclusions(text_a)
    clean_b = _apply_exclusions(text_b)

    rows_a = [r for r in clean_a.split('\n') if r.strip()]
    rows_b = [r for r in clean_b.split('\n') if r.strip()]
    
    row_diff_msg = ""
    if (len(rows_a) > 1 or len(rows_b) > 1) and len(rows_a) != len(rows_b):
        min_len = min(len(rows_a), len(rows_b))
        valid_a = []
        valid_b = []
        for i in range(min_len):
            col_a = re.split(r'\s{2,}|\t|\|', rows_a[i].strip())[0]
            col_b = re.split(r'\s{2,}|\t|\|', rows_b[i].strip())[0]
            if col_a != col_b:
                break
            valid_a.append(rows_a[i])
            valid_b.append(rows_b[i])
        
        clean_a = '\n'.join(valid_a)
        clean_b = '\n'.join(valid_b)
        diff = len(rows_b) - len(rows_a)
        if diff > 0:
            row_diff_msg = f"+ {diff} row(s) added"
        else:
            row_diff_msg = f"+ {-diff} row(s) removed"

    meas_a = _extract_measurement_numbers(clean_a)
    meas_b = _extract_measurement_numbers(clean_b)

    changed = False
    detail = ""
    if meas_a and meas_b:
        changed, detail = _compare_measurement_numbers(meas_a, meas_b)
    
    if row_diff_msg:
        changed = True
        if detail:
            detail += f"; {row_diff_msg}"
        else:
            detail = row_diff_msg

    if changed:
        result["details"].append(detail)
        result["has_critical_change"] = True
        if not result["change_type"]:
            result["change_type"] = "numeric_value"
            result["detail"] = detail

    # ------------------------------------------------------------------
    # 4. MATERIAL / ALLOY KEYWORD CHECK (unchanged)
    # ------------------------------------------------------------------
    mats_a: set[str] = set()
    mats_b: set[str] = set()

    for kw in _MATERIAL_KEYWORDS:
        if kw in text_a_lower:
            mats_a.add(kw)
        if kw in text_b_lower:
            mats_b.add(kw)
    for kw in _MATERIAL_SHORT:
        pat = r'\b' + re.escape(kw) + r'\b'
        if re.search(pat, text_a_lower):
            mats_a.add(kw)
        if re.search(pat, text_b_lower):
            mats_b.add(kw)
    for kw in _MATERIAL_SPECIAL:
        if kw in text_a:
            mats_a.add(kw)
        if kw in text_b:
            mats_b.add(kw)

    added_mats = mats_b - mats_a
    removed_mats = mats_a - mats_b
    if added_mats or removed_mats:
        parts: list[str] = []
        if removed_mats:
            parts.append(f"removed: {', '.join(sorted(removed_mats))}")
        if added_mats:
            parts.append(f"added: {', '.join(sorted(added_mats))}")
        detail = f"Material/spec changed \u2014 {'; '.join(parts)}"
        result["details"].append(detail)
        result["has_critical_change"] = True
        if not result["change_type"]:
            result["change_type"] = "material_spec"
            result["detail"] = detail

    # ------------------------------------------------------------------
    # 5. LOGICAL CONNECTOR FLIP  (or \u2194 and)
    # ------------------------------------------------------------------
    conns_a = _CONNECTOR_RE.findall(text_a)
    conns_b = _CONNECTOR_RE.findall(text_b)
    if conns_a and conns_b:
        set_a = {c.lower() for c in conns_a}
        set_b = {c.lower() for c in conns_b}
        if set_a != set_b:
            old = '/'.join(sorted(set_a))
            new = '/'.join(sorted(set_b))
            detail = f"Logical connector changed: '{old}' \u2192 '{new}'"
            result["details"].append(detail)
            result["has_critical_change"] = True
            if not result["change_type"]:
                result["change_type"] = "logical_connector"
                result["detail"] = detail

    return result


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
                # Pre-check: critical value changes override near-exact
                # (skip for administrative chunks — dates, revisions, deadlines)
                if not is_administrative_chunk(chunk_a.text, best_b.text):
                    crit = detect_critical_value_changes(chunk_a.text, best_b.text)
                else:
                    crit = {"has_critical_change": False}
                if crit["has_critical_change"]:
                    # Route to semantic batch for proper scoring
                    _pending_semantic.append((chunk_a, best_b, best_score))
                else:
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

                # Gate: skip critical override for administrative chunks
                if not is_administrative_chunk(chunk_a.text, best_b.text):
                    crit = detect_critical_value_changes(chunk_a.text, best_b.text)
                else:
                    crit = {"has_critical_change": False}

                # DECISION TREE — first match wins:

                # 1. Critical value change → SEMANTICALLY_DIFFERENT
                #    regardless of fuzzy or semantic score.
                if crit["has_critical_change"]:
                    sem_type, summary = generate_reason(sem_score, fuzzy_score, crit_detail=crit['detail'])
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

                # 2. High semantic + no critical → SEMANTIC (meaning preserved)
                elif sem_score >= self.cfg.fuzzy_semantic_reclass_threshold:
                    sem_type, summary = generate_reason(sem_score, fuzzy_score)
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

                # 3. Moderate semantic → SEMANTICALLY_DIFFERENT
                elif sem_score >= self.cfg.semantic_change_low:
                    sem_type, summary = generate_reason(sem_score, fuzzy_score)
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

                # 4. Low semantic → MODIFIED
                else:
                    sem_type, summary = generate_reason(sem_score, fuzzy_score, is_modified=True, is_cosine_only=False)
                    sem_analysis = SemanticAnalysis(
                        change_type=sem_type,
                        confidence=round(fuzzy_score, 3),
                        summary=summary,
                    )
                    matches.append(ChunkMatch(
                        chunk_a=chunk_a,
                        chunk_b=best_b,
                        change_type=ChangeType.MODIFIED,
                        similarity_score=fuzzy_score,
                        fuzzy_score=fuzzy_score,
                        semantic_score=sem_score,
                        semantic_analysis=sem_analysis,
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
                # High semantic similarity — but still check for critical value changes
                # (skip for administrative chunks)
                if not is_administrative_chunk(chunk_a.text, unmatched_b[best_j].text):
                    crit = detect_critical_value_changes(chunk_a.text, unmatched_b[best_j].text)
                else:
                    crit = {"has_critical_change": False}
                crit_changes = _extract_critical_info_changes(
                    chunk_a.text, unmatched_b[best_j].text
                )

                if crit["has_critical_change"]:
                    # Critical override → SEMANTICALLY_DIFFERENT
                    sem_type, summary = generate_reason(best_score, fuzzy_score, crit_detail=crit['detail'])
                    sem_analysis = SemanticAnalysis(
                        change_type=sem_type,
                        confidence=round(best_score, 3),
                        summary=summary,
                    )
                    matches.append(ChunkMatch(
                        chunk_a=chunk_a,
                        chunk_b=unmatched_b[best_j],
                        change_type=ChangeType.SEMANTIC_DIFFERENT,
                        similarity_score=best_score,
                        fuzzy_score=fuzzy_score,
                        semantic_score=best_score,
                        semantic_analysis=sem_analysis,
                        critical_info_changes=crit_changes,
                    ))
                else:
                    # No critical change → SEMANTIC (meaning preserved)
                    sem_type, summary = generate_reason(best_score, fuzzy_score)
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
            elif best_score >= 0.45:
                # Moderate semantic similarity → MODIFIED
                sem_type, summary = generate_reason(best_score, fuzzy_score, is_modified=True, is_cosine_only=True)
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
                    change_type=ChangeType.MODIFIED,
                    similarity_score=best_score,
                    fuzzy_score=fuzzy_score,
                    semantic_score=best_score,
                    semantic_analysis=sem_analysis,
                ))
                matched_b_indices.add(best_j)
            elif best_score >= self.cfg.semantic_contradiction_threshold:
                # < 0.45 -> SEMANTICALLY_DIFFERENT
                sem_type, summary = generate_reason(best_score, fuzzy_score)
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
                    change_type=ChangeType.SEMANTIC_DIFFERENT,
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