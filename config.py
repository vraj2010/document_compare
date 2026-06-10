"""
config.py — Central configuration for the Document Comparison System.

All thresholds, model names, and tuning knobs live here so nothing is
hard-coded across the codebase.
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Matching thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MatchingConfig:
    # A fuzzy ratio >= this value is treated as a "near-exact" match
    fuzzy_exact_threshold: float = 0.90

    # A fuzzy ratio in [fuzzy_change_low, fuzzy_exact_threshold) means "modified".
    # Raised from 0.60 → 0.75 so that loosely-similar chunks fall through to the
    # semantic stage instead of being prematurely claimed as "Modified".
    fuzzy_change_low: float = 0.75

    # Cosine similarity >= this is "semantically similar" (same meaning, diff wording)
    semantic_similar_threshold: float = 0.82

    # Cosine similarity in [semantic_change_low, semantic_similar_threshold)
    # means "meaning modified / different"
    semantic_change_low: float = 0.50

    # Cosine similarity below this → likely contradiction / unrelated
    semantic_contradiction_threshold: float = 0.20

    # For "Modified" fuzzy matches: if the semantic score is below this threshold
    # the chunk is re-classified as SEMANTIC / SEMANTIC_DIFFERENT with full analysis
    # instead of staying as a plain "Modified" with no semantic insight.
    fuzzy_semantic_reclass_threshold: float = 0.82


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkingConfig:
    # Strategy: "section" | "hybrid"
    strategy: str = "hybrid"

    # Each paragraph becomes its own chunk up to this token limit.
    # Lowered from 300 → 80 so individual sentences/clauses are compared
    # independently instead of being merged into one big blob — this lets
    # the semantic stage surface fine-grained meaning changes.
    max_chunk_tokens: int = 80

    # Minimum paragraph length (chars) to be considered its own chunk.
    # Lowered from 40 → 20 to keep short but meaningful clauses.
    min_paragraph_chars: int = 20


# ---------------------------------------------------------------------------
# Embedding / SentenceTransformer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbeddingConfig:
    # Lightweight, fast, high-quality for English & multilingual text
    model_name: str = "all-MiniLM-L6-v2"

    # Batch size for embedding generation
    batch_size: int = 64

    # Device: "cpu" is safest for broad deployment; change to "cuda" if GPU present
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReportingConfig:
    # Maximum characters of text shown in diff snippets in the HTML report
    snippet_max_chars: int = 500


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExtractionConfig:
    # Fallback: plain-text files are read with this encoding
    txt_encoding: str = "utf-8"


# ---------------------------------------------------------------------------
# Top-level config singleton
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AppConfig:
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)


# Single importable instance
CONFIG = AppConfig()
