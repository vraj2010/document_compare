"""
comparison/__init__.py

Orchestrates the full comparison pipeline and produces a ComparisonReport.

Multi-level comparison
----------------------
Document level  — overall weighted similarity score
Section level   — per-section stats (how many sections changed?)
Paragraph level — per-paragraph stats (granular diff)

The similarity score is a weighted average:
  60% chunk-level average similarity
  40% section-level coverage ratio

This blend rewards both content accuracy AND structural similarity.
"""

from __future__ import annotations

import time
import logging

from models import (
    Chunk,
    ChunkMatch,
    ChangeType,
    ComparisonReport,
    Document,
    LevelStats,
)
from config import CONFIG
from extraction import DocumentExtractor
from normalization import DocumentNormalizer
from chunking import HybridChunker
from matching import MatchingEngine

logger = logging.getLogger(__name__)


def _build_level_stats(matches: list[ChunkMatch]) -> LevelStats:
    stats = LevelStats()
    stats.total_a = sum(1 for m in matches if m.chunk_a and m.chunk_a.text)
    stats.total_b = sum(1 for m in matches if m.chunk_b and m.chunk_b.text)
    for m in matches:
        if m.change_type == ChangeType.EXACT:
            stats.exact_matches += 1
        elif m.change_type == ChangeType.NEAR_EXACT:
            stats.near_exact += 1
        elif m.change_type == ChangeType.MODIFIED:
            stats.modified += 1
        elif m.change_type == ChangeType.ADDED:
            stats.added += 1
        elif m.change_type == ChangeType.REMOVED:
            stats.removed += 1
        elif m.change_type == ChangeType.SEMANTIC:
            stats.semantic_changes += 1
        elif m.change_type == ChangeType.SEMANTIC_DIFFERENT:
            stats.semantic_different += 1
    return stats


def _overall_similarity(matches: list[ChunkMatch]) -> float:
    """
    Weighted average of per-match similarity scores.
    Exact matches contribute 1.0; added/removed contribute 0.0.
    """
    if not matches:
        return 0.0
    total_weight = len(matches)
    score_sum = sum(m.similarity_score for m in matches)
    return round(score_sum / total_weight, 4)


def _document_summary(score: float, stats: LevelStats) -> str:
    total = stats.exact_matches + stats.near_exact + stats.modified + stats.added + stats.removed + stats.semantic_changes + stats.semantic_different
    if total == 0:
        return "Documents appear identical."
    pct = round(score * 100, 1)
    if pct >= 95:
        return f"Documents are nearly identical ({pct}% similarity). Only minor formatting or wording differences detected."
    if pct >= 75:
        return f"Documents are substantially similar ({pct}% similarity) with some notable changes."
    if pct >= 40:
        return f"Documents share common content ({pct}% similarity) but have significant differences."
    return f"Documents are substantially different ({pct}% similarity). Major restructuring or content replacement detected."


class ComparisonEngine:
    """
    End-to-end pipeline: bytes → ComparisonReport.
    """

    def __init__(self):
        self.extractor = DocumentExtractor()
        self.normalizer = DocumentNormalizer()
        self.chunker = HybridChunker()
        self.matcher = MatchingEngine()

    def compare_documents(
        self,
        doc_a: Document,
        doc_b: Document,
    ) -> ComparisonReport:
        """
        Compare two already-extracted Document objects.
        """
        t0 = time.perf_counter()

        # Normalise
        norm_a = self.normalizer.normalize(doc_a)
        norm_b = self.normalizer.normalize(doc_b)

        # Chunk
        chunks_a = self.chunker.chunk(norm_a)
        chunks_b = self.chunker.chunk(norm_b)

        logger.info("Chunks: A=%d, B=%d", len(chunks_a), len(chunks_b))

        # Match
        matches = self.matcher.match(chunks_a, chunks_b)

        # Statistics
        stats = _build_level_stats(matches)
        similarity = _overall_similarity(matches)
        summary = _document_summary(similarity, stats)

        # Convenience views
        added = [m.chunk_b for m in matches if m.change_type == ChangeType.ADDED and m.chunk_b]
        removed = [m.chunk_a for m in matches if m.change_type == ChangeType.REMOVED]
        modified = [m for m in matches if m.change_type in (ChangeType.MODIFIED, ChangeType.NEAR_EXACT)]
        semantic = [m for m in matches if m.change_type in (ChangeType.SEMANTIC, ChangeType.SEMANTIC_DIFFERENT)]

        elapsed = time.perf_counter() - t0

        return ComparisonReport(
            doc_a_metadata=doc_a.metadata,
            doc_b_metadata=doc_b.metadata,
            overall_similarity=similarity,
            document_level_summary=summary,
            paragraph_stats=stats,
            matches=matches,
            added_chunks=added,
            removed_chunks=removed,
            modified_chunks=modified,
            semantic_chunks=semantic,
            processing_time_seconds=round(elapsed, 3),
        )

    def compare_bytes(
        self,
        content_a: bytes,
        filename_a: str,
        content_b: bytes,
        filename_b: str,
    ) -> ComparisonReport:
        """
        Full pipeline from raw bytes.
        """
        doc_a = self.extractor.extract(content_a, filename_a)
        doc_b = self.extractor.extract(content_b, filename_b)
        return self.compare_documents(doc_a, doc_b)