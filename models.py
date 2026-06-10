"""
models.py — Pydantic models that form the unified internal document
representation consumed by every stage of the pipeline.

Design rationale
----------------
A single canonical structure means each pipeline stage (extraction,
normalisation, chunking, matching) only needs to understand one schema.
Pydantic gives us free validation, serialisation, and IDE auto-complete.
"""

from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Primitive building blocks
# ---------------------------------------------------------------------------

class TableCell(BaseModel):
    row: int
    col: int
    text: str


class Table(BaseModel):
    caption: str = ""
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)

    def to_text(self) -> str:
        """Flatten table to a normalisation-friendly string."""
        parts: list[str] = []
        if self.caption:
            parts.append(self.caption)
        if self.headers:
            parts.append(" | ".join(self.headers))
        for row in self.rows:
            parts.append(" | ".join(row))
        return "\n".join(parts)


class DocumentSection(BaseModel):
    heading: str = ""
    heading_level: int = 0          # 0 = no heading, 1 = H1, 2 = H2, …
    page_start: int = 0
    page_end: int = 0
    paragraphs: list[str] = Field(default_factory=list)
    lists: list[list[str]] = Field(default_factory=list)   # each inner list = one bullet list
    tables: list[Table] = Field(default_factory=list)

    def to_text(self) -> str:
        parts: list[str] = []
        if self.heading:
            parts.append(self.heading)
        parts.extend(self.paragraphs)
        for lst in self.lists:
            parts.extend(lst)
        for tbl in self.tables:
            parts.append(tbl.to_text())
        return "\n".join(parts)


class DocumentMetadata(BaseModel):
    filename: str = ""
    file_type: str = ""             # "pdf" | "docx" | "txt"
    page_count: int = 0
    missing_pages: list[int] = Field(default_factory=list)
    word_count: int = 0
    char_count: int = 0


class Document(BaseModel):
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    sections: list[DocumentSection] = Field(default_factory=list)

    def full_text(self) -> str:
        return "\n\n".join(s.to_text() for s in self.sections)


# ---------------------------------------------------------------------------
# Chunking output
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    """A contiguous slice of document content used for comparison."""
    chunk_id: str                   # deterministic, e.g. "sec0_para2"
    section_index: int
    section_heading: str = ""
    text: str
    source: str = ""                # "paragraph" | "table" | "list" | "heading"
    index_in_doc: int = -1          # global order in the original document


# ---------------------------------------------------------------------------
# Matching / comparison results
# ---------------------------------------------------------------------------

class ChangeType(str, Enum):
    EXACT              = "exact_match"
    NEAR_EXACT         = "near_exact"
    MODIFIED           = "modified"
    SEMANTIC           = "semantic_change"
    SEMANTIC_DIFFERENT = "semantically_different"   # NEW: semantically different content
    ADDED              = "added"
    REMOVED            = "removed"
    UNCHANGED          = "unchanged"


class SemanticChangeType(str, Enum):
    PRESERVED    = "meaning_preserved"
    MODIFIED     = "meaning_modified"
    EXPANDED     = "meaning_expanded"
    REDUCED      = "meaning_reduced"
    CONTRADICTION = "contradiction"
    UNRELATED    = "unrelated"


class CriticalInfoChange(BaseModel):
    """Captures a specific date/number/critical-value change detected inside a Modified chunk."""
    info_type: str          # "date" | "number" | "percentage" | "currency" | "other"
    original: str           # value in Document A
    revised: str            # value in Document B
    context: str = ""       # surrounding words for readability


class SemanticAnalysis(BaseModel):
    change_type: SemanticChangeType
    confidence: float
    summary: str


class ChunkMatch(BaseModel):
    chunk_a: Chunk
    chunk_b: Chunk | None           # None → chunk was removed / added
    change_type: ChangeType
    similarity_score: float = 0.0
    fuzzy_score: float = 0.0
    semantic_score: float = 0.0
    semantic_analysis: SemanticAnalysis | None = None
    critical_info_changes: list[CriticalInfoChange] = Field(default_factory=list)  # NEW


# ---------------------------------------------------------------------------
# Final comparison report
# ---------------------------------------------------------------------------

class LevelStats(BaseModel):
    total_a: int = 0
    total_b: int = 0
    exact_matches: int = 0
    near_exact: int = 0
    modified: int = 0
    added: int = 0
    removed: int = 0
    semantic_changes: int = 0
    semantic_different: int = 0   # NEW: semantically different chunks


class ComparisonReport(BaseModel):
    doc_a_metadata: DocumentMetadata
    doc_b_metadata: DocumentMetadata

    overall_similarity: float       # 0.0 – 1.0
    document_level_summary: str

    section_stats: LevelStats = Field(default_factory=LevelStats)
    paragraph_stats: LevelStats = Field(default_factory=LevelStats)

    matches: list[ChunkMatch] = Field(default_factory=list)

    # Convenience views (populated by reporting layer)
    added_chunks: list[Chunk] = Field(default_factory=list)
    removed_chunks: list[Chunk] = Field(default_factory=list)
    modified_chunks: list[ChunkMatch] = Field(default_factory=list)
    semantic_chunks: list[ChunkMatch] = Field(default_factory=list)

    processing_time_seconds: float = 0.0