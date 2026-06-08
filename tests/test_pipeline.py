"""
tests/test_pipeline.py

Unit tests covering: normalization, chunking, hash/fuzzy matching,
semantic scoring, and report generation.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from models import (
    Chunk,
    Document,
    DocumentMetadata,
    DocumentSection,
    Table,
    ChangeType,
)
from normalization import DocumentNormalizer, normalize_text
from chunking import HybridChunker
from matching import MatchingEngine, _sha256
from reporting import generate_json_report, generate_html_report
from models import ComparisonReport, LevelStats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_doc(sections_data: list[dict]) -> Document:
    sections = []
    for d in sections_data:
        sections.append(DocumentSection(
            heading=d.get("heading", ""),
            heading_level=d.get("level", 1),
            paragraphs=d.get("paragraphs", []),
            lists=d.get("lists", []),
        ))
    return Document(
        metadata=DocumentMetadata(filename="test.txt", file_type="txt"),
        sections=sections,
    )


@pytest.fixture
def simple_doc_a():
    return _make_doc([
        {
            "heading": "Introduction",
            "paragraphs": [
                "This document describes the quarterly revenue results.",
                "Revenue increased by 20% compared to the previous quarter.",
            ],
        },
        {
            "heading": "Conclusion",
            "paragraphs": ["Overall performance was strong and exceeded targets."],
        },
    ])


@pytest.fixture
def simple_doc_b():
    return _make_doc([
        {
            "heading": "Introduction",
            "paragraphs": [
                "This document describes the quarterly revenue results.",
                "Revenue grew significantly this quarter.",   # semantic change
            ],
        },
        {
            "heading": "Summary",                            # renamed section
            "paragraphs": ["Overall performance exceeded targets by a wide margin."],  # modified
            "lists": [["Strong growth", "New markets entered"]],   # added
        },
    ])


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_unicode_nfc(self):
        # Combining accent → precomposed
        text = "cafe\u0301"   # 'e' + combining acute
        result = normalize_text(text)
        assert result == "café"

    def test_whitespace_collapse(self):
        text = "hello    world\n\n\n\nbye"
        result = normalize_text(text)
        assert "    " not in result
        assert result.count("\n") <= 2

    def test_smart_quotes(self):
        text = "\u201cHello\u201d and \u2018world\u2019"
        result = normalize_text(text)
        assert '"' in result or "'" in result
        assert "\u201c" not in result

    def test_document_normalizer(self, simple_doc_a):
        norm = DocumentNormalizer()
        result = norm.normalize(simple_doc_a)
        assert len(result.sections) == len(simple_doc_a.sections)
        for sec in result.sections:
            for para in sec.paragraphs:
                assert para == para.strip()


# ---------------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------------

class TestChunking:
    def test_produces_chunks(self, simple_doc_a):
        chunker = HybridChunker()
        norm = DocumentNormalizer().normalize(simple_doc_a)
        chunks = chunker.chunk(norm)
        assert len(chunks) > 0

    def test_chunk_ids_unique(self, simple_doc_a):
        chunker = HybridChunker()
        norm = DocumentNormalizer().normalize(simple_doc_a)
        chunks = chunker.chunk(norm)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_text_not_empty(self, simple_doc_a):
        chunker = HybridChunker()
        norm = DocumentNormalizer().normalize(simple_doc_a)
        chunks = chunker.chunk(norm)
        for c in chunks:
            assert c.text.strip(), f"Empty chunk: {c.chunk_id}"

    def test_section_heading_preserved(self, simple_doc_a):
        chunker = HybridChunker()
        norm = DocumentNormalizer().normalize(simple_doc_a)
        chunks = chunker.chunk(norm)
        heading_chunks = [c for c in chunks if c.source == "heading"]
        assert any("Introduction" in c.text for c in heading_chunks)

    def test_list_chunks(self):
        doc = _make_doc([{"heading": "Items", "lists": [["Alpha", "Beta", "Gamma"]]}])
        chunker = HybridChunker()
        norm = DocumentNormalizer().normalize(doc)
        chunks = chunker.chunk(norm)
        list_chunks = [c for c in chunks if c.source == "list"]
        assert len(list_chunks) == 1
        assert "Alpha" in list_chunks[0].text


# ---------------------------------------------------------------------------
# Hash matching tests
# ---------------------------------------------------------------------------

class TestHashMatching:
    def _make_chunks(self, texts: list[str], prefix="a") -> list[Chunk]:
        return [
            Chunk(chunk_id=f"{prefix}_{i}", section_index=0, text=t, source="paragraph")
            for i, t in enumerate(texts)
        ]

    def test_exact_match(self):
        engine = MatchingEngine()
        ca = self._make_chunks(["Hello world", "Foo bar"], "a")
        cb = self._make_chunks(["Hello world", "Baz qux"], "b")
        matches = engine.match(ca, cb)
        exact = [m for m in matches if m.change_type == ChangeType.EXACT]
        assert len(exact) == 1
        assert exact[0].chunk_a.text == "Hello world"

    def test_added_chunk_detected(self):
        engine = MatchingEngine()
        ca = self._make_chunks(["Hello world"], "a")
        cb = self._make_chunks(["Hello world", "Brand new content here"], "b")
        matches = engine.match(ca, cb)
        added = [m for m in matches if m.change_type == ChangeType.ADDED]
        assert len(added) == 1

    def test_removed_chunk_detected(self):
        engine = MatchingEngine()
        ca = self._make_chunks(["Hello world", "Something removed"], "a")
        cb = self._make_chunks(["Hello world"], "b")
        matches = engine.match(ca, cb)
        removed = [m for m in matches if m.change_type == ChangeType.REMOVED]
        assert len(removed) == 1

    def test_sha256_consistency(self):
        text = "The quick brown fox"
        assert _sha256(text) == _sha256(text)
        assert _sha256(text) != _sha256(text + "!")


# ---------------------------------------------------------------------------
# Fuzzy matching tests
# ---------------------------------------------------------------------------

class TestFuzzyMatching:
    def _make_chunks(self, texts: list[str], prefix="a") -> list[Chunk]:
        return [
            Chunk(chunk_id=f"{prefix}_{i}", section_index=0, text=t, source="paragraph")
            for i, t in enumerate(texts)
        ]

    def test_near_exact_typo(self):
        engine = MatchingEngine()
        ca = self._make_chunks(["The quick brown fox jumps over the lazy dog"], "a")
        cb = self._make_chunks(["The quick brwon fox jumps over the lazy dog"], "b")  # typo
        matches = engine.match(ca, cb)
        near = [m for m in matches if m.change_type in (ChangeType.NEAR_EXACT, ChangeType.MODIFIED)]
        assert len(near) >= 1

    def test_modified_content(self):
        engine = MatchingEngine()
        ca = self._make_chunks(["Revenue increased by 20% this year"], "a")
        cb = self._make_chunks(["Revenue increased by 35% this year"], "b")
        matches = engine.match(ca, cb)
        mods = [m for m in matches if m.change_type in (ChangeType.MODIFIED, ChangeType.NEAR_EXACT)]
        assert len(mods) >= 1


# ---------------------------------------------------------------------------
# Semantic matching (mocked to avoid model download in CI)
# ---------------------------------------------------------------------------

class TestSemanticAnalysis:
    def test_semantic_change_type_high_score(self):
        from matching import _semantic_change_type
        from models import SemanticChangeType
        sc, summary = _semantic_change_type(0.92, 0.90)
        assert sc == SemanticChangeType.PRESERVED

    def test_semantic_change_type_low_score(self):
        from matching import _semantic_change_type
        from models import SemanticChangeType
        sc, summary = _semantic_change_type(0.10, 0.10)
        assert sc == SemanticChangeType.UNRELATED

    def test_semantic_change_type_medium_score(self):
        from matching import _semantic_change_type
        from models import SemanticChangeType
        sc, summary = _semantic_change_type(0.65, 0.30)
        assert sc in (SemanticChangeType.EXPANDED, SemanticChangeType.MODIFIED)


# ---------------------------------------------------------------------------
# Reporting tests
# ---------------------------------------------------------------------------

class TestReporting:
    def _make_report(self) -> ComparisonReport:
        meta_a = DocumentMetadata(filename="a.txt", file_type="txt", word_count=100)
        meta_b = DocumentMetadata(filename="b.txt", file_type="txt", word_count=110)
        return ComparisonReport(
            doc_a_metadata=meta_a,
            doc_b_metadata=meta_b,
            overall_similarity=0.85,
            document_level_summary="High similarity.",
            paragraph_stats=LevelStats(exact_matches=5, modified=2, added=1, removed=1),
            matches=[],
        )

    def test_json_report_is_valid_json(self):
        import json
        report = self._make_report()
        json_str = generate_json_report(report)
        data = json.loads(json_str)
        assert data["overall_similarity"] == 0.85

    def test_html_report_contains_similarity(self):
        report = self._make_report()
        html = generate_html_report(report)
        assert "85%" in html
        assert "<!DOCTYPE html>" in html

    def test_html_report_has_table(self):
        report = self._make_report()
        html = generate_html_report(report)
        assert "<table" in html
        assert "<thead" in html


# ---------------------------------------------------------------------------
# Integration smoke test
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_txt_comparison(self, simple_doc_a, simple_doc_b):
        from normalization import DocumentNormalizer
        from chunking import HybridChunker

        norm = DocumentNormalizer()
        chunker = HybridChunker()
        engine = MatchingEngine()

        na = norm.normalize(simple_doc_a)
        nb = norm.normalize(simple_doc_b)
        ca = chunker.chunk(na)
        cb = chunker.chunk(nb)
        matches = engine.match(ca, cb)

        assert len(matches) > 0
        types = {m.change_type for m in matches}
        # Expect at least one exact match (identical first paragraph)
        assert ChangeType.EXACT in types or ChangeType.NEAR_EXACT in types
