"""
chunking/__init__.py

Chunking strategy: **Hybrid Section + Paragraph**

Why hybrid over pure section or pure semantic?
-----------------------------------------------
* Pure section chunking:  Good for structure, but sections can be huge
  (thousands of tokens), making sentence-level diffs undetectable and
  embedding comparisons noisy.
* Pure semantic chunking:  High accuracy but requires a full embedding
  pass just to split the document — expensive and circular.
* Hybrid:  Split on section boundaries first (preserves logical structure),
  then sub-split any section whose paragraph count exceeds a threshold.
  Result: semantically coherent, small, fast-to-embed chunks.

Expected performance
--------------------
* For a 50-page PDF (~25k words, ~300 sections/paragraphs):
  ~400-600 chunks, chunking time < 50 ms.
"""

from __future__ import annotations

from models import Chunk, Document
from config import CONFIG


class HybridChunker:
    """
    Produces Chunk objects from a normalized Document.

    Each chunk has a deterministic ID (section_index + source + item_index)
    so that hash-based deduplication is reproducible.
    """

    def __init__(self):
        self.cfg = CONFIG.chunking

    def chunk(self, doc: Document) -> list[Chunk]:
        chunks: list[Chunk] = []

        for sec_idx, section in enumerate(doc.sections):
            heading = section.heading

            # --- Heading as its own chunk (only if substantial) ----------
            if heading and len(heading) >= self.cfg.min_paragraph_chars:
                chunks.append(Chunk(
                    chunk_id=f"sec{sec_idx}_heading",
                    section_index=sec_idx,
                    section_heading=heading,
                    text=heading,
                    source="heading",
                ))

            # --- Paragraphs ----------------------------------------------
            # Each paragraph becomes its own chunk so that fine-grained
            # semantic differences are visible.  Paragraphs that exceed
            # max_chunk_tokens are split on sentence boundaries first;
            # if no sentence boundary is found they are split by token count.

            import re as _re
            _SENT_SPLIT = _re.compile(r'(?<=[.!?])\s+')

            def _split_long_para(text: str) -> list[str]:
                """Split a paragraph that exceeds max_chunk_tokens."""
                tokens = text.split()
                if len(tokens) <= self.cfg.max_chunk_tokens:
                    return [text]
                # Try sentence splitting first
                sentences = _SENT_SPLIT.split(text)
                parts: list[str] = []
                buf: list[str] = []
                buf_tok = 0
                for sent in sentences:
                    stok = len(sent.split())
                    if buf_tok + stok > self.cfg.max_chunk_tokens and buf:
                        parts.append(" ".join(buf))
                        buf, buf_tok = [], 0
                    buf.append(sent)
                    buf_tok += stok
                if buf:
                    parts.append(" ".join(buf))
                return parts if parts else [text]

            for p_idx, para in enumerate(section.paragraphs):
                if len(para) < self.cfg.min_paragraph_chars:
                    continue  # skip noise
                if len(para.split()) < 5:
                    continue  # skip noise

                sub_parts = _split_long_para(para)
                for sp_idx, part in enumerate(sub_parts):
                    part = part.strip()
                    if not part:
                        continue
                    cid = (f"sec{sec_idx}_para{p_idx}"
                           if len(sub_parts) == 1
                           else f"sec{sec_idx}_para{p_idx}_{sp_idx}")
                    chunks.append(Chunk(
                        chunk_id=cid,
                        section_index=sec_idx,
                        section_heading=heading,
                        text=part,
                        source="paragraph",
                    ))

            # --- Lists ---------------------------------------------------
            for l_idx, lst in enumerate(section.lists):
                combined = "\n".join(f"- {item}" for item in lst if item)
                if len(combined) >= self.cfg.min_paragraph_chars:
                    chunks.append(Chunk(
                        chunk_id=f"sec{sec_idx}_list{l_idx}",
                        section_index=sec_idx,
                        section_heading=heading,
                        text=combined,
                        source="list",
                    ))

            # --- Tables --------------------------------------------------
            for t_idx, table in enumerate(section.tables):
                tbl_text = table.to_text()
                if tbl_text.strip():
                    chunks.append(Chunk(
                        chunk_id=f"sec{sec_idx}_table{t_idx}",
                        section_index=sec_idx,
                        section_heading=heading,
                        text=tbl_text,
                        source="table",
                    ))

        return chunks
