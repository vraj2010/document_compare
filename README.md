# DocCompare — Production-Ready Document Comparison System

> Intelligent PDF / DOCX / TXT comparison using Exact, Fuzzy, and Semantic matching.  
> Built with Python · Streamlit · Docling · RapidFuzz · SentenceTransformers · Pydantic

---

## Quick Start

```bash
# 1. Clone / unzip project
cd document_compare

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies (CPU-only PyTorch for broad compatibility)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. Launch
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Project Structure

```
document_compare/
│
├── app.py                   # Streamlit entry point (3 tabs: Upload, Comparison, Reports)
├── config.py                # All thresholds & tuning knobs (single source of truth)
├── models.py                # Pydantic models for the unified document representation
├── requirements.txt
│
├── extraction/
│   └── extractor.py         # Docling-first extractor with fallbacks for PDF/DOCX/TXT
│
├── normalization/
│   └── __init__.py          # Unicode, whitespace, punctuation, heading, table normalisation
│
├── chunking/
│   └── __init__.py          # Hybrid section+paragraph chunker
│
├── matching/
│   └── __init__.py          # 3-stage cascade: Hash → Fuzzy → Semantic
│
├── comparison/
│   └── __init__.py          # Pipeline orchestrator → ComparisonReport
│
├── reporting/
│   └── __init__.py          # JSON + HTML report generators
│
├── ui/
│   ├── __init__.py
│   └── components.py        # Reusable Streamlit widgets
│
└── tests/
    └── test_pipeline.py     # Pytest unit + integration tests
```

---

## Architecture Deep Dive

### 1 · Extraction Layer

**Primary:** Docling's `DocumentConverter` handles PDF and DOCX with structural
awareness — it preserves headings, tables, lists, and reading order that
raw-text extractors lose.

**Fallbacks (if Docling fails):**
- DOCX → `python-docx` (style-based heading detection)  
- PDF  → `pdfplumber` → `pypdf`  
- TXT  → heuristic line parser (detects Markdown headings and bullet lists)

### 2 · Unified Document Model (Pydantic)

```
Document
├── DocumentMetadata  (filename, type, word_count, char_count)
└── List[DocumentSection]
    ├── heading / heading_level
    ├── List[str]       paragraphs
    ├── List[List[str]] lists
    └── List[Table]     tables
```

Every downstream stage consumes this model — it is the contract between
extraction and comparison.

### 3 · Normalization

Five passes applied before any comparison to reduce noise:

| Pass | What it does |
|------|-------------|
| Unicode NFC | Collapses combining characters to precomposed forms |
| Whitespace | Collapses runs of spaces/tabs; limits blank lines |
| Punctuation | Smart quotes, em-dashes, NBSP → ASCII equivalents |
| Heading | Strips trailing punctuation, normalises case |
| Table | Lowercases headers, trims all cells |

### 4 · Chunking Strategy — Hybrid Section + Paragraph

**Why not pure section chunking?**  
Sections can be thousands of words long, making embeddings noisy and
fine-grained diffs impossible.

**Why not pure semantic chunking?**  
Requires a full embedding pass just to split — expensive and circular.

**Hybrid approach:**
1. Split at section boundaries (preserves document structure)
2. Sub-split paragraphs if accumulated tokens exceed `max_chunk_tokens` (default 300)
3. Tables and lists each become their own chunk

Result: ~400-600 chunks for a 50-page document, chunking time < 50 ms.

### 5 · Three-Stage Matching Cascade

```
For each chunk in A:
  1. SHA-256 hash lookup in B  →  Exact Match  (0 embeddings)
     ↓ miss
  2. RapidFuzz token_sort_ratio  →  Near-Exact / Modified  (0 embeddings)
     ↓ miss
  3. SentenceTransformer cosine  →  Semantic / Unrelated  (batch embedding)
```

**Why this order?**  
Stages 1 and 2 are essentially free (microseconds). In a typical contract
or report document, 60-80% of chunks are unchanged boilerplate — they never
reach Stage 3. This means the expensive embedding model handles only a
small fraction of chunks.

**Batch embedding:**  
All Stage-3 candidates are embedded in a single `model.encode()` call,
which saturates CPU/GPU efficiently via batching.

**Nearest-neighbour matching:**  
Similarity matrix computed as `embeds_a @ embeds_b.T` (vectorised dot
product on pre-normalised embeddings = cosine similarity). No `faiss`
dependency needed for documents under ~1000 chunks.

### 6 · Semantic Change Detection

| Cosine Score | Fuzzy Score | Interpretation |
|---|---|---|
| ≥ 0.82 | ≥ 0.85 | Meaning Preserved (minor rewording) |
| ≥ 0.82 | < 0.85 | Meaning Preserved (different phrasing) |
| 0.50–0.82 | < 0.50 | Meaning Expanded |
| 0.50–0.82 | ≥ 0.50 | Meaning Modified |
| 0.20–0.50 | — | Meaning Reduced |
| < 0.20 | — | Contradiction / Unrelated |

Example:
```
Doc A: "Revenue increased by 20%"
Doc B: "Revenue grew significantly"
→ change_type: meaning_preserved, confidence: 0.91
```

### 7 · Similarity Scoring

Overall similarity is the weighted average of per-match similarity scores.
Exact matches = 1.0, Added/Removed = 0.0.

### 8 · Why `all-MiniLM-L6-v2`?

| Model | Dims | Speed | Quality |
|-------|------|-------|---------|
| all-MiniLM-L6-v2 | 384 | ⚡⚡⚡ | ★★★★ |
| all-mpnet-base-v2 | 768 | ⚡⚡ | ★★★★★ |
| paraphrase-MiniLM-L3-v2 | 384 | ⚡⚡⚡⚡ | ★★★ |

`all-MiniLM-L6-v2` hits the sweet spot: 22M parameters, 80ms/batch on CPU,
STS benchmark score competitive with much larger models.

---

## Configuration

All thresholds are in `config.py` and surfaced in the UI's **Advanced
Settings** expander:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `fuzzy_exact_threshold` | 0.95 | Above → near-exact match |
| `semantic_similar_threshold` | 0.82 | Above → same meaning |
| `max_chunk_tokens` | 300 | Chunk size (lower = finer diffs) |

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

Expected: all unit tests pass without needing a GPU.  
The semantic embedding tests use pure logic (no model download required).

---

## Expected Performance

| Document size | Chunks | Hash stage | Fuzzy stage | Semantic stage | Total |
|---|---|---|---|---|---|
| 2-page letter | ~30 | < 1ms | < 5ms | < 200ms | < 300ms |
| 20-page report | ~200 | < 5ms | < 50ms | < 1s | ~1.5s |
| 100-page contract | ~800 | < 20ms | < 500ms | ~3s | ~5s |

*Timings on a modern CPU. GPU reduces semantic stage by ~5×.*

---

## Future Scalability

- **GPU acceleration:** Change `device: "cpu"` → `"cuda"` in `config.py`.
- **Larger documents:** Swap nearest-neighbour with `faiss.IndexFlatIP` for
  sub-linear search above ~2000 chunks.
- **API mode:** Wrap `ComparisonEngine.compare_bytes()` in a FastAPI endpoint.
- **Caching:** Already uses `lru_cache`; for persistent caching across restarts
  add `diskcache` without other architectural changes.
- **Multilingual:** Swap to `paraphrase-multilingual-MiniLM-L12-v2` in config.
