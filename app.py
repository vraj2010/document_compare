"""
app.py — Main Streamlit entry point.

Run with:
    streamlit run app.py

Pages (implemented as tabs):
  1. Upload    — drag-and-drop or browse for File A & File B
  2. Comparison — visual diff with statistics
  3. Reports    — download JSON / HTML
"""

from __future__ import annotations

import sys
import os

# Ensure package root is on the path when running via `streamlit run app.py`
sys.path.insert(0, os.path.dirname(__file__))

import logging
import time

import streamlit as st

from comparison import ComparisonEngine
from extraction import DocumentExtractor
from reporting import generate_html_report, generate_json_report
from ui import (
    render_download_buttons,
    render_match_card,
    render_similarity_gauge,
    render_stats_bar,
)
from models import ChangeType

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DocCompare — Intelligent Document Comparison",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  .main-header {
    background: linear-gradient(135deg, #0d1b6e 0%, #1a237e 50%, #283593 100%);
    color: white;
    padding: 32px 36px;
    border-radius: 16px;
    margin-bottom: 28px;
  }
  .main-header h1 {
    font-size: 36px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 6px;
  }
  .main-header p {
    opacity: 0.75;
    font-size: 15px;
    font-weight: 400;
  }

  .upload-zone {
    border: 2px dashed #90caf9;
    border-radius: 12px;
    padding: 24px;
    background: #e3f2fd;
    text-align: center;
    margin-bottom: 8px;
  }

  .section-title {
    font-size: 20px;
    font-weight: 700;
    color: #1a237e;
    margin: 24px 0 12px;
    padding-bottom: 6px;
    border-bottom: 2px solid #e3f2fd;
  }

  div[data-testid="stExpander"] {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    margin-bottom: 8px;
  }

  .stTabs [data-baseweb="tab-list"] {
    gap: 8px;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    font-weight: 600;
    font-size: 15px;
  }

  .process-step {
    background: white;
    border-left: 4px solid #1a237e;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
    font-size: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
  }

  .info-chip {
    display: inline-block;
    background: #e8eaf6;
    color: #1a237e;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    margin: 2px;
  }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="main-header">
  <h1>📋 DocCompare</h1>
  <p>Intelligent document comparison &mdash;
     Exact, Fuzzy &amp; Semantic analysis &mdash;
     PDF &bull; DOCX &bull; TXT</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_upload, tab_compare, tab_reports = st.tabs(["📤 Upload", "🔍 Comparison", "📊 Reports"])

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "report" not in st.session_state:
    st.session_state.report = None
if "json_report" not in st.session_state:
    st.session_state.json_report = None
if "html_report" not in st.session_state:
    st.session_state.html_report = None


# ===========================================================================
# TAB 1 — Upload
# ===========================================================================

with tab_upload:
    st.markdown('<div class="section-title">Upload Documents</div>', unsafe_allow_html=True)

    col_a, col_spacer, col_b = st.columns([5, 1, 5])

    with col_a:
        st.markdown("#### 📄 Document A  *(original)*")
        file_a = st.file_uploader(
            "Drop File A here",
            type=["pdf", "docx", "txt"],
            key="file_a",
            label_visibility="collapsed",
        )
        if file_a:
            st.success(f"✔ **{file_a.name}**  ({file_a.size:,} bytes)")

    with col_b:
        st.markdown("#### 📄 Document B  *(revised)*")
        file_b = st.file_uploader(
            "Drop File B here",
            type=["pdf", "docx", "txt"],
            key="file_b",
            label_visibility="collapsed",
        )
        if file_b:
            st.success(f"✔ **{file_b.name}**  ({file_b.size:,} bytes)")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Settings expander
    # -----------------------------------------------------------------------
    with st.expander("⚙️ Advanced Settings", expanded=False):
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            fuzzy_threshold = st.slider(
                "Fuzzy exact threshold",
                0.7, 1.0, 0.95, 0.01,
                help="Fuzzy ratio above this is treated as near-exact match.",
            )
        with sc2:
            sem_threshold = st.slider(
                "Semantic similarity threshold",
                0.5, 1.0, 0.82, 0.01,
                help="Cosine similarity above this is treated as same meaning.",
            )
        with sc3:
            max_chunk_tokens = st.slider(
                "Max chunk tokens",
                20, 300, 80, 10,
                help="Lower = finer paragraph-level chunks → more semantic detail.",
            )

    st.markdown("")

    # -----------------------------------------------------------------------
    # Compare button
    # -----------------------------------------------------------------------
    compare_btn = st.button(
        "🚀 Compare Documents",
        type="primary",
        disabled=(file_a is None or file_b is None),
        use_container_width=True,
    )

    if compare_btn and file_a and file_b:
        # Apply custom config overrides
        from config import CONFIG
        import config as _cfg_module
        # Re-create config with overrides (frozen dataclass → rebuild)
        from config import MatchingConfig, ChunkingConfig
        new_matching = MatchingConfig(
            fuzzy_exact_threshold=fuzzy_threshold,
            semantic_similar_threshold=sem_threshold,
        )
        new_chunking = ChunkingConfig(max_chunk_tokens=max_chunk_tokens)
        from config import AppConfig, EmbeddingConfig, ReportingConfig, ExtractionConfig
        overridden = AppConfig(
            matching=new_matching,
            chunking=new_chunking,
        )

        progress = st.progress(0, text="Initialising…")

        try:
            progress.progress(10, text="📖 Extracting Document A…")
            extractor = DocumentExtractor()
            bytes_a = file_a.read()
            bytes_b = file_b.read()
            doc_a = extractor.extract(bytes_a, file_a.name)

            progress.progress(30, text="📖 Extracting Document B…")
            doc_b = extractor.extract(bytes_b, file_b.name)

            progress.progress(50, text="🔍 Normalising & chunking…")
            from normalization import DocumentNormalizer
            from chunking import HybridChunker

            normalizer = DocumentNormalizer()
            chunker = HybridChunker()
            norm_a = normalizer.normalize(doc_a)
            norm_b = normalizer.normalize(doc_b)
            chunks_a = chunker.chunk(norm_a)
            chunks_b = chunker.chunk(norm_b)

            progress.progress(65, text=f"⚡ Matching {len(chunks_a) + len(chunks_b)} chunks…")
            from matching import MatchingEngine
            engine = MatchingEngine()
            matches = engine.match(chunks_a, chunks_b)

            progress.progress(85, text="📊 Building report…")
            from comparison import ComparisonEngine, _build_level_stats, _overall_similarity, _document_summary
            stats = _build_level_stats(matches)
            similarity = _overall_similarity(matches)
            summary = _document_summary(similarity, stats)

            from models import ComparisonReport, Chunk, ChangeType
            added = [m.chunk_b for m in matches if m.change_type == ChangeType.ADDED and m.chunk_b]
            removed = [m.chunk_a for m in matches if m.change_type == ChangeType.REMOVED]
            modified = [m for m in matches if m.change_type in (ChangeType.MODIFIED, ChangeType.NEAR_EXACT)]
            semantic = [m for m in matches if m.change_type == ChangeType.SEMANTIC]

            report = ComparisonReport(
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
            )
            st.session_state.report = report
            st.session_state.json_report = generate_json_report(report)
            st.session_state.html_report = generate_html_report(report)

            progress.progress(100, text="✅ Done!")
            time.sleep(0.4)
            progress.empty()

            st.success(
                f"✅ Comparison complete!  "
                f"**{report.overall_similarity:.1%}** similarity &nbsp;·&nbsp; "
                f"{len(matches)} chunks analysed &nbsp;·&nbsp; "
                f"Switch to the **Comparison** tab to view results."
            )

        except Exception as exc:
            progress.empty()
            st.error(f"❌ Comparison failed: {exc}")
            logger.exception("Comparison error")


# ===========================================================================
# TAB 2 — Comparison
# ===========================================================================

with tab_compare:
    report = st.session_state.report

    if report is None:
        st.info("👆 Upload two documents and click **Compare Documents** to see results here.")
    else:
        # ---- Summary row ------------------------------------------------
        c_score, c_meta = st.columns([2, 5])
        with c_score:
            render_similarity_gauge(report.overall_similarity)
        with c_meta:
            st.markdown(f"""
            <div style="padding:16px 0;">
              <p style="font-size:15px; line-height:1.7; color:#424242;">{report.document_level_summary}</p>
              <div style="margin-top:12px;">
                <span class="info-chip">📄 {report.doc_a_metadata.filename}</span>
                <span class="info-chip">📄 {report.doc_b_metadata.filename}</span>
                <span class="info-chip">⏱ {report.processing_time_seconds}s</span>
                <span class="info-chip">📦 {len(report.matches)} chunks</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ---- Stats bar --------------------------------------------------
        render_stats_bar(report)
        st.markdown("")

        # ---- Filter controls -------------------------------------------
        st.markdown('<div class="section-title">Detailed Differences</div>', unsafe_allow_html=True)

        filter_col1, filter_col2 = st.columns([3, 2])
        with filter_col1:
            change_filter = st.multiselect(
                "Filter by change type",
                options=[
                    "All Differences",
                    "Semantic Change",
                    "Semantically Different",
                    "Modified",
                    "Added",
                    "Removed",
                    "Near Exact",
                    "Exact Match",
                ],
                default=["All Differences"],
                help=(
                    "**All Differences** — shows every change except exact matches.\n\n"
                    "**Semantic Change** — same meaning, different wording. Shows "
                    "the matched pair (Doc A ↔ Doc B) plus any content only present "
                    "in one document within those sections.\n\n"
                    "**Semantically Different** — chunks that are loosely related but "
                    "carry meaningfully different content. Includes surrounding "
                    "added/removed chunks in the same section for full context.\n\n"
                    "**Modified** — same chunk, partial text edits (fuzzy match).\n\n"
                    "**Added** — content only in Document B.\n\n"
                    "**Removed** — content only in Document A.\n\n"
                    "**Near Exact** — near-identical text (tiny typo / spacing).\n\n"
                    "**Exact Match** — byte-for-byte identical chunks."
                ),
            )
        with filter_col2:
            search_term = st.text_input("🔍 Search in content", placeholder="Enter keyword…")

        # ---- Filtering logic -------------------------------------------
        FILTER_MAP = {
            "Added":                  ChangeType.ADDED,
            "Removed":                ChangeType.REMOVED,
            "Modified":               ChangeType.MODIFIED,
            "Near Exact":             ChangeType.NEAR_EXACT,
            "Semantic Change":        ChangeType.SEMANTIC,
            "Semantically Different": ChangeType.SEMANTIC_DIFFERENT,
            "Exact Match":            ChangeType.EXACT,
        }

        # All non-exact change types
        _DIFFERENCE_TYPES = {
            ChangeType.ADDED, ChangeType.REMOVED, ChangeType.MODIFIED,
            ChangeType.SEMANTIC, ChangeType.SEMANTIC_DIFFERENT, ChangeType.NEAR_EXACT,
        }

        # Semantic filter types — when these are selected we want full context
        _SEMANTIC_FILTER_TYPES = {"Semantic Change", "Semantically Different"}

        show_all_diff   = "All Differences" in change_filter or not change_filter
        is_semantic_only = (
            not show_all_diff
            and bool(change_filter)
            and all(f in _SEMANTIC_FILTER_TYPES for f in change_filter)
        )

        allowed_types = (
            {FILTER_MAP[f] for f in change_filter if f in FILTER_MAP}
            if not show_all_diff else None
        )

        # When a semantic filter is active: collect the section indices of all
        # semantic matches so we can also surface their Add/Remove neighbours.
        _semantic_section_indices: set[int] = set()
        if is_semantic_only:
            sem_direct_types = {FILTER_MAP[f] for f in change_filter if f in FILTER_MAP}
            for m in report.matches:
                if m.change_type in sem_direct_types:
                    if m.chunk_a:
                        _semantic_section_indices.add(m.chunk_a.section_index)
                    if m.chunk_b:
                        _semantic_section_indices.add(m.chunk_b.section_index)

        def _match_passes_filter(m) -> bool:
            """Return True if this match should be shown given the current filter."""
            if show_all_diff:
                return m.change_type in _DIFFERENCE_TYPES

            # Direct type match
            if allowed_types and m.change_type in allowed_types:
                return True

            # Semantic context expansion: when filtering by semantic types,
            # also include Added/Removed chunks that belong to the same
            # sections as the matched semantic pairs — so the user sees
            # the full picture of what changed semantically in that section.
            if is_semantic_only and m.change_type in (ChangeType.ADDED, ChangeType.REMOVED):
                sec = (
                    m.chunk_a.section_index if m.chunk_a
                    else (m.chunk_b.section_index if m.chunk_b else -1)
                )
                if sec in _semantic_section_indices:
                    return True

            return False

        def _match_passes_search(m) -> bool:
            if not search_term:
                return True
            term = search_term.lower()
            text_a = (m.chunk_a.text or "") if m.chunk_a else ""
            text_b = (m.chunk_b.text or "") if m.chunk_b else ""
            return term in text_a.lower() or term in text_b.lower()

        filtered = [
            m for m in report.matches
            if _match_passes_filter(m) and _match_passes_search(m)
        ]

        # Sort: semantic types first, then by section index for readability
        PRIORITY = {
            ChangeType.SEMANTIC_DIFFERENT: 0,
            ChangeType.SEMANTIC:           1,
            ChangeType.REMOVED:            2,
            ChangeType.ADDED:              3,
            ChangeType.MODIFIED:           4,
            ChangeType.NEAR_EXACT:         5,
            ChangeType.EXACT:              6,
        }

        if is_semantic_only:
            # Group by section for semantic views so pairs stay together
            filtered.sort(key=lambda m: (
                m.chunk_a.section_index if m.chunk_a else
                (m.chunk_b.section_index if m.chunk_b else 999),
                PRIORITY.get(m.change_type, 99),
            ))
        else:
            filtered.sort(key=lambda m: PRIORITY.get(m.change_type, 99))

        # ---- Context banner for semantic filter -------------------------
        if is_semantic_only and filtered:
            sem_matched = [m for m in filtered if m.change_type in (
                ChangeType.SEMANTIC, ChangeType.SEMANTIC_DIFFERENT)]
            sem_context = [m for m in filtered if m.change_type in (
                ChangeType.ADDED, ChangeType.REMOVED)]
            st.info(
                f"🔍 **Semantic view** — showing **{len(sem_matched)} semantic pair(s)** "
                f"and **{len(sem_context)} surrounding add/remove chunk(s)** "
                f"from the same sections, so you can see the full picture of what changed."
            )

        st.caption(f"Showing {len(filtered)} of {len(report.matches)} chunks")

        PAGE_SIZE = 50
        if len(filtered) > PAGE_SIZE:
            page = st.number_input("Page", 1, (len(filtered) - 1) // PAGE_SIZE + 1, 1)
            page_matches = filtered[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]
        else:
            page_matches = filtered

        if is_semantic_only:
            # ---- Grouped section view for semantic filter ---------------
            # Walk through matches in section order and print a section
            # header whenever the section changes.
            last_section_idx = None
            for i, match in enumerate(page_matches):
                sec_idx = (
                    match.chunk_a.section_index if match.chunk_a
                    else (match.chunk_b.section_index if match.chunk_b else None)
                )
                sec_heading = (
                    match.chunk_a.section_heading if match.chunk_a and match.chunk_a.section_heading
                    else (match.chunk_b.section_heading if match.chunk_b and match.chunk_b.section_heading else "")
                )
                if sec_idx != last_section_idx:
                    last_section_idx = sec_idx
                    label = f"§ {sec_heading}" if sec_heading else f"Section {sec_idx}"
                    st.markdown(
                        f'<div style="margin:18px 0 6px; padding:8px 14px; '
                        f'background:#e8eaf6; border-left:4px solid #3949ab; '
                        f'border-radius:0 8px 8px 0; font-size:14px; font-weight:700; '
                        f'color:#1a237e;">{label}</div>',
                        unsafe_allow_html=True,
                    )
                render_match_card(match, i)
        else:
            for i, match in enumerate(page_matches):
                render_match_card(match, i)


# ===========================================================================
# TAB 3 — Reports
# ===========================================================================

with tab_reports:
    report = st.session_state.report

    if report is None:
        st.info("👆 Run a comparison first to generate downloadable reports.")
    else:
        st.markdown('<div class="section-title">Download Reports</div>', unsafe_allow_html=True)
        st.markdown(
            "Both reports contain the full comparison data. "
            "**JSON** is machine-readable; **HTML** is a self-contained visual report."
        )
        st.markdown("")
        render_download_buttons(
            st.session_state.json_report,
            st.session_state.html_report,
        )

        st.markdown("---")
        st.markdown('<div class="section-title">JSON Preview</div>', unsafe_allow_html=True)
        with st.expander("Expand JSON", expanded=False):
            st.code(st.session_state.json_report[:8000] + ("\n… (truncated)" if len(st.session_state.json_report) > 8000 else ""), language="json")