"""
app.py — Main Streamlit entry point.
# Cache invalidated again

Run with:
    streamlit run app.py

Pages (implemented as tabs):
  1. Upload    — drag-and-drop or browse for File A & File B
  2. Comparison — section-based side-by-side diff with 4 categories
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
from comparison.section_grouper import build_section_diffs
from ui import (
    render_download_buttons,
    render_section_diff_card,
    FILTER_OPTIONS,
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
  <p>Section-Based Document Comparison &mdash;
     Side-by-side diff with full section context &mdash;
     PDF &bull; DOCX &bull; TXT</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_upload, tab_compare = st.tabs(["📤 Upload", "⚠️ Differences"])

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "report" not in st.session_state:
    st.session_state.report = None
if "doc_a" not in st.session_state:
    st.session_state.doc_a = None
if "doc_b" not in st.session_state:
    st.session_state.doc_b = None
if "json_report" not in st.session_state:
    st.session_state.json_report = None
if "html_report" not in st.session_state:
    st.session_state.html_report = None
if "section_diffs" not in st.session_state:
    st.session_state.section_diffs = None
if "diff_filter" not in st.session_state:
    st.session_state.diff_filter = "All Differences"


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

            progress.progress(50, text="🔍 Chunking…")
            from chunking import HybridChunker

            chunker = HybridChunker()
            chunks_a = chunker.chunk(doc_a)
            chunks_b = chunker.chunk(doc_b)

            progress.progress(65, text=f"⚡ Matching {len(chunks_a) + len(chunks_b)} chunks…")
            from matching import MatchingEngine
            engine = MatchingEngine()
            matches = engine.match(chunks_a, chunks_b)

            progress.progress(85, text="📊 Building report…")
            from comparison import _build_level_stats, _overall_similarity, _document_summary

            stats = _build_level_stats(matches)
            similarity = _overall_similarity(matches)
            summary = _document_summary(similarity, stats)

            from models import ComparisonReport, Chunk
            added = [m.chunk_b for m in matches if m.change_type == ChangeType.ADDED and m.chunk_b]
            removed = [m.chunk_a for m in matches if m.change_type == ChangeType.REMOVED]
            modified = [m for m in matches if m.change_type in (ChangeType.MODIFIED, ChangeType.NEAR_EXACT)]
            semantic = [m for m in matches if m.change_type in (ChangeType.SEMANTIC, ChangeType.SEMANTIC_DIFFERENT)]

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

            # Build section-level diffs
            section_diffs = build_section_diffs(doc_a, doc_b, matches)

            st.session_state.report = report
            st.session_state.doc_a = doc_a
            st.session_state.doc_b = doc_b
            st.session_state.section_diffs = section_diffs
            st.session_state.json_report = generate_json_report(report)
            st.session_state.html_report = generate_html_report(report)

            progress.progress(100, text="✅ Done!")
            time.sleep(0.4)
            progress.empty()

            diff_count = len(section_diffs)
            st.success(
                f"✅ Comparison complete!  "
                f"**{diff_count}** section difference{'s' if diff_count != 1 else ''} found "
                f"across {len(matches)} chunks analysed. &nbsp;·&nbsp; "
                f"Switch to the **Differences** tab to view results."
            )

        except Exception as exc:
            progress.empty()
            st.error(f"❌ Comparison failed: {exc}")
            logger.exception("Comparison error")


# ===========================================================================
# TAB 2 — Section-Based Differences
# ===========================================================================

with tab_compare:
    report = st.session_state.report
    section_diffs = st.session_state.section_diffs

    if report is None or section_diffs is None:
        st.info("👆 Upload two documents and click **Compare Documents** to see results here.")
    else:
        diff_count = len(section_diffs)

        # ---- Header banner ------------------------------------------------
        if diff_count > 0:
            banner_bg = "#c62828"
            banner_icon = "⚠️"
            banner_text = f"{diff_count} Section Difference{'s' if diff_count != 1 else ''} Detected"
        else:
            banner_bg = "#2e7d32"
            banner_icon = "✅"
            banner_text = "No Differences Detected"

        st.markdown(f"""
        <div style="
            background: {banner_bg};
            color: white;
            padding: 20px 28px;
            border-radius: 12px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 14px;
        ">
          <span style="font-size: 32px;">{banner_icon}</span>
          <div>
            <div style="font-size: 22px; font-weight: 800; letter-spacing: -0.3px;">{banner_text}</div>
            <div style="font-size: 13px; opacity: 0.85; margin-top: 4px;">
              Compared
              <strong>{report.doc_a_metadata.filename}</strong> ↔
              <strong>{report.doc_b_metadata.filename}</strong>
              &nbsp;·&nbsp; {len(report.matches)} chunks analysed
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ---- Filter dropdown -----------------------------------------------
        filter_options = ["All Differences", "Modified", "Removed", "Added"]
        selected_filter = st.selectbox(
            "Filter by category",
            options=filter_options,
            index=0,
            key="diff_filter_select",
            label_visibility="collapsed",
        )

        # ---- Apply filter ---------------------------------------------------
        if selected_filter == "All Differences":
            filtered = section_diffs
        else:
            filtered = [d for d in section_diffs if d.status == selected_filter.upper()]

        # ---- Count by category ----------------------------------------------
        modified_diffs = [d for d in section_diffs if d.status == "MODIFIED"]
        removed_diffs = [d for d in section_diffs if d.status == "REMOVED"]
        added_diffs = [d for d in section_diffs if d.status == "ADDED"]

        # ---- Summary stats chips -------------------------------------------
        st.markdown(f"""
        <div style="display:flex; gap:12px; margin-bottom:20px; flex-wrap:wrap;">
          <span style="background:#fff3e0; color:#e65100; border-radius:20px; padding:6px 16px;
                font-size:13px; font-weight:600;">
            ✏️ {len(modified_diffs)} Modified</span>
          <span style="background:#ffebee; color:#c62828; border-radius:20px; padding:6px 16px;
                font-size:13px; font-weight:600;">
            ➖ {len(removed_diffs)} Removed</span>
          <span style="background:#e8f5e9; color:#2e7d32; border-radius:20px; padding:6px 16px;
                font-size:13px; font-weight:600;">
            ➕ {len(added_diffs)} Added</span>
        </div>
        """, unsafe_allow_html=True)

        # ---- Render section diff cards --------------------------------------
        if diff_count == 0:
            st.markdown("""
            <div style="
                text-align: center;
                padding: 48px 24px;
                color: #757575;
                font-size: 16px;
            ">
              <div style="font-size: 48px; margin-bottom: 12px;">🎉</div>
              <div style="font-weight: 600; color: #2e7d32; font-size: 18px; margin-bottom: 8px;">
                Documents are semantically consistent
              </div>
              <div>No statements with materially different meaning were found between the two documents.</div>
            </div>
            """, unsafe_allow_html=True)
        elif len(filtered) == 0:
            st.markdown(f"""
            <div style="
                text-align: center;
                padding: 48px 24px;
                color: #757575;
                font-size: 16px;
            ">
              <div style="font-size: 36px; margin-bottom: 12px;">🔍</div>
              <div style="font-weight: 600; color: #616161; font-size: 16px; margin-bottom: 8px;">
                No &ldquo;{selected_filter}&rdquo; differences found
              </div>
              <div>Try selecting a different category or &ldquo;All Differences&rdquo;.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # --- SECTION 1: Modified ---
            mod_in_filter = [d for d in filtered if d.status == "MODIFIED"]
            if mod_in_filter:
                st.markdown("""
                <div style="font-size:18px; font-weight:800; color:#e65100; margin:20px 0 12px;
                     padding-bottom:6px; border-bottom:2px solid #fff3e0;">
                  SECTION 1 — MODIFIED SECTIONS
                </div>
                """, unsafe_allow_html=True)
                for i, d in enumerate(mod_in_filter, 1):
                    render_section_diff_card(
                        diff=d,
                        index=i,
                        file_a_name=report.doc_a_metadata.filename,
                        file_b_name=report.doc_b_metadata.filename,
                    )

            # --- SECTION 2: Removed ---
            rem_in_filter = [d for d in filtered if d.status == "REMOVED"]
            if rem_in_filter:
                st.markdown("""
                <div style="font-size:18px; font-weight:800; color:#c62828; margin:20px 0 12px;
                     padding-bottom:6px; border-bottom:2px solid #ffebee;">
                  SECTION 2 — REMOVED SECTIONS
                </div>
                """, unsafe_allow_html=True)
                for i, d in enumerate(rem_in_filter, 1):
                    render_section_diff_card(
                        diff=d,
                        index=i,
                        file_a_name=report.doc_a_metadata.filename,
                        file_b_name=report.doc_b_metadata.filename,
                    )

            # --- SECTION 3: Added ---
            add_in_filter = [d for d in filtered if d.status == "ADDED"]
            if add_in_filter:
                st.markdown("""
                <div style="font-size:18px; font-weight:800; color:#2e7d32; margin:20px 0 12px;
                     padding-bottom:6px; border-bottom:2px solid #e8f5e9;">
                  SECTION 3 — ADDED SECTIONS
                </div>
                """, unsafe_allow_html=True)
                for i, d in enumerate(add_in_filter, 1):
                    render_section_diff_card(
                        diff=d,
                        index=i,
                        file_a_name=report.doc_a_metadata.filename,
                        file_b_name=report.doc_b_metadata.filename,
                    )

        # --- SECTION 4: Missing Pages ---
        has_missing_a = bool(report.doc_a_metadata.missing_pages)
        has_missing_b = bool(report.doc_b_metadata.missing_pages)
        if has_missing_a or has_missing_b:
            st.markdown("""
            <div style="font-size:18px; font-weight:800; color:#1565c0; margin:20px 0 12px;
                 padding-bottom:6px; border-bottom:2px solid #e3f2fd;">
              SECTION 4 — MISSING PAGES
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**{report.doc_a_metadata.filename}**")
                if has_missing_a:
                    pages = ", ".join(map(str, sorted(report.doc_a_metadata.missing_pages)))
                    st.error(f"Missing Pages: {pages}")
                else:
                    st.success("No missing pages detected.")
            with c2:
                st.markdown(f"**{report.doc_b_metadata.filename}**")
                if has_missing_b:
                    pages = ", ".join(map(str, sorted(report.doc_b_metadata.missing_pages)))
                    st.error(f"Missing Pages: {pages}")
                else:
                    st.success("No missing pages detected.")

        st.caption(f"Showing {len(filtered)} of {diff_count} section differences")

        # ---- Download reports -----------------------------------------------
        if st.session_state.json_report and st.session_state.html_report:
            st.markdown("---")
            with st.expander("📥 Download Reports", expanded=False):
                st.markdown(
                    "**JSON** is machine-readable; **HTML** is a self-contained visual report. "
                    "Both contain all differences across all categories."
                )
                render_download_buttons(
                    st.session_state.json_report,
                    st.session_state.html_report,
                )