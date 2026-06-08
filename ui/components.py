"""
ui/components.py

Reusable Streamlit UI building blocks.
"""

from __future__ import annotations

import streamlit as st

from models import ChangeType, ComparisonReport


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

_BADGE_CSS = {
    ChangeType.EXACT:              ("🟢", "#2e7d32", "#e8f5e9"),
    ChangeType.NEAR_EXACT:         ("🟩", "#558b2f", "#f1f8e9"),
    ChangeType.MODIFIED:           ("🟡", "#f57f17", "#fff8e1"),
    ChangeType.SEMANTIC:           ("🔵", "#1565c0", "#e3f2fd"),
    ChangeType.SEMANTIC_DIFFERENT: ("🔴", "#c62828", "#fce4ec"),   # NEW — red for semantically different
    ChangeType.ADDED:              ("➕", "#1b5e20", "#e8f5e9"),
    ChangeType.REMOVED:            ("➖", "#b71c1c", "#ffebee"),
    ChangeType.UNCHANGED:          ("⚪", "#616161", "#f5f5f5"),
}

_CHANGE_LABELS = {
    ChangeType.EXACT:              "Exact Match",
    ChangeType.NEAR_EXACT:         "Near Exact",
    ChangeType.MODIFIED:           "Modified",
    ChangeType.SEMANTIC:           "Semantic Change",
    ChangeType.SEMANTIC_DIFFERENT: "Semantically Different",   # NEW
    ChangeType.ADDED:              "Added",
    ChangeType.REMOVED:            "Removed",
    ChangeType.UNCHANGED:          "Unchanged",
}


# ---------------------------------------------------------------------------
# Similarity gauge
# ---------------------------------------------------------------------------

def render_similarity_gauge(score: float):
    pct = int(score * 100)
    if pct >= 85:
        color = "#2e7d32"
        label = "High Similarity"
    elif pct >= 60:
        color = "#f57f17"
        label = "Moderate Similarity"
    else:
        color = "#b71c1c"
        label = "Low Similarity"

    st.markdown(f"""
    <div style="text-align:center; padding: 20px 0;">
      <div style="font-size:72px; font-weight:900; color:{color}; line-height:1;">{pct}%</div>
      <div style="font-size:18px; color:{color}; font-weight:600; margin-top:4px;">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Stats bar
# ---------------------------------------------------------------------------

def render_stats_bar(report: ComparisonReport):
    ps = report.paragraph_stats
    cols = st.columns(6)
    metrics = [
        ("✏️ Modified",        ps.modified,             "#e65100"),
        ("💡 Semantic",        ps.semantic_changes,     "#1565c0"),
        ("🔴 Sem. Different",  ps.semantic_different,   "#c62828"),  # NEW
        ("➕ Added",           ps.added,                "#1b5e20"),
        ("➖ Removed",         ps.removed,              "#b71c1c"),
        ("🟩 Near-Exact",      ps.near_exact,           "#558b2f"),
    ]
    for col, (label, value, color) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div style="background:white; border-radius:10px; padding:16px; text-align:center;
                        box-shadow:0 2px 8px rgba(0,0,0,.08);">
              <div style="font-size:32px; font-weight:900; color:{color}">{value}</div>
              <div style="font-size:13px; color:#616161; margin-top:4px">{label}</div>
            </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Individual match card
# ---------------------------------------------------------------------------

def render_match_card(match, index: int):
    ct = match.change_type
    icon, fg, bg = _BADGE_CSS.get(ct, ("⚪", "#616161", "#f5f5f5"))
    label = _CHANGE_LABELS.get(ct, ct.value)
    sim_pct = f"{match.similarity_score:.0%}"

    text_a = match.chunk_a.text if match.chunk_a else ""
    text_b = match.chunk_b.text if match.chunk_b else ""
    section = (match.chunk_a.section_heading if match.chunk_a and match.chunk_a.section_heading
               else (match.chunk_b.section_heading if match.chunk_b else ""))

    title = f"{icon} {label} — {sim_pct}"
    if section:
        title += f"  |  §{section[:60]}"

    with st.expander(title, expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Document A (Original)**")
            st.markdown(f"""<div style="background:#fafafa; border-left:4px solid {fg};
                padding:12px; border-radius:6px; font-size:13px; white-space:pre-wrap;
                word-break:break-word; min-height:60px;">{text_a or '<em>— not present —</em>'}</div>""",
                unsafe_allow_html=True)
        with c2:
            st.markdown("**Document B (New)**")
            st.markdown(f"""<div style="background:{bg}; border-left:4px solid {fg};
                padding:12px; border-radius:6px; font-size:13px; white-space:pre-wrap;
                word-break:break-word; min-height:60px;">{text_b or '<em>— not present —</em>'}</div>""",
                unsafe_allow_html=True)

        if match.semantic_analysis:
            sa = match.semantic_analysis
            # Choose color: red for semantically different, blue for semantic change
            sem_bg = "#fce4ec" if ct == ChangeType.SEMANTIC_DIFFERENT else "#e3f2fd"
            sem_border = "#c62828" if ct == ChangeType.SEMANTIC_DIFFERENT else "#1565c0"
            sem_label = "⚠️ Semantically Different" if ct == ChangeType.SEMANTIC_DIFFERENT else "💡 Semantic Analysis"
            st.markdown(f"""
            <div style="background:{sem_bg}; border-left:3px solid {sem_border};
                padding:8px 12px; border-radius:4px; margin-top:8px; font-size:12px;">
              <strong>{sem_label}:</strong> {sa.change_type.value} &nbsp;|&nbsp;
              {sa.summary} &nbsp;|&nbsp; Confidence: <strong>{sa.confidence:.0%}</strong>
            </div>""", unsafe_allow_html=True)

        if match.critical_info_changes:
            rows_html = "".join(
                f'<tr><td style="padding:3px 8px; font-weight:600; color:#b71c1c;">{c.info_type.upper()}</td>'
                f'<td style="padding:3px 8px; text-decoration:line-through; color:#b71c1c;">{c.original}</td>'
                f'<td style="padding:3px 8px;">→</td>'
                f'<td style="padding:3px 8px; font-weight:700; color:#1b5e20;">{c.revised}</td></tr>'
                for c in match.critical_info_changes
            )
            st.markdown(f"""
            <div style="background:#fff3e0; border-left:3px solid #e65100;
                padding:8px 12px; border-radius:4px; margin-top:6px; font-size:12px;">
              <strong>⚠️ Critical Info Changes (Dates / Numbers):</strong>
              <table style="margin-top:6px; border-collapse:collapse;">{rows_html}</table>
            </div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Download buttons
# ---------------------------------------------------------------------------

def render_download_buttons(json_str: str, html_str: str):
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            label="📥 Download JSON Report",
            data=json_str,
            file_name="comparison_report.json",
            mime="application/json",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            label="📄 Download HTML Report",
            data=html_str,
            file_name="comparison_report.html",
            mime="text/html",
            use_container_width=True,
        )