"""
ui/components.py

Section-based side-by-side comparison UI for Streamlit.
Groups all differences by document section rather than raw chunks.
"""

from __future__ import annotations

import streamlit as st

from models import ChangeType, ComparisonReport, ChunkMatch


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

_BADGE_CSS = {
    ChangeType.EXACT:              ("🟢", "#2e7d32", "#e8f5e9"),
    ChangeType.NEAR_EXACT:         ("🟩", "#558b2f", "#f1f8e9"),
    ChangeType.MODIFIED:           ("🟡", "#f57f17", "#fff8e1"),
    ChangeType.SEMANTIC:           ("🔵", "#1565c0", "#e3f2fd"),
    ChangeType.SEMANTIC_DIFFERENT: ("🔴", "#c62828", "#fce4ec"),
    ChangeType.ADDED:              ("➕", "#1b5e20", "#e8f5e9"),
    ChangeType.REMOVED:            ("➖", "#b71c1c", "#ffebee"),
    ChangeType.UNCHANGED:          ("⚪", "#616161", "#f5f5f5"),
}

_CHANGE_LABELS = {
    ChangeType.EXACT:              "Exact Match",
    ChangeType.NEAR_EXACT:         "Near Exact",
    ChangeType.MODIFIED:           "Modified",
    ChangeType.SEMANTIC:           "Semantic Change",
    ChangeType.SEMANTIC_DIFFERENT: "Semantically Different",
    ChangeType.ADDED:              "Added",
    ChangeType.REMOVED:            "Removed",
    ChangeType.UNCHANGED:          "Unchanged",
}


# ---------------------------------------------------------------------------
# Display-category mapping
# ---------------------------------------------------------------------------

_DISPLAY_CATEGORY_MAP: dict[str, str] = {
    ChangeType.ADDED.value:              "Added",
    ChangeType.REMOVED.value:            "Removed",
    ChangeType.MODIFIED.value:           "Modified",
    ChangeType.SEMANTIC.value:           "Modified",
    ChangeType.SEMANTIC_DIFFERENT.value: "Modified",
}

# Badge colors per display category
_CATEGORY_BADGE: dict[str, tuple[str, str, str]] = {
    # (border_color, badge_bg, badge_label)
    "Semantically Different": ("#c62828", "#c62828", "SEMANTICALLY DIFFERENT"),
    "Modified":               ("#e65100", "#e65100", "MODIFIED"),
    "Added":                  ("#2e7d32", "#2e7d32", "ADDED"),
    "Removed":                ("#757575", "#757575", "REMOVED"),
}

# Dropdown options
FILTER_OPTIONS = [
    "All Differences",
    "Modified",
    "Added",
    "Removed",
]


def get_display_category(match: ChunkMatch) -> str | None:
    """Map a ChunkMatch to one of the 4 display categories, or None if excluded."""
    return _DISPLAY_CATEGORY_MAP.get(match.change_type.value, None)


def classify_all_differences(matches: list[ChunkMatch]) -> list[tuple[ChunkMatch, str]]:
    """
    Filter matches to only the displayable categories and return
    (match, category) pairs.
    """
    from comparison.display_utils import merge_matches
    result: list[tuple[ChunkMatch, str]] = []
    for m in matches:
        cat = get_display_category(m)
        if cat is not None:
            result.append((m, cat))
    return merge_matches(result)


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
        ("🔴 Sem. Different",  ps.semantic_different,   "#c62828"),
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
# Meaning difference helpers (backward compat)
# ---------------------------------------------------------------------------

def is_meaning_difference(match) -> bool:
    """Return True if this match represents a material meaning change."""
    if match.change_type == ChangeType.SEMANTIC_DIFFERENT:
        return True
    if match.change_type == ChangeType.MODIFIED and match.critical_info_changes:
        return True
    return False


# ---------------------------------------------------------------------------
# Section-based diff card (NEW — side-by-side)
# ---------------------------------------------------------------------------

def _html_escape(text: str) -> str:
    """Minimal HTML escaping for safe rendering."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_section_diff_card(
    diff,  # SectionDiff
    index: int,
    file_a_name: str = "",
    file_b_name: str = "",
):
    """
    Render a section-level difference card with side-by-side layout.
    Shows both original and revised text for MODIFIED sections,
    or just one side for ADDED/REMOVED.
    """
    status = diff.status
    heading = diff.heading or "(No heading)"

    # Colors per status
    if status == "MODIFIED":
        border_color = "#e65100"
        badge_bg = "#e65100"
        badge_label = "MODIFIED"
    elif status == "REMOVED":
        border_color = "#c62828"
        badge_bg = "#c62828"
        badge_label = "REMOVED"
    elif status == "ADDED":
        border_color = "#2e7d32"
        badge_bg = "#2e7d32"
        badge_label = "ADDED"
    else:
        border_color = "#757575"
        badge_bg = "#757575"
        badge_label = status.upper()

    file_a_hint = f" — {_html_escape(file_a_name)}" if file_a_name else ""
    file_b_hint = f" — {_html_escape(file_b_name)}" if file_b_name else ""

    # --- Badge + heading row ---
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">'
        f'<span style="background:{badge_bg};color:#fff;border-radius:20px;'
        f'padding:3px 12px;font-size:11px;font-weight:700;letter-spacing:.4px;">'
        f'{badge_label}</span>'
        f'<span style="font-size:14px;font-weight:700;color:#424242;">'
        f'#{index} &nbsp;|&nbsp; {_html_escape(heading)}</span></div>',
        unsafe_allow_html=True,
    )

    # Always use side-by-side columns
    col_a, col_b = st.columns(2)

    if status == "MODIFIED":
        orig_display = diff.highlighted_original if diff.highlighted_original else _html_escape(diff.original_text)
        rev_display = diff.highlighted_revised if diff.highlighted_revised else _html_escape(diff.revised_text)
        orig_bg = "#fafafa"
        rev_bg = "#fafafa"
    elif status == "REMOVED":
        orig_display = _html_escape(diff.original_text) if diff.original_text else '<em style="color:#9e9e9e;">— empty —</em>'
        rev_display = f'<em style="color:#9e9e9e;">— Content is missing from {_html_escape(file_b_name) if file_b_name else "Document B"} —</em>'
        orig_bg = "#ffebee"
        rev_bg = "#fafafa"
    elif status == "ADDED":
        orig_display = f'<em style="color:#9e9e9e;">— Content is missing from {_html_escape(file_a_name) if file_a_name else "Document A"} —</em>'
        rev_display = _html_escape(diff.revised_text) if diff.revised_text else '<em style="color:#9e9e9e;">— empty —</em>'
        orig_bg = "#fafafa"
        rev_bg = "#e8f5e9"
    else:
        orig_display = _html_escape(diff.original_text)
        rev_display = _html_escape(diff.revised_text)
        orig_bg = "#fafafa"
        rev_bg = "#fafafa"

    with col_a:
        st.markdown(
            f'<div style="font-size:12px;font-weight:700;color:{border_color};'
            f'text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;">'
            f'ORIGINAL TEXT'
            f'<span style="font-weight:400;color:#757575;font-size:11px;'
            f'text-transform:none;letter-spacing:0;">{file_a_hint}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="background:{orig_bg};border-left:4px solid {border_color};'
            f'border-radius:4px;padding:12px 16px;font-size:14px;line-height:1.6;'
            f'color:#424242;white-space:pre-wrap;word-break:break-word;'
            f'min-height:60px;">{orig_display}</div>',
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            f'<div style="font-size:12px;font-weight:700;color:{border_color};'
            f'text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;">'
            f'UPDATED TEXT'
            f'<span style="font-weight:400;color:#757575;font-size:11px;'
            f'text-transform:none;letter-spacing:0;">{file_b_hint}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="background:{rev_bg};border-left:4px solid {border_color};'
            f'border-radius:4px;padding:12px 16px;font-size:14px;line-height:1.6;'
            f'color:#424242;white-space:pre-wrap;word-break:break-word;'
            f'min-height:60px;">{rev_display}</div>',
            unsafe_allow_html=True,
        )

    # Separator
    st.markdown("---")


# ---------------------------------------------------------------------------
# Legacy card (kept for backward compat)
# ---------------------------------------------------------------------------

def render_category_diff_card(
    match: ChunkMatch,
    category: str,
    index: int,
    file_a_name: str = "",
    file_b_name: str = "",
):
    """Legacy chunk-level diff card — kept for backward compatibility."""
    from comparison.display_utils import highlight_diff
    border_color, badge_bg, badge_label = _CATEGORY_BADGE.get(
        category, ("#757575", "#757575", category.upper())
    )

    if category == "Added":
        text_a_display = '<em style="color:#9e9e9e;">—Not present in original document—</em>'
        raw_b = match.chunk_b.text if match.chunk_b else ""
        text_b_display = _html_escape(raw_b) if raw_b else '<em style="color:#9e9e9e;">— empty —</em>'
    elif category == "Removed":
        raw_a = match.chunk_a.text if match.chunk_a else ""
        text_a_display = _html_escape(raw_a) if raw_a else '<em style="color:#9e9e9e;">— empty —</em>'
        text_b_display = '<em style="color:#9e9e9e;">—Not present in updated document—</em>'
    else:
        raw_a = match.chunk_a.text if match.chunk_a else ""
        raw_b = match.chunk_b.text if match.chunk_b else ""
        html_a = _html_escape(raw_a)
        html_b = _html_escape(raw_b)
        if html_a and html_b:
            text_a_display, text_b_display = highlight_diff(html_a, html_b)
        else:
            text_a_display = html_a if html_a else '<em style="color:#9e9e9e;">— not present —</em>'
            text_b_display = html_b if html_b else '<em style="color:#9e9e9e;">— not present —</em>'

    file_a_hint = f" — {_html_escape(file_a_name)}" if file_a_name else ""
    file_b_hint = f" — {_html_escape(file_b_name)}" if file_b_name else ""

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">'
        f'<span style="background:{badge_bg};color:#fff;border-radius:20px;'
        f'padding:3px 12px;font-size:11px;font-weight:700;letter-spacing:.4px;">'
        f'{badge_label}</span>'
        f'<span style="font-size:14px;font-weight:700;color:#424242;">'
        f'DIFFERENCE #{index}</span></div>',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f'<div style="font-size:12px;font-weight:700;color:{border_color};'
            f'text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;">'
            f'ORIGINAL TEXT'
            f'<span style="font-weight:400;color:#757575;font-size:11px;'
            f'text-transform:none;letter-spacing:0;">{file_a_hint}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="background:#fafafa;border-left:4px solid {border_color};'
            f'border-radius:4px;padding:12px 16px;font-size:14px;line-height:1.6;'
            f'color:#424242;white-space:pre-wrap;word-break:break-word;'
            f'min-height:60px;">{text_a_display}</div>',
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f'<div style="font-size:12px;font-weight:700;color:{border_color};'
            f'text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;">'
            f'UPDATED TEXT'
            f'<span style="font-weight:400;color:#757575;font-size:11px;'
            f'text-transform:none;letter-spacing:0;">{file_b_hint}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="background:#fafafa;border-left:4px solid {border_color};'
            f'border-radius:4px;padding:12px 16px;font-size:14px;line-height:1.6;'
            f'color:#424242;white-space:pre-wrap;word-break:break-word;'
            f'min-height:60px;">{text_b_display}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")


def render_meaning_diff_card(match, index: int, file_a_name: str = "", file_b_name: str = ""):
    """Render a clean card showing Original Text, Updated Text."""
    text_a = match.chunk_a.text if match.chunk_a else ""
    text_b = match.chunk_b.text if match.chunk_b else ""

    file_a_label = f' <span style="font-weight:400; color:#757575; font-size:11px; text-transform:none; letter-spacing:0;">— {file_a_name}</span>' if file_a_name else ""
    file_b_label = f' <span style="font-weight:400; color:#757575; font-size:11px; text-transform:none; letter-spacing:0;">— {file_b_name}</span>' if file_b_name else ""

    st.markdown(f"""
    <div style="
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-left: 5px solid #c62828;
        border-radius: 0 12px 12px 0;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    ">
      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 14px;">
        <div>
          <div style="font-size: 12px; font-weight: 700; color: #b71c1c;
              text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
            Original Text{file_a_label}</div>
          <div style="background: #fafafa; border-left: 4px solid #c62828;
              border-radius: 4px; padding: 12px 16px; font-size: 14px;
              line-height: 1.6; color: #424242; white-space: pre-wrap;
              word-break: break-word; min-height: 60px;">
            {text_a or '<em style="color:#9e9e9e;">— not present —</em>'}</div>
        </div>
        <div>
          <div style="font-size: 12px; font-weight: 700; color: #b71c1c;
              text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
            Updated Text{file_b_label}</div>
          <div style="background: #fafafa; border-left: 4px solid #c62828;
              border-radius: 4px; padding: 12px 16px; font-size: 14px;
              line-height: 1.6; color: #424242; white-space: pre-wrap;
              word-break: break-word; min-height: 60px;">
            {text_b or '<em style="color:#9e9e9e;">— not present —</em>'}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


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