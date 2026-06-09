"""
ui/components.py

Reusable Streamlit UI building blocks.
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
# Display-category mapping
# ---------------------------------------------------------------------------

_DISPLAY_CATEGORY_MAP: dict[str, str] = {
    # ChangeType.value → display category
    ChangeType.ADDED.value:              "Added",
    ChangeType.REMOVED.value:            "Removed",
    ChangeType.MODIFIED.value:           "Modified",
    ChangeType.SEMANTIC.value:           "Modified",         # semantic_change → Modified
    ChangeType.SEMANTIC_DIFFERENT.value: "Modified",         # semantically_different → Modified
}

# Badge colors per display category
_CATEGORY_BADGE: dict[str, tuple[str, str, str]] = {
    # (border_color, badge_bg, badge_label)
    "Semantically Different": ("#c62828", "#c62828", "SEMANTICALLY DIFFERENT"),
    "Modified":               ("#e65100", "#e65100", "MODIFIED"),
    "Added":                  ("#2e7d32", "#2e7d32", "ADDED"),
    "Removed":                ("#757575", "#757575", "REMOVED"),
}

# Dropdown options — exact spec
FILTER_OPTIONS = [
    "All Differences",
    "Semantically Different",
    "Modified",
    "Added",
    "Removed",
]


def get_display_category(match: ChunkMatch) -> str | None:
    """Map a ChunkMatch to one of the 4 display categories, or None if excluded."""
    return _DISPLAY_CATEGORY_MAP.get(match.change_type.value, None)


def classify_all_differences(matches: list[ChunkMatch]) -> list[tuple[ChunkMatch, str]]:
    """
    Filter matches to only the 4 displayable categories and return
    (match, category) pairs.
    """
    result: list[tuple[ChunkMatch, str]] = []
    for m in matches:
        cat = get_display_category(m)
        if cat is not None:
            result.append((m, cat))
    return result


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
# Individual match card (original expander-style — kept for compatibility)
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
# Meaning difference helpers (original — kept for backward compat)
# ---------------------------------------------------------------------------

def is_meaning_difference(match) -> bool:
    """Return True if this match represents a material meaning change."""
    if match.change_type == ChangeType.SEMANTIC_DIFFERENT:
        return True
    if match.change_type == ChangeType.MODIFIED and match.critical_info_changes:
        return True
    return False


def _build_reason(match) -> str:
    """Build a human-readable reason explaining the meaning change."""
    parts: list[str] = []

    # Semantic analysis summary
    if match.semantic_analysis and match.semantic_analysis.summary:
        parts.append(match.semantic_analysis.summary)

    # Critical info changes (dates / numbers)
    if match.critical_info_changes:
        for c in match.critical_info_changes:
            if c.revised and c.revised not in ("(removed)", "(changed)"):
                parts.append(f"{c.info_type.capitalize()} changed from \"{c.original}\" to \"{c.revised}\".")
            elif c.revised == "(removed)":
                parts.append(f"{c.info_type.capitalize()} \"{c.original}\" was removed.")
            else:
                parts.append(f"{c.info_type.capitalize()} \"{c.original}\" was changed.")

    return " ".join(parts) if parts else "Content meaning has materially changed between documents."


# ---------------------------------------------------------------------------
# Category-aware reason builder (for all 4 categories)
# ---------------------------------------------------------------------------

def _build_category_reason(match: ChunkMatch, category: str) -> str:
    """Build the REASON text per the specification for each category."""
    if category == "Semantically Different":
        base = ""
        if match.semantic_analysis and match.semantic_analysis.summary:
            base = match.semantic_analysis.summary + " "
        base += f"Core meaning has changed significantly. Cosine similarity: {match.semantic_score:.2f}"
        return base

    if category == "Modified":
        base = ""
        if match.semantic_analysis and match.semantic_analysis.summary:
            base = match.semantic_analysis.summary + " "
        # Append critical info changes
        if match.critical_info_changes:
            for c in match.critical_info_changes:
                if c.revised and c.revised not in ("(removed)", "(changed)"):
                    base += f"{c.info_type.capitalize()} changed from \"{c.original}\" to \"{c.revised}\". "
                elif c.revised == "(removed)":
                    base += f"{c.info_type.capitalize()} \"{c.original}\" was removed. "
                else:
                    base += f"{c.info_type.capitalize()} \"{c.original}\" was changed. "
        base += f"Text has been revised with partial meaning change. Fuzzy score: {match.fuzzy_score:.2f}"
        return base.strip()

    if category == "Added":
        return "This section appears only in the updated document."

    if category == "Removed":
        return "This section was present in the original but removed in the updated document."

    return "Content has changed between documents."


# ---------------------------------------------------------------------------
# Meaning difference card (original — kept for backward compat)
# ---------------------------------------------------------------------------

def render_meaning_diff_card(match, index: int, file_a_name: str = "", file_b_name: str = ""):
    """Render a clean card showing Original Text, Updated Text, and Reason."""
    text_a = match.chunk_a.text if match.chunk_a else ""
    text_b = match.chunk_b.text if match.chunk_b else ""
    reason = _build_reason(match)

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
      <div style="
          display: flex;
          align-items: center;
          margin-bottom: 16px;
          gap: 8px;
      ">
        <span style="
            background: #c62828;
            color: white;
            border-radius: 20px;
            padding: 3px 12px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.3px;
        ">DIFFERENCE #{index + 1}</span>
      </div>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 14px;">
        <div>
          <div style="
              font-size: 12px;
              font-weight: 700;
              color: #b71c1c;
              text-transform: uppercase;
              letter-spacing: 0.5px;
              margin-bottom: 6px;
          ">Original Text{file_a_label}</div>
          <div style="
              background: #fafafa;
              border-left: 4px solid #c62828;
              border-radius: 4px;
              padding: 12px 16px;
              font-size: 14px;
              line-height: 1.6;
              color: #424242;
              white-space: pre-wrap;
              word-break: break-word;
              min-height: 60px;
          ">{text_a or '<em style="color:#9e9e9e;">— not present —</em>'}</div>
        </div>
        <div>
          <div style="
              font-size: 12px;
              font-weight: 700;
              color: #b71c1c;
              text-transform: uppercase;
              letter-spacing: 0.5px;
              margin-bottom: 6px;
          ">Updated Text{file_b_label}</div>
          <div style="
              background: #fafafa;
              border-left: 4px solid #c62828;
              border-radius: 4px;
              padding: 12px 16px;
              font-size: 14px;
              line-height: 1.6;
              color: #424242;
              white-space: pre-wrap;
              word-break: break-word;
              min-height: 60px;
          ">{text_b or '<em style="color:#9e9e9e;">— not present —</em>'}</div>
        </div>
      </div>

      <div>
        <div style="
            font-size: 12px;
            font-weight: 700;
            color: #e65100;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        ">Reason</div>
        <div style="
            background: #fff3e0;
            border-left: 4px solid #e65100;
            border-radius: 4px;
            padding: 12px 16px;
            font-size: 13px;
            line-height: 1.6;
            color: #5d4037;
            font-style: italic;
        ">{reason}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# NEW: Category-aware difference card (all 4 categories)
# ---------------------------------------------------------------------------

def render_category_diff_card(
    match: ChunkMatch,
    category: str,
    index: int,
    file_a_name: str = "",
    file_b_name: str = "",
):
    """
    Render a difference card with category-specific badge color, text panels,
    and auto-generated reason.  Uses Streamlit-native columns for layout so
    that HTML snippets stay small and flush-left (avoiding the Streamlit
    markdown parser treating indented HTML as code blocks).
    """
    border_color, badge_bg, badge_label = _CATEGORY_BADGE.get(
        category, ("#757575", "#757575", category.upper())
    )

    reason = _build_category_reason(match, category)

    # --- Determine text for left / right panels --------------------------
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
        text_a_display = _html_escape(raw_a) if raw_a else '<em style="color:#9e9e9e;">— not present —</em>'
        text_b_display = _html_escape(raw_b) if raw_b else '<em style="color:#9e9e9e;">— not present —</em>'

    file_a_hint = f" — {_html_escape(file_a_name)}" if file_a_name else ""
    file_b_hint = f" — {_html_escape(file_b_name)}" if file_b_name else ""

    # --- Badge row -------------------------------------------------------
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">'
        f'<span style="background:{badge_bg};color:#fff;border-radius:20px;'
        f'padding:3px 12px;font-size:11px;font-weight:700;letter-spacing:.4px;">'
        f'{badge_label}</span>'
        f'<span style="font-size:14px;font-weight:700;color:#424242;">'
        f'DIFFERENCE #{index}</span></div>',
        unsafe_allow_html=True,
    )

    # --- Side-by-side text panels using st.columns -----------------------
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

    # --- Reason ----------------------------------------------------------
    st.markdown(
        '<div style="font-size:12px;font-weight:700;color:#e65100;'
        'text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;">'
        'REASON</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="background:#fff3e0;border-left:4px solid #e65100;'
        f'border-radius:4px;padding:12px 16px;font-size:13px;line-height:1.6;'
        f'color:#5d4037;font-style:italic;">{_html_escape(reason)}</div>',
        unsafe_allow_html=True,
    )

    # --- Card separator --------------------------------------------------
    st.markdown("---")


def _html_escape(text: str) -> str:
    """Minimal HTML escaping for safe rendering."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


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