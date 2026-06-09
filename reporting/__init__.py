"""
reporting/__init__.py

Generates downloadable JSON and HTML reports from a ComparisonReport.
Reports contain **all 4 difference categories** — Semantically Different,
Modified, Added, and Removed — with per-card badge colors and reasons.

Design choices
--------------
* JSON — filtered subset of Pydantic .model_dump() serialization.
  Machine-readable, suitable for downstream processing or API responses.
* HTML — self-contained single-file report with inline CSS.
  No external dependencies, works offline.  Clean table of diffs with
  category badges.
"""

from __future__ import annotations

import json
from datetime import datetime

from models import ChangeType, ChunkMatch, ComparisonReport
from config import CONFIG


# ---------------------------------------------------------------------------
# Display-category mapping (mirrors ui/components.py)
# ---------------------------------------------------------------------------

_DISPLAY_CATEGORY_MAP: dict[str, str] = {
    ChangeType.ADDED.value:              "Added",
    ChangeType.REMOVED.value:            "Removed",
    ChangeType.MODIFIED.value:           "Modified",
    ChangeType.SEMANTIC.value:           "Modified",
    ChangeType.SEMANTIC_DIFFERENT.value: "Modified",
}

_CATEGORY_COLORS: dict[str, tuple[str, str]] = {
    # (border_color / badge_bg, badge_label)
    "Semantically Different": ("#c62828", "SEMANTICALLY DIFFERENT"),
    "Modified":               ("#e65100", "MODIFIED"),
    "Added":                  ("#2e7d32", "ADDED"),
    "Removed":                ("#757575", "REMOVED"),
}


def _get_display_category(match: ChunkMatch) -> str | None:
    return _DISPLAY_CATEGORY_MAP.get(match.change_type.value, None)


def _classify_all(matches: list[ChunkMatch]) -> list[tuple[ChunkMatch, str]]:
    result: list[tuple[ChunkMatch, str]] = []
    for m in matches:
        cat = _get_display_category(m)
        if cat is not None:
            result.append((m, cat))
    return result


# ---------------------------------------------------------------------------
# Reason builder (matches ui/components.py _build_category_reason)
# ---------------------------------------------------------------------------

def _build_category_reason(match: ChunkMatch, category: str) -> str:
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
# Legacy helpers (kept for backward compat)
# ---------------------------------------------------------------------------

def _is_meaning_difference(match: ChunkMatch) -> bool:
    """Return True if this match represents a material meaning change."""
    if match.change_type == ChangeType.SEMANTIC_DIFFERENT:
        return True
    if match.change_type == ChangeType.MODIFIED and match.critical_info_changes:
        return True
    return False


def _build_reason(match: ChunkMatch) -> str:
    """Build a human-readable reason explaining the meaning change."""
    parts: list[str] = []

    if match.semantic_analysis and match.semantic_analysis.summary:
        parts.append(match.semantic_analysis.summary)

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
# JSON report
# ---------------------------------------------------------------------------

def generate_json_report(report: ComparisonReport) -> str:
    all_diffs = _classify_all(report.matches)

    output = {
        "doc_a": report.doc_a_metadata.model_dump(),
        "doc_b": report.doc_b_metadata.model_dump(),
        "total_chunks_analysed": len(report.matches),
        "total_differences": len(all_diffs),
        "differences": [],
    }

    for m, category in all_diffs:
        text_a = m.chunk_a.text if m.chunk_a else ""
        text_b = m.chunk_b.text if m.chunk_b else ""

        # For Added chunks, original text is empty
        if category == "Added":
            text_a = ""
        # For Removed chunks, updated text is empty
        if category == "Removed":
            text_b = ""

        entry = {
            "category": category,
            "original_text": text_a,
            "updated_text": text_b,
            "reason": _build_category_reason(m, category),
            "change_type": m.change_type.value,
            "similarity_score": m.similarity_score,
            "fuzzy_score": m.fuzzy_score,
            "semantic_score": m.semantic_score,
        }
        if m.semantic_analysis:
            entry["semantic_analysis"] = {
                "type": m.semantic_analysis.change_type.value,
                "confidence": m.semantic_analysis.confidence,
                "summary": m.semantic_analysis.summary,
            }
        if m.critical_info_changes:
            entry["critical_info_changes"] = [
                {
                    "info_type": c.info_type,
                    "original": c.original,
                    "revised": c.revised,
                }
                for c in m.critical_info_changes
            ]
        output["differences"].append(entry)

    return json.dumps(output, indent=2, default=str)


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

_SNIPPET = CONFIG.reporting.snippet_max_chars


def _escape(text: str | None) -> str:
    if not text:
        return "<em style='color:#9e9e9e;'>— not present —</em>"
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    if len(safe) > _SNIPPET:
        return safe[:_SNIPPET] + "…"
    return safe


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def generate_html_report(report: ComparisonReport) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    all_diffs = _classify_all(report.matches)
    diff_count = len(all_diffs)

    file_a = report.doc_a_metadata.filename
    file_b = report.doc_b_metadata.filename

    # Build cards
    cards_html = ""
    for idx, (m, category) in enumerate(all_diffs):
        border_color, badge_label = _CATEGORY_COLORS.get(
            category, ("#757575", category.upper())
        )

        # Text panels per category
        if category == "Added":
            text_a = "<em style='color:#9e9e9e; font-style:italic;'>—Not present in original document—</em>"
            text_b = _escape(m.chunk_b.text if m.chunk_b else None)
        elif category == "Removed":
            text_a = _escape(m.chunk_a.text if m.chunk_a else None)
            text_b = "<em style='color:#9e9e9e; font-style:italic;'>—Not present in updated document—</em>"
        else:
            text_a = _escape(m.chunk_a.text if m.chunk_a else None)
            text_b = _escape(m.chunk_b.text if m.chunk_b else None)

        reason = _escape(_build_category_reason(m, category))

        cards_html += (
            f'<div class="diff-card" style="border-left-color:{border_color};">'
            f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:16px;">'
            f'<span class="badge" style="background:{border_color};">{badge_label}</span>'
            f'<span style="font-size:14px; font-weight:700; color:#424242;">DIFFERENCE #{idx + 1}</span>'
            f'</div>'
            '<div class="texts-row">'
            '<div class="text-block">'
            f'<div class="text-label" style="color:{border_color};">ORIGINAL TEXT <span class="file-hint">&mdash; {file_a}</span></div>'
            f'<div class="text-box" style="border-left-color:{border_color};">{text_a}</div>'
            '</div>'
            '<div class="text-block">'
            f'<div class="text-label" style="color:{border_color};">UPDATED TEXT <span class="file-hint">&mdash; {file_b}</span></div>'
            f'<div class="text-box" style="border-left-color:{border_color};">{text_b}</div>'
            '</div>'
            '</div>'
            '<div class="reason-section">'
            '<div class="reason-label">REASON</div>'
            f'<div class="reason-box">{reason}</div>'
            '</div>'
            '</div>'
        )

    # No-results card
    if not all_diffs:
        cards_html = (
            '<div style="text-align:center; padding:60px 24px; color:#757575; font-size:16px;">'
            '<div style="font-size:48px; margin-bottom:12px;">&#127881;</div>'
            '<div style="font-weight:600; color:#2e7d32; font-size:18px; margin-bottom:8px;">'
            'Documents are semantically consistent</div>'
            '<div>No statements with materially different meaning were found.</div>'
            '</div>'
        )

    diff_count_color = '#c62828' if diff_count > 0 else '#2e7d32'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Document Differences Report</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #f5f5f5; color: #212121; font-size: 14px; }}

    header {{ background: linear-gradient(135deg, #b71c1c 0%, #c62828 50%, #d32f2f 100%); color: white; padding: 28px 32px; }}
    header h1 {{ font-size: 24px; font-weight: 700; margin-bottom: 4px; }}
    header p {{ opacity: 0.85; font-size: 13px; }}

    .container {{ max-width: 1200px; margin: 24px auto; padding: 0 16px; }}

    .summary-card {{
        background: white; border-radius: 12px; padding: 24px; margin-bottom: 28px;
        box-shadow: 0 2px 8px rgba(0,0,0,.08);
        display: flex; align-items: center; gap: 20px;
    }}
    .diff-count {{
        font-size: 48px; font-weight: 900;
        color: {diff_count_color};
        min-width: 80px; text-align: center;
    }}
    .diff-count small {{ font-size: 14px; color: #757575; display: block; font-weight: 500; }}
    .summary-meta {{ color: #616161; font-size: 13px; line-height: 1.7; }}
    .summary-meta strong {{ color: #212121; }}

    .diff-card {{
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-left: 5px solid #c62828;
        border-radius: 0 12px 12px 0;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }}
    .badge {{
        display: inline-block;
        background: #c62828; color: white;
        border-radius: 20px; padding: 3px 12px;
        font-size: 11px; font-weight: 700;
        letter-spacing: 0.4px;
    }}
    .texts-row {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        margin-bottom: 14px;
    }}
    .text-label {{
        font-size: 12px; font-weight: 700; color: #b71c1c;
        text-transform: uppercase; letter-spacing: 0.5px;
        margin-bottom: 6px;
    }}
    .file-hint {{
        font-weight: 400; color: #757575; font-size: 11px;
        text-transform: none; letter-spacing: 0;
    }}
    .text-box {{
        background: #fafafa;
        border-left: 4px solid #c62828;
        border-radius: 4px;
        padding: 12px 16px;
        font-size: 14px; line-height: 1.6; color: #424242;
        white-space: pre-wrap; word-break: break-word;
        min-height: 60px;
    }}
    .reason-label {{
        font-size: 12px; font-weight: 700; color: #e65100;
        text-transform: uppercase; letter-spacing: 0.5px;
        margin-bottom: 6px;
    }}
    .reason-box {{
        background: #fff3e0;
        border-left: 4px solid #e65100;
        border-radius: 4px;
        padding: 12px 16px;
        font-size: 13px; line-height: 1.6;
        color: #5d4037; font-style: italic;
    }}
    
    footer {{ text-align: center; padding: 32px; color: #9e9e9e; font-size: 12px; }}
  </style>
</head>
<body>
  <header>
    <h1>&#9888;&#65039; Document Differences Report</h1>
    <p>Generated: {now} &nbsp;|&nbsp; {diff_count} difference{'s' if diff_count != 1 else ''} found</p>
  </header>

  <div class="container">

    <div class="summary-card">
      <div class="diff-count">{diff_count}<small>difference{'s' if diff_count != 1 else ''}</small></div>
      <div class="summary-meta">
        <div>&#128196; <strong>File A:</strong> {file_a}
          &nbsp;({report.doc_a_metadata.word_count:,} words)</div>
        <div>&#128196; <strong>File B:</strong> {file_b}
          &nbsp;({report.doc_b_metadata.word_count:,} words)</div>
        <div>&#128230; <strong>{len(report.matches)}</strong> total chunks analysed</div>
      </div>
    </div>

    {cards_html}

  </div>

  <footer>Document Differences Report &mdash; Built with Docling, RapidFuzz &amp; SentenceTransformers</footer>
</body>
</html>"""