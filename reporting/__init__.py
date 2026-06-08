"""
reporting/__init__.py

Generates downloadable JSON and HTML reports from a ComparisonReport.

Design choices
--------------
* JSON — direct Pydantic .model_dump() serialization.  Machine-readable,
  suitable for downstream processing or API responses.
* HTML — self-contained single-file report with inline CSS.
  No external dependencies, works offline.  Uses a color-coded diff view.
"""

from __future__ import annotations

import json
from datetime import datetime

from models import ChangeType, ChunkMatch, ComparisonReport
from config import CONFIG


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def generate_json_report(report: ComparisonReport) -> str:
    data = report.model_dump()
    return json.dumps(data, indent=2, default=str)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_CHANGE_COLORS = {
    ChangeType.EXACT:              ("#e8f5e9", "#2e7d32"),   # green
    ChangeType.NEAR_EXACT:         ("#f1f8e9", "#558b2f"),   # light green
    ChangeType.MODIFIED:           ("#fff8e1", "#f57f17"),   # amber
    ChangeType.SEMANTIC:           ("#e3f2fd", "#1565c0"),   # blue
    ChangeType.SEMANTIC_DIFFERENT: ("#fce4ec", "#c62828"),   # deep red — NEW
    ChangeType.ADDED:              ("#e8f5e9", "#1b5e20"),   # deep green
    ChangeType.REMOVED:            ("#ffebee", "#b71c1c"),   # red
    ChangeType.UNCHANGED:          ("#fafafa", "#424242"),
}

_CHANGE_LABELS = {
    ChangeType.EXACT:              "Exact Match",
    ChangeType.NEAR_EXACT:         "Near Exact",
    ChangeType.MODIFIED:           "Modified",
    ChangeType.SEMANTIC:           "Semantic Change",
    ChangeType.SEMANTIC_DIFFERENT: "Semantically Different",  # NEW
    ChangeType.ADDED:              "Added",
    ChangeType.REMOVED:            "Removed",
    ChangeType.UNCHANGED:          "Unchanged",
}

_SNIPPET = CONFIG.reporting.snippet_max_chars


def _truncate(text: str | None) -> str:
    if not text:
        return "<em>—</em>"
    if len(text) <= _SNIPPET:
        return text.replace("<", "&lt;").replace(">", "&gt;")
    return text[:_SNIPPET].replace("<", "&lt;").replace(">", "&gt;") + "…"



_DIFFERENCE_TYPES_REPORT = {
    ChangeType.ADDED, ChangeType.REMOVED, ChangeType.MODIFIED,
    ChangeType.NEAR_EXACT, ChangeType.SEMANTIC, ChangeType.SEMANTIC_DIFFERENT,
}


def _match_row(match: ChunkMatch) -> str:
    """Return HTML table row(s). Exact matches are excluded — only differences shown."""
    ct = match.change_type

    # Skip exact matches from the report
    if ct == ChangeType.EXACT:
        return ""

    bg, fg = _CHANGE_COLORS.get(ct, ("#fff", "#000"))
    label = _CHANGE_LABELS.get(ct, ct.value)

    text_a = _truncate(match.chunk_a.text if match.chunk_a else None)
    text_b = _truncate(match.chunk_b.text if match.chunk_b else None)
    section = match.chunk_a.section_heading if match.chunk_a else (
        match.chunk_b.section_heading if match.chunk_b else ""
    )

    extra_html = ""

    if match.semantic_analysis:
        sa = match.semantic_analysis
        sem_bg = "#fce4ec" if ct == ChangeType.SEMANTIC_DIFFERENT else "#e3f2fd"
        sem_border = "#c62828" if ct == ChangeType.SEMANTIC_DIFFERENT else "#1565c0"
        sem_icon = "&#x26A0;&#xFE0F; Semantically Different" if ct == ChangeType.SEMANTIC_DIFFERENT else "&#x1F4A1; Semantic Analysis"
        extra_html += (
            f'<div class="semantic-badge" style="background:{sem_bg};border-left-color:{sem_border};">' +
            f'<strong>{sem_icon}:</strong> {sa.change_type.value} &mdash; {sa.summary}' +
            f'<span class="confidence">Confidence: {sa.confidence:.0%}</span></div>'
        )

    if getattr(match, "critical_info_changes", None):
        ci_rows = "".join(
            f'<tr>' +
            f'<td class="ci-type">{c.info_type.upper()}</td>' +
            f'<td class="ci-old">{c.original}</td>' +
            f'<td class="ci-arrow">&rarr;</td>' +
            f'<td class="ci-new">{c.revised}</td>' +
            f'</tr>'
            for c in match.critical_info_changes
        )
        extra_html += (
            '<div class="critical-info-badge">' +
            '<strong>&#x26A0;&#xFE0F; Critical Info Changes (Dates / Numbers):</strong>' +
            f'<table class="ci-table">{ci_rows}</table>' +
            '</div>'
        )

    sim_pct = f"{match.similarity_score:.0%}"
    extra_row = (
        f'<tr style="background:{bg}"><td colspan="5">{extra_html}</td></tr>'
        if extra_html else ""
    )

    main_row = (
        f'<tr style="background:{bg}">' +
        f'<td><span class="badge" style="background:{fg};color:#fff">{label}</span></td>' +
        f'<td class="section-col">{section or "&mdash;"}</td>' +
        f'<td class="text-col">{text_a}</td>' +
        f'<td class="text-col">{text_b}</td>' +
        f'<td class="score-col">{sim_pct}</td>' +
        f'</tr>'
    )

    return main_row + extra_row

def generate_html_report(report: ComparisonReport) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sim_pct = f"{report.overall_similarity:.1%}"
    ps = report.paragraph_stats

    stats_html = f"""
    <div class="stats-grid">
      <div class="stat-card green"><div class="stat-num">{ps.exact_matches + ps.near_exact}</div><div>Exact / Near-Exact</div></div>
      <div class="stat-card amber"><div class="stat-num">{ps.modified}</div><div>Modified</div></div>
      <div class="stat-card blue"><div class="stat-num">{ps.semantic_changes}</div><div>Semantic Changes</div></div>
      <div class="stat-card red"><div class="stat-num">{ps.removed}</div><div>Removed</div></div>
      <div class="stat-card green2"><div class="stat-num">{ps.added}</div><div>Added</div></div>
    </div>
    """

    rows_html = "".join(_match_row(m) for m in report.matches)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Document Comparison Report</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #f5f5f5; color: #212121; font-size: 14px; }}
    header {{ background: #1a237e; color: white; padding: 24px 32px; }}
    header h1 {{ font-size: 24px; font-weight: 700; margin-bottom: 4px; }}
    header p {{ opacity: 0.8; font-size: 13px; }}
    .container {{ max-width: 1400px; margin: 24px auto; padding: 0 16px; }}
    .summary-card {{ background: white; border-radius: 12px; padding: 24px; margin-bottom: 20px;
                    box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
    .score-ring {{ font-size: 48px; font-weight: 900; color: #1a237e; }}
    .score-ring small {{ font-size: 18px; color: #757575; }}
    .summary-text {{ color: #424242; margin-top: 8px; line-height: 1.6; }}
    .meta-row {{ display: flex; gap: 32px; margin-top: 16px; flex-wrap: wrap; }}
    .meta-item {{ font-size: 13px; color: #616161; }}
    .meta-item strong {{ color: #212121; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin: 20px 0; }}
    .stat-card {{ background: white; border-radius: 10px; padding: 16px; text-align: center;
                 box-shadow: 0 2px 6px rgba(0,0,0,.07); font-size: 12px; color: #616161; }}
    .stat-num {{ font-size: 28px; font-weight: 800; margin-bottom: 4px; }}
    .green .stat-num {{ color: #2e7d32; }}
    .amber .stat-num {{ color: #e65100; }}
    .blue .stat-num {{ color: #1565c0; }}
    .red .stat-num {{ color: #b71c1c; }}
    .green2 .stat-num {{ color: #1b5e20; }}
    table {{ width: 100%; border-collapse: collapse; background: white;
            border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
    th {{ background: #1a237e; color: white; padding: 12px 14px; text-align: left;
         font-size: 12px; text-transform: uppercase; letter-spacing: .5px; font-weight: 600; }}
    td {{ padding: 10px 14px; border-bottom: 1px solid #f0f0f0; vertical-align: top; }}
    .badge {{ display: inline-block; padding: 3px 10px; border-radius: 99px;
             font-size: 11px; font-weight: 700; white-space: nowrap; }}
    .section-col {{ color: #616161; font-style: italic; font-size: 12px; max-width: 160px; }}
    .text-col {{ max-width: 380px; word-break: break-word; line-height: 1.5; }}
    .score-col {{ text-align: center; font-weight: 700; white-space: nowrap; }}
    .semantic-badge {{ background: #e3f2fd; border-left: 3px solid #1565c0;
                      padding: 8px 12px; margin: 4px 0; border-radius: 4px; font-size: 12px; }}
    .confidence {{ float: right; color: #757575; }}
    .critical-info-badge {{ background:#fff3e0; border-left:3px solid #e65100;
                      padding:8px 12px; margin:4px 0; border-radius:4px; font-size:12px; }}
    .ci-table {{ margin-top:4px; border-collapse:collapse; }}
    .ci-type {{ padding:2px 8px; font-weight:700; color:#c62828; font-size:11px; }}
    .ci-old {{ padding:2px 8px; text-decoration:line-through; color:#b71c1c; }}
    .ci-arrow {{ padding:2px 4px; color:#616161; }}
    .ci-new {{ padding:2px 8px; font-weight:700; color:#1b5e20; }}
    .deepred .stat-num {{ color:#c62828; }}
    footer {{ text-align: center; padding: 32px; color: #9e9e9e; font-size: 12px; }}
  </style>
</head>
<body>
  <header>
    <h1>&#128203; Document Comparison Report</h1>
    <p>Generated: {now} &nbsp;|&nbsp; Processing time: {report.processing_time_seconds}s</p>
  </header>

  <div class="container">

    <div class="summary-card">
      <div class="score-ring">{sim_pct} <small>similarity</small></div>
      <p class="summary-text">{report.document_level_summary}</p>
      <div class="meta-row">
        <div class="meta-item">&#128196; <strong>File A:</strong> {report.doc_a_metadata.filename}
          &nbsp;({report.doc_a_metadata.word_count:,} words)</div>
        <div class="meta-item">&#128196; <strong>File B:</strong> {report.doc_b_metadata.filename}
          &nbsp;({report.doc_b_metadata.word_count:,} words)</div>
      </div>
    </div>

    {stats_html}

    <table>
      <thead>
        <tr>
          <th>Change Type</th>
          <th>Section</th>
          <th>Document A (Original)</th>
          <th>Document B (New)</th>
          <th>Score</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>

  </div>

  <footer>Document Comparison System &mdash; Built with Docling, RapidFuzz &amp; SentenceTransformers</footer>
</body>
</html>"""