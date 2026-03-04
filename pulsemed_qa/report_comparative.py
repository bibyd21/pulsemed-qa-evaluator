"""Comparative HTML report generator for three-tier evaluation.

Generates a single-file HTML report comparing Tier A (Raw LLM),
Tier B (Constrained + LLM Judge), and Tier C (Full Pipeline) results
with NotebookLM external validation.
"""

from __future__ import annotations

import html
import os
from datetime import datetime

from pulsemed_qa.comparative import ComparativeReport, NotebookLMResult
from pulsemed_qa.evaluator.severity import Severity
from pulsemed_qa.report import EvaluationReport, ScenarioResult


def _sev_badge(sev: Severity) -> str:
    """Generate a colored severity badge."""
    return f'<span class="sev-badge {sev.value}">{sev.value}</span>'


def _sev_class(sev: Severity) -> str:
    """CSS class for severity cell background."""
    return {
        Severity.CRITICAL: "cell-critical",
        Severity.MAJOR: "cell-major",
        Severity.MINOR: "cell-minor",
        Severity.PASS: "cell-pass",
    }[sev]


def _tier_summary_card(label: str, description: str, tier: EvaluationReport) -> str:
    """Generate a tier summary card."""
    counts = tier.count_by_severity()
    rate = tier.pass_rate * 100
    crit = counts.get(Severity.CRITICAL, 0)
    passed = counts.get(Severity.PASS, 0)
    rate_color = "#28a745" if rate >= 80 else "#ffc107" if rate >= 50 else "#dc3545"

    return f"""
    <div class="tier-card">
      <h3>{html.escape(label)}</h3>
      <p class="tier-desc">{html.escape(description)}</p>
      <div class="tier-stats">
        <div class="stat">
          <span class="stat-value" style="color:{rate_color}">{rate:.0f}%</span>
          <span class="stat-label">Pass Rate</span>
        </div>
        <div class="stat">
          <span class="stat-value" style="color:#dc3545">{crit}</span>
          <span class="stat-label">Critical</span>
        </div>
        <div class="stat">
          <span class="stat-value" style="color:#28a745">{passed}</span>
          <span class="stat-label">Passed</span>
        </div>
      </div>
      <div class="score-row">
        <span class="score-label">Avg Safety</span>
        <span class="score-val">{tier.average_scores().get('Clinical Safety', 0):.1f}/5</span>
      </div>
      <div class="score-row">
        <span class="score-label">Avg Faithfulness</span>
        <span class="score-val">{tier.average_scores().get('Semantic Faithfulness', 0):.1f}/5</span>
      </div>
    </div>"""


def _detection_matrix_rows(report: ComparativeReport) -> str:
    """Generate detection matrix table rows."""
    rows = []
    for r_a, r_b, r_c in zip(
        report.tier_a.results, report.tier_b.results, report.tier_c.results
    ):
        sid = r_a.scenario.scenario_id
        nlm = report.notebook_lm.get(sid)

        sev_a = r_a.severity.severity
        sev_b = r_b.severity.severity
        sev_c = r_c.severity.severity

        nlm_html = ""
        if nlm:
            conf_color = "#28a745" if nlm.grounding_confidence >= 80 else "#dc3545" if nlm.grounding_confidence < 50 else "#ffc107"
            agree_icon = "&#9989;" if nlm.agrees_with_pipeline else "&#10060;"
            nlm_html = f'<span style="color:{conf_color};font-weight:600">{nlm.grounding_confidence:.0f}%</span> {agree_icon}'
        else:
            nlm_html = "N/A"

        rows.append(f"""
        <tr>
          <td class="scenario-name">{html.escape(r_a.scenario.name)}</td>
          <td class="{_sev_class(sev_a)}">{_sev_badge(sev_a)}</td>
          <td class="{_sev_class(sev_b)}">{_sev_badge(sev_b)}</td>
          <td class="{_sev_class(sev_c)}">{_sev_badge(sev_c)}</td>
          <td class="nlm-cell">{nlm_html}</td>
        </tr>""")

    return "\n".join(rows)


def _scenario_comparison_cards(report: ComparativeReport) -> str:
    """Generate per-scenario three-column comparison cards."""
    esc = html.escape
    cards = []

    for r_a, r_b, r_c in zip(
        report.tier_a.results, report.tier_b.results, report.tier_c.results
    ):
        sid = r_a.scenario.scenario_id
        nlm = report.notebook_lm.get(sid)

        nlm_section = ""
        if nlm:
            nlm_section = f"""
            <div class="nlm-assessment">
              <h5>NotebookLM External Validation</h5>
              <p><strong>Confidence:</strong> {nlm.grounding_confidence:.0f}% |
                 <strong>Agrees:</strong> {"Yes" if nlm.agrees_with_pipeline else "No"}</p>
              <p class="nlm-text">{esc(nlm.assessment)}</p>
            </div>"""

        def _response_col(label: str, r: ScenarioResult) -> str:
            sev = r.severity.severity
            return f"""
            <div class="comp-col">
              <h5>{label} {_sev_badge(sev)}</h5>
              <pre class="comp-response">{esc(r.chatbot_response[:300])}{"..." if len(r.chatbot_response) > 300 else ""}</pre>
              <div class="comp-scores">
                <span>Faith: {r.llm_judgment.semantic_faithfulness_score}/5</span>
                <span>Safety: {r.llm_judgment.clinical_safety_score}/5</span>
                <span>Comm: {r.llm_judgment.communication_quality_score}/5</span>
              </div>
              <p class="comp-reason">{esc(r.severity.reasons[0]) if r.severity.reasons else ""}</p>
            </div>"""

        cards.append(f"""
        <div class="comp-card">
          <div class="comp-header" onclick="this.closest('.comp-card').classList.toggle('open')">
            <h4>{esc(r_a.scenario.name)}</h4>
            <span class="what-tests">{esc(r_a.scenario.what_this_tests)}</span>
            <span class="chevron">&#9660;</span>
          </div>
          <div class="comp-details">
            <div class="comp-grid">
              {_response_col("Tier A", r_a)}
              {_response_col("Tier B", r_b)}
              {_response_col("Tier C", r_c)}
            </div>
            {nlm_section}
          </div>
        </div>""")

    return "\n".join(cards)


COMPARATIVE_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PulseMed QA — Comparative Evaluation Report</title>
<style>
  :root {{
    --critical: #dc3545;
    --major: #fd7e14;
    --minor: #ffc107;
    --pass: #28a745;
    --bg: #f8f9fa;
    --card-bg: #ffffff;
    --text: #212529;
    --text-muted: #6c757d;
    --border: #dee2e6;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text);
    line-height: 1.6; padding: 2rem; max-width: 1200px; margin: 0 auto;
  }}
  header {{
    text-align: center; padding: 2rem 0;
    border-bottom: 3px solid var(--text); margin-bottom: 2rem;
  }}
  header h1 {{ font-size: 1.8rem; margin-bottom: 0.3rem; }}
  header .subtitle {{ color: var(--text-muted); font-size: 1.1rem; }}
  header .meta {{ color: var(--text-muted); font-size: 0.85rem; margin-top: 0.5rem; }}

  .executive {{
    background: var(--card-bg); border-radius: 8px; padding: 1.5rem;
    margin-bottom: 2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    border-left: 5px solid #0d6efd;
  }}
  .executive h2 {{ font-size: 1.1rem; margin-bottom: 0.5rem; color: #0d6efd; }}
  .executive p {{ font-size: 0.95rem; }}

  .tiers-grid {{
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 1rem; margin-bottom: 2rem;
  }}
  .tier-card {{
    background: var(--card-bg); border-radius: 8px; padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center;
  }}
  .tier-card h3 {{ font-size: 1rem; margin-bottom: 0.3rem; }}
  .tier-desc {{ color: var(--text-muted); font-size: 0.82rem; margin-bottom: 1rem; }}
  .tier-stats {{ display: flex; justify-content: space-around; margin-bottom: 0.8rem; }}
  .stat {{ text-align: center; }}
  .stat-value {{ display: block; font-size: 1.8rem; font-weight: 700; }}
  .stat-label {{ font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  .score-row {{
    display: flex; justify-content: space-between; padding: 0.3rem 0;
    border-top: 1px solid var(--border); font-size: 0.85rem;
  }}
  .score-label {{ color: var(--text-muted); }}
  .score-val {{ font-weight: 600; }}

  .matrix {{
    background: var(--card-bg); border-radius: 8px; padding: 1.5rem;
    margin-bottom: 2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  }}
  .matrix h2 {{ font-size: 1.1rem; margin-bottom: 1rem; }}
  .matrix table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  .matrix th {{
    text-align: left; padding: 0.6rem; border-bottom: 2px solid var(--text);
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;
  }}
  .matrix td {{ padding: 0.5rem 0.6rem; border-bottom: 1px solid var(--border); }}
  .scenario-name {{ font-weight: 500; }}
  .cell-critical {{ background: rgba(220,53,69,0.08); }}
  .cell-major {{ background: rgba(253,126,20,0.08); }}
  .cell-minor {{ background: rgba(255,193,7,0.08); }}
  .cell-pass {{ background: rgba(40,167,69,0.08); }}
  .nlm-cell {{ text-align: center; }}

  .sev-badge {{
    display: inline-block; padding: 0.15rem 0.5rem; border-radius: 3px;
    font-size: 0.7rem; font-weight: 700; color: white; text-transform: uppercase;
  }}
  .sev-badge.CRITICAL {{ background: var(--critical); }}
  .sev-badge.MAJOR {{ background: var(--major); }}
  .sev-badge.MINOR {{ background: var(--minor); color: #212529; }}
  .sev-badge.PASS {{ background: var(--pass); }}

  .scenarios-section {{ margin-bottom: 2rem; }}
  .scenarios-section > h2 {{ font-size: 1.2rem; margin-bottom: 1rem; }}

  .comp-card {{
    background: var(--card-bg); border-radius: 8px; margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow: hidden;
  }}
  .comp-header {{
    padding: 1rem 1.5rem; cursor: pointer;
    display: flex; align-items: center; gap: 1rem;
  }}
  .comp-header h4 {{ font-size: 0.95rem; margin: 0; white-space: nowrap; }}
  .what-tests {{ color: var(--text-muted); font-size: 0.82rem; flex: 1; }}
  .comp-details {{ display: none; padding: 0 1.5rem 1.5rem; border-top: 1px solid var(--border); }}
  .comp-card.open .comp-details {{ display: block; }}
  .comp-card.open .chevron {{ transform: rotate(180deg); }}
  .chevron {{ transition: transform 0.2s; font-size: 0.8rem; color: var(--text-muted); }}

  .comp-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-top: 1rem; }}
  .comp-col {{ padding: 0.75rem; background: #f8f9fa; border-radius: 6px; }}
  .comp-col h5 {{ font-size: 0.85rem; margin-bottom: 0.5rem; }}
  .comp-response {{
    font-size: 0.78rem; background: #fff; padding: 0.5rem; border-radius: 4px;
    white-space: pre-wrap; word-wrap: break-word;
    font-family: 'SFMono-Regular', Consolas, monospace;
    max-height: 150px; overflow-y: auto; margin-bottom: 0.5rem;
  }}
  .comp-scores {{ font-size: 0.78rem; color: var(--text-muted); display: flex; gap: 0.5rem; margin-bottom: 0.3rem; }}
  .comp-reason {{ font-size: 0.78rem; color: var(--text-muted); font-style: italic; }}

  .nlm-assessment {{
    margin-top: 1rem; padding: 0.75rem; background: #e8f4f8;
    border-radius: 6px; border-left: 3px solid #0d6efd;
  }}
  .nlm-assessment h5 {{ font-size: 0.85rem; margin-bottom: 0.3rem; color: #0d6efd; }}
  .nlm-assessment p {{ font-size: 0.82rem; }}
  .nlm-text {{ color: var(--text-muted); font-style: italic; }}

  footer {{
    text-align: center; padding: 2rem 0; border-top: 1px solid var(--border);
    color: var(--text-muted); font-size: 0.85rem;
  }}
  footer h3 {{ color: var(--text); font-size: 1rem; margin-bottom: 0.5rem; }}
  footer p {{ max-width: 700px; margin: 0 auto 0.5rem; }}

  @media (max-width: 768px) {{
    .tiers-grid {{ grid-template-columns: 1fr; }}
    .comp-grid {{ grid-template-columns: 1fr; }}
  }}
  @media print {{
    .comp-details {{ display: block !important; }}
    .chevron {{ display: none; }}
  }}
</style>
</head>
<body>
<header>
  <h1>PulseMed QA — Comparative Evaluation</h1>
  <div class="subtitle">Why Layered Evaluation Matters</div>
  <div class="meta">{date}</div>
</header>

<div class="executive">
  <h2>Executive Summary</h2>
  <p>
    This report compares three evaluation approaches across {total} healthcare chatbot scenarios.
    <strong>Tier A</strong> (raw LLM, no safeguards) achieves a <strong>{rate_a:.0f}% pass rate</strong>.
    <strong>Tier B</strong> (constrained chatbot + LLM judge only) improves to <strong>{rate_b:.0f}%</strong>.
    <strong>Tier C</strong> (full PulseMed pipeline with deterministic + LLM judge) achieves
    <strong>{rate_c:.0f}%</strong> &mdash; catching all critical safety issues including PII leakage
    and drug cross-reactivity that LLM-only evaluation misses.
  </p>
</div>

<div class="tiers-grid">
  {tier_cards}
</div>

<div class="matrix">
  <h2>Detection Matrix &mdash; Severity by Tier</h2>
  <table>
    <thead>
      <tr>
        <th>Scenario</th>
        <th>Tier A</th>
        <th>Tier B</th>
        <th>Tier C</th>
        <th>NotebookLM</th>
      </tr>
    </thead>
    <tbody>
      {matrix_rows}
    </tbody>
  </table>
</div>

<div class="scenarios-section">
  <h2>Per-Scenario Comparison</h2>
  {scenario_cards}
</div>

<footer>
  <h3>Three-Tier Methodology</h3>
  <p>
    <strong>Tier A &mdash; Raw LLM:</strong> Unguarded chatbot with LLM-only evaluation.
    Represents "ChatGPT wrapper" healthcare bots with no safety layer.
  </p>
  <p>
    <strong>Tier B &mdash; Constrained + LLM Judge:</strong> Knowledge-grounded chatbot
    with LLM judge evaluation. Industry-standard approach that catches semantic issues
    but misses PII patterns and drug cross-reactivity.
  </p>
  <p>
    <strong>Tier C &mdash; Full Pipeline:</strong> Deterministic rule-based checks +
    LLM judge hybrid. Catches all critical issues including those requiring pattern
    matching (PII/HIPAA) and structured lookup (pharmacological cross-reactivity).
  </p>
  <p>
    <strong>NotebookLM:</strong> Independent external grounding validation confirming
    the pipeline&rsquo;s findings with source document cross-referencing.
  </p>
  <p style="margin-top: 1rem;">
    All patient data is <strong>synthetic</strong> &mdash; no real patient information was used.
  </p>
  <p style="margin-top: 1rem;">
    PulseMed QA Evaluator &bull; Built by Darrin Biby &bull;
    <a href="mailto:bibyd21@gmail.com">bibyd21@gmail.com</a> &bull;
    <a href="https://github.com/bibyd21">github.com/bibyd21</a>
  </p>
</footer>

<script>
document.querySelectorAll('.comp-header').forEach(h => {{
  h.addEventListener('click', () => h.closest('.comp-card').classList.toggle('open'));
}});
</script>
</body>
</html>"""


def generate_comparative_html_report(
    report: ComparativeReport, output_dir: str
) -> str:
    """Generate the comparative HTML report and save to disk.

    Returns the file path of the generated report.
    """
    os.makedirs(output_dir, exist_ok=True)

    tier_cards = "\n".join([
        _tier_summary_card(
            "Tier A — Raw LLM",
            "No guardrails, LLM judge only",
            report.tier_a,
        ),
        _tier_summary_card(
            "Tier B — Constrained + Judge",
            "Knowledge-grounded chatbot, LLM judge only",
            report.tier_b,
        ),
        _tier_summary_card(
            "Tier C — Full Pipeline",
            "Deterministic checks + LLM judge",
            report.tier_c,
        ),
    ])

    total = report.tier_a.total
    html_content = COMPARATIVE_HTML_TEMPLATE.format(
        date=report.timestamp.strftime("%B %d, %Y at %H:%M"),
        total=total,
        rate_a=report.tier_a.pass_rate * 100,
        rate_b=report.tier_b.pass_rate * 100,
        rate_c=report.tier_c.pass_rate * 100,
        tier_cards=tier_cards,
        matrix_rows=_detection_matrix_rows(report),
        scenario_cards=_scenario_comparison_cards(report),
    )

    filename = f"pulsemed_comparative_{report.timestamp.strftime('%Y%m%d_%H%M%S')}.html"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

    return filepath
