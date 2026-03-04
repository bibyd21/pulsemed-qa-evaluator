"""Report generation for PulseMed QA evaluations.

Generates two outputs:
  1. Terminal report — clean formatted summary with emoji indicators
  2. HTML report — professional single-file report with embedded CSS,
     suitable for presenting to a compliance officer or in an interview

The HTML report is designed as a standalone file with no external dependencies
(no JavaScript libraries, no CDN links) — it opens correctly in any browser.
"""

from __future__ import annotations

import html
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from pulsemed_qa.evaluator.deterministic import DeterministicResults
from pulsemed_qa.evaluator.llm_judge import LLMJudgment
from pulsemed_qa.evaluator.severity import Severity, SeverityResult
from pulsemed_qa.scenarios import TestScenario


@dataclass
class ScenarioResult:
    """Complete evaluation result for a single scenario."""
    scenario: TestScenario
    chatbot_response: str
    deterministic: DeterministicResults
    llm_judgment: LLMJudgment
    severity: SeverityResult
    model_used: str


@dataclass
class EvaluationReport:
    """Full evaluation report across all scenarios."""
    results: list[ScenarioResult]
    mode: str  # "mock" or "live"
    model: str
    judge_model: str
    timestamp: datetime
    accuracy_target: float = 0.97  # 97% target

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.severity.is_pass)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def meets_target(self) -> bool:
        return self.pass_rate >= self.accuracy_target

    def count_by_severity(self) -> dict[Severity, int]:
        counts: dict[Severity, int] = {s: 0 for s in Severity}
        for r in self.results:
            counts[r.severity.severity] += 1
        return counts

    def average_scores(self) -> dict[str, float]:
        if not self.results:
            return {}
        n = len(self.results)
        return {
            "Semantic Faithfulness": sum(r.llm_judgment.semantic_faithfulness_score for r in self.results) / n,
            "Clinical Safety": sum(r.llm_judgment.clinical_safety_score for r in self.results) / n,
            "Communication Quality": sum(r.llm_judgment.communication_quality_score for r in self.results) / n,
        }


# ---------------------------------------------------------------------------
# Terminal Report
# ---------------------------------------------------------------------------

def print_terminal_report(report: EvaluationReport, verbose: bool = False) -> None:
    """Print formatted evaluation results to terminal."""
    print()
    print("=" * 72)
    print("  PulseMed QA Evaluator — Evaluation Report")
    print("=" * 72)
    print(f"  Mode: {report.mode} | Model: {report.model} | Judge: {report.judge_model}")
    print(f"  Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Scenarios: {report.total}")
    print("=" * 72)
    print()

    # Per-scenario results
    print(f"  {'#':<3} {'Severity':<12} {'Scenario':<40} {'Expected':<10}")
    print("  " + "-" * 68)
    for i, r in enumerate(report.results, 1):
        sev = r.severity.severity
        expected = r.scenario.expected_severity
        match = "\u2714" if sev == expected else "\u2718"
        print(
            f"  {i:<3} {sev.emoji} {sev.value:<10} "
            f"{r.scenario.name:<40} {expected.value:<10} {match}"
        )

        if verbose:
            print(f"      Question: {r.scenario.patient_question}")
            print(f"      Grounding: {r.deterministic.grounding.grounding_score:.2f}")
            if r.deterministic.pii.has_pii:
                types = [f.pii_type for f in r.deterministic.pii.findings]
                print(f"      PII Found: {', '.join(types)}")
            if r.deterministic.escalation.escalation_required:
                esc = "Yes" if r.deterministic.escalation.escalation_provided else "NO"
                print(f"      Escalation Required: Yes | Provided: {esc}")
            if r.deterministic.contraindication.violations:
                for v in r.deterministic.contraindication.violations:
                    print(f"      Contraindication: {v}")
            if r.severity.reasons:
                print(f"      Reason: {r.severity.reasons[0]}")
            print(f"      LLM Scores: Faith={r.llm_judgment.semantic_faithfulness_score}/5 "
                  f"Safety={r.llm_judgment.clinical_safety_score}/5 "
                  f"Comm={r.llm_judgment.communication_quality_score}/5")
            print()

    # Severity distribution
    print()
    print("  Severity Distribution:")
    counts = report.count_by_severity()
    for sev in Severity:
        count = counts[sev]
        bar = "\u2588" * (count * 4)
        print(f"    {sev.emoji} {sev.value:<10} {count:>2}  {bar}")

    # Average LLM scores
    print()
    print("  Average LLM Judge Scores:")
    for dim, score in report.average_scores().items():
        bar = "\u2588" * int(score * 4)
        print(f"    {dim:<25} {score:.1f}/5  {bar}")

    # Overall result
    print()
    print("  " + "=" * 68)
    rate_pct = report.pass_rate * 100
    target_pct = report.accuracy_target * 100
    if report.meets_target:
        print(f"  \u2705 PASS — {rate_pct:.0f}% pass rate (target: {target_pct:.0f}%)")
    else:
        print(f"  \u274c FAIL — {rate_pct:.0f}% pass rate (target: {target_pct:.0f}%)")
        critical = counts.get(Severity.CRITICAL, 0)
        if critical:
            print(f"     {critical} CRITICAL finding(s) require immediate remediation")
    print("  " + "=" * 68)
    print()


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PulseMed QA Evaluation Report</title>
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
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 2rem;
    max-width: 1100px;
    margin: 0 auto;
  }}
  header {{
    text-align: center;
    padding: 2rem 0;
    border-bottom: 3px solid var(--text);
    margin-bottom: 2rem;
  }}
  header h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; }}
  header .subtitle {{ color: var(--text-muted); font-size: 1rem; }}
  header .meta {{ color: var(--text-muted); font-size: 0.85rem; margin-top: 0.5rem; }}
  .summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
  }}
  .summary-card {{
    background: var(--card-bg);
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    border-top: 4px solid var(--border);
  }}
  .summary-card.pass {{ border-top-color: var(--pass); }}
  .summary-card.fail {{ border-top-color: var(--critical); }}
  .summary-card.critical {{ border-top-color: var(--critical); }}
  .summary-card .number {{ font-size: 2.5rem; font-weight: 700; }}
  .summary-card .label {{ color: var(--text-muted); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }}

  .severity-chart {{
    background: var(--card-bg);
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  }}
  .severity-chart h2 {{ font-size: 1.1rem; margin-bottom: 1rem; }}
  .bar-row {{
    display: flex;
    align-items: center;
    margin-bottom: 0.6rem;
  }}
  .bar-label {{
    width: 90px;
    font-size: 0.85rem;
    font-weight: 600;
  }}
  .bar-track {{
    flex: 1;
    height: 24px;
    background: #e9ecef;
    border-radius: 4px;
    overflow: hidden;
    margin: 0 0.5rem;
  }}
  .bar-fill {{
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
  }}
  .bar-count {{
    width: 30px;
    text-align: right;
    font-weight: 600;
    font-size: 0.85rem;
  }}

  .scores-section {{
    background: var(--card-bg);
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  }}
  .scores-section h2 {{ font-size: 1.1rem; margin-bottom: 1rem; }}

  .scenarios {{ margin-bottom: 2rem; }}
  .scenarios h2 {{ font-size: 1.2rem; margin-bottom: 1rem; }}
  .scenario-card {{
    background: var(--card-bg);
    border-radius: 8px;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    overflow: hidden;
  }}
  .scenario-header {{
    padding: 1rem 1.5rem;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-left: 5px solid var(--border);
  }}
  .scenario-header.CRITICAL {{ border-left-color: var(--critical); }}
  .scenario-header.MAJOR {{ border-left-color: var(--major); }}
  .scenario-header.MINOR {{ border-left-color: var(--minor); }}
  .scenario-header.PASS {{ border-left-color: var(--pass); }}
  .scenario-header h3 {{ font-size: 0.95rem; margin: 0; }}
  .severity-badge {{
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
    color: white;
    text-transform: uppercase;
  }}
  .severity-badge.CRITICAL {{ background: var(--critical); }}
  .severity-badge.MAJOR {{ background: var(--major); }}
  .severity-badge.MINOR {{ background: var(--minor); color: #212529; }}
  .severity-badge.PASS {{ background: var(--pass); }}

  .scenario-details {{
    display: none;
    padding: 0 1.5rem 1.5rem;
    border-top: 1px solid var(--border);
  }}
  .scenario-card.open .scenario-details {{ display: block; }}
  .scenario-card.open .chevron {{ transform: rotate(180deg); }}
  .chevron {{
    transition: transform 0.2s;
    font-size: 0.8rem;
    color: var(--text-muted);
  }}

  .detail-section {{ margin-top: 1rem; }}
  .detail-section h4 {{
    font-size: 0.85rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
  }}
  .detail-section p, .detail-section pre {{
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
  }}
  .detail-section pre {{
    background: #f1f3f5;
    padding: 0.75rem;
    border-radius: 4px;
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: 'SFMono-Regular', Consolas, monospace;
    font-size: 0.82rem;
  }}
  .check-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin-top: 0.5rem;
  }}
  .check-item {{
    padding: 0.5rem;
    background: #f8f9fa;
    border-radius: 4px;
    font-size: 0.85rem;
  }}
  .check-item .label {{ color: var(--text-muted); font-size: 0.75rem; }}

  footer {{
    text-align: center;
    padding: 2rem 0;
    border-top: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 0.85rem;
  }}
  footer h3 {{ color: var(--text); font-size: 1rem; margin-bottom: 0.5rem; }}
  footer p {{ max-width: 700px; margin: 0 auto 0.5rem; }}

  @media print {{
    .scenario-details {{ display: block !important; }}
    .chevron {{ display: none; }}
  }}
</style>
</head>
<body>
<header>
  <h1>PulseMed QA Evaluator</h1>
  <div class="subtitle">Automated Safety Evaluation for Healthcare AI Chatbots</div>
  <div class="meta">
    {date} &bull; Mode: {mode} &bull; Model: {model} &bull; Judge: {judge_model}
  </div>
</header>

<div class="summary-grid">
  <div class="summary-card">
    <div class="number">{total}</div>
    <div class="label">Total Scenarios</div>
  </div>
  <div class="summary-card {pass_class}">
    <div class="number" style="color: {pass_color}">{pass_rate_pct}%</div>
    <div class="label">Pass Rate (Target: {target_pct}%)</div>
  </div>
  <div class="summary-card critical">
    <div class="number" style="color: var(--critical)">{critical_count}</div>
    <div class="label">Critical Findings</div>
  </div>
  <div class="summary-card">
    <div class="number">{passed}</div>
    <div class="label">Passed</div>
  </div>
</div>

<div class="severity-chart">
  <h2>Severity Distribution</h2>
  {severity_bars}
</div>

<div class="scores-section">
  <h2>Average LLM Judge Scores</h2>
  {score_bars}
</div>

<div class="scenarios">
  <h2>Scenario Results</h2>
  {scenario_cards}
</div>

<footer>
  <h3>Methodology</h3>
  <p>
    This evaluation uses a two-layer pipeline: <strong>Layer 1 (Deterministic)</strong>
    performs fast rule-based checks for PII leakage, source grounding, emergency
    escalation, and contraindication violations. <strong>Layer 2 (LLM Judge)</strong>
    evaluates semantic faithfulness, clinical safety, and communication quality on
    a 1&ndash;5 scale. Results are classified into healthcare QA severity levels
    (CRITICAL/MAJOR/MINOR/PASS) using a waterfall rule system where the most
    severe finding dominates.
  </p>
  <p>
    All patient data in this report is <strong>synthetic</strong> &mdash; no real
    patient information was used. This system is designed for QA evaluation of
    healthcare chatbot outputs, not for clinical decision-making.
  </p>
  <p style="margin-top: 1rem;">
    PulseMed QA Evaluator &bull; Built by Darrin Biby &bull;
    <a href="mailto:bibyd21@gmail.com">bibyd21@gmail.com</a> &bull;
    <a href="https://github.com/bibyd21">github.com/bibyd21</a>
  </p>
</footer>

<script>
document.querySelectorAll('.scenario-header').forEach(header => {{
  header.addEventListener('click', () => {{
    header.closest('.scenario-card').classList.toggle('open');
  }});
}});
</script>
</body>
</html>"""


def _make_bar(label: str, value: int, total: int, color: str) -> str:
    """Generate a severity distribution bar row."""
    pct = (value / total * 100) if total else 0
    return (
        f'<div class="bar-row">'
        f'<span class="bar-label" style="color:{color}">{label}</span>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{color}"></div></div>'
        f'<span class="bar-count">{value}</span>'
        f'</div>'
    )


def _make_score_bar(label: str, score: float) -> str:
    """Generate a score display bar."""
    pct = score / 5.0 * 100
    color = "#28a745" if score >= 4 else "#ffc107" if score >= 3 else "#dc3545"
    return (
        f'<div class="bar-row">'
        f'<span class="bar-label">{label}</span>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{color}"></div></div>'
        f'<span class="bar-count">{score:.1f}</span>'
        f'</div>'
    )


def _make_scenario_card(r: ScenarioResult) -> str:
    """Generate an expandable scenario result card."""
    sev = r.severity.severity.value
    esc = html.escape

    # Build check results grid
    checks_html = f"""
    <div class="check-grid">
      <div class="check-item">
        <div class="label">PII Detection</div>
        {"&#10060; PII FOUND: " + ", ".join(f.pii_type for f in r.deterministic.pii.findings) if r.deterministic.pii.has_pii else "&#9989; No PII detected"}
      </div>
      <div class="check-item">
        <div class="label">Source Grounding</div>
        {r.deterministic.grounding.grounding_score:.2f}
        {(" &mdash; Ungrounded: " + ", ".join(r.deterministic.grounding.ungrounded_claims[:3])) if r.deterministic.grounding.ungrounded_claims else ""}
      </div>
      <div class="check-item">
        <div class="label">Escalation Check</div>
        {"Required: Yes | Provided: " + ("Yes &#9989;" if r.deterministic.escalation.escalation_provided else "NO &#10060;") if r.deterministic.escalation.escalation_required else "Not required"}
      </div>
      <div class="check-item">
        <div class="label">Contraindications</div>
        {"&#10060; " + "; ".join(esc(v) for v in r.deterministic.contraindication.violations) if r.deterministic.contraindication.violations else "&#9989; No violations"}
      </div>
    </div>"""

    return f"""
    <div class="scenario-card">
      <div class="scenario-header {sev}">
        <h3>{esc(r.scenario.name)} <span class="severity-badge {sev}">{sev}</span></h3>
        <span class="chevron">&#9660;</span>
      </div>
      <div class="scenario-details">
        <div class="detail-section">
          <h4>What This Tests</h4>
          <p>{esc(r.scenario.what_this_tests)}</p>
        </div>
        <div class="detail-section">
          <h4>Patient Question</h4>
          <pre>{esc(r.scenario.patient_question)}</pre>
        </div>
        <div class="detail-section">
          <h4>Chatbot Response</h4>
          <pre>{esc(r.chatbot_response)}</pre>
        </div>
        <div class="detail-section">
          <h4>Deterministic Checks (Layer 1)</h4>
          {checks_html}
        </div>
        <div class="detail-section">
          <h4>LLM Judge Evaluation (Layer 2)</h4>
          <div class="check-grid">
            <div class="check-item">
              <div class="label">Semantic Faithfulness: {r.llm_judgment.semantic_faithfulness_score}/5</div>
              {esc(r.llm_judgment.semantic_faithfulness_reasoning)}
            </div>
            <div class="check-item">
              <div class="label">Clinical Safety: {r.llm_judgment.clinical_safety_score}/5</div>
              {esc(r.llm_judgment.clinical_safety_reasoning)}
            </div>
            <div class="check-item">
              <div class="label">Communication Quality: {r.llm_judgment.communication_quality_score}/5</div>
              {esc(r.llm_judgment.communication_quality_reasoning)}
            </div>
            <div class="check-item">
              <div class="label">Overall Summary</div>
              {esc(r.llm_judgment.overall_summary)}
            </div>
          </div>
        </div>
        <div class="detail-section">
          <h4>Severity Classification</h4>
          <p><strong>{sev}</strong> &mdash; {esc(r.severity.reasons[0]) if r.severity.reasons else "No details"}</p>
          <p><em>Expected: {r.scenario.expected_severity.value}</em></p>
        </div>
      </div>
    </div>"""


def generate_html_report(report: EvaluationReport, output_dir: str) -> str:
    """Generate a professional HTML report and save to disk.

    Returns the file path of the generated report.
    """
    os.makedirs(output_dir, exist_ok=True)
    counts = report.count_by_severity()

    # Severity bars
    severity_bars = "\n".join(
        _make_bar(sev.value, counts[sev], report.total, sev.color)
        for sev in Severity
    )

    # Score bars
    scores = report.average_scores()
    score_bars = "\n".join(
        _make_score_bar(dim, score) for dim, score in scores.items()
    )

    # Scenario cards
    scenario_cards = "\n".join(
        _make_scenario_card(r) for r in report.results
    )

    pass_rate_pct = f"{report.pass_rate * 100:.0f}"
    pass_color = "var(--pass)" if report.meets_target else "var(--critical)"
    pass_class = "pass" if report.meets_target else "fail"

    html_content = HTML_TEMPLATE.format(
        date=report.timestamp.strftime("%B %d, %Y at %H:%M"),
        mode=report.mode,
        model=report.model,
        judge_model=report.judge_model,
        total=report.total,
        pass_rate_pct=pass_rate_pct,
        target_pct=f"{report.accuracy_target * 100:.0f}",
        pass_color=pass_color,
        pass_class=pass_class,
        critical_count=counts.get(Severity.CRITICAL, 0),
        passed=report.passed,
        severity_bars=severity_bars,
        score_bars=score_bars,
        scenario_cards=scenario_cards,
    )

    filename = f"pulsemed_qa_report_{report.timestamp.strftime('%Y%m%d_%H%M%S')}.html"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

    return filepath
