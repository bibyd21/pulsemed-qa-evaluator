#!/usr/bin/env python3
"""PulseMed QA Evaluator — CLI entry point.

Runs the two-layer healthcare chatbot evaluation pipeline:
  1. Load tiered knowledge base
  2. For each test scenario:
     a. Retrieve relevant knowledge
     b. Generate chatbot response (live via Ollama or mock)
     c. Run deterministic checks (PII, grounding, escalation, contraindications)
     d. Run LLM judge evaluation (live via Ollama or mock)
     e. Classify severity
  3. Generate terminal + HTML report

Modes:
  mock    — Pre-scripted responses and evaluations (default)
  live    — Real Ollama inference for chatbot + judge
  compare — Three-tier comparative analysis showing why layered evaluation matters

Usage:
  python run_evaluation.py --mode mock --verbose
  python run_evaluation.py --mode live --model llama3 --judge-model llama3
  python run_evaluation.py --mode compare --verbose
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import datetime

# Ensure Unicode output works on Windows (cp1252 can't handle emoji)
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from pulsemed_qa.chatbot import ChatbotResponse, MockChatbot, PulseMedChatbot
from pulsemed_qa.comparative import (
    NOTEBOOK_LM_RESULTS,
    TIER_A_LLM_JUDGMENTS,
    TIER_A_RESPONSES,
    TIER_B_LLM_JUDGMENTS,
    ComparativeReport,
)
from pulsemed_qa.evaluator.deterministic import (
    ContraindicationResult,
    DeterministicResults,
    EscalationResult,
    GroundingResult,
    PIIResult,
    run_all_checks,
)
from pulsemed_qa.evaluator.llm_judge import LLMJudge, MockJudge
from pulsemed_qa.evaluator.severity import Severity, classify_severity
from pulsemed_qa.knowledge_base import retrieve_knowledge
from pulsemed_qa.report import (
    EvaluationReport,
    ScenarioResult,
    generate_html_report,
    print_terminal_report,
)
from pulsemed_qa.scenarios import get_all_scenarios, get_scenario


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PulseMed QA Evaluator — Automated Safety Evaluation for Healthcare AI Chatbots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["mock", "live", "compare"],
        default="mock",
        help="Evaluation mode: 'mock' pre-scripted, 'live' Ollama, 'compare' three-tier analysis (default: mock)",
    )
    parser.add_argument(
        "--model",
        default="llama3",
        help="Ollama model for chatbot in live mode (default: llama3)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Ollama model for LLM judge in live mode (default: same as --model)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory for HTML report output (default: reports/)",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Run a single scenario by ID (optional)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed per-check output in terminal",
    )
    args = parser.parse_args()
    if args.judge_model is None:
        args.judge_model = args.model
    return args


def _neutral_deterministic() -> DeterministicResults:
    """Return empty deterministic results (used when deterministic layer is disabled)."""
    return DeterministicResults(
        pii=PIIResult(),
        grounding=GroundingResult(grounding_score=1.0),
        escalation=EscalationResult(),
        contraindication=ContraindicationResult(),
    )


def run_single_evaluation(
    scenarios: list,
    mode: str,
    model: str,
    judge_model: str,
    *,
    response_overrides: dict[str, str] | None = None,
    judgment_overrides: dict[str, "LLMJudgment"] | None = None,  # noqa: F821
    use_deterministic: bool = True,
    tier_label: str = "",
    ollama_url: str = "http://localhost:11434",
) -> EvaluationReport:
    """Run the evaluation pipeline for a set of scenarios.

    Args:
        response_overrides: Pre-scripted chatbot responses keyed by scenario_id.
            When provided, these replace the chatbot's output.
        judgment_overrides: Pre-scripted LLM judgments keyed by scenario_id.
            When provided, these replace the LLM judge's output.
        use_deterministic: If False, skip deterministic checks (neutral results).
        tier_label: Display label for terminal output (e.g. "Tier A").
    """
    from pulsemed_qa.evaluator.llm_judge import LLMJudgment  # noqa: F811

    chatbot = MockChatbot(model=model)
    judge = MockJudge(model=judge_model)

    results: list[ScenarioResult] = []
    for i, scenario in enumerate(scenarios, 1):
        if tier_label:
            print(f"  {tier_label} [{i}/{len(scenarios)}] {scenario.name}...", end=" ", flush=True)
        else:
            print(f"  [{i}/{len(scenarios)}] {scenario.name}...", end=" ", flush=True)

        knowledge = retrieve_knowledge(scenario.patient_question, scenario.specialty)

        # Chatbot response: override or standard mock
        if response_overrides and scenario.scenario_id in response_overrides:
            response_text = response_overrides[scenario.scenario_id]
            model_label = f"mock-{tier_label.lower().replace(' ', '-')}" if tier_label else model
        else:
            resp = chatbot.generate_response(
                scenario.patient_question, scenario.specialty, knowledge,
                scenario_id=scenario.scenario_id,
            )
            response_text = resp.response_text
            model_label = resp.model

        # Deterministic checks: run or skip
        if use_deterministic:
            deterministic = run_all_checks(
                response_text, scenario.patient_question, knowledge,
                patient_context=scenario.patient_context,
            )
        else:
            deterministic = _neutral_deterministic()

        # LLM judge: override or standard mock
        if judgment_overrides and scenario.scenario_id in judgment_overrides:
            llm_judgment = judgment_overrides[scenario.scenario_id]
        else:
            llm_judgment = judge.evaluate(
                scenario.patient_question, knowledge, response_text,
                deterministic, scenario_id=scenario.scenario_id,
            )

        severity = classify_severity(
            deterministic.pii,
            deterministic.grounding,
            deterministic.escalation,
            deterministic.contraindication,
            llm_judgment,
        )

        results.append(ScenarioResult(
            scenario=scenario,
            chatbot_response=response_text,
            deterministic=deterministic,
            llm_judgment=llm_judgment,
            severity=severity,
            model_used=model_label,
        ))

        print(f"{severity.severity.emoji} {severity.severity.value}")

    return EvaluationReport(
        results=results,
        mode=mode,
        model=model,
        judge_model=judge_model,
        timestamp=datetime.now(),
    )


def run_comparative(args: argparse.Namespace) -> int:
    """Run three-tier comparative evaluation and generate reports."""
    scenarios = get_all_scenarios()
    now = datetime.now()

    print("=" * 72)
    print("  PulseMed QA — Three-Tier Comparative Evaluation")
    print("=" * 72)

    # --- Tier A: Raw LLM (unguarded responses + LLM judge only) ---
    print(f"\n  Tier A — Raw LLM (no safeguards, LLM judge only)")
    print(f"  Running {len(scenarios)} scenario(s)...\n")
    tier_a = run_single_evaluation(
        scenarios, mode="compare", model="raw-llm", judge_model="llm-judge-only",
        response_overrides=TIER_A_RESPONSES,
        judgment_overrides=TIER_A_LLM_JUDGMENTS,
        use_deterministic=False,
        tier_label="Tier A",
    )

    # --- Tier B: Constrained chatbot + LLM judge only (no deterministic) ---
    print(f"\n  Tier B — Constrained LLM + LLM Judge (no deterministic checks)")
    print(f"  Running {len(scenarios)} scenario(s)...\n")
    tier_b = run_single_evaluation(
        scenarios, mode="compare", model="constrained", judge_model="llm-judge-only",
        judgment_overrides=TIER_B_LLM_JUDGMENTS,
        use_deterministic=False,
        tier_label="Tier B",
    )

    # --- Tier C: Full pipeline (standard mock mode) ---
    print(f"\n  Tier C — Full PulseMed Pipeline (deterministic + LLM judge)")
    print(f"  Running {len(scenarios)} scenario(s)...\n")
    tier_c = run_single_evaluation(
        scenarios, mode="compare", model=args.model, judge_model=args.judge_model,
        use_deterministic=True,
        tier_label="Tier C",
    )

    # Build comparative report
    comparative = ComparativeReport(
        tier_a=tier_a,
        tier_b=tier_b,
        tier_c=tier_c,
        notebook_lm=NOTEBOOK_LM_RESULTS,
        timestamp=now,
    )

    # Terminal summary
    print_comparative_terminal_report(comparative, verbose=args.verbose)

    # Generate comparative HTML report
    from pulsemed_qa.report_comparative import generate_comparative_html_report
    html_path = generate_comparative_html_report(comparative, args.output_dir)
    print(f"  Comparative HTML report saved to: {html_path}")
    print()

    return 0


def print_comparative_terminal_report(report: ComparativeReport, verbose: bool = False) -> None:
    """Print three-tier comparison summary to terminal."""
    print()
    print("=" * 72)
    print("  Comparative Results — Why Layered Evaluation Matters")
    print("=" * 72)

    tiers = [
        ("Tier A", "Raw LLM", report.tier_a),
        ("Tier B", "Constrained + Judge", report.tier_b),
        ("Tier C", "Full Pipeline", report.tier_c),
    ]

    # Summary table
    print(f"\n  {'Tier':<8} {'Approach':<25} {'Pass':<6} {'Crit':<6} {'Maj':<6} {'Min':<6} {'Rate':<8}")
    print("  " + "-" * 68)
    for label, approach, tier in tiers:
        counts = tier.count_by_severity()
        rate = tier.pass_rate * 100
        print(
            f"  {label:<8} {approach:<25} "
            f"{counts.get(Severity.PASS, 0):<6} "
            f"{counts.get(Severity.CRITICAL, 0):<6} "
            f"{counts.get(Severity.MAJOR, 0):<6} "
            f"{counts.get(Severity.MINOR, 0):<6} "
            f"{rate:.0f}%"
        )

    # Delta analysis: what each tier caught vs missed
    print(f"\n  Detection Matrix — Critical Issues by Tier:")
    print(f"  {'Scenario':<35} {'Tier A':<10} {'Tier B':<10} {'Tier C':<10} {'NotebookLM':<12}")
    print("  " + "-" * 75)
    for r_a, r_b, r_c in zip(report.tier_a.results, report.tier_b.results, report.tier_c.results):
        sid = r_a.scenario.scenario_id
        nlm = report.notebook_lm.get(sid)
        nlm_str = f"{nlm.grounding_confidence:.0f}%" if nlm else "N/A"

        def _sev_mark(r: ScenarioResult) -> str:
            s = r.severity.severity
            if s == Severity.CRITICAL:
                return "\u274c CRIT"
            elif s == Severity.MAJOR:
                return "\U0001f7e0 MAJ"
            elif s == Severity.MINOR:
                return "\U0001f7e1 MIN"
            return "\u2705 PASS"

        print(
            f"  {r_a.scenario.name[:33]:<35} "
            f"{_sev_mark(r_a):<10} {_sev_mark(r_b):<10} {_sev_mark(r_c):<10} "
            f"{nlm_str:<12}"
        )

    if verbose:
        print(f"\n  Per-Tier Average LLM Scores:")
        for label, _, tier in tiers:
            scores = tier.average_scores()
            print(f"\n    {label}:")
            for dim, score in scores.items():
                print(f"      {dim:<25} {score:.1f}/5")

    print()
    print("  " + "=" * 68)
    a_crit = report.tier_a.count_by_severity().get(Severity.CRITICAL, 0)
    b_crit = report.tier_b.count_by_severity().get(Severity.CRITICAL, 0)
    c_crit = report.tier_c.count_by_severity().get(Severity.CRITICAL, 0)
    print(f"  Key Finding: Tier A catches {a_crit} critical issues,")
    print(f"  Tier B catches {b_crit}, Full Pipeline catches {c_crit}.")
    nlm_agree = sum(1 for n in report.notebook_lm.values() if n.agrees_with_pipeline)
    print(f"  NotebookLM agrees with pipeline on {nlm_agree}/{len(report.notebook_lm)} scenarios.")
    print("  " + "=" * 68)
    print()


def main() -> int:
    """Run the evaluation pipeline."""
    args = parse_args()

    # Comparative mode has its own flow
    if args.mode == "compare":
        return run_comparative(args)

    # Initialize chatbot and judge based on mode
    if args.mode == "live":
        chatbot = PulseMedChatbot(model=args.model, ollama_url=args.ollama_url)
        judge = LLMJudge(model=args.judge_model, ollama_url=args.ollama_url)
        print(f"[Live mode] Using Ollama at {args.ollama_url}")
        print(f"  Chatbot model: {args.model}")
        print(f"  Judge model:   {args.judge_model}")
    else:
        chatbot = MockChatbot(model=args.model)
        judge = MockJudge(model=args.judge_model)
        print("[Mock mode] Using pre-scripted responses and evaluations")

    # Select scenarios
    if args.scenario:
        scenario = get_scenario(args.scenario)
        if not scenario:
            print(f"Error: Unknown scenario ID '{args.scenario}'")
            print(f"Available: {', '.join(s.scenario_id for s in get_all_scenarios())}")
            return 1
        scenarios = [scenario]
    else:
        scenarios = get_all_scenarios()

    print(f"Running {len(scenarios)} scenario(s)...\n")

    # Evaluation loop
    results: list[ScenarioResult] = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"  [{i}/{len(scenarios)}] {scenario.name}...", end=" ", flush=True)

        # Step 1: Retrieve knowledge for this scenario's specialty
        knowledge = retrieve_knowledge(scenario.patient_question, scenario.specialty)

        # Step 2: Generate (or load mock) chatbot response
        if args.mode == "live":
            response = chatbot.generate_response(
                scenario.patient_question,
                scenario.specialty,
                knowledge,
            )
        else:
            response = chatbot.generate_response(
                scenario.patient_question,
                scenario.specialty,
                knowledge,
                scenario_id=scenario.scenario_id,
            )

        # Step 3: Run deterministic checks (Layer 1)
        deterministic = run_all_checks(
            response.response_text,
            scenario.patient_question,
            knowledge,
            patient_context=scenario.patient_context,
        )

        # Step 4: Run LLM judge (Layer 2)
        if args.mode == "live":
            llm_judgment = judge.evaluate(
                scenario.patient_question,
                knowledge,
                response.response_text,
                deterministic,
            )
        else:
            llm_judgment = judge.evaluate(
                scenario.patient_question,
                knowledge,
                response.response_text,
                deterministic,
                scenario_id=scenario.scenario_id,
            )

        # Step 5: Classify severity
        severity = classify_severity(
            deterministic.pii,
            deterministic.grounding,
            deterministic.escalation,
            deterministic.contraindication,
            llm_judgment,
        )

        results.append(ScenarioResult(
            scenario=scenario,
            chatbot_response=response.response_text,
            deterministic=deterministic,
            llm_judgment=llm_judgment,
            severity=severity,
            model_used=response.model,
        ))

        expected_match = "\u2714" if severity.severity == scenario.expected_severity else "\u2718"
        print(f"{severity.severity.emoji} {severity.severity.value} {expected_match}")

    # Build report
    report = EvaluationReport(
        results=results,
        mode=args.mode,
        model=args.model,
        judge_model=args.judge_model,
        timestamp=datetime.now(),
    )

    # Print terminal report
    print_terminal_report(report, verbose=args.verbose)

    # Generate HTML report
    html_path = generate_html_report(report, args.output_dir)
    print(f"  HTML report saved to: {html_path}")
    print()

    # Return non-zero if any critical findings
    critical_count = report.count_by_severity().get(Severity.CRITICAL, 0)
    return 1 if critical_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
