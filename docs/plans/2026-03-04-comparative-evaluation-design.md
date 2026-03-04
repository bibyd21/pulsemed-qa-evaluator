# Comparative Evaluation Tiers — Design Doc

## Problem
The current demo shows the full PulseMed pipeline but doesn't show *why* each layer matters. An interviewer might ask: "Why not just use an LLM judge?" We need a visual, data-driven answer.

## Solution
Add `--mode compare` that runs all 12 scenarios through three evaluation tiers and generates a comparative report showing progressive improvement.

## Three Evaluation Tiers

### Tier A — "Raw LLM" (no safeguards)
- **Chatbot**: Unguarded responses (no system prompt, no knowledge grounding)
- **Evaluation**: LLM judge only (no deterministic checks)
- **Purpose**: Shows what most "ChatGPT wrapper" healthcare bots look like
- **Expected**: Catches ~2/6 critical issues (obvious ones the judge notices)

### Tier B — "Constrained LLM + LLM Judge" (partial)
- **Chatbot**: Knowledge-grounded, constrained system prompt (same as current)
- **Evaluation**: LLM judge only (deterministic layer disabled)
- **Purpose**: Shows "industry standard" LLM-as-judge approach
- **Expected**: Catches ~4/6 critical issues (misses PII patterns, drug cross-reactivity)

### Tier C — "Full PulseMed Pipeline" (complete)
- **Chatbot + Evaluation**: Exactly what exists today
- **Purpose**: The gold standard — deterministic + LLM hybrid
- **Expected**: Catches 6/6 critical issues

### NotebookLM Validation
- Pre-scripted external grounding assessments per scenario
- Shows independent AI confirming our pipeline's findings
- Adds third-party credibility to the evaluation methodology

## New Files
- `pulsemed_qa/comparative.py` — Tier A/B mock data, NotebookLM results, ComparativeReport dataclass
- `pulsemed_qa/report_comparative.py` — Comparative HTML report generator

## Modified Files
- `run_evaluation.py` — Add `--mode compare` handling
- `pulsemed_qa/report.py` — Add `print_comparative_terminal_report()`

## Data Model
```python
@dataclass
class NotebookLMResult:
    scenario_id: str
    grounding_confidence: float  # 0-100%
    assessment: str
    agrees_with_pipeline: bool

@dataclass
class ComparativeReport:
    tier_a: EvaluationReport
    tier_b: EvaluationReport
    tier_c: EvaluationReport
    notebook_lm: dict[str, NotebookLMResult]
    timestamp: datetime
```

## Comparative HTML Report Structure
1. Executive summary — "Why layered evaluation matters" with 3 pass rates
2. Tier comparison cards — Pass rate, critical count, avg safety score per tier
3. Delta visualization — What each tier caught vs missed
4. Per-scenario comparison — Side-by-side results, color-coded agreement
5. NotebookLM column — External grounding confidence per scenario
6. Methodology footer — Three-tier rationale

## CLI
```
python run_evaluation.py --mode compare          # all tiers, mock data
python run_evaluation.py --mode compare --verbose # with per-check details
```

## Verification
- `--mode compare` produces comparative HTML report
- Tier A detects fewer issues than Tier B, Tier B fewer than Tier C
- NotebookLM results align with Tier C findings
- Existing `--mode mock` and `--mode live` unchanged
