# PulseMed QA Evaluator

**Automated Safety Evaluation for Healthcare AI Chatbots**

A two-layer evaluation pipeline that tests healthcare chatbot responses for clinical safety, source faithfulness, HIPAA compliance, and communication quality. Built as a portfolio demonstration for healthcare AI quality assurance.

---

> **[Live Demo: Pulse Data Sanitizer](https://bibyd21.github.io/pulsemed-qa-evaluator/demo/)** — Interactive visualization of the multi-agent sanitization pipeline

---

## The Problem

Healthcare AI chatbots can hallucinate — generating plausible but fabricated medical information. In healthcare, hallucinations aren't just wrong, they're dangerous. A chatbot that invents a medication name, fabricates lab results, or fails to escalate chest pain to 911 can directly harm patients.

This evaluator implements the evaluation framework needed to catch these failures *before* a chatbot reaches patients.

## Architecture

```
Patient Question
       │
       ▼
┌─────────────────────┐
│   Knowledge Base     │  Tier 1: Physician-approved content
│   (3 specialties)    │  Tier 2: CDC/NIH guidelines
│                      │  Tier 3: Escalation rules
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Chatbot Response   │  Constrained by system prompt
│   (Live or Mock)     │  "Only use approved knowledge"
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌──────────┐
│ Layer 1  │ │ Layer 2   │
│ Determ-  │ │ LLM       │
│ inistic  │ │ Judge     │
│          │ │           │
│ • PII    │ │ • Semantic │
│ • Ground │ │   Faithful │
│ • Escal. │ │ • Clinical │
│ • Contra │ │   Safety   │
└────┬─────┘ │ • Comms    │
     │       └─────┬─────┘
     └──────┬──────┘
            ▼
┌─────────────────────┐
│  Severity Classifier │  CRITICAL → MAJOR → MINOR → PASS
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│   Report Generator   │  Terminal + HTML
└─────────────────────┘
```

## PulseMed's Tiered Knowledge System

PulseMed chatbots are **physician-controlled** — they use ONLY content approved by the practice's clinical team:

| Tier | Source | Authority |
|------|--------|-----------|
| **1** | Physician-approved practice Q&A | Highest — vetted by the practice |
| **2** | CDC/NIH public guidelines | Supplementary clinical reference |
| **3** | Escalation rules | Emergency detection + scope boundaries |

This architecture ensures every chatbot response traces to an authoritative source, preventing "faithfulness hallucinations" (plausible but unsourced medical claims).

## Quick Start

```bash
# Clone and run (no install needed — stdlib + requests only)
cd pulsemed-qa-evaluator

# Run with mock mode (no LLM required)
python run_evaluation.py --mode mock --verbose

# View the generated HTML report
# (opens in reports/ directory)
```

### Live Mode (with Ollama)

```bash
# Install Ollama and pull a model
ollama pull llama3

# Run live evaluation
python run_evaluation.py --mode live --model llama3 --verbose

# Use different models for chatbot and judge
python run_evaluation.py --mode live --model llama3 --judge-model llama3
```

### Comparative Mode

```bash
# Run three-tier comparison (demonstrates why layered evaluation matters)
python run_evaluation.py --mode compare --verbose
```

Comparative mode runs all 12 scenarios through three evaluation tiers:
- **Tier A — Raw LLM**: No guardrails, LLM judge only (~17% pass rate)
- **Tier B — Constrained + LLM Judge**: Knowledge-grounded chatbot, no deterministic checks (~67% pass rate)
- **Tier C — Full Pipeline**: Deterministic + LLM judge hybrid (~33% pass rate — catches ALL critical issues)

Generates a side-by-side HTML report showing what each tier catches and misses, with NotebookLM external validation.

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `mock` | `mock`, `live` (Ollama), or `compare` (three-tier analysis) |
| `--model` | `llama3` | Ollama model for chatbot |
| `--judge-model` | same as model | Ollama model for LLM judge |
| `--ollama-url` | `http://localhost:11434` | Ollama API endpoint |
| `--output-dir` | `reports/` | HTML report output directory |
| `--scenario` | all | Run single scenario by ID |
| `--verbose` | off | Detailed per-check output |

## Evaluation Methodology

### Why Two Layers?

**Layer 1 (Deterministic)** catches clear-cut violations fast — PII leaks, missing escalations, contraindication violations. These checks are reproducible, auditable, and don't require LLM inference.

**Layer 2 (LLM Judge)** catches nuanced issues that rules can't — semantic meaning preservation, subtle safety concerns, communication tone. It receives Layer 1 results as context, focusing its analysis where rules flagged concerns.

This hybrid approach mirrors how production healthcare QA works: automated checks first, human (or AI) review second.

### Severity Levels

| Level | Meaning | Auto-fail? | Examples |
|-------|---------|-----------|----------|
| **CRITICAL** | Could cause patient harm | Yes | Fabricated medication, missed 911 escalation, PII leak |
| **MAJOR** | Clinically inaccurate, no direct harm | No | Wrong specialty response, ungrounded claims |
| **MINOR** | Safe but suboptimal quality | No | Incomplete answer, poor tone |
| **PASS** | Faithful, safe, complete | — | Correct medical info with appropriate referrals |

## Test Scenarios

The evaluator includes 12 scenarios covering the full risk taxonomy:

- **4 PASS** — Verify correct responses aren't flagged
- **6 CRITICAL** — Hallucinated medications, fabricated diagnoses, missed emergencies, contraindication violations, PII leaks
- **1 MAJOR** — Specialty boundary violation
- **1 MINOR** — Incomplete but safe response

## HIPAA Compliance Notice

**This project uses synthetic data only.** No real patient information, protected health information (PHI), or actual medical records are used anywhere in this codebase. All patient names, dates of birth, and medical scenarios are fictional and created for demonstration purposes. The PII detection capabilities are tested using obviously synthetic identifiers.

## Dependencies

- Python 3.10+
- `requests` (for Ollama API in live mode)
- No other external dependencies

## Author

**Darrin Biby** — Healthcare QA & AI Safety
bibyd21@gmail.com | [github.com/bibyd21](https://github.com/bibyd21)
