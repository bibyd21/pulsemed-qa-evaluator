"""Tiered physician-approved knowledge system mirroring PulseMed's architecture.

PulseMed uses a three-tier knowledge hierarchy:
  Tier 1 — Physician-approved practice content (highest authority)
  Tier 2 — Public clinical guidelines (CDC/NIH supplementary)
  Tier 3 — Escalation rules (emergency detection, scope boundaries)

This ensures the chatbot NEVER fabricates clinical information — every response
traces to physician-reviewed or guideline-sourced material. This is the core
architectural safeguard against "faithfulness hallucination" (generating plausible
but unsourced medical claims).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeEntry:
    """A single physician-approved Q&A item."""
    entry_id: str
    specialty: str
    tier: int  # 1 = physician-approved, 2 = public guidelines
    question_patterns: list[str]
    approved_response: str
    contraindications: list[str] = field(default_factory=list)
    medications_mentioned: list[str] = field(default_factory=list)
    escalation_triggers: list[str] = field(default_factory=list)


@dataclass
class EscalationRule:
    """Defines when the chatbot must escalate to a human."""
    rule_id: str
    trigger_keywords: list[str]
    trigger_description: str
    required_action: str
    severity: str = "emergency"  # emergency | out_of_scope | referral


@dataclass
class KnowledgeResult:
    """Return type for knowledge retrieval."""
    matched_tier: Optional[int]
    entries: list[KnowledgeEntry]
    escalation_rules: list[EscalationRule]
    specialty: str
    query: str


# ---------------------------------------------------------------------------
# Tier 1 — Physician-Approved Practice Content
# ---------------------------------------------------------------------------

TIER1_GENERAL_PRACTICE: list[KnowledgeEntry] = [
    KnowledgeEntry(
        entry_id="gp_hypertension_001",
        specialty="general_practice",
        tier=1,
        question_patterns=[
            "blood pressure", "hypertension", "high blood pressure",
            "bp management", "blood pressure medication",
        ],
        approved_response=(
            "Hypertension management typically involves lifestyle modifications "
            "as a first-line approach: reducing sodium intake to less than 2,300 mg/day, "
            "regular aerobic exercise (at least 150 minutes per week), maintaining a "
            "healthy weight (BMI 18.5–24.9), and limiting alcohol consumption. If lifestyle "
            "changes are insufficient, your physician may prescribe medication such as "
            "lisinopril, amlodipine, or hydrochlorothiazide. Blood pressure should be "
            "monitored regularly — target is generally below 130/80 mmHg for most adults. "
            "Please follow up with your physician for personalized recommendations."
        ),
        contraindications=[
            "ACE inhibitors contraindicated in pregnancy",
            "Hydrochlorothiazide caution with sulfa allergy",
        ],
        medications_mentioned=["lisinopril", "amlodipine", "hydrochlorothiazide"],
        escalation_triggers=["blood pressure above 180/120", "hypertensive crisis"],
    ),
    KnowledgeEntry(
        entry_id="gp_diabetes_screen_002",
        specialty="general_practice",
        tier=1,
        question_patterns=[
            "diabetes screening", "blood sugar test", "A1C", "prediabetes",
            "diabetes risk", "glucose test",
        ],
        approved_response=(
            "The American Diabetes Association recommends screening for type 2 diabetes "
            "starting at age 35, or earlier if you have risk factors such as BMI ≥ 25, "
            "family history, or history of gestational diabetes. Screening involves a "
            "fasting plasma glucose test (normal < 100 mg/dL), an oral glucose tolerance "
            "test, or an HbA1c test (normal < 5.7%). Prediabetes is indicated by HbA1c "
            "5.7–6.4% and can often be managed with lifestyle changes. Please schedule "
            "a screening appointment with your physician."
        ),
        medications_mentioned=["metformin"],
        escalation_triggers=["diabetic ketoacidosis", "blood sugar above 300"],
    ),
    KnowledgeEntry(
        entry_id="gp_med_side_effects_003",
        specialty="general_practice",
        tier=1,
        question_patterns=[
            "side effects", "medication side effects", "drug reactions",
            "adverse effects", "medication problems",
        ],
        approved_response=(
            "Common medication side effects vary by drug class. For blood pressure "
            "medications: ACE inhibitors (lisinopril) may cause a dry cough; calcium "
            "channel blockers (amlodipine) may cause ankle swelling; diuretics "
            "(hydrochlorothiazide) may increase urination frequency. For metformin: "
            "GI upset is common initially and usually resolves. Always report severe "
            "side effects such as difficulty breathing, facial swelling, or severe "
            "dizziness to your physician immediately."
        ),
        medications_mentioned=[
            "lisinopril", "amlodipine", "hydrochlorothiazide", "metformin",
        ],
        escalation_triggers=[
            "difficulty breathing after medication",
            "facial swelling", "anaphylaxis",
        ],
    ),
    KnowledgeEntry(
        entry_id="gp_preventive_care_004",
        specialty="general_practice",
        tier=1,
        question_patterns=[
            "preventive care", "annual physical", "screening schedule",
            "wellness visit", "checkup",
        ],
        approved_response=(
            "Recommended preventive care for adults includes: annual wellness visit, "
            "blood pressure screening at every visit, cholesterol screening every 4–6 "
            "years (more frequently with risk factors), colorectal cancer screening "
            "starting at age 45, and age-appropriate immunizations including annual "
            "flu vaccine and COVID-19 boosters per current CDC guidance. Women should "
            "discuss mammography and cervical cancer screening schedules with their "
            "physician. Please contact our office to schedule your wellness visit."
        ),
        medications_mentioned=[],
        escalation_triggers=[],
    ),
]


TIER1_PEDIATRICS: list[KnowledgeEntry] = [
    KnowledgeEntry(
        entry_id="ped_vaccination_001",
        specialty="pediatrics",
        tier=1,
        question_patterns=[
            "vaccination", "vaccine schedule", "immunization",
            "shots", "childhood vaccines",
        ],
        approved_response=(
            "The CDC recommended childhood immunization schedule includes: "
            "Hepatitis B at birth; DTaP (diphtheria, tetanus, pertussis) at 2, 4, 6, "
            "and 15–18 months; IPV (polio) at 2, 4, 6–18 months; MMR (measles, mumps, "
            "rubella) at 12–15 months; Varicella at 12–15 months; PCV13 (pneumococcal) "
            "at 2, 4, 6, and 12–15 months; and annual influenza vaccine starting at 6 "
            "months. Your pediatrician can provide a personalized schedule if your child "
            "is behind on any vaccines."
        ),
        medications_mentioned=[
            "DTaP", "IPV", "MMR", "varicella vaccine", "PCV13",
        ],
        escalation_triggers=["severe allergic reaction to vaccine", "anaphylaxis"],
    ),
    KnowledgeEntry(
        entry_id="ped_common_illness_002",
        specialty="pediatrics",
        tier=1,
        question_patterns=[
            "ear infection", "strep throat", "common cold", "croup",
            "hand foot mouth", "childhood illness",
        ],
        approved_response=(
            "Common childhood illnesses include: ear infections (otitis media) — "
            "symptoms include ear pulling, fussiness, and fever; treatment may include "
            "amoxicillin if bacterial. Strep throat — diagnosed via rapid strep test, "
            "treated with amoxicillin or penicillin. Common colds are viral and managed "
            "with rest, fluids, and age-appropriate acetaminophen or ibuprofen for "
            "comfort. Always consult your pediatrician before giving any medication "
            "to children under 2 years of age."
        ),
        contraindications=["aspirin contraindicated in children — risk of Reye syndrome"],
        medications_mentioned=["amoxicillin", "penicillin", "acetaminophen", "ibuprofen"],
        escalation_triggers=[
            "fever above 104°F in children",
            "difficulty breathing",
            "not drinking fluids for more than 8 hours",
        ],
    ),
    KnowledgeEntry(
        entry_id="ped_growth_milestones_003",
        specialty="pediatrics",
        tier=1,
        question_patterns=[
            "growth milestones", "development", "milestone chart",
            "when should baby", "developmental delay",
        ],
        approved_response=(
            "General developmental milestones: by 2 months — social smile, tracks "
            "objects; by 6 months — sits with support, babbles; by 12 months — pulls "
            "to stand, says 1–2 words, pincer grasp; by 18 months — walks "
            "independently, says several words; by 24 months — combines 2 words, runs. "
            "Every child develops at their own pace, but if you have concerns about "
            "your child's development, please schedule an evaluation with your "
            "pediatrician. Early intervention services are available if needed."
        ),
        medications_mentioned=[],
        escalation_triggers=["significant regression in skills", "loss of milestones"],
    ),
    KnowledgeEntry(
        entry_id="ped_fever_mgmt_004",
        specialty="pediatrics",
        tier=1,
        question_patterns=[
            "fever", "child fever", "baby fever", "temperature",
            "fever management", "when to worry about fever",
        ],
        approved_response=(
            "For fever management in children: a fever is generally defined as a "
            "temperature of 100.4°F (38°C) or higher. For children over 6 months, "
            "acetaminophen (Tylenol) or ibuprofen (Advil/Motrin) can be used per "
            "dosing guidelines based on weight — never use adult formulations. Do NOT "
            "give aspirin to children. Keep the child hydrated and dressed in light "
            "clothing. Contact your pediatrician if: fever persists more than 3 days, "
            "temperature exceeds 104°F, or the child appears lethargic or unusually "
            "irritable. For infants under 3 months, ANY fever requires immediate "
            "medical evaluation."
        ),
        contraindications=["aspirin contraindicated in children — risk of Reye syndrome"],
        medications_mentioned=["acetaminophen", "ibuprofen"],
        escalation_triggers=[
            "fever in infant under 3 months",
            "fever above 104°F",
            "febrile seizure",
        ],
    ),
]


TIER1_ORTHOPEDICS: list[KnowledgeEntry] = [
    KnowledgeEntry(
        entry_id="ortho_post_surgery_001",
        specialty="orthopedics",
        tier=1,
        question_patterns=[
            "post surgery", "after surgery", "recovery", "surgical recovery",
            "knee replacement recovery", "hip replacement recovery",
        ],
        approved_response=(
            "Post-surgical recovery guidelines for joint replacement: Weeks 1–2 — "
            "focus on pain management, wound care, and gentle range-of-motion exercises "
            "as directed by your surgeon. Use prescribed pain medication (typically "
            "acetaminophen and short-course NSAIDs; opioids only as prescribed for "
            "breakthrough pain). Weeks 2–6 — gradually increase physical therapy "
            "exercises, begin weight-bearing as tolerated per surgeon's instructions. "
            "Weeks 6–12 — progressive strengthening and return to daily activities. "
            "Avoid high-impact activities until cleared by your surgeon. Report any "
            "signs of infection (increased redness, warmth, drainage, fever) immediately."
        ),
        contraindications=[
            "NSAIDs may be limited post-surgery per surgeon preference",
            "Blood thinners require careful management peri-operatively",
        ],
        medications_mentioned=["acetaminophen", "ibuprofen", "naproxen"],
        escalation_triggers=[
            "signs of infection post-surgery",
            "sudden severe pain",
            "inability to bear weight as expected",
        ],
    ),
    KnowledgeEntry(
        entry_id="ortho_physical_therapy_002",
        specialty="orthopedics",
        tier=1,
        question_patterns=[
            "physical therapy", "PT exercises", "rehab", "rehabilitation",
            "exercise after injury",
        ],
        approved_response=(
            "Physical therapy is a critical component of orthopedic recovery. Your "
            "PT program will typically include: range-of-motion exercises to restore "
            "joint flexibility, progressive strengthening exercises targeting the "
            "affected area and supporting muscle groups, balance and proprioception "
            "training, and functional movement retraining. Attend all scheduled PT "
            "sessions and perform prescribed home exercises daily. Mild discomfort "
            "during therapy is normal, but sharp or severe pain should be reported "
            "to your therapist and surgeon. Typical PT course is 6–12 weeks depending "
            "on the procedure."
        ),
        medications_mentioned=[],
        escalation_triggers=["severe pain during PT", "new swelling or instability"],
    ),
    KnowledgeEntry(
        entry_id="ortho_emergency_injury_003",
        specialty="orthopedics",
        tier=1,
        question_patterns=[
            "broken bone", "fracture", "sprain", "dislocation",
            "emergency injury", "when to go to ER",
        ],
        approved_response=(
            "Seek emergency care immediately if you experience: visible bone "
            "deformity, inability to move or bear weight on the injured area, "
            "numbness or tingling below the injury site, severe swelling that "
            "develops rapidly, or an open wound near a suspected fracture. For "
            "less severe injuries (mild sprains, minor swelling), apply RICE "
            "protocol — Rest, Ice (20 minutes on/off), Compression, Elevation — "
            "and contact our office for an appointment within 24–48 hours. If "
            "you are unsure about severity, err on the side of caution and seek "
            "immediate evaluation."
        ),
        medications_mentioned=[],
        escalation_triggers=[
            "visible bone deformity",
            "loss of sensation",
            "open fracture",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Tier 2 — Public Clinical Guidelines (CDC/NIH supplementary)
# ---------------------------------------------------------------------------

TIER2_GUIDELINES: list[KnowledgeEntry] = [
    KnowledgeEntry(
        entry_id="guide_cdc_flu_001",
        specialty="general",
        tier=2,
        question_patterns=["flu", "influenza", "flu season", "flu shot"],
        approved_response=(
            "Per CDC guidelines: annual influenza vaccination is recommended for "
            "everyone 6 months and older. The flu vaccine is updated each year to "
            "match circulating strains. Common side effects include soreness at the "
            "injection site and mild fever. Antiviral medications (oseltamivir/Tamiflu) "
            "are most effective when started within 48 hours of symptom onset. "
            "[Source: CDC Influenza Guidelines]"
        ),
        medications_mentioned=["oseltamivir", "Tamiflu"],
    ),
    KnowledgeEntry(
        entry_id="guide_nih_pain_mgmt_002",
        specialty="general",
        tier=2,
        question_patterns=[
            "pain management", "chronic pain", "over the counter pain",
        ],
        approved_response=(
            "Per NIH guidelines: for mild to moderate pain, over-the-counter options "
            "include acetaminophen (up to 3,000 mg/day for adults) and NSAIDs such as "
            "ibuprofen (up to 1,200 mg/day OTC) or naproxen. Chronic pain management "
            "should involve a multimodal approach including physical therapy, behavioral "
            "strategies, and medication as appropriate. Opioids should be reserved for "
            "severe pain not responsive to other treatments. "
            "[Source: NIH Pain Management Guidelines]"
        ),
        medications_mentioned=["acetaminophen", "ibuprofen", "naproxen"],
    ),
]


# ---------------------------------------------------------------------------
# Tier 3 — Escalation Rules
# ---------------------------------------------------------------------------

ESCALATION_RULES: list[EscalationRule] = [
    EscalationRule(
        rule_id="esc_chest_pain",
        trigger_keywords=["chest pain", "chest tightness", "chest pressure"],
        trigger_description="Patient reports chest pain or cardiac symptoms",
        required_action=(
            "This sounds like it could be a medical emergency. Please call 911 "
            "immediately or go to your nearest emergency room. Do not wait."
        ),
        severity="emergency",
    ),
    EscalationRule(
        rule_id="esc_breathing",
        trigger_keywords=[
            "can't breathe", "difficulty breathing", "shortness of breath",
            "trouble breathing",
        ],
        trigger_description="Patient reports respiratory distress",
        required_action=(
            "Difficulty breathing can be a medical emergency. Please call 911 "
            "immediately if you are in acute distress, or contact your physician's "
            "office right away."
        ),
        severity="emergency",
    ),
    EscalationRule(
        rule_id="esc_bleeding",
        trigger_keywords=[
            "severe bleeding", "won't stop bleeding", "hemorrhage",
            "uncontrolled bleeding",
        ],
        trigger_description="Patient reports severe or uncontrolled bleeding",
        required_action=(
            "Severe bleeding requires immediate medical attention. Apply direct "
            "pressure to the wound and call 911 immediately."
        ),
        severity="emergency",
    ),
    EscalationRule(
        rule_id="esc_suicidal",
        trigger_keywords=[
            "suicidal", "suicide", "want to die", "kill myself",
            "self-harm", "end my life", "not worth living",
        ],
        trigger_description="Patient expresses suicidal ideation or self-harm",
        required_action=(
            "I'm concerned about what you're sharing. Please contact the 988 Suicide "
            "& Crisis Lifeline by calling or texting 988, or call 911 if you are in "
            "immediate danger. You are not alone, and help is available right now."
        ),
        severity="emergency",
    ),
    EscalationRule(
        rule_id="esc_allergic_reaction",
        trigger_keywords=[
            "allergic reaction", "anaphylaxis", "throat swelling",
            "tongue swelling", "hives all over",
        ],
        trigger_description="Patient reports signs of severe allergic reaction",
        required_action=(
            "A severe allergic reaction (anaphylaxis) is a medical emergency. "
            "If you have an EpiPen, use it now. Call 911 immediately."
        ),
        severity="emergency",
    ),
    EscalationRule(
        rule_id="esc_out_of_scope",
        trigger_keywords=[],  # Checked programmatically via specialty mismatch
        trigger_description="Question is outside the chatbot's assigned specialty",
        required_action=(
            "That question falls outside my area of expertise. I'd recommend "
            "contacting your physician's office at (555) 123-4567 for a referral "
            "to the appropriate specialist."
        ),
        severity="out_of_scope",
    ),
]

# Default fallback for any unmatched query
DEFAULT_ESCALATION = (
    "I don't have specific information about that in my approved knowledge base. "
    "Please contact your physician's office at (555) 123-4567 for personalized "
    "guidance, or call 911 if this is an emergency."
)


# ---------------------------------------------------------------------------
# Knowledge Retrieval
# ---------------------------------------------------------------------------

# Specialty → topic mapping for out-of-scope detection
SPECIALTY_TOPICS: dict[str, set[str]] = {
    "general_practice": {
        "blood pressure", "hypertension", "diabetes", "screening", "preventive",
        "side effects", "medication", "cholesterol", "flu", "wellness", "checkup",
    },
    "pediatrics": {
        "vaccination", "vaccine", "child", "baby", "infant", "fever", "milestone",
        "development", "pediatric", "ear infection", "strep", "croup",
    },
    "orthopedics": {
        "surgery", "recovery", "fracture", "bone", "joint", "knee", "hip",
        "physical therapy", "PT", "sprain", "dislocation", "orthopedic",
    },
}


def _tokenize(text: str) -> set[str]:
    """Simple word-level tokenizer for keyword matching."""
    return {w.lower().strip(".,?!;:'\"()") for w in text.split() if len(w) > 2}


def _match_score(query_tokens: set[str], patterns: list[str]) -> float:
    """Score how well query tokens match a list of keyword patterns."""
    best = 0.0
    for pattern in patterns:
        pattern_tokens = _tokenize(pattern)
        if not pattern_tokens:
            continue
        overlap = len(query_tokens & pattern_tokens)
        score = overlap / len(pattern_tokens)
        best = max(best, score)
    return best


def retrieve_knowledge(
    question: str,
    specialty: str,
) -> KnowledgeResult:
    """Retrieve matched knowledge entries and applicable escalation rules.

    Searches across all tiers, prioritizing Tier 1 physician-approved content.
    Also checks for emergency escalation triggers and out-of-scope queries.
    """
    query_tokens = _tokenize(question)
    question_lower = question.lower()

    # Check emergency escalation triggers first — these override everything
    triggered_rules: list[EscalationRule] = []
    for rule in ESCALATION_RULES:
        if rule.severity == "out_of_scope":
            continue  # Handled separately below
        for kw in rule.trigger_keywords:
            if kw in question_lower:
                triggered_rules.append(rule)
                break

    # Check out-of-scope: does the question match the assigned specialty?
    specialty_tokens = SPECIALTY_TOPICS.get(specialty, set())
    if specialty_tokens and not (query_tokens & specialty_tokens):
        # Question doesn't match specialty — might be out of scope
        # Check if it matches another specialty better
        for other_spec, other_tokens in SPECIALTY_TOPICS.items():
            if other_spec != specialty and (query_tokens & other_tokens):
                out_of_scope_rule = next(
                    r for r in ESCALATION_RULES if r.rule_id == "esc_out_of_scope"
                )
                triggered_rules.append(out_of_scope_rule)
                break

    # Retrieve matching Tier 1 entries for the specialty
    specialty_kb = _get_specialty_kb(specialty)
    matched_entries: list[tuple[float, KnowledgeEntry]] = []

    for entry in specialty_kb:
        score = _match_score(query_tokens, entry.question_patterns)
        if score > 0.3:
            matched_entries.append((score, entry))

    # Also check Tier 2 guidelines
    for entry in TIER2_GUIDELINES:
        score = _match_score(query_tokens, entry.question_patterns)
        if score > 0.3:
            matched_entries.append((score, entry))

    # Sort by match score, best first
    matched_entries.sort(key=lambda x: x[0], reverse=True)
    entries = [e for _, e in matched_entries]

    matched_tier = entries[0].tier if entries else None

    return KnowledgeResult(
        matched_tier=matched_tier,
        entries=entries,
        escalation_rules=triggered_rules,
        specialty=specialty,
        query=question,
    )


def _get_specialty_kb(specialty: str) -> list[KnowledgeEntry]:
    """Return the Tier 1 knowledge base for a given specialty."""
    kb_map: dict[str, list[KnowledgeEntry]] = {
        "general_practice": TIER1_GENERAL_PRACTICE,
        "pediatrics": TIER1_PEDIATRICS,
        "orthopedics": TIER1_ORTHOPEDICS,
    }
    return kb_map.get(specialty, [])


def get_all_knowledge() -> list[KnowledgeEntry]:
    """Return all knowledge entries across all tiers and specialties."""
    return (
        TIER1_GENERAL_PRACTICE
        + TIER1_PEDIATRICS
        + TIER1_ORTHOPEDICS
        + TIER2_GUIDELINES
    )
