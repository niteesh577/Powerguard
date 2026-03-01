"""
agents/risk_agent.py
====================
Risk Synthesis Agent — Node 3 (final) in the LangGraph pipeline.

Receives: outputs from biomech + fatigue agents, recovery score, user profile.
Computes: master risk equation + personalized thresholds.
Outputs: RiskSynthesisOutput (full set of guardrail-validated recommendations).

Master risk equation:
  Risk = (Stress × Fatigue × ProgressionRate) / (Recovery × TissueCapacity)

Risk level thresholds (after personalization):
  LOW      < 0.25
  MODERATE 0.25 – 0.50
  HIGH     0.50 – 0.75
  CRITICAL > 0.75
"""

import json
import math
import time

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from pydantic import ValidationError

from config.settings import get_settings
from engines.recovery_engine import compute_tissue_capacity_multiplier
from data.models import UserProfile, ExperienceLevel
from observability.logger import get_logger
from schemas.agent_state import AgentState
from schemas.response import RiskLevel, RiskSynthesisOutput, Recommendation

logger = get_logger("risk_agent")

_GUARDRAIL_DISCLAIMER = (
    "This report contains risk indicators only and is NOT a medical diagnosis. "
    "Consult a qualified sports medicine professional for any health concerns."
)

_SYSTEM_PROMPT = """You are a sports medicine risk synthesis specialist for powerlifting injury prevention.

Your role: synthesize biomechanics and fatigue data into an actionable injury risk report.

ABSOLUTE GUARDRAILS — NON-NEGOTIABLE:
1. NEVER diagnose an injury or medical condition
2. ALWAYS use probabilistic risk language ("elevated risk indicators suggest", "data consistent with")
3. Every recommendation MUST specify a concrete action (not vague advice)
4. Include category for each recommendation: "Technique" | "Load" | "Recovery" | "Monitoring"
5. Include evidence_basis for each recommendation if possible
6. Confidence must reflect data quality — do NOT claim >0.90 confidence
7. Respond ONLY with valid JSON

REQUIRED JSON FORMAT:
{
  "primary_concern": "1-2 sentence most critical finding",
  "recommendations": [
    {
      "category": "Technique|Load|Recovery|Monitoring",
      "action": "Specific, measurable action",
      "priority": 1-5,
      "evidence_basis": "Brief scientific grounding or empty string"
    }
  ],
  "confidence": 0.0-1.0
}

Maximum 5 recommendations. Order by priority (1 = most urgent).
"""


def _compute_master_risk_score(state: AgentState, profile: UserProfile) -> tuple[float, str]:
    """
    Physics-inspired master risk equation.
    Risk = (Stress × Fatigue × ProgressionAmplifier) / (Recovery × TissueCapacity)

    All inputs are [0, 1]. Output is normalized and clamped to [0, 1].
    """
    stress     = state.get("stress_indices", {}).get("overall_stress", 0.5)
    fatigue    = state.get("fatigue_score", 0.5)
    recovery   = max(0.01, state.get("recovery_score", 0.6))  # never 0 (avoid div/0)
    progression = state.get("progression_rate", 0.05)

    # Progression rate amplifier: safe (5%) = 1.0×, warning (15%) = 1.5×, >25% = 2.0×
    prog_amp = 1.0 + min(progression / 0.25, 1.0)

    # Tissue capacity (experience-based personalization)
    capacity = compute_tissue_capacity_multiplier(profile)

    raw_risk = (stress * fatigue * prog_amp) / (recovery * capacity)

    # Normalize: 1.0 is the reference risk (moderate-high) at capacity=1.0
    # Clamp to [0, 1]
    normalized = math.tanh(raw_risk * 0.8)   # tanh keeps output smooth in [0, 1)

    # Boost if overreaching detected
    if state.get("fatigue_output", {}).get("overreaching_detected"):
        normalized = min(1.0, normalized * 1.25)

    return round(normalized, 3), _risk_level(normalized).value


def _risk_level(score: float) -> RiskLevel:
    if score < 0.25: return RiskLevel.LOW
    if score < 0.50: return RiskLevel.MODERATE
    if score < 0.75: return RiskLevel.HIGH
    return RiskLevel.CRITICAL


def _build_human_prompt(state: AgentState, risk_score: float, risk_level_str: str) -> str:
    b_out = state.get("biomech_output") or {}
    f_out = state.get("fatigue_output") or {}
    profile = state.get("user_profile", {})

    return f"""RISK SYNTHESIS REQUEST

Pre-computed Risk Score: {risk_score:.3f}  →  {risk_level_str}
(This score is physics-derived — use it as ground truth for your recommendations)

User Profile:
  Experience: {profile.get('experience_level', '?')} | Age: {profile.get('age', '?')} | BW: {profile.get('bodyweight_kg', '?')} kg
  Injury history: {profile.get('injury_history', [])}

Lift: {state.get('lift_type', '?').upper()} @ {state.get('weight_kg', 0)} kg, RPE {state.get('rpe', 0)}

Biomechanics Agent Findings:
  Primary risk joint:  {b_out.get('primary_risk_joint', 'N/A')}
  Stress flags:        {b_out.get('joint_stress_flags', [])}
  Risk signals:        {b_out.get('risk_signals', [])}
  Summary:             {b_out.get('technique_summary', 'N/A')}
  Confidence:          {b_out.get('confidence', 0):.2f}

Fatigue & Load Agent Findings:
  Fatigue level:          {f_out.get('fatigue_level', 'N/A')}
  Overreaching detected:  {f_out.get('overreaching_detected', False)}
  Progression concern:    {f_out.get('weekly_progression_concern', False)} ({f_out.get('weekly_progression_rate_pct', 0)}% this week)
  Velocity loss concern:  {f_out.get('velocity_loss_concern', False)}
  Assessment:             {f_out.get('load_assessment', 'N/A')}
  Confidence:             {f_out.get('confidence', 0):.2f}

Recovery Context:
  Recovery score: {state.get('recovery_score', 0.5):.3f}

Synthesize into a final risk assessment. Be specific, actionable, and evidence-grounded.
Return JSON only."""


def risk_agent_node(state: AgentState) -> dict:
    """LangGraph node: Risk Synthesis Agent."""
    settings  = get_settings()
    t_start   = time.perf_counter()
    trace_id  = state.get("trace_id", "")

    logger.info("risk_agent.start", trace_id=trace_id)

    # Reconstruct UserProfile for personalization
    profile_data = state.get("user_profile", {})
    try:
        profile = UserProfile(**profile_data)
    except Exception:
        profile = UserProfile(
            user_id="unknown", name="Unknown", age=25, sex="male",
            bodyweight_kg=80.0, training_age_years=1.0,
            experience_level=ExperienceLevel.intermediate,
        )

    # Compute master risk equation
    risk_score, risk_level_str = _compute_master_risk_score(state, profile)
    injury_prob_pct = round(risk_score * 85, 1)   # calibrated: max ~85% (not 100% — epistemic humility)

    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=settings.groq_temperature,
        max_tokens=settings.groq_max_tokens,
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_build_human_prompt(state, risk_score, risk_level_str)),
    ]

    errors = list(state.get("errors", []))
    output_dict = {}

    try:
        response        = llm.invoke(messages)
        raw_output      = JsonOutputParser().parse(response.content)

        # Build the full RiskSynthesisOutput
        recs = [Recommendation(**r) for r in raw_output.get("recommendations", [])]
        validated = RiskSynthesisOutput(
            risk_level=_risk_level(risk_score),
            risk_score=risk_score,
            injury_probability_pct=injury_prob_pct,
            primary_concern=raw_output.get("primary_concern", "See recommendations."),
            recommendations=recs,
            confidence=float(raw_output.get("confidence", 0.5)),
            guardrail_disclaimer=_GUARDRAIL_DISCLAIMER,
        )
        output_dict = validated.model_dump()

        logger.info(
            "risk_agent.success",
            trace_id=trace_id,
            risk_level=risk_level_str,
            risk_score=risk_score,
            injury_prob_pct=injury_prob_pct,
            confidence=validated.confidence,
        )

    except (ValidationError, Exception) as e:
        logger.error("risk_agent.error", trace_id=trace_id, error=str(e))
        errors.append(f"risk_agent error: {str(e)}")
        output_dict = _fallback_output(risk_score, risk_level_str, injury_prob_pct)

    duration_ms = (time.perf_counter() - t_start) * 1000
    trace_entry = {
        "agent_name":    "risk_agent",
        "input_summary": f"risk_score={risk_score}, level={risk_level_str}",
        "output":        output_dict,
        "duration_ms":   round(duration_ms, 1),
        "confidence":    output_dict.get("confidence", 0.0),
    }

    return {
        "risk_output":  output_dict,
        "agent_traces": state.get("agent_traces", []) + [trace_entry],
        "errors":       errors,
    }


def _fallback_output(risk_score: float, risk_level_str: str, injury_prob_pct: float) -> dict:
    """Rule-based fallback if LLM fails."""
    level = RiskLevel(risk_level_str)
    recs  = []
    if risk_score > 0.5:
        recs.append({"category": "Load", "action": "Reduce training load by 10% this week.", "priority": 1, "evidence_basis": ""})
    if risk_score > 0.25:
        recs.append({"category": "Recovery", "action": "Prioritize 8+ hours sleep and target protein intake of 2.0 g/kg.", "priority": 2, "evidence_basis": ""})
    recs.append({"category": "Monitoring", "action": "Record RPE and bar velocity every session to track fatigue trend.", "priority": 3, "evidence_basis": ""})

    return {
        "risk_level":             level.value,
        "risk_score":             risk_score,
        "injury_probability_pct": injury_prob_pct,
        "primary_concern":        "Automated risk synthesis unavailable. Physics-based risk score applied.",
        "recommendations":        recs,
        "confidence":             0.15,
        "guardrail_disclaimer":   _GUARDRAIL_DISCLAIMER,
    }
