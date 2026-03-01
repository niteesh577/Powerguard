"""
agents/biomech_agent.py
=======================
Biomechanics Agent — Node 1 in the LangGraph pipeline.

Receives: biomechanics summary, stress indices, user profile.
Reasons about: joint stress, technique deviations, mechanical risk.
Outputs: BiomechAgentOutput (validated by Pydantic before state injection).

Guardrails enforced here:
  - No diagnostic language (validated by BiomechAgentOutput.technique_summary)
  - Confidence required
  - Structured JSON output only
"""

import json
import time

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from pydantic import ValidationError

from config.settings import get_settings
from observability.logger import get_logger
from schemas.agent_state import AgentState
from schemas.response import BiomechAgentOutput

logger = get_logger("biomech_agent")

_SYSTEM_PROMPT = """You are an expert sports biomechanist specializing in powerlifting injury prevention.

Your role is to analyze biomechanics data and identify mechanical risk indicators.

STRICT GUARDRAILS — VIOLATION WILL CAUSE SYSTEM REJECTION:
1. NEVER use diagnostic language: "injured", "torn", "ruptured", "diagnosed", "fractured"
2. ALWAYS use probabilistic language: "elevated risk indicator", "suggests", "may indicate", "consistent with"
3. ALWAYS include a confidence score between 0.0 and 1.0
4. Respond ONLY with valid JSON — no prose, no markdown

REQUIRED JSON FORMAT:
{
  "joint_stress_flags": ["string — up to 5 specific observations"],
  "technique_summary": "2-4 sentence probabilistic assessment",
  "primary_risk_joint": "the joint with highest stress",
  "risk_signals": ["string — specific movement pattern concerns, up to 5"],
  "confidence": 0.0-1.0
}

Base your reasoning strictly on the provided data. Do not speculate beyond the evidence.
"""


def _build_human_prompt(state: AgentState) -> str:
    bio = state.get("biomech_summary", {})
    stress = state.get("stress_indices", {})
    profile = state.get("user_profile", {})

    return f"""LIFT ANALYSIS REQUEST

User Profile:
  - Experience: {profile.get('experience_level', 'unknown')}
  - Bodyweight: {profile.get('bodyweight_kg', '?')} kg
  - Training age: {profile.get('training_age_years', '?')} years
  - Injury history: {profile.get('injury_history', [])}

Lift: {state.get('lift_type', 'unknown').upper()}
Load: {state.get('weight_kg', 0)} kg  |  RPE: {state.get('rpe', 0)}

Biomechanics Summary:
{json.dumps(bio.get('overall_summary', {}), indent=2)}

Rep-Level Data ({len(bio.get('reps', []))} reps):
{json.dumps(bio.get('reps', [])[:3], indent=2)}

Computed Stress Indices (physics-based):
  Primary joint stress:    {stress.get('primary_joint_stress', 'N/A')} (label: {stress.get('stress_label', 'N/A')})
  Secondary joint stress:  {stress.get('secondary_joint_stress', 'N/A')}
  Symmetry stress:         {stress.get('symmetry_stress', 'N/A')}
  Overall stress index:    {stress.get('overall_stress', 'N/A')}

Analyze the mechanics and return your JSON assessment."""


def biomech_agent_node(state: AgentState) -> dict:
    """LangGraph node: Biomechanics Agent."""
    settings  = get_settings()
    t_start   = time.perf_counter()
    trace_id  = state.get("trace_id", "")

    logger.info("biomech_agent.start", trace_id=trace_id, lift=state.get("lift_type"))

    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=settings.groq_temperature,
        max_tokens=settings.groq_max_tokens,
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=_build_human_prompt(state)),
    ]

    raw_output = {}
    errors = list(state.get("errors", []))

    try:
        response = llm.invoke(messages)
        raw_output = JsonOutputParser().parse(response.content)

        # Validate through Pydantic guardrails
        validated = BiomechAgentOutput(**raw_output)
        output_dict = validated.model_dump()

        logger.info(
            "biomech_agent.success",
            trace_id=trace_id,
            primary_joint=validated.primary_risk_joint,
            confidence=validated.confidence,
        )

    except ValidationError as e:
        logger.error("biomech_agent.guardrail_violation", trace_id=trace_id, error=str(e))
        errors.append(f"biomech_agent validation error: {e.error_count()} violations")
        # Fallback: safe minimal output
        output_dict = {
            "joint_stress_flags": ["Data insufficient for detailed assessment"],
            "technique_summary": "Insufficient data to complete biomechanical risk assessment. Manual review recommended.",
            "primary_risk_joint": "unknown",
            "risk_signals": [],
            "confidence": 0.10,
        }
    except Exception as e:
        logger.error("biomech_agent.error", trace_id=trace_id, error=str(e))
        errors.append(f"biomech_agent error: {str(e)}")
        output_dict = {
            "joint_stress_flags": [],
            "technique_summary": "Agent unavailable — manual review required.",
            "primary_risk_joint": "unknown",
            "risk_signals": [],
            "confidence": 0.0,
        }

    duration_ms = (time.perf_counter() - t_start) * 1000
    trace_entry = {
        "agent_name":    "biomech_agent",
        "input_summary": f"lift={state.get('lift_type')}, load={state.get('weight_kg')}kg, stress={state.get('stress_indices', {}).get('overall_stress', 'N/A')}",
        "output":        output_dict,
        "duration_ms":   round(duration_ms, 1),
        "confidence":    output_dict.get("confidence", 0.0),
    }

    return {
        "biomech_output": output_dict,
        "agent_traces":   state.get("agent_traces", []) + [trace_entry],
        "errors":         errors,
    }
