"""
agents/fatigue_agent.py
=======================
Fatigue & Load Agent — Node 2 in the LangGraph pipeline.

Receives: fatigue metrics, workout history summary, user profile, biomech output.
Reasons about: accumulated load, overreaching, progression rate risk.
Outputs: FatigueAgentOutput (validated by Pydantic).
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
from schemas.response import FatigueAgentOutput

logger = get_logger("fatigue_agent")

_SYSTEM_PROMPT = """You are an expert in sports science load management and powerlifting periodization.

Your role is to analyze fatigue accumulation, training load, and overreaching risk.

STRICT GUARDRAILS:
1. NEVER diagnose injury or illness — only identify load-related risk indicators
2. Use probabilistic language: "suggests overreaching", "consistent with accumulated fatigue"
3. Always include a confidence score [0.0, 1.0]
4. Respond ONLY with valid JSON

REQUIRED JSON FORMAT:
{
  "fatigue_level": "low" | "moderate" | "high" | "critical",
  "load_assessment": "2-4 sentence evidence-based assessment",
  "overreaching_detected": true | false,
  "velocity_loss_concern": true | false,
  "weekly_progression_concern": true | false,
  "weekly_progression_rate_pct": float,
  "confidence": 0.0-1.0
}

Be specific and quantitative. Reference the actual numbers provided.
"""


def _build_human_prompt(state: AgentState) -> str:
    profile  = state.get("user_profile", {})
    logs     = state.get("workout_logs", [])
    settings = get_settings()

    # Summarize workout volume by week
    prog_rate = state.get("progression_rate", 0.0)
    prog_pct  = round(prog_rate * 100, 1)
    safe_pct  = settings.safe_weekly_progression_rate * 100
    warning_pct = settings.warning_weekly_progression_rate * 100

    # Recent session summaries (last 6 sessions of this exercise)
    exercise = state.get("lift_type", "").replace("_", " ")
    recent_sessions = [
        w for w in logs
        if exercise.replace(" ", "_") in w.get("exercise", "")
    ][-6:]

    biomech_flags = (state.get("biomech_output") or {}).get("joint_stress_flags", [])

    return f"""FATIGUE & LOAD ANALYSIS REQUEST

User: {profile.get('experience_level', 'unknown')} lifter | {profile.get('training_age_years', '?')} years training

Current Session:
  Lift:       {state.get('lift_type', '?').upper()}
  Load:       {state.get('weight_kg', 0)} kg
  RPE:        {state.get('rpe', 0)} / 10

Computed Fatigue Metrics:
  Session fatigue score:      {state.get('fatigue_score', 0):.3f}
  Weekly progression rate:    {prog_pct}%  (Safe: <{safe_pct}%  |  Warning: >{warning_pct}%)
  Velocity loss this week:    {round((state.get("progression_rate", 0)) * 10, 1)}%

Recent Training Sessions (last 6, this lift):
{json.dumps(recent_sessions[:6], indent=2, default=str)}

Biomechanics red flags already detected:
{biomech_flags}

Recovery context:
  Recovery score: {state.get('recovery_score', 0.5):.3f}

Assess fatigue and load accumulation risk. Return JSON only."""


def fatigue_agent_node(state: AgentState) -> dict:
    """LangGraph node: Fatigue & Load Agent."""
    settings  = get_settings()
    t_start   = time.perf_counter()
    trace_id  = state.get("trace_id", "")

    logger.info("fatigue_agent.start", trace_id=trace_id, fatigue_score=state.get("fatigue_score"))

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

    errors = list(state.get("errors", []))
    output_dict = {}

    try:
        response   = llm.invoke(messages)
        raw_output = JsonOutputParser().parse(response.content)

        # Ensure progression rate is from the engine (source of truth)
        raw_output["weekly_progression_rate_pct"] = round(state.get("progression_rate", 0.0) * 100, 1)

        validated   = FatigueAgentOutput(**raw_output)
        output_dict = validated.model_dump()

        logger.info(
            "fatigue_agent.success",
            trace_id=trace_id,
            fatigue_level=validated.fatigue_level,
            overreaching=validated.overreaching_detected,
            confidence=validated.confidence,
        )

    except ValidationError as e:
        logger.error("fatigue_agent.validation_error", trace_id=trace_id, error=str(e))
        errors.append(f"fatigue_agent validation: {e}")
        output_dict = _fallback_output(state)
    except Exception as e:
        logger.error("fatigue_agent.error", trace_id=trace_id, error=str(e))
        errors.append(f"fatigue_agent error: {str(e)}")
        output_dict = _fallback_output(state)

    duration_ms = (time.perf_counter() - t_start) * 1000
    trace_entry = {
        "agent_name":    "fatigue_agent",
        "input_summary": f"fatigue_score={state.get('fatigue_score'):.3f}, progression={round(state.get('progression_rate', 0) * 100, 1)}%",
        "output":        output_dict,
        "duration_ms":   round(duration_ms, 1),
        "confidence":    output_dict.get("confidence", 0.0),
    }

    return {
        "fatigue_output": output_dict,
        "agent_traces":   state.get("agent_traces", []) + [trace_entry],
        "errors":         errors,
    }


def _fallback_output(state: AgentState) -> dict:
    prog = state.get("progression_rate", 0.0)
    return {
        "fatigue_level":               "moderate",
        "load_assessment":             "Automated fatigue assessment unavailable. Manual review recommended.",
        "overreaching_detected":       prog > 0.15,
        "velocity_loss_concern":       False,
        "weekly_progression_concern":  prog > 0.10,
        "weekly_progression_rate_pct": round(prog * 100, 1),
        "confidence":                  0.10,
    }
