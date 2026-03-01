"""
schemas/agent_state.py
======================
LangGraph StateGraph state definition.
Flows read-left-to-right through: biomech_agent → fatigue_agent → risk_agent.
"""

from typing import Any
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # ── Inputs (populated before graph entry) ────────────────────────────────
    trace_id:        str
    user_id:         str
    lift_type:       str
    weight_kg:       float
    rpe:             float

    # User context
    user_profile:    dict        # UserProfile.model_dump()
    workout_logs:    list[dict]
    sleep_logs:      list[dict]
    nutrition_logs:  list[dict]

    # CV layer output
    biomechanics_data: dict      # Full JSON from the lift analyzer
    biomech_summary:   dict      # Condensed version sent to LLM (avoids token blowout)

    # Physics engine outputs
    stress_indices:    dict      # StressIndices.model_dump()
    fatigue_score:     float
    progression_rate:  float     # decimal (0.12 = 12%)
    recovery_score:    float

    # ── Agent outputs (populated by each node) ────────────────────────────────
    biomech_output:   dict | None   # BiomechAgentOutput.model_dump()
    fatigue_output:   dict | None   # FatigueAgentOutput.model_dump()
    risk_output:      dict | None   # RiskSynthesisOutput.model_dump()

    # ── Observability ─────────────────────────────────────────────────────────
    agent_traces:    list[dict]   # one entry per agent: name, input, output, timing
    errors:          list[str]    # non-fatal errors / warnings accumulated
