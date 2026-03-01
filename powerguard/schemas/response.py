"""
schemas/response.py
===================
Validated output models for the PowerGuard API.
All agent outputs are validated here before returning to the client.
Guardrails are enforced at the model level.
"""

from enum import Enum
from typing import Annotated
from pydantic import BaseModel, Field, field_validator


class RiskLevel(str, Enum):
    LOW      = "LOW"
    MODERATE = "MODERATE"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


# ── Per-agent outputs ────────────────────────────────────────────────────────

class StressIndices(BaseModel):
    """Physics-derived mechanical stress indices (all normalized 0–1)."""
    primary_joint_stress:   float = Field(..., ge=0.0, le=1.0)
    secondary_joint_stress: float = Field(..., ge=0.0, le=1.0)
    symmetry_stress:        float = Field(..., ge=0.0, le=1.0)
    overall_stress:         float = Field(..., ge=0.0, le=1.0)
    stress_label:           str   = Field(..., description="Human-readable stress label")


class BiomechAgentOutput(BaseModel):
    """Output from the Biomechanics Agent."""
    joint_stress_flags:   list[str] = Field(default_factory=list, max_length=8)
    technique_summary:    str       = Field(..., min_length=10, max_length=500)
    primary_risk_joint:   str       = Field(..., description="Joint with highest stress")
    risk_signals:         list[str] = Field(default_factory=list, max_length=6)
    confidence:           Annotated[float, Field(ge=0.0, le=1.0)]

    @field_validator("technique_summary")
    @classmethod
    def no_diagnostic_language(cls, v: str) -> str:
        banned = ["diagnosed", "injury", "torn", "ruptured", "fractured", "herniated"]
        lower = v.lower()
        for word in banned:
            if word in lower:
                raise ValueError(
                    f"Guardrail: diagnostic language detected ('{word}'). "
                    "Use probabilistic phrasing instead."
                )
        return v


class FatigueAgentOutput(BaseModel):
    """Output from the Fatigue & Load Agent."""
    fatigue_level:               str   = Field(..., pattern="^(low|moderate|high|critical)$")
    load_assessment:             str   = Field(..., min_length=10, max_length=400)
    overreaching_detected:       bool
    velocity_loss_concern:       bool
    weekly_progression_concern:  bool
    weekly_progression_rate_pct: float = Field(..., description="Actual weekly load change %")
    confidence:                  Annotated[float, Field(ge=0.0, le=1.0)]


class Recommendation(BaseModel):
    category:    str = Field(..., description="E.g. 'Technique', 'Load', 'Recovery'")
    action:      str = Field(..., min_length=10, max_length=300)
    priority:    int = Field(..., ge=1, le=5, description="1=highest priority")
    evidence_basis: str = Field(default="", description="Scientific grounding if available")


class RiskSynthesisOutput(BaseModel):
    """Final output from the Risk Synthesis Agent — returned to the client."""
    risk_level:             RiskLevel
    risk_score:             Annotated[float, Field(ge=0.0, le=1.0)]
    injury_probability_pct: Annotated[float, Field(ge=0.0, le=100.0)]
    primary_concern:        str       = Field(..., min_length=10, max_length=300)
    recommendations:        list[Recommendation] = Field(..., min_length=1, max_length=5)
    confidence:             Annotated[float, Field(ge=0.0, le=1.0)]
    guardrail_disclaimer:   str       = Field(
        default=(
            "This report contains risk indicators only and is NOT a medical diagnosis. "
            "Consult a qualified sports medicine professional for any health concerns."
        )
    )


# ── Full API response ─────────────────────────────────────────────────────────

class AgentTrace(BaseModel):
    agent_name:    str
    input_summary: str
    output:        dict
    duration_ms:   float
    confidence:    float


class AnalysisResponse(BaseModel):
    """Complete response returned by POST /v1/analyze."""
    trace_id:        str
    user_id:         str
    lift_type:       str
    weight_kg:       float

    # Computed intermediates
    stress_indices:  StressIndices
    fatigue_score:   float = Field(..., ge=0.0, le=1.0)
    recovery_score:  float = Field(..., ge=0.0, le=1.0)
    progression_rate_pct: float

    # Final risk result
    risk_result:     RiskSynthesisOutput

    # Observability: full agent trace (for debugging / audit)
    agent_traces:    list[AgentTrace]

    processing_time_ms: float
