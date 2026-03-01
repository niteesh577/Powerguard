"""
schemas/request.py
==================
Validated input models for the PowerGuard API.
"""

from enum import Enum
from pydantic import BaseModel, Field, field_validator


class LiftType(str, Enum):
    bench_press = "bench_press"
    deadlift    = "deadlift"
    squat       = "squat"


class AnalysisRequest(BaseModel):
    """
    Everything the API needs to run a full injury-risk analysis.
    The video file is received as a separate multipart upload.
    """
    user_id:   str        = Field(..., min_length=1, max_length=128, description="Unique user identifier")
    lift_type: LiftType   = Field(..., description="Type of lift performed in the video")
    weight_kg: float      = Field(..., ge=1.0, le=600.0, description="Weight used in kg")
    rpe:       float      = Field(..., ge=1.0, le=10.0,  description="Rate of perceived exertion (1–10)")

    @field_validator("weight_kg")
    @classmethod
    def weight_must_be_realistic(cls, v: float) -> float:
        if v > 500:
            raise ValueError("Weight > 500 kg is unrealistic. Please verify the input.")
        return round(v, 1)

    @field_validator("rpe")
    @classmethod
    def rpe_half_points_only(cls, v: float) -> float:
        # RPE is typically recorded in 0.5 increments
        return round(v * 2) / 2
