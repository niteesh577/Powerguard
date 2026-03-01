"""
data/models.py
==============
Pydantic models that define the shape of user data.
These are the **shared contract** between:
  - MockDataProvider (Phase 1)
  - SupabaseDataProvider (Phase 2 — swap when tables are ready)

Align these models exactly with your Supabase table schemas when you create them.
"""

from datetime import date
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ExperienceLevel(str, Enum):
    beginner     = "beginner"      # < 1 year
    intermediate = "intermediate"  # 1–3 years
    advanced     = "advanced"      # 3+ years
    elite        = "elite"         # competitive


class UserProfile(BaseModel):
    """Maps to: supabase table `users` / `user_profiles`."""
    user_id:               str
    name:                  str
    age:                   int = Field(..., ge=13, le=100)
    sex:                   str = Field(..., pattern="^(male|female|other)$")
    bodyweight_kg:         float = Field(..., ge=30.0, le=300.0)
    training_age_years:    float = Field(..., ge=0.0, le=50.0)
    experience_level:      ExperienceLevel
    injury_history:        list[str] = Field(default_factory=list)
    # Estimated maxes (for progression rate context)
    max_bench_kg:          Optional[float] = None
    max_deadlift_kg:       Optional[float] = None
    max_squat_kg:          Optional[float] = None


class WorkoutLog(BaseModel):
    """Maps to: supabase table `workout_logs`."""
    date:        date
    exercise:    str   = Field(..., description="e.g. 'bench_press', 'deadlift', 'squat'")
    weight_kg:   float = Field(..., ge=0.0)
    reps:        int   = Field(..., ge=1, le=100)
    sets:        int   = Field(..., ge=1, le=20)
    rpe:         float = Field(..., ge=1.0, le=10.0)
    # Bar velocity (m/s) — optional, from velocity tracker or wearable
    velocity_ms: Optional[float] = Field(default=None, ge=0.0, le=3.0)

    @property
    def volume_load(self) -> float:
        """Total volume load: weight × reps × sets."""
        return self.weight_kg * self.reps * self.sets


class NutritionLog(BaseModel):
    """Maps to: supabase table `nutrition_logs`."""
    date:      date
    calories:  float = Field(..., ge=0.0, le=10000.0)
    protein_g: float = Field(..., ge=0.0, le=500.0)
    carbs_g:   float = Field(..., ge=0.0, le=1000.0)
    fats_g:    float = Field(..., ge=0.0, le=300.0)


class SleepLog(BaseModel):
    """Maps to: supabase table `sleep_logs`."""
    date:           date
    hours:          float = Field(..., ge=0.0, le=24.0)
    quality_score:  float = Field(..., ge=0.0, le=1.0,
                                  description="Subjective or wearable quality 0–1")
    hrv:            Optional[float] = Field(default=None, ge=0.0,
                                            description="Heart rate variability (ms)")
    soreness_score: float = Field(default=5.0, ge=0.0, le=10.0,
                                  description="Muscle soreness 0 (none) – 10 (extreme)")
