"""
engines/fatigue_engine.py
=========================
Fatigue Accumulation Engine.

Computes:
  1. Session fatigue score   (RPE + relative load + velocity loss)
  2. 7-day rolling fatigue   (volume load accumulation)
  3. Weekly progression rate (load ramp — the #1 injury predictor)
"""

from collections import defaultdict
from datetime import timedelta
from statistics import mean

from data.models import WorkoutLog


def compute_velocity_loss(logs: list[WorkoutLog], exercise: str) -> float:
    """
    Velocity loss % = (V_first_set_week – V_last_set_week) / V_first_set_week
    Higher loss = more neuromuscular fatigue.
    Returns 0.0 if velocity data is unavailable.
    """
    exercise_logs = [w for w in logs if w.exercise == exercise and w.velocity_ms is not None]
    if len(exercise_logs) < 2:
        return 0.0
    first_vel = exercise_logs[0].velocity_ms
    last_vel  = exercise_logs[-1].velocity_ms
    if first_vel <= 0:
        return 0.0
    return max(0.0, (first_vel - last_vel) / first_vel)


def compute_session_fatigue(
    logs: list[WorkoutLog],
    exercise: str,
    current_rpe: float,
    current_weight_kg: float,
    current_reps: int = 1,
) -> float:
    """
    Session fatigue score [0, 1].

    Components:
      RPE factor          = (RPE - 5) / 5         (RPE 5 → 0, RPE 10 → 1)
      Velocity loss       = % velocity dropped this week
      Relative intensity  = current_weight / estimated_1RM
    """
    # RPE factor
    rpe_factor = max(0.0, (current_rpe - 5.0) / 5.0)

    # Velocity loss
    vel_loss = compute_velocity_loss(logs, exercise)

    # Relative intensity (Epley 1RM estimate from most recent session)
    exercise_logs = [w for w in logs if w.exercise == exercise]
    relative_intensity = 0.5
    if exercise_logs:
        recent = exercise_logs[-1]
        est_1rm = recent.weight_kg * (1 + recent.reps / 30.0)
        relative_intensity = min(current_weight_kg / est_1rm, 1.0) if est_1rm > 0 else 0.5

    session_fatigue = (
        0.40 * rpe_factor +
        0.35 * vel_loss +
        0.25 * relative_intensity
    )
    return min(1.0, max(0.0, session_fatigue))


def compute_rolling_fatigue(logs: list[WorkoutLog], days: int = 7) -> float:
    """
    7-day accumulated volume load normalized to [0, 1].
    Normalization constant: 15,000 kg (heavy week for an intermediate lifter).
    """
    if not logs:
        return 0.0
    cutoff = logs[-1].date - timedelta(days=days - 1)
    recent = [w for w in logs if w.date >= cutoff]
    total_load = sum(w.volume_load for w in recent)
    return min(1.0, total_load / 15_000.0)


def compute_weekly_progression_rate(logs: list[WorkoutLog], exercise: str) -> float:
    """
    Weekly load increase % = (current_week_load - prev_week_load) / prev_week_load

    Returns 0.0 if insufficient history.
    Safe zone: < 0.10 (10%)
    Warning:   0.10 – 0.15 (10–15%)
    Red flag:  > 0.15 (>15%)
    """
    exercise_logs = [w for w in logs if w.exercise == exercise]
    if len(exercise_logs) < 2:
        return 0.0

    # Group by ISO week
    by_week: dict[str, float] = defaultdict(float)
    for w in exercise_logs:
        week_key = w.date.strftime("%G-%V")   # ISO year + week
        by_week[week_key] += w.volume_load

    weeks = sorted(by_week.keys())
    if len(weeks) < 2:
        return 0.0

    current_load  = by_week[weeks[-1]]
    previous_load = by_week[weeks[-2]]
    if previous_load <= 0:
        return 0.0

    return (current_load - previous_load) / previous_load


def get_fatigue_label(score: float) -> str:
    if score < 0.30: return "low"
    if score < 0.55: return "moderate"
    if score < 0.75: return "high"
    return "critical"


def compute_all_fatigue_metrics(
    logs: list[WorkoutLog],
    exercise: str,
    current_rpe: float,
    current_weight_kg: float,
) -> dict:
    """
    Convenience function — returns all fatigue metrics as a single dict
    for injection into the LangGraph state.
    """
    session_fatigue   = compute_session_fatigue(logs, exercise, current_rpe, current_weight_kg)
    rolling_fatigue   = compute_rolling_fatigue(logs)
    progression_rate  = compute_weekly_progression_rate(logs, exercise)
    velocity_loss_pct = compute_velocity_loss(logs, exercise) * 100

    # Combined fatigue score: weight session + rolling equally
    combined = 0.60 * session_fatigue + 0.40 * rolling_fatigue

    return {
        "session_fatigue_score":    round(session_fatigue, 3),
        "rolling_7d_fatigue_score": round(rolling_fatigue, 3),
        "combined_fatigue_score":   round(combined, 3),
        "fatigue_label":            get_fatigue_label(combined),
        "weekly_progression_rate":  round(progression_rate, 4),
        "velocity_loss_pct":        round(velocity_loss_pct, 1),
        "overreaching_flag":        progression_rate > 0.15 or combined > 0.75,
    }
