"""
engines/recovery_engine.py
==========================
Capacity / Recovery Engine.

Recovery score [0, 1]:
  1.0 = fully recovered, high tissue capacity
  0.0 = severely under-recovered

Formula:
  recovery = 0.35 * sleep_score  +  0.25 * hrv_score
           + 0.20 * protein_score +  0.20 * soreness_score

Tissue capacity = f(experience_level) × recovery_score
"""

from statistics import mean, stdev

from data.models import ExperienceLevel, NutritionLog, SleepLog, UserProfile


# Target protein: 2.0 g/kg bodyweight (evidence-based for strength athletes)
PROTEIN_TARGET_G_PER_KG = 2.0

# HRV: we compare latest reading to the 7-day rolling mean
HRV_WINDOW = 7


def _compute_sleep_score(sleep_logs: list[SleepLog]) -> float:
    """
    Blend of sleep duration (target 8 h) and subjective quality.
    Returns [0, 1].
    """
    if not sleep_logs:
        return 0.60  # unknown = assume moderate
    avg_hours   = mean(s.hours for s in sleep_logs)
    avg_quality = mean(s.quality_score for s in sleep_logs)
    duration_score = min(avg_hours / 8.0, 1.0)
    return 0.55 * duration_score + 0.45 * avg_quality


def _compute_hrv_score(sleep_logs: list[SleepLog]) -> float:
    """
    HRV ratio = today's HRV / rolling mean HRV.
    Ratio > 1.0 = well-recovered; < 0.85 = suppressed.
    Returns [0, 1].
    """
    hrv_values = [s.hrv for s in sleep_logs if s.hrv is not None]
    if len(hrv_values) < 2:
        return 0.70  # unknown → assume moderate

    rolling_mean = mean(hrv_values[:-1])
    current_hrv  = hrv_values[-1]
    if rolling_mean <= 0:
        return 0.70

    ratio = current_hrv / rolling_mean
    # Map ratio [0.7 – 1.3] → [0, 1]
    score = (ratio - 0.70) / 0.60
    return min(1.0, max(0.0, score))


def _compute_protein_score(nutrition_logs: list[NutritionLog], bodyweight_kg: float) -> float:
    """
    Protein adequacy = actual / target  (target = 2.0 g/kg).
    Capped at 1.0 even if intake exceeds target.
    """
    if not nutrition_logs:
        return 0.65
    target = bodyweight_kg * PROTEIN_TARGET_G_PER_KG
    avg_protein = mean(n.protein_g for n in nutrition_logs)
    return min(1.0, avg_protein / target)


def _compute_soreness_score(sleep_logs: list[SleepLog]) -> float:
    """
    Inverse of soreness (higher soreness = lower recovery capacity).
    Soreness 0 = 1.0 score; Soreness 10 = 0.0 score.
    """
    if not sleep_logs:
        return 0.65
    avg_soreness = mean(s.soreness_score for s in sleep_logs)
    return max(0.0, 1.0 - avg_soreness / 10.0)


def compute_recovery_score(
    sleep_logs: list[SleepLog],
    nutrition_logs: list[NutritionLog],
    bodyweight_kg: float,
) -> dict:
    """
    Full recovery computation returning both the composite score and its components.
    """
    sleep_score    = _compute_sleep_score(sleep_logs)
    hrv_score      = _compute_hrv_score(sleep_logs)
    protein_score  = _compute_protein_score(nutrition_logs, bodyweight_kg)
    soreness_score = _compute_soreness_score(sleep_logs)

    composite = (
        0.35 * sleep_score   +
        0.25 * hrv_score     +
        0.20 * protein_score +
        0.20 * soreness_score
    )
    composite = max(0.01, min(1.0, composite))  # floor at 0.01 to avoid division by zero in risk eq

    return {
        "recovery_score":   round(composite, 3),
        "sleep_score":      round(sleep_score, 3),
        "hrv_score":        round(hrv_score, 3),
        "protein_score":    round(protein_score, 3),
        "soreness_score":   round(soreness_score, 3),
        "recovery_label":   _recovery_label(composite),
    }


def _recovery_label(score: float) -> str:
    if score >= 0.80: return "excellent"
    if score >= 0.65: return "good"
    if score >= 0.50: return "moderate"
    if score >= 0.35: return "poor"
    return "critical"


def compute_tissue_capacity_multiplier(profile: UserProfile) -> float:
    """
    Personalization: advanced athletes have higher tissue capacity → lower effective risk.
    Beginner: 0.75 (lower capacity, more vulnerable)
    Intermediate: 1.00
    Advanced: 1.25
    Elite: 1.40
    """
    multipliers = {
        ExperienceLevel.beginner:     0.75,
        ExperienceLevel.intermediate: 1.00,
        ExperienceLevel.advanced:     1.25,
        ExperienceLevel.elite:        1.40,
    }
    return multipliers.get(profile.experience_level, 1.00)
