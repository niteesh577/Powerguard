"""
engines/stress_engine.py
========================
Mechanical Stress Engine — converts biomechanics JSON + load into normalized joint stress indices.

All indices are normalized to [0, 1]:
  0.0 = minimal stress / excellent mechanics
  1.0 = maximum stress / critical mechanics

Physics formulas:
  Bench  → shoulder_stress ∝ load × sin(elbow_flare_rad) × grip_width_ratio
  DL     → lumbar_stress   ∝ load × bar_horizontal_drift
  Squat  → knee_stress     ∝ load × sin(knee_angle_rad) × (1 + valgus_factor)
"""

import math
from dataclasses import dataclass

from schemas.response import StressIndices


@dataclass
class RawBiomechanics:
    """Condensed metrics extracted from the full analyzer JSON."""
    # Bench press
    avg_elbow_flare_deg: float = 0.0
    avg_grip_width_ratio: float = 1.8
    avg_elbow_angle_symmetry: float = 1.0
    avg_wrist_symmetry: float = 1.0

    # Deadlift
    bar_path_x_std_px: float = 0.0
    avg_back_angle_deg: float = 15.0
    avg_hip_symmetry: float = 1.0

    # Squat
    avg_bottom_knee_angle_deg: float = 90.0
    avg_max_left_valgus_px: float = 0.0
    avg_max_right_valgus_px: float = 0.0
    avg_knee_symmetry: float = 1.0
    below_parallel_pct: float = 0.5
    avg_torso_lean_deg: float = 20.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _symmetry_penalty(symmetry_score: float) -> float:
    """Convert symmetry [0,1] to a stress multiplier: asymmetry amplifies stress."""
    return 1.0 + (1.0 - symmetry_score) * 0.5  # max 1.5× at zero symmetry


# ── Lift-specific stress calculators ────────────────────────────────────────

def compute_bench_stress(load_kg: float, bio: RawBiomechanics) -> StressIndices:
    """
    Shoulder stress:
      raw = load_kg × sin(elbow_flare_rad) × grip_width_ratio
    Normalization constant 250 chosen so that:
      100 kg × sin(60°) × 2.5 ≈ 216 → index ≈ 0.86 (high)
    """
    flare_rad = math.radians(max(bio.avg_elbow_flare_deg, 1.0))
    primary   = _clamp(load_kg * math.sin(flare_rad) * bio.avg_grip_width_ratio / 250.0)

    # Secondary: wrist asymmetry under load
    secondary = _clamp(load_kg * (1.0 - bio.avg_wrist_symmetry) / 100.0)

    # Symmetry stress: overall technique asymmetry
    sym_stress = _clamp(1.0 - bio.avg_elbow_angle_symmetry)

    overall = _clamp(0.55 * primary + 0.25 * secondary + 0.20 * sym_stress)

    label = _stress_label(overall)
    return StressIndices(
        primary_joint_stress=primary,
        secondary_joint_stress=secondary,
        symmetry_stress=sym_stress,
        overall_stress=overall,
        stress_label=label,
    )


def compute_deadlift_stress(load_kg: float, bio: RawBiomechanics) -> StressIndices:
    """
    Lumbar shear stress:
      raw = load_kg × (bar_horizontal_drift / frame_height_proxy)
    A well-performed DL has bar drift std < 20 px; dangerous > 80 px.
    Normalization: 180 kg × 0.6 = 108 → index ≈ 1.0
    """
    drift_factor  = _clamp(bio.bar_path_x_std_px / 80.0)
    primary       = _clamp(load_kg * drift_factor / 110.0)

    # Back angle: more forward lean = more lumbar stress
    back_factor   = _clamp(bio.avg_back_angle_deg / 60.0)
    secondary     = _clamp(load_kg * back_factor / 180.0 * 0.5)

    sym_stress    = _clamp(1.0 - bio.avg_hip_symmetry)
    overall       = _clamp(0.60 * primary + 0.25 * secondary + 0.15 * sym_stress)

    return StressIndices(
        primary_joint_stress=primary,
        secondary_joint_stress=secondary,
        symmetry_stress=sym_stress,
        overall_stress=overall,
        stress_label=_stress_label(overall),
    )


def compute_squat_stress(load_kg: float, bio: RawBiomechanics) -> StressIndices:
    """
    Knee stress:
      raw = load_kg × sin(knee_angle_rad) × valgus_factor
    Lower knee angle = more knee flexion = more stress.
    Valgus amplifies stress.
    Normalization: 150 kg × sin(70°) × 1.3 ≈ 183 → index close to 1.0
    """
    knee_rad     = math.radians(max(bio.avg_bottom_knee_angle_deg, 10.0))
    avg_valgus   = (abs(bio.avg_max_left_valgus_px) + abs(bio.avg_max_right_valgus_px)) / 2
    valgus_factor = 1.0 + _clamp(avg_valgus / 60.0) * 0.5   # max 1.5×

    primary      = _clamp(load_kg * math.sin(knee_rad) * valgus_factor / 200.0)

    # Torso lean: anterior shear on lumbar
    lean_factor  = _clamp(bio.avg_torso_lean_deg / 45.0)
    secondary    = _clamp(load_kg * lean_factor / 200.0 * 0.5)

    sym_stress   = _clamp(1.0 - bio.avg_knee_symmetry)
    overall      = _clamp(0.55 * primary + 0.25 * secondary + 0.20 * sym_stress)

    return StressIndices(
        primary_joint_stress=primary,
        secondary_joint_stress=secondary,
        symmetry_stress=sym_stress,
        overall_stress=overall,
        stress_label=_stress_label(overall),
    )


def _stress_label(index: float) -> str:
    if index < 0.25: return "minimal"
    if index < 0.50: return "low"
    if index < 0.70: return "moderate"
    if index < 0.85: return "high"
    return "critical"


# ── Main dispatcher ──────────────────────────────────────────────────────────

def compute_stress_indices(lift_type: str, load_kg: float, biomech_summary: dict) -> StressIndices:
    """
    Entry point called by the API layer.
    `biomech_summary` is a condensed dict extracted from the full analyzer JSON.
    """
    bio = _extract_biomechanics(lift_type, biomech_summary)

    if lift_type == "bench_press":
        return compute_bench_stress(load_kg, bio)
    elif lift_type == "deadlift":
        return compute_deadlift_stress(load_kg, bio)
    elif lift_type == "squat":
        return compute_squat_stress(load_kg, bio)
    else:
        raise ValueError(f"Unknown lift_type: {lift_type!r}")


def _extract_biomechanics(lift_type: str, summary: dict) -> RawBiomechanics:
    """Map the analyzer JSON summary onto RawBiomechanics fields."""
    b = RawBiomechanics()
    os = summary.get("overall_summary", {})

    if lift_type == "bench_press":
        b.avg_elbow_flare_deg       = os.get("avg_bottom_elbow_flare") or 45.0
        b.avg_grip_width_ratio      = os.get("avg_grip_width_ratio")  or 1.8
        b.avg_elbow_angle_symmetry  = os.get("avg_elbow_angle_symmetry") or 0.90
        b.avg_wrist_symmetry        = os.get("avg_wrist_symmetry") or 0.92

    elif lift_type == "deadlift":
        b.bar_path_x_std_px    = os.get("avg_bar_path_x_std_px") or 20.0
        b.avg_back_angle_deg   = os.get("avg_back_angle") or 15.0
        b.avg_hip_symmetry     = os.get("avg_hip_symmetry") or 0.90

    elif lift_type == "squat":
        b.avg_bottom_knee_angle_deg  = os.get("avg_bottom_left_knee_angle") or 85.0
        b.avg_max_left_valgus_px     = os.get("avg_max_left_knee_valgus_px") or 0.0
        b.avg_max_right_valgus_px    = os.get("avg_max_right_knee_valgus_px") or 0.0
        b.avg_knee_symmetry          = os.get("avg_knee_symmetry") or 0.90
        b.avg_torso_lean_deg         = os.get("avg_bottom_torso_lean") or 20.0

    return b


def summarize_biomechanics(lift_type: str, full_json: dict) -> dict:
    """
    Condense the full analyzer JSON to a small dict for LLM prompts.
    Avoids sending thousands of frame_data rows to the LLM.
    """
    os_ = full_json.get("overall_summary", {})
    reps = full_json.get("reps", [])

    summary = {
        "lift_type":     lift_type,
        "total_reps":    os_.get("total_reps", 0),
        "video_duration_s": os_.get("video_duration_s", 0),
        "overall_summary": os_,
    }

    # Include per-rep summary (without raw bar_path arrays — too large)
    rep_summaries = []
    for r in reps:
        rep_summaries.append({k: v for k, v in r.items() if k != "bar_path" and k != "hip_path"})
    summary["reps"] = rep_summaries

    return summary
