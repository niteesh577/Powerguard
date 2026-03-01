"""
deadlift_analyzer.py
====================
Comprehensive deadlift biomechanics data collector.

Usage
-----
    python deadlift_analyzer.py

Edit INPUT_VIDEO at the bottom to point to your file.

Metrics collected
-----------------
Per frame:
  - Left / right hip hinge angle       (shoulder → hip → knee)
  - Left / right knee angle             (hip → knee → ankle)
  - Back (torso) angle from vertical    (lower = more upright)
  - Shoulder position offset from bar   (px; negative = behind bar)
  - Bar velocity (px/s)
  - Symmetry: hip angle, knee angle
  - Phase: idle / pull / lower

Per rep:
  - Pull time, lower time, total duration
  - Range of motion (px)
  - Back angle at pull start (setup) → lockout → re-set
  - Hip / knee angles at lockout (should be fully extended ≈ 180°)
  - Knee angle at start of pull (how much knee bend at setup)
  - Bar path coordinates + horizontal drift (x std dev)
  - Sticking point (frame of minimum bar velocity during pull)
  - Average symmetry scores

Outputs
-------
  VIDEOS/OUTPUTS/<name>_analysis.mp4  – annotated video
  VIDEOS/OUTPUTS/<name>_analysis.json – structured report for agent
"""

import cv2 as cv
import json
import math
import os
import numpy as np

from Pose_estimationModule import PoseDetector
from biomechanics_metrics import (
    calculate_angle,
    calculate_velocity,
    symmetry_score,
    DeadliftRepTracker,
)
from utils import draw_text_with_bg

INPUT_VIDEO = "VIDEOS/INPUTS/deadlift.mp4"

# ── Landmark indices ──────────────────────────────────────────────────────────
_IDX = {
    "left_shoulder":  11, "right_shoulder": 12,
    "left_elbow":     13, "right_elbow":    14,
    "left_wrist":     15, "right_wrist":    16,
    "left_hip":       23, "right_hip":      24,
    "left_knee":      25, "right_knee":     26,
    "left_ankle":     27, "right_ankle":    28,
}

_PHASE_COLOR = {
    "idle":  (160, 160, 160),
    "pull":  (0,   220,  60),   # green – concentric effort
    "lower": (0,   100, 255),   # orange – eccentric
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _g(lm, key):
    return lm.get(_IDX[key])

def _mid(a, b):
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)

def _back_angle(shoulder_mid, hip_mid) -> float | None:
    """Angle of the torso vector from vertical (0° = perfectly upright)."""
    if shoulder_mid is None or hip_mid is None:
        return None
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]   # negative = shoulder above hip
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None
    # Vertical reference points upward in screen coords → (0, -1)
    cos_a = np.clip(-dy / length, -1.0, 1.0)
    return round(math.degrees(math.acos(cos_a)), 1)


# ── Per-frame angle computation ───────────────────────────────────────────────

def compute_angles(lm: dict) -> dict:
    angles = {}

    ls, rs = _g(lm, "left_shoulder"),  _g(lm, "right_shoulder")
    le, re = _g(lm, "left_elbow"),     _g(lm, "right_elbow")
    lw, rw = _g(lm, "left_wrist"),     _g(lm, "right_wrist")
    lh, rh = _g(lm, "left_hip"),       _g(lm, "right_hip")
    lk, rk = _g(lm, "left_knee"),      _g(lm, "right_knee")
    la, ra = _g(lm, "left_ankle"),     _g(lm, "right_ankle")

    # Hip hinge angle – shoulder → hip → knee (how much the torso is hinged)
    if ls and lh and lk:
        angles["left_hip_hinge"]  = calculate_angle(ls, lh, lk)
    if rs and rh and rk:
        angles["right_hip_hinge"] = calculate_angle(rs, rh, rk)

    # Knee angle – hip → knee → ankle
    if lh and lk and la:
        angles["left_knee"]  = calculate_angle(lh, lk, la)
    if rh and rk and ra:
        angles["right_knee"] = calculate_angle(rh, rk, ra)

    # Arm angle (shoulder → elbow → wrist) – checks for locked-out arms
    if ls and le and lw:
        angles["left_arm"]  = calculate_angle(ls, le, lw)
    if rs and re and rw:
        angles["right_arm"] = calculate_angle(rs, re, rw)

    # Back angle from vertical
    shoulder_mid = _mid(ls, rs)
    hip_mid      = _mid(lh, rh)
    ba = _back_angle(shoulder_mid, hip_mid)
    if ba is not None:
        angles["back_angle"] = ba

    # Shoulder-bar horizontal offset (negative = shoulders behind bar = good)
    bar_mid = _mid(lw, rw)
    if shoulder_mid and bar_mid:
        angles["shoulder_bar_offset_px"] = round(shoulder_mid[0] - bar_mid[0], 1)

    # Hip height relative to knee (helps detect setup position)
    knee_mid = _mid(lk, rk)
    if hip_mid and knee_mid:
        # Positive → hip above knee; negative → hip below knee
        angles["hip_above_knee_px"] = round(knee_mid[1] - hip_mid[1], 1)

    # Symmetry
    if "left_hip_hinge" in angles and "right_hip_hinge" in angles:
        angles["hip_symmetry"]  = symmetry_score(
            angles["left_hip_hinge"], angles["right_hip_hinge"])
    if "left_knee" in angles and "right_knee" in angles:
        angles["knee_symmetry"] = symmetry_score(
            angles["left_knee"], angles["right_knee"])
    if "left_arm" in angles and "right_arm" in angles:
        angles["arm_symmetry"]  = symmetry_score(
            angles["left_arm"], angles["right_arm"])

    return angles


# ── Overlay ───────────────────────────────────────────────────────────────────

def _draw_panel(frame, phase, rep_count, bar_vel, angles, panel_w=270):
    overlay = frame.copy()
    cv.rectangle(overlay, (0, 0), (panel_w, frame.shape[0]), (15, 15, 15), cv.FILLED)
    cv.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    color = _PHASE_COLOR.get(phase, (200, 200, 200))

    def put(text, y, col=(220, 220, 220), scale=0.52, bold=False):
        cv.putText(frame, text, (12, y),
                   cv.FONT_HERSHEY_SIMPLEX, scale, col,
                   2 if bold else 1, cv.LINE_AA)

    put(f"REPS: {rep_count}",      30, col=(0, 255, 180), scale=0.70, bold=True)
    put(f"PHASE: {phase.upper()}", 60, col=color, scale=0.58, bold=True)
    put(f"BAR VEL: {bar_vel:>6.0f} px/s", 88, col=(200, 180, 255))

    cv.line(frame, (12, 98), (panel_w - 12, 98), (60, 60, 60), 1)

    rows = [
        ("L Hip Hinge",  angles.get("left_hip_hinge",  "--"), "°"),
        ("R Hip Hinge",  angles.get("right_hip_hinge", "--"), "°"),
        ("L Knee",       angles.get("left_knee",        "--"), "°"),
        ("R Knee",       angles.get("right_knee",       "--"), "°"),
        ("L Arm",        angles.get("left_arm",         "--"), "°"),
        ("R Arm",        angles.get("right_arm",        "--"), "°"),
        ("Back Angle",   angles.get("back_angle",       "--"), "°"),
        ("Shld-Bar Δx",  angles.get("shoulder_bar_offset_px", "--"), " px"),
        ("Hip Sym",      angles.get("hip_symmetry",     "--"), ""),
        ("Knee Sym",     angles.get("knee_symmetry",    "--"), ""),
    ]
    y = 120
    for label, val, unit in rows:
        v = f"{val:.1f}{unit}" if isinstance(val, float) else f"{val}{unit}"
        put(f"{label:<12}: {v:>8}", y)
        y += 26


def _draw_angle_label(frame, pt, value, color=(255, 255, 0)):
    if pt is None or value is None:
        return
    x, y = pt
    cv.putText(frame, f"{value:.0f}",
               (x + 8, y - 8), cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv.LINE_AA)


def _draw_bar_path(frame, path, color=(80, 200, 100)):
    for i in range(1, len(path)):
        cv.line(frame, tuple(path[i - 1]), tuple(path[i]), color, 2, cv.LINE_AA)


# ── Per-rep derived metrics ───────────────────────────────────────────────────

def _derive_rep_metrics(r: dict, fps: float) -> dict:
    fa = [a for a in r.get("frame_angles", []) if a]
    bp = r.get("bar_path", [])

    entry = {
        "rep_number":           r["rep_number"],
        "start_frame":          r["start_frame"],
        "end_frame":            r.get("end_frame"),
        "lockout_frame":        r.get("lockout_frame"),
        "duration_s":           r.get("duration_s"),
        "pull_time_s":          r.get("pull_time_s"),
        "lower_time_s":         r.get("lower_time_s"),
        "range_of_motion_px":   r.get("range_of_motion_px"),
        "bar_path":             bp,
    }

    if not fa:
        return entry

    ys = [p[1] for p in bp] if bp else []

    # Setup angles (first 10% of frames = start of pull)
    n_setup = max(1, len(fa) // 10)
    setup_fa = fa[:n_setup]
    if setup_fa:
        entry["setup_back_angle"]   = round(np.mean([a.get("back_angle", 0) for a in setup_fa if a.get("back_angle")]), 1)
        entry["setup_left_knee"]    = round(np.mean([a.get("left_knee",  0) for a in setup_fa if a.get("left_knee")]),  1)
        entry["setup_right_knee"]   = round(np.mean([a.get("right_knee", 0) for a in setup_fa if a.get("right_knee")]), 1)

    # Lockout angles (frames near lockout_frame)
    if r.get("lockout_frame") and bp:
        lock_off = r["lockout_frame"] - r["start_frame"]
        win = slice(max(0, lock_off - 5), min(len(fa), lock_off + 5))
        lock_fa = fa[win]
        if lock_fa:
            entry["lockout_back_angle"]        = round(np.mean([a.get("back_angle", 0) for a in lock_fa if a.get("back_angle")]), 1)
            entry["lockout_left_hip_hinge"]    = round(np.mean([a.get("left_hip_hinge",  0) for a in lock_fa if a.get("left_hip_hinge")]),  1)
            entry["lockout_right_hip_hinge"]   = round(np.mean([a.get("right_hip_hinge", 0) for a in lock_fa if a.get("right_hip_hinge")]), 1)
            entry["lockout_left_knee"]         = round(np.mean([a.get("left_knee",  0) for a in lock_fa if a.get("left_knee")]),  1)
            entry["lockout_right_knee"]        = round(np.mean([a.get("right_knee", 0) for a in lock_fa if a.get("right_knee")]), 1)

    # Sticking point – minimum bar velocity during pull (y decreasing)
    if ys and r.get("lockout_frame"):
        lock_off = r["lockout_frame"] - r["start_frame"]
        pull_ys  = ys[:lock_off]
        if len(pull_ys) > 3:
            vels = [abs(pull_ys[i] - pull_ys[i - 1]) for i in range(1, len(pull_ys))]
            stick_off = int(np.argmin(vels))
            entry["sticking_point_rep_frame_offset"] = stick_off
            entry["sticking_point_time_s"]           = round((r["start_frame"] + stick_off) / fps, 2)

    # Bar path linearity (x std dev – lower = more vertical)
    if len(bp) > 2:
        xs = [p[0] for p in bp]
        entry["bar_path_x_std_px"]  = round(float(np.std(xs)), 1)
        entry["bar_path_y_range_px"] = round(float(max(ys) - min(ys)), 1) if ys else None

    # Averages across all frames
    def _avg(key):
        vals = [a.get(key) for a in fa if a.get(key) is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    entry["avg_back_angle"]        = _avg("back_angle")
    entry["avg_hip_symmetry"]      = _avg("hip_symmetry")
    entry["avg_knee_symmetry"]     = _avg("knee_symmetry")
    entry["avg_arm_symmetry"]      = _avg("arm_symmetry")
    entry["avg_shoulder_bar_offset_px"] = _avg("shoulder_bar_offset_px")

    return entry


# ── Main pipeline ─────────────────────────────────────────────────────────────

def analyze(video_path: str = INPUT_VIDEO) -> dict:
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps   = cap.get(cv.CAP_PROP_FPS) or 30.0
    w     = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    base     = os.path.splitext(os.path.basename(video_path))[0]
    out_dir  = os.path.join(os.path.dirname(video_path), "..", "OUTPUTS")
    os.makedirs(out_dir, exist_ok=True)
    vid_out  = os.path.join(out_dir, f"{base}_analysis.mp4")
    json_out = os.path.join(out_dir, f"{base}_analysis.json")

    writer   = cv.VideoWriter(vid_out, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    detector = PoseDetector()
    tracker  = DeadliftRepTracker(fps=fps)

    frame_data = []
    prev_bar   = None
    frame_idx  = 0

    print(f"\n[INFO] Analyzing (Deadlift): {video_path}")
    print(f"       {total} frames @ {fps:.1f} fps  ({w}×{h})\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        ts = round(frame_idx / fps, 3)

        frame = detector.find_pose(frame, draw=True)
        lm    = detector.get_positions(frame)

        if not lm:
            writer.write(frame)
            frame_data.append({"frame": frame_idx, "timestamp_s": ts,
                                "phase": "no_detection", "angles": {}, "bar_pt": None})
            continue

        angles  = compute_angles(lm)
        bar_pt  = _mid(_g(lm, "left_wrist"), _g(lm, "right_wrist"))
        bar_vel = calculate_velocity(prev_bar, bar_pt, fps) if prev_bar else 0.0
        prev_bar = bar_pt

        phase = tracker.update(bar_pt[1] if bar_pt else 0.0)

        if tracker._current_rep is not None:
            tracker._current_rep["frame_angles"].append(angles)
            if bar_pt:
                tracker._current_rep["bar_path"].append(list(bar_pt))

        frame_data.append({
            "frame":        frame_idx,
            "timestamp_s":  ts,
            "phase":        phase,
            "bar_pt":       list(bar_pt) if bar_pt else None,
            "bar_vel_px_s": bar_vel,
            "angles":       angles,
        })

        # Overlays
        _draw_panel(frame, phase, tracker.rep_count, bar_vel, angles)
        _draw_angle_label(frame, _g(lm, "left_hip"),    angles.get("left_hip_hinge"),  color=(255, 200, 80))
        _draw_angle_label(frame, _g(lm, "right_hip"),   angles.get("right_hip_hinge"), color=(255, 200, 80))
        _draw_angle_label(frame, _g(lm, "left_knee"),   angles.get("left_knee"))
        _draw_angle_label(frame, _g(lm, "right_knee"),  angles.get("right_knee"))
        _draw_angle_label(frame, _g(lm, "left_elbow"),  angles.get("left_arm"),  color=(180, 255, 100))
        _draw_angle_label(frame, _g(lm, "right_elbow"), angles.get("right_arm"), color=(180, 255, 100))

        if tracker._current_rep:
            _draw_bar_path(frame, tracker._current_rep.get("bar_path", []))
        if bar_pt:
            cv.circle(frame, bar_pt, 6, (0, 200, 255), cv.FILLED)

        writer.write(frame)
        if frame_idx % 60 == 0:
            print(f"  [{frame_idx:>5}/{total}]  reps={tracker.rep_count}  phase={phase}")

    cap.release()
    writer.release()
    cv.destroyAllWindows()
    tracker.flush()

    reps_summary = [_derive_rep_metrics(r, fps) for r in tracker.completed_reps]

    def _avg(key):
        vals = [r.get(key) for r in reps_summary if r.get(key) is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    summary = {
        "total_reps":                  tracker.rep_count,
        "video_duration_s":            round(frame_idx / fps, 2),
        "avg_rep_duration_s":          _avg("duration_s"),
        "avg_pull_time_s":             _avg("pull_time_s"),
        "avg_lower_time_s":            _avg("lower_time_s"),
        "avg_range_of_motion_px":      _avg("range_of_motion_px"),
        "avg_setup_back_angle":        _avg("setup_back_angle"),
        "avg_lockout_back_angle":      _avg("lockout_back_angle"),
        "avg_lockout_left_hip_hinge":  _avg("lockout_left_hip_hinge"),
        "avg_lockout_right_hip_hinge": _avg("lockout_right_hip_hinge"),
        "avg_lockout_left_knee":       _avg("lockout_left_knee"),
        "avg_lockout_right_knee":      _avg("lockout_right_knee"),
        "avg_back_angle":              _avg("avg_back_angle"),
        "avg_hip_symmetry":            _avg("avg_hip_symmetry"),
        "avg_knee_symmetry":           _avg("avg_knee_symmetry"),
        "avg_bar_path_x_std_px":       _avg("bar_path_x_std_px"),
        "avg_shoulder_bar_offset_px":  _avg("avg_shoulder_bar_offset_px"),
    }

    report = {
        "lift":            "deadlift",
        "video_info":      {"path": video_path, "fps": fps, "width": w,
                            "height": h, "total_frames": frame_idx},
        "overall_summary": summary,
        "reps":            reps_summary,
        "frame_data":      frame_data,
    }

    with open(json_out, "w") as f:
        json.dump(report, f, indent=2, default=_json_safe)

    print(f"\n{'─'*55}")
    print(f"  Total reps detected   : {tracker.rep_count}")
    print(f"  Avg rep duration      : {summary['avg_rep_duration_s']} s")
    print(f"  Avg pull / lower      : {summary['avg_pull_time_s']} s / {summary['avg_lower_time_s']} s")
    print(f"  Avg ROM               : {summary['avg_range_of_motion_px']} px")
    print(f"  Avg back angle @ setup: {summary['avg_setup_back_angle']}°")
    print(f"  Avg back angle overall: {summary['avg_back_angle']}°")
    print(f"  Avg lockout hip hinge : L={summary['avg_lockout_left_hip_hinge']}°  R={summary['avg_lockout_right_hip_hinge']}°")
    print(f"  Avg lockout knee      : L={summary['avg_lockout_left_knee']}°  R={summary['avg_lockout_right_knee']}°")
    print(f"  Avg bar path x drift  : {summary['avg_bar_path_x_std_px']} px")
    print(f"  Avg shoulder-bar Δx   : {summary['avg_shoulder_bar_offset_px']} px")
    print(f"{'─'*55}")
    print(f"  Annotated video → {vid_out}")
    print(f"  Analysis JSON   → {json_out}")
    print(f"{'─'*55}\n")

    return report


def _json_safe(obj):
    if isinstance(obj, (np.int64, np.int32, np.int16)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    return str(obj)


if __name__ == "__main__":
    analyze(INPUT_VIDEO)
