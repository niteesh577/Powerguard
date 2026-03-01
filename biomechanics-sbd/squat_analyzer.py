"""
squat_analyzer.py
=================
Comprehensive squat biomechanics data collector.

Usage
-----
    python squat_analyzer.py

Edit INPUT_VIDEO at the bottom to point to your file.

Metrics collected
-----------------
Per frame:
  - Left / right knee angle            (hip → knee → ankle)
  - Left / right hip angle             (shoulder → hip → knee)
  - Torso lean angle from vertical     (0° = perfectly upright)
  - Knee valgus (cave) per side        (knee x offset from ankle x)
  - Stance width                       (ankle-to-ankle distance px)
  - Hip depth relative to knee         (positive = hip above knee = not at parallel)
  - Below-parallel flag                (hip y > knee y in screen coords)
  - Bar position                       (wrist midpoint – proxy for bar on back)
  - Hip midpoint velocity              (how fast the lifter is moving)
  - Knee angle symmetry, hip symmetry
  - Phase: idle / descent / ascent

Per rep:
  - Descent / ascent time, total duration
  - Range of motion (hip vertical travel, px)
  - Minimum knee angle (maximum squat depth)
  - Below-parallel achieved (bool)
  - Torso lean at bottom
  - Max knee valgus per side
  - Average stance width
  - Hip / knee symmetry
  - Sticking point (frame of minimum hip velocity on the way up)

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
    RepTracker,
)
from utils import draw_text_with_bg

INPUT_VIDEO = "VIDEOS/INPUTS/squats.mp4"

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
    "idle":    (160, 160, 160),
    "descent": (0,   100, 255),   # orange – going down
    "ascent":  (60,  220,  60),   # green  – coming up
}

# Squat-specific phase labels for display
_PHASE_LABELS = {
    "descent": "descent",
    "ascent":  "ascent",
    "idle":    "idle",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _g(lm, key):
    return lm.get(_IDX[key])

def _mid(a, b):
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)

def _torso_angle(shoulder_mid, hip_mid) -> float | None:
    """Angle of the torso from vertical (0° = perfectly upright)."""
    if shoulder_mid is None or hip_mid is None:
        return None
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]   # negative = shoulder above hip
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None
    cos_a = np.clip(-dy / length, -1.0, 1.0)
    return round(math.degrees(math.acos(cos_a)), 1)


# ── Per-frame angle computation ───────────────────────────────────────────────

def compute_angles(lm: dict, frame_h: int = 0) -> dict:
    angles = {}

    ls, rs = _g(lm, "left_shoulder"),  _g(lm, "right_shoulder")
    lh, rh = _g(lm, "left_hip"),       _g(lm, "right_hip")
    lk, rk = _g(lm, "left_knee"),      _g(lm, "right_knee")
    la, ra = _g(lm, "left_ankle"),     _g(lm, "right_ankle")
    lw, rw = _g(lm, "left_wrist"),     _g(lm, "right_wrist")

    # Knee angles – hip → knee → ankle
    if lh and lk and la:
        angles["left_knee"]  = calculate_angle(lh, lk, la)
    if rh and rk and ra:
        angles["right_knee"] = calculate_angle(rh, rk, ra)

    # Hip angles – shoulder → hip → knee
    if ls and lh and lk:
        angles["left_hip"]  = calculate_angle(ls, lh, lk)
    if rs and rh and rk:
        angles["right_hip"] = calculate_angle(rs, rh, rk)

    # Torso lean from vertical
    shoulder_mid = _mid(ls, rs)
    hip_mid      = _mid(lh, rh)
    ta = _torso_angle(shoulder_mid, hip_mid)
    if ta is not None:
        angles["torso_lean"] = ta

    # Squat depth: hip y vs knee y (higher y = lower in frame = deeper)
    knee_mid = _mid(lk, rk)
    if hip_mid and knee_mid:
        diff = knee_mid[1] - hip_mid[1]   # positive = hip above knee
        angles["hip_above_knee_px"] = round(diff, 1)
        angles["below_parallel"]    = bool(diff < 0)  # hip below knee

    # Knee valgus (cave): knee x offset from ankle x
    # Positive (left side) = knee tracking inside ankle = valgus
    if lk and la:
        angles["left_knee_valgus_px"]  = round(la[0] - lk[0], 1)  # + = knee collapsing in
    if rk and ra:
        angles["right_knee_valgus_px"] = round(rk[0] - ra[0], 1)  # + = knee collapsing in

    # Stance width (ankle distance)
    if la and ra:
        angles["stance_width_px"] = round(
            math.hypot(ra[0] - la[0], ra[1] - la[1]), 1)

    # Bar position (wrist midpoint – proxy for bar on traps)
    bar_mid = _mid(lw, rw)
    if bar_mid:
        angles["bar_x"] = bar_mid[0]
        angles["bar_y"] = bar_mid[1]

    # Symmetry
    if "left_knee" in angles and "right_knee" in angles:
        angles["knee_angle_symmetry"] = symmetry_score(
            angles["left_knee"], angles["right_knee"])
    if "left_hip" in angles and "right_hip" in angles:
        angles["hip_angle_symmetry"]  = symmetry_score(
            angles["left_hip"], angles["right_hip"])

    return angles


# ── Overlay ───────────────────────────────────────────────────────────────────

def _draw_panel(frame, phase, rep_count, hip_vel, angles, panel_w=275):
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
    put(f"HIP VEL: {hip_vel:>6.0f} px/s", 88, col=(200, 180, 255))

    # Depth indicator
    below = angles.get("below_parallel")
    depth_col = (0, 255, 100) if below else (0, 80, 255)
    depth_str = "BELOW PAR ✓" if below else "ABOVE PAR  "
    put(f"DEPTH: {depth_str}", 114, col=depth_col, scale=0.52, bold=(below is True))

    cv.line(frame, (12, 126), (panel_w - 12, 126), (60, 60, 60), 1)

    rows = [
        ("L Knee",        angles.get("left_knee",      "--"), "°"),
        ("R Knee",        angles.get("right_knee",     "--"), "°"),
        ("L Hip",         angles.get("left_hip",       "--"), "°"),
        ("R Hip",         angles.get("right_hip",      "--"), "°"),
        ("Torso Lean",    angles.get("torso_lean",     "--"), "°"),
        ("L Knee Valg",  angles.get("left_knee_valgus_px",  "--"), " px"),
        ("R Knee Valg",  angles.get("right_knee_valgus_px", "--"), " px"),
        ("Stance W",     angles.get("stance_width_px", "--"), " px"),
        ("Knee Sym",     angles.get("knee_angle_symmetry",  "--"), ""),
        ("Hip Sym",      angles.get("hip_angle_symmetry",   "--"), ""),
    ]
    y = 148
    for label, val, unit in rows:
        v = f"{val:.1f}{unit}" if isinstance(val, float) else f"{val}{unit}"
        put(f"{label:<12}: {v:>8}", y)
        y += 25


def _draw_angle_label(frame, pt, value, color=(255, 255, 0)):
    if pt is None or value is None:
        return
    x, y = pt
    cv.putText(frame, f"{value:.0f}",
               (x + 8, y - 8), cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv.LINE_AA)


def _draw_hip_path(frame, path, color=(255, 100, 200)):
    for i in range(1, len(path)):
        cv.line(frame, tuple(path[i - 1]), tuple(path[i]), color, 2, cv.LINE_AA)


# ── Per-rep derived metrics ───────────────────────────────────────────────────

def _derive_rep_metrics(r: dict, fps: float) -> dict:
    fa = [a for a in r.get("frame_angles", []) if a]
    bp = r.get("bar_path", [])    # hip path in this context

    entry = {
        "rep_number":           r["rep_number"],
        "start_frame":          r["start_frame"],
        "end_frame":            r.get("end_frame"),
        "duration_s":           r.get("duration_s"),
        "descent_time_s":       r.get("descent_time_s"),
        "ascent_time_s":        r.get("ascent_time_s"),
        "range_of_motion_px":   r.get("range_of_motion_px"),
        "hip_path":             bp,
    }

    if not fa:
        return entry

    ys = [p[1] for p in bp] if bp else []

    # Bottom of the squat – frames near max y (lowest hip position)
    if ys:
        bottom_idx = int(np.argmax(ys))
        win = slice(max(0, bottom_idx - 5), min(len(fa), bottom_idx + 5))
        bottom_fa = fa[win]
        if bottom_fa:
            lk_vals = [a.get("left_knee")  for a in bottom_fa if a.get("left_knee")]
            rk_vals = [a.get("right_knee") for a in bottom_fa if a.get("right_knee")]
            if lk_vals: entry["bottom_left_knee_angle"]  = round(float(np.mean(lk_vals)), 1)
            if rk_vals: entry["bottom_right_knee_angle"] = round(float(np.mean(rk_vals)), 1)

            tl_vals = [a.get("torso_lean") for a in bottom_fa if a.get("torso_lean")]
            if tl_vals: entry["bottom_torso_lean"]       = round(float(np.mean(tl_vals)), 1)

            # Knee valgus at bottom
            lv = [a.get("left_knee_valgus_px")  for a in bottom_fa if a.get("left_knee_valgus_px") is not None]
            rv = [a.get("right_knee_valgus_px") for a in bottom_fa if a.get("right_knee_valgus_px") is not None]
            if lv: entry["bottom_left_knee_valgus_px"]  = round(float(np.max(np.abs(lv))), 1)
            if rv: entry["bottom_right_knee_valgus_px"] = round(float(np.max(np.abs(rv))), 1)

        # Below parallel achieved
        bp_flags = [a.get("below_parallel") for a in fa if a.get("below_parallel") is not None]
        entry["below_parallel_achieved"] = any(bp_flags) if bp_flags else False

        # Sticking point – min hip velocity during ascent (y decreasing)
        if len(ys) > 5:
            ascent_ys = ys[bottom_idx:]
            if len(ascent_ys) > 3:
                vels = [abs(ascent_ys[i] - ascent_ys[i - 1]) for i in range(1, len(ascent_ys))]
                stick_off = int(np.argmin(vels))
                entry["sticking_point_rep_frame_offset"] = bottom_idx + stick_off
                entry["sticking_point_time_s"]           = round(
                    (r["start_frame"] + bottom_idx + stick_off) / fps, 2)

    # Max knee valgus across the whole rep
    lv_all = [abs(a.get("left_knee_valgus_px",  0)) for a in fa if a.get("left_knee_valgus_px")  is not None]
    rv_all = [abs(a.get("right_knee_valgus_px", 0)) for a in fa if a.get("right_knee_valgus_px") is not None]
    if lv_all: entry["max_left_knee_valgus_px"]  = round(float(max(lv_all)), 1)
    if rv_all: entry["max_right_knee_valgus_px"] = round(float(max(rv_all)), 1)

    # Averages
    def _avg(key):
        vals = [a.get(key) for a in fa if a.get(key) is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    entry["avg_knee_angle_symmetry"] = _avg("knee_angle_symmetry")
    entry["avg_hip_angle_symmetry"]  = _avg("hip_angle_symmetry")
    entry["avg_torso_lean"]          = _avg("torso_lean")
    entry["avg_stance_width_px"]     = _avg("stance_width_px")
    entry["min_knee_angle_left"]     = round(min((a.get("left_knee",  180) for a in fa), default=180), 1)
    entry["min_knee_angle_right"]    = round(min((a.get("right_knee", 180) for a in fa), default=180), 1)

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
    # Track hip midpoint y; hip goes DOWN on descent (y increases) — default direction
    tracker  = RepTracker(fps=fps, phase_labels=_PHASE_LABELS)

    frame_data = []
    prev_hip   = None
    frame_idx  = 0

    print(f"\n[INFO] Analyzing (Squat): {video_path}")
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
                                "phase": "no_detection", "angles": {}, "hip_pt": None})
            continue

        angles  = compute_angles(lm, frame_h=h)
        hip_pt  = _mid(_g(lm, "left_hip"), _g(lm, "right_hip"))
        hip_vel = calculate_velocity(prev_hip, hip_pt, fps) if prev_hip else 0.0
        prev_hip = hip_pt

        phase = tracker.update(hip_pt[1] if hip_pt else 0.0)

        # Accumulate into current rep
        if tracker._current_rep is not None:
            tracker._current_rep["frame_angles"].append(angles)
            if hip_pt:
                tracker._current_rep["bar_path"].append(list(hip_pt))

        frame_data.append({
            "frame":        frame_idx,
            "timestamp_s":  ts,
            "phase":        phase,
            "hip_pt":       list(hip_pt) if hip_pt else None,
            "hip_vel_px_s": hip_vel,
            "angles":       angles,
        })

        # Overlays
        _draw_panel(frame, phase, tracker.rep_count, hip_vel, angles)
        _draw_angle_label(frame, _g(lm, "left_knee"),  angles.get("left_knee"))
        _draw_angle_label(frame, _g(lm, "right_knee"), angles.get("right_knee"))
        _draw_angle_label(frame, _g(lm, "left_hip"),   angles.get("left_hip"),  color=(255, 200, 80))
        _draw_angle_label(frame, _g(lm, "right_hip"),  angles.get("right_hip"), color=(255, 200, 80))

        if tracker._current_rep:
            _draw_hip_path(frame, tracker._current_rep.get("bar_path", []))
        if hip_pt:
            cv.circle(frame, hip_pt, 7, (255, 80, 200), cv.FILLED)

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

    below_par_count = sum(1 for r in reps_summary if r.get("below_parallel_achieved"))

    summary = {
        "total_reps":                   tracker.rep_count,
        "reps_below_parallel":          below_par_count,
        "video_duration_s":             round(frame_idx / fps, 2),
        "avg_rep_duration_s":           _avg("duration_s"),
        "avg_descent_time_s":           _avg("descent_time_s"),
        "avg_ascent_time_s":            _avg("ascent_time_s"),
        "avg_range_of_motion_px":       _avg("range_of_motion_px"),
        "avg_bottom_left_knee_angle":   _avg("bottom_left_knee_angle"),
        "avg_bottom_right_knee_angle":  _avg("bottom_right_knee_angle"),
        "avg_bottom_torso_lean":        _avg("bottom_torso_lean"),
        "avg_max_left_knee_valgus_px":  _avg("max_left_knee_valgus_px"),
        "avg_max_right_knee_valgus_px": _avg("max_right_knee_valgus_px"),
        "avg_knee_symmetry":            _avg("avg_knee_angle_symmetry"),
        "avg_hip_symmetry":             _avg("avg_hip_angle_symmetry"),
        "avg_stance_width_px":          _avg("avg_stance_width_px"),
        "avg_min_left_knee_angle":      _avg("min_knee_angle_left"),
        "avg_min_right_knee_angle":     _avg("min_knee_angle_right"),
    }

    report = {
        "lift":            "squat",
        "video_info":      {"path": video_path, "fps": fps, "width": w,
                            "height": h, "total_frames": frame_idx},
        "overall_summary": summary,
        "reps":            reps_summary,
        "frame_data":      frame_data,
    }

    with open(json_out, "w") as f:
        json.dump(report, f, indent=2, default=_json_safe)

    print(f"\n{'─'*55}")
    print(f"  Total reps detected      : {tracker.rep_count}")
    print(f"  Reps below parallel      : {below_par_count}/{tracker.rep_count}")
    print(f"  Avg rep duration         : {summary['avg_rep_duration_s']} s")
    print(f"  Avg descent / ascent     : {summary['avg_descent_time_s']} s / {summary['avg_ascent_time_s']} s")
    print(f"  Avg ROM                  : {summary['avg_range_of_motion_px']} px")
    print(f"  Avg knee angle @ bottom  : L={summary['avg_bottom_left_knee_angle']}°  R={summary['avg_bottom_right_knee_angle']}°")
    print(f"  Avg torso lean @ bottom  : {summary['avg_bottom_torso_lean']}°")
    print(f"  Avg max knee valgus      : L={summary['avg_max_left_knee_valgus_px']} px  R={summary['avg_max_right_knee_valgus_px']} px")
    print(f"  Avg knee symmetry        : {summary['avg_knee_symmetry']}")
    print(f"  Avg stance width         : {summary['avg_stance_width_px']} px")
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
