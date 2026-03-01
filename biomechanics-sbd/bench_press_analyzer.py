"""
bench_press_analyzer.py
=======================
Comprehensive bench-press biomechanics data collector.

Usage
-----
    python bench_press_analyzer.py

Outputs (saved to VIDEOS/OUTPUTS/)
-------
  * <name>_analysis.mp4   – annotated video with live overlays
  * <name>_analysis.json  – structured metrics report for downstream agent

Metrics collected
-----------------
Per frame:
  - Left / right elbow angles
  - Left / right shoulder angles
  - Elbow flare angle (between upper arms)
  - Grip width ratio (grip / shoulder width)
  - Bar (wrist midpoint) position + velocity
  - Wrist symmetry, elbow angle symmetry
  - Current phase (idle / descent / ascent)

Per rep:
  - Duration, descent time, ascent time
  - Range of motion (px)
  - Elbow angles at bottom (chest touch) and lockout
  - Elbow flare at bottom
  - Average symmetry score
  - Bar path coordinates
  - Bar path linearity (x std dev)
  - Sticking point frame + time
  - Avg grip width ratio

Session overall:
  - Totals and averages of all per-rep metrics
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


# ── Video to analyze ──────────────────────────────────────────────────────────
INPUT_VIDEO = "VIDEOS/INPUTS/bench_press.mp4"

# ── MediaPipe landmark indices ────────────────────────────────────────────────
_IDX = {
    "nose":           0,
    "left_shoulder":  11, "right_shoulder": 12,
    "left_elbow":     13, "right_elbow":    14,
    "left_wrist":     15, "right_wrist":    16,
    "left_hip":       23, "right_hip":      24,
}

# Phase → BGR colour for the overlay panel header
_PHASE_COLOR = {
    "idle":    (160, 160, 160),
    "descent": (0,  100, 255),   # orange-ish
    "ascent":  (60, 220,  60),   # green
}


# ── Landmark helpers ──────────────────────────────────────────────────────────

def _g(lm: dict, key: str):
    """Return landmark (cx, cy) by name, or None."""
    return lm.get(_IDX[key])


def _midpoint(a, b):
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)


# ── Per-frame metric computation ──────────────────────────────────────────────

def compute_angles(lm: dict) -> dict:
    """Return all biomechanically relevant angles for one frame."""
    angles = {}

    ls  = _g(lm, "left_shoulder");   rs  = _g(lm, "right_shoulder")
    le  = _g(lm, "left_elbow");      re  = _g(lm, "right_elbow")
    lw  = _g(lm, "left_wrist");      rw  = _g(lm, "right_wrist")
    lh  = _g(lm, "left_hip");        rh  = _g(lm, "right_hip")

    # Elbow angles – shoulder→elbow→wrist
    if ls and le and lw:
        angles["left_elbow"]  = calculate_angle(ls, le, lw)
    if rs and re and rw:
        angles["right_elbow"] = calculate_angle(rs, re, rw)

    # Shoulder angles – hip→shoulder→elbow
    if lh and ls and le:
        angles["left_shoulder"]  = calculate_angle(lh, ls, le)
    if rh and rs and re:
        angles["right_shoulder"] = calculate_angle(rh, rs, re)

    # Elbow flare – angle between the two upper-arm vectors
    if ls and le and rs and re:
        vl = np.array(le) - np.array(ls)
        vr = np.array(re) - np.array(rs)
        d  = np.linalg.norm(vl) * np.linalg.norm(vr)
        if d > 1e-6:
            cos_a = np.clip(np.dot(vl, vr) / d, -1.0, 1.0)
            angles["elbow_flare"] = round(float(np.degrees(np.arccos(cos_a))), 1)

    # Grip width ratio  (grip distance / shoulder width)
    if lw and rw and ls and rs:
        grip_w  = math.hypot(rw[0] - lw[0], rw[1] - lw[1])
        shoul_w = math.hypot(rs[0] - ls[0], rs[1] - ls[1])
        if shoul_w > 1e-6:
            angles["grip_width_ratio"] = round(grip_w / shoul_w, 2)

    # Wrist height symmetry
    if lw and rw:
        angles["wrist_height_diff_px"] = round(abs(lw[1] - rw[1]), 1)
        angles["wrist_symmetry"]        = symmetry_score(float(lw[1]), float(rw[1]))

    # Elbow angle symmetry (L vs R)
    if "left_elbow" in angles and "right_elbow" in angles:
        angles["elbow_angle_symmetry"] = symmetry_score(
            angles["left_elbow"], angles["right_elbow"]
        )
    # Shoulder angle symmetry
    if "left_shoulder" in angles and "right_shoulder" in angles:
        angles["shoulder_angle_symmetry"] = symmetry_score(
            angles["left_shoulder"], angles["right_shoulder"]
        )

    return angles


# ── Overlay drawing helpers ───────────────────────────────────────────────────

def _draw_panel(frame, phase: str, rep_count: int, bar_vel: float,
                angles: dict, panel_w: int = 265):
    """Draw a translucent info panel on the left of the frame."""
    overlay = frame.copy()
    cv.rectangle(overlay, (0, 0), (panel_w, frame.shape[0]), (15, 15, 15), cv.FILLED)
    cv.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    color = _PHASE_COLOR.get(phase, (200, 200, 200))

    def put(text: str, y: int, col=(220, 220, 220), scale=0.52, bold=False):
        thickness = 2 if bold else 1
        cv.putText(frame, text, (12, y),
                   cv.FONT_HERSHEY_SIMPLEX, scale, col, thickness, cv.LINE_AA)

    # ── Header ───────────────────────────────────────────────────────────────
    put(f"REPS: {rep_count}",     30, col=(0, 255, 180), scale=0.70, bold=True)
    put(f"PHASE: {phase.upper()}", 60, col=color, scale=0.58, bold=True)
    put(f"BAR VEL: {bar_vel:>6.0f} px/s", 88, col=(200, 180, 255))

    # ── Divider ───────────────────────────────────────────────────────────────
    cv.line(frame, (12, 98), (panel_w - 12, 98), (60, 60, 60), 1)

    # ── Angles ───────────────────────────────────────────────────────────────
    rows = [
        ("L Elbow",    angles.get("left_elbow",  "--"), "°"),
        ("R Elbow",    angles.get("right_elbow", "--"), "°"),
        ("L Shoulder", angles.get("left_shoulder",  "--"), "°"),
        ("R Shoulder", angles.get("right_shoulder", "--"), "°"),
        ("Flare",      angles.get("elbow_flare",    "--"), "°"),
        ("Grip Ratio", angles.get("grip_width_ratio", "--"), "x"),
        ("↔ Sym",      angles.get("elbow_angle_symmetry", "--"), ""),
        ("Wrist Δy",   angles.get("wrist_height_diff_px", "--"), " px"),
    ]
    y = 120
    for label, val, unit in rows:
        v = f"{val:.1f}{unit}" if isinstance(val, float) else f"{val}{unit}"
        put(f"{label:<11}: {v:>9}", y)
        y += 26


def _draw_angle_label(frame, pt, value, color=(255, 255, 0)):
    if pt is None or value is None:
        return
    x, y = pt
    cv.putText(frame, f"{value:.0f}",
               (x + 8, y - 8), cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv.LINE_AA)


def _draw_bar_path(frame, path: list, color=(80, 160, 255)):
    for i in range(1, len(path)):
        cv.line(frame, tuple(path[i - 1]), tuple(path[i]), color, 2, cv.LINE_AA)


# ── Rep-level derived metrics ─────────────────────────────────────────────────

def _derive_rep_metrics(r: dict, fps: float) -> dict:
    """
    Given a completed rep dict (from RepTracker), derive higher-level
    biomechanics metrics suitable for the analysis agent.
    """
    fa = [a for a in r.get("frame_angles", []) if a]
    bp = r.get("bar_path", [])

    entry = {
        "rep_number":           r["rep_number"],
        "start_frame":          r["start_frame"],
        "end_frame":            r.get("end_frame"),
        "duration_s":           r.get("duration_s"),
        "descent_time_s":       r.get("descent_time_s"),
        "ascent_time_s":        r.get("ascent_time_s"),
        "range_of_motion_px":   r.get("range_of_motion_px"),
        "bar_path":             bp,
    }

    if not fa:
        return entry

    # Bottom of the rep (chest touch) – frames near max bar y
    ys = [p[1] for p in bp] if bp else []
    if ys:
        bottom_idx = int(np.argmax(ys))
        win = slice(max(0, bottom_idx - 5), min(len(fa), bottom_idx + 5))
        bottom_fa = fa[win]
        if bottom_fa:
            entry["bottom_left_elbow_angle"]  = round(
                min((a.get("left_elbow",  180) for a in bottom_fa), default=None), 1)
            entry["bottom_right_elbow_angle"] = round(
                min((a.get("right_elbow", 180) for a in bottom_fa), default=None), 1)
            flares = [a.get("elbow_flare") for a in bottom_fa if a.get("elbow_flare")]
            if flares:
                entry["bottom_elbow_flare"] = round(float(np.mean(flares)), 1)

        # Lockout = top 10% of frames (start + end of bar_path)
        top_n = max(1, len(fa) // 10)
        top_fa = fa[:top_n] + fa[-top_n:]
        entry["lockout_left_elbow_angle"]  = round(
            max((a.get("left_elbow",  0) for a in top_fa), default=None), 1)
        entry["lockout_right_elbow_angle"] = round(
            max((a.get("right_elbow", 0) for a in top_fa), default=None), 1)

        # Sticking point (min bar velocity during ascent)
        if len(ys) > 5:
            ascent_ys = ys[bottom_idx:]
            if len(ascent_ys) > 3:
                vels = [abs(ascent_ys[i] - ascent_ys[i - 1])
                        for i in range(1, len(ascent_ys))]
                stick_off = int(np.argmin(vels))
                entry["sticking_point_rep_frame_offset"] = bottom_idx + stick_off
                entry["sticking_point_time_s"] = round(
                    (r["start_frame"] + bottom_idx + stick_off) / fps, 2)

        # Bar path linearity (x-axis std dev – lower = more vertical)
        if len(bp) > 2:
            entry["bar_path_x_std_px"] = round(float(np.std([p[0] for p in bp])), 1)
            entry["bar_path_y_range_px"] = round(float(max(ys) - min(ys)), 1)

    # Averages across all frames of the rep
    def _avg_field(key):
        vals = [a.get(key) for a in fa if a.get(key) is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    entry["avg_elbow_angle_symmetry"]     = _avg_field("elbow_angle_symmetry")
    entry["avg_shoulder_angle_symmetry"]  = _avg_field("shoulder_angle_symmetry")
    entry["avg_wrist_symmetry"]           = _avg_field("wrist_symmetry")
    entry["avg_grip_width_ratio"]         = _avg_field("grip_width_ratio")
    entry["avg_elbow_flare"]              = _avg_field("elbow_flare")

    return entry


# ── Main pipeline ─────────────────────────────────────────────────────────────

def analyze(video_path: str = INPUT_VIDEO) -> dict:
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv.CAP_PROP_FPS) or 30.0
    w      = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    # ── Output paths ─────────────────────────────────────────────────────────
    base     = os.path.splitext(os.path.basename(video_path))[0]
    out_dir  = os.path.join(os.path.dirname(video_path), "..", "OUTPUTS")
    os.makedirs(out_dir, exist_ok=True)
    vid_out  = os.path.join(out_dir, f"{base}_analysis.mp4")
    json_out = os.path.join(out_dir, f"{base}_analysis.json")

    writer     = cv.VideoWriter(vid_out, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    detector   = PoseDetector()
    tracker    = RepTracker(fps=fps,
                            smooth_window=9,       # heavier smoothing to tame noise
                            vel_threshold=5.0,     # low enough to detect all real reps
                            min_rep_frames=35,     # at 25fps ≈ 1.4 s – blocks unracking
                            min_rom_px=160.0)      # real reps ≈217px; small motions filtered
    frame_data = []
    prev_bar   = None
    frame_idx  = 0

    print(f"\n[INFO] Analyzing: {video_path}")
    print(f"       {total} frames @ {fps:.1f} fps  ({w}×{h})\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        ts = round(frame_idx / fps, 3)

        # ── Pose ─────────────────────────────────────────────────────────────
        frame = detector.find_pose(frame, draw=True)
        lm    = detector.get_positions(frame)

        if not lm:
            writer.write(frame)
            frame_data.append({"frame": frame_idx, "timestamp_s": ts,
                                "phase": "no_detection", "angles": {}, "bar_pt": None})
            continue

        # ── Metrics ───────────────────────────────────────────────────────────
        angles  = compute_angles(lm)
        bar_pt  = _midpoint(_g(lm, "left_wrist"), _g(lm, "right_wrist"))
        bar_vel = calculate_velocity(prev_bar, bar_pt, fps) if prev_bar else 0.0
        prev_bar = bar_pt

        phase = tracker.update(bar_pt[1] if bar_pt else 0.0)

        # Accumulate angles and bar path into the current rep
        if tracker._current_rep is not None:
            tracker._current_rep["frame_angles"].append(angles)
            if bar_pt:
                tracker._current_rep["bar_path"].append(list(bar_pt))

        # ── Store frame record ────────────────────────────────────────────────
        frame_data.append({
            "frame":        frame_idx,
            "timestamp_s":  ts,
            "phase":        phase,
            "bar_pt":       list(bar_pt) if bar_pt else None,
            "bar_vel_px_s": bar_vel,
            "angles":       angles,
        })

        # ── Draw overlays ─────────────────────────────────────────────────────
        _draw_panel(frame, phase, tracker.rep_count, bar_vel, angles)

        # Angle labels near joints
        _draw_angle_label(frame, _g(lm, "left_elbow"),     angles.get("left_elbow"))
        _draw_angle_label(frame, _g(lm, "right_elbow"),    angles.get("right_elbow"),    color=(255, 255, 0))
        _draw_angle_label(frame, _g(lm, "left_shoulder"),  angles.get("left_shoulder"),  color=(180, 255, 100))
        _draw_angle_label(frame, _g(lm, "right_shoulder"), angles.get("right_shoulder"), color=(180, 255, 100))

        # Live bar path for current rep
        if tracker._current_rep:
            _draw_bar_path(frame, tracker._current_rep.get("bar_path", []))

        # Bar midpoint dot
        if bar_pt:
            cv.circle(frame, bar_pt, 6, (0, 200, 255), cv.FILLED)

        writer.write(frame)

        if frame_idx % 60 == 0:
            print(f"  [{frame_idx:>5}/{total}]  reps={tracker.rep_count}  phase={phase}")

    cap.release()
    writer.release()
    cv.destroyAllWindows()

    # ── Finalise ──────────────────────────────────────────────────────────────
    tracker.flush()

    # ── Per-rep derived metrics ───────────────────────────────────────────────
    reps_summary = [_derive_rep_metrics(r, fps) for r in tracker.completed_reps]

    # ── Session summary ───────────────────────────────────────────────────────
    def _avg(key):
        vals = [r.get(key) for r in reps_summary if r.get(key) is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    summary = {
        "total_reps":               tracker.rep_count,
        "video_duration_s":         round(frame_idx / fps, 2),
        "avg_rep_duration_s":       _avg("duration_s"),
        "avg_descent_time_s":       _avg("descent_time_s"),
        "avg_ascent_time_s":        _avg("ascent_time_s"),
        "avg_range_of_motion_px":   _avg("range_of_motion_px"),
        "avg_bottom_left_elbow":    _avg("bottom_left_elbow_angle"),
        "avg_bottom_right_elbow":   _avg("bottom_right_elbow_angle"),
        "avg_lockout_left_elbow":   _avg("lockout_left_elbow_angle"),
        "avg_lockout_right_elbow":  _avg("lockout_right_elbow_angle"),
        "avg_bottom_elbow_flare":   _avg("bottom_elbow_flare"),
        "avg_elbow_angle_symmetry": _avg("avg_elbow_angle_symmetry"),
        "avg_wrist_symmetry":       _avg("avg_wrist_symmetry"),
        "avg_grip_width_ratio":     _avg("avg_grip_width_ratio"),
        "avg_bar_path_x_std_px":    _avg("bar_path_x_std_px"),
    }

    # ── Assemble full report ───────────────────────────────────────────────────
    report = {
        "video_info": {
            "path":          video_path,
            "fps":           fps,
            "width":         w,
            "height":        h,
            "total_frames":  frame_idx,
        },
        "overall_summary": summary,
        "reps":             reps_summary,
        "frame_data":       frame_data,   # full per-frame time-series
    }

    with open(json_out, "w") as f:
        json.dump(report, f, indent=2, default=_json_safe)

    # ── Summary print ─────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  Total reps detected   : {tracker.rep_count}")
    print(f"  Avg rep duration      : {summary['avg_rep_duration_s']} s")
    print(f"  Avg descent / ascent  : {summary['avg_descent_time_s']} s / {summary['avg_ascent_time_s']} s")
    print(f"  Avg ROM               : {summary['avg_range_of_motion_px']} px")
    print(f"  Avg elbow @ bottom    : L={summary['avg_bottom_left_elbow']}°  R={summary['avg_bottom_right_elbow']}°")
    print(f"  Avg elbow @ lockout   : L={summary['avg_lockout_left_elbow']}°  R={summary['avg_lockout_right_elbow']}°")
    print(f"  Avg elbow flare       : {summary['avg_bottom_elbow_flare']}°")
    print(f"  Avg grip ratio        : {summary['avg_grip_width_ratio']}x shoulder width")
    print(f"  Avg elbow symmetry    : {summary['avg_elbow_angle_symmetry']}")
    print(f"{'─'*55}")
    print(f"  Annotated video → {vid_out}")
    print(f"  Analysis JSON   → {json_out}")
    print(f"{'─'*55}\n")

    return report


# ── JSON serialisation helper ─────────────────────────────────────────────────

def _json_safe(obj):
    if isinstance(obj, (np.int64, np.int32, np.int16)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    return str(obj)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    analyze(INPUT_VIDEO)
