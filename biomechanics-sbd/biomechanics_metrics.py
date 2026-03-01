"""
biomechanics_metrics.py
=======================
Math utilities and RepTracker for bench-press biomechanics analysis.
"""

import math
import numpy as np


# ── Geometry ──────────────────────────────────────────────────────────────────

def calculate_angle(a, b, c) -> float:
    """
    Angle at vertex *b* (degrees), formed by vectors b→a and b→c.
    Returns 0.0 when points are coincident.
    """
    a, b, c = (np.array(p, dtype=float) for p in (a, b, c))
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return 0.0
    cos_a = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return round(float(np.degrees(np.arccos(cos_a))), 1)


def calculate_velocity(prev_pt, curr_pt, fps: float) -> float:
    """Euclidean pixel speed (px/s) between two consecutive-frame positions."""
    if prev_pt is None or curr_pt is None:
        return 0.0
    return round(math.hypot(curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1]) * fps, 1)


def symmetry_score(left_val: float, right_val: float) -> float:
    """
    Symmetry ratio in [0, 1]  →  1.0 = perfect symmetry.
    Formula: 1 - |L - R| / max(|L|, |R|)
    """
    mx = max(abs(left_val), abs(right_val))
    if mx < 1e-6:
        return 1.0
    return round(1.0 - abs(left_val - right_val) / mx, 3)


def moving_average(buf: list, window: int) -> float:
    """Simple moving average of the last *window* values."""
    if not buf:
        return 0.0
    w = buf[-window:]
    return sum(w) / len(w)


# ── Smoothed signal helper ─────────────────────────────────────────────────────

def _full_smooth(values: list, window: int) -> list:
    """Causal moving average – O(n) with a running sum."""
    out, s = [], 0.0
    buf = []
    for v in values:
        buf.append(v)
        s += v
        if len(buf) > window:
            s -= buf[-window - 1]
            buf = buf[-window:]
        out.append(s / len(buf))
    return out


# ── RepTracker ────────────────────────────────────────────────────────────────

class RepTracker:
    """
    Detects bench-press / squat repetitions from the vertical position of the
    tracked point using a velocity-sign state machine with hysteresis.

    Phase labels
    ------------
    "idle"    – no significant movement detected yet
    "descent" – tracked point moving downward  (y increases in screen coords)
    "ascent"  – tracked point moving upward    (y decreases in screen coords)

    A rep is completed on every ascent→descent transition.

    Parameters
    ----------
    phase_labels : dict, optional
        Remap internal phase names for display, e.g.
        {"descent": "going down", "ascent": "going up"}
    """

    def __init__(self, fps: float, smooth_window: int = 7, vel_threshold: float = 4.0,
                 min_rep_frames: int = 15, min_rom_px: float = 0.0,
                 phase_labels: dict | None = None):
        self.fps            = fps
        self.smooth_window  = smooth_window
        self.vel_threshold  = vel_threshold
        self.min_rep_frames = min_rep_frames
        self.min_rom_px     = min_rom_px
        self._phase_labels  = phase_labels or {}

        self._y_raw: list[float] = []
        self._phase         = "idle"
        self._current_rep: dict | None = None
        self.completed_reps: list[dict] = []
        self._frame_idx     = 0

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def rep_count(self) -> int:
        return len(self.completed_reps)

    @property
    def phase(self) -> str:
        return self._phase_labels.get(self._phase, self._phase)

    def update(self, bar_y: float) -> str:
        """
        Feed the bar y-coordinate for the current frame.
        Returns the (remapped) current phase string.
        """
        self._frame_idx += 1
        self._y_raw.append(bar_y)

        # Track true min/max y seen in the current rep (ROM calculation)
        if self._current_rep is not None:
            self._current_rep["_min_y"] = min(self._current_rep["_min_y"], bar_y)
            self._current_rep["_max_y"] = max(self._current_rep["_max_y"], bar_y)

        if len(self._y_raw) < self.smooth_window + 1:
            return self.phase

        smoothed = _full_smooth(self._y_raw, self.smooth_window)
        lookback = max(2, self.smooth_window // 2 + 1)
        vel = smoothed[-1] - smoothed[-lookback]

        # ── State machine ────────────────────────────────────────────────────
        if self._phase == "idle":
            if vel > self.vel_threshold:
                self._phase = "descent"
                self._open_rep()

        elif self._phase == "descent":
            if vel < -self.vel_threshold:
                self._phase = "ascent"
                if self._current_rep is not None:
                    self._current_rep["bottom_frame"] = self._frame_idx
                    self._current_rep["bottom_y"]     = bar_y

        elif self._phase == "ascent":
            if vel > self.vel_threshold:
                if (self._current_rep is not None and
                        self._frame_idx - self._current_rep["start_frame"] >= self.min_rep_frames):
                    self._close_rep()
                self._phase = "descent"
                self._open_rep()

        return self.phase

    def flush(self):
        """Call after the last frame to finalize any open rep."""
        if (self._current_rep is not None and
                self._current_rep.get("bottom_frame") is not None):
            self._close_rep()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _open_rep(self):
        start_y = self._y_raw[-1]
        self._current_rep = {
            "rep_number":   self.rep_count + 1,
            "start_frame":  self._frame_idx,
            "top_y":        start_y,
            "bottom_frame": None,
            "bottom_y":     None,
            "frame_angles": [],
            "bar_path":     [],
            # Running min/max for true ROM measurement
            "_min_y":       start_y,
            "_max_y":       start_y,
        }

    def _close_rep(self):
        r = self._current_rep
        if r is None:
            return

        # Use actual min/max y seen across ALL frames of the rep
        # (more accurate than transition-point y values, which are offset by smoothing)
        rom = r.get("_max_y", r["top_y"]) - r.get("_min_y", r["top_y"])

        # ── Minimum ROM guard: discard noise oscillations ─────────────────
        if rom < self.min_rom_px:
            self._current_rep = None   # silently discard; do NOT count
            return

        r["end_frame"]      = self._frame_idx
        total_f             = r["end_frame"] - r["start_frame"]
        bottom_f            = r.get("bottom_frame") or self._frame_idx
        descent_f           = bottom_f - r["start_frame"]
        ascent_f            = r["end_frame"] - bottom_f

        r["duration_s"]     = round(total_f   / self.fps, 2)
        r["descent_time_s"] = round(descent_f / self.fps, 2)
        r["ascent_time_s"]  = round(ascent_f  / self.fps, 2)
        r["range_of_motion_px"] = round(rom, 1)
        self.completed_reps.append(r)
        self._current_rep = None


# ── DeadliftRepTracker ────────────────────────────────────────────────────────

class DeadliftRepTracker:
    """
    Rep tracker tailored to deadlift kinematics.

    Tracks the wrist midpoint y-coordinate (proxy for bar height).
    In screen coordinates, bar going UP  → y decreases (pull phase).
    Bar going DOWN                        → y increases (lower phase).

    Phase labels
    ------------
    "idle"   – bar stationary at floor
    "pull"   – bar moving upward (concentric)
    "lower"  – bar moving downward (eccentric / returning to floor)

    A rep is completed on every lower → pull transition.
    """

    def __init__(self, fps: float, smooth_window: int = 7, vel_threshold: float = 4.0,
                 min_rep_frames: int = 15):
        self.fps            = fps
        self.smooth_window  = smooth_window
        self.vel_threshold  = vel_threshold
        self.min_rep_frames = min_rep_frames

        self._y_raw: list[float] = []
        self._phase         = "idle"
        self._current_rep: dict | None = None
        self.completed_reps: list[dict] = []
        self._frame_idx     = 0

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def rep_count(self) -> int:
        return len(self.completed_reps)

    @property
    def phase(self) -> str:
        return self._phase

    def update(self, bar_y: float) -> str:
        self._frame_idx += 1
        self._y_raw.append(bar_y)

        if len(self._y_raw) < self.smooth_window + 1:
            return self._phase

        smoothed = _full_smooth(self._y_raw, self.smooth_window)
        lookback = max(2, self.smooth_window // 2 + 1)
        vel = smoothed[-1] - smoothed[-lookback]   # negative = bar going UP

        if self._phase == "idle":
            if vel < -self.vel_threshold:           # bar starting to rise
                self._phase = "pull"
                self._open_rep()

        elif self._phase == "pull":
            if vel > self.vel_threshold:            # bar starting to descend
                self._phase = "lower"
                if self._current_rep is not None:
                    self._current_rep["lockout_frame"] = self._frame_idx
                    self._current_rep["lockout_y"]     = bar_y

        elif self._phase == "lower":
            if vel < -self.vel_threshold:           # starting a new pull
                if (self._current_rep is not None and
                        self._frame_idx - self._current_rep["start_frame"] >= self.min_rep_frames):
                    self._close_rep()
                self._phase = "pull"
                self._open_rep()

        return self._phase

    def flush(self):
        if (self._current_rep is not None and
                self._current_rep.get("lockout_frame") is not None):
            self._close_rep()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _open_rep(self):
        self._current_rep = {
            "rep_number":    self.rep_count + 1,
            "start_frame":   self._frame_idx,
            "floor_y":       self._y_raw[-1],   # bar at floor = high y
            "lockout_frame": None,
            "lockout_y":     None,
            "frame_angles":  [],
            "bar_path":      [],
        }

    def _close_rep(self):
        r = self._current_rep
        if r is None:
            return
        r["end_frame"]       = self._frame_idx
        total_f              = r["end_frame"] - r["start_frame"]
        lockout_f            = r.get("lockout_frame") or self._frame_idx
        pull_f               = lockout_f - r["start_frame"]
        lower_f              = r["end_frame"] - lockout_f

        r["duration_s"]      = round(total_f  / self.fps, 2)
        r["pull_time_s"]     = round(pull_f   / self.fps, 2)
        r["lower_time_s"]    = round(lower_f  / self.fps, 2)
        r["range_of_motion_px"] = round(
            abs((r.get("lockout_y") or r["floor_y"]) - r["floor_y"]), 1
        )
        self.completed_reps.append(r)
        self._current_rep = None

