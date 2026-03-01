import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import time
import os


# ---------------------------------------------------------------------------
# Model setup – lite model (~7 MB), downloaded once next to this module
# ---------------------------------------------------------------------------
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(_MODULE_DIR, "pose_landmarker_lite.task")
MODEL_URL   = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# Landmark connections – mirrors the old mp.solutions.pose.POSE_CONNECTIONS
_POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26),
    (25, 27), (26, 28),
    (27, 29), (28, 30),
    (29, 31), (30, 32),
    (27, 31), (28, 32),
]


def _ensure_model() -> None:
    """Download the pose landmarker model if it is not present locally."""
    if not os.path.exists(MODEL_PATH):
        print("[PoseDetector] Downloading pose landmarker model (~7 MB) …")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[PoseDetector] Model saved to: {MODEL_PATH}")


# ---------------------------------------------------------------------------
# PoseDetector – same external API as the original (find_pose / get_positions)
# ---------------------------------------------------------------------------
class PoseDetector:
    """
    Pose detector using the MediaPipe Tasks API (mediapipe >= 0.10).

    Parameters
    ----------
    mode         : bool  – True → static IMAGE mode; False → VIDEO mode (default)
    complexity   : int   – ignored (Tasks API uses the model file for complexity)
    smooth       : bool  – ignored (Tasks API handles smoothing internally)
    detection_con: float – minimum pose detection confidence
    track_con    : float – minimum tracking confidence (VIDEO mode only)
    """

    def __init__(self, mode=False, complexity=1, smooth=True,
                 detection_con=0.5, track_con=0.5):
        self.results        = None
        self._frame_count   = 0

        _ensure_model()

        running_mode = (mp_vision.RunningMode.IMAGE if mode
                        else mp_vision.RunningMode.VIDEO)
        self._running_mode = running_mode

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_poses=1,
            min_pose_detection_confidence=detection_con,
            min_pose_presence_confidence=detection_con,
            min_tracking_confidence=track_con,
            output_segmentation_masks=False,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    def find_pose(self, frame, draw=True):
        """
        Detect pose landmarks in a BGR frame (same signature as the old API).

        Returns the (optionally annotated) BGR frame.
        """
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        if self._running_mode == mp_vision.RunningMode.VIDEO:
            # Timestamps must be strictly increasing (in ms)
            self._frame_count += 1
            timestamp_ms  = self._frame_count * 33          # ~30 fps equivalent
            self.results  = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            self.results  = self.landmarker.detect(mp_image)

        if draw and self.results.pose_landmarks:
            self._draw_landmarks(frame, self.results.pose_landmarks[0])

        return frame

    # ------------------------------------------------------------------
    def get_positions(self, frame):
        """
        Return a dict mapping landmark index → (cx, cy) pixel coordinates.
        Same return type as the old API.
        """
        landmarks = {}
        if self.results and self.results.pose_landmarks:
            h, w = frame.shape[:2]
            for idx, lm in enumerate(self.results.pose_landmarks[0]):
                landmarks[idx] = (int(lm.x * w), int(lm.y * h))
        return landmarks

    # ------------------------------------------------------------------
    def _draw_landmarks(self, frame, landmarks):
        """Draw pose skeleton on frame using OpenCV (no drawing_utils needed)."""
        h, w = frame.shape[:2]

        # Compute pixel positions for all landmarks
        pts = {}
        for i, lm in enumerate(landmarks):
            pts[i] = (int(lm.x * w), int(lm.y * h))

        # Draw connections first (underneath the dots)
        for a, b in _POSE_CONNECTIONS:
            if a in pts and b in pts:
                cv.line(frame, pts[a], pts[b], (0, 255, 0), 2, cv.LINE_AA)

        # Draw landmark dots
        for pt in pts.values():
            cv.circle(frame, pt, 5, (0, 255, 255), cv.FILLED)


# ---------------------------------------------------------------------------
# Convenience wrapper (mirrors the old pose_estimator_in_video function)
# ---------------------------------------------------------------------------
def pose_estimator_in_video(video_path, filename, resizing_factor, save_video=False):
    cap = cv.VideoCapture(0 if video_path == 0 else video_path)
    if not cap.isOpened():
        print("Couldn't capture the video file")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error reading the video file.")
        return

    frame_height  = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_width   = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    fps           = int(cap.get(cv.CAP_PROP_FPS))

    detector = PoseDetector()

    if save_video:
        video_dir       = os.path.dirname(filename) if os.path.dirname(filename) else "."
        os.makedirs(video_dir, exist_ok=True)
        fourcc          = cv.VideoWriter_fourcc(*'mp4v')
        out             = cv.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))

    resized_frame_size = (int(resizing_factor * frame_width),
                          int(resizing_factor * frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame     = detector.find_pose(frame)
        landmarks = detector.get_positions(frame)

        if save_video:
            out.write(frame)

        if video_path == 0:
            frame = cv.flip(frame, 1)

        resized_frame = cv.resize(frame, resized_frame_size)
        cv.imshow('Video', resized_frame)
        if cv.waitKey(1) & 0xff == ord('p'):
            break

    cap.release()
    if save_video:
        out.release()
    cv.destroyAllWindows()


def main():
    video_path     = input("Enter video path or '0' for webcam: ").strip().strip('"').strip("'")
    resizing_factor = float(input("Enter resizing factor (e.g., 0.5 for 50%): "))
    save_video     = input("Do you want to save the video? (yes or no): ").lower() == 'yes'

    filename = None
    if save_video:
        filename = input("Enter filename to save the video: ")

    if video_path == '0':
        video_path = 0

    print(f"Video path is: {video_path}")
    pose_estimator_in_video(video_path, filename, resizing_factor, save_video)


if __name__ == "__main__":
    main()