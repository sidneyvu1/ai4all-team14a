"""Gradio webcam UI: live face-landmark overlay + timestamped emotion-peaks graph.

Each frame's MediaPipe blendshapes are pushed into a rolling time window and
max-pooled before being fed to emotion_intensity_regressor_live.joblib, mirroring
how training clips were max-pooled (src/models/model.py's pool_features) rather
than predicting from a single noisy frame. Predictions are accumulated per
browser session (gr.State, so concurrent viewers don't share a history) into a
timestamped intensity-over-time plot, with detected peaks marked -- this is the
actual product deliverable: a timestamped graph of emotional peaks for an ad.

This is the third pass on the UI, iterating on (not replacing) the previous
polish pass. Layout/navigation are unchanged: webcam left, landmarks right,
predictions, graph, reset button. New in this pass: a real app title, a
corrected "dominant emotion" definition (see note below), a tracking-confidence
indicator, an FPS readout, session export (CSV + graph PNG), a light/dark
theme toggle, and further CSS polish. The core pipeline (landmark detection ->
blendshape pooling -> regressor call -> history/peaks) is still untouched.
"""

import csv
import os
import tempfile
import time
from collections import deque
from pathlib import Path

# Set environment variables for headless/server environment before importing MediaPipe
# This prevents MediaPipe from trying to use GPU acceleration or graphics libraries
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import cv2
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import mediapipe as mp
import numpy as np
from joblib import load
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)
from scipy.signal import find_peaks

try:
    import spaces
except ImportError:  # not running on a Hugging Face Space
    spaces = None


@(spaces.GPU if spaces else (lambda fn: fn))
def _zerogpu_dummy():
    """Unused. Its only purpose is the @spaces.GPU decorator: this app is
    CPU-only, but Hugging Face's free tier requires ZeroGPU hardware, which
    in turn requires at least one @spaces.GPU function to exist -- without
    this, the Space fails its startup check. Everything else still runs on
    the normal CPU allocation; this function is never called."""
    return None


# Real app name (was the placeholder "EmotionPulse"). Still a single
# constant so it's a one-line change if the team renames the product later.
APP_TITLE = "Real-Time Emotion Analysis"
APP_SUBTITLE = (
    "Live webcam feed with facial-landmark tracking, real-time emotion "
    "intensity, and a timestamped graph of emotional peaks for this session."
)

EMOTIONS = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
EMOTION_COLORS = {
    "happy": "#2ca02c",
    "sad": "#1f77b4",
    "anger": "#d62728",
    "surprise": "#ff7f0e",
    "disgust": "#8c564b",
    "fear": "#9467bd",
}
EMOTION_EMOJI = {
    "happy": "🙂", "sad": "😢", "anger": "😠",
    "surprise": "😮", "disgust": "🤢", "fear": "😨",
}

LANDMARKER_MODEL_PATH = Path(__file__).parent / "assets" / "face_landmarker.task"
REGRESSOR_PATH = (
    Path(__file__).parents[1] / "models" / "artifacts" / "emotion_intensity_regressor_live.joblib"
)

# --- Unchanged from prior versions ------------------------------------------
WINDOW_SECONDS = 0.5

# How often Gradio invokes process_frame, decoupled from WINDOW_SECONDS: the
# pooling window can stay wide for robustness while the stream itself runs
# much faster, since each call only has to re-pool a small deque rather than
# rebuild anything expensive.
STREAM_EVERY_SECONDS = 0.1

# Rebuilding the peaks figure (matplotlib figure creation + legend + layout)
# costs ~150-200ms regardless of history size -- far more than detect+predict
# combined -- so it's driven by its own timer instead of every stream tick.
PLOT_REFRESH_SECONDS = 1.0
PEAK_MIN_HEIGHT = 1.0
PEAK_MIN_PROMINENCE = 0.3
CALIBRATION_SECONDS = 1.0
FACE_LOST_GRACE_SECONDS = 1.0

# --- New in this pass ---------------------------------------------------
# How many recent frames feed the tracking-confidence and FPS readouts. Small
# and fixed-size (a deque with maxlen auto-evicts), so this stays cheap --
# no unbounded growth over a long session, unlike `history` which is meant
# to keep every prediction.
ROLLING_WINDOW_FRAMES = 30

# `history` is capped to the most recent 30 minutes of predictions (as a
# fixed-size deque) rather than growing forever -- past that horizon every
# per-tick summary/plot rebuild would otherwise keep getting more expensive
# for the life of an unattended session, with no benefit since the UI only
# ever shows/exports "this session so far".
MAX_HISTORY_SECONDS = 30 * 60
MAX_HISTORY_FRAMES = int(MAX_HISTORY_SECONDS / STREAM_EVERY_SECONDS)


def _new_history():
    return deque(maxlen=MAX_HISTORY_FRAMES)


def _new_landmarker():
    # Use IMAGE mode instead of VIDEO mode to avoid graphics library dependencies
    # (libGLESv2.so.2) that aren't available in headless environments like HF Spaces.
    # IMAGE mode doesn't exploit temporal coherence between frames, but it's
    # compatible with headless servers and requires no graphics acceleration.
    try:
        print(f"Initializing FaceLandmarker (IMAGE mode) with model at {LANDMARKER_MODEL_PATH}", flush=True)
        return FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(LANDMARKER_MODEL_PATH)),
                running_mode=RunningMode.IMAGE,
                num_faces=1,
                output_face_blendshapes=True,
            )
        )
    except Exception as e:
        print(f"ERROR initializing FaceLandmarker: {e}", flush=True)
        print(f"Model path exists: {LANDMARKER_MODEL_PATH.exists()}", flush=True)
        if LANDMARKER_MODEL_PATH.exists():
            print(f"Model size: {LANDMARKER_MODEL_PATH.stat().st_size} bytes", flush=True)
        import traceback
        traceback.print_exc()
        raise


# A FaceLandmarker is per-session state (created lazily in process_frame),
# not a module-level singleton: MediaPipe's detector isn't guaranteed
# thread-safe for concurrent calls, and a shared instance would let two
# simultaneous browser tabs interfere with each other's detections. The
# regressor stays shared -- RandomForestRegressor.predict() is a stateless,
# read-only operation, so concurrent calls across sessions are safe.
try:
    _regressor = load(REGRESSOR_PATH)
    # n_jobs=-1 (set at training time for fitting on the full dataset) makes
    # single-row inference *slower*: joblib pays multiprocess/thread dispatch
    # overhead on every predict() call that dwarfs the actual tree traversal.
    # Force in-process sequential prediction for live use (~75ms -> ~25ms/call).
    _regressor.n_jobs = 1
    print(f"Loaded regressor from {REGRESSOR_PATH}", flush=True)
except Exception as e:
    print(f"ERROR: Failed to load regressor from {REGRESSOR_PATH}: {e}", flush=True)
    import traceback
    traceback.print_exc()
    _regressor = None

# Verify landmarker model exists
if not LANDMARKER_MODEL_PATH.exists():
    print(f"ERROR: Landmarker model not found at {LANDMARKER_MODEL_PATH}", flush=True)
else:
    print(f"Landmarker model found at {LANDMARKER_MODEL_PATH} ({LANDMARKER_MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)


# =============================================================================
# Backend prediction pipeline (unchanged)
# =============================================================================

def _pool_recent_blendshapes(window, now):
    while window and now - window[0][0] > WINDOW_SECONDS:
        window.popleft()
    if not window:
        return None
    return np.max([scores for _, scores in window], axis=0, keepdims=True)


def draw_landmarks(frame, landmarks):
    height, width = frame.shape[:2]
    for landmark in landmarks:
        x, y = int(landmark.x * width), int(landmark.y * height)
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    return frame


# =============================================================================
# Presentation helpers
# =============================================================================

def build_bars_html(intensities):
    rows = []
    for emotion, value in zip(EMOTIONS, intensities):
        pct = max(0.0, min(100.0, (value / 3.0) * 100.0))
        color = EMOTION_COLORS[emotion]
        rows.append(f"""
        <div class="bar-row">
          <div class="bar-row-label">
            <span>{EMOTION_EMOJI[emotion]} {emotion.capitalize()}</span>
            <span class="bar-value">{value:.2f}</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
          </div>
        </div>""")
    return f'<div class="bars-panel">{"".join(rows)}</div>'


def build_status_html(state):
    if state == "calibrating":
        return '<div class="status-pill status-calibrating">● Calibrating…</div>'
    if state == "lost":
        return '<div class="status-pill status-lost">● Face tracking lost — showing last reading</div>'
    return '<div class="status-pill status-tracking">● Tracking</div>'


# --- Tracking confidence (new) ----------------------------------------------
# Explicitly NOT a model-confidence score -- RandomForestRegressor doesn't
# produce one. This instead reflects how trustworthy the *input* to that
# model currently is, from cheap signals already available in the pipeline:
#   - continuous face detection: recent per-frame hit rate
#   - blendshape completeness: same hit rate doubles as a proxy for this,
#     since a frame with no face also has no blendshape scores
#   - calibration status: still-calibrating is always reported as Low
#   - a face-loss longer than the grace period is always reported as Low
# A single detection-history deque covers "continuous detection" and
# "completeness" together rather than tracking them separately, since for
# this pipeline they're the same underlying event (was a face found?).
def compute_confidence(detection_history, still_calibrating, face_lost_duration):
    if still_calibrating:
        return "Low", 0.15
    if face_lost_duration > FACE_LOST_GRACE_SECONDS:
        return "Low", 0.15
    if not detection_history:
        return "Low", 0.2
    rate = sum(detection_history) / len(detection_history)
    if rate >= 0.9:
        return "High", rate
    if rate >= 0.6:
        return "Medium", rate
    return "Low", rate


def build_meta_html(confidence_label, confidence_ratio, fps):
    conf_class = f"conf-{confidence_label.lower()}"
    pct = max(4.0, min(100.0, confidence_ratio * 100.0))
    return f"""
    <div class="meta-row">
      <div class="meta-item">
        <span class="meta-label">Tracking confidence</span>
        <span class="conf-pill {conf_class}">{confidence_label}</span>
        <div class="conf-track"><div class="conf-fill {conf_class}" style="width:{pct:.0f}%;"></div></div>
      </div>
      <div class="meta-item meta-fps">
        <span class="meta-label">FPS</span>
        <span class="meta-fps-value">{fps:.1f}</span>
      </div>
    </div>"""


def compute_fps(frame_timestamps):
    if len(frame_timestamps) < 2:
        return 0.0
    span = frame_timestamps[-1] - frame_timestamps[0]
    if span <= 0:
        return 0.0
    return (len(frame_timestamps) - 1) / span


# --- Session summary ---------------------------------------------------------
# Dominant emotion: the original app (before any UI-polish pass) had no
# "dominant emotion" concept at all -- it was introduced in the last revision
# as "whichever emotion has the most detected peaks", which is really a
# volatility measure, not a dominance measure (an emotion with one huge spike
# could lose to one with lots of small blips). Switched to average intensity
# across the session, which is both more intuitive and gives this stat a
# distinct meaning from "highest intensity" (a peak) shown right next to it.
def compute_session_stats(history):
    if not history:
        return {"elapsed": 0.0, "highest": 0.0, "dominant": None, "total_peaks": 0}

    times = np.array([t for t, _ in history])
    values = np.array([v for _, v in history])
    elapsed = float(times[-1])
    highest = float(values.max())

    mean_intensity = values.mean(axis=0)
    dominant_idx = int(np.argmax(mean_intensity))
    dominant = EMOTIONS[dominant_idx] if mean_intensity[dominant_idx] > 0 else None

    total_peaks = 0
    for i in range(len(EMOTIONS)):
        peak_idx, _ = find_peaks(values[:, i], height=PEAK_MIN_HEIGHT, prominence=PEAK_MIN_PROMINENCE)
        total_peaks += len(peak_idx)

    return {"elapsed": elapsed, "highest": highest, "dominant": dominant, "total_peaks": total_peaks}


def build_summary_html(stats):
    minutes, seconds = divmod(int(stats["elapsed"]), 60)
    if stats["dominant"]:
        dominant_str = f'{EMOTION_EMOJI[stats["dominant"]]} {stats["dominant"].capitalize()}'
    else:
        dominant_str = "—"

    def stat(label, value):
        return f'<div class="stat"><div class="stat-value">{value}</div><div class="stat-label">{label}</div></div>'

    return (
        '<div class="summary-strip">'
        + stat("Dominant emotion (avg)", dominant_str)
        + stat("Elapsed", f"{minutes:02d}:{seconds:02d}")
        + stat("Highest intensity", f'{stats["highest"]:.2f}')
        + stat("Total peaks", stats["total_peaks"])
        + "</div>"
    )


def build_peaks_figure(history):
    # Built via Figure() directly (not plt.subplots()): plt.subplots() would
    # register the figure in pyplot's global figure manager, which never
    # gets cleaned up unless something calls plt.close() on it -- and this
    # runs once a second for the life of the session, so that leak would
    # accumulate one never-freed Figure per tick. A bare Figure() is never
    # registered with pyplot in the first place, so there's nothing to leak.
    plt.rcParams["font.family"] = "sans-serif"
    fig = Figure(figsize=(8, 3.4), dpi=110)
    ax = fig.subplots()
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")

    ax.set_xlabel("Time since session start (s)", fontsize=10, color="#444")
    ax.set_ylabel("Predicted intensity (0–3)", fontsize=10, color="#444")
    ax.set_ylim(0, 3.2)
    ax.grid(True, linestyle="--", linewidth=0.5, color="#ddd", alpha=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#ccc")
    ax.tick_params(colors="#555", labelsize=9)

    if not history:
        ax.set_xlim(0, 1)
        ax.set_title("Emotion intensity over time (no data yet)", fontsize=11, color="#333")
        fig.tight_layout()
        return fig

    times = np.array([t for t, _ in history])
    values = np.array([v for _, v in history])

    for i, emotion in enumerate(EMOTIONS):
        y = values[:, i]
        ax.plot(times, y, label=emotion, color=EMOTION_COLORS[emotion], linewidth=1.8, alpha=0.9)
        peak_idx, _ = find_peaks(y, height=PEAK_MIN_HEIGHT, prominence=PEAK_MIN_PROMINENCE)
        if len(peak_idx):
            ax.scatter(
                times[peak_idx], y[peak_idx],
                color=EMOTION_COLORS[emotion], marker="^", s=55, zorder=3,
                edgecolors="white", linewidths=0.6,
            )

    ax.set_title("Emotion intensity over session (▲ marks a detected peak)", fontsize=11, color="#333")
    ax.legend(loc="upper right", fontsize=8, ncol=3, frameon=False)
    fig.tight_layout()
    return fig


# =============================================================================
# Session export (new)
# =============================================================================

def export_session_csv(history):
    """Writes the full timestamped history to a temp CSV and hands the path
    back to the DownloadButton. No new state needed -- history already has
    everything."""
    if not history:
        return None
    fd, path = tempfile.mkstemp(suffix=".csv", prefix="emotion_session_")
    with open(fd, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_seconds"] + EMOTIONS)
        for t, values in history:
            writer.writerow([f"{t:.3f}"] + [f"{v:.4f}" for v in values])
    return path


def export_session_graph(history):
    """Rebuilds the graph from history (rather than caching the last Figure
    object in extra state) and saves it as a PNG for download. Cheap: this
    only runs once, on button click, not per-frame."""
    if not history:
        return None
    fig = build_peaks_figure(history)
    fd, path = tempfile.mkstemp(suffix=".png", prefix="emotion_session_graph_")
    os.close(fd)  # fig.savefig writes via `path`, not `fd` -- close it to avoid leaking the descriptor
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


# =============================================================================
# Frame processing (core pipeline unchanged; confidence/FPS tracking added)
# =============================================================================

def process_frame(frame, session_start, history, blendshape_window, last_face_seen,
                   frame_timestamps, detection_history, landmarker, last_video_timestamp_ms):
    if frame is None:
        return (
            None, gr.skip(), gr.skip(), gr.skip(),
            session_start, history, blendshape_window, last_face_seen,
            frame_timestamps, detection_history, landmarker, last_video_timestamp_ms,
        )

    try:
        if session_start is None:
            session_start = time.monotonic()
        if history is None:
            history = _new_history()
        if blendshape_window is None:
            blendshape_window = deque()
        if frame_timestamps is None:
            frame_timestamps = deque(maxlen=ROLLING_WINDOW_FRAMES)
        if detection_history is None:
            detection_history = deque(maxlen=ROLLING_WINDOW_FRAMES)
        if landmarker is None:
            # Created per session (not a module-level singleton) since a shared
            # FaceLandmarker isn't guaranteed thread-safe across concurrent
            # browser sessions -- see the note by _new_landmarker's definition.
            landmarker = _new_landmarker()
        if last_video_timestamp_ms is None:
            last_video_timestamp_ms = -1

        now = time.monotonic()
        frame_timestamps.append(now)  # for FPS -- cheap append/evict, no extra work per frame

        # Ensure frame is in the right format (RGB numpy array)
        if isinstance(frame, np.ndarray):
            if len(frame.shape) != 3 or frame.shape[2] not in (3, 4):
                print(f"Warning: Unexpected frame shape {frame.shape}, skipping", flush=True)
                return (
                    frame, gr.skip(), gr.skip(), gr.skip(),
                    session_start, history, blendshape_window, last_face_seen,
                    frame_timestamps, detection_history, landmarker, last_video_timestamp_ms,
                )
            # Convert BGR (OpenCV default) to RGB if needed
            if frame.dtype == np.uint8:
                # Frame is already in the right format
                pass
            else:
                # Normalize if it's float
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # IMAGE mode: no timestamp tracking needed, simpler and works on headless servers
        result = landmarker.detect(mp_image)
    except Exception as e:
        print(f"Error in process_frame: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return (
            frame, gr.skip(), gr.skip(), gr.skip(),
            session_start, history, blendshape_window, last_face_seen,
            frame_timestamps, detection_history, landmarker, last_video_timestamp_ms,
        )

    if result.face_landmarks:
        annotated = frame.copy()
        for face_landmarks in result.face_landmarks:
            annotated = draw_landmarks(annotated, face_landmarks)
    else:
        annotated = frame

    face_detected = bool(result.face_blendshapes)
    detection_history.append(face_detected)  # for tracking-confidence
    if face_detected:
        scores = np.array([c.score for c in result.face_blendshapes[0]])
        blendshape_window.append((now, scores))  # unchanged
        last_face_seen = now

    pooled = _pool_recent_blendshapes(blendshape_window, now)  # unchanged
    fps = compute_fps(frame_timestamps)

    still_calibrating = (now - session_start) < CALIBRATION_SECONDS
    face_lost_duration = now - last_face_seen if last_face_seen else float("inf")
    confidence_label, confidence_ratio = compute_confidence(
        detection_history, still_calibrating, face_lost_duration
    )
    meta_html = build_meta_html(confidence_label, confidence_ratio, fps)

    if still_calibrating:
        status_html = build_status_html("calibrating")
        return (
            annotated, gr.skip(), status_html, meta_html,
            session_start, history, blendshape_window, last_face_seen,
            frame_timestamps, detection_history, landmarker, last_video_timestamp_ms,
        )

    if pooled is None:
        if face_lost_duration > FACE_LOST_GRACE_SECONDS:
            status_html = build_status_html("lost")
            return (
                annotated, gr.skip(), status_html, meta_html,
                session_start, history, blendshape_window, last_face_seen,
                frame_timestamps, detection_history, landmarker, last_video_timestamp_ms,
            )
        return (
            annotated, gr.skip(), gr.skip(), meta_html,
            session_start, history, blendshape_window, last_face_seen,
            frame_timestamps, detection_history, landmarker, last_video_timestamp_ms,
        )

    if _regressor is None:
        # Regressor failed to load at startup
        intensities = np.zeros(len(EMOTIONS))
    else:
        intensities = _regressor.predict(pooled)[0]  # unchanged
    history.append((now - session_start, intensities))  # capped via _new_history()'s maxlen

    bars_html = build_bars_html(intensities)
    status_html = build_status_html("tracking")

    return (
        annotated, bars_html, status_html, meta_html,
        session_start, history, blendshape_window, last_face_seen,
        frame_timestamps, detection_history, landmarker, last_video_timestamp_ms,
    )


def refresh_summary(history):
    return build_summary_html(compute_session_stats(history))


def refresh_plot(history):
    """Runs on its own timer, fully decoupled from the webcam stream --
    bundling this into process_frame's return meant every stream tick's
    round trip periodically ballooned while the browser downloaded and
    rendered the plot image before it could capture the next frame."""
    return build_peaks_figure(history)


def reset_session(landmarker):
    # Close the old per-session landmarker's native resources up front
    # instead of waiting on garbage collection -- process_frame lazily
    # creates a fresh one (with its own fresh VIDEO-mode timestamp clock,
    # since last_video_timestamp_ms is also reset to None here) on the next
    # frame.
    if landmarker is not None:
        landmarker.close()
    empty_fig = build_peaks_figure(_new_history())
    return (
        None, _new_history(), None, None, None, None, None, None,
        # session_start, history, blendshape_window, last_face_seen,
        # frame_timestamps, detection_history, landmarker, last_video_timestamp_ms
        None,                                # landmarks image
        build_bars_html(np.zeros(len(EMOTIONS))),
        empty_fig,
        build_status_html("calibrating"),
        build_meta_html("Low", 0.15, 0.0),
        build_summary_html(compute_session_stats(_new_history())),
    )


# =============================================================================
# Layout -- same structure as the previous pass:
#   Header -> Row(webcam | landmarks) -> summary -> predictions -> graph -> actions
# New this pass: theme toggle in the header, confidence/FPS line under status,
# CSV/PNG export buttons alongside Reset. No component has moved out of its
# established section.
# =============================================================================

CUSTOM_CSS = """
.gradio-container { max-width: 1100px !important; margin: auto; }
#header-row { align-items: center; }
#header-block h1 { margin-bottom: 2px; }
#header-block p { color: var(--body-text-color-subdued); margin-top: 0; }

.card { background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
        border-radius: 14px; padding: 16px 18px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06); transition: box-shadow 0.18s ease; }
.card:hover { box-shadow: 0 3px 10px rgba(0,0,0,0.10); }

.summary-strip { display: flex; flex-wrap: wrap; justify-content: space-around;
                  align-items: center; gap: 10px; padding: 8px 4px; font-family: sans-serif; }
.summary-strip .stat { text-align: center; min-width: 90px; }
.summary-strip .stat-value { font-size: 18px; font-weight: 700; color: var(--body-text-color); }
.summary-strip .stat-label { font-size: 11px; color: var(--body-text-color-subdued);
                              text-transform: uppercase; letter-spacing: 0.04em; margin-top: 2px; }

.status-pill { display: inline-block; font-family: sans-serif; font-size: 12px;
               font-weight: 600; padding: 3px 10px; border-radius: 10px; margin-bottom: 8px; }
.status-tracking { background: #2ca02c26; color: #2ca02c; }
.status-calibrating { background: #ff7f0e26; color: #ff7f0e; }
.status-lost { background: #d6272826; color: #d62728; }

.meta-row { display: flex; gap: 22px; align-items: center; font-family: sans-serif;
            margin-bottom: 10px; flex-wrap: wrap; }
.meta-item { display: flex; align-items: center; gap: 6px; }
.meta-label { font-size: 11px; color: var(--body-text-color-subdued); text-transform: uppercase;
              letter-spacing: 0.03em; }
.conf-pill { font-size: 11px; font-weight: 700; padding: 1px 7px; border-radius: 8px; }
.conf-high { background: #2ca02c26; color: #2ca02c; }
.conf-medium { background: #ff7f0e26; color: #ff7f0e; }
.conf-low { background: #d6272826; color: #d62728; }
.conf-track { width: 46px; height: 6px; background: var(--border-color-primary);
              border-radius: 4px; overflow: hidden; }
.conf-fill { height: 100%; border-radius: 4px; transition: width 0.2s ease; }
.conf-fill.conf-high { background: #2ca02c; }
.conf-fill.conf-medium { background: #ff7f0e; }
.conf-fill.conf-low { background: #d62728; }
.meta-fps-value { font-variant-numeric: tabular-nums; font-size: 12px; font-weight: 600;
                   color: var(--body-text-color); }

.bars-panel { font-family: sans-serif; padding: 4px 2px; }
.bar-row { margin-bottom: 10px; }
.bar-row-label { display: flex; justify-content: space-between; font-size: 13px;
                  margin-bottom: 3px; color: var(--body-text-color); }
.bar-value { font-variant-numeric: tabular-nums; color: var(--body-text-color-subdued); }
.bar-track { background: var(--border-color-primary); border-radius: 6px; height: 10px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 6px; transition: width 0.15s ease-out; }

#actions-row { gap: 10px; }
#actions-row button { transition: transform 0.1s ease, box-shadow 0.1s ease; }
#actions-row button:hover { transform: translateY(-1px); box-shadow: 0 2px 6px rgba(0,0,0,0.12); }

#theme-toggle-btn { min-width: 40px; }
"""

THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")

with gr.Blocks(title=APP_TITLE, theme=THEME, css=CUSTOM_CSS) as demo:
    with gr.Row(elem_id="header-row"):
        with gr.Column(scale=6, elem_id="header-block"):
            gr.Markdown(f"# {APP_TITLE}")
            gr.Markdown(APP_SUBTITLE)
        with gr.Column(scale=1, min_width=60):
            # Toggles Gradio's own built-in `.dark` CSS-variable set (the
            # same mechanism behind the `?__theme=dark` URL param) via a
            # client-side-only JS handler -- no server round trip, no
            # restart, and every color above uses Gradio's CSS variables
            # (var(--body-text-color) etc.) rather than hardcoded hex, so it
            # stays consistent across both themes. This is the standard,
            # non-hacky way to do a runtime toggle in current Gradio; a true
            # first-class `gr.themes` runtime-switch API doesn't exist yet.
            theme_toggle_btn = gr.Button("🌓", elem_id="theme-toggle-btn", size="sm")

    session_start_state = gr.State(None)
    history_state = gr.State(_new_history())
    blendshape_window_state = gr.State(None)
    last_face_seen_state = gr.State(None)
    frame_timestamps_state = gr.State(None)
    detection_history_state = gr.State(None)
    landmarker_state = gr.State(None)
    last_video_timestamp_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1, elem_classes="card"):
            # Use simple webcam without constraints to maximize compatibility.
            # Gradio 6.20.0 may not support WebcamOptions, and Hugging Face Spaces
            # sometimes has issues with complex webcam configurations.
            cam = gr.Image(
                sources=["webcam"],
                streaming=True,
                type="numpy",
                label="Webcam",
            )
        with gr.Column(scale=1, elem_classes="card"):
            landmarks_out = gr.Image(type="numpy", label="Landmarks")

    with gr.Group(elem_classes="card"):
        summary_out = gr.HTML(value=build_summary_html(compute_session_stats([])))

    with gr.Group(elem_classes="card"):
        status_out = gr.HTML(value=build_status_html("calibrating"))
        meta_out = gr.HTML(value=build_meta_html("Low", 0.15, 0.0))
        bars_out = gr.HTML(value=build_bars_html(np.zeros(len(EMOTIONS))), label="Predicted emotion intensity")

    with gr.Group(elem_classes="card"):
        plot_out = gr.Plot(label="Emotional peaks this session")

    with gr.Row(elem_id="actions-row"):
        reset_btn = gr.Button("Reset session", variant="primary")
        export_csv_btn = gr.DownloadButton("⬇ Export CSV")
        export_png_btn = gr.DownloadButton("⬇ Export graph (PNG)")

    cam.stream(
        fn=process_frame,
        inputs=[
            cam, session_start_state, history_state, blendshape_window_state, last_face_seen_state,
            frame_timestamps_state, detection_history_state, landmarker_state, last_video_timestamp_state,
        ],
        outputs=[
            landmarks_out, bars_out, status_out, meta_out,
            session_start_state, history_state, blendshape_window_state, last_face_seen_state,
            frame_timestamps_state, detection_history_state, landmarker_state, last_video_timestamp_state,
        ],
        stream_every=STREAM_EVERY_SECONDS,
        # Every output of a .stream() handler gets Gradio's default "pending"
        # overlay (a translucent gray pulse) while that call is in flight.
        # At STREAM_EVERY_SECONDS cadence that overlay toggles on/off ~10x/sec,
        # which reads as constant flicker on bars_out/status_out/meta_out even
        # though the underlying values update instantly. Nothing here is slow
        # enough to need a busy indicator, so hide it instead of showing it on
        # every frame.
        show_progress="hidden",
    )

    summary_timer = gr.Timer(1.0)
    summary_timer.tick(fn=refresh_summary, inputs=[history_state], outputs=[summary_out])

    plot_timer = gr.Timer(PLOT_REFRESH_SECONDS)
    plot_timer.tick(fn=refresh_plot, inputs=[history_state], outputs=[plot_out])

    reset_btn.click(
        fn=reset_session,
        inputs=[landmarker_state],
        outputs=[
            session_start_state, history_state, blendshape_window_state, last_face_seen_state,
            frame_timestamps_state, detection_history_state, landmarker_state, last_video_timestamp_state,
            landmarks_out, bars_out, plot_out, status_out, meta_out, summary_out,
        ],
    )
    export_csv_btn.click(fn=export_session_csv, inputs=[history_state], outputs=[export_csv_btn])
    export_png_btn.click(fn=export_session_graph, inputs=[history_state], outputs=[export_png_btn])

    theme_toggle_btn.click(
        fn=None, inputs=None, outputs=None,
        js="() => { document.documentElement.classList.toggle('dark'); }",
    )

if __name__ == "__main__":
    demo.launch()