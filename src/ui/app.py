"""Gradio webcam UI: live face-landmark overlay + timestamped emotion-peaks graph.

Each frame's MediaPipe blendshapes are pushed into a rolling time window and
max-pooled before being fed to emotion_intensity_regressor_live.joblib, mirroring
how training clips were max-pooled (src/models/model.py's pool_features) rather
than predicting from a single noisy frame. Predictions are accumulated per
browser session (gr.State, so concurrent viewers don't share a history) into a
timestamped intensity-over-time plot, with detected peaks marked -- this is the
actual product deliverable: a timestamped graph of emotional peaks for an ad.
"""

import time
from collections import deque
from pathlib import Path

import cv2
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

EMOTIONS = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
EMOTION_COLORS = {
    "happy": "#2ca02c",
    "sad": "#1f77b4",
    "anger": "#d62728",
    "surprise": "#ff7f0e",
    "disgust": "#8c564b",
    "fear": "#9467bd",
}

LANDMARKER_MODEL_PATH = Path(__file__).parent / "assets" / "face_landmarker.task"
REGRESSOR_PATH = (
    Path(__file__).parents[1] / "models" / "artifacts" / "emotion_intensity_regressor_live.joblib"
)

# Window is measured in wall-clock seconds, not frame count: Gradio's webcam
# stream delivers frames at an unpredictable, browser/hardware-dependent rate,
# so a frame-count window represents a different real duration on every
# machine (and can drift mid-session if frames drop). 0.5s is a first-pass
# default sized to ride out brief tracking dropouts (blinks, momentary
# occlusion) and frame-to-frame jitter while staying short enough that a
# displayed peak still lines up closely with when it actually happened -- a
# longer window trades that timing precision for more robustness. Worth
# re-tuning once the peaks graph makes that trade-off visible.
WINDOW_SECONDS = 0.5

# A local maximum in an emotion's intensity-over-time series only gets marked
# as a "peak" on the graph if it clears both thresholds -- height alone would
# flag e.g. a sustained mild-but-noisy signal as a string of peaks.
PEAK_MIN_HEIGHT = 1.0
PEAK_MIN_PROMINENCE = 0.3

_landmarker = FaceLandmarker.create_from_options(
    FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(LANDMARKER_MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
    )
)
_regressor = load(REGRESSOR_PATH)


def _pool_recent_blendshapes(window, now):
    """Evict entries older than WINDOW_SECONDS, then max-pool what remains."""
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


def build_peaks_figure(history):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.set_xlabel("Time since session start (s)")
    ax.set_ylabel("Predicted intensity (0-3)")
    ax.set_ylim(0, 3.2)

    if not history:
        ax.set_xlim(0, 1)
        ax.set_title("Emotion intensity over time (no data yet)")
        fig.tight_layout()
        return fig

    times = np.array([t for t, _ in history])
    values = np.array([v for _, v in history])

    for i, emotion in enumerate(EMOTIONS):
        y = values[:, i]
        ax.plot(times, y, label=emotion, color=EMOTION_COLORS[emotion], linewidth=1.5)

        peak_idx, _ = find_peaks(y, height=PEAK_MIN_HEIGHT, prominence=PEAK_MIN_PROMINENCE)
        if len(peak_idx):
            ax.scatter(
                times[peak_idx], y[peak_idx],
                color=EMOTION_COLORS[emotion], marker="^", s=50, zorder=3,
            )

    ax.set_title("Emotion intensity over session (^ marks a detected peak)")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    fig.tight_layout()
    return fig


def process_frame(frame, session_start, history, blendshape_window):
    if frame is None:
        return None, None, gr.skip(), session_start, history, blendshape_window

    if session_start is None:
        session_start = time.monotonic()
    if blendshape_window is None:
        blendshape_window = deque()

    now = time.monotonic()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = _landmarker.detect(mp_image)

    annotated = frame.copy()
    for face_landmarks in result.face_landmarks:
        annotated = draw_landmarks(annotated, face_landmarks)

    if result.face_blendshapes:
        scores = np.array([c.score for c in result.face_blendshapes[0]])
        blendshape_window.append((now, scores))

    # Eviction runs every call (not just when a face is detected), so a
    # dropout longer than WINDOW_SECONDS correctly empties the window instead
    # of holding a stale prediction indefinitely.
    pooled = _pool_recent_blendshapes(blendshape_window, now)
    if pooled is None:
        return annotated, None, gr.skip(), session_start, history, blendshape_window

    intensities = _regressor.predict(pooled)[0]
    history.append((now - session_start, intensities))

    label = dict(zip(EMOTIONS, intensities.tolist()))
    fig = build_peaks_figure(history)
    return annotated, label, fig, session_start, history, blendshape_window


def reset_session():
    return None, [], None, None, None, build_peaks_figure([])


with gr.Blocks(title="Neuromarketing Ad-Testing — Live Emotion Intensity") as demo:
    gr.Markdown("# Neuromarketing Ad-Testing — Live Emotion Intensity")
    gr.Markdown(
        "Live webcam feed with MediaPipe face landmarks, predicted emotion "
        "intensity, and a timestamped graph of emotional peaks for this session."
    )

    session_start_state = gr.State(None)
    history_state = gr.State([])
    blendshape_window_state = gr.State(None)

    with gr.Row():
        cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam")
        landmarks_out = gr.Image(type="numpy", label="Landmarks")

    label_out = gr.Label(label="Predicted emotion intensity (0-3 scale)")
    plot_out = gr.Plot(label="Emotional peaks this session")
    reset_btn = gr.Button("Reset session")

    cam.stream(
        fn=process_frame,
        inputs=[cam, session_start_state, history_state, blendshape_window_state],
        outputs=[
            landmarks_out, label_out, plot_out,
            session_start_state, history_state, blendshape_window_state,
        ],
        stream_every=WINDOW_SECONDS,
    )
    reset_btn.click(
        fn=reset_session,
        outputs=[
            session_start_state, history_state, blendshape_window_state,
            landmarks_out, label_out, plot_out,
        ],
    )

if __name__ == "__main__":
    demo.launch()
