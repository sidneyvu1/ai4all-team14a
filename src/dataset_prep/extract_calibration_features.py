"""Extracts MediaPipe blendshape features from the self-recorded calibration clips.

Each clip in data/calibration/ is named <emotion>_<level>.mp4 (or neutral.mp4),
which doubles as its label: level in {mild, moderate, strong} maps to intensity
{1, 2, 3} on CMU-MOSEI's 0-3 scale, matching every other emotion column to 0.
"""

import os
import re

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

CALIBRATION_DIR = "data/calibration"
MODEL_PATH = "src/ui/assets/face_landmarker.task"
OUTPUT_PATH = "data/calibration_features.npz"

EMOTIONS = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
LEVEL_TO_INTENSITY = {"mild": 1, "moderate": 2, "strong": 3}

def _new_landmarker():
    return FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        )
    )


def parse_label(filename):
    """Returns a (N_EMOTIONS,) intensity vector from a <emotion>_<level>.mp4 filename."""
    stem = os.path.splitext(filename)[0]
    y = np.zeros(len(EMOTIONS))
    if stem == "neutral":
        return y
    match = re.match(r"(\w+)_(mild|moderate|strong)$", stem)
    if not match:
        raise ValueError(f"Filename doesn't match <emotion>_<level>.mp4 convention: {filename}")
    emotion, level = match.groups()
    y[EMOTIONS.index(emotion)] = LEVEL_TO_INTENSITY[level]
    return y


def extract_frame_scores(path):
    """Runs blendshape detection on every frame of a clip -> (T, 52) raw sequence.

    Uses a fresh FaceLandmarker per clip: VIDEO mode requires strictly
    increasing timestamps for the lifetime of the instance, and each clip
    restarts its own timeline at 0.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frame_scores = []
    names = None
    frame_idx = 0
    with _new_landmarker() as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(frame_idx * (1000 / fps))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.face_blendshapes:
                categories = result.face_blendshapes[0]
                if names is None:
                    names = [c.category_name for c in categories]
                frame_scores.append([c.score for c in categories])
            frame_idx += 1

    cap.release()

    if not frame_scores:
        raise RuntimeError(f"No face detected in any frame of {path}")

    return np.array(frame_scores), names


def extract_clip_features(path):
    """Per-clip feature: max-pool the full raw frame sequence -> (52,)."""
    frame_scores, names = extract_frame_scores(path)
    return frame_scores.max(axis=0), names


def main():
    files = sorted(f for f in os.listdir(CALIBRATION_DIR) if f.endswith(".mp4"))

    X, y, clip_names, raw_sequences, feature_names = [], [], [], [], None
    for fname in files:
        path = os.path.join(CALIBRATION_DIR, fname)
        print(f"Processing {fname}...")
        frame_scores, names = extract_frame_scores(path)
        feature_names = feature_names or names

        X.append(frame_scores.max(axis=0))
        y.append(parse_label(fname))
        clip_names.append(fname)
        raw_sequences.append(frame_scores)

    X = np.array(X)
    y = np.array(y)

    print(f"\nFinal X shape: {X.shape}")
    print(f"Final y shape: {y.shape}")

    np.savez(
        OUTPUT_PATH,
        X=X, y=y,
        clip_names=np.array(clip_names),
        feature_names=np.array(feature_names),
        raw_sequences=np.array(raw_sequences, dtype=object),
    )
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
