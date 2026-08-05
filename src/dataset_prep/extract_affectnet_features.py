"""Extracts MediaPipe blendshape features from the AffectNet subset.

AffectNet images are static photos, not clips, so each image needs only a
single IMAGE-mode blendshape detection -- no multi-frame max-pooling like
extract_cremad_features.py (a one-frame "clip" already max-pools to itself).
IMAGE mode has no timestamp/ordering constraint, so (unlike the VIDEO-mode
per-clip landmarker in extract_calibration_features.py) one landmarker
instance is reused across every image here.

AffectNet has no intensity levels (unlike CREMA-D's LO/MD/HI) -- a labeled
image just means "this emotion is present", so it's encoded as CMU-MOSEI's
top intensity (3), matching CREMA-D's "HI" convention. `contempt` isn't one
of the six modeled emotions and is skipped; `neutral` anchors the 0-intensity
baseline for all six, same as CREMA-D's NEU clips.

Each image's folder placement and labels.csv's own `label` column disagree
37% of the time, with no README to explain why -- consistent with AffectNet's
well-documented crowd-sourced label noise, not an error in this repackaging.
Rather than guess which source is right, only images where both agree are
used (17,705 of 28,175 labeled rows), the same "skip the ambiguous ones"
approach already used for CREMA-D's unspecified-intensity clips.
"""

import csv
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

# Override via AFFECTNET_DIR env var -- the default only exists on the
# original author's machine (kagglehub's local download cache).
AFFECTNET_DIR = os.environ.get(
    "AFFECTNET_DIR",
    r"C:\Users\andys\.cache\kagglehub\datasets\mstjebashazida"
    r"\affectnet\versions\1\archive (3)",
)
LABELS_CSV = os.path.join(AFFECTNET_DIR, "labels.csv")
MODEL_PATH = str(Path(__file__).resolve().parents[2] / "src" / "ui" / "assets" / "face_landmarker.task")
OUTPUT_PATH = "data/affectnet_features.npz"

EMOTIONS = ["happy", "sad", "anger", "surprise", "disgust", "fear"]
FULL_INTENSITY = 3


def _new_landmarker():
    return FaceLandmarker.create_from_options(
        FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
        )
    )


def load_agreeing_rows():
    """Rows where the folder-assigned label and labels.csv's label agree."""
    rows = []
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            folder_label = row["pth"].split("/")[0].lower()
            csv_label = row["label"].lower()
            if folder_label == csv_label:
                rows.append((row["pth"], csv_label))
    return rows


def resolve_path(rel_path):
    rel = rel_path.replace("/", os.sep)
    for split in ("Train", "Test"):
        candidate = os.path.join(AFFECTNET_DIR, split, rel)
        if os.path.isfile(candidate):
            return candidate
    return None


def label_to_vector(label):
    if label == "contempt":
        return None  # not one of the six modeled emotions
    y = np.zeros(len(EMOTIONS))
    if label == "neutral":
        return y
    if label not in EMOTIONS:
        return None
    y[EMOTIONS.index(label)] = FULL_INTENSITY
    return y


def main():
    rows = load_agreeing_rows()
    print(f"Found {len(rows)} agreeing-label images")

    landmarker = _new_landmarker()
    X, y, paths, feature_names = [], [], [], None
    skipped = 0
    for i, (rel_path, label) in enumerate(rows):
        vec = label_to_vector(label)
        if vec is None:
            skipped += 1
            continue

        full_path = resolve_path(rel_path)
        if full_path is None:
            skipped += 1
            continue

        img = cv2.imread(full_path)
        if img is None:
            skipped += 1
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_image)
        if not result.face_blendshapes:
            skipped += 1
            continue

        categories = result.face_blendshapes[0]
        feature_names = feature_names or [c.category_name for c in categories]
        X.append([c.score for c in categories])
        y.append(vec)
        paths.append(rel_path)

        if (i + 1) % 1000 == 0:
            print(f"  processed {i + 1}/{len(rows)} ({skipped} skipped so far)")

    X = np.array(X)
    y = np.array(y)
    print(f"\nFinal X shape: {X.shape}")
    print(f"Final y shape: {y.shape}")
    print(f"Skipped: {skipped}")

    np.savez(
        OUTPUT_PATH,
        X=X, y=y,
        paths=np.array(paths),
        feature_names=np.array(feature_names),
    )
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
