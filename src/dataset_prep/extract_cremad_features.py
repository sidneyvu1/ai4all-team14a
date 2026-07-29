"""Extracts MediaPipe blendshape features from the full CREMA-D actor set.

CREMA-D clips are named <ActorID>_<Sentence>_<Emotion>_<Intensity>.flv, where
Intensity in {LO, MD, HI} maps to CMU-MOSEI's 0-3 scale (1/2/3), XX means
"unspecified" and is skipped for non-neutral emotions (ambiguous target
level), and NEU clips anchor the 0-intensity baseline for all six emotions.

This mirrors src/dataset_prep/extract_calibration_features.py. An earlier
version of this script used a hand-picked 15-actor subset (chosen for
demographic diversity via VideoDemographics.csv) instead of a single
self-recorded person, which fixed the backwards intensity ordering seen on
the tiny solo calibration set. This version uses all 91 actors instead: with
only 15 actors, each emotion's held-out val/test set had as few as 6-9
clips -- too small to tell "the model is bad at this emotion" apart from
"the eval sample is too small to measure it". All 91 actors are already
downloaded locally, so this needs no new recording.
"""

import os
import re

import numpy as np

from extract_calibration_features import (
    EMOTIONS,
    extract_frame_scores,
)

CREMAD_DIR = (
    r"C:\Users\andys\.cache\kagglehub\datasets\orvile"
    r"\crema-d-emotional-multimodal-dataset\versions\1\content\CREMA-D\VideoFlash"
)
OUTPUT_PATH = "data/cremad_features_full.npz"

EMOTION_CODE_MAP = {
    "ANG": "anger", "DIS": "disgust", "FEA": "fear",
    "HAP": "happy", "SAD": "sad",
}
INTENSITY_CODE_MAP = {"LO": 1, "MD": 2, "HI": 3}


def parse_label(filename):
    """Returns (label_vector, actor_id) or (None, actor_id) to skip the clip."""
    stem = os.path.splitext(filename)[0]
    match = re.match(r"(\d{4})_(\w+)_(ANG|DIS|FEA|HAP|SAD|NEU)_(LO|MD|HI|XX|X)$", stem)
    if not match:
        raise ValueError(f"Unexpected CREMA-D filename: {filename}")
    actor_id, _sentence, emotion_code, intensity_code = match.groups()

    y = np.zeros(len(EMOTIONS))
    if emotion_code == "NEU":
        return y, actor_id

    if intensity_code not in INTENSITY_CODE_MAP:
        return None, actor_id  # unspecified intensity on a non-neutral emotion: skip

    y[EMOTIONS.index(EMOTION_CODE_MAP[emotion_code])] = INTENSITY_CODE_MAP[intensity_code]
    return y, actor_id


def main():
    files = sorted(f for f in os.listdir(CREMAD_DIR) if f.endswith(".flv"))
    actor_count = len({f.split("_")[0] for f in files})
    print(f"Found {len(files)} clips for {actor_count} actors")

    X, y, clip_names, actor_ids, feature_names = [], [], [], [], None
    skipped = 0
    for i, fname in enumerate(files):
        label, actor_id = parse_label(fname)
        if label is None:
            skipped += 1
            continue

        path = os.path.join(CREMAD_DIR, fname)
        try:
            frame_scores, names = extract_frame_scores(path)
        except RuntimeError:
            skipped += 1
            continue
        feature_names = feature_names or names

        X.append(frame_scores.max(axis=0))
        y.append(label)
        clip_names.append(fname)
        actor_ids.append(actor_id)

        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(files)} ({skipped} skipped so far)")

    X = np.array(X)
    y = np.array(y)

    print(f"\nFinal X shape: {X.shape}")
    print(f"Final y shape: {y.shape}")
    print(f"Skipped: {skipped}")

    np.savez(
        OUTPUT_PATH,
        X=X, y=y,
        clip_names=np.array(clip_names),
        actor_ids=np.array(actor_ids),
        feature_names=np.array(feature_names),
    )
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
