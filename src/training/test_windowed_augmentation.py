"""Diagnostic: does slicing each calibration clip into overlapping windows
(more training examples, no new recordings) fix the backwards LOOCV pattern
seen when training on one max-pooled vector per clip?

Evaluation is leave-one-CLIP-out (not leave-one-window-out): all windows from
a held-out clip are excluded from training together, so windows from the same
recording never leak across the train/test split.

CONCLUSION: yes -- overall LOOCV MAE dropped from 0.415 (train_calibration.py,
whole-clip pooling) to 0.286 with these exact window/stride settings, and the
backwards ordering resolved. Folded into the real training pipeline in
train_live_model.py (window_calibration_clips), which windows the calibration
slice of the merged live-model training set using this same WINDOW_SIZE/
STRIDE. This script is kept standalone since it also reports the full
leave-one-clip-out prediction table, which the merged pipeline doesn't.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.model import EMOTIONS, build_model

DATA_PATH = "data/calibration_features.npz"
WINDOW_SIZE = 20
STRIDE = 10


def make_windows(sequence, window_size=WINDOW_SIZE, stride=STRIDE):
    t = len(sequence)
    if t <= window_size:
        return [sequence.max(axis=0)]
    windows = []
    for start in range(0, t - window_size + 1, stride):
        windows.append(sequence[start:start + window_size].max(axis=0))
    return windows


def build_windowed_dataset(raw_sequences, y):
    X, Y, groups = [], [], []
    for clip_idx, sequence in enumerate(raw_sequences):
        for window_feats in make_windows(sequence):
            X.append(window_feats)
            Y.append(y[clip_idx])
            groups.append(clip_idx)
    return np.array(X), np.array(Y), np.array(groups)


def main():
    data = np.load(DATA_PATH, allow_pickle=True)
    y, clip_names, raw_sequences = data["y"], data["clip_names"], data["raw_sequences"]

    X_win, y_win, groups = build_windowed_dataset(raw_sequences, y)
    print(f"Expanded {len(clip_names)} clips into {len(X_win)} windows "
          f"({WINDOW_SIZE}-frame windows, stride {STRIDE})\n")

    n_clips = len(clip_names)
    clip_preds = np.zeros_like(y)

    for held_out in range(n_clips):
        train_mask = groups != held_out
        model = build_model()
        model.fit(X_win[train_mask], y_win[train_mask])

        held_out_windows = X_win[groups == held_out]
        clip_preds[held_out] = model.predict(held_out_windows).mean(axis=0)

    print("Leave-one-clip-out predictions (true -> predicted, per emotion):\n")
    header = f"{'clip':22s}" + "".join(f"{e:>12s}" for e in EMOTIONS)
    print(header)
    for i, name in enumerate(clip_names):
        row = f"{name:22s}"
        for j in range(len(EMOTIONS)):
            row += f"{y[i, j]:.0f}->{clip_preds[i, j]:<7.2f}".rjust(12)
        print(row)

    print("\nLeave-one-clip-out MAE per emotion:")
    for j, emotion in enumerate(EMOTIONS):
        mae = mean_absolute_error(y[:, j], clip_preds[:, j])
        print(f"  {emotion:10s}  MAE={mae:.3f}")
    print(f"  {'overall':10s}  MAE={mean_absolute_error(y, clip_preds):.3f}")


if __name__ == "__main__":
    main()
