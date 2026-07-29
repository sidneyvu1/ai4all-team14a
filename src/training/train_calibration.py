"""Trains and evaluates the MediaPipe-blendshape calibration model.

Only 19 self-recorded clips exist, so there's no valid train/val/test split.
Leave-one-out cross-validation is used instead to get an honest read on
generalization before trusting this for the live app.
"""

import os

import numpy as np
from joblib import dump
from sklearn.metrics import mean_absolute_error

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.model import EMOTIONS, build_model

DATA_PATH = "data/calibration_features.npz"
MODEL_DIR = "src/models/artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_intensity_regressor_mediapipe.joblib")


def load_data(path=DATA_PATH):
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"], data["clip_names"]


def leave_one_out_eval(X, y, clip_names):
    n = len(X)
    preds = np.zeros_like(y)

    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        model = build_model()
        model.fit(X[train_idx], y[train_idx])
        preds[i] = model.predict(X[i:i + 1])[0]

    print("Leave-one-out predictions (true -> predicted, per emotion):\n")
    header = f"{'clip':22s}" + "".join(f"{e:>12s}" for e in EMOTIONS)
    print(header)
    for i, name in enumerate(clip_names):
        row = f"{name:22s}"
        for j in range(len(EMOTIONS)):
            row += f"{y[i, j]:.0f}->{preds[i, j]:<7.2f}".rjust(12)
        print(row)

    print("\nLeave-one-out MAE per emotion:")
    for j, emotion in enumerate(EMOTIONS):
        mae = mean_absolute_error(y[:, j], preds[:, j])
        print(f"  {emotion:10s}  MAE={mae:.3f}")
    print(f"  {'overall':10s}  MAE={mean_absolute_error(y, preds):.3f}")


def main():
    X, y, clip_names = load_data()

    leave_one_out_eval(X, y, clip_names)

    final_model = build_model()
    final_model.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    dump(final_model, MODEL_PATH)
    print(f"\nSaved final model (trained on all {len(X)} clips) to {MODEL_PATH}")


if __name__ == "__main__":
    main()
