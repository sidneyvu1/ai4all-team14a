"""Trains the final live-inference model: CREMA-D (diverse actors, 5 emotions)
merged with the self-recorded calibration clips (adds `surprise`, which
CREMA-D doesn't have, plus a little personalization).

Evaluation only covers the 5 CREMA-D emotions, held out by actor exactly as
in train_cremad.py. `surprise` has no held-out data at all -- only 3 training
examples total (from the solo calibration set) -- so it is trained but
UNVERIFIED. Treat any live surprise predictions with real skepticism until
more surprise examples (ideally from multiple people) are available.
"""

import os
import sys
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.model import EMOTIONS, build_model

CREMAD_PATH = "data/cremad_features.npz"
CALIBRATION_PATH = "data/calibration_features.npz"
MODEL_DIR = "src/models/artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_intensity_regressor_live.joblib")

VAL_ACTORS = {"1013", "1091"}
TEST_ACTORS = {"1050", "1017", "1082"}


def load_merged():
    cremad = np.load(CREMAD_PATH, allow_pickle=True)
    calibration = np.load(CALIBRATION_PATH, allow_pickle=True)

    assert list(cremad["feature_names"]) == list(calibration["feature_names"])

    X = np.vstack([cremad["X"], calibration["X"]])
    y = np.vstack([cremad["y"], calibration["y"]])
    # calibration clips get a group id that never matches a held-out actor,
    # so they always stay in train.
    groups = np.concatenate([
        cremad["actor_ids"],
        np.array(["self"] * len(calibration["X"])),
    ])
    return X, y, groups


def split_by_group(X, y, groups):
    val_mask = np.isin(groups, list(VAL_ACTORS))
    test_mask = np.isin(groups, list(TEST_ACTORS))
    train_mask = ~val_mask & ~test_mask
    return (
        X[train_mask], y[train_mask],
        X[val_mask], y[val_mask],
        X[test_mask], y[test_mask],
    )


def evaluate(model, X, y, split_name):
    preds = model.predict(X)
    print(f"\n{split_name} performance (n={len(X)}):")
    for i, emotion in enumerate(EMOTIONS):
        if emotion == "surprise":
            continue  # no held-out data for this emotion -- see module docstring
        mae = mean_absolute_error(y[:, i], preds[:, i])
        r2 = r2_score(y[:, i], preds[:, i])
        print(f"  {emotion:10s}  MAE={mae:.3f}  R2={r2:.3f}")


def main():
    X, y, groups = load_merged()
    X_train, y_train, X_val, y_val, X_test, y_test = split_by_group(X, y, groups)
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    model = build_model()
    model.fit(X_train, y_train)

    evaluate(model, X_val, y_val, "Validation (held-out CREMA-D actors)")
    evaluate(model, X_test, y_test, "Test (held-out CREMA-D actors)")
    print("\nNOTE: 'surprise' has zero held-out test data (CREMA-D lacks it; "
          "only 3 self-recorded training examples exist) -- trained but unverified.")

    os.makedirs(MODEL_DIR, exist_ok=True)
    dump(model, MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
