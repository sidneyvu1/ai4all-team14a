"""Trains and evaluates the MediaPipe-blendshape model on the CREMA-D subset.

Split by actor (not by clip) so no person's clips leak across train/val/test --
the only way to honestly test whether this generalizes to a face the model
has never seen, rather than just memorizing a familiar face.

Note: CREMA-D has no "surprise" category, so that column is all-zero here --
it isn't evaluated by this script.
"""

import os
import sys
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.model import EMOTIONS, build_model

DATA_PATH = "data/cremad_features.npz"
MODEL_DIR = "src/models/artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_intensity_regressor_cremad.joblib")

# Held out entirely by actor so results reflect generalization to new faces.
VAL_ACTORS = {"1013", "1091"}
TEST_ACTORS = {"1050", "1017", "1082"}


def load_data():
    data = np.load(DATA_PATH, allow_pickle=True)
    return data["X"], data["y"], data["clip_names"], data["actor_ids"]


def split_by_actor(X, y, actor_ids):
    actor_ids = np.array(actor_ids)
    val_mask = np.isin(actor_ids, list(VAL_ACTORS))
    test_mask = np.isin(actor_ids, list(TEST_ACTORS))
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
            continue  # not present in CREMA-D
        mae = mean_absolute_error(y[:, i], preds[:, i])
        r2 = r2_score(y[:, i], preds[:, i])
        print(f"  {emotion:10s}  MAE={mae:.3f}  R2={r2:.3f}")


def main():
    X, y, clip_names, actor_ids = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = split_by_actor(X, y, actor_ids)
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    model = build_model()
    model.fit(X_train, y_train)

    evaluate(model, X_val, y_val, "Validation (held-out actors)")
    evaluate(model, X_test, y_test, "Test (held-out actors)")

    os.makedirs(MODEL_DIR, exist_ok=True)
    dump(model, MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
