"""Trains the emotion-intensity regressor on the CMU-MOSEI OpenFace2 splits."""

import os
import sys
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.model import EMOTIONS, build_model, pool_features

DATA_PATH = "data/mosei_video_splits.npz"
MODEL_DIR = "src/models/artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_intensity_regressor.joblib")


def load_splits(path=DATA_PATH):
    data = np.load(path)
    return (
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        data["X_test"], data["y_test"],
    )


def evaluate(model, X, y, split_name):
    preds = model.predict(X)
    print(f"\n{split_name} performance:")
    for i, emotion in enumerate(EMOTIONS):
        mae = mean_absolute_error(y[:, i], preds[:, i])
        r2 = r2_score(y[:, i], preds[:, i])
        print(f"  {emotion:10s}  MAE={mae:.3f}  R2={r2:.3f}")
    print(f"  {'overall':10s}  MAE={mean_absolute_error(y, preds):.3f}")


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()

    X_train_pooled = pool_features(X_train)
    X_val_pooled = pool_features(X_val)
    X_test_pooled = pool_features(X_test)

    model = build_model()
    model.fit(X_train_pooled, y_train)

    evaluate(model, X_val_pooled, y_val, "Validation")
    evaluate(model, X_test_pooled, y_test, "Test")

    os.makedirs(MODEL_DIR, exist_ok=True)
    dump(model, MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
