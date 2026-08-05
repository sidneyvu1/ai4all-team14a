"""Trains and evaluates the MediaPipe-blendshape model on the CREMA-D subset.

Split by actor (not by clip) so no person's clips leak across train/val/test --
the only way to honestly test whether this generalizes to a face the model
has never seen, rather than just memorizing a familiar face.

Hyperparameters are tuned via actor-grouped cross-validation *within* the 70%
train split, never against the 30% test split -- so test stays a single,
honest final read instead of something repeatedly peeked at during tuning.

Note: CREMA-D has no "surprise" category, so that column is all-zero here --
it isn't evaluated by this script.
"""

import os
import sys
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.model import EMOTIONS, build_model

DATA_PATH = "data/cremad_features_full.npz"
MODEL_DIR = "src/models/artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_intensity_regressor_cremad.joblib")

EVAL_INDICES = [i for i, e in enumerate(EMOTIONS) if e != "surprise"]  # not present in CREMA-D

# Held out by actor as a random 70/30 split rather than a handful of fixed
# IDs: with only the old 15-actor subset, a handful of held-out actors gave
# some emotions (e.g. fear) as few as 6-9 held-out clips -- too small to tell
# "the model is bad at this" apart from "the eval sample is too small to
# measure it". A fixed seed keeps the split reproducible.
TEST_FRACTION = 0.30
SPLIT_SEED = 42

# Deliberately small: an earlier 3x3 grid (150/250/400 x 15/20/30, 9 configs)
# found the best config beat model.py's defaults (150, 15) by only 0.003 R2 --
# within noise, at 2.6x the model size (see train_live_model.py, which skips
# retuning entirely on that basis). This grid exists to confirm that finding
# still holds if the data changes, not to search for a real improvement.
HYPERPARAM_GRID = [
    {"n_estimators": n, "max_depth": d}
    for n in (150, 250)
    for d in (15, 20)
]


def load_data():
    data = np.load(DATA_PATH, allow_pickle=True)
    return data["X"], data["y"], data["actor_ids"], data["feature_names"]


def split_by_actor(X, y, actor_ids):
    actor_ids = np.array(actor_ids)
    unique_actors = sorted(set(actor_ids))
    shuffled = np.random.RandomState(SPLIT_SEED).permutation(unique_actors)
    n_test = round(len(shuffled) * TEST_FRACTION)
    test_actors = set(shuffled[:n_test])

    test_mask = np.isin(actor_ids, list(test_actors))
    train_mask = ~test_mask
    return (
        X[train_mask], y[train_mask], actor_ids[train_mask],
        X[test_mask], y[test_mask],
    )


def mean_r2(y_true, y_pred):
    """R2 averaged over the 5 real CREMA-D emotions (excludes the all-zero surprise column)."""
    return r2_score(y_true[:, EVAL_INDICES], y_pred[:, EVAL_INDICES], multioutput="uniform_average")


def tune_hyperparameters(X_train, y_train, actor_ids_train):
    """Actor-grouped CV over HYPERPARAM_GRID, scored on the 5 real emotions."""
    n_splits = min(5, len(set(actor_ids_train)))
    folds = list(GroupKFold(n_splits=n_splits).split(X_train, y_train, groups=actor_ids_train))

    best_params, best_score = None, -np.inf
    for params in HYPERPARAM_GRID:
        fold_scores = []
        for fold_train_idx, fold_val_idx in folds:
            model = build_model(**params)
            model.fit(X_train[fold_train_idx], y_train[fold_train_idx])
            fold_scores.append(mean_r2(y_train[fold_val_idx], model.predict(X_train[fold_val_idx])))
        mean_score = float(np.mean(fold_scores))
        print(f"  {params}  CV mean R2={mean_score:.3f}")
        if mean_score > best_score:
            best_params, best_score = params, mean_score

    print(f"\nBest hyperparameters: {best_params} (CV mean R2={best_score:.3f})")
    return best_params


def evaluate(model, X, y, split_name):
    preds = model.predict(X)
    print(f"\n{split_name} performance (n={len(X)}):")
    for i, emotion in enumerate(EMOTIONS):
        if emotion == "surprise":
            continue  # not present in CREMA-D
        mae = mean_absolute_error(y[:, i], preds[:, i])
        r2 = r2_score(y[:, i], preds[:, i])
        print(f"  {emotion:10s}  MAE={mae:.3f}  R2={r2:.3f}")


def report_feature_importance(model, X_test, y_test, feature_names, emotion, top_n=8):
    """Permutation importance for one emotion's output column, on the held-out test set."""
    idx = EMOTIONS.index(emotion)
    baseline_r2 = r2_score(y_test[:, idx], model.predict(X_test)[:, idx])

    rng = np.random.RandomState(SPLIT_SEED)
    drops = []
    for feat_idx in range(X_test.shape[1]):
        X_perm = X_test.copy()
        X_perm[:, feat_idx] = rng.permutation(X_perm[:, feat_idx])
        perm_r2 = r2_score(y_test[:, idx], model.predict(X_perm)[:, idx])
        drops.append(baseline_r2 - perm_r2)

    order = np.argsort(drops)[::-1][:top_n]
    print(f"\nTop features for '{emotion}' (R2 drop when permuted, baseline R2={baseline_r2:.3f}):")
    for feat_idx in order:
        print(f"  {feature_names[feat_idx]:20s}  {drops[feat_idx]:+.3f}")


def main():
    X, y, actor_ids, feature_names = load_data()
    X_train, y_train, actor_ids_train, X_test, y_test = split_by_actor(X, y, actor_ids)
    print(f"Train: {len(X_train)} ({len(set(actor_ids_train))} actors)  Test: {len(X_test)}")

    print("\nTuning hyperparameters via actor-grouped CV on the train split only...")
    best_params = tune_hyperparameters(X_train, y_train, actor_ids_train)

    model = build_model(**best_params)
    model.fit(X_train, y_train)

    evaluate(model, X_test, y_test, "Test (held-out actors, final read)")

    for emotion in ("fear", "sad"):
        report_feature_importance(model, X_test, y_test, feature_names, emotion)

    os.makedirs(MODEL_DIR, exist_ok=True)
    dump(model, MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
