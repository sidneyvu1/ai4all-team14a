"""Trains the final live-inference model: full CREMA-D actor set + the
self-recorded calibration clips + an AffectNet subset (static "in the wild"
photos, not posed/acted like CREMA-D).

AffectNet is included specifically to address two gaps found by testing the
live app against a real face: (1) `surprise` previously had zero held-out
data at all (CREMA-D has no surprise category; only 3 self-recorded training
examples existed) -- AffectNet contributes ~2,700 real surprise photos, so
surprise now gets a real held-out evaluation for the first time. (2) the live
app was found to read as "sad" for almost any live expression -- diagnosed as
a domain-shift problem: the model had only ever seen short, deliberately
*posed* CREMA-D clips from 91 actors, never ordinary, candid, naturally-lit
photos of unfamiliar faces. AffectNet's candid, in-the-wild images are a much
closer match to what a live webcam actually sees.

AffectNet has no intensity levels (unlike CREMA-D's LO/MD/HI); see
extract_affectnet_features.py for how "labeled present" is encoded, and for
why only the ~63% of images where AffectNet's folder label and its labels.csv
relabeling agree are used (the other 37% is unresolved label-noise, not
guessed at).

Held out by group, same actor-grouped 70/30 split as train_cremad.py for
CREMA-D. AffectNet photos have no shared identity to leak across a split
(each is an unrelated scraped photo), so each gets its own singleton group --
this reduces the same split mechanism to a plain per-image random 70/30 split
for AffectNet's portion, while still doing proper actor-level holdout for
CREMA-D. Calibration clips are excluded from the split entirely (always
train) since they exist to patch specific personal/surprise gaps, not to be
evaluated on.

Calibration clips are windowed rather than reduced to one max-pooled vector
per clip: test_windowed_augmentation.py found that slicing each clip into
overlapping 20-frame windows (all windows from a held-out clip excluded
together, so no leakage) cut leave-one-clip-out MAE from 0.415 to 0.286 and
fixed a backwards mild/moderate/strong ordering seen with whole-clip pooling.
It also matches the live app's own inference-time pooling, which maxes over a
short rolling window (ROLLING_WINDOW_FRAMES in app.py), not an entire clip.
CREMA-D and AffectNet are NOT windowed here: CREMA-D's raw per-frame scores
aren't cached (only pooling max is; windowing it would mean re-running
extract_cremad_features.py over all ~7,400 clips), and AffectNet is single
static images with no time axis to window.
"""

import os
import sys
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.model import EMOTIONS, build_model

CREMAD_PATH = "data/cremad_features_full.npz"
CALIBRATION_PATH = "data/calibration_features.npz"
AFFECTNET_PATH = "data/affectnet_features.npz"
MODEL_DIR = "src/models/artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_intensity_regressor_live.joblib")

# Held out by group, as a random 70/30 split rather than a handful of fixed
# IDs -- see train_cremad.py for why (the old 5-actor holdout gave some
# emotions as few as 6-9 held-out clips). Same seed/fraction as
# train_cremad.py. Hyperparameters are NOT re-tuned here -- train_cremad.py's
# grouped-CV grid search found tuning made no real difference (best config
# beat the current defaults by 0.003 R2, within noise, at 2.6x the model
# size) -- so this keeps model.py's defaults.
TEST_FRACTION = 0.30
SPLIT_SEED = 42

# See test_windowed_augmentation.py for why calibration clips are windowed
# instead of max-pooled whole -- must match that script's config exactly for
# the validated MAE improvement to carry over.
CALIBRATION_WINDOW_SIZE = 20
CALIBRATION_WINDOW_STRIDE = 10


def window_calibration_clips(raw_sequences, y):
    """Expand each clip's raw per-frame sequence into overlapping max-pooled windows."""
    X, Y = [], []
    for clip_idx, sequence in enumerate(raw_sequences):
        t = len(sequence)
        if t <= CALIBRATION_WINDOW_SIZE:
            windows = [sequence.max(axis=0)]
        else:
            windows = [
                sequence[start:start + CALIBRATION_WINDOW_SIZE].max(axis=0)
                for start in range(0, t - CALIBRATION_WINDOW_SIZE + 1, CALIBRATION_WINDOW_STRIDE)
            ]
        for window_feats in windows:
            X.append(window_feats)
            Y.append(y[clip_idx])
    return np.array(X), np.array(Y)


def load_merged():
    cremad = np.load(CREMAD_PATH, allow_pickle=True)
    calibration = np.load(CALIBRATION_PATH, allow_pickle=True)
    affectnet = np.load(AFFECTNET_PATH, allow_pickle=True)

    assert (
        list(cremad["feature_names"])
        == list(calibration["feature_names"])
        == list(affectnet["feature_names"])
    )

    calibration_X, calibration_y = window_calibration_clips(
        calibration["raw_sequences"], calibration["y"]
    )

    X = np.vstack([cremad["X"], calibration_X, affectnet["X"]])
    y = np.vstack([cremad["y"], calibration_y, affectnet["y"]])
    groups = np.concatenate([
        cremad["actor_ids"],
        np.array(["self"] * len(calibration_X)),
        np.array([f"affectnet_{i}" for i in range(len(affectnet["X"]))]),
    ])
    return X, y, groups


def split_by_group(X, y, groups):
    # "self" is deliberately excluded from the test candidate pool so
    # calibration clips always stay in train -- see module docstring.
    candidate_groups = sorted(set(groups) - {"self"})
    shuffled = np.random.RandomState(SPLIT_SEED).permutation(candidate_groups)
    n_test = round(len(shuffled) * TEST_FRACTION)
    test_groups = set(shuffled[:n_test])

    test_mask = np.isin(groups, list(test_groups))
    train_mask = ~test_mask
    return (
        X[train_mask], y[train_mask],
        X[test_mask], y[test_mask],
    )


def evaluate(model, X, y, split_name):
    preds = model.predict(X)
    print(f"\n{split_name} performance (n={len(X)}):")
    for i, emotion in enumerate(EMOTIONS):
        mae = mean_absolute_error(y[:, i], preds[:, i])
        r2 = r2_score(y[:, i], preds[:, i])
        print(f"  {emotion:10s}  MAE={mae:.3f}  R2={r2:.3f}")


def main():
    X, y, groups = load_merged()
    X_train, y_train, X_test, y_test = split_by_group(X, y, groups)
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    model = build_model()
    model.fit(X_train, y_train)

    evaluate(model, X_test, y_test, "Test (held-out actors + held-out AffectNet photos, final read)")

    os.makedirs(MODEL_DIR, exist_ok=True)
    dump(model, MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
