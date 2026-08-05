"""Generates real performance/fairness/XAI results for the shipped live model
(src/models/artifacts/emotion_intensity_regressor_live.joblib) against the
actual feature data it was trained on.

Produces, into analysis/results/:
  - metrics_summary.csv               per-emotion MAE/RMSE/R2 + baseline comparison
  - domain_breakdown.csv              MAE broken out by source dataset
  - cv_stability.csv                  5-fold grouped-CV MAE, mean +/- std per emotion
  - fig_domain_shift.png              MAE broken out by source dataset (CREMA-D/AffectNet/calibration)
  - fig_cv_stability.png              5-fold grouped-CV MAE, mean +/- std per emotion

SHAP/XAI figures and the detection-style visuals (confusion matrices, ROC/PR,
calibration, scorecard) live in presentation_figures.py, not here.

Re-derives the held-out test split using the exact same grouped 70/30 logic as
src/training/train_live_model.py (same seed, same group keys) so results are a
true out-of-sample read on the model that is actually shipped, not on this
script's own guesswork.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from models.model import EMOTIONS  # noqa: E402
from training.train_live_model import (  # noqa: E402
    CALIBRATION_PATH,
    CREMAD_PATH,
    AFFECTNET_PATH,
    SPLIT_SEED,
    TEST_FRACTION,
    window_calibration_clips,
)

RESULTS_DIR = REPO_ROOT / "analysis" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = REPO_ROOT / "src" / "models" / "artifacts" / "emotion_intensity_regressor_live.joblib"

# ---- palette (validated categorical order + sequential blue ramp; see dataviz skill) ----
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_LINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "axes.edgecolor": BASELINE_LINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def load_merged_with_source():
    """Same merge as train_live_model.load_merged(), but also returns a
    per-row source label (cremad / calibration / affectnet) for the domain
    breakdown, and the shared feature_names."""
    cremad = np.load(CREMAD_PATH, allow_pickle=True)
    calibration = np.load(CALIBRATION_PATH, allow_pickle=True)
    affectnet = np.load(AFFECTNET_PATH, allow_pickle=True)

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
    source = np.array(
        ["CREMA-D"] * len(cremad["X"])
        + ["Calibration (self)"] * len(calibration_X)
        + ["AffectNet"] * len(affectnet["X"])
    )
    return X, y, groups, source, list(cremad["feature_names"])


def split_by_group(X, y, groups, source):
    candidate_groups = sorted(set(groups) - {"self"})
    shuffled = np.random.RandomState(SPLIT_SEED).permutation(candidate_groups)
    n_test = round(len(shuffled) * TEST_FRACTION)
    test_groups = set(shuffled[:n_test])

    test_mask = np.isin(groups, list(test_groups))
    train_mask = ~test_mask
    return (
        X[train_mask], y[train_mask], source[train_mask],
        X[test_mask], y[test_mask], source[test_mask],
    )


def per_emotion_metrics(y_true, y_pred):
    rows = []
    for i, emotion in enumerate(EMOTIONS):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        rows.append({"emotion": emotion, "mae": mae, "rmse": rmse, "r2": r2, "n": len(y_true)})
    return pd.DataFrame(rows)


def main():
    print("Loading merged feature data (CREMA-D + calibration + AffectNet)...")
    X, y, groups, source, feature_names = load_merged_with_source()
    X_train, y_train, source_train, X_test, y_test, source_test = split_by_group(X, y, groups, source)
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")
    print("Test set source composition:")
    print(pd.Series(source_test).value_counts())

    model = load(MODEL_PATH)
    y_pred = model.predict(X_test)

    # ---------------------------------------------------------------
    # 1. Held-out test performance: MAE, RMSE, R2 per emotion, + naive baseline
    # ---------------------------------------------------------------
    metrics = per_emotion_metrics(y_test, y_pred)

    baseline_pred = np.tile(y_train.mean(axis=0), (len(y_test), 1))
    baseline_mae = np.array([
        mean_absolute_error(y_test[:, i], baseline_pred[:, i]) for i in range(len(EMOTIONS))
    ])
    metrics["baseline_mae"] = baseline_mae
    metrics["improvement_over_baseline_pct"] = 100 * (1 - metrics["mae"] / metrics["baseline_mae"])

    metrics_path = RESULTS_DIR / "metrics_summary.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"\nSaved {metrics_path}")
    print(metrics.round(3).to_string(index=False))

    # ---------------------------------------------------------------
    # 2. Domain-shift breakdown: MAE by source dataset (CREMA-D vs AffectNet vs calibration)
    # ---------------------------------------------------------------
    domain_rows = []
    for src_name in ["CREMA-D", "AffectNet", "Calibration (self)"]:
        mask = source_test == src_name
        if mask.sum() == 0:
            continue
        for i, emotion in enumerate(EMOTIONS):
            domain_rows.append({
                "source": src_name,
                "emotion": emotion,
                "mae": mean_absolute_error(y_test[mask, i], y_pred[mask, i]),
                "n": int(mask.sum()),
            })
    domain_df = pd.DataFrame(domain_rows)
    domain_df.to_csv(RESULTS_DIR / "domain_breakdown.csv", index=False)
    print(f"\nSaved {RESULTS_DIR / 'domain_breakdown.csv'}")
    print(domain_df.pivot(index="emotion", columns="source", values="mae").round(3).to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    sources_present = domain_df["source"].unique().tolist()
    colors_by_source = {sources_present[i]: c for i, c in zip(range(len(sources_present)), [BLUE, ORANGE, AQUA])}
    x = np.arange(len(EMOTIONS))
    w = 0.8 / len(sources_present)
    for j, src_name in enumerate(sources_present):
        sub = domain_df[domain_df["source"] == src_name].set_index("emotion").reindex(EMOTIONS)
        ax.bar(x + (j - (len(sources_present) - 1) / 2) * w, sub["mae"], width=w,
               color=colors_by_source[src_name], label=f"{src_name} (n={int(sub['n'].iloc[0])})")
    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in EMOTIONS])
    ax.set_ylabel("MAE (intensity scale 0-3)")
    ax.set_title("Domain shift check — error by data source, per emotion", color=INK_PRIMARY, fontsize=13, loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_domain_shift.png")
    plt.close(fig)

    # ---------------------------------------------------------------
    # 3. Grouped 5-fold CV stability (on the training pool, group = actor/self/affectnet-id)
    # ---------------------------------------------------------------
    print("\nRunning grouped 5-fold CV (this refits the model 5x, ~1-2 min)...")
    from models.model import build_model
    # rebuild the same train_mask used in split_by_group, to get aligned train-side groups
    candidate_groups = sorted(set(groups) - {"self"})
    shuffled = np.random.RandomState(SPLIT_SEED).permutation(candidate_groups)
    n_test = round(len(shuffled) * TEST_FRACTION)
    test_groups = set(shuffled[:n_test])
    train_mask = ~np.isin(groups, list(test_groups))
    groups_train = groups[train_mask]

    gkf = GroupKFold(n_splits=5)
    fold_maes = {e: [] for e in EMOTIONS}
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups_train)):
        m = build_model()
        m.fit(X_train[tr_idx], y_train[tr_idx])
        preds = m.predict(X_train[val_idx])
        for i, emotion in enumerate(EMOTIONS):
            fold_maes[emotion].append(mean_absolute_error(y_train[val_idx, i], preds[:, i]))
        print(f"  fold {fold+1}/5 done (n_val={len(val_idx)}, groups={len(set(groups_train[val_idx]))})")

    cv_df = pd.DataFrame({
        "emotion": EMOTIONS,
        "cv_mae_mean": [np.mean(fold_maes[e]) for e in EMOTIONS],
        "cv_mae_std": [np.std(fold_maes[e]) for e in EMOTIONS],
    })
    cv_df.to_csv(RESULTS_DIR / "cv_stability.csv", index=False)
    print(f"\nSaved {RESULTS_DIR / 'cv_stability.csv'}")
    print(cv_df.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(EMOTIONS, cv_df["cv_mae_mean"], yerr=cv_df["cv_mae_std"], color=BLUE, capsize=4,
           error_kw={"ecolor": INK_SECONDARY, "linewidth": 1.5})
    ax.set_xticklabels([e.capitalize() for e in EMOTIONS])
    ax.set_ylabel("MAE (5-fold grouped CV, mean ± std)")
    ax.set_title("Cross-validation stability — is performance consistent across actors?", color=INK_PRIMARY, fontsize=13, loc="left")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_cv_stability.png")
    plt.close(fig)

    print(f"\nAll results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
