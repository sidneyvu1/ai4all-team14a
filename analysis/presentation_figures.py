"""Industry-standard model-evaluation figures for presentation use.

Complements evaluate_model.py (which produces the regression-metric read) with
the standard classification-style visuals stakeholders expect from a model
evaluation: dataset composition, confusion matrices, ROC / precision-recall
curves, a scorecard heatmap, calibration curves, SHAP beeswarms, and a
face-region x emotion importance heatmap.

Framing: the ground-truth labels are effectively binary per emotion (AffectNet
contributes only 0 or 3; CREMA-D adds some 1/2 levels). So each emotion is
scored as a detection task -- "is this emotion clearly present (label >= 1.5)?"
-- with the model's continuous intensity output used as the detection score.
That is exactly the setting ROC / PR / confusion analysis is designed for.

Outputs into analysis/results/:
  fig_dataset_composition.png   what the model was trained/tested on
  fig_confusion_matrices.png    6-panel confusion matrix (present vs absent)
  fig_roc_curves.png            ROC curve + AUC per emotion
  fig_pr_curves.png             precision-recall curve + AP per emotion
  fig_model_scorecard.png       heatmap: accuracy/precision/recall/F1/AUC per emotion
  fig_calibration.png           reliability curves (does 80% confidence mean 80% right?)
  fig_face_region_importance.png  which part of the face drives each emotion (SHAP)
  fig_shap_beeswarm_<emotion>.png standard SHAP beeswarm, one per emotion
  scorecard.csv                 the scorecard numbers
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
)
from sklearn.calibration import calibration_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "analysis"))
from models.model import EMOTIONS  # noqa: E402
from evaluate_model import load_merged_with_source, split_by_group  # noqa: E402

RESULTS_DIR = REPO_ROOT / "analysis" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = REPO_ROOT / "src" / "models" / "artifacts" / "emotion_intensity_regressor_live.joblib"

PRESENT_THRESHOLD = 1.5  # label >= 1.5 counts as "emotion clearly present"

# ---- validated categorical palette (slots 1-6) + chrome, from the dataviz reference ----
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BLUE, ORANGE = SERIES[0], SERIES[1]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_LINE = "#c3c2b7"
SURFACE = "#fcfcfb"
# sequential blue ramp, light -> dark (steps 100..700)
SEQ_BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

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

from matplotlib.colors import LinearSegmentedColormap
BLUES_CMAP = LinearSegmentedColormap.from_list("seq_blue", [SURFACE] + SEQ_BLUES)

EMO_LABELS = [e.capitalize() for e in EMOTIONS]


def fig_dataset_composition(y, source):
    """Stacked bars: for each emotion, how many 'present' examples each dataset contributes."""
    sources = ["CREMA-D", "AffectNet", "Calibration (self)"]
    colors = {s: c for s, c in zip(sources, [BLUE, ORANGE, "#1baf7a"])}
    counts = {s: [(y[source == s, i] >= PRESENT_THRESHOLD).sum() for i in range(6)] for s in sources}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(6)
    bottom = np.zeros(6)
    for s in sources:
        vals = np.array(counts[s], dtype=float)
        ax.bar(x, vals, bottom=bottom, color=colors[s], label=s, width=0.62,
               edgecolor=SURFACE, linewidth=2)
        bottom += vals
    for xi, total in zip(x, bottom):
        ax.text(xi, total + 40, f"{int(total):,}", ha="center", fontsize=10, color=INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(EMO_LABELS)
    ax.set_ylabel("Examples where emotion is clearly present")
    ax.set_title("What the model learned from — examples per emotion, by dataset",
                 color=INK_PRIMARY, fontsize=13, loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_dataset_composition.png")
    plt.close(fig)


def fig_confusion_matrices(y_true_bin, y_pred_bin):
    """6-panel confusion matrices, row-normalized, with plain-language cell labels."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    for i, (emotion, ax) in enumerate(zip(EMO_LABELS, axes.flat)):
        cm = confusion_matrix(y_true_bin[:, i], y_pred_bin[:, i], normalize="true")
        raw = confusion_matrix(y_true_bin[:, i], y_pred_bin[:, i])
        ax.imshow(cm, cmap=BLUES_CMAP, vmin=0, vmax=1)
        for r in range(2):
            for c in range(2):
                txt_color = "#ffffff" if cm[r, c] > 0.55 else INK_PRIMARY
                ax.text(c, r, f"{cm[r, c]:.0%}\n({raw[r, c]:,})",
                        ha="center", va="center", fontsize=11, color=txt_color)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted\nabsent", "Predicted\npresent"], fontsize=9)
        ax.set_yticklabels(["Actually\nabsent", "Actually\npresent"], fontsize=9)
        ax.set_title(emotion, fontsize=12, color=INK_PRIMARY)
        ax.grid(False)
    fig.suptitle("Confusion matrices — when the emotion is truly there, does the model see it?",
                 color=INK_PRIMARY, fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_confusion_matrices.png")
    plt.close(fig)


def fig_roc(y_true_bin, y_score):
    fig, ax = plt.subplots(figsize=(8, 7))
    aucs = {}
    for i, emotion in enumerate(EMO_LABELS):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        aucs[emotion] = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=SERIES[i], linewidth=2,
                label=f"{emotion}  (AUC = {aucs[emotion]:.2f})")
    ax.plot([0, 1], [0, 1], color=BASELINE_LINE, linewidth=1.5, linestyle="--")
    ax.text(0.62, 0.55, "Random guessing", color=INK_MUTED, fontsize=9, rotation=38)
    ax.set_xlabel("False positive rate (false alarms)")
    ax.set_ylabel("True positive rate (real emotions caught)")
    ax.set_title("ROC curves — detection quality per emotion\n(1.0 = perfect, 0.5 = coin flip)",
                 color=INK_PRIMARY, fontsize=13, loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_roc_curves.png")
    plt.close(fig)
    return aucs


def fig_pr(y_true_bin, y_score):
    fig, ax = plt.subplots(figsize=(8, 7))
    aps = {}
    for i, emotion in enumerate(EMO_LABELS):
        prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        aps[emotion] = average_precision_score(y_true_bin[:, i], y_score[:, i])
        ax.plot(rec, prec, color=SERIES[i], linewidth=2,
                label=f"{emotion}  (AP = {aps[emotion]:.2f})")
        base_rate = y_true_bin[:, i].mean()
        ax.axhline(base_rate, color=SERIES[i], linewidth=0.8, linestyle=":", alpha=0.35)
    ax.set_xlabel("Recall (share of real emotions caught)")
    ax.set_ylabel("Precision (share of alerts that were right)")
    ax.set_title("Precision–recall curves per emotion\n(dotted line = guessing at random for that emotion)",
                 color=INK_PRIMARY, fontsize=13, loc="left")
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_pr_curves.png")
    plt.close(fig)
    return aps


def fig_scorecard(y_true_bin, y_pred_bin, aucs, aps):
    rows = []
    for i, emotion in enumerate(EMO_LABELS):
        rows.append({
            "Emotion": emotion,
            "Accuracy": accuracy_score(y_true_bin[:, i], y_pred_bin[:, i]),
            "Precision": precision_score(y_true_bin[:, i], y_pred_bin[:, i]),
            "Recall": recall_score(y_true_bin[:, i], y_pred_bin[:, i]),
            "F1": f1_score(y_true_bin[:, i], y_pred_bin[:, i]),
            "ROC AUC": aucs[emotion],
            "Avg Precision": aps[emotion],
        })
    df = pd.DataFrame(rows).set_index("Emotion")
    df.to_csv(RESULTS_DIR / "scorecard.csv")

    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.imshow(df.values, cmap=BLUES_CMAP, vmin=0.3, vmax=1.0, aspect="auto")
    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            v = df.values[r, c]
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=11,
                    color="#ffffff" if v > 0.78 else INK_PRIMARY,
                    fontweight="bold" if c >= 4 else "normal")
    ax.set_xticks(range(df.shape[1])); ax.set_xticklabels(df.columns, fontsize=10)
    ax.set_yticks(range(df.shape[0])); ax.set_yticklabels(df.index, fontsize=11)
    ax.set_title("Model scorecard — emotion detection on held-out test data (0–1, higher is better)",
                 color=INK_PRIMARY, fontsize=13, loc="left")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_model_scorecard.png")
    plt.close(fig)
    return df


def fig_calibration(y_true_bin, y_score):
    """Reliability curves: when the model outputs intensity ~x, how often is the emotion truly present?"""
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, emotion in enumerate(EMO_LABELS):
        conf = np.clip(y_score[:, i] / 3.0, 0, 1)  # map 0-3 intensity onto 0-1 confidence
        frac_pos, mean_pred = calibration_curve(y_true_bin[:, i], conf, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, color=SERIES[i], linewidth=2, marker="o",
                markersize=5, label=emotion)
    ax.plot([0, 1], [0, 1], color=BASELINE_LINE, linewidth=1.5, linestyle="--")
    ax.text(0.68, 0.73, "Perfectly calibrated", color=INK_MUTED, fontsize=9, rotation=38)
    ax.set_xlabel("Model's predicted intensity (scaled 0–1)")
    ax.set_ylabel("How often the emotion was actually present")
    ax.set_title("Calibration — can you trust the score as a confidence?\n(on the diagonal = yes)",
                 color=INK_PRIMARY, fontsize=13, loc="left")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_calibration.png")
    plt.close(fig)


FACE_REGIONS = {
    "Brows": ["browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight"],
    "Eyes": ["eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
             "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
             "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
             "eyeWideLeft", "eyeWideRight"],
    "Cheeks": ["cheekPuff", "cheekSquintLeft", "cheekSquintRight"],
    "Nose": ["noseSneerLeft", "noseSneerRight"],
    "Mouth / smile": ["mouthSmileLeft", "mouthSmileRight", "mouthDimpleLeft", "mouthDimpleRight"],
    "Mouth / other": ["mouthClose", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
                      "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
                      "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper",
                      "mouthShrugLower", "mouthShrugUpper", "mouthStretchLeft", "mouthStretchRight",
                      "mouthUpperUpLeft", "mouthUpperUpRight"],
    "Jaw": ["jawForward", "jawLeft", "jawOpen", "jawRight"],
    "Overall neutrality": ["_neutral"],
}


def fig_face_regions(shap_values, feature_names):
    """Heatmap: emotion x face region, share of total SHAP importance."""
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    regions = list(FACE_REGIONS.keys())
    mat = np.zeros((len(EMOTIONS), len(regions)))
    for i in range(len(EMOTIONS)):
        mean_abs = np.abs(shap_values[i]).mean(axis=0)
        total = mean_abs.sum()
        for j, region in enumerate(regions):
            idxs = [name_to_idx[f] for f in FACE_REGIONS[region] if f in name_to_idx]
            mat[i, j] = mean_abs[idxs].sum() / total

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.imshow(mat, cmap=BLUES_CMAP, vmin=0, vmax=mat.max(), aspect="auto")
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            ax.text(c, r, f"{mat[r, c]:.0%}", ha="center", va="center", fontsize=10,
                    color="#ffffff" if mat[r, c] > 0.6 * mat.max() else INK_PRIMARY)
    ax.set_xticks(range(len(regions))); ax.set_xticklabels(regions, fontsize=10)
    ax.set_yticks(range(len(EMOTIONS))); ax.set_yticklabels(EMO_LABELS, fontsize=11)
    ax.set_title("Which part of the face the model relies on, per emotion\n(share of total feature influence — SHAP)",
                 color=INK_PRIMARY, fontsize=13, loc="left")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_face_region_importance.png")
    plt.close(fig)
    return pd.DataFrame(mat, index=EMOTIONS, columns=regions)


def fig_beeswarms(shap_values, X_sample, feature_names):
    import shap
    for i, emotion in enumerate(EMOTIONS):
        fig = plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values[i], X_sample, feature_names=feature_names,
                          max_display=10, show=False, plot_size=None)
        plt.title(f"What pushes the '{emotion}' score up or down\n"
                  "(each dot = one test face; red = strong facial movement, blue = weak)",
                  fontsize=12, color=INK_PRIMARY, loc="left")
        plt.gcf().set_facecolor(SURFACE)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"fig_shap_beeswarm_{emotion}.png", dpi=150,
                    facecolor=SURFACE, bbox_inches="tight")
        plt.close("all")


def main():
    print("Loading data and model...")
    X, y, groups, source, feature_names = load_merged_with_source()
    X_train, y_train, source_train, X_test, y_test, source_test = split_by_group(X, y, groups, source)
    model = load(MODEL_PATH)
    y_score = model.predict(X_test)

    y_true_bin = (y_test >= PRESENT_THRESHOLD).astype(int)
    y_pred_bin = (y_score >= PRESENT_THRESHOLD).astype(int)

    print("1/7 dataset composition...")
    fig_dataset_composition(y, source)

    print("2/7 confusion matrices...")
    fig_confusion_matrices(y_true_bin, y_pred_bin)

    print("3/7 ROC curves...")
    aucs = fig_roc(y_true_bin, y_score)

    print("4/7 precision-recall curves...")
    aps = fig_pr(y_true_bin, y_score)

    print("5/7 scorecard...")
    df = fig_scorecard(y_true_bin, y_pred_bin, aucs, aps)
    print(df.round(3).to_string())

    print("6/7 calibration...")
    fig_calibration(y_true_bin, y_score)

    print("7/7 SHAP (region heatmap + beeswarms)...")
    import shap
    rng = np.random.RandomState(0)
    sample_idx = rng.choice(len(X_test), size=min(400, len(X_test)), replace=False)
    X_shap = X_test[sample_idx]
    explainer = shap.TreeExplainer(model)
    # shape (n_samples, n_features, n_outputs) — slice per emotion as [:, :, i]
    shap_raw = explainer.shap_values(X_shap)
    shap_values = [shap_raw[:, :, i] for i in range(len(EMOTIONS))]
    region_df = fig_face_regions(shap_values, feature_names)
    region_df.to_csv(RESULTS_DIR / "face_region_importance.csv")
    print(region_df.round(3).to_string())
    fig_beeswarms(shap_values, X_shap, feature_names)

    shap_rows = []
    for i, emotion in enumerate(EMOTIONS):
        mean_abs = np.abs(shap_values[i]).mean(axis=0)
        for rank, j in enumerate(np.argsort(mean_abs)[::-1][:10]):
            shap_rows.append({"emotion": emotion, "rank": rank + 1,
                              "feature": feature_names[j], "mean_abs_shap": mean_abs[j]})
    pd.DataFrame(shap_rows).to_csv(RESULTS_DIR / "shap_top_features_per_emotion.csv", index=False)

    print(f"\nAll presentation figures written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
