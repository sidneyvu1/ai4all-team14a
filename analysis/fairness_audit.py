"""Demographic fairness audit for the shipped emotion-intensity model.

Implements the protocol in docs/MODEL_DOCUMENTATION_AND_TESTING.md §2.2:
subgroup performance on the held-out CREMA-D test actors, sliced by the
demographic metadata CREMA-D publishes per actor (data/crema_d_demographics.csv,
from the official CREMA-D repository). AffectNet has no reliable per-image
demographics, so this audit covers the CREMA-D portion of the test set only —
stated as a scope limit, not implied away.

Method:
  - Slices: Sex (Male/Female), Race (Caucasian / African American / Asian),
    Ethnicity (Hispanic / Not Hispanic), Age bracket (20-34 / 35-49 / 50+).
  - Metric: overall MAE (all 6 emotions pooled) per subgroup, plus per-emotion MAE.
  - Uncertainty: cluster bootstrap over ACTORS (not clips) — resampling actors
    with replacement 2,000x — because clips from one actor are correlated and a
    per-clip bootstrap would understate the uncertainty.
  - Disparity flag (threshold pre-registered in the docs): a subgroup is flagged
    if its pooled MAE differs from the overall CREMA-D test MAE by more than
    25% relative.
  - Sample floor: subgroups with fewer than 3 test actors are reported as
    "insufficient data", not as a pass or fail.

Outputs into analysis/results/:
  fairness_audit.csv        pooled MAE, CI, n per subgroup + flag status
  fairness_by_emotion.csv   per-emotion MAE per subgroup
  fig_fairness_audit.png    pooled MAE per subgroup with bootstrap 95% CIs
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "analysis"))
from models.model import EMOTIONS  # noqa: E402
from evaluate_model import load_merged_with_source, split_by_group  # noqa: E402

RESULTS_DIR = REPO_ROOT / "analysis" / "results"
MODEL_PATH = REPO_ROOT / "src" / "models" / "artifacts" / "emotion_intensity_regressor_live.joblib"
DEMOGRAPHICS_PATH = REPO_ROOT / "data" / "crema_d_demographics.csv"

MIN_ACTORS = 3          # below this: report "insufficient data"
DISPARITY_REL = 0.25    # flag if subgroup MAE differs from overall by >25% relative
N_BOOT = 2000
BOOT_SEED = 7

BLUE = "#2a78d6"
ORANGE = "#eb6834"
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


def age_bracket(age):
    if age < 35:
        return "20-34"
    if age < 50:
        return "35-49"
    return "50+"


def pooled_mae(y_true, y_pred):
    return float(np.abs(y_true - y_pred).mean())


def actor_bootstrap_ci(df_rows, n_boot=N_BOOT, seed=BOOT_SEED):
    """Cluster bootstrap over actors: resample actors with replacement, recompute pooled MAE."""
    rng = np.random.RandomState(seed)
    actors = df_rows["actor"].unique()
    by_actor = {a: g for a, g in df_rows.groupby("actor")}
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(actors, size=len(actors), replace=True)
        errs = np.concatenate([by_actor[a]["abs_err"].values for a in sample])
        stats.append(errs.mean())
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main():
    print("Loading model, data, and demographics...")
    X, y, groups, source, feature_names = load_merged_with_source()
    X_train, y_train, s_train, X_test, y_test, s_test = split_by_group(X, y, groups, source)

    # groups aligned with X rows; rebuild test mask to recover per-row actor ids
    from training.train_live_model import SPLIT_SEED, TEST_FRACTION
    candidate_groups = sorted(set(groups) - {"self"})
    shuffled = np.random.RandomState(SPLIT_SEED).permutation(candidate_groups)
    test_groups = set(shuffled[:round(len(shuffled) * TEST_FRACTION)])
    test_mask = np.isin(groups, list(test_groups))
    groups_test = groups[test_mask]
    assert len(groups_test) == len(X_test)

    model = load(MODEL_PATH)
    y_pred = model.predict(X_test)

    # CREMA-D rows only (the portion with known demographics)
    cremad_mask = s_test == "CREMA-D"
    demo = pd.read_csv(DEMOGRAPHICS_PATH)
    demo["ActorID"] = demo["ActorID"].astype(str)
    demo["AgeBracket"] = demo["Age"].apply(age_bracket)
    demo = demo.set_index("ActorID")

    rows = []
    idxs = np.where(cremad_mask)[0]
    for i in idxs:
        actor = groups_test[i]
        d = demo.loc[actor]
        for j, emotion in enumerate(EMOTIONS):
            rows.append({
                "actor": actor, "emotion": emotion,
                "abs_err": abs(y_test[i, j] - y_pred[i, j]),
                "Sex": d["Sex"], "Race": d["Race"],
                "Ethnicity": d["Ethnicity"], "AgeBracket": d["AgeBracket"],
            })
    df = pd.DataFrame(rows)
    overall_mae = df["abs_err"].mean()
    n_actors_total = df["actor"].nunique()
    print(f"CREMA-D test rows: {len(idxs)} clips from {n_actors_total} held-out actors")
    print(f"Overall CREMA-D test MAE (pooled over emotions): {overall_mae:.3f}\n")

    audit_rows, emotion_rows = [], []
    for dim in ["Sex", "Race", "Ethnicity", "AgeBracket"]:
        for value, sub in df.groupby(dim):
            n_actors = sub["actor"].nunique()
            mae = sub["abs_err"].mean()
            if n_actors < MIN_ACTORS:
                status = f"insufficient data (<{MIN_ACTORS} actors)"
                lo = hi = np.nan
            else:
                lo, hi = actor_bootstrap_ci(sub)
                rel_gap = (mae - overall_mae) / overall_mae
                status = "FLAGGED" if abs(rel_gap) > DISPARITY_REL and lo > overall_mae else "ok"
            audit_rows.append({
                "dimension": dim, "subgroup": value, "n_actors": n_actors,
                "n_rows": len(sub) // len(EMOTIONS), "pooled_mae": mae,
                "ci_low": lo, "ci_high": hi,
                "rel_gap_vs_overall_pct": 100 * (mae - overall_mae) / overall_mae,
                "status": status,
            })
            for emotion, esub in sub.groupby("emotion"):
                emotion_rows.append({"dimension": dim, "subgroup": value,
                                     "emotion": emotion, "mae": esub["abs_err"].mean()})

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(RESULTS_DIR / "fairness_audit.csv", index=False)
    pd.DataFrame(emotion_rows).to_csv(RESULTS_DIR / "fairness_by_emotion.csv", index=False)
    print(audit.round(3).to_string(index=False))

    # ---- figure: pooled MAE per subgroup with bootstrap CIs ----
    plot_df = audit[audit["ci_low"].notna()].reset_index(drop=True)
    insufficient = audit[audit["ci_low"].isna()]

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [f"{r.subgroup}\n({r.dimension}, {r.n_actors} actors)" for r in plot_df.itertuples()]
    colors = [ORANGE if r.status == "FLAGGED" else BLUE for r in plot_df.itertuples()]
    ypos = np.arange(len(plot_df))
    ax.barh(ypos, plot_df["pooled_mae"], color=colors, height=0.62)
    ax.errorbar(plot_df["pooled_mae"], ypos,
                xerr=[plot_df["pooled_mae"] - plot_df["ci_low"],
                      plot_df["ci_high"] - plot_df["pooled_mae"]],
                fmt="none", ecolor=INK_SECONDARY, capsize=4, linewidth=1.5)
    ax.axvline(overall_mae, color=INK_MUTED, linewidth=1.5, linestyle="--")
    ax.text(overall_mae + 0.004, len(plot_df) - 0.4, f"overall ({overall_mae:.3f})",
            color=INK_MUTED, fontsize=9)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("MAE, all emotions pooled (whiskers = 95% bootstrap CI over actors)")
    title = "Fairness audit — error by demographic subgroup (CREMA-D held-out actors)"
    if len(insufficient):
        skipped = ", ".join(f"{r.subgroup} ({r.n_actors})" for r in insufficient.itertuples())
        title += f"\nNot shown, insufficient data: {skipped}"
    ax.set_title(title, color=INK_PRIMARY, fontsize=12, loc="left")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_fairness_audit.png")
    plt.close(fig)

    print(f"\nSaved fairness_audit.csv, fairness_by_emotion.csv, fig_fairness_audit.png to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
