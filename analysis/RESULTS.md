# Emotion-Intensity Model — Evaluation Report

**Model:** `src/models/artifacts/emotion_intensity_regressor_live.joblib` — a
multi-output Random Forest regressor (150 trees, max depth 15) that maps 52
MediaPipe face-blendshape scores to intensity estimates (0–3) for six emotions:
happy, sad, anger, surprise, disgust, fear.

**Evaluation data:** 5,522 held-out examples the model never saw during
training — 782 CREMA-D clips from held-out actors and 4,740 AffectNet photos.
The split is identity-grouped (all clips from a given actor land entirely in
train or entirely in test), so scores measure generalization to *new faces*,
not recognition of familiar ones.

All numbers and figures in this report are reproducible:

```
.venv/Scripts/python.exe analysis/evaluate_model.py          # regression metrics, CV, domain breakdown
.venv/Scripts/python.exe analysis/presentation_figures.py    # detection metrics, calibration, XAI
.venv/Scripts/python.exe analysis/fairness_audit.py          # demographic subgroup audit
```

---

## 1. Summary of findings

1. **The model detects all six emotions far better than chance.** ROC AUC
   ranges from 0.87 (sad) to 0.98 (happy); chance is 0.50.
2. **Its intensity score is well calibrated** — the score can be read as a
   confidence. Across all six emotions, when the model outputs a scaled
   confidence of *x*, the emotion is truly present about *x* of the time.
3. **It is a conservative detector at the default threshold**: precision is
   high (74–92% of its "present" calls are correct) but recall varies widely —
   84% of true happy expressions are caught vs. only 17% of true sad ones.
   This is a tunable threshold trade-off, not a fixed property (see §3).
4. **The model attends to the same facial regions humans do** (SHAP analysis):
   happiness is read mostly from the mouth/smile (53% of feature influence),
   fear from the eyes (43%), anger from the brows (32%), disgust from
   non-smile mouth actions like sneer/press (35%).
5. **No demographic subgroup is flagged for disparate error** in the audit of
   held-out CREMA-D actors (sex, race, ethnicity, age), though accuracy
   degrades modestly for actors aged 50+ (+14% error) and the Asian subgroup
   is too small to evaluate at all — see §7.
6. **Candid photos are harder than posed clips.** Error is consistently higher
   on in-the-wild AffectNet photos than on posed CREMA-D clips for every
   emotion except happy — the expected domain gap, and the reason AffectNet
   was added to training in the first place.
7. **Results are stable, not a lucky split.** 5-fold identity-grouped
   cross-validation reproduces the held-out MAE within ±0.02 for every emotion.

---

## 2. What the model was trained and tested on

→ `fig_dataset_composition.png`

| Source | Type | Contribution |
|---|---|---|
| CREMA-D | Posed actor clips (91 actors), LO/MD/HI intensity labels | ~2,450 clips; no surprise category |
| AffectNet | Candid in-the-wild photos, binary present/absent labels | ~15,800 photos; only images where two independent labelings agree |
| Self-recorded calibration clips | Short webcam clips, windowed | ~19 clips, train-only |

Label structure caveat: AffectNet labels are effectively binary (absent = 0 or
present = 3), and CREMA-D adds a minority of graded 1/2 levels. Ground truth
is therefore mostly a presence signal rather than a smooth intensity scale,
which is why this report presents both a detection view (§3) and a regression
view (§4).

## 3. Detection performance (primary view)

Each emotion is scored as a detection task — "is this emotion clearly present
(label ≥ 1.5)?" — using the model's continuous intensity output as the score.

→ `fig_model_scorecard.png`, `fig_confusion_matrices.png`, `fig_roc_curves.png`,
`fig_pr_curves.png`; numbers in `scorecard.csv`

| Emotion | Accuracy | Precision | Recall | F1 | ROC AUC | Avg Precision |
|---|---|---|---|---|---|---|
| Happy | 0.95 | 0.92 | 0.84 | 0.88 | **0.98** | 0.95 |
| Surprise | 0.89 | 0.74 | 0.51 | 0.60 | **0.93** | 0.73 |
| Anger | 0.91 | 0.74 | 0.41 | 0.52 | **0.91** | 0.62 |
| Fear | 0.91 | 0.74 | 0.34 | 0.46 | **0.89** | 0.61 |
| Disgust | 0.91 | 0.76 | 0.25 | 0.38 | **0.89** | 0.57 |
| Sad | 0.91 | 0.81 | 0.17 | 0.27 | **0.87** | 0.54 |

Reading this honestly:

- **Ranking quality (AUC) is strong across the board** — the model reliably
  scores faces showing an emotion above faces that don't.
- **The default threshold trades recall for precision.** When the model says
  an emotion is present, it is right 74–92% of the time; but for subtle
  negative emotions (sad, disgust, fear) it misses the majority of true
  cases. The ROC curves show what recall could be bought by accepting more
  false alarms — e.g. sad reaches ~80% recall at a ~20% false-positive rate.
  Any deployment should pick its threshold from these curves based on whether
  missed emotions or false alarms are costlier for the use case.
- **Accuracy alone overstates performance** (89–95%) because most test faces
  don't show any given emotion; precision/recall/AUC are the honest metrics
  here, and both are reported for that reason.

## 4. Intensity accuracy (regression view)

→ `metrics_summary.csv`

| Emotion | MAE | RMSE | R² | vs. always-predict-the-mean |
|---|---|---|---|---|
| Happy | 0.240 | 0.567 | 0.78 | 76% lower error |
| Surprise | 0.458 | 0.814 | 0.45 | 41% lower |
| Anger | 0.404 | 0.748 | 0.38 | 35% lower |
| Fear | 0.434 | 0.777 | 0.35 | 30% lower |
| Disgust | 0.416 | 0.765 | 0.31 | 27% lower |
| Sad | 0.435 | 0.768 | 0.27 | 22% lower |

On the 0–3 intensity scale, the average prediction error is 0.24 (happy) to
0.46 (surprise). Every emotion beats the naive baseline. R² is high for happy
and moderate for the rest — consistent with the mostly-binary label structure
(§2), which limits how much graded-intensity signal exists to learn.

### Calibration

→ `fig_calibration.png`

All six reliability curves sit close to the diagonal: the model's scaled
output can be treated as a probability that the emotion is truly present.
This is a deployment-relevant property — downstream consumers (e.g. the live
app) can threshold or display the score as a confidence without recalibration.

### Stability (cross-validation)

→ `fig_cv_stability.png`, `cv_stability.csv`

5-fold cross-validation with identity-grouped folds (no actor appears in both
sides of any fold) reproduces the held-out results: e.g. happy MAE
0.255 ± 0.003, anger 0.408 ± 0.021. The headline numbers are not an artifact
of one favorable split.

## 5. Domain shift: posed clips vs. candid photos

→ `fig_domain_shift.png`, `domain_breakdown.csv`

MAE is higher on AffectNet (candid, unposed, variable lighting) than on
CREMA-D (posed actor clips) for every emotion except happy. Two callouts:

- **Surprise on CREMA-D (MAE 0.108) must not be cited as a strength.**
  CREMA-D contains no surprise category, so every CREMA-D test row has a true
  surprise label of 0 — trivially easy. The real surprise evaluation is the
  AffectNet column (MAE 0.516, the hardest cell in the table).
- The candid-photo gap is the same domain-shift phenomenon that motivated
  adding AffectNet to training (documented in `train_live_model.py`); it is
  reduced, not eliminated. Live-webcam conditions resemble AffectNet more
  than CREMA-D, so AffectNet-side numbers are the better predictor of live
  performance.

## 6. Explainability — what the model actually looks at

Grad-CAM does not apply to this architecture (no convolutional layers; the
regressor never sees pixels). The equivalent for tree models is SHAP
(`shap.TreeExplainer`, exact for tree ensembles), applied to the 52 named
MediaPipe blendshape features — which are already human-readable facial
actions, so attributions point at named muscle movements rather than pixel
regions.

→ `fig_face_region_importance.png` (summary), `fig_shap_beeswarm_<emotion>.png`
(per-feature detail), `face_region_importance.csv`, `shap_top_features_per_emotion.csv`

Share of total feature influence by facial region, per emotion:

| Emotion | Dominant region | Share | Matches human intuition? |
|---|---|---|---|
| Happy | Mouth / smile | 53% | Yes — smiling |
| Fear | Eyes | 43% | Yes — eye widening |
| Disgust | Mouth (non-smile) | 35% | Yes — sneer, lip press |
| Anger | Brows | 32% | Yes — brow furrowing |
| Surprise | Eyes | 30% | Yes — widened eyes (plus jaw/mouth opening) |
| Sad | Mouth (frown) / eyes | 27% / 25% | Yes — frown, heavy eyes |

The per-feature beeswarms confirm the direction, not just the magnitude: e.g.
for happy, high `mouthSmileLeft`/`mouthSmileRight` values push the score up
and their absence pushes it down. The model's reasoning aligns with how
humans read faces, which supports (though does not by itself prove) that it
has learned expression rather than dataset shortcuts.

## 7. Demographic fairness audit

→ `fig_fairness_audit.png`, `fairness_audit.csv`, `fairness_by_emotion.csv`
(produced by `analysis/fairness_audit.py` using CREMA-D's published per-actor
demographics; protocol as specified in `docs/MODEL_DOCUMENTATION_AND_TESTING.md` §2.2)

Scope: the 29 held-out CREMA-D test actors (782 clips). AffectNet carries no
reliable per-image demographics, so its 4,740 test photos cannot be audited
this way — the audit covers what is auditable and says so.

Method: pooled MAE per subgroup with 95% confidence intervals from a cluster
bootstrap over actors (clips from one actor are correlated, so resampling is
done at the actor level). Pre-registered disparity flag: >25% relative gap
from the overall MAE with the CI clear of it. Subgroups with fewer than 3
test actors are reported as *insufficient data*, not as a result.

| Slice | Result |
|---|---|
| Sex (Male 18 / Female 11 actors) | No gap: 0.323 vs 0.313, CIs fully overlap |
| Race — Caucasian (20) vs African American (7) | No gap: 0.322 vs 0.307, CIs fully overlap |
| Race — Asian (1 actor) | **Insufficient data — cannot conclude anything** |
| Ethnicity — Hispanic (3) vs Not Hispanic (26) | No gap detected; Hispanic CI is wide (n=3) |
| Age 20–34 (12) / 35–49 (9) | At or below overall error |
| Age 50+ (8 actors) | **+14% higher error (0.364 vs 0.319 overall), CI [0.332, 0.395] sits above the overall line** — below the 25% flag threshold, but a real, consistent degradation |

Honest reading:

- **No subgroup meets the pre-registered disparity flag.** Sex, race (where
  measurable), and ethnicity slices show near-identical error.
- **The age-50+ gap is the audit's substantive finding.** It is modest but
  not noise (the bootstrap CI excludes the overall mean). Plausible causes
  include age-related differences in facial dynamics and older faces being a
  minority of training data. Any deployment serving older users should
  expect somewhat less accurate readings until this is addressed (e.g. by
  adding older-face training data).
- **"Not flagged" is not "proven fair."** The Asian subgroup has one test
  actor (seven in the entire dataset) — the audit cannot say anything about
  it, and the AffectNet majority of the test set is unaudited entirely. The
  honest claim is: *no disparity was detected on the slices with enough data
  to test*, which is weaker than *the model is equitable*.

## 8. Limitations and responsible-use notes

- **Fairness audit coverage is partial** (§7): one race subgroup and the
  entire AffectNet portion are unauditable with current metadata, and the
  age-50+ degradation is real. A skin-tone-based audit (e.g. Monk scale
  annotation of a test sample) would cover what the race labels miss.
- **Recall on subtle negative emotions is low at the default threshold**
  (sad 17%, disgust 25%). Applications must not assume "no alert" means "no
  emotion."
- **Ground truth is mostly binary** (§2), so graded-intensity accuracy
  between 0 and 3 is only weakly validated for AffectNet-dominated emotions.
- **Upstream dependency:** all predictions inherit MediaPipe's face-landmark
  quality. Poor lighting, occlusion, or extreme head pose degrade the input
  features before the model runs; the live app should surface detection
  failure as "no reading," never as zero emotion.
- **Acted vs. felt emotion:** CREMA-D expressions are performed by actors and
  AffectNet labels describe apparent expression. The model estimates how a
  face *looks*, not what a person *feels*, and outputs should always be
  described that way.

## 9. File inventory

| File | Contents |
|---|---|
| `fig_dataset_composition.png` | training/evaluation data per emotion, by source |
| `fig_model_scorecard.png` | detection metrics heatmap (the one-slide summary) |
| `fig_confusion_matrices.png` | per-emotion confusion matrices, % + counts |
| `fig_roc_curves.png` / `fig_pr_curves.png` | detection quality curves with AUC / AP |
| `fig_calibration.png` | reliability curves (score-as-confidence check) |
| `fig_cv_stability.png` | 5-fold grouped-CV MAE, mean ± std |
| `fig_domain_shift.png` | MAE by source dataset per emotion |
| `fig_fairness_audit.png` | subgroup MAE with bootstrap CIs (§7) |
| `fairness_audit.csv` / `fairness_by_emotion.csv` | fairness audit data (§7) |
| `fig_face_region_importance.png` | SHAP influence by facial region × emotion |
| `fig_shap_beeswarm_*.png` | per-feature SHAP detail, one per emotion |
| `scorecard.csv` | detection metrics (table in §3) |
| `metrics_summary.csv` | regression metrics + baseline comparison (§4) |
| `cv_stability.csv` | cross-validation results (§4) |
| `domain_breakdown.csv` | per-source MAE (§5) |
| `face_region_importance.csv` / `shap_top_features_per_emotion.csv` | XAI data (§6) |
