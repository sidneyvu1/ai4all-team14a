---
title: Live Emotion Intensity Demo
emoji: 🎭
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.20.0
python_version: "3.12"
app_file: src/ui/app.py
pinned: false
short_description: Webcam emotion-intensity tracking (AI4ALL Team 14a)
---

# Team 14a AI4ALL Project

## Members
Andy Sosa, Sidney Vu

## About

A live emotion-intensity recognition tool for neuromarketing-style testing: point a
webcam at someone watching content, and see real-time intensity estimates (0-3)
for six emotions -- happy, sad, anger, surprise, disgust, fear -- alongside a
rolling peaks graph of the session.

The pipeline:

1. **Face -> features**: [MediaPipe FaceLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
   extracts 52 blendshape scores per frame from the webcam feed.
2. **Features -> intensity**: a `RandomForestRegressor` (`src/models/model.py`) maps a
   max-pooled window of blendshape scores to a 6-emotion intensity vector.
3. **Live app**: `src/ui/app.py` is a Gradio dashboard that streams webcam frames,
   runs the pipeline above per-frame, and renders live intensity bars plus a
   peaks-over-time graph per session.

## Repo layout

```
src/
  dataset_prep/   # feature extraction from each raw dataset -> data/*.npz
  models/         # shared model definition (pooling + RandomForest)
  training/       # trains the models that live in src/models/artifacts/
  ui/app.py       # the live Gradio app
experiments/      # exploratory scratch work, not part of the shipped pipeline
```

## Dataset

The model is trained on a merge of three sources, all reduced to the same
MediaPipe-blendshape feature space:

| Source | What it contributes | Labels |
|---|---|---|
| [CREMA-D](https://www.kaggle.com/datasets/orvile/crema-d-emotional-multimodal-dataset) | ~7,400 short acted clips, 91 actors | LO/MD/HI intensity per emotion |
| Self-recorded calibration clips (`data/calibration/`) | 19 clips of one person, deliberately posed at 3 intensity levels | mild/moderate/strong -> 1/2/3 |
| [AffectNet](https://www.kaggle.com/datasets/mstjebashazida/affectnet) | ~17,700 candid, in-the-wild photos | present/absent only (encoded as full intensity) |

CREMA-D and the calibration clips give real intensity gradation but are posed,
not candid, so live webcam input reads differently than train data
(diagnosed early on: the model defaulted to predicting "sad" for almost any
live expression). AffectNet's candid photos close that domain gap and add the
only real held-out `surprise` data, at the cost of only binary present/absent
labels -- see `src/training/train_live_model.py`'s docstring for the full
reasoning and known limitation (AffectNet's binary labels push the merged
model toward predicting intensity extremes).

An early CMU-MOSEI + OpenFace2 attempt (`src/training/training.py`) is kept
for reference but superseded -- OpenFace2 features aren't producible from a
live MediaPipe-only pipeline, so that model is never loaded by the app.

## Project setup

1. `uv sync` in the project root.
2. Run `dataset_download_script.py` to fetch the raw datasets (large; not
   committed to git).
3. `CREMAD_DIR` / `AFFECTNET_DIR` env vars can point dataset extraction at a
   different local cache path than the defaults in
   `src/dataset_prep/extract_cremad_features.py` /
   `extract_affectnet_features.py`.
4. Run the `src/dataset_prep/extract_*.py` scripts to build `data/*.npz`
   feature files, then `src/training/train_live_model.py` to produce
   `src/models/artifacts/emotion_intensity_regressor_live.joblib`.

## Running the app

```
uv run python src/ui/app.py
```

This starts a local Gradio server and prints a `localhost` link. Opening it
prompts for webcam permission in the browser; once granted, you'll see live
per-emotion intensity bars and a graph of peak emotions over the session.

@ us in Discord with any questions or trouble running the program.
