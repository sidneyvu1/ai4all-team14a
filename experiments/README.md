# Experiments

Exploratory scratch work, kept for reference but not part of the shipped pipeline (`src/dataset_prep` → `src/training` → `src/ui/app.py`).

- **DataSet.ipynb** — an earlier, abandoned approach: Keras `ImageDataGenerator` augmentation over raw AffectNet images, unrelated to the MediaPipe-blendshape + RandomForest pipeline actually shipped.
- **explore_mosei_csd.py** — a one-off script for inspecting the raw CMU-MOSEI `.csd` file structure. Superseded by `src/dataset_prep/align_mosei_video.py`, which does the real extraction.
