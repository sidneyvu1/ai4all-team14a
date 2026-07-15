"""
Visualizes the cleaned CMU-MOSEI video-only dataset:
1. Bar chart of how common each emotion is (class balance)
2. A sample clip's facial features changing over time (the "flipbook" idea)
3. Heatmap showing which emotions tend to co-occur in the same clip

Run this after cleaning.py has produced mosei_video_splits.npz
"""

import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "data/mosei_video_splits.npz"
LABEL_NAMES = ["happy", "sad", "anger", "surprise", "disgust", "fear"]

data = np.load(DATA_PATH)
X_train, y_train = data["X_train"], data["y_train"]

# ---- 1. Class balance bar chart ----
pct_present = (y_train > 0).mean(axis=0) * 100

plt.figure(figsize=(8, 5))
bars = plt.bar(LABEL_NAMES, pct_present, color="#4C72B0")
plt.ylabel("% of clips with this emotion present")
plt.title("Emotion Frequency in Training Set")
for bar, pct in zip(bars, pct_present):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
              f"{pct:.1f}%", ha="center")
plt.tight_layout()
plt.savefig("emotion_frequency.png", dpi=150)
plt.show()
print("Saved emotion_frequency.png")

# ---- 2. Sample clip: how facial features change over the 50 timesteps ----
# Pick a clip that has a clearly present emotion, more interesting to look at
# than a random empty one.
sample_idx = np.argmax(y_train[:, 0])  # a clip with the strongest "happy" signal
sample_clip = X_train[sample_idx]      # shape (50, 35)

plt.figure(figsize=(10, 5))
# Plot just a handful of the 35 features so the plot stays readable
for feat_idx in [0, 5, 10, 15, 20]:
    plt.plot(sample_clip[:, feat_idx], label=f"Feature {feat_idx}")
plt.xlabel("Timestep (across the clip)")
plt.ylabel("Feature value")
plt.title(f"Sample Facial Feature Trajectories (clip index {sample_idx}, labeled 'happy')")
plt.legend()
plt.tight_layout()
plt.savefig("sample_trajectory.png", dpi=150)
plt.show()
print("Saved sample_trajectory.png")

# ---- 3. Emotion co-occurrence heatmap ----
# Shows, e.g., how often "anger" and "disgust" show up together in the same clip.
binary_labels = (y_train > 0).astype(int)
co_occurrence = binary_labels.T @ binary_labels  # (6, 6) counts

plt.figure(figsize=(7, 6))
im = plt.imshow(co_occurrence, cmap="Blues")
plt.xticks(range(6), LABEL_NAMES, rotation=45)
plt.yticks(range(6), LABEL_NAMES)
plt.title("Emotion Co-occurrence (how often two emotions appear together)")
for i in range(6):
    for j in range(6):
        plt.text(j, i, co_occurrence[i, j], ha="center", va="center",
                  color="white" if co_occurrence[i, j] > co_occurrence.max() / 2 else "black")
plt.colorbar(im, label="Number of clips")
plt.tight_layout()
plt.savefig("emotion_cooccurrence.png", dpi=150)
plt.show()
print("Saved emotion_cooccurrence.png")