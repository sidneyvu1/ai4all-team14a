import numpy as np

data = np.load("data/CMU-MOSEI/mosei_visualopenface_only_processed.npz")
print(data)
X, y, ids = data["X"], data["y"], data["ids"]

LABEL_NAMES = ["sentiment", "happy", "sad", "anger", "surprise", "disgust", "fear"]

# 1. Drop the ~2% all-zero (face-not-detected) sequences
non_empty_mask = ~np.all(X == 0, axis=(1, 2))
X, y, ids = X[non_empty_mask], y[non_empty_mask], ids[non_empty_mask]
print("After dropping empty sequences:", X.shape, y.shape)

# 2. Split off just the 6 emotion columns (drop sentiment for now, since your task is emotion, not sentiment)
emotions_only = y[:, 1:]  # shape (N, 6)
print("Emotions-only shape:", emotions_only.shape)

# 3. Check emotion distribution, since MOSEI is known to be imbalanced (happy dominates)
for i, name in enumerate(LABEL_NAMES[1:]):
    nonzero_pct = np.mean(emotions_only[:, i] > 0) * 100
    print(f"{name}: {nonzero_pct:.1f}% of clips have nonzero intensity")

# 4. Train/val/test split
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, emotions_only, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("\nTrain:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

np.savez("data/mosei_video_splits.npz",
         X_train=X_train, y_train=y_train,
         X_val=X_val, y_val=y_val,
         X_test=X_test, y_test=y_test)