import numpy as np

data = np.load("data/CMU-MOSEI/mosei_video_only_processed.npz")
print("Keys:", list(data.keys()))

X = data["X"]
y = data["y"]
ids = data["ids"]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Number of ids:", len(ids))

print("\nSample id:", ids[0])
print("Sample y row (should be 7 values: sentiment + 6 emotions):", y[0])
print("Sample X row, first timestep (should be 35 values):", X[0][0])

print("\nAny all-zero X sequences? (would indicate empty slices):", np.sum(np.all(X == 0, axis=(1,2))))
print("Any NaN in y?", np.isnan(y).any())
print("Any NaN in X?", np.isnan(X).any())