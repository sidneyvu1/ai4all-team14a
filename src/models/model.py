"""Lightweight multi-output regressor mapping pooled facial features to emotion intensity."""

from sklearn.ensemble import RandomForestRegressor

EMOTIONS = ["happy", "sad", "anger", "surprise", "disgust", "fear"]


def pool_features(X):
    """Collapse a (N, timesteps, features) clip sequence into (N, features) via per-feature max.

    Max (not mean) is used because the emotion-intensity labels reflect the peak
    expression observed during a clip, not its average.
    """
    return X.max(axis=1)


def build_model():
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
