#!/usr/bin/env python3
"""Diagnostic script to test all components of the emotion analysis app."""

import sys
from pathlib import Path

def test_imports():
    """Test that all required dependencies can be imported."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)

    deps = {
        "numpy": "numpy",
        "cv2": "opencv-python",
        "gradio": "gradio",
        "mediapipe": "mediapipe",
        "matplotlib": "matplotlib",
        "joblib": "joblib",
        "scipy": "scipy",
    }

    all_ok = True
    for name, package in deps.items():
        try:
            __import__(name)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            all_ok = False

    return all_ok


def test_model_files():
    """Test that model files exist and are valid."""
    print("\n" + "=" * 60)
    print("Testing model files...")
    print("=" * 60)

    landmarker_path = Path(__file__).parent / "src/ui/assets/face_landmarker.task"
    regressor_path = Path(__file__).parent / "src/models/artifacts/emotion_intensity_regressor_live.joblib"

    all_ok = True

    # Check landmarker
    if landmarker_path.exists():
        size_mb = landmarker_path.stat().st_size / 1024 / 1024
        print(f"✓ Face landmarker: {size_mb:.1f} MB")
        # Check if it's a real file or LFS pointer
        with open(landmarker_path, 'rb') as f:
            header = f.read(4)
            if header == b'version' or (len(header) > 0 and header[0:2] == b'PK'):
                print(f"  └─ File format looks valid")
            else:
                print(f"  └─ WARNING: File header is {header}, might be LFS pointer")
    else:
        print(f"✗ Face landmarker not found at {landmarker_path}")
        all_ok = False

    # Check regressor
    if regressor_path.exists():
        size_mb = regressor_path.stat().st_size / 1024 / 1024
        print(f"✓ Emotion regressor: {size_mb:.1f} MB")
    else:
        print(f"✗ Emotion regressor not found at {regressor_path}")
        all_ok = False

    return all_ok


def test_mediapipe():
    """Test MediaPipe FaceLandmarker initialization."""
    print("\n" + "=" * 60)
    print("Testing MediaPipe...")
    print("=" * 60)

    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
        from pathlib import Path

        landmarker_path = Path(__file__).parent / "src/ui/assets/face_landmarker.task"

        print("Initializing FaceLandmarker...")
        landmarker = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(landmarker_path)),
                running_mode=RunningMode.VIDEO,
                num_faces=1,
                output_face_blendshapes=True,
            )
        )
        print("✓ FaceLandmarker initialized successfully")
        landmarker.close()
        return True

    except Exception as e:
        print(f"✗ FaceLandmarker initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_regressor():
    """Test regressor loading and basic inference."""
    print("\n" + "=" * 60)
    print("Testing Regressor...")
    print("=" * 60)

    try:
        from joblib import load
        from pathlib import Path
        import numpy as np

        regressor_path = Path(__file__).parent / "src/models/artifacts/emotion_intensity_regressor_live.joblib"

        print("Loading regressor...")
        regressor = load(regressor_path)
        regressor.n_jobs = 1
        print("✓ Regressor loaded successfully")

        # Test with dummy input
        print("Testing inference with dummy input...")
        dummy_features = np.zeros((1, 52))  # 52 MediaPipe blendshapes
        predictions = regressor.predict(dummy_features)
        print(f"✓ Inference successful, output shape: {predictions.shape}")
        print(f"  └─ Predictions: {predictions[0]}")

        return True

    except Exception as e:
        print(f"✗ Regressor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("EMOTION ANALYSIS APP - DIAGNOSTIC TEST")
    print("=" * 60)

    results = {
        "imports": test_imports(),
        "model_files": test_model_files(),
        "mediapipe": test_mediapipe(),
        "regressor": test_regressor(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All checks passed! The app should work locally.")
        print("\nTo run the app:")
        print("  uv run python src/ui/app.py")
    else:
        print("\n✗ Some checks failed. Please fix the issues above before running the app.")
        sys.exit(1)


if __name__ == "__main__":
    main()
