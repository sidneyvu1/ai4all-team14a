"""Gradio webcam UI shell.

Captures webcam frames and overlays detected MediaPipe face landmarks so the
capture pipeline can be verified visually. Emotion-intensity prediction is not
wired in yet: the trained regressor (src/models/artifacts/) was trained on
CMU-MOSEI's OpenFace2 features, which live MediaPipe landmarks don't directly
match, so that bridge is still an open decision (see project notes).
"""

from pathlib import Path

import cv2
import gradio as gr
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

MODEL_PATH = Path(__file__).parent / "assets" / "face_landmarker.task"

_landmarker = FaceLandmarker.create_from_options(
    FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
    )
)


def draw_landmarks(frame, landmarks):
    height, width = frame.shape[:2]
    for landmark in landmarks:
        x, y = int(landmark.x * width), int(landmark.y * height)
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    return frame


def process_frame(frame):
    if frame is None:
        return None

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = _landmarker.detect(mp_image)

    annotated = frame.copy()
    for face_landmarks in result.face_landmarks:
        annotated = draw_landmarks(annotated, face_landmarks)
    return annotated


demo = gr.Interface(
    fn=process_frame,
    inputs=gr.Image(sources=["webcam"], streaming=True, type="numpy"),
    outputs=gr.Image(type="numpy"),
    live=True,
    title="Neuromarketing Ad-Testing — Face Landmark Capture",
    description="Live webcam feed with MediaPipe face landmarks overlaid.",
)

if __name__ == "__main__":
    demo.launch()
