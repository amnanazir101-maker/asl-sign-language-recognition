"""Load the trained CNN and recognise American Sign Language (ASL) letters in images."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "asl_cnn.h5"
CLASSES_PATH = ROOT / "models" / "classes.json"
IMG_SIZE = 64  # the network takes 64 x 64 RGB images, as in training


@dataclass
class Prediction:
    label: str  # "A"-"Z", "space", "del" or "nothing"
    confidence: float  # 0-1


class ASLRecognizer:
    """The improved CNN (4 convolutional blocks, batch norm, 1.67M parameters)."""

    def __init__(self, model_path: Path = MODEL_PATH, classes_path: Path = CLASSES_PATH) -> None:
        from tensorflow import keras  # imported here: TensorFlow takes a few seconds to load

        self.model = keras.models.load_model(model_path, compile=False)
        self.classes: list[str] = json.loads(Path(classes_path).read_text(encoding="utf-8"))

    @staticmethod
    def preprocess(image_bgr: np.ndarray) -> np.ndarray:
        """OpenCV (BGR) image -> a batch of one 64 x 64 RGB image scaled to [0, 1]."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        return np.expand_dims(resized.astype("float32") / 255.0, axis=0)

    def predict(self, image_bgr: np.ndarray, top: int = 3) -> list[Prediction]:
        """The `top` most likely letters, most likely first."""
        # Calling the model directly is much faster than model.predict() for one image.
        probabilities = np.asarray(self.model(self.preprocess(image_bgr), training=False))[0]
        best = np.argsort(probabilities)[::-1][:top]
        return [Prediction(self.classes[i], float(probabilities[i])) for i in best]
