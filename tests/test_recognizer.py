import json
import string
from pathlib import Path

import cv2
import numpy as np
import pytest

from asl_recognizer import CLASSES_PATH, ASLRecognizer

SAMPLE = Path(__file__).parents[1] / "assets" / "sample_I.jpg"


@pytest.fixture(scope="module")
def recognizer() -> ASLRecognizer:
    return ASLRecognizer()


def test_classes_are_the_alphabet_plus_three_controls():
    classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    assert classes == [*string.ascii_uppercase, "del", "nothing", "space"]


def test_preprocessing_matches_the_training_input():
    blue = np.zeros((300, 200, 3), dtype=np.uint8)
    blue[..., 0] = 255  # OpenCV images are BGR
    batch = ASLRecognizer.preprocess(blue)
    assert batch.shape == (1, 64, 64, 3)
    assert batch.dtype == np.float32
    assert batch[0, 0, 0].tolist() == [0.0, 0.0, 1.0]  # RGB order, scaled to [0, 1]


def test_model_recognises_the_letter_i(recognizer: ASLRecognizer):
    guesses = recognizer.predict(cv2.imread(str(SAMPLE)), top=3)
    assert guesses[0].label == "I"
    assert guesses[0].confidence > 0.5
    assert [g.confidence for g in guesses] == sorted((g.confidence for g in guesses), reverse=True)
