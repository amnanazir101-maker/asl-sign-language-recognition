"""Recognise ASL letters in image files.

python predict_image.py assets/sample_I.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from asl_recognizer import ASLRecognizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", nargs="+", type=Path, help="image files of a single hand sign")
    parser.add_argument("--top", type=int, default=3, help="how many guesses to show (default 3)")
    args = parser.parse_args()

    recognizer = ASLRecognizer()
    for path in args.images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"{path}: could not read the image")
            continue
        guesses = recognizer.predict(image, top=args.top)
        print(f"{path.name}: " + ", ".join(f"{g.label} {g.confidence:.1%}" for g in guesses))


if __name__ == "__main__":
    main()
