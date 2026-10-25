"""Real-time ASL alphabet recognition from a webcam with OpenCV.

python detect_webcam.py        # show a sign inside the green box; q = quit, s = screenshot
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from asl_recognizer import ASLRecognizer

WINDOW = "ASL Sign Language Recognition"
BOX = (320, 100, 584, 364)  # x1, y1, x2, y2: a 264 x 264 region on a 640 x 480 frame
GREEN, RED, WHITE = (0, 200, 0), (0, 0, 255), (255, 255, 255)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    parser.add_argument(
        "--threshold", type=float, default=0.70,
        help="minimum confidence before a letter is shown (default 0.70)",
    )  # fmt: skip
    parser.add_argument("--every", type=int, default=5, help="predict every N frames (default 5)")
    args = parser.parse_args()

    print("Loading the model...")
    recognizer = ASLRecognizer()
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try --camera 1.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Show a sign inside the green box. Press q to quit, s to save a screenshot.")

    x1, y1, x2, y2 = BOX
    predictions, frame_no, fps, last = [], 0, 0.0, time.perf_counter()
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror view, like looking in a mirror
            if frame_no % args.every == 0:
                # Crop before drawing on the frame, so the model never sees the green box.
                predictions = recognizer.predict(frame[y1:y2, x1:x2])

            cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
            best = predictions[0] if predictions else None
            if best and best.confidence >= args.threshold:
                cv2.putText(frame, f"{best.label} ({best.confidence:.0%})", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, GREEN, 3)  # fmt: skip
            else:
                cv2.putText(frame, "Show a sign in the box", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)  # fmt: skip
            runners_up = "   ".join(f"{p.label} {p.confidence:.0%}" for p in predictions[1:])
            cv2.putText(frame, runners_up, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            cv2.putText(frame, f"{fps:.0f} FPS | q: quit | s: screenshot", (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)  # fmt: skip
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("s"):
                folder = Path("screenshots")
                folder.mkdir(exist_ok=True)
                path = folder / f"asl_{frame_no}.jpg"
                cv2.imwrite(str(path), frame)
                print(f"Saved {path}")
            frame_no += 1
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
