"""Simple camera preview utility.

Usage: run this file with Python. It opens a window showing the camera feed.
Optional command-line args (see --help): device index, width, height, flip.
"""

from __future__ import annotations

import argparse
import sys
import signal
from typing import Optional, Tuple

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Open a camera preview window using OpenCV")
    p.add_argument(
        "-d", "--device", type=int, default=0, help="camera device index (default: 0)"
    )
    p.add_argument("--width", type=int, default=640, help="frame width (default: 640)")
    p.add_argument(
        "--height", type=int, default=480, help="frame height (default: 480)"
    )
    p.add_argument(
        "--flip",
        choices=["none", "h", "v", "hv"],
        default="none",
        help="flip frames horizontally/vertically",
    )
    p.add_argument("--window-name", default="Camera Preview", help="OpenCV window name")
    return p.parse_args()


def set_capture_resolution(cap: cv2.VideoCapture, size: Tuple[int, int]) -> None:
    w, h = size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))


def apply_flip(frame, mode: str):
    if mode == "h":
        return cv2.flip(frame, 1)
    if mode == "v":
        return cv2.flip(frame, 0)
    if mode == "hv":
        return cv2.flip(frame, -1)
    return frame


def open_preview(
    device: int = 0,
    size: Tuple[int, int] = (640, 480),
    flip: str = "none",
    window_name: str = "Camera Preview",
) -> None:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera device {device}")

    set_capture_resolution(cap, size)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    running = True

    def _handle_signal(signum, frame):
        nonlocal running
        running = False

    # handle Ctrl+C cleanly
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while running:
            ret, frame = cap.read()
            if not ret or frame is None:
                # small sleep to avoid busy loop if camera disconnects
                cv2.waitKey(100)
                continue

            frame = apply_flip(frame, flip)

            cv2.imshow(window_name, frame)

            # waitKey returns -1 if no key was pressed; also required for imshow to update
            k = cv2.waitKey(1) & 0xFF
            # press 'q' or Esc to quit
            if k == ord("q") or k == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args() if argv is None else parse_args()

    try:
        open_preview(
            device=args.device,
            size=(args.width, args.height),
            flip=args.flip,
            window_name=args.window_name,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
