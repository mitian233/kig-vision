"""A lightweight picamera-compatible layer backed by OpenCV.

This implements the subset of the picamera API used by main.py:
- PiCamera class
- properties: resolution, framerate, hflip, vflip, recording
- methods: start_preview, stop_preview, capture, start_recording, stop_recording, close
- preview attribute with resolution property

Recording behavior:
- format='yuv' -> writes raw YUV420p frames to output.write()
- format='h264' -> spawns ffmpeg to encode raw BGR frames to h264 and forwards ffmpeg stdout to output.write()

This is intentionally minimal and intended to be a drop-in replacement when the `picamera` package
is not available.
"""

from __future__ import annotations

import threading
import subprocess
import time
from typing import Any, Optional, Tuple
import os

import cv2
import numpy as np


class Preview:
    def __init__(self, camera: "PiCamera"):
        self._camera = camera
        self._resolution = camera.resolution

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._resolution

    @resolution.setter
    def resolution(self, value: Tuple[int, int]):
        self._resolution = tuple(value)
        # preview resolution doesn't change the capture resolution for this simple implementation


class PiCamera:
    def __init__(self, camera_id: int = 0):
        self._camera_id = camera_id
        self._cap = cv2.VideoCapture(camera_id)
        # defaults
        self._resolution = (640, 480)
        self._framerate = 30
        self._hflip = False
        self._vflip = False
        self.preview = Preview(self)
        self._previewing = False
        self._preview_thread: Optional[threading.Thread] = None
        self._preview_stop = threading.Event()
        self._window_name = f"PiCamera-{camera_id}"
        self._preview_supported = True

        # recording internals
        self._recording = False
        self._record_thread: Optional[threading.Thread] = None
        self._record_stop = threading.Event()
        self._encoder_proc: Optional[subprocess.Popen] = None
        self._forward_thread: Optional[threading.Thread] = None

        # apply defaults to capture device
        self.resolution = self._resolution
        self.framerate = self._framerate

    def _apply_flip(self, frame: np.ndarray) -> np.ndarray:
        if self._hflip:
            frame = cv2.flip(frame, 1)
        if self._vflip:
            frame = cv2.flip(frame, 0)
        return frame

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._resolution

    @resolution.setter
    def resolution(self, value: Tuple[int, int]):
        w, h = int(value[0]), int(value[1])
        self._resolution = (w, h)
        # try to set on VideoCapture; many devices accept these
        try:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        except Exception:
            pass

    @property
    def framerate(self) -> int:
        return self._framerate

    @framerate.setter
    def framerate(self, value: int):
        self._framerate = int(value)
        try:
            self._cap.set(cv2.CAP_PROP_FPS, float(self._framerate))
        except Exception:
            pass

    @property
    def hflip(self) -> bool:
        return self._hflip

    @hflip.setter
    def hflip(self, value: bool):
        self._hflip = bool(value)

    @property
    def vflip(self) -> bool:
        return self._vflip

    @vflip.setter
    def vflip(self, value: bool):
        self._vflip = bool(value)

    @property
    def recording(self) -> bool:
        return self._recording

    def start_preview(self) -> None:
        if self._previewing:
            return
        self._previewing = True
        self._preview_stop.clear()

        def preview_loop():
            try:
                # try to create a named window; may fail in headless environments
                # quick headless check: skip preview if no DISPLAY or WAYLAND_DISPLAY
                if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
                    self._preview_supported = False
                    return
                try:
                    # Attempt to create a resizable window and set it to fullscreen.
                    # Some OpenCV builds or headless environments may not support
                    # fullscreen window properties, so we gracefully fall back.
                    cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
                    try:
                        # Try to make the window fullscreen. This may raise on
                        # certain backends; ignore errors and continue with
                        # a normal window.
                        cv2.setWindowProperty(
                            self._window_name,
                            cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_FULLSCREEN,
                        )
                    except Exception:
                        # If fullscreen isn't supported, keep WINDOW_NORMAL.
                        pass
                except Exception:
                    # no GUI available / plugin error
                    self._preview_supported = False
                    return

                while not self._preview_stop.is_set():
                    frame = self._read_frame()
                    if frame is None:
                        time.sleep(0.01)
                        continue
                    # resize to preview resolution if set
                    pr = self.preview.resolution
                    if pr is not None and (frame.shape[1], frame.shape[0]) != (
                        int(pr[0]),
                        int(pr[1]),
                    ):
                        try:
                            frame = cv2.resize(frame, (int(pr[0]), int(pr[1])))
                        except Exception:
                            pass

                    try:
                        cv2.imshow(self._window_name, frame)
                        # waitKey is required for imshow to update; small delay
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            # stop preview on 'q' press
                            break
                    except Exception:
                        # if imshow fails (headless), stop preview
                        break
            finally:
                try:
                    cv2.destroyWindow(self._window_name)
                except Exception:
                    pass

        self._preview_thread = threading.Thread(target=preview_loop, daemon=True)
        self._preview_thread.start()

    def stop_preview(self) -> None:
        if not self._previewing:
            return
        self._previewing = False
        self._preview_stop.set()
        if self._preview_thread is not None:
            self._preview_thread.join(timeout=1.0)
            self._preview_thread = None
        try:
            cv2.destroyWindow(self._window_name)
        except Exception:
            pass

    def _read_frame(self) -> Optional[np.ndarray]:
        if not self._cap or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        frame = self._apply_flip(frame)
        return frame

    def capture(self, filename: str) -> None:
        frame = self._read_frame()
        if frame is None:
            raise RuntimeError("Failed to read frame for capture")
        # write JPEG
        # ensure parent dir exists is responsibility of caller
        cv2.imwrite(filename, frame)

    def start_recording(self, output: Any, format: str = "h264") -> None:
        if self._recording:
            raise RuntimeError("Already recording")

        self._record_stop.clear()
        self._recording = True

        if format == "yuv":
            # write raw yuv420p frames directly to output.write()
            def record_loop():
                w, h = self._resolution
                # target framerate
                interval = 1.0 / max(1, self._framerate)
                while not self._record_stop.is_set():
                    frame = self._read_frame()
                    if frame is None:
                        time.sleep(0.01)
                        continue
                    # ensure frame matches requested resolution
                    try:
                        if (frame.shape[1], frame.shape[0]) != (w, h):
                            frame = cv2.resize(frame, (w, h))
                    except Exception:
                        pass
                    # convert to I420 (YUV420P)
                    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
                    try:
                        output.write(yuv.tobytes())
                    except Exception:
                        # stop on any write error
                        break
                    time.sleep(interval)

                # flush if available
                try:
                    output.flush()
                except Exception:
                    pass

            self._record_thread = threading.Thread(target=record_loop, daemon=True)
            self._record_thread.start()

        elif format == "h264":
            # use ffmpeg to encode raw BGR frames to h264 and forward stdout to output.write
            w, h = self._resolution

            cmd = [
                "ffmpeg",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{w}x{h}",
                "-r",
                str(float(self._framerate)),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-f",
                "h264",
                "-",
            ]

            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._encoder_proc = proc

            def feed_loop():
                interval = 1.0 / max(1, self._framerate)
                try:
                    while not self._record_stop.is_set():
                        frame = self._read_frame()
                        if frame is None:
                            time.sleep(0.01)
                            continue
                        # ensure frame matches requested resolution
                        try:
                            if (frame.shape[1], frame.shape[0]) != (w, h):
                                frame = cv2.resize(frame, (w, h))
                        except Exception:
                            pass
                        # write raw BGR bytes
                        try:
                            proc.stdin.write(frame.tobytes())
                        except Exception:
                            break
                        # flush to keep low latency
                        try:
                            proc.stdin.flush()
                        except Exception:
                            pass
                        time.sleep(interval)
                finally:
                    try:
                        proc.stdin.close()
                    except Exception:
                        pass

            def forward_loop():
                try:
                    while True:
                        buf = proc.stdout.read1(32768)
                        if not buf:
                            if proc.poll() is not None:
                                break
                            time.sleep(0.01)
                            continue
                        try:
                            output.write(buf)
                        except Exception:
                            break
                finally:
                    try:
                        output.flush()
                    except Exception:
                        pass

            self._record_thread = threading.Thread(target=feed_loop, daemon=True)
            self._forward_thread = threading.Thread(target=forward_loop, daemon=True)
            self._record_thread.start()
            self._forward_thread.start()

        else:
            raise ValueError(f"Unsupported recording format: {format}")

    def stop_recording(self) -> None:
        if not self._recording:
            return
        self._record_stop.set()
        # join threads
        if self._record_thread is not None:
            self._record_thread.join(timeout=2.0)
            self._record_thread = None
        if self._encoder_proc is not None:
            # wait for encoder stdout to finish being forwarded
            try:
                self._encoder_proc.wait(timeout=1.0)
            except Exception:
                try:
                    self._encoder_proc.terminate()
                except Exception:
                    pass
            self._encoder_proc = None
        if self._forward_thread is not None:
            self._forward_thread.join(timeout=1.0)
            self._forward_thread = None
        self._recording = False

    def close(self) -> None:
        try:
            self.stop_preview()
        except Exception:
            pass
        try:
            self.stop_recording()
        except Exception:
            pass
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        try:
            # destroy any remaining OpenCV windows
            cv2.destroyAllWindows()
        except Exception:
            pass


__all__ = ["PiCamera", "Preview"]
