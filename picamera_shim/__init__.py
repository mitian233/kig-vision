"""picamera_shim — a tiny compatibility shim exposing a subset of the
picamera API implemented on top of linuxpy's video device.

This package is intentionally named differently from the upstream
`picamera` to avoid collisions with an installed picamera distribution.
Use this shim when you want to run the project on non-RPi hardware that
provides a V4L2 device via `linuxpy`.
"""

from __future__ import annotations

import threading
import logging
from typing import Optional

from pathlib import Path

try:
    # our helper that detects preferred device
    from pythonv4l2 import preferred_device_factory, get_preferred_video_path
except Exception:  # pragma: no cover - environment dependent
    preferred_device_factory = None

    def get_preferred_video_path():
        return None


LOG = logging.getLogger(__name__)


class Preview:
    def __init__(self):
        # only resolution is used by main.py
        self.resolution = None


class PiCamera:
    def __init__(self):
        if preferred_device_factory is None:
            raise ImportError(
                "linuxpy-based picamera shim requires pythonv4l2.preferred_device_factory"
            )

        self._dev = preferred_device_factory()
        self._opened = False
        self.resolution = (640, 480)
        self.framerate = 30
        self.vflip = False
        self.hflip = False
        self.preview = Preview()
        self._recording = False
        self._record_thread: Optional[threading.Thread] = None
        self._record_stop = threading.Event()

    def _open(self):
        if not self._opened:
            # The Device returned by linuxpy is a context manager; call __enter__
            # to actually open the device for use.
            try:
                self._dev.__enter__()
                self._opened = True
            except Exception:
                LOG.exception("failed to open video device")
                raise

    def start_preview(self):
        # No real preview support; just ensure device is opened and set preview resolution
        self._open()
        if self.preview.resolution is None:
            self.preview.resolution = self.resolution

    def stop_preview(self):
        # nothing to do; keep device open for captures
        return

    def capture(self, output_path: str):
        """Capture a single frame and write to output_path as JPEG if possible.

        The linuxpy frame produced is a raw buffer (format depends on driver).
        If it's already JPEG we write it directly, otherwise we attempt a best-effort
        conversion using Pillow if available.
        """
        self._open()
        # pull a single frame from the device iterator
        try:
            frame = next(iter(self._dev))
        except Exception as e:
            LOG.exception("failed capturing frame: %s", e)
            raise

        # Try to detect JPEG header
        if isinstance(frame, (bytes, bytearray)) and frame[:3] == b"\xff\xd8\xff":
            Path(output_path).write_bytes(frame)
            return

        # Fallback: try Pillow to convert raw frame to JPEG if available. This is best-effort.
        try:
            from PIL import Image
            import numpy as np

            # Attempt to interpret the frame as a flat YUV420 buffer or RGB. This is heuristic.
            w, h = self.resolution
            # try RGB first
            arr = np.frombuffer(frame, dtype=np.uint8)
            if arr.size == w * h * 3:
                img = Image.frombytes("RGB", (w, h), arr.tobytes())
            else:
                # try YCbCr (pack into Image)
                img = Image.frombytes("YCbCr", (w, h), arr.tobytes())
            img.save(output_path, format="JPEG")
            return
        except Exception:
            # Last-resort: write raw bytes
            Path(output_path).write_bytes(
                frame if isinstance(frame, (bytes, bytearray)) else bytes(frame)
            )

    @property
    def recording(self) -> bool:
        return self._recording

    def start_recording(self, output, format: str = "h264"):
        """Start recording frames and write raw/frame data to output.

        output is expected to be a file-like object with write() and flush(),
        as used in main.py (e.g., MP4Outputer or BroadcastOutput).
        This implementation pulls frames and forwards their bytes to output.
        """
        if self._recording:
            raise Exception("Already recording")
        self._open()

        def record_loop(dev, out, stop_event):
            try:
                for frame in dev:
                    if stop_event.is_set():
                        break
                    try:
                        out.write(frame)
                    except Exception:
                        LOG.exception("failed writing frame to output")
                        break
            finally:
                try:
                    out.flush()
                except Exception:
                    pass

        self._record_stop.clear()
        self._record_thread = threading.Thread(
            target=record_loop, args=(self._dev, output, self._record_stop), daemon=True
        )
        self._recording = True
        self._record_thread.start()

    def stop_recording(self):
        if not self._recording:
            return
        self._record_stop.set()
        if self._record_thread is not None:
            self._record_thread.join(timeout=5)
        self._recording = False

    def close(self):
        try:
            # If device is a context manager open, call __exit__
            if self._opened:
                try:
                    self._dev.__exit__(None, None, None)
                except Exception:
                    LOG.exception("error closing device")
                self._opened = False
        finally:
            pass


__all__ = ["PiCamera"]
