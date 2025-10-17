"""Utility to detect the preferred USB V4L2 device and export helpers for other modules.

This module attempts to detect which /dev/video* node corresponds to a USB
camera (by inspecting sysfs). It does NOT open the device at import time.
Other modules can import `PREFERRED_VIDEO_DEVICE_PATH` or call
`preferred_device_factory()` to get a linuxpy Device instance for use with
`with`.
"""

from __future__ import annotations

import glob
import os
import logging
from typing import Optional, Tuple

try:
    from linuxpy.video.device import Device
except Exception:  # pragma: no cover - linuxpy may not be installed on analysis machine
    Device = None  # type: ignore

LOG = logging.getLogger(__name__)


def _is_usb_video_node(dev_path: str) -> Tuple[bool, Optional[str]]:
    """Check sysfs to determine whether a /dev/videoN node is on a USB bus.

    Returns (is_usb, detail) where detail is a realpath or modalias content when available.
    """
    name = os.path.basename(dev_path)
    sys_path = f"/sys/class/video4linux/{name}/device"
    try:
        if not os.path.exists(sys_path):
            return False, None
        real = os.path.realpath(sys_path)
        # Heuristic: if the resolved path contains 'usb' it's very likely a USB device
        if "usb" in real.lower():
            return True, real
        modalias_file = os.path.join(sys_path, "modalias")
        if os.path.exists(modalias_file):
            try:
                with open(modalias_file, "r", encoding="utf-8", errors="ignore") as f:
                    modalias = f.read().lower()
                if "usb" in modalias:
                    return True, modalias.strip()
            except Exception:
                pass
        return False, real
    except Exception as e:
        LOG.debug("Failed probing %s: %s", dev_path, e)
        return False, None


def _detect_preferred_video_node() -> Optional[str]:
    """Detect the preferred /dev/video* node.

    Strategy:
    1. Prefer the first video node whose sysfs 'device' path or modalias mentions 'usb'.
    2. Fallback to the first existing /dev/video* node.
    """
    nodes = sorted(glob.glob("/dev/video*"))
    if not nodes:
        return None
    # prefer USB
    for node in nodes:
        is_usb, detail = _is_usb_video_node(node)
        if is_usb:
            LOG.info("Selected USB video node %s (detail=%s)", node, detail)
            return node
    # fallback
    LOG.info("No USB-specific video node found; falling back to %s", nodes[0])
    return nodes[0]


# Compute once at import time the preferred device path (may be None)
PREFERRED_VIDEO_DEVICE_PATH: Optional[str] = _detect_preferred_video_node()


def get_preferred_video_path() -> Optional[str]:
    """Return the detected preferred /dev/video path or None if none found."""
    return PREFERRED_VIDEO_DEVICE_PATH


def preferred_device_factory():
    """Return a linuxpy.video.device.Device for the preferred device.

    The caller should use the returned object as a context manager, e.g.:

            dev = preferred_device_factory()
            with dev:
                    for frame in dev:
                            ...

    Raises FileNotFoundError if no video node was detected or RuntimeError if
    linuxpy is not available.
    """
    if PREFERRED_VIDEO_DEVICE_PATH is None:
        raise FileNotFoundError("no /dev/video* nodes found on the system")
    if Device is None:
        raise RuntimeError(
            "linuxpy.video.device.Device is not available (linuxpy not installed)"
        )

    # Prefer a from_path constructor if provided by the installed linuxpy
    if hasattr(Device, "from_path"):
        return Device.from_path(PREFERRED_VIDEO_DEVICE_PATH)

    # Otherwise fall back to extracting the numeric id and using from_id
    base = os.path.basename(PREFERRED_VIDEO_DEVICE_PATH)
    try:
        idx = int(base.replace("video", ""))
    except Exception as e:  # pragma: no cover - defensive
        raise RuntimeError(
            f"failed to parse video device index from {PREFERRED_VIDEO_DEVICE_PATH}: {e}"
        )
    if hasattr(Device, "from_id"):
        return Device.from_id(idx)
    # As a last resort try instantiating Device with the path
    try:
        return Device(PREFERRED_VIDEO_DEVICE_PATH)  # type: ignore
    except Exception as e:  # pragma: no cover - defensive
        raise RuntimeError(
            f"cannot create Device for {PREFERRED_VIDEO_DEVICE_PATH}: {e}"
        )


__all__ = [
    "PREFERRED_VIDEO_DEVICE_PATH",
    "get_preferred_video_path",
    "preferred_device_factory",
]
