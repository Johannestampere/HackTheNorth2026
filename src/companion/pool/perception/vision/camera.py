"""Webcam access for the calibration pipeline."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from types import TracebackType

import cv2
import numpy as np

WARMUP_FRAMES = 10
MAX_PROBE_INDEX = 8

# Preferred external webcams, matched case-insensitively against the OS device
# names. An external camera on a stand sees the table far better than a laptop
# lid camera, so it wins when both are present.
PREFERRED_NAMES = ("logitech", "c925", "c920", "c922", "brio", "webcam")


class CameraError(RuntimeError):
    """The camera could not be opened or would not deliver frames."""


@dataclass
class CameraConfig:
    index: int | None = None  # None selects the preferred camera automatically
    width: int = 1280
    height: int = 720
    autofocus: bool = False
    auto_exposure: bool = False


def list_device_names() -> list[str]:
    """OS camera device names, in the order OpenCV indexes them.

    Windows only (via PowerShell/CIM); returns an empty list elsewhere, which
    simply disables name-based selection.
    """
    if sys.platform != "win32":
        return []
    script = (
        "Get-CimInstance Win32_PnPEntity | "
        "Where-Object { $_.PNPClass -eq 'Camera' -or $_.Service -eq 'usbvideo' } | "
        "ForEach-Object { $_.Name }"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _is_blank(frame: np.ndarray) -> bool:
    """True if the frame carries no real image content.

    A camera disabled in hardware (privacy shutter, or the laptop's camera
    kill key) still opens and still returns frames - flat grey ones with the
    Windows 'camera off' glyph. Without this check the pipeline would just
    report that the paper was not found and leave the real cause
    unexplained.
    """
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.std()) < 8.0


def find_camera_index(width: int = 1280, height: int = 720) -> tuple[int, str]:
    """Pick the best working camera index, preferring an external webcam.

    Returns (index, reason). Raises CameraError if nothing usable is found.
    """
    names = list_device_names()
    preferred_order = sorted(
        range(len(names)),
        key=lambda i: next(
            (rank for rank, needle in enumerate(PREFERRED_NAMES)
             if needle in names[i].lower()), len(PREFERRED_NAMES)))

    probe = preferred_order + [i for i in range(MAX_PROBE_INDEX)
                               if i not in preferred_order]
    blank: list[int] = []
    for index in probe:
        cap = cv2.VideoCapture(index, _backend())
        try:
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            for _ in range(WARMUP_FRAMES):
                cap.read()
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if _is_blank(frame):
                blank.append(index)
                continue
        finally:
            cap.release()
        name = names[index] if index < len(names) else f"index {index}"
        return index, name

    if blank:
        raise CameraError(
            f"camera index {blank[0]} opens but only returns blank frames - it "
            "is switched off in hardware. Check the privacy shutter, the "
            "camera key (often F8/F9), and Windows Settings > Privacy & "
            "security > Camera.")
    raise CameraError(
        "no working camera found. Check the webcam is plugged in and not in "
        "use by another app (Zoom, Teams, the Camera app).")


def _backend() -> int:
    # CAP_DSHOW avoids the slow MSMF startup path on Windows; ignored elsewhere.
    return cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY


class Camera:
    """A webcam opened at a fixed resolution with auto-adjust turned off.

    Autofocus and auto-exposure are disabled on a best-effort basis: many UVC
    webcams ignore these properties, and that is not fatal, so failures are
    reported through `notes` rather than raised.
    """

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self.notes: list[str] = []
        self.cap: cv2.VideoCapture | None = None
        self.index: int = self.config.index if self.config.index is not None else -1

    def open(self) -> "Camera":
        cfg = self.config
        if cfg.index is None:
            index, name = find_camera_index(cfg.width, cfg.height)
            self.index = index
            self.notes.append(f"using camera {index}: {name}")
        else:
            self.index = cfg.index

        cap = cv2.VideoCapture(self.index, _backend())
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise CameraError(
                f"cannot open camera index {self.index}. Check that the webcam "
                "is connected, not in use by another app, and that the index "
                "is right (run with no --camera flag to auto-select)."
            )
        self.cap = cap

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self._best_effort(cv2.CAP_PROP_AUTOFOCUS, 0 if not cfg.autofocus else 1,
                          "autofocus")
        # 0.25 is the widely-used "manual" value for CAP_PROP_AUTO_EXPOSURE on
        # V4L2/DSHOW backends; 0.75 means auto.
        self._best_effort(cv2.CAP_PROP_AUTO_EXPOSURE,
                          0.25 if not cfg.auto_exposure else 0.75, "auto-exposure")

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) != (cfg.width, cfg.height):
            self.notes.append(
                f"camera gave {actual_w}x{actual_h}, not the requested "
                f"{cfg.width}x{cfg.height}")

        for _ in range(WARMUP_FRAMES):  # let exposure and focus settle
            cap.read()

        ok, frame = cap.read()
        if ok and frame is not None and _is_blank(frame):
            self.release()
            raise CameraError(
                f"camera {self.index} opens but returns blank frames - it is "
                "switched off in hardware. Check the privacy shutter, the "
                "camera key (often F8/F9), and Windows Settings > Privacy & "
                "security > Camera.")
        return self

    def _best_effort(self, prop: int, value: float, label: str) -> None:
        try:
            if not self.cap.set(prop, value):
                self.notes.append(f"could not disable {label} (unsupported)")
        except cv2.error:
            self.notes.append(f"could not disable {label} (unsupported)")

    def read(self) -> np.ndarray:
        if self.cap is None:
            raise CameraError("camera is not open")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise CameraError("camera stopped delivering frames")
        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.release()
