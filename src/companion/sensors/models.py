"""File-backed captures keep the initial boundary independent of camera SDKs."""

from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path

from companion.serialization import read_document


@dataclass(frozen=True)
class CameraCapture:
    """One RGB image, its acquisition time, and references to camera calibration.

    Perception estimates depth from this image internally; there is no depth sensor.
    ``view_id`` identifies the camera pose used for this capture.
    """

    capture_id: str
    captured_at_s: float
    rgb_path: str
    calibration_id: str
    view_id: str

    def __post_init__(self) -> None:
        if not all((self.capture_id, self.rgb_path, self.calibration_id, self.view_id)):
            raise ValueError("A capture needs an ID, RGB path, calibration ID, and view ID")
        if not isfinite(self.captured_at_s) or self.captured_at_s < 0:
            raise ValueError("Capture time must be finite Unix seconds")


@dataclass(frozen=True)
class CaptureBatch:
    """One or more RGB views to combine into a single stationary table observation."""

    batch_id: str
    captures: tuple[CameraCapture, ...]

    def __post_init__(self) -> None:
        if not self.batch_id or not self.captures:
            raise ValueError("A capture batch needs an ID and at least one capture")
        if len({c.capture_id for c in self.captures}) != len(self.captures):
            raise ValueError("Capture IDs must be unique within a batch")


def load_capture_batch(path: Path) -> CaptureBatch:
    """Resolve image paths against the manifest directory, not the process cwd."""
    fields = read_document(path, "capture_batch")
    captures = []
    for data in fields["captures"]:
        capture = CameraCapture(**data)
        captures.append(replace(
            capture,
            rgb_path=str((path.parent / capture.rgb_path).resolve()),
        ))
    fields["captures"] = tuple(captures)
    return CaptureBatch(**fields)
