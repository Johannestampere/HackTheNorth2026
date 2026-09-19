from typing import Protocol

from .models import CameraCapture


class Camera(Protocol):
    def capture(self) -> CameraCapture:
        """Acquire an RGB image with its timestamp and calibrated view ID."""
        ...
