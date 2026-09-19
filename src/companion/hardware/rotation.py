from typing import Protocol


class RotationController(Protocol):
    def rotate_to(self, angle_degrees: float) -> None:
        """Move to an absolute base angle in [0, 360), then wait for settling."""
        ...

    def current_pose_id(self) -> str:
        """Return the calibrated pose ID only after reaching a known, repeatable pose."""
        ...
