from typing import Protocol

class Projector(Protocol):
    def project(self, rgb: bytes, width_px: int, height_px: int) -> None:
        """Display at native resolution with scaling/keystone behavior matching calibration."""
        ...

    def blank(self) -> None:
        """Output black before camera acquisition or robot movement."""
        ...
