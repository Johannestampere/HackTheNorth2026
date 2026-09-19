from typing import Protocol

from companion.pool.contracts import ShotPlan

from .models import ProjectionFrame, ProjectionTarget


class ShotRenderer(Protocol):
    def render(self, plan: ShotPlan, target: ProjectionTarget) -> ProjectionFrame:
        """Render shot geometry for the supplied calibrated pose; do not move hardware."""
        ...
