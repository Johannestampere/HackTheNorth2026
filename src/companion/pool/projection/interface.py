from typing import Protocol

from companion.pool.contracts import ShotPlan

from .models import ProjectionFrame, ProjectionTarget


class ShotRenderer(Protocol):
    def render(self, plan: ShotPlan, target: ProjectionTarget) -> ProjectionFrame:
        """Render the supplied plan at its calibrated pose; do not move hardware.

        cue_aim is the authoritative alignment even when guides is empty. Guide
        coordinates are ball-center paths in meters, associated by ball_id.
        cue_stick_speed_mps is cue-tip speed, not ball speed or a power percent;
        keep it separate from line length. Do not invent omitted trajectories.
        """
        ...
