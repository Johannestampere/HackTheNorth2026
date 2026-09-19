from companion.pool.contracts import ShotPlan

from .models import ProjectionFrame, ProjectionTarget


class ProjectionService:
    """Implementation owned by teammate 3. Rasterization/warping is intentionally unimplemented."""

    def render(self, plan: ShotPlan, target: ProjectionTarget) -> ProjectionFrame:
        raise NotImplementedError("Projection: implement calibrated mapping and guidance rendering")
