from companion.pool.contracts import PerceptionResult, TableGeometry
from companion.sensors.models import CaptureBatch


class PerceptionService:
    """Implementation owned by teammate 1. See docs/team-guide.md."""

    def estimate(self, captures: CaptureBatch, geometry: TableGeometry) -> PerceptionResult:
        raise NotImplementedError("Perception: implement detection, localization, and coverage checks")
