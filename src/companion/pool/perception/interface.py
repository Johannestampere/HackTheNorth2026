from typing import Protocol

from companion.pool.contracts import PerceptionResult, TableGeometry
from companion.sensors.models import CaptureBatch


class TablePerception(Protocol):
    def estimate(self, captures: CaptureBatch, geometry: TableGeometry) -> PerceptionResult:
        """Localize/classify balls or explain why the captures are insufficient."""
        ...
