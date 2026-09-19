from companion.sensors.models import CameraCapture


class GeneralizationTool:
    def run(self, first_capture: CameraCapture) -> None:
        """Output contract is intentionally deferred until this tool has a defined task."""
        raise NotImplementedError("Generalization is outside the initial pool implementation")
