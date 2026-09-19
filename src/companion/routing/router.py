from typing import Protocol

from companion.sensors.models import CameraCapture

from .models import RoutingDecision


class TaskRouter(Protocol):
    def route(self, first_capture: CameraCapture) -> RoutingDecision: ...


class LLMRouter:
    def route(self, first_capture: CameraCapture) -> RoutingDecision:
        raise NotImplementedError("Select a multimodal LLM and parse its tool choice into RoutingDecision")
