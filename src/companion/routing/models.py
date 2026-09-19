from dataclasses import dataclass
from enum import Enum


class ToolName(str, Enum):
    POOL = "pool"
    GENERALIZATION = "generalization"


@dataclass(frozen=True)
class RoutingDecision:
    """The LLM's allowed tool choice and a short explanation of that choice."""

    tool: ToolName
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.tool, ToolName):
            raise ValueError("Router must select an allowlisted ToolName")
