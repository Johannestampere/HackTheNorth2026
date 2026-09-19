"""Expected stage outcomes. Invalid contracts still raise ValueError."""

from dataclasses import dataclass

from .geometry import Point2
from .shot import ShotPlan
from .table import CoverageStatus, TableState


@dataclass(frozen=True)
class PerceptionReady:
    """A complete table observation that can be handed to the planner."""

    state: TableState

    def __post_init__(self) -> None:
        if self.state.coverage is not CoverageStatus.COMPLETE:
            raise ValueError("PerceptionReady requires complete coverage; request more views otherwise")


@dataclass(frozen=True)
class NeedsMoreViews:
    """Request more captures, optionally identifying missing table-region polygons."""

    reason: str
    missing_regions: tuple[tuple[Point2, ...], ...] = ()
    partial_state: TableState | None = None


@dataclass(frozen=True)
class UnusableCapture:
    """Explain why the input cannot yield a usable state, such as ball motion."""

    reason: str


@dataclass(frozen=True)
class PlanningReady:
    """A recommended shot ready for the projection stage."""

    plan: ShotPlan


@dataclass(frozen=True)
class InsufficientInformation:
    """Explain what information the planner is missing before it can choose a shot."""

    reason: str


@dataclass(frozen=True)
class NoFeasibleShot:
    """No acceptable shot was found under the planner's current model and search."""

    reason: str


PerceptionResult = PerceptionReady | NeedsMoreViews | UnusableCapture
PlanningResult = PlanningReady | InsufficientInformation | NoFeasibleShot
