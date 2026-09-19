"""Public pool boundary types. Coordinate and serialization rules: docs/contracts.md."""

from .geometry import Point2, Segment2, UnitVector2
from .results import (
    InsufficientInformation, NeedsMoreViews, NoFeasibleShot,
    PerceptionReady, PerceptionResult, PlanningReady, PlanningResult, UnusableCapture,
)
from .shot import CueAim, GuideRole, GuideSegment, ShotPlan
from .table import (
    Ball, BallType, CoverageStatus, GameContext, GameMode,
    PlayerGroup, Pocket, TableGeometry, TableState,
)

__all__ = [
    "Point2", "Segment2", "UnitVector2", "Ball", "BallType", "CoverageStatus",
    "GameContext", "GameMode", "PlayerGroup", "Pocket", "TableGeometry", "TableState",
    "CueAim", "GuideRole", "GuideSegment", "ShotPlan", "InsufficientInformation",
    "NeedsMoreViews", "NoFeasibleShot", "PerceptionReady", "PerceptionResult",
    "PlanningReady", "PlanningResult", "UnusableCapture",
]
