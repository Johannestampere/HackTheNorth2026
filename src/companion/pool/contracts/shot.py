"""The contract between planning and projection. Contains no pixel coordinates."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from .geometry import Point2, Segment2, UnitVector2


class GuideRole(str, Enum):
    CUE_ALIGNMENT = "cue_alignment"
    CUE_BALL_BEFORE_CONTACT = "cue_ball_before_contact"
    CUE_BALL_AFTER_CONTACT = "cue_ball_after_contact"
    OBJECT_BALL_PATH = "object_ball_path"


@dataclass(frozen=True)
class CueAim:
    """Anchor a forward unit strike direction at the observed cue-ball center.

    ``origin`` is copied from the ball's position in table meters; ``cue_ball_id``
    identifies that ball. Direction describes aim, not power or speed.
    """

    cue_ball_id: str
    origin: Point2
    direction: UnitVector2

    def __post_init__(self) -> None:
        if not self.cue_ball_id:
            raise ValueError("Cue aim must identify the cue ball")


@dataclass(frozen=True)
class GuideSegment:
    """A physical line segment plus its purpose, so projection can style it."""

    role: GuideRole
    segment: Segment2

    def __post_init__(self) -> None:
        if not isinstance(self.role, GuideRole):
            raise ValueError("Guide role must be a GuideRole enum")


@dataclass(frozen=True)
class ShotPlan:
    """Planner output for one table observation, expressed entirely in table meters.

    V1 supplies cue aim; additional guide segments and ghost-ball/target fields
    describe V2 guidance. Optional speed is initial cue-ball speed in m/s.
    Source IDs let the pipeline reject plans for the wrong observation or table.
    """

    observation_id: str
    table_id: str
    cue_aim: CueAim
    guides: tuple[GuideSegment, ...] = ()
    ghost_ball: Point2 | None = None
    target_ball_id: str | None = None
    target_pocket_id: str | None = None
    suggested_cue_ball_speed_mps: float | None = None

    def __post_init__(self) -> None:
        if not self.observation_id or not self.table_id:
            raise ValueError("A shot must identify its source observation and table frame")
        if self.suggested_cue_ball_speed_mps is not None and (
            not isfinite(self.suggested_cue_ball_speed_mps)
            or self.suggested_cue_ball_speed_mps <= 0
        ):
            raise ValueError("Suggested cue-ball speed must be positive and finite")
