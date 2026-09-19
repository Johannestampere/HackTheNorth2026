"""The contract between planning and projection. Contains no pixel coordinates."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from .geometry import Point2, Segment2, UnitVector2


class GuideRole(str, Enum):
    """Meaning of a segment, independent of the renderer's colors or stroke widths."""

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
    """One drawable segment in table meters, associated with an observed ball.

    ``ball_id`` identifies the moving ball for a trajectory. For CUE_ALIGNMENT
    it identifies the cue ball being aimed at; the segment lies behind that ball
    and points toward it. Other roles describe ball-center motion from start to
    end. Multiple segments for the same ball appear in travel order in guides.
    The renderer styles the role and must not infer ball identity from the order.
    """

    role: GuideRole
    ball_id: str
    segment: Segment2

    def __post_init__(self) -> None:
        if not isinstance(self.role, GuideRole):
            raise ValueError("Guide role must be a GuideRole enum")
        if not self.ball_id:
            raise ValueError("A guide must identify its observed ball")


@dataclass(frozen=True)
class ShotPlan:
    """One recommended pot attempt, chosen to improve the shooter's rack outcome.

    ``cue_aim`` anchors a forward UNIT direction at the observed cue-ball center.
    ``cue_stick_speed_mps`` is the cue tip's linear speed immediately before
    impact, in m/s: NOT initial cue-ball speed and NOT a unitless power percentage.
    The MVP assumes a level, center-ball strike (no intentional tip offset).
    Physics may generate spin after impact; those assumptions do not mean balls
    never spin. Predicted paths are conditional on this strike and the model.

    ``target_ball_id`` and ``target_pocket_id`` identify the intended called pot,
    including the eight ball when legal. The MVP does not encode safety shots.
    ``guides`` are optional nominal predicted paths/alignment in table meters;
    an empty tuple means display aim only, not that nothing moves. ``ghost_ball``
    is an optional cue-ball center at first contact for a direct shot.

    IDs bind the plan to its source state/frame. A recommendation is neither a
    guarantee of a pot nor a calibrated win probability. Search scores, trial
    samples, and simulator objects stay private to planning.
    """

    observation_id: str
    table_id: str
    cue_aim: CueAim
    cue_stick_speed_mps: float
    target_ball_id: str
    target_pocket_id: str
    guides: tuple[GuideSegment, ...] = ()
    ghost_ball: Point2 | None = None

    def __post_init__(self) -> None:
        if not self.observation_id or not self.table_id:
            raise ValueError("A shot must identify its source observation and table frame")
        if not self.target_ball_id or not self.target_pocket_id:
            raise ValueError("A pot attempt must identify its called ball and pocket")
        # Keep strike speed separate from direction so renderers never treat
        # vector length as power or confuse cue-stick speed with cue-ball speed.
        if not isfinite(self.cue_stick_speed_mps) or self.cue_stick_speed_mps <= 0:
            raise ValueError("Cue-stick speed must be positive and finite")
