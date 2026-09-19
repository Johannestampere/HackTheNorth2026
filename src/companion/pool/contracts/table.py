"""The contract between perception and planning. No image or ML types."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from .geometry import Point2


class BallType(str, Enum):
    CUE = "cue"
    SOLID = "solid"
    STRIPE = "stripe"
    EIGHT = "eight"
    UNKNOWN = "unknown"


class CoverageStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Ball:
    """One observed ball: ID, center position in table meters, and classified type.

    Optional confidence describes the type prediction; optional uncertainty is
    position error in meters. None means unavailable, not perfect certainty.
    """

    id: str
    position: Point2
    type: BallType
    type_confidence: float | None = None
    position_uncertainty_m: float | None = None

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.type, BallType):
            raise ValueError("A ball needs an ID and a BallType enum")
        if self.type_confidence is not None and not 0 <= self.type_confidence <= 1:
            raise ValueError("Type confidence must be in [0, 1]")
        if self.position_uncertainty_m is not None and (
            not isfinite(self.position_uncertainty_m) or self.position_uncertainty_m < 0
        ):
            raise ValueError("Position uncertainty must be finite and nonnegative")


@dataclass(frozen=True)
class Pocket:
    """A target pocket's ID, mouth-center position, and mouth width in meters."""

    id: str
    position: Point2
    mouth_width_m: float

    def __post_init__(self) -> None:
        if not self.id or not isfinite(self.mouth_width_m) or self.mouth_width_m <= 0:
            raise ValueError("A pocket needs an ID and positive finite mouth width")


@dataclass(frozen=True)
class TableGeometry:
    """Measured playing-surface dimensions, ball radius, and six pocket locations.

    All lengths are meters. ``table_id`` also identifies the fixed origin and
    axes, so every stage interprets positions in the same coordinate frame.
    """

    table_id: str
    length_m: float
    width_m: float
    ball_radius_m: float
    pockets: tuple[Pocket, ...]

    def __post_init__(self) -> None:
        if not self.table_id:
            raise ValueError("table_id identifies the table AND its coordinate frame")
        if not all(isfinite(v) and v > 0 for v in (
            self.length_m, self.width_m, self.ball_radius_m
        )):
            raise ValueError("Table dimensions and ball radius must be positive and finite")
        if 2 * self.ball_radius_m >= min(self.length_m, self.width_m):
            raise ValueError("Ball diameter must be smaller than the playing surface")
        if len(self.pockets) != 6 or len({p.id for p in self.pockets}) != 6:
            raise ValueError("The current pool contract requires six uniquely identified pockets")


@dataclass(frozen=True)
class TableState:
    """Perception's ball map for one observation, passed to the shot planner.

    Capture times are Unix seconds spanning all contributing views. Coverage
    records whether the full playing area was observed; missing space is not empty.
    """

    observation_id: str
    capture_started_at_s: float
    capture_ended_at_s: float
    geometry: TableGeometry
    balls: tuple[Ball, ...]
    coverage: CoverageStatus

    def __post_init__(self) -> None:
        if not self.observation_id or not isinstance(self.coverage, CoverageStatus):
            raise ValueError("An observation needs an ID and a CoverageStatus enum")
        if not all(isfinite(v) and v >= 0 for v in (
            self.capture_started_at_s, self.capture_ended_at_s
        )) or self.capture_ended_at_s < self.capture_started_at_s:
            raise ValueError("Capture timestamps must be an ordered, finite Unix-time interval")
        if len({ball.id for ball in self.balls}) != len(self.balls):
            raise ValueError("Ball IDs must be unique within an observation")
        if sum(ball.type is BallType.CUE for ball in self.balls) > 1:
            raise ValueError("An observation cannot contain multiple cue balls")
        if sum(ball.type is BallType.EIGHT for ball in self.balls) > 1:
            raise ValueError("An observation cannot contain multiple eight balls")
        for ball in self.balls:
            if not (0 <= ball.position.x <= self.geometry.length_m
                    and 0 <= ball.position.y <= self.geometry.width_m):
                raise ValueError(f"Ball {ball.id} is outside the playing surface")


class GameMode(str, Enum):
    DEMO = "demo"
    GROUP_PRACTICE = "group_practice"


class PlayerGroup(str, Enum):
    SOLIDS = "solids"
    STRIPES = "stripes"


@dataclass(frozen=True)
class GameContext:
    """Choose any solid/stripe target in demo mode, or a selected practice group.

    This is not a full eight-ball rules engine; the eight ball stays an obstacle.
    """

    mode: GameMode = GameMode.DEMO
    player_group: PlayerGroup | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, GameMode):
            raise ValueError("mode must be a GameMode enum")
        if self.mode is GameMode.GROUP_PRACTICE:
            if not isinstance(self.player_group, PlayerGroup):
                raise ValueError("Group practice requires a player group")
        elif self.player_group is not None:
            raise ValueError("Demo mode does not take a player group")
