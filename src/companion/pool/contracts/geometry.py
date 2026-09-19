"""2D geometry in table-length units: the long side is 1, origin at top-left."""

from dataclasses import dataclass
from math import hypot, isclose, isfinite


@dataclass(frozen=True)
class Point2:
    """A position in two dimensions; the ``2`` means x and y, not a version.

    One unit is the playing surface's long-side length. In the agreed top-down
    view, (0, 0) is top-left, +x runs right along the long side, and +y runs down
    along the short side. For a 2:1 table, the bottom-right is Point2(1, 0.5),
    NOT Point2(1, 1). These are table coordinates, not raw camera pixels.
    """

    x: float
    y: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.x, self.y)):
            raise ValueError("Point coordinates must be finite")


@dataclass(frozen=True)
class UnitVector2:
    """A direction in two dimensions with length one and no physical units.

    The ``2`` means x and y. UnitVector2(1, 0) points along positive table x.
    Cue aim points forward into the shot. This stores no origin, distance, or
    speed; callers must normalize their vector before constructing it.
    """

    x: float
    y: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.x, self.y)):
            raise ValueError("Direction components must be finite")
        if not isclose(hypot(self.x, self.y), 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError("Direction must have unit length")


@dataclass(frozen=True)
class Segment2:
    """A finite line in two dimensions, from ``start`` to ``end``.

    The ``2`` means both endpoints are Point2 positions in table-length units.
    Used for cue alignment and predicted ball paths; endpoints must differ.
    """

    start: Point2
    end: Point2

    def __post_init__(self) -> None:
        if self.start == self.end:
            raise ValueError("A segment must have nonzero length")
