"""Small, simulator-independent geometry checks for direct pot candidates."""

from dataclasses import dataclass
from math import hypot

from companion.pool.contracts import BallType, GameContext, PlayerGroup, Point2, TableState


@dataclass(frozen=True)
class Candidate:
    """Private direct-pot seed; phi is Pooltool's angle in degrees, not our frame."""

    ball_id: str
    pocket_id: str
    phi: float
    ghost: Point2
    difficulty: float


def target_ids(state: TableState, game: GameContext) -> set[str]:
    """Return the assigned group, or the eight only when that group is cleared."""
    kind = BallType.SOLID if game.player_group is PlayerGroup.SOLIDS else BallType.STRIPE
    own = {b.id for b in state.balls if b.type is kind}
    return own or {b.id for b in state.balls if b.type is BallType.EIGHT}


def path_clear(start: Point2, end: Point2, state: TableState, excluded: set[str]) -> bool:
    """Check a swept ball against other ball centers on a finite line segment."""
    dx, dy = end.x - start.x, end.y - start.y
    distance2 = dx * dx + dy * dy
    if distance2 < 1e-12:
        return False
    for ball in state.balls:
        if ball.id in excluded:
            continue
        t = max(0.0, min(1.0, ((ball.position.x - start.x) * dx
                              + (ball.position.y - start.y) * dy) / distance2))
        if hypot(ball.position.x - start.x - t * dx,
                 ball.position.y - start.y - t * dy) < 2 * state.geometry.ball_radius:
            return False
    return True
