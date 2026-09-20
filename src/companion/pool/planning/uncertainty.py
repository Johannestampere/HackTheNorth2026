"""Reproducible, assumed execution/perception errors and empirical shot scoring."""

from dataclasses import dataclass, replace
from math import cos, sin, sqrt, tau, isfinite
from random import Random

from companion.pool.contracts import Point2, TableState


@dataclass(frozen=True)
class ErrorModel:
    """Uncalibrated errors; angular standard deviation is degrees.

    Speed is multiplied by a uniform factor in [1-fraction, 1+fraction].
    Position is uniform in a disk: its radius uses each ball's reported
    position_uncertainty, or position_radius when unavailable. Interpreting
    reported uncertainty as a bound is a planner assumption, not a schema change.
    Layout samples are conditioned on nonoverlap and centers inside the rails.
    """

    aim_std_degrees: float = 0.5
    speed_fraction: float = 0.10
    position_radius: float = 0.001

    def __post_init__(self):
        if any(not isfinite(v) or v < 0 for v in (
            self.aim_std_degrees, self.speed_fraction, self.position_radius
        )) or self.speed_fraction >= 1:
            raise ValueError('Errors must be finite/nonnegative and speed_fraction below 1')

    def strike(self, rng: Random) -> tuple[float, float]:
        """Sample angle error and positive speed multiplier; never re-aim after error."""
        return rng.gauss(0, self.aim_std_degrees), rng.uniform(1-self.speed_fraction, 1+self.speed_fraction)

    def layout(self, state: TableState, rng: Random) -> TableState:
        """Joint rejection sampling preserves valid geometry without clipping balls."""
        r, width = state.geometry.ball_radius, state.geometry.width
        for _ in range(200):
            balls = []
            valid = True
            for ball in state.balls:
                bound = self.position_radius if ball.position_uncertainty is None else ball.position_uncertainty
                angle, distance = rng.random()*tau, sqrt(rng.random())*bound
                p = Point2(ball.position.x+cos(angle)*distance, ball.position.y+sin(angle)*distance)
                if not (r <= p.x <= 1-r and r <= p.y <= width-r) or any(
                    (p.x-b.position.x)**2+(p.y-b.position.y)**2 < (2*r)**2 for b in balls
                ):
                    valid = False
                    break
                balls.append(replace(ball, position=p))
            if valid:
                return replace(state, balls=tuple(balls))
        raise ValueError('Could not sample valid layouts under the supplied position uncertainty')


@dataclass(frozen=True)
class TrialStats:
    """Mutually exclusive sampled outcomes; rates are empirical, not calibrated.

    Utility: win +3, legal called pot +1, legal miss 0, foul -2, rack loss -5.
    Loss takes precedence over foul; win over pot, avoiding double counting.
    This immediate utility feeds the discounted multi-shot lookahead score.
    """

    trials: int
    wins: int
    pots: int
    misses: int
    fouls: int
    losses: int

    @property
    def utility(self) -> float:
        return (3*self.wins+self.pots-2*self.fouls-5*self.losses)/self.trials

    @property
    def success_rate(self) -> float:
        return (self.wins+self.pots)/self.trials

    @classmethod
    def from_outcomes(cls, outcomes):
        counts = dict(wins=0, pots=0, misses=0, fouls=0, losses=0)
        for outcome in outcomes:
            key = ('losses' if outcome.loss else 'fouls' if outcome.foul else
                   'wins' if outcome.win else 'pots' if outcome.made_call else 'misses')
            counts[key] += 1
        if not sum(counts.values()):
            raise ValueError('At least one uncertainty trial is required')
        return cls(trials=sum(counts.values()), **counts)


def diverse_shortlist(successful, limit):
    """Round-robin nominal successes by ball/pocket so one pot cannot fill the list."""
    groups = {}
    for item in sorted(successful, key=lambda x: x[0], reverse=True):
        candidate = item[1]
        groups.setdefault((candidate.ball_id, candidate.pocket_id), []).append(item)
    selected = []
    depth = 0
    while len(selected) < limit:
        added = False
        for group in groups.values():
            if depth < len(group):
                selected.append(group[depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    return selected
